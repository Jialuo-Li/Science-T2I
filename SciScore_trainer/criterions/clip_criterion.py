from dataclasses import dataclass
import torch
from omegaconf import II
from torch.nn.modules.loss import _Loss


@dataclass
class CLIPCriterionConfig:
    _target_: str = "criterions.clip_criterion.CLIPCriterion"
    is_distributed: bool = True
    input_ids_column_name: str = II("dataset.input_ids_column_name")
    pos_input_ids_column_name: str = II("dataset.pos_input_ids_column_name")
    neg_input_ids_column_name: str = II("dataset.neg_input_ids_column_name")
    pixels_pos_column_name: str = II("dataset.pixels_pos_column_name")
    pixels_neg_column_name: str = II("dataset.pixels_neg_column_name")
    lambda_: float = 0.1

class CLIPCriterion(_Loss):
    def __init__(self, cfg: CLIPCriterionConfig):
        super().__init__()
        self.cfg = cfg

    @staticmethod
    def get_features(model, input_ids, pos_input_ids, neg_input_ids, pixels_pos_values, pixels_neg_values):
        all_pixel_values = torch.cat([pixels_pos_values, pixels_neg_values], dim=0)
        text_features, all_image_features = model(text_inputs=input_ids, image_inputs=all_pixel_values)
        text_pos_features = model(text_inputs=pos_input_ids)[0]
        text_neg_features = model(text_inputs=neg_input_ids)[0]

        def normalize(features):
            return features / features.norm(dim=-1, keepdim=True)

        text_pos_features = normalize(text_pos_features)
        text_neg_features = normalize(text_neg_features)
        all_image_features = normalize(all_image_features)
        text_features = normalize(text_features)
        image_pos_features, image_neg_features = all_image_features.chunk(2, dim=0)
        return image_pos_features, image_neg_features, text_features, text_pos_features, text_neg_features

    @staticmethod
    def gather_features(features):
        return torch.cat(torch.distributed.nn.all_gather(features), dim=0)

    def calc_loss(self, text_features, text_pos_features, text_neg_features, image_pos_features, image_neg_features, logit_scale, *args, **kwargs):
        device = image_pos_features.device

        if self.cfg.is_distributed:
            image_pos_features = self.gather_features(image_pos_features)
            image_neg_features = self.gather_features(image_neg_features)
            text_features = self.gather_features(text_features)
            text_pos_features = self.gather_features(text_pos_features)
            text_neg_features = self.gather_features(text_neg_features)

        all_image_features = torch.cat([image_pos_features, image_neg_features], dim=0)

        text_logits = logit_scale * text_features @ all_image_features.T
        pos_text_logits = logit_scale * text_pos_features @ all_image_features.T
        neg_text_logits = logit_scale * text_neg_features @ all_image_features.T

        def get_text_loss(logits, align_pos=True):
            pos_logits, neg_logits = logits.chunk(2, dim=-1)
            index = torch.arange(pos_logits.shape[0], device=device)
            pos_logits = pos_logits[index, index]
            neg_logits = neg_logits[index, index]
            logits = torch.stack([pos_logits, neg_logits], dim=-1)
            labels = torch.zeros(logits.shape[0], device=device, dtype=torch.long)
            if not align_pos:
                labels = labels + 1
            return torch.nn.functional.cross_entropy(logits, labels, reduction="none")

        def get_image_loss(pos_logits, neg_logits):
            pos_logits_pos, pos_logits_neg = pos_logits.chunk(2, dim=-1)
            neg_logits_pos, neg_logits_neg = neg_logits.chunk(2, dim=-1)
            index = torch.arange(pos_logits_pos.shape[0], device=device)
            pos_labels = torch.zeros(pos_logits.shape[0], device=device, dtype=torch.long)
            neg_labels = torch.ones(neg_logits.shape[0], device=device, dtype=torch.long)
            pos_combined_logits = torch.stack([pos_logits_pos[index, index], neg_logits_pos[index, index]], dim=-1)
            neg_combined_logits = torch.stack([pos_logits_neg[index, index], neg_logits_neg[index, index]], dim=-1)
            return torch.nn.functional.cross_entropy(pos_combined_logits, pos_labels, reduction="none"), torch.nn.functional.cross_entropy(neg_combined_logits, neg_labels, reduction="none")

        text_loss = get_text_loss(text_logits)
        pos_img_loss, neg_img_loss = get_image_loss(pos_text_logits, neg_text_logits)

        loss = text_loss + self.cfg.lambda_ * (pos_img_loss +  neg_img_loss)
        return loss.sum()

    def forward(self, model, batch):
        image_pos_features, image_neg_features, text_features, text_pos_features, text_neg_features = self.get_features(
            model,
            batch[self.cfg.input_ids_column_name],
            batch[self.cfg.pos_input_ids_column_name],
            batch[self.cfg.neg_input_ids_column_name],
            batch[self.cfg.pixels_pos_column_name],
            batch[self.cfg.pixels_neg_column_name],
        )
        return self.calc_loss(
            text_features,
            text_pos_features,
            text_neg_features,
            image_pos_features,
            image_neg_features,
            model.logit_scale.exp(),
        )