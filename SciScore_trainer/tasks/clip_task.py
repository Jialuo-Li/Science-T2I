import collections
from dataclasses import dataclass

import torch
from PIL import Image
from accelerate.logging import get_logger
from accelerate.utils import LoggerType
from omegaconf import II
from transformers import AutoTokenizer

from accelerators.base_accelerator import BaseAccelerator
from tasks.base_task import BaseTaskConfig, BaseTask

logger = get_logger(__name__)


@dataclass
class CLIPTaskConfig(BaseTaskConfig):
    _target_: str = "tasks.clip_task.CLIPTask"
    pretrained_model_name_or_path: str = II("model.pretrained_model_name_or_path")
    input_ids_column_name: str = II("dataset.input_ids_column_name")
    pos_input_ids_column_name: str = II("dataset.pos_input_ids_column_name")
    neg_input_ids_column_name: str = II("dataset.neg_input_ids_column_name")
    pixels_pos_column_name: str = II("dataset.pixels_pos_column_name")
    pixels_neg_column_name: str = II("dataset.pixels_neg_column_name")
    category_column_name: str = II("dataset.category_column_name")


def numpy_to_pil(images):
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images


class CLIPTask(BaseTask):
    def __init__(self, cfg: CLIPTaskConfig, accelerator: BaseAccelerator):
        super().__init__(cfg, accelerator)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained_model_name_or_path)
        self.cfg = cfg
        self.roc_data_list = []

    def train_step(self, model, criterion, batch):
        loss = criterion(model, batch)
        return loss

    @staticmethod
    def features2probs(model, text_features, image_pos_features, image_neg_features):
        image_pos_scores = model.logit_scale.exp() * torch.diag(
            torch.einsum('bd,cd->bc', text_features, image_pos_features))
        image_neg_scores = model.logit_scale.exp() * torch.diag(
            torch.einsum('bd,cd->bc', text_features, image_neg_features))
        scores = torch.stack([image_pos_scores, image_neg_scores], dim=-1)
        probs = torch.softmax(scores, dim=-1)
        image_pos_probs, image_neg_probs = probs[:, 0], probs[:, 1]
        return image_pos_probs, image_neg_probs
    
    @torch.no_grad()
    def valid_step(self, model, criterion, batch):
        image_pos_features, image_neg_features, text_features, _, _ = criterion.get_features(
            model,
            batch[self.cfg.input_ids_column_name],
            batch[self.cfg.pos_input_ids_column_name],
            batch[self.cfg.neg_input_ids_column_name],
            batch[self.cfg.pixels_pos_column_name],
            batch[self.cfg.pixels_neg_column_name]
        )
        
        image_pos_probs, image_neg_probs = self.features2probs(model, text_features, image_pos_features, image_neg_features)
        
        return image_pos_probs, image_neg_probs, batch[self.cfg.category_column_name]

    @staticmethod
    def pixel_values_to_pil_images(pixel_values):
        images = (pixel_values / 2 + 0.5).clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).float().numpy()
        images = numpy_to_pil(images)
        return images

    def run_inference(self, model, criterion, dataloader):
        eval_dict = collections.defaultdict(list)
        logger.info("Running clip score...")
        for batch in dataloader:
            image_pos_probs, image_neg_probs, categories = self.valid_step(model, criterion, batch)
            
            is_correct = (image_pos_probs > image_neg_probs)
            
            eval_dict["is_correct"] += is_correct.tolist()
            eval_dict["categories"] += categories
            
            eval_dict["true_labels"] += [1] * len(image_pos_probs) + [0] * len(image_neg_probs)
            eval_dict["predicted_probs"] += image_pos_probs.tolist() + image_neg_probs.tolist()
            
            eval_dict["captions"] += self.tokenizer.batch_decode(
                batch[self.cfg.input_ids_column_name],
                skip_special_tokens=True
            )
            eval_dict["image_pos"] += self.pixel_values_to_pil_images(batch[self.cfg.pixels_pos_column_name])
            eval_dict["image_neg"] += self.pixel_values_to_pil_images(batch[self.cfg.pixels_neg_column_name])
            eval_dict["prob_pos"] += image_pos_probs.tolist()
            eval_dict["prob_neg"] += image_neg_probs.tolist()

        return eval_dict

    @torch.no_grad()
    def evaluate(self, model, criterion, dataloader):
        eval_dict = self.run_inference(model, criterion, dataloader)
        eval_dict = self.gather_dict(eval_dict)
        metrics = {
            "accuracy": sum(eval_dict["is_correct"]) / len(eval_dict["is_correct"])
        }

        category_correct = collections.defaultdict(int)
        category_total = collections.defaultdict(int)
        for cat, correct in zip(eval_dict["categories"], eval_dict["is_correct"]):
            category_total[cat] += 1
            if correct:
                category_correct[cat] += 1
        for cat in category_total:
            acc = category_correct[cat] / category_total[cat]
            metrics[f"accuracy_{cat}"] = acc
            
        if LoggerType.WANDB == self.accelerator.cfg.log_with:
            self.log_to_wandb(eval_dict)
        return metrics