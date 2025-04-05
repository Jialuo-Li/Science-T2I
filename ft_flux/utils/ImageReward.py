import os
import torch
import torch.nn as nn
from transformers import CLIPProcessor
from ImageReward.models.BLIP.blip_pretrain import BLIP_Pretrain
from ImageReward import ImageReward_download


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    @torch.no_grad()
    def forward(self, embed):
        return self.layers(embed)


class ImageRewardScorer(nn.Module):
    def __init__(self, dtype, device):
        super().__init__()
        self.dtype = dtype
        self.device = device

        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

        download_root = f"{os.path.expanduser('~')}/.cache/ImageReward"
        config_path = ImageReward_download("https://huggingface.co/THUDM/ImageReward/blob/main/med_config.json", download_root)
        model_path = ImageReward_download("https://huggingface.co/THUDM/ImageReward/blob/main/ImageReward.pt", download_root)

        self.blip = BLIP_Pretrain(image_size=224, vit='large', med_config=config_path).to(self.device, dtype=self.dtype)
        self.mlp = MLP().to(self.device, dtype=self.dtype)

        state_dict = torch.load(model_path, map_location=self.device)
        self.load_state_dict(state_dict, strict=False)
        self.eval()

    @torch.no_grad()
    def __call__(self, images, prompts):
        images = (images * 255).round().clamp(0, 255).to(torch.uint8)
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.dtype).to(self.device) for k, v in inputs.items()}["pixel_values"]
        image_embeds = self.blip.visual_encoder(inputs)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(self.device)
        text_input = self.blip.tokenizer(
            prompts,
            padding='max_length',
            truncation=True,
            max_length=35,
            return_tensors="pt"
        ).to(self.device)
        text_output = self.blip.text_encoder(
            text_input.input_ids,
            attention_mask=text_input.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        txt_features = text_output.last_hidden_state[:, 0, :].to(dtype=self.dtype)
        scores = self.mlp(txt_features).squeeze(1)

        return scores