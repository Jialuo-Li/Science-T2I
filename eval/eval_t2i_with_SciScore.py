import torch
import argparse
from diffusers import FluxPipeline
from accelerate import Accelerator
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoModel
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms

class SciScore(nn.Module):
    def __init__(self, device, dtype):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.processor = AutoProcessor.from_pretrained('Jialuo21/SciScore')
        self.model = AutoModel.from_pretrained('Jialuo21/SciScore').eval().to(device, dtype=dtype)
        
    @torch.no_grad()
    def forward(self, images, prompts):
        # Preprocess images
        transform = transforms.ToTensor()
        images = transform(images)
        images = (images * 255).round().clamp(0, 255).to(torch.uint8)

        image_inputs = self.processor(
            images=images,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)

        # Preprocess text
        text_inputs = self.processor(
            text=prompts, 
            padding=True, 
            truncation=True, 
            max_length=77, 
            return_tensors="pt"
        )
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        
        # Get embeddings
        image_embeds = F.normalize(self.model.get_image_features(**image_inputs), dim=-1)
        text_embeds = F.normalize(self.model.get_text_features(**text_inputs), dim=-1)
        logits_per_image = image_embeds @ text_embeds.T
        scores = torch.diagonal(logits_per_image) * self.model.logit_scale.exp()
        # Compute scores
        return scores

def compute_scores(pipe, sci_scorer, prompt, ip, num_samples=2):
    scores = []
    for _ in range(num_samples):
        image = pipe(prompt, guidance_scale=0.0, num_inference_steps=4, max_sequence_length=256, height=512, width=512).images[0]
        score = sci_scorer(image, ip)
        scores.append(score.item())
    return scores

def main(dataset_name):
    accelerator = Accelerator()
    
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
    pipe.to(accelerator.device)

    sci_scorer = SciScore(device=accelerator.device, dtype=torch.float32)
    
    dataset = load_dataset(dataset_name)["test"].select_columns(['implicit_prompt', 'explicit_prompt', 'superficial_prompt'])
    dataloader = DataLoader(dataset, batch_size=1)
    dataloader = accelerator.prepare(dataloader)

    sciscores = {'ip': [], 'ep': [], 'sp': []}
    for batch in dataloader:
        prompts = {
            'ip': batch['implicit_prompt'][0],
            'ep': batch['explicit_prompt'][0],
            'sp': batch['superficial_prompt'][0]
        }
        
        for key, prompt in prompts.items():
            scores = compute_scores(pipe, sci_scorer, prompt, prompts['ip'])
            sciscores[key].extend(scores)

    # Gather and compute averages
    for key in sciscores:
        scores_tensor = torch.tensor(sciscores[key], device=accelerator.device)
        gathered_scores = accelerator.gather(scores_tensor).cpu().tolist()
        
        if accelerator.is_main_process:
            avg_score = sum(gathered_scores) / len(gathered_scores)
            print(f"Average SciScore for {key}: {avg_score}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a dataset using a pretrained model.")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset to evaluate.")
    args = parser.parse_args()
    main(dataset_name=args.dataset_name)