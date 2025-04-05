from PIL import Image
import io
import numpy as np
import torch

def light_reward(device: str = "cuda"):
    def _fn(images, prompts, metadata):
        reward = images.reshape(images.shape[0],-1).mean(1)
        return np.array(reward.cpu().detach().float()),{}
    return _fn

def ImageReward(device: str = "cuda"):
    from .ImageReward import ImageRewardScorer

    scorer = ImageRewardScorer(dtype=torch.float32, device=device)
    scorer.requires_grad_(False)

    def _fn(images, prompts, metadata):
        scores = scorer(images, prompts)
        return scores, {}

    return _fn

def SciScore(device: str = "cuda"):
    from .SciScore import SciScore

    scorer = SciScore(dtype=torch.float32, device=device)
    scorer.requires_grad_(False)

    def _fn(images, prompts, objects, metadata):
        scores, box = scorer(images, prompts, objects)
        return scores, box, {}

    return _fn


def aesthetic_score(device: str = "cuda"):
    from utils.aesthetic_scorer import AestheticScorer

    scorer = AestheticScorer(dtype=torch.float32, device=device).to(device)
    scorer.requires_grad_(False)
    
    def _fn(images, prompts, metadata):
        scores = scorer(images)
        return scores, {}

    return _fn