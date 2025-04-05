from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import math
import torch

from diffusers.utils import logging
from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteSchedulerOutput

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

def unpack_latents(latents, height, width):
    batch_size, num_patches, channels = latents.shape

    latents = latents.view(batch_size, height, width, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    latents = latents.reshape(batch_size, channels // (2 * 2), height * 2, width * 2)

    return latents


def apply_box_mask(tensor, box):
    batchsize, _, height, width = tensor.shape
    cx, cy, w, h = box[:, 0], box[:, 1], box[:, 2], box[:, 3]

    x1 = (cx - w / 2) * width
    x2 = (cx + w / 2) * width
    y1 = (cy - h / 2) * height
    y2 = (cy + h / 2) * height

    mask = torch.zeros((batchsize, height, width), device=tensor.device)
    for i in range(batchsize):
        mask[i, int(y1[i]):int(y2[i]), int(x1[i]):int(x2[i])] = 1
    mask = mask.unsqueeze(1) 
    masked_tensor = tensor * mask
    return masked_tensor




def flowmatching_with_logprob(
    self,
    model_output: torch.FloatTensor,
    timestep: Union[float, torch.FloatTensor],
    sample: torch.FloatTensor,
    s_churn: float = 0.0,
    s_tmin: float = 0.0,
    s_tmax: float = float("inf"),
    s_noise: float = 1.0,
    generator: Optional[torch.Generator] = None,
    prev_sample: Optional[torch.FloatTensor] = None,
    step_index: Optional[torch.IntTensor] = None,
    box: Optional[torch.FloatTensor] = None,
) -> Union[FlowMatchEulerDiscreteSchedulerOutput, Tuple]:
    """
    Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
    process from the learned model outputs (most often the predicted noise).

    Args:
        model_output (`torch.FloatTensor`):
            The direct output from learned diffusion model.
        timestep (`float`):
            The current discrete timestep in the diffusion chain.
        sample (`torch.FloatTensor`):
            A current instance of a sample created by the diffusion process.
        s_churn (`float`):
        s_tmin  (`float`):
        s_tmax  (`float`):
        s_noise (`float`, defaults to 1.0):
            Scaling factor for noise added to the sample.
        generator (`torch.Generator`, *optional*):
            A random number generator.
        return_dict (`bool`):
            Whether or not to return a [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] or
            tuple.

    Returns:
        [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] or `tuple`:
            If return_dict is `True`, [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] is
            returned, otherwise a tuple is returned where the first element is the sample tensor.
    """
    if (
        isinstance(timestep, int)
        or isinstance(timestep, torch.IntTensor)
        or isinstance(timestep, torch.LongTensor)
    ):
        raise ValueError(
            (
                "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                " `EulerDiscreteScheduler.step()` is not supported. Make sure to pass"
                " one of the `scheduler.timesteps` as a timestep."
            ),
        )

    step_index_is_none = step_index is None
    # Upcast to avoid precision issues when computing prev_sample
    original_dtype = sample.dtype
    if step_index_is_none:
        if self.step_index is None:
            self._init_step_index(timestep)
            
        step_index = self.step_index 

        sigma = self.sigmas[step_index]
        sigma_next = self.sigmas[step_index + 1]

        if s_churn > 0 and s_tmin <= sigma <= s_tmax:
            gamma = min(s_churn / (len(self.sigmas) - 1), math.sqrt(2) - 1)
            sigma_hat = sigma * (gamma + 1)
            std_dev_t = s_noise * (sigma_hat**2 - sigma**2) ** 0.5
        else:
            std_dev_t = 0.0

        delta = model_output + (std_dev_t **2 / (2 * sigma)) * (-sample + (1-sigma) * model_output)
        
        prev_sample_mean = sample + (sigma_next - sigma) * delta

        if prev_sample is None:
            noise = randn_tensor(
                model_output.shape, dtype=model_output.dtype, device=model_output.device, generator=generator
            )
            prev_sample = prev_sample_mean + std_dev_t * noise

        log_prob = (
            -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * (std_dev_t**2))
            - torch.log(std_dev_t)
            - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
        )
    
        # mean along all but batch dimension
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

        out_step_index = self._step_index
        self._step_index += 1
        return prev_sample.to(original_dtype), log_prob, torch.tensor([out_step_index] * log_prob.shape[0]).to(sample.device)
    
    else:

        sigma = self.sigmas[step_index]
        sigma_next = self.sigmas[step_index + 1]

        if s_churn > 0:
            mask = (sigma >= s_tmin) & (sigma <= s_tmax)
            gamma = torch.minimum(torch.full_like(sigma, s_churn / (len(self.sigmas) - 1)), torch.sqrt(torch.tensor(2.0)) - 1)
            sigma_hat = sigma * (gamma + 1)
            std_dev_t = torch.where(mask, s_noise * torch.sqrt(sigma_hat**2 - sigma**2), torch.zeros_like(sigma))
        else:
            std_dev_t = torch.zeros_like(sigma)

        # Reshape sigma and sigma_next for broadcasting
        sigma = sigma.view(-1, 1, 1)  # Adjust based on sample dimensions
        sigma_next = sigma_next.view(-1, 1, 1)
        std_dev_t = std_dev_t.view(-1, 1, 1)
        

        delta = model_output + (std_dev_t **2 / (2 * sigma)) * (-sample + (1-sigma) * model_output)
        prev_sample_mean = sample + (sigma_next - sigma) * delta

        log_prob = (
            -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * (std_dev_t.detach()**2))
            - torch.log(std_dev_t.detach())
            - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
        )
        if box is not None:
            log_prob = unpack_latents(log_prob, 32, 32) # config.resolution // vae_scale
            log_prob = apply_box_mask(log_prob, box).to(original_dtype)
        
        # mean along all but batch dimension
        log_prob = log_prob.view(log_prob.size(0), -1).mean(dim=1)
        
        return prev_sample.to(original_dtype), log_prob, None