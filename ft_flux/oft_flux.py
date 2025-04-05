import os
import copy
import tempfile
import contextlib
import datetime
from collections import defaultdict
from functools import partial

import torch
import wandb
from PIL import Image
from absl import app, flags
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import FluxPipeline
from diffusers.training_utils import cast_training_params
from diffusers.utils import convert_unet_state_dict_to_peft
from diffusers.utils.torch_utils import is_compiled_module
from ml_collections import config_flags
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from tqdm import tqdm

import utils.prompts
import utils.rewards
from utils.diffusers_patch.flux_encoder import encode_prompt
from utils.diffusers_patch.flux_pipeline_with_logprob import flux_pipeline_with_logprob
from utils.diffusers_patch.flow_matching_with_logprob import flowmatching_with_logprob

tqdm = partial(tqdm, dynamic_ncols=True)


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

logger = get_logger(__name__)


def main(_):
    # basic Accelerate and logging setup
    config = FLAGS.config

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id

    if config.resume_from:
        config.resume_from = os.path.normpath(os.path.expanduser(config.resume_from))
        if "checkpoint_" not in os.path.basename(config.resume_from):
            # get the most recent checkpoint in this directory
            checkpoints = list(
                filter(lambda x: "checkpoint_" in x, os.listdir(config.resume_from))
            )
            if len(checkpoints) == 0:
                raise ValueError(f"No checkpoints found in {config.resume_from}")
            config.resume_from = os.path.join(
                config.resume_from,
                sorted(checkpoints, key=lambda x: int(x.split("_")[-1]))[-1],
            )

    # number of timesteps within each trajectory to train on
    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=False,
        total_limit=config.num_checkpoint_limit,
    )


    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        # we always accumulate gradients across timesteps; we want config.train.gradient_accumulation_steps to be the
        # number of *samples* we accumulate across, so we need to multiply by the number of training timesteps to get
        # the total number of optimizer steps to accumulate across.
        gradient_accumulation_steps=config.train.gradient_accumulation_steps
        * num_train_timesteps,
    )
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="fine-tune-flux-oft",
            config=config.to_dict(),
            init_kwargs={"wandb": {"name": config.run_name}},
        )
    logger.info(f"\n{config}")

    # set seed (device_specific is very important to get different prompts on different devices)
    set_seed(config.seed, device_specific=True)

    # load scheduler, tokenizer and models.
    pipeline = FluxPipeline.from_pretrained(config.pretrained.model, revision=config.pretrained.revision, torch_dtype=torch.bfloat16).to(accelerator.device)

    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.transformer.requires_grad_(not config.use_lora) # pipeline.unet.requires_grad_(not config.use_lora)
    # disable safety checker
    pipeline.safety_checker = None
    # make the progress bar nicer
    pipeline.set_progress_bar_config(
        position=1,
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )
    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    if not config.resume_from:
        transformer_ref = copy.deepcopy(pipeline.transformer)
        for param in transformer_ref.parameters():
            param.requires_grad = False
    # store transformer in variable
    transformer = pipeline.transformer

    if config.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
            
    if config.use_lora:
        transformer_lora_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_rank, 
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"]
        )
        transformer.add_adapter(transformer_lora_config)

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            transformer_lora_layers_to_save = None
            text_encoder_one_lora_layers_to_save = None

            for model in models:
                if isinstance(model, type(unwrap_model(transformer))):
                    transformer_lora_layers_to_save = get_peft_model_state_dict(model)
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

            FluxPipeline.save_lora_weights(
                output_dir,
                transformer_lora_layers=transformer_lora_layers_to_save,
                text_encoder_lora_layers=text_encoder_one_lora_layers_to_save,
            )

    def load_model_hook(models, input_dir):
        transformer_ = None
        text_encoder_one_ = None

        while len(models) > 0:
            model = models.pop()

            if isinstance(model, type(unwrap_model(transformer))):
                transformer_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        lora_state_dict = FluxPipeline.lora_state_dict(input_dir)

        transformer_state_dict = {
            f'{k.replace("transformer.", "")}': v for k, v in lora_state_dict.items() if k.startswith("transformer.")
        }
        transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
        incompatible_keys = set_peft_model_state_dict(transformer_, transformer_state_dict, adapter_name="default")
        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )

        # Make sure the trainable params are in float32. This is again needed since the base models
        # are in `weight_dtype`. More details:
        # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
        if config.mixed_precision == "fp16":
            models = [transformer_]
            # only upcast trainable parameters (LoRA) into fp32
            cast_training_params(models)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    
    # Make sure the trainable params are in float32.
    if accelerator.mixed_precision == "fp16":
        models = [transformer]
        cast_training_params(models, dtype=torch.float32)

    # Initialize the optimizer
    if config.train.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    params_to_optimize = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    optimizer = optimizer_cls(
        params_to_optimize,
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    # prepare prompt and reward fn
    prompt_fn = getattr(utils.prompts, config.prompt_fn)
    eval_prompt_fn = getattr(utils.prompts, config.eval_prompt_fn)
    reward_fn = getattr(utils.rewards, config.reward_fn)(device=accelerator.device) # ensure reward model is on same device

    # for some reason, autocast is necessary for non-lora training but for lora training it isn't necessary and it uses
    # more memory
    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast
    # autocast = accelerator.autocast

    # Prepare everything with our `accelerator`.
    transformer = accelerator.prepare(transformer)
    # Train!
    samples_per_epoch = (
        config.sample.batch_size
        * accelerator.num_processes
        * config.sample.num_batches_per_epoch
    )
    total_train_batch_size = (
        config.train.batch_size
        * accelerator.num_processes
        * config.train.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {config.num_epochs}")
    logger.info(f"  Sample batch size per device = {config.sample.batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}"
    )
    logger.info("")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}"
    )
    assert config.sample.batch_size == config.train.batch_size # for dpo loss
    assert config.sample.batch_size % config.train.batch_size == 0
    assert samples_per_epoch % total_train_batch_size == 0
    assert config.sample.batch_size == 2 # for dpo loss

    if config.resume_from:
        logger.info(f"Resuming from {config.resume_from}")
        accelerator.load_state(config.resume_from)
        first_epoch = int(config.resume_from.split("_")[-1]) + 1
        transformer_ref = copy.deepcopy(pipeline.transformer)
        for param in transformer_ref.parameters():
            param.requires_grad = False
        optimizer = optimizer_cls(
            params_to_optimize,
            lr=config.train.learning_rate,
            betas=(config.train.adam_beta1, config.train.adam_beta2),
            weight_decay=config.train.adam_weight_decay,
            eps=config.train.adam_epsilon,
        )
        optimizer = accelerator.prepare(optimizer)
    else:
        first_epoch = 0

    global_step = 0
    for epoch in range(first_epoch, config.num_epochs):
        #################### SAMPLING ####################
        pipeline.transformer.eval()

        samples = []
        prompts = []
        for i in tqdm(
            range(config.sample.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):  
            if config.prompt_fn == 'sci':
                single_prompts, single_objects, prompt_metadata = prompt_fn(**config.prompt_fn_kwargs)
                prompts = (single_prompts,) * config.sample.batch_size
                objects = (single_objects,) * config.sample.batch_size
            else:
                single_prompts, prompt_metadata = prompt_fn(**config.prompt_fn_kwargs)
                prompts = (single_prompts,) * config.sample.batch_size
            

            prompt_metadata = tuple(copy.deepcopy(prompt_metadata) for _ in range(config.sample.batch_size))


            prompt_embeds, pooled_prompt_embeds, text_ids, prompt_ids = encode_prompt(
                pipeline, prompts, prompt_2=None, device=accelerator.device
            )
            
            with autocast():
                images, latents, log_probs, pipeline_ret_timesteps, step_idxs = flux_pipeline_with_logprob(
                    pipeline,
                    height=config.resolution,
                    width=config.resolution,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    num_inference_steps=config.sample.num_steps,
                    guidance_scale=config.sample.guidance_scale,
                    s_churn=config.sample.s_churn,
                    output_type="pt"
                )
            
            latents = torch.stack(latents, dim=1) 
            log_probs = torch.stack(log_probs, dim=1)  # (batch_size, num_steps, 1)
            step_idxs = torch.stack(step_idxs, dim=1)  # (batch_size, num_steps, 1)
            timesteps = pipeline_ret_timesteps.repeat(
                config.sample.batch_size, 1
            )  # (batch_size, num_steps)

            sample_data = {
                "prompt_ids": prompt_ids,
                "text_ids": text_ids,
                "prompt_embeds": prompt_embeds,
                "pooled_prompt_embeds": pooled_prompt_embeds,
                "timesteps": timesteps,
                "latents": latents[:, :-1],
                "next_latents": latents[:, 1:],
                "log_probs": log_probs,
                "step_idxs": step_idxs,
            }

            if config.reward_fn == "SciScore":
                rewards, boxes, metadata = reward_fn(images, prompts, objects, prompt_metadata)
                sample_data.update({"boxes": boxes, "rewards": rewards})
            else:
                rewards, metadata = reward_fn(images, prompts, prompt_metadata)
                sample_data.update({"rewards": rewards})
            samples.append(sample_data)

        for sample in tqdm(
            samples,
            desc="Waiting for rewards",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            rewards = sample["rewards"]
            sample["rewards"] = torch.as_tensor(rewards, device=accelerator.device)

        samples = {k: torch.cat([s[k] for s in samples]) for k in samples[0].keys()}

        def save_and_prepare_image(image, tmpdir, filename):
            pil = Image.fromarray(((image * 255).clamp(0, 255).to(torch.uint8).cpu().numpy().transpose(1, 2, 0)))
            pil = pil.resize((256, 256))
            output_path = os.path.join(tmpdir, filename)
            pil.save(output_path, format="PNG")
            return output_path 

        # Modified version of "this is a hack to force wandb to log the images as JPEGs instead of PNGs"
        with tempfile.TemporaryDirectory() as tmpdir:
            logged_images = []
            for i, image in enumerate(images):
                img_path = save_and_prepare_image(image, tmpdir, f"{i}.png")
                logged_images.append(
                    wandb.Image(
                        img_path,
                        caption=f"{prompts[i]:.25} | {rewards[i]:.2f}" # only log rewards from process 0
                    )
                )

            accelerator.log(
                {
                    "images": logged_images,
                },
                step=global_step
            )

        rewards = accelerator.gather(samples["rewards"]).cpu().numpy()

        accelerator.log(
            {
                "reward": rewards,
                "epoch": epoch,
                "reward_mean": rewards.mean(),
                "reward_std": rewards.std(),
            },
            step=global_step,
        )

        del samples["prompt_ids"]

        total_batch_size, num_timesteps = samples["timesteps"].shape
        assert (total_batch_size == config.sample.batch_size * config.sample.num_batches_per_epoch)
        assert num_timesteps == config.sample.num_steps * config.sample.denoising_split

        #################### EVALUATION ####################
        eval_samples = []
        eval_prompts = []

        for i in tqdm(range(config.eval_num_batches), desc="Evaluation", position=0, disable=not accelerator.is_local_main_process):  
            if config.eval_prompt_fn == 'sci_eval':
                eval_prompts, eval_objects, eval_prompt_metadata = zip(*[eval_prompt_fn(**config.prompt_fn_kwargs) for _ in range(config.eval_batch_size)])
            else:
                eval_prompts, eval_prompt_metadata = zip(*[eval_prompt_fn(**config.prompt_fn_kwargs) for _ in range(config.eval_batch_size)])
            

            # encode prompts using custom encoding function to track prompt_ids
            eval_prompt_embeds, eval_pooled_prompt_embeds, eval_text_ids, eval_prompt_ids = encode_prompt(pipeline, eval_prompts, prompt_2=None, device=accelerator.device) 
                    
            
            with autocast():
                eval_images, _, _, _, _ = flux_pipeline_with_logprob(
                    pipeline,
                    prompt_embeds=eval_prompt_embeds,
                    pooled_prompt_embeds=eval_pooled_prompt_embeds,
                    num_inference_steps=config.sample.num_steps,
                    guidance_scale=config.sample.guidance_scale,
                    s_churn=config.sample.s_churn,
                    output_type="pt"
                )

            # compute rewards 
            if config.reward_fn == "SciScore":
                eval_rewards, eval_boxes, eval_metadata = reward_fn(eval_images, eval_prompts, eval_objects, eval_prompt_metadata)
            else:
                eval_rewards, eval_metadata = reward_fn(eval_images, eval_prompts, eval_prompt_metadata)

            eval_samples.append(
                {   
                    "rewards": eval_rewards
                }
            )

        # wait for all rewards to be computed
        for sample in tqdm(
            eval_samples,
            desc="Waiting for rewards",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            rewards = sample["rewards"]
            sample["rewards"] = torch.as_tensor(rewards, device=accelerator.device)

        # collate samples into dict where each entry has shape (num_batches_per_epoch * sample.batch_size, ...)
        eval_samples = {k: torch.cat([s[k] for s in eval_samples]) for k in eval_samples[0].keys()}

        # gather rewards across processes
        eval_rewards = accelerator.gather(eval_samples["rewards"]).cpu().numpy()

        # log rewards and images
        accelerator.log(
            {
                "eval_reward": eval_rewards,
                "eval_reward_mean": eval_rewards.mean(),
            },
            step=global_step,
        )
            
        #################### TRAINING ####################
        for inner_epoch in range(config.train.num_inner_epochs):
            # rebatch for training
            samples_batched = {k: v.reshape(-1, config.train.batch_size, *v.shape[1:]) for k, v in samples.items()}

            # dict of lists -> list of dicts for easier iteration
            samples_batched = [dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())]

            pipeline.transformer.train()
            info = defaultdict(list)
            for i, sample in tqdm(
                list(enumerate(samples_batched)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not accelerator.is_local_main_process,
            ):

                for j in tqdm(
                    range(num_train_timesteps),
                    desc="Timestep",
                    position=1,
                    leave=False,
                    disable=not accelerator.is_local_main_process,
                ):
                    
                    with accelerator.accumulate(transformer):
                        with autocast():
                            
                            latent_image_ids = FluxPipeline._prepare_latent_image_ids(
                                sample["latents"].shape[0],
                                2 * (int(config.resolution) // pipeline.vae_scale_factor),
                                2 * (int(config.resolution) // pipeline.vae_scale_factor),
                                accelerator.device,
                                inference_dtype,
                            )
                            
                            # handle guidance
                            if transformer.module.config.guidance_embeds:
                                guidance = torch.tensor([config.sample.guidance_scale], device=accelerator.device)
                                guidance = guidance.expand(sample["latents"].shape[0])
                            else:
                                guidance = None
                        
                            noise_pred = transformer(
                                sample["latents"][:, j],
                                timestep=sample["timesteps"][:, j].expand(sample["latents"][:, j].shape[0]).to(sample["latents"][:, j].dtype)/1000,
                                guidance=guidance,
                                pooled_projections=sample["pooled_prompt_embeds"],
                                encoder_hidden_states=sample["prompt_embeds"],
                                txt_ids=sample["text_ids"],
                                img_ids=latent_image_ids,
                                return_dict=False,
                            )[0]
                            
                            noise_pred_ref = transformer_ref(
                                sample["latents"][:, j],
                                timestep=sample["timesteps"][:, j].expand(sample["latents"][:, j].shape[0]).to(sample["latents"][:, j].dtype)/1000,
                                guidance=guidance,
                                pooled_projections=sample["pooled_prompt_embeds"],
                                encoder_hidden_states=sample["prompt_embeds"],
                                txt_ids=sample["text_ids"],
                                img_ids=latent_image_ids,
                                return_dict=False,
                            )[0]
                            
                        # compute the log prob of next_latents given latents under the current model
                        common_args = {
                            'timestep': sample["timesteps"][:, j],
                            'sample': sample["latents"][:, j],
                            's_churn': config.sample.s_churn,
                            'prev_sample': sample["next_latents"][:, j],
                            'step_index': sample["step_idxs"][:, j]
                        }
                        # Add box parameter if reward_fn is "SciScore"
                        extra_args = {'box': sample["boxes"]} if config.reward_fn == "SciScore" else {}

                        # Calculate log probabilities
                        _, log_prob, _ = flowmatching_with_logprob(
                            pipeline.scheduler,
                            model_output=noise_pred,
                            **common_args,
                            **extra_args
                        )

                        _, log_prob_ref, _ = flowmatching_with_logprob(
                            pipeline.scheduler,
                            model_output=noise_pred_ref,
                            **common_args,
                            **extra_args
                        )

                        reward_diff = sample['rewards'][:, None] - sample['rewards']
                        ratio = torch.log(torch.clamp(torch.exp(log_prob-log_prob_ref), 1 - config.train.eps, 1 + config.train.eps))
                        ratio_diff = ratio[:, None] - ratio
                        loss = -torch.log(torch.sigmoid(config.train.beta * ratio_diff * (reward_diff > 0))).mean()
                       
                        info["loss"].append(loss)
                        # backward pass
                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(
                                transformer.parameters(), config.train.max_grad_norm
                            )
                        optimizer.step()
                        optimizer.zero_grad()

                    # Checks if the accelerator has performed an optimization step behind the scenes
                    if accelerator.sync_gradients:
                        assert (j == num_train_timesteps - 1) and (
                            i + 1
                        ) % config.train.gradient_accumulation_steps == 0
                        # log training-related stuff
                        info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                        info = accelerator.reduce(info, reduction="mean")
                        info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                        accelerator.log(info, step=global_step)
                        global_step += 1
                        info = defaultdict(list)

            # make sure we did an optimization step at the end of the inner epoch
            assert accelerator.sync_gradients

        if epoch != 0 and epoch % config.save_freq == 0 and accelerator.is_main_process:
            accelerator.save_state(os.path.join(os.path.join(config.logdir, config.run_name), f"checkpoint_{epoch}"))


if __name__ == "__main__":
    app.run(main)


