import os
import copy
import wandb
import torch
import random
import tempfile
import datetime
import contextlib
from tqdm import tqdm
from collections import defaultdict
from functools import partial

from absl import app, flags
from PIL import Image
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from diffusers import FluxPipeline
from diffusers.training_utils import cast_training_params
from diffusers.utils import convert_unet_state_dict_to_peft
from diffusers.utils.torch_utils import is_compiled_module
from ml_collections import config_flags
from peft import LoraConfig, set_peft_model_state_dict, get_peft_model_state_dict
from torchvision import transforms

import utils.prompts
import utils.rewards
from utils.diffusers_patch.flux_encoder import encode_prompt  # Import custom encoding function
from utils.diffusers_patch.flux_pipeline_with_logprob import flux_pipeline_with_logprob

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
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
    )
    
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="fine-tune-flux-sft",
            config=config.to_dict(),
            init_kwargs={"wandb": {"name": config.run_name}},
        )
    logger.info(f"\n{config}")

    # set seed (device_specific is very important to get different prompts on different devices)
    set_seed(config.seed, device_specific=True)

    # load scheduler, tokenizer and models.
    pipeline = FluxPipeline.from_pretrained(config.pretrained.model, revision=config.pretrained.revision, torch_dtype=torch.bfloat16).to(accelerator.device)

    # freeze parameters of models to save more memory
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.transformer.requires_grad_(not config.use_lora) # pipeline.unet.requires_grad_(not config.use_lora)
    noise_scheduler_copy = copy.deepcopy(pipeline.scheduler)
    
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

    transformer = pipeline.transformer
    if config.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
            
    if config.use_lora:
        # LORA CONFIG ADAPTED FROM:
        # https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora_sdxl.py
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
    eval_prompt_fn = getattr(utils.prompts, config.eval_prompt_fn)
    reward_fn = getattr(utils.rewards, config.reward_fn)(device=accelerator.device) # ensure reward model is on same device

    # for some reason, autocast is necessary for non-lora training but for lora training it isn't necessary and it uses
    # more memory
    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast
    # autocast = accelerator.autocast
    train_dataset = load_dataset(config.dataset_name)[config.dataset_split].select_columns(["implicit_prompt", "explicit_image"])

    def get_dataset_preprocessor():
        # Preprocessing the datasets.
        image_transforms = transforms.Compose(
            [
                transforms.Resize(config.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        def preprocess_train(examples):
            all_pixel_values = []
            col_name = "explicit_image"
        
            for img_list in examples[col_name]:
                pixel_values = [image_transforms(image) for image in img_list]
                all_pixel_values.append(pixel_values)

            examples[col_name] = all_pixel_values
            return examples

        return preprocess_train
    
    preprocess_train_fn = get_dataset_preprocessor()

    with accelerator.main_process_first():
        train_dataset = train_dataset.with_transform(preprocess_train_fn)


    def collate_fn(examples):
        prompt_list = []
        positive_imgs_list = []
        for example in examples:
            idx_cap = random.randrange(0, len(example["implicit_prompt"]))
            idx_pos = random.randrange(0, len(example["explicit_image"]))
            prompt_list.append(example["implicit_prompt"][idx_cap])
            positive_imgs_list.append(example["explicit_image"][idx_pos])

        positive_imgs_tensor = torch.stack(positive_imgs_list)
        return {"prompt": prompt_list, "pixel_values": positive_imgs_tensor}
        
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    # Prepare everything with our `accelerator`.
    transformer, optimizer, train_dataloader = accelerator.prepare(transformer, optimizer, train_dataloader)

    # Train!
    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {config.num_epochs}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}")
    if config.resume_from:
        logger.info(f"Resuming from {config.resume_from}")
        accelerator.load_state(config.resume_from)
        first_epoch = int(config.resume_from.split("_")[-1]) + 1
    else:
        first_epoch = 0


    global_step = 0
    for epoch in tqdm(range(first_epoch, config.num_epochs)):

        #################### EVALUATION ####################
        pipeline.transformer.eval()
        # sample examples for evaluation
        samples = []
        for i in tqdm(
            range(config.eval_num_batches),
            desc=f"Epoch {epoch}: sampling",
            disable=not accelerator.is_local_main_process,
            position=0,
        ): 
            eval_prompts, eval_objects, eval_prompt_metadata = zip(*[eval_prompt_fn(**config.prompt_fn_kwargs) for _ in range(config.eval_batch_size)])

            # encode prompts using custom encoding function to track prompt_ids
            eval_prompt_embeds, eval_pooled_prompt_embeds, eval_text_ids, eval_prompt_ids = encode_prompt(pipeline, eval_prompts, prompt_2=None, device=accelerator.device) 

            with autocast():
                images, latents, log_probs, pipeline_ret_timesteps, step_idxs = flux_pipeline_with_logprob(
                    pipeline,
                    height=config.resolution,
                    width=config.resolution,
                    prompt_embeds=eval_prompt_embeds,
                    pooled_prompt_embeds=eval_pooled_prompt_embeds,
                    num_inference_steps=config.sample.num_steps,
                    guidance_scale=config.sample.guidance_scale,
                    s_churn=config.sample.s_churn,
                    output_type="pt"
                )

            # compute rewards 
            rewards, boxes, metadata = reward_fn(images, eval_prompts, eval_objects, eval_prompt_metadata)

            samples.append({"prompt_ids": eval_prompt_ids, "rewards": rewards})

        # wait for all rewards to be computed
        for sample in tqdm(
            samples,
            desc="Waiting for rewards",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            rewards = sample["rewards"]
            sample["rewards"] = torch.as_tensor(rewards, device=accelerator.device)

        # collate samples into dict where each entry has shape (num_batches_per_epoch * sample.batch_size, ...)
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
                        caption=f"{eval_prompts[i]:.25} | {rewards[i]:.2f}" # only log rewards from process 0
                    )
                )

            accelerator.log({"images": logged_images,},step=global_step)

        # gather rewards across processes
        rewards = accelerator.gather(samples["rewards"]).cpu().numpy()

        # log rewards and images
        accelerator.log(
            {
                "reward": rewards,
                "epoch": epoch,
                "reward_mean": rewards.mean(),
                "reward_std": rewards.std(),
            },
            step=global_step,
        )
            
        def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
            sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
            schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
            timesteps = timesteps.to(accelerator.device)
            step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

            sigma = sigmas[step_indices].flatten()
            while len(sigma.shape) < n_dim:
                sigma = sigma.unsqueeze(-1)
            return sigma
        
        #################### TRAINING ####################
        pipeline.transformer.train()
        for batch in tqdm(train_dataloader):
            
            info = defaultdict(list)

            with accelerator.accumulate(transformer):
                pixel_values = batch["pixel_values"].to(dtype=inference_dtype)
                prompts = batch["prompt"]
                    
                prompt_embeds, pooled_prompt_embeds, text_ids, prompt_ids = encode_prompt(pipeline, prompts, prompt_2=None, device=accelerator.device) 
                
                model_input = pipeline.vae.encode(pixel_values).latent_dist.sample()
                model_input = (model_input - pipeline.vae.config.shift_factor) * pipeline.vae.config.scaling_factor
                model_input = model_input.to(dtype=inference_dtype)

                vae_scale_factor = 2 ** (len(pipeline.vae.config.block_out_channels))
                
                latent_image_ids = FluxPipeline._prepare_latent_image_ids(
                    model_input.shape[0],
                    model_input.shape[2],
                    model_input.shape[3],
                    accelerator.device,
                    inference_dtype,
                )
                
                # Sample noise that we'll add to the latents
                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]

                # Sample a random timestep for each image
                # for weighting schemes where we sample timesteps non-uniformly
                u = torch.rand(size=(bsz,), device="cpu")
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)

                # Add noise according to flow matching.
                # zt = (1 - texp) * x + texp * z1
                sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
                noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

                packed_noisy_model_input = FluxPipeline._pack_latents(
                    noisy_model_input,
                    batch_size=model_input.shape[0],
                    num_channels_latents=model_input.shape[1],
                    height=model_input.shape[2],
                    width=model_input.shape[3],
                )

                # handle guidance
                if transformer.module.config.guidance_embeds:
                    guidance = torch.tensor([config.sample.guidance_scale], device=accelerator.device)
                    guidance = guidance.expand(model_input.shape[0])
                else:
                    guidance = None

                # Predict the noise residual
                model_pred = transformer(
                    hidden_states=packed_noisy_model_input,
                    # YiYi notes: divide it by 1000 for now because we scale it by 1000 in the transforme rmodel (we should not keep it but I want to keep the inputs same for the model for testing)
                    timestep=timesteps / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    return_dict=False,
                )[0]
                
                # upscaling height & width as discussed in https://github.com/huggingface/diffusers/pull/9257#discussion_r1731108042
                model_pred = FluxPipeline._unpack_latents(
                    model_pred,
                    height=int(model_input.shape[2] * vae_scale_factor / 2),
                    width=int(model_input.shape[3] * vae_scale_factor / 2),
                    vae_scale_factor=vae_scale_factor,
                )
                
                # these weighting schemes use a uniform timestep sampling
                # and instead post-weight the loss
                weighting = torch.ones_like(sigmas)

                # flow matching loss
                target = noise - model_input

                # Compute regular loss.
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()

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
                # log training-related stuff
                info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                info = accelerator.reduce(info, reduction="mean")
                info.update({"epoch": epoch})
                accelerator.log(info, step=global_step)
                global_step += 1
                info = defaultdict(list)


        if epoch != 0 and epoch % config.save_freq == 0 and accelerator.is_main_process:
            accelerator.save_state(os.path.join(os.path.join(config.logdir, config.run_name), f"checkpoint_{epoch}"))


if __name__ == "__main__":
    app.run(main)


