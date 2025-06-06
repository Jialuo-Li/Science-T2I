import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    ############ General ############
    # run name for wandb logging and checkpoint saving -- if not provided, will be auto-generated based on the datetime.
    config.run_name = ""
    # random seed for reproducibility.
    config.seed = 0
    # top-level logging directory for checkpoint saving.
    config.logdir = "logs"
    # number of epochs to train for. each epoch is one round of sampling from the model followed by training on those samples.
    config.num_epochs = 100
    # number of epochs between saving model checkpoints.
    config.save_freq = 1
    # number of checkpoints to keep before overwriting old ones.
    config.num_checkpoint_limit = 100000000
    # mixed precision training. options are "fp16", "bf16", and "no". half-precision speeds up training significantly.
    config.mixed_precision = "bf16"
    # allow tf32 on Ampere GPUs, which can speed up training.
    config.allow_tf32 = True
    # resume training from a checkpoint. either an exact checkpoint directory (e.g. checkpoint_50), or a directory
    # containing checkpoints, in which case the latest one will be used. `config.use_lora` must be set to the same value
    # as the run that generated the saved checkpoint.
    config.resume_from = ""
    # whether or not to use LoRA. LoRA reduces memory usage significantly by injecting small weight matrices into the
    # attention layers of the UNet. with LoRA, fp16, and a batch size of 1, finetuning Stable Diffusion should take
    # about 10GB of GPU memory. beware that if LoRA is disabled, training will take a lot of memory and saved checkpoint
    # files will also be large.
    config.use_lora = True
    # whether or not to use xFormers to reduce memory usage.
    config.use_xformers = False
    # lora rank
    config.lora_rank = 16
    # gradient checkpointing. this reduces memory usage at the cost of some additional compute.
    config.gradient_checkpointing = True
    ############ Pretrained Model ############
    config.pretrained = pretrained = ml_collections.ConfigDict()
    # base model to load. either a path to a local directory, or a model name from the HuggingFace model hub.
    # pretrained.model = "stablediffusionapi/anything-v5"
    pretrained.model = "black-forest-labs/FLUX.1-dev"
    # revision of the model to load.
    pretrained.revision = "main"
    ############ Sampling ############
    config.sample = sample = ml_collections.ConfigDict()
    # number of sampler inference steps.
    sample.num_steps = 30
    # eta parameter for the DDIM sampler. this controls the amount of noise injected into the sampling process, with 0.0
    # being fully deterministic and 1.0 being equivalent to the DDPM sampler.
    sample.eta = 1.0
    # classifier-free guidance weight. 1.0 is no guidance.
    sample.guidance_scale = 0.0
    # control the intensity of additional noise added during sampling
    sample.s_churn = 0.1
    # batch size (per GPU!) to use for sampling.
    sample.batch_size = 2
    # number of batches to sample per epoch. the total number of samples per epoch is `num_batches_per_epoch *
    # batch_size * num_gpus`.
    sample.num_batches_per_epoch = 8
    # save interval
    sample.save_interval = 100
    # denoising split
    sample.denoising_split = 1.0
    ############ Training ############
    config.train = train = ml_collections.ConfigDict()
    # batch size (per GPU!) to use for training.
    train.batch_size = 4
    # whether to use the 8bit Adam optimizer from bitsandbytes.
    train.use_8bit_adam = False
    # learning rate.
    train.learning_rate = 3e-5
    # Adam beta1.
    train.adam_beta1 = 0.9
    # Adam beta2.
    train.adam_beta2 = 0.999
    # Adam weight decay.
    train.adam_weight_decay = 1e-4
    # Adam epsilon.
    train.adam_epsilon = 1e-8
    # number of gradient accumulation steps. the effective batch size is `batch_size * num_gpus *
    # gradient_accumulation_steps`.
    train.gradient_accumulation_steps = 8
    # maximum gradient norm for gradient clipping.
    train.max_grad_norm = 1.0
    # number of inner epochs per outer epoch. each inner epoch is one iteration through the data collected during one
    # outer epoch's round of sampling.
    train.num_inner_epochs = 1
    # enable activation checkpointing or not. 
    # this reduces memory usage at the cost of some additional compute.
    train.activation_checkpoint = True
    # whether or not to use classifier-free guidance during training. if enabled, the same guidance scale used during
    # sampling will be used during training.
    train.cfg = True
    # clip advantages to the range [-adv_clip_max, adv_clip_max].
    train.adv_clip_max = 5
    # the fraction of timesteps to train on. if set to less than 1.0, the model will be trained on a subset of the
    # timesteps for each sample. this will speed up training but reduce the accuracy of policy gradient estimates.
    train.timestep_fraction = sample.denoising_split
    # coefficient of the KL divergence
    train.beta = 10
    # The coefficient constraining the probability ratio. Equivalent to restricting the Q-values within a certain range.
    train.eps = 0.1
    # save_interval
    train.save_interval = 50
    # sample path
    train.sample_path = ""
    # json path
    train.json_path = ""

    ############ Other method Config ############
    # DDPO: clip advantages to the range [-adv_clip_max, adv_clip_max].
    train.adv_clip_max = 5
    # DDPO: the PPO clip range.
    train.clip_range = 3e-4

    ############ Prompt Function ############
    # prompt function to use. see `prompts.py` for available prompt functisons.
    config.prompt_fn = "sci"
    config.eval_prompt_fn = "sci_eval"
    # kwargs to pass to the prompt function.
    config.prompt_fn_kwargs = {}
    

    ############ Reward Function ############
    # reward function to use. see `rewards.py` for available reward functions.
    # if the reward_fn is "jpeg_compressibility" or "jpeg_incompressibility", using the default config can reproduce our results.
    # if the reward_fn is "aesthetic_score" and you want to reproduce our results, 
    # set config.num_epochs = 1000, sample.num_batches_per_epoch=1, sample.batch_size=8 and sample.eval_batch_size=8
    config.reward_fn = "SciScore"

    ############ Dataset ############
    config.dataset_name = "Jialuo21/Science-T2I-Fullset"
    config.dataset_split = "train"

    ############ Evaluation ############
    config.eval_batch_size = 4
    config.eval_num_batches = 8
    # resolution of the images to train on.
    config.resolution = 512
    
    return config
