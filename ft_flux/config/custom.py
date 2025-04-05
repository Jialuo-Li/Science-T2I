import ml_collections
import imp
import os

base = imp.load_source("base", os.path.join(os.path.dirname(__file__), "base.py"))


def sft():
    config = base.get_config()

    config.pretrained.model = "black-forest-labs/FLUX.1-dev"
    config.num_epochs = 60

    config.use_lora = True
    config.lora_rank = 16
    
    config.gradient_checkpointing = False

    config.save_freq = 1
    config.num_checkpoint_limit = 100000000
    config.logdir = "sft"

    config.sample.num_steps = 30 # inference step
    config.sample.guidance_scale = 0.0

    # for evaluation
    config.eval_batch_size = 4
    config.eval_num_batches = 8

    config.train.batch_size = 4
    config.train.gradient_accumulation_steps = 8
    config.train.learning_rate = 2e-5

    # dataset
    config.dataset_name = "Jialuo21/Science-T2I-Fullset"
    config.dataset_split = "train"
    config.resolution = 512
    
    # prompting
    config.eval_prompt_fn = "sci_eval"
    config.prompt_fn_kwargs = {}

    # rewards
    config.reward_fn = "SciScore"

    return config

def oft():
    config = sft()
    config.num_epochs = 100
    config.logdir = "oft"

    config.gradient_checkpointing = True
    # for sampling when training
    config.sample.batch_size = 2 # must be 2 if use dpo loss
    config.sample.num_batches_per_epoch = 8 # control the number of samples per epoch
    config.sample.s_churn = 0.1

    config.train.batch_size = config.sample.batch_size # must be same as sample.batch_size when using DPO loss
    config.train.gradient_accumulation_steps = 2
    config.train.learning_rate = 3e-4
    config.train.beta = 10 # DPO loss beta

    config.resume_from = "./utils/assets/sft_ckpt/checkpoint_60"
    # prompting
    config.prompt_fn = "sci"

    # evaluation
    config.eval_prompt_fn = "sci_eval"
    config.eval_batch_size = 4
    config.eval_num_batches = 8

    return config

def oft_white():

    config = sft()
    config.num_epochs = 100
    config.logdir = "oft_white"

    config.pretrained.model = "black-forest-labs/FLUX.1-schnell"
    config.sample.num_steps = 4 # inference step

    # for sampling when training
    config.sample.batch_size = 2 # must be 2 if use dpo loss
    config.sample.num_batches_per_epoch = 8 # control the number of samples per epoch
    config.sample.s_churn = 0.1

    config.train.batch_size = config.sample.batch_size # must be same as sample.batch_size when using DPO loss
    config.train.gradient_accumulation_steps = 2
    config.train.learning_rate = 2e-4
    config.train.beta = 10 # DPO loss beta

    # prompting
    config.prompt_fn = "simple_animals"
    config.eval_prompt_fn = "simple_animals_test"
    config.reward_fn = "light_reward"
    return config


def oft_aes_score():

    config = sft()
    config.num_epochs = 100
    config.logdir = "oft_aes"

    config.pretrained.model = "black-forest-labs/FLUX.1-schnell"
    config.sample.num_steps = 4 # inference step

    # for sampling when training
    config.sample.batch_size = 2 # must be 2 if use dpo loss
    config.sample.num_batches_per_epoch = 16 # control the number of samples per epoch
    config.sample.s_churn = 0.1

    config.train.batch_size = config.sample.batch_size # must be same as sample.batch_size when using DPO loss
    config.train.gradient_accumulation_steps = 8
    config.train.learning_rate = 6e-5
    config.train.beta = 10 # DPO loss beta

    # prompting
    config.prompt_fn = "simple_animals"
    config.eval_prompt_fn = "simple_animals_test"
    config.reward_fn = "aesthetic_score"
    return config

def oft_image_reward():

    config = sft()
    config.num_epochs = 100
    config.logdir = "oft_imgrw"

    config.pretrained.model = "black-forest-labs/FLUX.1-schnell"
    config.sample.num_steps = 4 # inference step

    # for sampling when training
    config.sample.batch_size = 2 # must be 2 if use dpo loss
    config.sample.num_batches_per_epoch = 16 # control the number of samples per epoch
    config.sample.s_churn = 0.1

    config.train.batch_size = config.sample.batch_size # must be same as sample.batch_size when using DPO loss
    config.train.gradient_accumulation_steps = 8
    config.train.learning_rate = 6e-5
    config.train.beta = 10 # DPO loss beta

    # prompting
    config.prompt_fn = "simple_animals"
    config.eval_prompt_fn = "simple_animals_test"
    config.reward_fn = "ImageReward"
    return config


def get_config(name):
    return globals()[name]()