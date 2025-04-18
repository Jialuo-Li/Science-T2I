<div align="center" style="font-family: charter;">
<h1><img src="examples/icon.png" width="5%"/>&nbsp;<i>Science-T2I</i>:</br>Addressing Scientific Illusions in Image Synthesis</h1>


<a href="https://arxiv.org/pdf/2504.13129/" target="_blank">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-Science--T2I-red?logo=arxiv" height="20" /></a>
<a href="https://jialuo-li.github.io/Science-T2I-Web/" target="_blank">
    <img alt="Website" src="https://img.shields.io/badge/🌎_Website-Science--T2I-blue.svg" height="20" /></a>
<a href="https://huggingface.co/collections/Jialuo21/science-t2i-67d3bfe43253da2bc7cfaf06" target="_blank">
    <img alt="HF Dataset: Science-T2I" src="https://img.shields.io/badge/%F0%9F%A4%97%20_Hugging Face-Science--T2I-ffc107?color=ffc107&logoColor=white" height="20" /></a>

<div>
    <a href="https://jialuo-li.github.io/" target="_blank">Jialuo Li</a><sup>1</sup>,</span>
    <a href="https://rese1f.github.io/" target="_blank">Wenhao Chai</a><sup>2</sup>, </span>
    <a href="https://zeyofu.github.io/" target="_blank">Xingyu Fu</a><sup>3</sup>,</span>
    <a href="https://xxuhaiyang.github.io/" target="_blank">Haiyang Xu</a><sup>4</sup>,</span>
    <a href="https://www.sainingxie.com/" target="_blank">Saining Xie</a><sup>1</sup></span>
</div>

<div>
    <sup>1</sup>New York University&emsp;
    <sup>2</sup>University of Washington&emsp;
    <sup>3</sup>University of Pennsylvania&emsp;
    <sup>4</sup>University of California, San Diego&emsp;
</div>

<img src="examples/teaser.png" width="100%"/>

<p align="justify"><i>Given a prompt (in grey) requiring scientific knowledge, FLUX generates imaginary images (lower row) that are far from reality (upper row). Moreover, LMMs like GPT-4o fail to identify the realistic image, whereas our end-to-end reward model succeeds. Notice that the prompts here are summarization of the real prompts that we used for illustration purposes.</i></p>

</div>      

## :fire: News

- [2025/4/18] Release [paper](https://arxiv.org/abs/2504.13129).
- [2025/4/05] Release Science-T2I dataset, as well as the training and evaluation code.

## ✨ Quick Start  

### Installation

We recommend installing Science-T2I in a virtual environment from Conda (Python>=3.10).
```
conda create -n science-t2i python=3.10
conda activate science-t2i
```
Clone the repository and the submodule.
```
git clone git@github.com:Jialuo-Li/Science-T2I.git
cd Science-T2I
git submodule update --init
```
Install PyTorch following [instruction](https://pytorch.org/get-started/locally/).
```
pip install torch torchvision
```
Install additional dependencies.
```
pip install -r requirements.txt
```

## 🚀 Benchmark: Science-T2I-S&C

### Introduction
In addition to the Science-T2I training dataset, we have also curated two novel benchmarks specifically designed for evaluating vision-based scientific understanding tasks: Science-T2I-S and Science-T2I-C . These benchmarks contain 671 and 227 tuples, respectively. Each tuple consists of:
- An **implicit prompt** and its corresponding **explicit prompt**, **superficial prompt**.
- Two images: one that aligns with the explicit prompt and another that corresponds to the superficial prompt.

We encourage you to evaluate your models on our benchmarks and submit a pull request with your results to refresh the [Leaderboard](https://jialuo-li.github.io/Science-T2I-Web/)!

### Evaluation on VLM
To evaluate VLMs using our benchmarks, we provide an example script for assessing SciScore on the Science-T2I-S benchmark. You can adapt this script by modifying the input arguments suit your specific VLM.
```
python eval/eval_vlm.py \
  --dataset_name Jialuo21/Science-T2I-S \
  --processor_name Jialuo21/SciScore \
  --model_name Jialuo21/SciScore
```

### Evaluation on LMM
For evaluating LMMs, we offer an example script to assess [LLaVA-OV](https://github.com/LLaVA-VL/LLaVA-NeXT/blob/main/docs/LLaVA_OneVision.md) on the Science-T2I-S benchmark. To adapt this script for your own LMM, simply modify the dataset name and adjust the code accordingly.

First, install the required LLaVA-OV package:
```
pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git einops flash_attn
```
Then, run the evaluation script:
```
python eval/eval_lmm.py \
  --dataset_name Jialuo21/Science-T2I-S
```


## 🤖 Reward model: SciScore
### Inference with SciScore
We display here an example for running inference with SciScore:
```python
from transformers import AutoProcessor, AutoModel
from PIL import Image
import torch

device = "cuda"
processor_name_or_path = "Jialuo21/SciScore"
model_pretrained_name_or_path = "Jialuo21/SciScore"

processor = AutoProcessor.from_pretrained(processor_name_or_path)
model = AutoModel.from_pretrained(model_pretrained_name_or_path).eval().to(device)

def calc_probs(prompt, images):
    image_inputs = processor(
        images=images,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)
    text_inputs = processor(
        text=prompt,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        image_embs = model.get_image_features(**image_inputs)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
        text_embs = model.get_text_features(**text_inputs)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
        scores = model.logit_scale.exp() * (text_embs @ image_embs.T)[0]
        probs = torch.softmax(scores, dim=-1)
    return probs.cpu().tolist()

pil_images = [Image.open("./examples/camera_1.png"), Image.open("./examples/camera_2.png")]
prompt = "A camera screen without electricity sits beside the window, realistic."
print(calc_probs(prompt, pil_images))
```

### Benchmarking T2I Models
Using SciScore, you can assess how well T2I models align with real-world scenarios in our predefined tasks. Below is an example evaluation script for testing [FLUX.1\[schnell\]](https://huggingface.co/black-forest-labs/FLUX.1-schnell) on SciScore, utilizing the prompts from the Science-T2I-S dataset:
```
accelerate launch eval/eval_t2i_with_SciScore.py \
  --dataset_name Jialuo21/Science-T2I-S
```

### Train SciScore from Scratch
To train SciScore from scratch, execute the following commands. This process takes approximately one hour on a system with 8 A6000 GPUs.
```
pip install deepspeed==0.14.5 # First install deepspeed for training
cd SciScore_trainer
bash train_sciscore.sh
```

## ⚡ Two-Stage Fine-tuning on FLUX
### Installation
Install [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) dependencies and download pretrained weights.
```
cd ft_flux/GroundingDINO
pip install -e .
mkdir -p weights && cd weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd ..
pip uninstall deepspeed # Uninstall deepspeed if it's currently installed (not needed for this section)
```

### Stage 1: Supervised Fine-tuning (SFT)
We begin by performing supervised fine-tuning (SFT) on FLUX-1.\[dev\] for domain adaptation, using the training set from Science-T2I. The example command to run this stage is:
```
accelerate launch sft_flux.py --config config/custom.py:sft
```

### Stage 2: Online Fine-tuning (OFT)
In this stage, we further fine-tune FLUX.1\[dev\] using online fine-tuning (OFT) with the DPO training objective, SciScore is used as reward model to guide the optimization process. The example command is: 
```
accelerate launch oft_flux.py --config config/custom.py:oft
```
#### Additional Reward Model Option
We also provide examples for fine-tuning FLUX with different reward models.
* Whiteness Reward (higher reward for whiter images):
```
accelerate launch oft_flux.py --config config/custom.py:oft_white
```
* [ImageReward](https://github.com/THUDM/ImageReward):
```
pip install image-reward
accelerate launch oft_flux.py --config config/custom.py:oft_image_reward
```
* [LAION-Aesthetic-Scorer](https://github.com/LAION-AI/aesthetic-predictor)
```
accelerate launch oft_flux.py --config config/custom.py:oft_aes_score
```

## 🤝 Acknowledgement
We are deeply grateful for the following GitHub repositories, as their valuable code and efforts have been incredibly helpful:

* PickScore (https://github.com/yuvalkirstain/PickScore)
* DDPO (https://github.com/kvablack/ddpo-pytorch)
* Diffusers (https://github.com/huggingface/diffusers)
* GroundingDINO (https://github.com/IDEA-Research/GroundingDINO)
* ImageReward (https://github.com/THUDM/ImageReward)

## ✏️ Citation

If you find Science-T2I useful for your your research and applications, please cite using this BibTeX:

```bibtex
@misc{li2025sciencet2iaddressingscientificillusions,
      title={Science-T2I: Addressing Scientific Illusions in Image Synthesis}, 
      author={Jialuo Li and Wenhao Chai and Xingyu Fu and Haiyang Xu and Saining Xie},
      year={2025},
      eprint={2504.13129},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2504.13129}, 
}
```