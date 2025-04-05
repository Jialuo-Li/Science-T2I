import argparse
import torch
import json
import copy
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from datasets import load_dataset
from tqdm import tqdm

TASK_PROMPT = """You are presented with a prompt followed by two images. Your task is to critically analyze and compare both images, selecting the one that most accurately aligns with and represents the overall meaning of the given prompt.\nPlease answer in the json format {"choice": "image-1" / "image-2"}.\n"""
DEVICE = 'cuda:0'

def get_response(data, tokenizer, model, image_processor, reverse):
    prompt_text = data['implicit_prompt']
    pos_img = data['explicit_image'][0]
    neg_img = data['superficial_image'][0]

    images = [neg_img, pos_img] if reverse else [pos_img, neg_img]

    prompt = f"Image-1: {DEFAULT_IMAGE_TOKEN}\nImage-2: {DEFAULT_IMAGE_TOKEN}\nPrompt: {prompt_text}\n"
    content = prompt + TASK_PROMPT

    ############################# Modify the code to adapt to your own model #############################
    image_tensor = process_images(images, image_processor, model.config)
    image_tensor = [_image.to(dtype=torch.float16, device=DEVICE) for _image in image_tensor]

    conv = copy.deepcopy(conv_templates["qwen_1_5"])
    conv.append_message(conv.roles[0], content)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(DEVICE)
    image_sizes = [img.size for img in images]

    output_ids = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=image_sizes,
        do_sample=False,
        temperature=0,
        max_new_tokens=4096,
    )
    text_output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    #####################################################################################################
    try:
        return json.loads(text_output)
    except json.JSONDecodeError:
        return {"choice": text_output}

def evaluate_dataset(dataset, tokenizer, model, image_processor, reverse):
    acc_dict_law = {}
    acc_dict_category = {}

    for batch in tqdm(dataset):
        score = get_response(batch, tokenizer, model, image_processor, reverse)
        choice = score.get("choice", "")

        law = batch['law']
        category = batch['category']

        if law not in acc_dict_law:
            acc_dict_law[law] = [0, 0]
        if category not in acc_dict_category:
            acc_dict_category[category] = [0, 0]

        if (not reverse and choice == "image-1") or (reverse and choice == "image-2"):
            acc_dict_law[law][0] += 1
            acc_dict_category[category][0] += 1

        acc_dict_law[law][1] += 1
        acc_dict_category[category][1] += 1

    return acc_dict_law, acc_dict_category

def calculate_accuracy(acc_dict):
    total_correct = sum(v[0] for v in acc_dict.values())
    total_samples = sum(v[1] for v in acc_dict.values())
    total_accuracy = total_correct / total_samples if total_samples > 0 else 0
    accuracy_per_category = {k: v[0] / v[1] if v[1] > 0 else 0 for k, v in acc_dict.items()}
    return accuracy_per_category, total_accuracy

def main(dataset_name):
    ############################# Modify the code to adapt to your own model #############################
    pretrained = "lmms-lab/llava-onevision-qwen2-7b-ov"
    model_name = "llava_qwen"
    llava_model_args = {"multimodal": True, "overwrite_config": {"image_aspect_ratio": "pad"}}

    tokenizer, model, image_processor, _ = load_pretrained_model(pretrained, None, model_name, device_map=None, **llava_model_args)
    model = model.to(DEVICE).eval()
    #####################################################################################################
    dataset = load_dataset(dataset_name)['test'].select_columns(['implicit_prompt', 'category', 'law', 'explicit_image', 'superficial_image'])

    acc_law, acc_category = evaluate_dataset(dataset, tokenizer, model, image_processor, reverse=False)
    acc_law_re, acc_category_re = evaluate_dataset(dataset, tokenizer, model, image_processor, reverse=True)

    law_accuracy, total_accuracy = calculate_accuracy(acc_law)
    category_accuracy, _ = calculate_accuracy(acc_category)
    law_accuracy_re, total_accuracy_re = calculate_accuracy(acc_law_re)
    category_accuracy_re, _ = calculate_accuracy(acc_category_re)

    final_law_accuracy = {k: (law_accuracy[k] + law_accuracy_re[k]) / 2 for k in law_accuracy}
    final_category_accuracy = {k: (category_accuracy[k] + category_accuracy_re[k]) / 2 for k in category_accuracy}
    final_total_accuracy = (total_accuracy + total_accuracy_re) / 2

    results = {
        'law': law_accuracy,
        'category': category_accuracy,
        'total': total_accuracy,
        'law_re': law_accuracy_re,
        'category_re': category_accuracy_re,
        'total_re': total_accuracy_re,
        'final_law': final_law_accuracy,
        'final_category': final_category_accuracy,
        'final_total': final_total_accuracy,
    }

    save_path = f"./{str(dataset_name).split('/')[-1]}_lmm_acc.json"
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a dataset using a pretrained model.")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset to evaluate.")
    args = parser.parse_args()

    main(args.dataset_name)