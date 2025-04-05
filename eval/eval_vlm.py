import torch
from transformers import AutoProcessor, AutoModel
from datasets import load_dataset
from tqdm.auto import tqdm
from collections import defaultdict
import argparse
import json

def calculate_accuracy(probs, dataset):
    correct = sum(1 for prob_0, prob_1 in probs if prob_0 > prob_1)
    return correct / len(dataset)

def calculate_category_accuracy(probs, categories):
    category_metrics = defaultdict(lambda: {"correct": 0, "total": 0})
    for (prob_0, prob_1), category in zip(probs, categories):
        category_metrics[category]["total"] += 1
        if prob_0 > prob_1:
            category_metrics[category]["correct"] += 1
    return {cat: data["correct"] / data["total"] for cat, data in category_metrics.items()}

def calculate_law_accuracy(probs, laws):
    law_metrics = defaultdict(lambda: {"correct": 0, "total": 0})
    for (prob_0, prob_1), law in zip(probs, laws):
        law_metrics[law]["total"] += 1
        if prob_0 > prob_1:
            law_metrics[law]["correct"] += 1
    return {law: data["correct"] / data["total"] for law, data in law_metrics.items()}


############################# Modify the following functions to adapt to your own model #############################  

@torch.no_grad()
def calculate_similarity(images, prompt, model, processor, device):
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

    image_embs = model.get_image_features(**image_inputs)
    image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)

    text_embs = model.get_text_features(**text_inputs)
    text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

    scores = model.logit_scale.exp() * (text_embs @ image_embs.T)[0]
    return torch.softmax(scores, dim=-1).cpu().tolist()

def load_model(model_pretrained_name, processor_pretrained_name, device):
    model = AutoModel.from_pretrained(model_pretrained_name).to(device)
    processor = AutoProcessor.from_pretrained(processor_pretrained_name)
    return model, processor

#########################################################################################################################

def evaluate_dataset(dataset, model, processor, device):

    probs, categories, laws = [], [], []
    for example in tqdm(dataset, desc="Processing examples"):
        prob_0, prob_1 = calculate_similarity(
            [example["explicit_image"][0], example["superficial_image"][0]],
            example["implicit_prompt"],
            model,
            processor,
            device,
        )
        probs.append((prob_0, prob_1))
        categories.append(example["category"])
        laws.append(example["law"])
    return probs, categories, laws

def evaluate_results(dataset_name, processor_pretrained_name, model_pretrained_name):

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print(f"Loading dataset '{dataset_name}'")
    dataset = load_dataset(dataset_name)["test"]

    print(f"Loading model '{model_pretrained_name}'")
    model, processor = load_model(model_pretrained_name, processor_pretrained_name, device)
    probs, categories, laws = evaluate_dataset(dataset, model, processor, device)

    overall_acc = calculate_accuracy(probs, dataset)
    category_acc = calculate_category_accuracy(probs, categories)
    law_acc = calculate_law_accuracy(probs, laws)

    results = {
        "overall_accuracy": overall_acc,
        "category_accuracy": category_acc,
        "law_accuracy": law_acc
    }

    save_path = f"./{str(dataset_name).split('/')[-1]}_vlm_acc.json"
    with open(save_path, 'w') as json_file:
        json.dump(results, json_file, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="Jialuo21/Science-T2I-C",
        help="Name of the dataset.",
    )
    parser.add_argument(
        "--processor_name",
        type=str,
        default="Jialuo21/SciScore",
        help="Name of the processor pretrained model.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Jialuo21/SciScore",
        help="Name of the model pretrained model.",
    )

    args = parser.parse_args()

    evaluate_results(
        dataset_name=args.dataset_name,
        processor_pretrained_name=args.processor_name,
        model_pretrained_name=args.model_name,
    )
