import argparse
import base64
import concurrent.futures
import json
import re
import urllib.request
from collections import defaultdict
from pathlib import Path

import torch
from accelerate import Accelerator
from datasets import load_dataset
from diffusers import FluxPipeline
from torch.utils.data import DataLoader

SYSTEM_PROMPT = (
    "You are an experienced scientist. Begin by evaluating the provided image using the specified\n"
    "scene composition criteria. If the image does not fully satisfy these criteria, assign a reality score of\n"
    "0. However, if the scene meets all the criteria, proceed to assess its realism based on the given reality\n"
    "scoring guidelines, disregarding stylistic aspects and minor background details. Please first describe the\n"
    "image in detail and then adhere strictly to these criteria to ensure an accurate scoring of the image.\n"
    'Please present your evaluation in the following format: {"description":, "scene score": , "reality score": }'
)

PROMPT_TEMPLATE = (
    'Input: {{"Prompt": "{prompt}", "Scene Grading": "{scene_grading}", '
    '"Reality Grading": "{reality_grading}"}}.\n'
    'Please present your evaluation in the following format: {{"description":, "scene score": , "reality score": }}'
)


def parse_score_response(text_output):
    """Extract overall/reality score from LMM JSON response."""
    payload = None
    try:
        payload = json.loads(text_output)
    except json.JSONDecodeError:
        for match in re.finditer(r"\{[^{}]*\}", text_output):
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict) and (
                "scene score" in parsed or "reality score" in parsed or "overall score" in parsed
            ):
                payload = parsed
                break
        if payload is None:
            start, end = text_output.find("{"), text_output.rfind("}")
            if start != -1 and end > start:
                try:
                    payload = json.loads(text_output[start : end + 1])
                except json.JSONDecodeError:
                    pass

    if not isinstance(payload, dict):
        return None
    for key in ("overall score", "reality score"):
        val = payload.get(key)
        if val is None:
            continue
        if isinstance(val, (int, float)):
            return float(val)
        match = re.search(r"-?\d+(\.\d+)?", str(val))
        if match:
            return float(match.group(0))
    return None


def request_openai_chat(args, prompt_text, image_path):
    """Send image + prompt to an OpenAI-compatible multimodal API."""
    image_b64 = base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")
    payload = {
        "model": args.api_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                    {"type": "text", "text": prompt_text},
                ],
            },
        ],
        "temperature": 0,
        "max_tokens": args.max_new_tokens,
    }
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"

    req = urllib.request.Request(
        args.api_base.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body.get("choices", [{}])[0].get("message", {}).get("content", "")


def score_one_record(record, args):
    """Score a single generated image via the LMM API."""
    prompt_text = PROMPT_TEMPLATE.format(
        prompt=record["prompt"],
        scene_grading=record["scene_grading"],
        reality_grading=record["reality_grading"],
    )
    try:
        output_text = request_openai_chat(args, prompt_text, record["image_path"])
    except Exception:
        return record, None
    return record, parse_score_response(output_text)


def generate_images(args, accelerator, pipe, output_dir):
    """Generate images for each test sample and save metadata."""
    cols = ["implicit_prompt", "scene_scoring", "real_scoring", "category", "law"]
    dataset = load_dataset(args.dataset_name)["test"].select_columns(cols)
    dataloader = accelerator.prepare(DataLoader(dataset, batch_size=1))

    output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = output_dir / f"meta_rank{accelerator.process_index}.jsonl"

    with meta_path.open("w", encoding="utf-8") as f:
        for batch_idx, batch in enumerate(dataloader):
            prompt = batch["implicit_prompt"]
            if isinstance(prompt, (list, tuple)):
                prompt = prompt[0]
            if not prompt:
                continue

            for sample_idx in range(args.num_samples):
                image = pipe(
                    prompt,
                    guidance_scale=args.guidance_scale,
                    num_inference_steps=args.num_inference_steps,
                    max_sequence_length=256,
                    height=args.height,
                    width=args.width,
                ).images[0]
                image_path = output_dir / f"rank{accelerator.process_index}_{batch_idx}_{sample_idx}.png"
                image.save(image_path)

                get = lambda key, default="": (lambda v: v[0] if isinstance(v, (list, tuple)) else v)(batch.get(key, default))
                meta = {
                    "prompt": prompt,
                    "scene_grading": get("scene_scoring"),
                    "reality_grading": get("real_scoring"),
                    "category": get("category", "unknown"),
                    "law": get("law", "unknown"),
                    "image_path": str(image_path),
                }
                f.write(json.dumps(meta, ensure_ascii=False) + "\n")


def load_meta_records(output_dir):
    records = []
    for path in sorted(output_dir.glob("meta_rank*.jsonl")):
        with path.open("r", encoding="utf-8") as f:
            records.extend(json.loads(line) for line in f)
    return records


def compute_scores(records, args):
    """Score all records via LMM API and aggregate by category/law."""
    scores, parse_failures = [], 0
    category_scores = defaultdict(list)
    law_scores = defaultdict(list)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.api_concurrency) as pool:
        futures = [pool.submit(score_one_record, r, args) for r in records]
        for future in concurrent.futures.as_completed(futures):
            meta, score = future.result()
            if score is None:
                parse_failures += 1
                continue
            scores.append(score)
            category_scores[str(meta.get("category", "unknown"))].append(score)
            law_scores[str(meta.get("law", "unknown"))].append(score)

    avg = lambda s: sum(s) / len(s) / 3.0 if s else 0.0
    return {
        "overall_score": avg(scores),
        "num_scored_samples": len(scores),
        "parse_failures": parse_failures,
        "category_scores": {k: avg(v) for k, v in category_scores.items()},
        "law_scores": {k: avg(v) for k, v in law_scores.items()},
    }


def main(args):
    accelerator = Accelerator()
    output_dir = Path(args.output_dir)

    pipe = FluxPipeline.from_pretrained(args.t2i_model, torch_dtype=torch.bfloat16)
    pipe.to(accelerator.device)
    generate_images(args, accelerator, pipe, output_dir)

    del pipe
    torch.cuda.empty_cache()
    accelerator.wait_for_everyone()

    if not accelerator.is_main_process:
        return

    records = load_meta_records(output_dir)
    result = compute_scores(records, args)

    save_path = Path("./t2i_lmm_scores.json")
    with save_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)
    print(f"overall_score={result['overall_score']:.4f}")


def build_parser():
    p = argparse.ArgumentParser(description="Evaluate T2I models on Science-T2I using a multimodal LMM judge.")
    p.add_argument("--dataset_name", type=str, default="Jialuo21/Science-T2I")
    p.add_argument("--num_samples", type=int, default=1)
    p.add_argument("--guidance_scale", type=float, default=4.0)
    p.add_argument("--num_inference_steps", type=int, default=4)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--output_dir", type=str, default="./t2i_lmm_cache")
    p.add_argument("--t2i_model", type=str, default="black-forest-labs/FLUX.1-schnell")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--api_base", type=str, default="http://127.0.0.1:8080/v1")
    p.add_argument("--api_model", type=str, default="Qwen3-VL")
    p.add_argument("--api_key", type=str, default="")
    p.add_argument("--api_concurrency", type=int, default=32)
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
