import argparse
import json
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from llm.config import MODEL_CONFIGS, TEMP_DIR, OUTPUTS_DIR


def get_bf16_flag() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability(0)
    return major >= 8


def load_jsonl(path: Path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def parse_label(text: str) -> int:
    text = text.strip()
    match = re.search(r"\b([01])\b", text)
    if match:
        return int(match.group(1))

    if text.startswith("1"):
        return 1
    if text.startswith("0"):
        return 0

    return 0


def build_model_and_tokenizer(model_key: str):
    model_cfg = MODEL_CONFIGS[model_key]
    model_id = model_cfg["model_id"]
    adapter_path = model_cfg["output_dir"] / "final_adapter"

    tokenizer = AutoTokenizer.from_pretrained(adapter_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if get_bf16_flag() else torch.float16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if get_bf16_flag() else torch.float16,
    )

    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    return model, tokenizer


def generate_label(model, tokenizer, prompt: str) -> tuple[int, str]:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=4,
            do_sample=False,
            temperature=None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_part = full_text[len(prompt):].strip() if full_text.startswith(prompt) else full_text.strip()
    pred_label = parse_label(generated_part)
    return pred_label, generated_part


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["gemma", "llama"], required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    eval_path = TEMP_DIR / "square_eval.jsonl"
    records = load_jsonl(eval_path)

    if args.max_samples is not None:
        records = records[: args.max_samples]

    model, tokenizer = build_model_and_tokenizer(args.model)

    predictions = []
    for idx, record in enumerate(records, start=1):
        pred_label, raw_output = generate_label(model, tokenizer, record["prompt"])

        predictions.append(
            {
                "index": idx - 1,
                "text": record["text"],
                "gold_label": int(record["label"]),
                "pred_label": int(pred_label),
                "category": record.get("category"),
                "raw_output": raw_output,
            }
        )

        if idx % 20 == 0:
            print(f"Processed {idx}/{len(records)}")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUTS_DIR / f"{args.model}_square_predictions.json"

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    print(f"Saved predictions to: {output_file}")


if __name__ == "__main__":
    main()