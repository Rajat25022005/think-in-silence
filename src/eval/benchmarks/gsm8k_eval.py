import re
import torch
from datasets import load_dataset
from tqdm import tqdm
from typing import Dict


def _extract_number(text: str) -> str:
    matches = re.findall(r"-?[\d,]+\.?\d*", text.replace(",", ""))
    return matches[-1] if matches else text.strip()


def evaluate_gsm8k(
    model,
    device: torch.device,
    n_steps_list,
    tokenizer,
    max_samples: int = 500,
    split: str = "test"
) -> Dict:
    dataset = load_dataset("gsm8k", "main", split=split)
    results  = {}

    for n_steps in n_steps_list:
        correct = 0
        total   = 0

        model.eval()
        for sample in tqdm(dataset, desc=f"GSM8K k={n_steps}", total=min(max_samples, len(dataset))):
            if total >= max_samples:
                break

            question = sample["question"]
            gold_raw = sample["answer"]
            gold     = _extract_number(gold_raw.split("####")[-1].strip())

            enc = tokenizer(
                question, return_tensors="pt",
                padding="max_length", truncation=True, max_length=128
            ).to(device)

            generated = model.generate(
                enc["input_ids"], enc["attention_mask"], n_steps=n_steps
            )
            pred = _extract_number(generated[0]) if generated else ""

            if pred == gold:
                correct += 1
            total += 1

        results[f"k={n_steps}"] = {"accuracy": correct / total if total > 0 else 0.0,
                                    "correct": correct, "total": total}

    return results
