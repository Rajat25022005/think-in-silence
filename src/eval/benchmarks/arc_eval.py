import torch
from datasets import load_dataset
from tqdm import tqdm
from typing import Dict


def evaluate_arc(
    model,
    device: torch.device,
    n_steps_list,
    tokenizer,
    max_samples: int = 500,
    split: str = "test"
) -> Dict:
    dataset = load_dataset("ai2_arc", "ARC-Challenge", split=split)
    results  = {}

    num_to_letter = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}

    for n_steps in n_steps_list:
        correct = 0
        total   = 0

        model.eval()
        for sample in tqdm(dataset, desc=f"ARC k={n_steps}", total=min(max_samples, len(dataset))):
            if total >= max_samples:
                break

            question  = sample["question"]
            key       = num_to_letter.get(sample["answerKey"], sample["answerKey"])
            labels    = sample["choices"]["label"]
            texts     = sample["choices"]["text"]
            gold_text = ""
            for label, text in zip(labels, texts):
                if num_to_letter.get(label, label) == key:
                    gold_text = text.strip().lower()
                    break

            enc = tokenizer(
                question, return_tensors="pt",
                padding="max_length", truncation=True, max_length=128
            ).to(device)

            generated = model.generate(enc["input_ids"], enc["attention_mask"], n_steps=n_steps)
            pred      = generated[0].strip().lower() if generated else ""

            if gold_text and (gold_text in pred or pred in gold_text):
                correct += 1
            total += 1

        results[f"k={n_steps}"] = {
            "accuracy": correct / total if total > 0 else 0.0,
            "total": total
        }

    return results
