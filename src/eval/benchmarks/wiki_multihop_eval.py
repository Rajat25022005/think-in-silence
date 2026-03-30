import torch
from datasets import load_dataset
from tqdm import tqdm
from typing import Dict


def evaluate_wiki_multihop(
    model,
    device: torch.device,
    n_steps_list,
    tokenizer,
    max_samples: int = 500,
    split: str = "validation"
) -> Dict:
    dataset = load_dataset("rajat5039/wiki-multihop-qa-500k", None, split=split)
    results  = {}

    for n_steps in n_steps_list:
        exact_match = 0
        total        = 0

        model.eval()
        for sample in tqdm(dataset, desc=f"WikiMultihopQA k={n_steps}", total=min(max_samples, len(dataset))):
            if total >= max_samples:
                break

            question = sample["question"]
            gold     = sample["answer"].strip().lower()

            enc = tokenizer(
                question, return_tensors="pt",
                padding="max_length", truncation=True, max_length=128
            ).to(device)

            generated = model.generate(enc["input_ids"], enc["attention_mask"], n_steps=n_steps)
            pred      = generated[0].strip().lower() if generated else ""

            if pred == gold or gold in pred:
                exact_match += 1
            total += 1

        results[f"k={n_steps}"] = {
            "exact_match": exact_match / total if total > 0 else 0.0,
            "total": total
        }

    return results
