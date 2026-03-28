import torch
from typing import List, Dict
from tqdm import tqdm

import evaluate


def evaluate_generation(
    model,
    dataloader,
    device: torch.device,
    n_steps_list: List[int],
    max_samples: int = 500
) -> Dict:
    bleu_metric = evaluate.load("bleu")
    rouge_metric = evaluate.load("rouge")

    results = {}

    for n_steps in n_steps_list:
        predictions = []
        references  = []
        count       = 0

        model.eval()
        tokenizer = model.encoder.tokenizer

        for batch in tqdm(dataloader, desc=f"Generating k={n_steps}", leave=False):
            q_ids  = batch["q_ids"].to(device)
            q_mask = batch["q_mask"].to(device)
            a_ids  = batch["a_ids"].to(device)

            generated = model.generate(q_ids, q_mask, n_steps=n_steps)

            for gen_text, gold_ids in zip(generated, a_ids):
                gold_text = tokenizer.decode(
                    gold_ids.tolist(), skip_special_tokens=True
                ).strip()
                predictions.append(gen_text.strip())
                references.append([gold_text])

            count += q_ids.size(0)
            if count >= max_samples:
                break

        bleu_score  = bleu_metric.compute(predictions=predictions,
                                          references=references)
        rouge_score = rouge_metric.compute(predictions=predictions,
                                           references=[r[0] for r in references])

        results[f"k={n_steps}"] = {
            "bleu":    bleu_score["bleu"],
            "rouge1":  rouge_score["rouge1"],
            "rouge2":  rouge_score["rouge2"],
            "rougeL":  rouge_score["rougeL"],
        }

    return results
