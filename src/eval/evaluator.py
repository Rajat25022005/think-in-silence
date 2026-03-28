import torch
import torch.nn.functional as F
from typing import List, Dict
from tqdm import tqdm


def recall_at_k(
    model,
    dataloader,
    device: torch.device,
    k_values: List[int],
    n_steps_list: List[int],
    max_samples: int = 2000
) -> Dict:
    results = {}

    for n_steps in n_steps_list:
        q_embeds = []
        a_embeds = []
        count    = 0

        model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Encoding k={n_steps}", leave=False):
                q_ids  = batch["q_ids"].to(device)
                q_mask = batch["q_mask"].to(device)
                a_ids  = batch["a_ids"].to(device)
                a_mask = batch["a_mask"].to(device)

                ctx    = model.encoder.encode_question(q_ids, q_mask)
                h      = model.thought(ctx, n_steps=n_steps)
                pred   = model.thought.predict(h)
                target = model.encoder.encode_answer(a_ids, a_mask)

                q_embeds.append(pred.cpu())
                a_embeds.append(target.cpu())

                count += q_ids.size(0)
                if count >= max_samples:
                    break

        q_mat = torch.cat(q_embeds, dim=0)
        a_mat = torch.cat(a_embeds, dim=0)

        q_norm = F.normalize(q_mat, dim=-1)
        a_norm = F.normalize(a_mat, dim=-1)
        sims   = q_norm @ a_norm.T

        step_results = {}
        for k in k_values:
            topk_idx = sims.topk(k, dim=-1).indices
            hits      = sum(
                1 for i, row in enumerate(topk_idx.tolist()) if i in row
            )
            step_results[f"R@{k}"] = hits / len(q_mat)

        results[f"k={n_steps}"] = step_results

    return results
