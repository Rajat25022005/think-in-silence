import torch
import numpy as np
from typing import List, Dict
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


def probe_intermediate_states(
    model,
    dataloader,
    device: torch.device,
    n_steps: int,
    max_samples: int = 2000
) -> Dict:
    all_states  = []
    all_labels  = []
    count       = 0

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Collecting states", leave=False):
            q_ids  = batch["q_ids"].to(device)
            q_mask = batch["q_mask"].to(device)
            a_ids  = batch["a_ids"].to(device)
            a_mask = batch["a_mask"].to(device)

            ctx       = model.encoder.encode_question(q_ids, q_mask)
            states    = model.thought(ctx, n_steps=n_steps, return_all_states=True)
            target    = model.encoder.encode_answer(a_ids, a_mask)

            all_states.append(states.cpu())

            correct = (model.thought.predict(states[:, -1]) - target).norm(dim=-1) < 1.0
            all_labels.extend(correct.cpu().tolist())

            count += q_ids.size(0)
            if count >= max_samples:
                break

    states_tensor = torch.cat(all_states, dim=0)
    n_samples, n_steps_stored, _, dim = states_tensor.shape

    step_accs = {}
    for k in range(n_steps_stored):
        step_vecs = states_tensor[:, k, 0, :].numpy()
        labels    = np.array(all_labels[:n_samples]).astype(int)

        scaler     = StandardScaler()
        step_vecs  = scaler.fit_transform(step_vecs)

        split      = int(0.8 * n_samples)
        clf        = LogisticRegression(max_iter=200, C=1.0, solver="lbfgs")
        clf.fit(step_vecs[:split], labels[:split])
        preds      = clf.predict(step_vecs[split:])
        acc        = accuracy_score(labels[split:], preds)
        step_accs[k] = acc

    return {"step_probing_accuracy": step_accs}
