import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List


def plot_k_scaling(
    results: Dict,
    metric: str = "R@1",
    save_path: str = "results/plots/k_scaling.png"
):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    k_vals    = []
    metric_vals = []

    for key, val in results.items():
        k = int(key.split("=")[1])
        k_vals.append(k)
        metric_vals.append(val[metric])

    k_vals, metric_vals = zip(*sorted(zip(k_vals, metric_vals)))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_vals, metric_vals, marker="o", linewidth=2, markersize=8)
    ax.set_xlabel("Thinking Steps (K)", fontsize=13)
    ax.set_ylabel(metric, fontsize=13)
    ax.set_title(f"{metric} vs Number of Thinking Steps", fontsize=14)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[visualise] Saved K-scaling plot to {save_path}")


def plot_tsne(
    embeddings: np.ndarray,
    labels: List[str],
    save_path: str = "results/plots/tsne.png",
    perplexity: int = 30
):
    from sklearn.manifold import TSNE

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    tsne   = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    coords = tsne.fit_transform(embeddings)

    unique = list(set(labels))
    cmap   = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(10, 8))

    for i, cat in enumerate(unique):
        mask = [l == cat for l in labels]
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   label=cat, alpha=0.6, s=20, color=cmap(i))

    ax.legend(markerscale=2, fontsize=9)
    ax.set_title("t-SNE of Latent Thought Vectors", fontsize=14)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[visualise] Saved t-SNE plot to {save_path}")


def plot_probing_accuracy(
    step_accs: Dict,
    save_path: str = "results/plots/probing.png"
):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    steps = sorted(step_accs.keys())
    accs  = [step_accs[k] for k in steps]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, accs, marker="s", linewidth=2, color="darkorange", markersize=7)
    ax.set_xlabel("Thinking Step k", fontsize=13)
    ax.set_ylabel("Linear Probe Accuracy", fontsize=13)
    ax.set_title("Answer Information Accumulation Across Thinking Steps", fontsize=13)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[visualise] Saved probing plot to {save_path}")
