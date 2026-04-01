import argparse
import json
import yaml
from pathlib import Path
from types import SimpleNamespace

import torch

from src.utils.seed      import set_seed
from src.utils.device    import get_device
from src.utils.logging   import get_logger
from src.utils.checkpoint import latest_checkpoint, load_checkpoint
from src.models.lc_thought import LCThought
from src.datasets.qa_datasets import build_dataloader
from src.eval.evaluator      import recall_at_k
from src.eval.generation_eval import evaluate_generation
from src.eval.visualise      import plot_k_scaling


logger = get_logger()


def load_config(path: str) -> SimpleNamespace:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    def _to_ns(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: _to_ns(v) for k, v in d.items()})
        return d

    return _to_ns(raw)


def main():
    parser = argparse.ArgumentParser(description="Think-in-Silence — Evaluation")
    parser.add_argument("--config",     type=str, default="configs/base.yaml")
    parser.add_argument("--ckpt",       type=str, default=None)
    parser.add_argument("--ckpt_dir",   type=str, default=None)
    parser.add_argument("--eval_type",  type=str, default="retrieval",
                        choices=["retrieval", "generation", "all"])
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(args.seed)
    device = get_device()
    logger.info(f"Device: {device}")

    model     = LCThought(cfg).to(device)
    tokenizer = model.encoder.tokenizer

    ckpt_path = args.ckpt
    if ckpt_path is None and args.ckpt_dir:
        ckpt_path = latest_checkpoint(args.ckpt_dir)
    if ckpt_path is None:
        ckpt_path = latest_checkpoint(cfg.training.ckpt_dir)

    if ckpt_path:
        load_checkpoint(ckpt_path, model, device=str(device))
        logger.info(f"Loaded checkpoint: {ckpt_path}")
    else:
        logger.warning("No checkpoint found. Evaluating with random init.")

    model.eval()
    dataloader  = build_dataloader(cfg, tokenizer)
    k_list      = cfg.eval.k_values
    results_dir = Path("results/metrics")
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.eval_type in ("retrieval", "all"):
        recall_results = recall_at_k(
            model, dataloader, device,
            k_values=[1, 5, 10],
            n_steps_list=k_list,
            max_samples=args.max_samples
        )
        logger.info("Recall@K results:")
        for k, metrics in recall_results.items():
            logger.info(f"  {k}: {metrics}")

        with open(results_dir / "recall.json", "w") as f:
            json.dump(recall_results, f, indent=2)

        plot_k_scaling(
            {k: {"R@1": v["R@1"]} for k, v in recall_results.items()},
            metric="R@1",
            save_path="results/plots/k_scaling_recall.png"
        )

    if args.eval_type in ("generation", "all"):
        gen_results = evaluate_generation(
            model, dataloader, device,
            n_steps_list=k_list,
            max_samples=args.max_samples
        )
        logger.info("Generation results:")
        for k, metrics in gen_results.items():
            logger.info(f"  {k}: {metrics}")

        with open(results_dir / "generation.json", "w") as f:
            json.dump(gen_results, f, indent=2)

        plot_k_scaling(
            {k: {"BLEU": v["bleu"]} for k, v in gen_results.items()},
            metric="BLEU",
            save_path="results/plots/k_scaling_bleu.png"
        )


if __name__ == "__main__":
    main()
