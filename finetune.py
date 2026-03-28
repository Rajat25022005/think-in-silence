import argparse
import yaml
from types import SimpleNamespace

import torch

from src.utils.seed     import set_seed
from src.utils.device   import get_device
from src.utils.logging  import get_logger
from src.utils.checkpoint import latest_checkpoint, load_checkpoint
from src.models.lc_thought import LCThought
from src.datasets.qa_datasets import build_dataloader
from src.training.finetune_trainer import train_stage3


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
    parser = argparse.ArgumentParser(description="Think-in-Silence — Stage 3 Joint Fine-tuning")
    parser.add_argument("--config",       type=str, default="configs/base.yaml")
    parser.add_argument("--stage2_ckpt",  type=str, default=None,
                        help="Path to stage 2 checkpoint to initialize from")
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--wandb",        action="store_true")
    parser.add_argument("--no_resume",    action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg.wandb = args.wandb

    set_seed(args.seed)
    device = get_device()
    logger.info(f"Device: {device} | Config: {args.config}")

    model     = LCThought(cfg, vocab_size=30522).to(device)
    tokenizer = model.encoder.tokenizer

    stage2_dir  = cfg.training.ckpt_dir.replace("stage1", "stage2")
    stage2_ckpt = args.stage2_ckpt or latest_checkpoint(stage2_dir)
    if stage2_ckpt:
        load_checkpoint(stage2_ckpt, model, device=str(device))
        logger.info(f"Initialized from stage 2 checkpoint: {stage2_ckpt}")

    dataloader = build_dataloader(cfg, tokenizer)
    train_stage3(model, dataloader, cfg, device, resume=not args.no_resume)


if __name__ == "__main__":
    main()
