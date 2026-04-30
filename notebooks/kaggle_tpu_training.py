"""
Think-in-Silence — Kaggle TPU Training Notebook
================================================
Runtime: TPU v4-8 or TPU v5e-8 (Kaggle Accelerator)

How to use:
1. Create a new Kaggle Notebook
2. Settings → Accelerator → TPU v4-8
3. Settings → Internet → ON
4. Paste this entire file into a single cell (or split at the marked sections)
5. Run all cells
"""

# ============================================================
# CELL 1: Setup — Clone repo & install dependencies
# ============================================================

import os

# Clone your repo (replace with your actual repo URL)
REPO_URL = "https://github.com/Rajat25022005/think-in-silence.git"
REPO_DIR = "/kaggle/working/think-in-silence"

if not os.path.exists(REPO_DIR):
    os.system(f"git clone {REPO_URL} {REPO_DIR}")

os.chdir(REPO_DIR)

# Install dependencies (torch & torch_xla are pre-installed on Kaggle TPU)
os.system("pip install -q transformers datasets sentence-transformers pyyaml wandb einops tqdm")

# ============================================================
# CELL 2: Verify TPU is available
# ============================================================

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

print(f"PyTorch version:     {torch.__version__}")
print(f"torch_xla version:   {torch_xla.__version__}")

device = xm.xla_device()
print(f"XLA device:          {device}")
print(f"Device type:         {device.type}")

# Quick sanity check — tensor on TPU
t = torch.randn(2, 3, device=device)
print(f"Tensor on TPU:       {t.device}")
print(f"TPU is ready! ✓")

# ============================================================
# CELL 3: Configure & Launch Training
# ============================================================

import sys
sys.path.insert(0, REPO_DIR)

from src.utils.seed    import set_seed
from src.utils.device  import get_device
from src.utils.logging import get_logger
from src.models.lc_thought import LCThought
from src.datasets.qa_datasets import build_dataloader
from src.training.trainer import train_stage1

import yaml
from types import SimpleNamespace

logger = get_logger()


def load_config(path: str) -> SimpleNamespace:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    def _to_ns(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: _to_ns(v) for k, v in d.items()})
        return d

    return _to_ns(raw)


# ── Load config ──────────────────────────────────────────────
cfg = load_config("configs/base.yaml")
cfg.wandb = False  # Set to True if you want WandB logging

# ── Override config for Kaggle TPU ───────────────────────────
# Adjust these based on your TPU memory budget:
cfg.training.batch_size = 16        # Per-core batch size (total = 16 * 8 cores = 128 effective)
cfg.training.max_steps  = 50000     # Adjust as needed
cfg.training.log_every  = 50        # Log more frequently to see progress
cfg.training.ckpt_every = 2500      # Save every 2500 steps
cfg.training.keep_last_n_ckpts = 3  # Only keep last 3 checkpoints

# ── Seed & Device ────────────────────────────────────────────
set_seed(42)
device = get_device()
logger.info(f"Device: {device}")
logger.info(f"Config loaded from: configs/base.yaml")

# ── Build Model ──────────────────────────────────────────────
logger.info("Loading model...")
model     = LCThought(cfg).to(device)
tokenizer = model.encoder.tokenizer

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logger.info(f"Trainable parameters: {n_params:,}")

# ── Build DataLoader ─────────────────────────────────────────
logger.info("Building dataloader...")
dataloader = build_dataloader(cfg, tokenizer)

# ── Train Stage 1 (JEPA) ────────────────────────────────────
logger.info("Starting Stage 1 training on TPU...")
train_stage1(model, dataloader, cfg, device, resume=True)
logger.info("Stage 1 training complete!")

# ============================================================
# CELL 4 (Optional): Stage 2 — Decoder Training
# ============================================================

# Uncomment the block below to run Stage 2 after Stage 1 finishes:
#
# from src.training.decoder_trainer import train_stage2
# from src.utils.checkpoint import latest_checkpoint, load_checkpoint
#
# # Load best stage1 checkpoint into model
# stage1_ckpt = latest_checkpoint(cfg.training.ckpt_dir)
# if stage1_ckpt:
#     load_checkpoint(stage1_ckpt, model, device="cpu")
#     model.to(device)
#     logger.info(f"Loaded stage 1 checkpoint: {stage1_ckpt}")
#
# dataloader = build_dataloader(cfg, tokenizer)
# train_stage2(model, dataloader, cfg, device, resume=True)
# logger.info("Stage 2 training complete!")

# ============================================================
# CELL 5 (Optional): Stage 3 — Joint Fine-tuning
# ============================================================

# Uncomment the block below to run Stage 3:
#
# from src.training.finetune_trainer import train_stage3
# from src.utils.checkpoint import latest_checkpoint, load_checkpoint
#
# stage2_dir  = cfg.training.ckpt_dir.replace("stage1", "stage2")
# stage2_ckpt = latest_checkpoint(stage2_dir)
# if stage2_ckpt:
#     load_checkpoint(stage2_ckpt, model, device="cpu")
#     model.to(device)
#     logger.info(f"Loaded stage 2 checkpoint: {stage2_ckpt}")
#
# dataloader = build_dataloader(cfg, tokenizer)
# train_stage3(model, dataloader, cfg, device, resume=True)
# logger.info("Stage 3 training complete!")

# ============================================================
# CELL 6 (Optional): Save final model to Kaggle output
# ============================================================

# import shutil
# output_dir = "/kaggle/working/final_model"
# shutil.copytree("checkpoints", output_dir, dirs_exist_ok=True)
# logger.info(f"Final checkpoints copied to {output_dir}")
