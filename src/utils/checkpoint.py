import os
from pathlib import Path
from typing import Optional

import torch

from src.utils.device import is_tpu


def _cleanup_old_checkpoints(ckpt_dir: str, keep_last_n: int = 3):
    """Delete all but the most recent `keep_last_n` checkpoints."""
    ckpt_path = Path(ckpt_dir)
    checkpoints = sorted(ckpt_path.glob("step_*.pt"))
    if len(checkpoints) <= keep_last_n:
        return
    for old_ckpt in checkpoints[:-keep_last_n]:
        old_ckpt.unlink(missing_ok=True)


def save_checkpoint(
    model,
    optimizer,
    scaler,
    step: int,
    ckpt_dir: str,
    teacher_model=None,
    extra: Optional[dict] = None,
    keep_last_n: int = 3,
):
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(ckpt_dir, f"step_{step:07d}.pt")

    payload = {
        "step":           step,
        "model":          model.state_dict(),
        "optimizer":      optimizer.state_dict(),
        "scaler":         scaler.state_dict() if scaler is not None else None,
    }
    if teacher_model is not None:
        payload["teacher"] = teacher_model.state_dict()
    if extra:
        payload.update(extra)

    if is_tpu():
        import torch_xla.core.xla_model as xm
        xm.save(payload, path)
    else:
        torch.save(payload, path)

    _cleanup_old_checkpoints(ckpt_dir, keep_last_n=keep_last_n)
    return path


def load_checkpoint(
    path: str,
    model,
    optimizer=None,
    scaler=None,
    teacher_model=None,
    device: str = "cpu"
) -> int:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])

    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    if teacher_model is not None and "teacher" in ckpt:
        teacher_model.load_state_dict(ckpt["teacher"])

    return ckpt.get("step", 0)


def latest_checkpoint(ckpt_dir: str) -> Optional[str]:
    ckpt_path = Path(ckpt_dir)
    if not ckpt_path.exists():
        return None
    checkpoints = sorted(ckpt_path.glob("step_*.pt"))
    return str(checkpoints[-1]) if checkpoints else None
