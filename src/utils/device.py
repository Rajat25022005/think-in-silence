import torch


def _has_tpu() -> bool:
    try:
        import torch_xla.core.xla_model as xm
        dev = xm.xla_device()
        return dev is not None
    except Exception:
        return False


_TPU_AVAILABLE = None


def is_tpu() -> bool:
    global _TPU_AVAILABLE
    if _TPU_AVAILABLE is None:
        _TPU_AVAILABLE = _has_tpu()
    return _TPU_AVAILABLE


def get_device() -> torch.device:
    if is_tpu():
        import torch_xla.core.xla_model as xm
        return xm.xla_device()
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def move_batch(batch: dict, device: torch.device) -> dict:
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }
