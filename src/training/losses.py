import torch
import torch.nn.functional as F


def jepa_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse  = F.mse_loss(pred, target.detach())
    cos  = 1 - F.cosine_similarity(pred, target.detach(), dim=-1).mean()
    return mse + 0.5 * cos   # cosine weighted at 0.5 — tune if needed



def decoder_loss(logits: torch.Tensor, target_ids: torch.Tensor,
                 pad_id: int = 0) -> torch.Tensor:
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        target_ids.reshape(-1),
        ignore_index=pad_id
    )


def joint_loss(pred: torch.Tensor, target: torch.Tensor,
               logits: torch.Tensor, target_ids: torch.Tensor,
               mse_weight: float = 1.0, ce_weight: float = 1.0,
               pad_id: int = 0) -> torch.Tensor:
    mse = jepa_loss(pred, target)
    ce  = decoder_loss(logits, target_ids, pad_id)
    return mse_weight * mse + ce_weight * ce
