import copy
import random
import torch

from src.training.schedulers import get_lr, get_ema_momentum, update_ema
from src.utils.checkpoint    import save_checkpoint, load_checkpoint, latest_checkpoint
from src.utils.logging       import get_logger, MetricLogger
from src.utils.device        import is_tpu

K_SCHEDULE = [1, 2, 4, 8, 16]

logger = get_logger()


def build_teacher(model):
    """Deep-copy student → freeze → return as EMA teacher."""
    teacher = copy.deepcopy(model)
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()
    return teacher


def train_stage1(model, dataloader, cfg, device, resume: bool = True):
    _on_tpu = is_tpu()
    if _on_tpu:
        import torch_xla.core.xla_model as xm

    # Teacher is a frozen EMA copy of the student
    # It encodes answers to provide stable regression targets (JEPA objective)
    teacher = build_teacher(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=0.04,
        betas=(0.9, 0.999)
    )

    # GradScaler is only needed for CUDA fp16 — TPU uses native bf16
    use_scaler = (not _on_tpu) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    metric_log = MetricLogger(
        use_wandb=getattr(cfg, "wandb", False),
        project="think-in-silence",
        run_name="stage1",
        config={"stage": 1, "n_steps": cfg.model.n_steps}
    )

    keep_last_n = getattr(cfg.training, "keep_last_n_ckpts", 3)
    step = 0
    if resume:
        ckpt = latest_checkpoint(cfg.training.ckpt_dir)
        if ckpt:
            step = load_checkpoint(
                ckpt, model, optimizer, scaler if use_scaler else None,
                teacher, device="cpu"
            )
            # Re-move to device after loading
            model.to(device)
            teacher.to(device)
            logger.info(f"Resumed from {ckpt} at step {step}")

    model.train()
    model.encoder.backbone.model.eval()
    teacher.eval()

    # Wrap dataloader for TPU — MpDeviceLoader handles host-to-device transfer
    if _on_tpu:
        import torch_xla.distributed.parallel_loader as pl
        dataloader = pl.MpDeviceLoader(dataloader, device)

    data_iter = iter(dataloader)

    while step < cfg.training.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch     = next(data_iter)

        if _on_tpu:
            # MpDeviceLoader already moves tensors to XLA device
            q_ids  = batch["q_ids"]
            q_mask = batch["q_mask"]
            a_ids  = batch["a_ids"]
            a_mask = batch["a_mask"]
        else:
            q_ids  = batch["q_ids"].to(device)
            q_mask = batch["q_mask"].to(device)
            a_ids  = batch["a_ids"].to(device)
            a_mask = batch["a_mask"].to(device)

        lr = get_lr(step, cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        k = random.choice(K_SCHEDULE)

        if _on_tpu:
            # TPU: bf16 is native, no autocast needed
            loss, pred, target = model(
                q_ids, q_mask, a_ids, a_mask,
                mode="stage1",
                teacher=teacher,
                n_steps=k
            )
        else:
            with torch.amp.autocast("cuda", enabled=device.type == "cuda", dtype=torch.bfloat16):
                loss, pred, target = model(
                    q_ids, q_mask, a_ids, a_mask,
                    mode="stage1",
                    teacher=teacher,
                    n_steps=k
                )

        metric_log.log({'train/k': k}, step=step)

        optimizer.zero_grad(set_to_none=True)

        if _on_tpu:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.grad is not None],
                cfg.training.grad_clip
            )
            xm.optimizer_step(optimizer)
            xm.mark_step()
        else:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.grad is not None],
                cfg.training.grad_clip
            )
            scaler.step(optimizer)
            scaler.update()

        # Update EMA teacher after every student step
        momentum = get_ema_momentum(step, cfg)
        update_ema(model, teacher, momentum)

        step += 1

        if step % cfg.training.log_every == 0:
            loss_val = loss.item()
            metrics = {
                "train/loss":         loss_val,
                "train/lr":           lr,
                "train/ema_momentum": momentum,
            }
            metric_log.log(metrics, step=step)
            if not _on_tpu or xm.is_master_ordinal():
                logger.info(
                    f"step={step:6d}  loss={loss_val:.4f}  "
                    f"lr={lr:.2e}  ema={momentum:.4f}"
                )

        if step % cfg.training.ckpt_every == 0:
            if not _on_tpu or xm.is_master_ordinal():
                path = save_checkpoint(
                    model, optimizer, scaler if use_scaler else None, step,
                    cfg.training.ckpt_dir, teacher_model=teacher,
                    keep_last_n=keep_last_n
                )
                logger.info(f"Checkpoint saved: {path}")
            if _on_tpu:
                xm.rendezvous("checkpoint_saved")

    metric_log.finish()
    return model