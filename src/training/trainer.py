import copy
import torch
from torch.cuda.amp import GradScaler, autocast

from src.training.schedulers import get_lr, get_ema_momentum, update_ema
from src.utils.checkpoint    import save_checkpoint, load_checkpoint, latest_checkpoint
from src.utils.logging       import get_logger, MetricLogger


logger = get_logger()


def build_teacher(model):
    """Deep-copy student → freeze → return as EMA teacher."""
    teacher = copy.deepcopy(model)
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()
    return teacher


def train_stage1(model, dataloader, cfg, device, resume: bool = True):
    # Teacher is a frozen EMA copy of the student
    # It encodes answers to provide stable regression targets (JEPA objective)
    teacher = build_teacher(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=0.04,
        betas=(0.9, 0.999)
    )
    amp_dtype = torch.bfloat16
    scaler    = GradScaler(enabled=(device.type == "cuda" and amp_dtype == torch.float16))
    metric_log = MetricLogger(
        use_wandb=getattr(cfg, "wandb", False),
        project="think-in-silence",
        run_name="stage1",
        config={"stage": 1, "n_steps": cfg.model.n_steps}
    )

    step = 0
    if resume:
        ckpt = latest_checkpoint(cfg.training.ckpt_dir)
        if ckpt:
            step = load_checkpoint(
                ckpt, model, optimizer, scaler, teacher, device=str(device)
            )
            logger.info(f"Resumed from {ckpt} at step {step}")

    model.train()
    teacher.eval()
    data_iter = iter(dataloader)

    while step < cfg.training.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch     = next(data_iter)

        q_ids  = batch["q_ids"].to(device)
        q_mask = batch["q_mask"].to(device)
        a_ids  = batch["a_ids"].to(device)
        a_mask = batch["a_mask"].to(device)

        lr = get_lr(step, cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        with autocast(enabled=device.type == "cuda", dtype=amp_dtype):
            # Pass teacher so stage1 uses EMA answer embeddings as targets
            loss, pred, target = model(
                q_ids, q_mask, a_ids, a_mask,
                mode="stage1",
                teacher=teacher       # ← key fix: teacher provides stable targets
            )

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            filter(lambda p: p.grad is not None, model.parameters()),
            cfg.training.grad_clip
        )
        scaler.step(optimizer)
        scaler.update()

        # Update EMA teacher after every student step
        momentum = get_ema_momentum(step, cfg)
        update_ema(model, teacher, momentum)

        step += 1

        if step % cfg.training.log_every == 0:
            metrics = {
                "train/loss":         loss.item(),
                "train/lr":           lr,
                "train/ema_momentum": momentum,
            }
            metric_log.log(metrics, step=step)
            logger.info(
                f"step={step:6d}  loss={loss.item():.4f}  "
                f"lr={lr:.2e}  ema={momentum:.4f}"
            )

        if step % cfg.training.ckpt_every == 0:
            path = save_checkpoint(
                model, optimizer, scaler, step,
                cfg.training.ckpt_dir, teacher_model=teacher
            )
            logger.info(f"Checkpoint saved: {path}")

    metric_log.finish()
    return model