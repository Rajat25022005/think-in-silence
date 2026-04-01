import torch
from torch.cuda.amp import GradScaler, autocast

from src.training.schedulers import get_lr
from src.utils.checkpoint    import save_checkpoint, load_checkpoint, latest_checkpoint
from src.utils.logging       import get_logger, MetricLogger


logger = get_logger()


def train_stage2(model, dataloader, cfg, device, resume: bool = True):
    for name, param in model.thought.named_parameters():
        param.requires_grad = False
    logger.info("ThoughtModule frozen for stage 2 decoder training")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.training.lr,
        weight_decay=0.04,
        betas=(0.9, 0.999)
    )
    amp_dtype  = torch.bfloat16
    scaler     = GradScaler(enabled=(device.type == "cuda" and amp_dtype == torch.float16))
    metric_log = MetricLogger(
        use_wandb=getattr(cfg, "wandb", False),
        project="think-in-silence",
        run_name="stage2"
    )

    import os
    ckpt_dir = getattr(cfg.training, "stage2_ckpt_dir", os.path.join(os.path.dirname(cfg.training.ckpt_dir.rstrip("/")), "stage2"))
    step     = 0
    if resume:
        ckpt = latest_checkpoint(ckpt_dir)
        if ckpt:
            step = load_checkpoint(ckpt, model, optimizer, scaler, device=str(device))
            logger.info(f"Resumed from {ckpt} at step {step}")

    model.train()
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
            loss, logits = model(q_ids, q_mask, a_ids, a_mask, mode="stage2")

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            filter(lambda p: p.grad is not None, model.parameters()),
            cfg.training.grad_clip
        )
        scaler.step(optimizer)
        scaler.update()

        step += 1

        if step % cfg.training.log_every == 0:
            metric_log.log({"train/loss": loss.item(), "train/lr": lr}, step=step)
            logger.info(f"step={step:6d}  loss={loss.item():.4f}  lr={lr:.2e}")

        if step % cfg.training.ckpt_every == 0:
            path = save_checkpoint(model, optimizer, scaler, step, ckpt_dir)
            logger.info(f"Checkpoint saved: {path}")

    metric_log.finish()
    return model
