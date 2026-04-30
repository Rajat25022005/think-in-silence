import torch

from src.training.schedulers import get_lr
from src.utils.checkpoint    import save_checkpoint, load_checkpoint, latest_checkpoint
from src.utils.logging       import get_logger, MetricLogger
from src.utils.device        import is_tpu


logger = get_logger()


def train_stage2(model, dataloader, cfg, device, resume: bool = True):
    _on_tpu = is_tpu()
    if _on_tpu:
        import torch_xla.core.xla_model as xm

    for name, param in model.thought.named_parameters():
        param.requires_grad = False
    logger.info("ThoughtModule frozen for stage 2 decoder training")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
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
        run_name="stage2"
    )

    import os
    ckpt_dir = getattr(cfg.training, "stage2_ckpt_dir", os.path.join(os.path.dirname(cfg.training.ckpt_dir.rstrip("/")), "stage2"))
    keep_last_n = getattr(cfg.training, "keep_last_n_ckpts", 3)
    step     = 0
    if resume:
        ckpt = latest_checkpoint(ckpt_dir)
        if ckpt:
            step = load_checkpoint(
                ckpt, model, optimizer, scaler if use_scaler else None,
                device="cpu"
            )
            model.to(device)
            logger.info(f"Resumed from {ckpt} at step {step}")

    model.train()

    # Wrap dataloader for TPU
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

        if _on_tpu:
            loss, logits = model(q_ids, q_mask, a_ids, a_mask, mode="stage2")
        else:
            with torch.amp.autocast("cuda", enabled=device.type == "cuda", dtype=torch.bfloat16):
                loss, logits = model(q_ids, q_mask, a_ids, a_mask, mode="stage2")

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

        step += 1

        if step % cfg.training.log_every == 0:
            loss_val = loss.item()
            non_pad = (a_ids[:, 1:] != 0).sum().item()
            metric_log.log({
                "train/loss": loss_val,
                "train/lr": lr,
                "train/non_pad_tokens": non_pad,
            }, step=step)
            if not _on_tpu or xm.is_master_ordinal():
                logger.info(f"step={step:6d}  loss={loss_val:.4f}  lr={lr:.2e}")

        if step % cfg.training.ckpt_every == 0:
            if not _on_tpu or xm.is_master_ordinal():
                path = save_checkpoint(
                    model, optimizer, scaler if use_scaler else None, step,
                    ckpt_dir, keep_last_n=keep_last_n
                )
                logger.info(f"Checkpoint saved: {path}")
            if _on_tpu:
                xm.rendezvous("checkpoint_saved")

    metric_log.finish()
    return model
