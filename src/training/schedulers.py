import math


def get_lr(step: int, cfg) -> float:
    warmup = cfg.training.warmup_steps
    total  = cfg.training.max_steps
    base   = cfg.training.lr

    if step < warmup:
        return base * step / max(warmup, 1)

    progress = (step - warmup) / max(total - warmup, 1)
    return base * 0.5 * (1.0 + math.cos(math.pi * progress))


def get_ema_momentum(step: int, cfg) -> float:
    progress = step / max(cfg.training.max_steps, 1)
    start    = cfg.training.ema_momentum_start
    end      = cfg.training.ema_momentum_end
    return end - (end - start) * (math.cos(math.pi * progress) + 1) / 2


def update_ema(student, teacher, momentum: float):
    with __import__("torch").no_grad():
        for s_param, t_param in zip(student.parameters(), teacher.parameters()):
            t_param.data.mul_(momentum).add_(s_param.data, alpha=1.0 - momentum)
