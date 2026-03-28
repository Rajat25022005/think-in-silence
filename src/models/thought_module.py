import torch
import torch.nn as nn
from src.models.thought_block import ThoughtBlock


class ThoughtModule(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        dim      = cfg.model.proj_dim
        n_heads  = cfg.model.n_heads
        ffn_dim  = cfg.model.ffn_dim
        n_steps  = cfg.model.n_steps
        dropout  = cfg.model.dropout
        shared   = cfg.model.shared_weights

        if shared:
            block = ThoughtBlock(dim, n_heads, ffn_dim, dropout)
            self.blocks = nn.ModuleList([block] * n_steps)
        else:
            self.blocks = nn.ModuleList(
                [ThoughtBlock(dim, n_heads, ffn_dim, dropout) for _ in range(n_steps)]
            )

        self.h0 = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.h0, std=0.02)

        self.predictor = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim)
        )

        self.n_steps = n_steps

    def forward(
        self,
        ctx: torch.Tensor,
        n_steps: int = None,
        return_all_states: bool = False
    ) -> torch.Tensor:
        B = ctx.size(0)
        K = n_steps if n_steps is not None else self.n_steps

        h = self.h0.expand(B, -1, -1)
        states = [h]

        for k in range(K):
            block = self.blocks[k % len(self.blocks)]
            h     = block(h, ctx)
            states.append(h)

        if return_all_states:
            return torch.stack(states, dim=1)

        return h

    def predict(self, h: torch.Tensor) -> torch.Tensor:
        return self.predictor(h.squeeze(1))
