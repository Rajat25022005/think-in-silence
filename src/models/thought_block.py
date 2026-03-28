import torch
import torch.nn as nn


class ThoughtBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=n_heads,
            dropout=dropout, batch_first=True
        )

        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=n_heads,
            dropout=dropout, batch_first=True
        )

        self.norm3 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, h: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(h)
        attn_out, _ = self.self_attn(normed, normed, normed)
        h = h + attn_out

        normed = self.norm2(h)
        attn_out, _ = self.cross_attn(normed, ctx, ctx)
        h = h + attn_out

        h = h + self.ffn(self.norm3(h))
        return h
