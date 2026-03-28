import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderLayer(nn.Module):
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

    def forward(self, x: torch.Tensor, memory: torch.Tensor,
                causal_mask: torch.Tensor = None) -> torch.Tensor:
        normed = self.norm1(x)
        attn_out, _ = self.self_attn(normed, normed, normed, attn_mask=causal_mask)
        x = x + attn_out

        normed = self.norm2(x)
        attn_out, _ = self.cross_attn(normed, memory, memory)
        x = x + attn_out

        x = x + self.ffn(self.norm3(x))
        return x


class LatentDecoder(nn.Module):
    def __init__(self, cfg, vocab_size: int):
        super().__init__()

        dim      = cfg.model.proj_dim
        n_heads  = cfg.decoder.n_heads
        ffn_dim  = dim * 4
        n_layers = cfg.decoder.n_layers
        dropout  = cfg.model.dropout

        self.embed   = nn.Embedding(vocab_size, dim)
        self.layers  = nn.ModuleList(
            [DecoderLayer(dim, n_heads, ffn_dim, dropout) for _ in range(n_layers)]
        )
        self.norm_out = nn.LayerNorm(dim)
        self.lm_head  = nn.Linear(dim, vocab_size, bias=False)

        if cfg.decoder.tie_embeddings:
            self.lm_head.weight = self.embed.weight

        self.max_gen_len = cfg.decoder.max_gen_len

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.masked_fill(mask == 1, float("-inf"))

    def forward(self, token_ids: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        seq_len = token_ids.size(1)
        x       = self.embed(token_ids)
        causal  = self._causal_mask(seq_len, token_ids.device)

        for layer in self.layers:
            x = layer(x, memory, causal_mask=causal)

        x = self.norm_out(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(self, memory: torch.Tensor, bos_id: int, eos_id: int,
                 max_len: int = None) -> torch.Tensor:
        B      = memory.size(0)
        device = memory.device
        max_len = max_len or self.max_gen_len

        generated = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        done      = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_len):
            logits      = self.forward(generated, memory)
            next_token  = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated   = torch.cat([generated, next_token], dim=1)
            done        = done | (next_token.squeeze(1) == eos_id)
            if done.all():
                break

        return generated
