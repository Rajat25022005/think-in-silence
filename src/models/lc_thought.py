import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from src.models.encoder       import FrozenEncoder
from src.models.thought_module import ThoughtModule
from src.models.decoder        import LatentDecoder


class LCThought(nn.Module):
    def __init__(self, cfg, vocab_size: int):
        super().__init__()
        self.cfg          = cfg
        self.encoder      = FrozenEncoder(cfg)
        self.thought      = ThoughtModule(cfg)
        self.decoder      = LatentDecoder(cfg, vocab_size)
        self.mse_loss     = nn.MSELoss()

    def forward(
        self,
        q_ids:  torch.Tensor,
        q_mask: torch.Tensor,
        a_ids:  torch.Tensor,
        a_mask: torch.Tensor,
        n_steps: Optional[int] = None,
        return_all_states: bool = False,
        mode: str = "stage1"
    ):
        ctx    = self.encoder.encode_question(q_ids, q_mask)
        target = self.encoder.encode_answer(a_ids, a_mask)

        if mode == "stage1":
            h    = self.thought(ctx, n_steps=n_steps, return_all_states=return_all_states)
            if return_all_states:
                pred = self.thought.predict(h[:, -1, :].unsqueeze(1))
            else:
                pred = self.thought.predict(h)
            loss = self.mse_loss(pred, target.detach())
            return loss, pred, target

        elif mode == "stage2":
            with torch.no_grad():
                h = self.thought(ctx, n_steps=n_steps)
            logits = self.decoder(a_ids[:, :-1], h)
            loss   = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                a_ids[:, 1:].reshape(-1),
                ignore_index=0
            )
            return loss, logits

        elif mode == "stage3":
            h     = self.thought(ctx, n_steps=n_steps)
            pred  = self.thought.predict(h)
            mse   = self.mse_loss(pred, target.detach())
            logits = self.decoder(a_ids[:, :-1], h)
            ce     = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                a_ids[:, 1:].reshape(-1),
                ignore_index=0
            )
            loss = mse + ce
            return loss, pred, target, logits

        elif mode == "generate":
            return self.generate(q_ids, q_mask, n_steps=n_steps)

        raise ValueError(f"Unknown mode: {mode}. Use stage1 | stage2 | stage3 | generate")

    @torch.no_grad()
    def generate(
        self,
        q_ids:   torch.Tensor,
        q_mask:  torch.Tensor,
        n_steps: Optional[int] = None,
        max_len: int = 64
    ) -> list:
        tokenizer = self.encoder.tokenizer
        bos_id    = tokenizer.cls_token_id or tokenizer.bos_token_id or 1
        eos_id    = tokenizer.sep_token_id or tokenizer.eos_token_id or 2

        ctx       = self.encoder.encode_question(q_ids, q_mask)
        h         = self.thought(ctx, n_steps=n_steps)
        token_ids = self.decoder.generate(h, bos_id=bos_id, eos_id=eos_id, max_len=max_len)

        results = []
        for ids in token_ids:
            ids_list = ids.tolist()
            try:
                end = ids_list.index(eos_id)
                ids_list = ids_list[1:end]
            except ValueError:
                ids_list = ids_list[1:]
            results.append(tokenizer.decode(ids_list, skip_special_tokens=True))
        return results
