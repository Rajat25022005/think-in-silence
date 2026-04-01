import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from src.models.encoder        import FrozenEncoder
from src.models.thought_module import ThoughtModule
from src.models.decoder        import LatentDecoder


class LCThought(nn.Module):
    """
    Full think-in-silence model.

    Stage 1 (JEPA):
        student:  encodes question → runs K thought steps → predictor → pred
        teacher:  encodes answer  → mean-pooled embedding (EMA, no grad)
        loss:     MSE(pred, teacher_answer_emb.detach())

    The teacher is maintained externally in trainer.py via EMA.
    LCThought receives the teacher as an argument in stage1 forward.
    """

    def __init__(self, cfg, vocab_size: int):
        super().__init__()
        self.cfg      = cfg
        self.encoder  = FrozenEncoder(cfg)
        self.thought  = ThoughtModule(cfg)
        self.decoder  = LatentDecoder(cfg, vocab_size)
        self.mse_loss = nn.MSELoss()

    def forward(
        self,
        q_ids:   torch.Tensor,
        q_mask:  torch.Tensor,
        a_ids:   torch.Tensor,
        a_mask:  torch.Tensor,
        n_steps: Optional[int] = None,
        return_all_states: bool = False,
        mode: str = "stage1",
        teacher=None,          # ← EMA teacher model, required for stage1
    ):
        if mode == "stage1":
            # ── Student: question → latent thought → prediction ──────────
            ctx  = self.encoder.encode_question(q_ids, q_mask)
            h    = self.thought(ctx, n_steps=n_steps,
                                return_all_states=return_all_states)

            if return_all_states:
                pred = self.thought.predict(h[:, -1, :].unsqueeze(1))
            else:
                pred = self.thought.predict(h)

            # ── Teacher: answer → target embedding (EMA, no grad) ────────
            if teacher is not None:
                # Use EMA teacher encoder for stable targets — this is JEPA
                with torch.no_grad():
                    target = teacher.encoder.encode_answer(a_ids, a_mask)
            else:
                # Fallback (testing only) — uses student encoder
                target = self.encoder.encode_answer(a_ids, a_mask)

            loss = self.mse_loss(pred, target.detach())
            return loss, pred, target

        elif mode == "stage2":
            # ThoughtModule frozen externally in decoder_trainer.py
            ctx = self.encoder.encode_question(q_ids, q_mask)
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
            ctx    = self.encoder.encode_question(q_ids, q_mask)
            target = self.encoder.encode_answer(a_ids, a_mask)
            h      = self.thought(ctx, n_steps=n_steps)
            pred   = self.thought.predict(h)
            mse    = self.mse_loss(pred, target.detach())
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

        raise ValueError(
            f"Unknown mode: '{mode}'. Use: stage1 | stage2 | stage3 | generate"
        )

    @torch.no_grad()
    def generate(
        self,
        q_ids:   torch.Tensor,
        q_mask:  torch.Tensor,
        n_steps: Optional[int] = None,
        max_len: int = 64,
    ) -> list:
        tokenizer = self.encoder.tokenizer
        bos_id    = tokenizer.cls_token_id or tokenizer.bos_token_id or 1
        eos_id    = tokenizer.sep_token_id or tokenizer.eos_token_id or 2

        ctx       = self.encoder.encode_question(q_ids, q_mask)
        h         = self.thought(ctx, n_steps=n_steps)
        token_ids = self.decoder.generate(
            h, bos_id=bos_id, eos_id=eos_id, max_len=max_len
        )

        results = []
        for ids in token_ids:
            ids_list = ids.tolist()
            try:
                end      = ids_list.index(eos_id)
                ids_list = ids_list[1:end]
            except ValueError:
                ids_list = ids_list[1:]
            results.append(
                tokenizer.decode(ids_list, skip_special_tokens=True)
            )
        return results