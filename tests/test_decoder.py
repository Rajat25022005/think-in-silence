import pytest
import torch
from types import SimpleNamespace
from src.models.decoder import LatentDecoder


def make_cfg():
    return SimpleNamespace(
        model=SimpleNamespace(proj_dim=64, n_heads=4, dropout=0.0),
        decoder=SimpleNamespace(n_layers=2, n_heads=4, max_gen_len=12, tie_embeddings=True)
    )


VOCAB = 100


def test_forward_logits_shape():
    decoder   = LatentDecoder(make_cfg(), vocab_size=VOCAB)
    B, seq    = 3, 8
    token_ids = torch.zeros(B, seq, dtype=torch.long)
    memory    = torch.randn(B, 1, 64)
    logits    = decoder(token_ids, memory)
    assert logits.shape == (B, seq, VOCAB)


def test_generate_nonempty():
    decoder   = LatentDecoder(make_cfg(), vocab_size=VOCAB)
    memory    = torch.randn(2, 1, 64)
    generated = decoder.generate(memory, bos_id=1, eos_id=2)
    assert generated.shape[0] == 2
    assert generated.shape[1] >= 2


def test_cross_attention_receives_memory():
    decoder   = LatentDecoder(make_cfg(), vocab_size=VOCAB)
    token_ids = torch.zeros(2, 5, dtype=torch.long)
    mem_a     = torch.zeros(2, 1, 64)
    mem_b     = torch.ones(2, 1, 64)
    out_a     = decoder(token_ids, mem_a)
    out_b     = decoder(token_ids, mem_b)
    assert not torch.allclose(out_a, out_b)


def test_causal_mask_blocks_future():
    torch.manual_seed(0)
    decoder   = LatentDecoder(make_cfg(), vocab_size=VOCAB)
    token_ids = torch.zeros(1, 6, dtype=torch.long)
    memory    = torch.randn(1, 1, 64)
    logits    = decoder(token_ids, memory)
    assert logits.shape == (1, 6, VOCAB)
