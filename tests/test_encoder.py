import pytest
import torch
from types import SimpleNamespace


def make_cfg():
    return SimpleNamespace(
        model=SimpleNamespace(
            backbone="distilbert",
            n_steps=4,
            proj_dim=64,
            n_heads=4,
            ffn_dim=128,
            dropout=0.0,
            shared_weights=False
        ),
        decoder=SimpleNamespace(
            n_layers=2,
            n_heads=4,
            max_gen_len=16,
            tie_embeddings=True
        )
    )


def make_encoder(cfg):
    from src.models.encoder import FrozenEncoder
    return FrozenEncoder(cfg)


def test_encode_question_shape():
    cfg     = make_cfg()
    encoder = make_encoder(cfg)
    B, L    = 2, 32
    ids     = torch.zeros(B, L, dtype=torch.long)
    mask    = torch.ones(B, L, dtype=torch.long)
    out     = encoder.encode_question(ids, mask)
    assert out.shape == (B, L, cfg.model.proj_dim)


def test_encode_answer_shape():
    cfg     = make_cfg()
    encoder = make_encoder(cfg)
    B, L    = 2, 16
    ids     = torch.zeros(B, L, dtype=torch.long)
    mask    = torch.ones(B, L, dtype=torch.long)
    out     = encoder.encode_answer(ids, mask)
    assert out.shape == (B, cfg.model.proj_dim)


def test_encoder_frozen_backbone():
    cfg     = make_cfg()
    encoder = make_encoder(cfg)
    for param in encoder.backbone.model.parameters():
        assert not param.requires_grad


def test_proj_layers_trainable():
    cfg     = make_cfg()
    encoder = make_encoder(cfg)
    for param in encoder.question_proj.parameters():
        assert param.requires_grad
    for param in encoder.answer_proj.parameters():
        assert param.requires_grad
