import pytest
import torch
from types import SimpleNamespace
from src.models.thought_module import ThoughtModule


def make_cfg(n_steps=4, shared=False):
    return SimpleNamespace(model=SimpleNamespace(
        proj_dim=64, n_heads=4, ffn_dim=128, n_steps=n_steps,
        dropout=0.0, shared_weights=shared
    ))


def make_ctx(B=2, seq=10, dim=64):
    return torch.randn(B, seq, dim)


def test_output_shape():
    module = ThoughtModule(make_cfg())
    ctx    = make_ctx()
    h      = module(ctx)
    assert h.shape == (2, 1, 64)


def test_n_steps_override():
    module = ThoughtModule(make_cfg(n_steps=4))
    ctx    = make_ctx()
    h      = module(ctx, n_steps=16)
    assert h.shape == (2, 1, 64)


def test_return_all_states():
    K      = 4
    module = ThoughtModule(make_cfg(n_steps=K))
    ctx    = make_ctx()
    states = module(ctx, return_all_states=True)
    assert states.shape == (2, K + 1, 1, 64)


def test_h0_gradient():
    module = ThoughtModule(make_cfg())
    ctx    = make_ctx()
    h      = module(ctx)
    pred   = module.predict(h)
    pred.sum().backward()
    assert module.h0.grad is not None


def test_shared_weights_same_object():
    module = ThoughtModule(make_cfg(n_steps=4, shared=True))
    unique_ids = set(id(b) for b in module.blocks)
    assert len(unique_ids) == 1
