import pytest
import torch
from types import SimpleNamespace
from src.models.thought_block import ThoughtBlock


def make_block(dim=64, n_heads=4, ffn_dim=128):
    return ThoughtBlock(dim=dim, n_heads=n_heads, ffn_dim=ffn_dim, dropout=0.0)


def test_output_shape():
    block = make_block()
    B, seq = 3, 10
    h   = torch.randn(B, 1, 64)
    ctx = torch.randn(B, seq, 64)
    out = block(h, ctx)
    assert out.shape == (B, 1, 64)


def test_residual_connection():
    block = make_block()
    for p in block.parameters():
        p.data.zero_()
    h   = torch.ones(2, 1, 64)
    ctx = torch.zeros(2, 5, 64)
    out = block(h, ctx)
    assert out.shape == (2, 1, 64)


def test_gradient_flows():
    block = make_block()
    h   = torch.randn(2, 1, 64, requires_grad=True)
    ctx = torch.randn(2, 5, 64)
    out = block(h, ctx)
    out.sum().backward()
    assert h.grad is not None
