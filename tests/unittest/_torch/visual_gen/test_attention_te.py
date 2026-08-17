# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the TransformerEngine FP8 attention backend (TEAttention).

GPU tests (requires transformer_engine) cover forward correctness vs VANILLA.
"""

import pytest
import torch

from tensorrt_llm._torch.visual_gen.attention_backend.interface import (
    AttentionTensorLayout,
    PredefinedAttentionMask,
)
from tensorrt_llm._torch.visual_gen.attention_backend.te import TEAttention

try:
    import transformer_engine  # noqa: F401

    _te_available = True
except ImportError:
    _te_available = False

pytestmark = pytest.mark.skipif(
    not _te_available or not torch.cuda.is_available(),
    reason="transformer_engine and GPU required",
)


@pytest.fixture
def make_te_attn():
    def _make(num_heads=4, head_dim=64, num_kv_heads=None):
        return TEAttention(
            layer_idx=0,
            num_heads=num_heads,
            head_dim=head_dim,
            num_kv_heads=num_kv_heads or num_heads,
        ).cuda()

    return _make


def _vanilla_ref(q, k, v):
    q_ = q.transpose(1, 2).float()
    k_ = k.transpose(1, 2).float()
    v_ = v.transpose(1, 2).float()
    out = torch.nn.functional.scaled_dot_product_attention(q_, k_, v_, is_causal=False)
    return out.transpose(1, 2).to(q.dtype)


def test_output_shape(make_te_attn):
    B, S, H, D = 2, 64, 4, 64
    attn = make_te_attn(num_heads=H, head_dim=D)
    q = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        out = attn(q, k, v)
    assert out.shape == (B, S, H, D)


def test_output_finite(make_te_attn):
    B, S, H, D = 1, 128, 4, 64
    attn = make_te_attn(num_heads=H, head_dim=D)
    q = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        out = attn(q, k, v)
    assert torch.isfinite(out).all()


def test_preferred_layout_nhd(make_te_attn):
    attn = make_te_attn()
    assert attn.preferred_layout == AttentionTensorLayout.NHD


@pytest.mark.parametrize("S", [64, 256, 1024])
def test_output_close_to_vanilla_ref(make_te_attn, S):
    B, H, D = 1, 4, 64
    attn = make_te_attn(num_heads=H, head_dim=D)
    torch.manual_seed(42)
    q = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        out_te = attn(q, k, v)
        out_ref = _vanilla_ref(q, k, v)
    cos_sim = torch.nn.functional.cosine_similarity(
        out_te.reshape(-1).float(), out_ref.reshape(-1).float(), dim=0
    ).item()
    assert cos_sim > 0.99, f"Cosine similarity {cos_sim:.4f} < 0.99 at S={S}"


def test_causal_mask(make_te_attn):
    B, S, H, D = 1, 64, 4, 64
    attn = make_te_attn(num_heads=H, head_dim=D)
    q = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        out_full = attn(q, k, v, attention_mask=PredefinedAttentionMask.FULL)
        out_causal = attn(q, k, v, attention_mask=PredefinedAttentionMask.CAUSAL)
    assert out_full.shape == out_causal.shape
    assert not torch.allclose(out_full, out_causal, atol=1e-3)


def test_key_padding_mask_raises(make_te_attn):
    B, S, H, D = 1, 64, 4, 64
    attn = make_te_attn(num_heads=H, head_dim=D)
    q = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
    mask = torch.ones(B, S, device="cuda", dtype=torch.bool)
    with pytest.raises(NotImplementedError, match="key_padding_mask"):
        attn(q, k, v, key_padding_mask=mask)


def test_attn_op_rebuilt_on_trait_change(make_te_attn):
    attn = make_te_attn(num_heads=4, head_dim=64)
    B, S = 1, 64
    q = torch.randn(B, S, 4, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, S, 4, 64, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, S, 4, 64, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        attn(q, k, v)
    op_first = attn._attn_op

    with torch.no_grad():
        attn(q, k, v, attention_mask=PredefinedAttentionMask.CAUSAL)
    op_causal = attn._attn_op

    assert op_first is not op_causal

    with torch.no_grad():
        attn(q, k, v, attention_mask=PredefinedAttentionMask.CAUSAL)
    assert attn._attn_op is op_causal
