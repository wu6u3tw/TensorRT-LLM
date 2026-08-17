# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the TransformerEngine FP8 attention backend (TEAttention).

CPU-only tests cover args validation and the import-guard path.
GPU tests (requires transformer_engine) cover forward correctness vs VANILLA.
"""

import pytest
import torch
from pydantic import ValidationError

from tensorrt_llm._torch.visual_gen.attention_backend.interface import (
    AttentionTensorLayout,
    PredefinedAttentionMask,
)
from tensorrt_llm._torch.visual_gen.attention_backend.te import TEAttention
from tensorrt_llm._torch.visual_gen.attention_backend.utils import get_visual_gen_attention_backend
from tensorrt_llm.visual_gen.args import AttentionConfig, QuantAttentionConfig

try:
    import transformer_engine  # noqa: F401

    _te_available = True
except ImportError:
    _te_available = False

requires_te = pytest.mark.skipif(not _te_available, reason="transformer_engine not installed")


# =============================================================================
# CPU-only: args validation
# =============================================================================


@pytest.mark.cpu_only
class TestTEAttentionArgsValidation:
    def test_te_backend_accepted(self):
        cfg = AttentionConfig(backend="TE")
        assert cfg.backend == "TE"

    def test_te_backend_lowercase_accepted(self):
        # Pydantic Literal is case-sensitive; lowercase should fail.
        with pytest.raises(ValidationError):
            AttentionConfig(backend="te")

    def test_quant_attention_config_rejected_with_te_backend(self):
        with pytest.raises(ValidationError, match="requires backend in"):
            AttentionConfig(backend="TE", quant_attention_config=QuantAttentionConfig())

    def test_get_visual_gen_attention_backend_returns_te_class(self):
        cls = get_visual_gen_attention_backend("TE")
        assert cls is TEAttention

    def test_get_visual_gen_attention_backend_case_insensitive(self):
        assert get_visual_gen_attention_backend("te") is TEAttention


# =============================================================================
# CPU-only: import guard (TE absent path)
# =============================================================================


@pytest.mark.cpu_only
class TestTEAttentionImportGuard:
    @pytest.mark.skipif(_te_available, reason="TE is installed; guard only fires when TE is absent")
    def test_construction_raises_import_error_when_te_absent(self):
        with pytest.raises(ImportError, match="TransformerEngine is required"):
            TEAttention(num_heads=4, head_dim=64)

    def test_te_attention_always_importable(self):
        from tensorrt_llm._torch.visual_gen.attention_backend.te import TEAttention as _TE

        assert _TE is not None


# =============================================================================
# CPU-only: interface contract (no instantiation needed)
# =============================================================================


@pytest.mark.cpu_only
class TestTEAttentionInterface:
    def test_preferred_layout_is_nhd(self):
        assert TEAttention.support_fused_qkv() is False

    def test_support_fused_qkv_false(self):
        assert TEAttention.support_fused_qkv() is False

    def test_support_lse_false(self):
        assert TEAttention.support_lse() is False


# =============================================================================
# GPU tests: forward correctness vs VANILLA reference
# =============================================================================


@requires_te
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
class TestTEAttentionForward:
    @pytest.fixture
    def make_te_attn(self):
        def _make(num_heads=4, head_dim=64, num_kv_heads=None):
            return TEAttention(
                layer_idx=0,
                num_heads=num_heads,
                head_dim=head_dim,
                num_kv_heads=num_kv_heads or num_heads,
            ).cuda()

        return _make

    def _vanilla_ref(self, q, k, v):
        # q/k/v: [B, S, H, D] (NHD) -> need [B, H, S, D] for SDPA
        q_ = q.transpose(1, 2).float()
        k_ = k.transpose(1, 2).float()
        v_ = v.transpose(1, 2).float()
        out = torch.nn.functional.scaled_dot_product_attention(q_, k_, v_, is_causal=False)
        return out.transpose(1, 2).to(q.dtype)  # [B, S, H, D]

    def test_output_shape(self, make_te_attn):
        B, S, H, D = 2, 64, 4, 64
        attn = make_te_attn(num_heads=H, head_dim=D)
        q = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            out = attn(q, k, v)
        assert out.shape == (B, S, H, D), f"Expected {(B, S, H, D)}, got {out.shape}"

    def test_output_finite(self, make_te_attn):
        B, S, H, D = 1, 128, 4, 64
        attn = make_te_attn(num_heads=H, head_dim=D)
        q = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            out = attn(q, k, v)
        assert torch.isfinite(out).all(), "TEAttention output contains NaN or Inf"

    def test_preferred_layout_nhd(self, make_te_attn):
        attn = make_te_attn()
        assert attn.preferred_layout == AttentionTensorLayout.NHD

    @pytest.mark.parametrize("S", [64, 256, 1024])
    def test_output_close_to_vanilla_ref(self, make_te_attn, S):
        """TE FP8 output should have cosine similarity > 0.99 vs BF16 SDPA."""
        B, H, D = 1, 4, 64
        attn = make_te_attn(num_heads=H, head_dim=D)
        torch.manual_seed(42)
        q = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            out_te = attn(q, k, v)
            out_ref = self._vanilla_ref(q, k, v)
        cos_sim = torch.nn.functional.cosine_similarity(
            out_te.reshape(-1).float(), out_ref.reshape(-1).float(), dim=0
        ).item()
        assert cos_sim > 0.99, f"Cosine similarity {cos_sim:.4f} < 0.99 at S={S}"

    def test_causal_mask(self, make_te_attn):
        B, S, H, D = 1, 64, 4, 64
        attn = make_te_attn(num_heads=H, head_dim=D)
        q = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            out_full = attn(q, k, v, attention_mask=PredefinedAttentionMask.FULL)
            out_causal = attn(q, k, v, attention_mask=PredefinedAttentionMask.CAUSAL)
        assert out_full.shape == out_causal.shape
        # Causal and full outputs should differ (causal masks upper triangle)
        assert not torch.allclose(out_full, out_causal, atol=1e-3)

    def test_key_padding_mask_raises(self, make_te_attn):
        B, S, H, D = 1, 64, 4, 64
        attn = make_te_attn(num_heads=H, head_dim=D)
        q = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
        mask = torch.ones(B, S, device="cuda", dtype=torch.bool)
        with pytest.raises(NotImplementedError, match="key_padding_mask"):
            attn(q, k, v, key_padding_mask=mask)

    def test_attn_op_rebuilt_on_trait_change(self, make_te_attn):
        """DotProductAttention is rebuilt when head_dim changes between calls."""
        attn = make_te_attn(num_heads=4, head_dim=64)
        B, S = 1, 64

        q = torch.randn(B, S, 4, 64, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, S, 4, 64, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, S, 4, 64, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            attn(q, k, v)
        op_first = attn._attn_op

        # Simulate a different attn_mask_type triggering a rebuild
        with torch.no_grad():
            attn(q, k, v, attention_mask=PredefinedAttentionMask.CAUSAL)
        op_causal = attn._attn_op

        assert op_first is not op_causal, "Expected _attn_op to be rebuilt on trait change"

        # Same trait again - should reuse
        with torch.no_grad():
            attn(q, k, v, attention_mask=PredefinedAttentionMask.CAUSAL)
        assert attn._attn_op is op_causal, "Expected _attn_op to be reused when traits unchanged"
