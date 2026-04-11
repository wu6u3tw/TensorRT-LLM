# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for SkipSoftmax visual generation config and wiring."""

import math

import pytest

from tensorrt_llm._torch.visual_gen.config import (
    AttentionConfig,
    BaseSparseAttentionConfig,
    SkipSoftmaxConfig,
    SkipSoftmaxFormula,
    apply_skip_softmax_overrides,
)

# =============================================================================
# SkipSoftmaxConfig construction
# =============================================================================


class TestSkipSoftmaxConfigConstruction:
    def test_threshold_scale_factor_only(self):
        cfg = SkipSoftmaxConfig(threshold_scale_factor=5000.0)
        assert cfg.threshold_scale_factor == 5000.0
        assert cfg.target_sparsity is None
        assert cfg.formula is None
        assert cfg.layer_overrides is None

    def test_target_sparsity_with_formula(self):
        cfg = SkipSoftmaxConfig(
            target_sparsity=0.5,
            formula=SkipSoftmaxFormula(a=0.0003, b=7.5),
        )
        assert cfg.target_sparsity == 0.5
        assert cfg.formula.a == 0.0003
        assert cfg.formula.b == 7.5

    def test_with_layer_overrides(self):
        cfg = SkipSoftmaxConfig(
            threshold_scale_factor=5000.0,
            layer_overrides={
                "transformer_blocks.0.*": 0,
                "single_transformer_blocks.*": 8000.0,
            },
        )
        assert len(cfg.layer_overrides) == 2

    def test_attention_config_with_skip_softmax(self):
        cfg = AttentionConfig(
            backend="TRTLLM",
            sparse_attention_config=SkipSoftmaxConfig(threshold_scale_factor=5000.0),
        )
        assert cfg.sparse_attention_config is not None
        assert cfg.sparse_attention_config.algorithm == "skip_softmax"
        assert cfg.sparse_attention_config.threshold_scale_factor == 5000.0

    def test_attention_config_from_dict(self):
        cfg = AttentionConfig(
            **{
                "backend": "TRTLLM",
                "sparse_attention_config": {
                    "algorithm": "skip_softmax",
                    "threshold_scale_factor": 5000.0,
                },
            }
        )
        assert cfg.sparse_attention_config.threshold_scale_factor == 5000.0

    def test_attention_config_from_dict_with_formula(self):
        cfg = AttentionConfig(
            **{
                "backend": "TRTLLM",
                "sparse_attention_config": {
                    "algorithm": "skip_softmax",
                    "target_sparsity": 0.5,
                    "formula": {"a": 0.0003, "b": 7.5},
                },
            }
        )
        assert cfg.sparse_attention_config.target_sparsity == 0.5
        assert cfg.sparse_attention_config.formula.a == 0.0003

    def test_attention_config_no_sparse(self):
        cfg = AttentionConfig(backend="VANILLA")
        assert cfg.sparse_attention_config is None

    def test_base_class_inheritance(self):
        cfg = SkipSoftmaxConfig(threshold_scale_factor=5000.0)
        assert isinstance(cfg, BaseSparseAttentionConfig)
        assert cfg.algorithm == "skip_softmax"


# =============================================================================
# resolve_threshold_scale_factor
# =============================================================================


class TestResolveThresholdScaleFactor:
    def test_direct_threshold_returns_immediately(self):
        cfg = SkipSoftmaxConfig(threshold_scale_factor=5000.0)
        assert cfg.resolve_threshold_scale_factor() == 5000.0

    def test_direct_threshold_ignores_formula(self):
        cfg = SkipSoftmaxConfig(
            threshold_scale_factor=5000.0,
            target_sparsity=0.5,
            formula=SkipSoftmaxFormula(a=0.0003, b=7.5),
        )
        # threshold_scale_factor takes precedence
        assert cfg.resolve_threshold_scale_factor() == 5000.0

    def test_target_sparsity_with_user_formula(self):
        cfg = SkipSoftmaxConfig(
            target_sparsity=0.5,
            formula=SkipSoftmaxFormula(a=7e-5, b=7.929109),
        )
        expected = 7e-5 * math.exp(7.929109 * 0.5)
        assert cfg.resolve_threshold_scale_factor() == pytest.approx(expected)

    def test_target_sparsity_with_checkpoint_formula(self):
        cfg = SkipSoftmaxConfig(target_sparsity=0.5)
        checkpoint = {"a": 7e-5, "b": 7.929109}
        expected = 7e-5 * math.exp(7.929109 * 0.5)
        assert cfg.resolve_threshold_scale_factor(checkpoint) == pytest.approx(expected)

    def test_user_formula_overrides_checkpoint(self):
        cfg = SkipSoftmaxConfig(
            target_sparsity=0.5,
            formula=SkipSoftmaxFormula(a=0.001, b=5.0),  # user
        )
        checkpoint = {"a": 7e-5, "b": 7.929109}  # checkpoint (lower priority)
        expected = 0.001 * math.exp(5.0 * 0.5)  # should use user formula
        assert cfg.resolve_threshold_scale_factor(checkpoint) == pytest.approx(expected)

    def test_modelopt_checkpoint_formula_format(self):
        """Test with the actual ModelOpt config.json format."""
        cfg = SkipSoftmaxConfig(target_sparsity=0.5)
        # ModelOpt format: sparse_attention_config.threshold_scale_factor.prefill
        modelopt_prefill = {"a": 7.93, "b": 8.61}
        expected = 7.93 * math.exp(8.61 * 0.5)
        assert cfg.resolve_threshold_scale_factor(modelopt_prefill) == pytest.approx(expected)

    def test_no_threshold_no_sparsity_returns_none(self):
        cfg = SkipSoftmaxConfig()
        assert cfg.resolve_threshold_scale_factor() is None

    def test_target_sparsity_no_formula_raises(self):
        cfg = SkipSoftmaxConfig(target_sparsity=0.5)
        with pytest.raises(ValueError, match="calibration formula"):
            cfg.resolve_threshold_scale_factor()

    def test_target_sparsity_zero(self):
        cfg = SkipSoftmaxConfig(
            target_sparsity=0.0,
            formula=SkipSoftmaxFormula(a=7e-5, b=7.929109),
        )
        # exp(0) = 1, so result = a
        assert cfg.resolve_threshold_scale_factor() == pytest.approx(7e-5)

    def test_target_sparsity_one(self):
        cfg = SkipSoftmaxConfig(
            target_sparsity=1.0,
            formula=SkipSoftmaxFormula(a=7e-5, b=7.929109),
        )
        expected = 7e-5 * math.exp(7.929109)
        assert cfg.resolve_threshold_scale_factor() == pytest.approx(expected)


# =============================================================================
# resolve_threshold (layer overrides)
# =============================================================================


class TestResolveThreshold:
    def test_no_overrides_returns_default(self):
        cfg = SkipSoftmaxConfig(threshold_scale_factor=5000.0)
        assert cfg.resolve_threshold("transformer_blocks.5.attn1") == 5000.0

    def test_matching_override(self):
        cfg = SkipSoftmaxConfig(
            threshold_scale_factor=5000.0,
            layer_overrides={"transformer_blocks.0.*": 0},
        )
        assert cfg.resolve_threshold("transformer_blocks.0.attn1") is None  # disabled

    def test_non_matching_returns_default(self):
        cfg = SkipSoftmaxConfig(
            threshold_scale_factor=5000.0,
            layer_overrides={"transformer_blocks.0.*": 0},
        )
        assert cfg.resolve_threshold("transformer_blocks.5.attn1") == 5000.0

    def test_override_with_custom_value(self):
        cfg = SkipSoftmaxConfig(
            threshold_scale_factor=5000.0,
            layer_overrides={"single_transformer_blocks.*": 8000.0},
        )
        assert cfg.resolve_threshold("single_transformer_blocks.10.attn") == 8000.0

    def test_first_match_wins(self):
        cfg = SkipSoftmaxConfig(
            threshold_scale_factor=5000.0,
            layer_overrides={
                "transformer_blocks.0.*": 0,
                "transformer_blocks.*": 3000.0,
            },
        )
        # First pattern matches
        assert cfg.resolve_threshold("transformer_blocks.0.attn1") is None
        # Second pattern matches for other blocks
        assert cfg.resolve_threshold("transformer_blocks.5.attn1") == 3000.0

    def test_wildcard_patterns(self):
        cfg = SkipSoftmaxConfig(
            threshold_scale_factor=5000.0,
            layer_overrides={"*.attn2": 0},  # disable all cross-attention
        )
        assert cfg.resolve_threshold("transformer.blocks.3.attn2") is None
        assert cfg.resolve_threshold("transformer.blocks.3.attn1") == 5000.0

    def test_no_threshold_returns_none(self):
        cfg = SkipSoftmaxConfig(target_sparsity=0.5)
        # threshold_scale_factor not resolved yet
        assert cfg.resolve_threshold("any_layer") is None


# =============================================================================
# apply_skip_softmax_overrides
# =============================================================================


class TestApplySkipSoftmaxOverrides:
    def _make_mock_model(self):
        """Create a minimal mock model with TrtllmAttention-like backends."""
        import torch.nn as nn

        class MockTrtllmAttention:
            sparse_attention_config = None

        class MockAttentionModule(nn.Module):
            def __init__(self, name):
                super().__init__()
                self.attn = MockTrtllmAttention()

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.block0 = MockAttentionModule("block0")
                self.block1 = MockAttentionModule("block1")
                self.block2 = MockAttentionModule("block2")

        # Monkey-patch isinstance check since we can't import the real class
        # in unit tests without GPU
        return MockModel()

    def test_no_overrides_returns_zero(self):
        model = self._make_mock_model()
        cfg = SkipSoftmaxConfig(threshold_scale_factor=5000.0)
        # No layer_overrides → returns 0
        assert apply_skip_softmax_overrides(model, cfg) == 0

    def test_overrides_count(self):
        # This test would need real TrtllmAttention instances
        # which require GPU. Skip in unit test, test in integration.
        pass
