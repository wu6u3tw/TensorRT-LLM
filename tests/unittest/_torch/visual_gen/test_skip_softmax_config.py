# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for SkipSoftmax visual generation config and wiring."""

import math

import pytest

from tensorrt_llm._torch.visual_gen.config import (
    AttentionConfig,
    SkipSoftmaxConfig,
    SkipSoftmaxFormula,
    apply_skip_softmax_overrides,
)
from tensorrt_llm.llmapi.llm_args import SkipSoftmaxAttentionConfig

# =============================================================================
# SkipSoftmaxFormula — accepts both log_a (diffusion) and a (LLM) formats
# =============================================================================


class TestSkipSoftmaxFormulaFormats:
    def test_accepts_log_a(self):
        """Diffusion format: log_a stored directly."""
        f = SkipSoftmaxFormula(log_a=-14.409, b=37.457)
        assert f.log_a == pytest.approx(-14.409)
        assert f.b == pytest.approx(37.457)

    def test_accepts_linear_a_and_normalizes(self):
        """LLM format: a is normalized to log_a = log(a)."""
        f = SkipSoftmaxFormula(a=7e-5, b=7.929109)
        assert f.log_a == pytest.approx(math.log(7e-5))
        assert f.b == pytest.approx(7.929109)

    def test_rejects_both_log_a_and_a(self):
        """Specifying both is ambiguous — error rather than silently pick one."""
        with pytest.raises(ValueError, match="not both"):
            SkipSoftmaxFormula(log_a=-10.0, a=999.0, b=5.0)

    def test_rejects_non_positive_a(self):
        """Linear 'a' must be positive (log of 0/negative is undefined)."""
        with pytest.raises(ValueError, match="must be positive"):
            SkipSoftmaxFormula(a=0.0, b=5.0)
        with pytest.raises(ValueError, match="must be positive"):
            SkipSoftmaxFormula(a=-1.0, b=5.0)


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
            formula=SkipSoftmaxFormula(log_a=math.log(0.0003), b=7.5),
        )
        assert cfg.target_sparsity == 0.5
        assert cfg.formula.log_a == pytest.approx(math.log(0.0003))
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
                    "formula": {"log_a": math.log(0.0003), "b": 7.5},
                },
            }
        )
        assert cfg.sparse_attention_config.target_sparsity == 0.5
        assert cfg.sparse_attention_config.formula.log_a == pytest.approx(math.log(0.0003))

    def test_attention_config_no_sparse(self):
        cfg = AttentionConfig(backend="VANILLA")
        assert cfg.sparse_attention_config is None

    def test_base_class_inheritance(self):
        cfg = SkipSoftmaxConfig(threshold_scale_factor=5000.0)
        # Inherits from the LLM-shared SkipSoftmaxAttentionConfig (reuse, no duplication)
        assert isinstance(cfg, SkipSoftmaxAttentionConfig)
        assert cfg.algorithm == "skip_softmax"


# =============================================================================
# Use case scenarios
# =============================================================================


class TestUseCaseScenarios:
    """End-to-end use case tests matching the PR documentation.

    Case 1: Normal HF checkpoint (no skip_softmax metadata in config.json)
      1a: User provides threshold_scale_factor → all layers get same threshold
      1b: User provides target_sparsity without formula → helpful error
      1c: User provides target_sparsity + formula → resolves correctly
      1d: User provides full config with layer_overrides → per-layer thresholds

    Case 2: ModelOpt checkpoint (has calibrated a, b in config.json)
      2a: User provides nothing → auto-enable from checkpoint
      2b: User provides threshold_scale_factor → user overrides checkpoint
      2c: User provides target_sparsity → uses checkpoint formula
    """

    MODELOPT_CHECKPOINT = {
        "sparse_attention_config": {
            "config_groups": {
                "group_0": {
                    "sparse_algo": "softmax_skip",
                    "targets": ["Attention"],
                }
            },
            "threshold_scale_factor": {
                "formula": "a * exp(b * target_sparsity)",
                "prefill": {"a": 7.93, "b": 8.61},
                "decode": {"a": 0.12, "b": 9.85},
            },
            "producer": {"name": "modelopt", "version": "0.37.0"},
        }
    }

    # --- Case 1: Normal HF checkpoint ---

    def test_case_1a_user_threshold_only(self):
        """Normal checkpoint + user threshold → works."""
        cfg = SkipSoftmaxConfig(threshold_scale_factor=5000.0)
        result = cfg.resolve_threshold_scale_factor(checkpoint_formula=None)
        assert result == 5000.0

    def test_case_1b_user_target_sparsity_no_formula(self):
        """Normal checkpoint + target_sparsity without formula → helpful error."""
        cfg = SkipSoftmaxConfig(target_sparsity=0.5)
        with pytest.raises(ValueError, match="calibration formula"):
            cfg.resolve_threshold_scale_factor(checkpoint_formula=None)

    def test_case_1c_user_target_sparsity_with_formula(self):
        """Normal checkpoint + target_sparsity + user formula → resolves."""
        cfg = SkipSoftmaxConfig(
            target_sparsity=0.5,
            formula=SkipSoftmaxFormula(log_a=math.log(0.0003), b=7.5),
        )
        result = cfg.resolve_threshold_scale_factor(checkpoint_formula=None)
        expected = 0.0003 * math.exp(7.5 * 0.5)
        assert result == pytest.approx(expected)

    def test_case_1d_user_with_layer_overrides(self):
        """Normal checkpoint + layer_overrides → per-layer thresholds."""
        cfg = SkipSoftmaxConfig(
            threshold_scale_factor=5000.0,
            layer_overrides={"blocks.0*": 0, "blocks.5*": 8000.0},
        )
        assert cfg.resolve_threshold("blocks.0.attn1") is None  # disabled
        assert cfg.resolve_threshold("blocks.5.attn1") == 8000.0  # override
        assert cfg.resolve_threshold("blocks.3.attn1") == 5000.0  # default

    # --- Case 2: ModelOpt checkpoint ---

    def test_case_2a_modelopt_checkpoint_auto_enable(self):
        """ModelOpt checkpoint + no user config → auto-enable from checkpoint.

        The pipeline should detect sparse_attention_config in checkpoint
        config.json and create a SkipSoftmaxConfig automatically.
        """
        from tensorrt_llm._torch.visual_gen.config import auto_detect_sparse_attention_config

        ckpt = self.MODELOPT_CHECKPOINT
        result = auto_detect_sparse_attention_config(ckpt)
        assert result is not None
        assert isinstance(result, SkipSoftmaxConfig)
        # Should have the formula from checkpoint
        assert result.formula is not None
        assert result.formula.log_a == pytest.approx(math.log(7.93))
        assert result.formula.b == pytest.approx(8.61)

    def test_case_2b_modelopt_user_threshold_overrides(self):
        """ModelOpt checkpoint + user threshold → user wins."""
        cfg = SkipSoftmaxConfig(threshold_scale_factor=3000.0)
        ckpt_formula = self.MODELOPT_CHECKPOINT["sparse_attention_config"][
            "threshold_scale_factor"
        ]["prefill"]
        result = cfg.resolve_threshold_scale_factor(checkpoint_formula=ckpt_formula)
        # User threshold takes precedence, checkpoint formula ignored
        assert result == 3000.0

    def test_case_2c_modelopt_user_target_sparsity(self):
        """ModelOpt checkpoint + user target_sparsity → uses checkpoint formula."""
        cfg = SkipSoftmaxConfig(target_sparsity=0.5)
        ckpt_formula = self.MODELOPT_CHECKPOINT["sparse_attention_config"][
            "threshold_scale_factor"
        ]["prefill"]
        result = cfg.resolve_threshold_scale_factor(checkpoint_formula=ckpt_formula)
        expected = 7.93 * math.exp(8.61 * 0.5)
        assert result == pytest.approx(expected)

    def test_case_2a_no_sparse_config_returns_none(self):
        """Normal checkpoint (no sparse_attention_config) → returns None."""
        from tensorrt_llm._torch.visual_gen.config import auto_detect_sparse_attention_config

        result = auto_detect_sparse_attention_config({})
        assert result is None

        result = auto_detect_sparse_attention_config({"other_key": 123})
        assert result is None


# =============================================================================
# YAML loading
# =============================================================================


class TestYamlLoading:
    def test_load_modelopt_yaml(self, tmp_path):
        """Load from ModelOpt sparse YAML file."""
        from tensorrt_llm._torch.visual_gen.config import load_sparse_config_from_yaml

        yaml_content = """
config_groups:
  group_0:
    sparse_algo: softmax_skip
    targets:
    - WanAttention
    threshold_scale_factor:
      formula: log_a + b * target_sparsity
      prefill:
        log_a: -14.14
        b: 36.64
    disabled_layers:
    - blocks.0.attn1
    - blocks.0.attn2
    - blocks.39.attn2
"""
        yaml_file = tmp_path / "sparse.yaml"
        yaml_file.write_text(yaml_content)

        cfg = load_sparse_config_from_yaml(str(yaml_file))
        assert cfg is not None
        assert cfg.formula.log_a == pytest.approx(-14.14)
        assert cfg.formula.b == pytest.approx(36.64)
        assert cfg.layer_overrides is not None
        assert cfg.layer_overrides["blocks.0.attn1"] == 0
        assert cfg.layer_overrides["blocks.0.attn2"] == 0
        assert cfg.layer_overrides["blocks.39.attn2"] == 0
        assert len(cfg.layer_overrides) == 3

    def test_load_modelopt_yaml_llm_format_a(self, tmp_path):
        """Load from LLM-format YAML where prefill uses 'a' instead of 'log_a'."""
        from tensorrt_llm._torch.visual_gen.config import load_sparse_config_from_yaml

        yaml_content = """
config_groups:
  group_0:
    sparse_algo: softmax_skip
    threshold_scale_factor:
      formula: a * exp(b * target_sparsity)
      prefill:
        a: 7.0e-5
        b: 7.929109
"""
        yaml_file = tmp_path / "sparse.yaml"
        yaml_file.write_text(yaml_content)

        cfg = load_sparse_config_from_yaml(str(yaml_file))
        assert cfg is not None
        # 'a' should be normalized to log_a = log(a)
        assert cfg.formula.log_a == pytest.approx(math.log(7e-5))
        assert cfg.formula.b == pytest.approx(7.929109)

    def test_load_yaml_no_skip_softmax(self, tmp_path):
        """YAML without softmax_skip algo returns None."""
        from tensorrt_llm._torch.visual_gen.config import load_sparse_config_from_yaml

        yaml_content = """
config_groups:
  group_0:
    sparse_algo: other_algo
"""
        yaml_file = tmp_path / "sparse.yaml"
        yaml_file.write_text(yaml_content)

        cfg = load_sparse_config_from_yaml(str(yaml_file))
        assert cfg is None

    def test_auto_detect_yaml(self, tmp_path):
        """Auto-detect sparse YAML files in checkpoint directory."""
        from tensorrt_llm._torch.visual_gen.config import auto_detect_sparse_yaml

        yaml_content = """
config_groups:
  group_0:
    sparse_algo: softmax_skip
    threshold_scale_factor:
      prefill:
        log_a: -14.14
        b: 36.64
"""
        (tmp_path / "transformer").mkdir()
        (tmp_path / "transformer" / "sparse.yaml").write_text(yaml_content)

        configs = auto_detect_sparse_yaml(str(tmp_path))
        assert configs is not None
        assert len(configs) >= 1


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
            formula=SkipSoftmaxFormula(log_a=math.log(0.0003), b=7.5),
        )
        # threshold_scale_factor takes precedence
        assert cfg.resolve_threshold_scale_factor() == 5000.0

    def test_target_sparsity_with_user_formula(self):
        cfg = SkipSoftmaxConfig(
            target_sparsity=0.5,
            formula=SkipSoftmaxFormula(log_a=math.log(7e-5), b=7.929109),
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
            formula=SkipSoftmaxFormula(log_a=math.log(0.001), b=5.0),  # user
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
            formula=SkipSoftmaxFormula(log_a=math.log(7e-5), b=7.929109),
        )
        # exp(0) = 1, so result = a
        assert cfg.resolve_threshold_scale_factor() == pytest.approx(7e-5)

    def test_target_sparsity_one(self):
        cfg = SkipSoftmaxConfig(
            target_sparsity=1.0,
            formula=SkipSoftmaxFormula(log_a=math.log(7e-5), b=7.929109),
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
        """Create a mock model with patched TrtllmAttention instances."""
        from unittest.mock import MagicMock

        import torch.nn as nn

        from tensorrt_llm._torch.visual_gen.attention_backend.trtllm import TrtllmAttention

        def make_mock_backend():
            mock = MagicMock(spec=TrtllmAttention)
            mock.sparse_attention_config = None
            return mock

        class MockAttentionModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = make_mock_backend()

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.block0 = MockAttentionModule()
                self.block1 = MockAttentionModule()
                self.block2 = MockAttentionModule()

        return MockModel()

    def test_no_overrides_returns_zero(self):
        model = self._make_mock_model()
        cfg = SkipSoftmaxConfig(threshold_scale_factor=5000.0)
        assert apply_skip_softmax_overrides(model, cfg) == 0

    def test_overrides_applied(self):
        model = self._make_mock_model()
        cfg = SkipSoftmaxConfig(
            threshold_scale_factor=5000.0,
            layer_overrides={"block0*": 0, "block2*": 8000.0},
        )
        n = apply_skip_softmax_overrides(model, cfg)
        assert n == 3

        # block0: disabled (threshold=0 → None)
        assert model.block0.attn.sparse_attention_config is None
        # block1: default threshold
        assert model.block1.attn.sparse_attention_config is not None
        assert model.block1.attn.sparse_attention_config.threshold_scale_factor_prefill == 5000.0
        # block2: overridden to 8000
        assert model.block2.attn.sparse_attention_config is not None
        assert model.block2.attn.sparse_attention_config.threshold_scale_factor_prefill == 8000.0
