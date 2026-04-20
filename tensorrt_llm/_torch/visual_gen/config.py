import fnmatch
import json
import math
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated, Any, Dict, List, Literal, Optional, Tuple, Union

import torch
import yaml
from pydantic import BaseModel, ConfigDict, model_validator
from pydantic import Field as PydanticField

from tensorrt_llm.functional import AllReduceStrategy
from tensorrt_llm.llmapi.utils import StrictBaseModel, set_api_status
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

# =============================================================================
# Type aliases
# =============================================================================

CacheBackendName = Literal["teacache", "cache_dit"]

# =============================================================================
# Pipeline component identifiers
# =============================================================================


class PipelineComponent(str, Enum):
    """Identifiers for pipeline components that can be loaded or skipped.

    Inherits from str so values compare equal to plain strings,
    e.g. PipelineComponent.VAE == "vae" is True.
    """

    TRANSFORMER = "transformer"
    VAE = "vae"
    TEXT_ENCODER = "text_encoder"
    TEXT_ENCODER_2 = "text_encoder_2"
    TOKENIZER = "tokenizer"
    TOKENIZER_2 = "tokenizer_2"
    SCHEDULER = "scheduler"
    IMAGE_ENCODER = "image_encoder"
    IMAGE_PROCESSOR = "image_processor"


# =============================================================================
# Sub-configuration classes for VisualGenArgs
# =============================================================================


class SageAttentionConfig(StrictBaseModel):
    """Configuration for SageAttention quantization (TRTLLM backend only).

    SageAttention quantizes Q/K/V into FP8 (or INT8 for Q/K) with per-block
    scaling factors, enabling faster attention kernels. Providing this config
    to AttentionConfig enables SageAttention; omitting it (None) disables it.

    Similar to ``sparse_attention_config`` for the base TRTLLM attention
    backend — the presence of the config object signals enablement.

    Currently these (num_elts_per_blk_q, num_elts_per_blk_k, num_elts_per_blk_v)
    combinations are enabled:
    - (1, 1, 1)
    - (1, 4, 1)
    - (1, 16, 1) [for qk_int8 == True only]
    """

    num_elts_per_blk_q: int = PydanticField(
        1, ge=0, description="Elements per quantization block for Q (0 disables)"
    )
    num_elts_per_blk_k: int = PydanticField(
        4, ge=0, description="Elements per quantization block for K (0 disables)"
    )
    num_elts_per_blk_v: int = PydanticField(
        1, ge=0, description="Elements per quantization block for V (0 disables)"
    )
    qk_int8: bool = PydanticField(True, description="Use INT8 (vs E4M3) for Q/K quantization")


class BaseSparseAttentionConfig(StrictBaseModel):
    """Base class for sparse attention configurations.

    Subclasses must set ``algorithm`` to a unique literal string.
    The ``algorithm`` field acts as a Pydantic discriminator so that
    users can write ``sparse_attention_config: {algorithm: skip_softmax, ...}``
    and the correct subclass is instantiated automatically.
    """

    algorithm: str = PydanticField(description="Sparse attention algorithm name")
    layer_overrides: Optional[Dict[str, float]] = PydanticField(
        None,
        description="Per-layer threshold/parameter overrides. Keys are fnmatch "
        "patterns matched against module names. Set to 0 to disable for "
        "matching layers. Example: {'transformer_blocks.0.*': 0}",
    )

    def resolve_threshold(self, module_name: str) -> Optional[float]:
        """Resolve the threshold for a specific layer by module name.

        Subclasses should set ``self.threshold_scale_factor`` before calling.
        Checks layer_overrides patterns first (fnmatch), falls back to default.
        Returns None if threshold is 0 (disabled for this layer).
        """
        threshold = getattr(self, "threshold_scale_factor", None)
        if threshold is None:
            return None
        if self.layer_overrides:
            for pattern, override in self.layer_overrides.items():
                if fnmatch.fnmatch(module_name, pattern):
                    threshold = override
                    break
        return threshold if threshold > 0 else None


class SkipSoftmaxFormula(StrictBaseModel):
    """Exponential calibration formula: threshold = exp(log_a + b * sparsity).

    Equivalent to: threshold = a * exp(b * sparsity) where a = exp(log_a).
    Stored in log-space (log_a) to match ModelOpt diffusion format and
    avoid precision loss. Accepts either 'log_a' (diffusion format) or 'a'
    (LLM format) at construction; 'a' is normalized to log_a = log(a).
    """

    log_a: float = PydanticField(description="Log of coefficient a (log-space)")
    b: float = PydanticField(description="Coefficient b")

    @model_validator(mode="before")
    @classmethod
    def _accept_linear_a(cls, values):
        """Normalize LLM-format 'a' to diffusion-format 'log_a'."""
        if not isinstance(values, dict) or "a" not in values:
            return values
        if "log_a" in values:
            raise ValueError(
                "SkipSoftmaxFormula: specify either 'log_a' (diffusion format) "
                "or 'a' (LLM format), not both."
            )
        a = values["a"]
        if a <= 0:
            raise ValueError(
                f"SkipSoftmaxFormula: 'a' must be positive (got {a}). "
                "Use 'log_a' directly if you need log(a) of a non-positive value."
            )
        values = {**values}
        values["log_a"] = math.log(a)
        values.pop("a")
        return values


class SkipSoftmaxConfig(BaseSparseAttentionConfig):
    """SkipSoftmax sparse attention configuration.

    Dynamically skips softmax + BMM2 for KV blocks whose contribution falls
    below a threshold. The kernel decision rule is:
        skip if exp(local_max - global_max) < threshold_scale_factor / seq_len

    Requires backend='TRTLLM'. See docs/source/features/sparse-attention.md.

    Two ways to specify the threshold:
    1. threshold_scale_factor: raw value, resolution-dependent
    2. target_sparsity + formula: resolution-aware, uses exp(log_a + b * sparsity)
       Formula coefficients can come from checkpoint/YAML or user config.
    """

    algorithm: Literal["skip_softmax"] = "skip_softmax"
    threshold_scale_factor: Optional[float] = PydanticField(
        None,
        description="Default threshold scale factor for all attention layers. "
        "Higher = more aggressive skipping. Takes precedence over target_sparsity.",
    )
    target_sparsity: Optional[float] = PydanticField(
        None,
        description="Target sparsity (0.0-1.0). Converted to threshold_scale_factor "
        "via calibration formula. Requires formula coefficients from checkpoint "
        "config.json or from the 'formula' field.",
    )
    formula: Optional[SkipSoftmaxFormula] = PydanticField(
        None,
        description="Calibration formula coefficients for target_sparsity → "
        "threshold_scale_factor conversion. Takes precedence over checkpoint "
        "config.json formula.",
    )

    def resolve_threshold_scale_factor(
        self,
        checkpoint_formula: Optional[Dict[str, float]] = None,
    ) -> Optional[float]:
        """Resolve to a concrete threshold_scale_factor.

        Priority:
        1. self.threshold_scale_factor (if set, use directly)
        2. self.target_sparsity + formula from:
           a. self.formula (user config — highest precedence)
           b. checkpoint_formula (from checkpoint config.json — fallback)

        Returns:
            Resolved threshold_scale_factor, or None if cannot resolve.
        """
        if self.threshold_scale_factor is not None:
            return self.threshold_scale_factor

        if self.target_sparsity is None:
            return None

        # User formula takes precedence over checkpoint formula
        if self.formula:
            log_a, b = self.formula.log_a, self.formula.b
        elif checkpoint_formula:
            # Support both log_a (diffusion) and a (LLM) formats
            if "log_a" in checkpoint_formula:
                log_a = checkpoint_formula["log_a"]
            elif "a" in checkpoint_formula:
                log_a = math.log(checkpoint_formula["a"])
            else:
                log_a = None
            b = checkpoint_formula.get("b")
        else:
            log_a, b = None, None

        if log_a is None or b is None:
            raise ValueError(
                "SkipSoftmaxConfig: target_sparsity requires calibration formula "
                "coefficients. Provide via ModelOpt YAML (log_a, b), checkpoint "
                "config.json (a, b), or user config (formula field)."
            )
        return math.exp(log_a + b * self.target_sparsity)


# Discriminated union of all sparse attention configs.
# Add new algorithms here as: Annotated[Union[SkipSoftmaxConfig, ...], Field(discriminator="algorithm")]
SparseAttentionConfig = Annotated[
    Union[SkipSoftmaxConfig],
    PydanticField(discriminator="algorithm"),
]


class AttentionConfig(StrictBaseModel):
    """Configuration for Attention layers."""

    backend: Literal["VANILLA", "TRTLLM", "FA4"] = PydanticField(
        "VANILLA", description="Attention backend: VANILLA (PyTorch SDPA), TRTLLM, FA4"
    )
<<<<<<< HEAD
    sage_attention_config: Optional[SageAttentionConfig] = PydanticField(
        None,
        description=(
            "SageAttention config (TRTLLM backend only). "
            "Set to a SageAttentionConfig instance to enable SageAttention; "
            "leave as None to disable."
        ),
    )

    @model_validator(mode="after")
    def _validate_sage_attn_config(self) -> "AttentionConfig":
        SUPPORTED_SAGE_CONFIGS = {
            (1, 1, 1, False),
            (1, 4, 1, False),
            (1, 1, 1, True),
            (1, 4, 1, True),
            (1, 16, 1, True),
        }

        if self.sage_attention_config is not None:
            if self.backend != "TRTLLM":
                raise ValueError(
                    f"sage_attention_config requires backend='TRTLLM', "
                    f"got backend='{self.backend}'. Either set backend='TRTLLM' "
                    f"or remove sage_attention_config."
                )
            if (
                self.sage_attention_config.num_elts_per_blk_q,
                self.sage_attention_config.num_elts_per_blk_k,
                self.sage_attention_config.num_elts_per_blk_v,
                self.sage_attention_config.qk_int8,
            ) not in SUPPORTED_SAGE_CONFIGS:
                raise ValueError(f"Unsupported {self.sage_attention_config=}.")
        return self

    sparse_attention_config: Optional[SparseAttentionConfig] = PydanticField(
        None, description="Sparse attention configuration. Currently supports: skip_softmax."
    )
    sparse_config_path: Optional[str] = PydanticField(
        None,
        description="Path to ModelOpt sparse attention YAML config file. "
        "Overrides auto-detection from checkpoint directory.",
    )


def load_sparse_config_from_yaml(yaml_path: str) -> Optional[SkipSoftmaxConfig]:
    """Load SkipSoftmaxConfig from a ModelOpt sparse attention YAML file.

    Supports both ModelOpt diffusion format (log_a) and LLM format (a):

        # Diffusion format
        config_groups:
          group_0:
            sparse_algo: softmax_skip
            threshold_scale_factor:
              formula: log_a + b * target_sparsity
              prefill:
                log_a: -14.14
                b: 36.64

        # LLM format (auto-converted: log_a = log(a))
        config_groups:
          group_0:
            sparse_algo: softmax_skip
            threshold_scale_factor:
              formula: a * exp(b * target_sparsity)
              prefill:
                a: 7e-5
                b: 7.93

    Args:
        yaml_path: Path to the YAML file.

    Returns:
        SkipSoftmaxConfig, or None if the file doesn't contain skip_softmax config.
    """
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        return None

    # Find the first config group with sparse_algo: softmax_skip
    config_groups = data.get("config_groups", {})
    for group in config_groups.values():
        if group.get("sparse_algo") != "softmax_skip":
            continue

        tsf = group.get("threshold_scale_factor", {})
        prefill = tsf.get("prefill", {})
        if "b" not in prefill or ("log_a" not in prefill and "a" not in prefill):
            continue

        # Build layer_overrides from disabled_layers (threshold=0 → disabled)
        disabled = group.get("disabled_layers", [])
        layer_overrides = {name: 0 for name in disabled} if disabled else None

        # Filter to known keys; SkipSoftmaxFormula validator normalizes 'a' → 'log_a'.
        formula_kwargs = {k: prefill[k] for k in ("log_a", "a", "b") if k in prefill}
        return SkipSoftmaxConfig(
            formula=SkipSoftmaxFormula(**formula_kwargs),
            layer_overrides=layer_overrides,
        )

    return None


def auto_detect_sparse_yaml(checkpoint_dir: str) -> Optional[Dict[str, SkipSoftmaxConfig]]:
    """Auto-detect ModelOpt sparse YAML files in a checkpoint directory.

    Looks for files matching ``sparse.yaml`` or ``sparse.*.yaml`` in the
    checkpoint's component directories (e.g. transformer/, transformer_2/).

    Returns:
        Dict mapping component name to SkipSoftmaxConfig, or None.
    """

    checkpoint_path = Path(checkpoint_dir)
    configs = {}

    # Look in component subdirectories (diffusers layout)
    for yaml_path in sorted(checkpoint_path.glob("**/sparse*.yaml")):
        cfg = load_sparse_config_from_yaml(str(yaml_path))
        if cfg is not None:
            # Derive component name from filename or parent directory
            name = yaml_path.stem  # e.g. "sparse" or "sparse.transformer_2"
            configs[name] = cfg

    return configs if configs else None


def auto_detect_sparse_attention_config(
    checkpoint_config: Dict[str, Any],
) -> Optional[SkipSoftmaxConfig]:
    """Auto-detect sparse attention config from a ModelOpt checkpoint config.json.

    If the checkpoint contains calibrated skip_softmax metadata (formula coefficients),
    creates a SkipSoftmaxConfig with the formula so users can just set target_sparsity
    without providing their own formula. If no sparse config is found, returns None.

    Args:
        checkpoint_config: Parsed contents of the checkpoint's config.json.

    Returns:
        SkipSoftmaxConfig with formula from checkpoint, or None.
    """
    sparse_cfg = checkpoint_config.get("sparse_attention_config")
    if not isinstance(sparse_cfg, dict):
        return None

    tsf = sparse_cfg.get("threshold_scale_factor")
    if not isinstance(tsf, dict):
        return None

    prefill = tsf.get("prefill")
    if not isinstance(prefill, dict):
        return None

    # Support both LLM format (a, b) and ModelOpt diffusion format (log_a, b)
    if "log_a" in prefill and "b" in prefill:
        return SkipSoftmaxConfig(
            formula=SkipSoftmaxFormula(log_a=prefill["log_a"], b=prefill["b"]),
        )
    elif "a" in prefill and "b" in prefill:
        return SkipSoftmaxConfig(
            formula=SkipSoftmaxFormula(log_a=math.log(prefill["a"]), b=prefill["b"]),
        )

    return None


def apply_skip_softmax_overrides(model: "torch.nn.Module", skip_softmax: SkipSoftmaxConfig) -> int:
    """Apply layer_overrides from SkipSoftmaxConfig to a constructed model.

    Walks named_modules(), matches names against layer_overrides patterns,
    and sets per-layer sparse_attention_config on TRTLLM backends.

    Call this after model construction when layer_overrides is specified.

    Returns:
        Number of backends modified.
    """
    if skip_softmax.layer_overrides is None:
        return 0

    from tensorrt_llm._torch.visual_gen.attention_backend.trtllm import TrtllmAttention
    from tensorrt_llm.llmapi.llm_args import SkipSoftmaxAttentionConfig

    modified = 0
    for name, module in model.named_modules():
        threshold = skip_softmax.resolve_threshold(name)
        # Find TRTLLM backend: could be direct .attn or inside UlyssesAttention
        attn = getattr(module, "attn", None)
        targets = []
        if isinstance(attn, TrtllmAttention):
            targets.append(attn)
        inner = getattr(attn, "inner_backend", None)
        if isinstance(inner, TrtllmAttention):
            targets.append(inner)

        for target in targets:
            if threshold is not None:
                target.sparse_attention_config = SkipSoftmaxAttentionConfig(
                    threshold_scale_factor={"prefill": threshold, "decode": 0}
                )
            else:
                target.sparse_attention_config = None
            modified += 1

    return modified


class ParallelConfig(StrictBaseModel):
    """Configuration for distributed parallelism.

    Currently Supported:
        - dit_cfg_size: CFG (Classifier-Free Guidance) parallelism
        - dit_ulysses_size: Ulysses head-sharding parallelism
        - dit_attn2d_row_size, dit_attn2d_col_size: Attention2D context parallelism

    Not Yet Supported:
        - dit_tp_size: Tensor parallelism (not implemented)
        - dit_ring_size: Ring attention context parallelism (not implemented)
        - dit_dp_size, dit_fsdp_size: Other parallelism types
        - Combining Ulysses and Attention2D (orthogonal in principle, not yet implemented)

    See mapping.py for more details.

    Example Configurations:
        1. cfg_size=1, ulysses_size=2 -> 2 GPUs (Ulysses only)
           GPU 0-1: Single prompt, heads sharded across 2 GPUs

        2. cfg_size=2, ulysses_size=1 -> 2 GPUs (CFG only)
           GPU 0: Positive prompt
           GPU 1: Negative prompt

        3. cfg_size=2, ulysses_size=2 -> 4 GPUs (CFG + Ulysses)
           GPU 0-1: CFG group 0 (positive), Ulysses parallel
           GPU 2-3: CFG group 1 (negative), Ulysses parallel

        4. cfg_size=2, ulysses_size=4 -> 8 GPUs (CFG + Ulysses)
           GPU 0-3: CFG group 0 (positive), Ulysses parallel
           GPU 4-7: CFG group 1 (negative), Ulysses parallel

        5. cfg_size=1, attn2d_row_size=2, attn2d_col_size=2 -> 4 GPUs (Attention2D only)
           2x2 mesh: Q gathered across row group, K/V gathered across col group
    """

    enable_parallel_vae: bool = True
    parallel_vae_split_dim: Literal["width", "height"] = "width"

    # DiT Parallelism
    dit_dp_size: int = PydanticField(1, ge=1)  # Not yet supported
    dit_tp_size: int = PydanticField(1, ge=1)  # Not yet supported
    dit_ulysses_size: int = PydanticField(1, ge=1)  # Supported
    dit_ring_size: int = PydanticField(1, ge=1)  # Supported
    dit_attn2d_row_size: int = PydanticField(1, ge=1)  # Supported
    dit_attn2d_col_size: int = PydanticField(1, ge=1)  # Supported
    dit_cfg_size: int = PydanticField(1, ge=1)  # Supported
    dit_fsdp_size: int = PydanticField(1, ge=1)

    # Refiner Parallelism (Optional)
    refiner_dit_dp_size: int = 1
    refiner_dit_tp_size: int = 1
    refiner_dit_ulysses_size: int = 1
    refiner_dit_ring_size: int = 1
    refiner_dit_cp_size: int = 1
    refiner_dit_cfg_size: int = 1
    refiner_dit_fsdp_size: int = 1

    t5_fsdp_size: int = 1

    @property
    def seq_parallel_size(self) -> int:
        """Parallelism degree over the sequence/context axis.

        Returns ``cp_size × dit_ulysses_size`` where ``cp_size`` is the context-parallel
        degree: Attention2D tile (``dit_attn2d_row_size × dit_attn2d_col_size``) if
        Attention2D is enabled, else ``dit_ring_size`` if ring CP is enabled, else ``1``.
        Attention2D and Ulysses can be combined (sequence × head sharding).
        """
        attn2d_size = self.dit_attn2d_row_size * self.dit_attn2d_col_size
        if attn2d_size > 1:
            cp_size = attn2d_size
        elif self.dit_ring_size > 1:
            cp_size = self.dit_ring_size
        else:
            cp_size = 1
        return cp_size * self.dit_ulysses_size

    @property
    def n_workers(self) -> int:
        return self.dit_cfg_size * self.dit_tp_size * self.seq_parallel_size

    @property
    def total_parallel_size(self) -> int:
        return self.dit_cfg_size * self.dit_tp_size * self.seq_parallel_size

    def validate_world_size(self, world_size: int) -> None:
        if self.total_parallel_size > world_size:
            raise ValueError(
                f"total_parallel_size ({self.total_parallel_size}) "
                f"exceeds world_size ({world_size})"
            )


class BaseCacheConfig(StrictBaseModel):
    """Base class for diffusion step caching acceleration configs."""

    cache_backend: str


class TeaCacheConfig(BaseCacheConfig):
    """TeaCache step-caching acceleration config."""

    cache_backend: Literal["teacache"] = "teacache"
    teacache_thresh: float = PydanticField(0.2, gt=0.0)
    use_ret_steps: bool = False

    coefficients: List[float] = PydanticField(default_factory=lambda: [1.0, 0.0])

    # Runtime state fields (initialized by TeaCacheBackend.refresh)
    ret_steps: Optional[int] = None
    cutoff_steps: Optional[int] = None
    num_steps: Optional[int] = None

    # State tracking (reset per generation)
    _cnt: int = 0

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def validate_teacache(self) -> "TeaCacheConfig":
        """Validate TeaCache configuration."""
        # Validate coefficients
        if len(self.coefficients) == 0:
            raise ValueError("TeaCache coefficients list cannot be empty")

        # Validate ret_steps if set
        if self.ret_steps is not None and self.ret_steps < 0:
            raise ValueError(f"ret_steps must be non-negative, got {self.ret_steps}")

        # Validate cutoff_steps vs num_steps if both set
        if self.cutoff_steps is not None and self.num_steps is not None:
            if self.cutoff_steps > self.num_steps:
                raise ValueError(
                    f"cutoff_steps ({self.cutoff_steps}) cannot exceed num_steps ({self.num_steps})"
                )

        return self


class CacheDiTConfig(BaseCacheConfig):
    """Configuration for Cache-DiT (DBCache, TaylorSeer, SCM).

    Requires the cache-dit package.
    """

    cache_backend: Literal["cache_dit"] = "cache_dit"
    Fn_compute_blocks: int = PydanticField(
        1, ge=0, description="First n blocks always computed (Fn)."
    )
    Bn_compute_blocks: int = PydanticField(
        0, ge=0, description="Last n blocks use residual cache (Bn)."
    )
    max_warmup_steps: int = PydanticField(
        4,
        ge=0,
        description="Initial steps that do not use cache (default tuned for few-step runs).",
    )
    max_cached_steps: int = PydanticField(
        -1,
        description="Cap on cached steps; -1 means no limit.",
    )
    max_continuous_cached_steps: int = PydanticField(
        3,
        ge=-1,
        description="Cap on consecutive cached steps (-1 = library unlimited; default 3).",
    )
    residual_diff_threshold: float = PydanticField(
        0.24,
        ge=0.0,
        description="L1 diff threshold for DBCache (default pairs with max_continuous_cached_steps).",
    )
    enable_separate_cfg: Optional[bool] = PydanticField(
        None,
        description=(
            "If set, forwarded to DBCacheConfig.enable_separate_cfg. "
            "If None, enablers pick defaults for each pipeline (Wan: batched CFG → False)."
        ),
    )
    enable_taylorseer: bool = False
    taylorseer_order: int = PydanticField(1, ge=1, le=4)

    scm_steps_mask_policy: Optional[str] = PydanticField(
        None,
        description="Policy name for cache_dit.steps_mask (e.g. fast, medium, slow, ultra).",
    )
    scm_steps_policy: Literal["dynamic", "static"] = "dynamic"

    force_refresh_step_hint: Optional[int] = PydanticField(
        None,
        description="Optional step index hint for forced cache refresh (cache_dit DBCacheConfig).",
    )
    force_refresh_step_policy: Literal["once", "repeat"] = PydanticField(
        "once",
        description="Policy for force_refresh_step_hint: once or repeat.",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)


CacheConfig = Annotated[
    Union[TeaCacheConfig, CacheDiTConfig],
    PydanticField(discriminator="cache_backend"),
]


class TorchCompileConfig(StrictBaseModel):
    """Configuration for torch.compile and autotuning.

    Warmup shapes for torch.compile specialization are configured via
    CompilationConfig (resolutions + num_frames), not here.
    """

    enable_torch_compile: bool = True
    enable_fullgraph: bool = False
    enable_autotune: bool = True


class CudaGraphConfig(StrictBaseModel):
    """Configuration for CUDA graph capture/replay.

    Warmup shapes for CUDA graph pre-capture are configured via
    CompilationConfig (resolutions + num_frames), not here.
    """

    enable_cuda_graph: bool = False


class CompilationConfig(StrictBaseModel):
    """Configuration for torch.compile / CUDA graph warmup shapes.

    Warmup shapes are the Cartesian product of ``resolutions`` and ``num_frames``.
    For example, 2 resolutions x 2 frame counts = 4 warmup shapes.

    More warmup shapes = slower startup, but lower risk of torch.compile
    recompilation delays on first requests. Fewer shapes = faster startup,
    but first request with an un-warmed shape triggers recompilation.

    If not configured, each model pipeline uses its own defaults
    (e.g., Wan uses [(480, 832), (720, 1280)] x [33, 81]).

    YAML usage (via ``--extra_visual_gen_options``)::

        # Custom warmup: 2 resolutions x 2 frame counts = 4 shapes
        compilation:
          resolutions:
            - [480, 832]
            - [720, 1280]
          num_frames: [33, 81]

        # Only override resolutions (frame counts use model defaults)
        compilation:
          resolutions:
            - [1920, 1080]

        # Skip warmup entirely
        compilation:
          resolutions: []
          num_frames: []
    """

    resolutions: Optional[List[Tuple[int, int]]] = PydanticField(
        default=None,
        description=(
            "List of (height, width) resolutions to warmup at startup. "
            "Combined with num_frames via Cartesian product. "
            "If None, uses model-specific defaults."
        ),
    )
    num_frames: Optional[List[int]] = PydanticField(
        default=None,
        description=(
            "List of frame counts to warmup at startup. "
            "Combined with resolutions via Cartesian product. "
            "If None, uses model-specific defaults. "
            "For image models, use [1]."
        ),
    )


class PipelineConfig(StrictBaseModel):
    """Model-specific pipeline configuration."""

    fuse_qkv: bool = True
    enable_layerwise_nvtx_marker: bool = False

    # Offloading
    enable_offloading: bool = False
    offload_device: Literal["cpu", "cuda"] = "cpu"
    offload_param_pin_memory: bool = True


# =============================================================================
# VisualGenArgs - User-facing configuration (CLI / YAML)
# =============================================================================


class VisualGenArgs(StrictBaseModel):
    """User-facing configuration for diffusion model loading and inference.

    This is the main config class used in CLI args and YAML config files.
    PipelineLoader converts this to DiffusionModelConfig internally.

    Example:
        args = VisualGenArgs(
            checkpoint_path="/path/to/model",
            quant_config={"quant_algo": "FP8_BLOCK_SCALES", "dynamic": True},
            parallel=ParallelConfig(dit_tp_size=2),
        )
        loader = PipelineLoader()
        pipeline = loader.load(args)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Required: Path to checkpoint or HuggingFace Hub model ID
    checkpoint_path: str = PydanticField(
        "",
        description=(
            "Local directory path or HuggingFace Hub model ID "
            "(e.g., 'Wan-AI/Wan2.1-T2V-1.3B-Diffusers'). "
            "Hub models are downloaded and cached automatically."
        ),
    )

    # Path to the text encoder model (e.g. Gemma3 directory) used by LTX-2 pipelines.
    text_encoder_path: str = PydanticField(
        "",
        description=(
            "Path to the text encoder model directory (e.g. Gemma3). "
            "Required for LTX-2 pipelines. Must contain model weights, "
            "tokenizer files, and preprocessor config."
        ),
    )

    # Two-stage LTX-2: learned spatial upsampler checkpoint path.
    spatial_upsampler_path: str = PydanticField(
        "",
        description=(
            "Path to the learned LatentUpsampler checkpoint (.safetensors). "
            "Required for LTX-2 two-stage pipelines. When provided, the "
            "pipeline auto-selects LTX2TwoStagesPipeline."
        ),
    )

    # Two-stage LTX-2: distilled LoRA checkpoint path for stage 2 refinement.
    distilled_lora_path: str = PydanticField(
        "",
        description=(
            "Path to the distilled LoRA checkpoint (.safetensors) used in "
            "the stage 2 refinement pass. The LoRA weights are merged into "
            "the transformer for stage 2 denoising and un-merged afterwards."
        ),
    )

    # HuggingFace Hub options
    revision: Optional[str] = PydanticField(
        None,
        description="HuggingFace Hub revision (branch, tag, or commit SHA) to download.",
    )

    # Device/dtype options
    device: str = "cuda"
    dtype: str = "bfloat16"

    # Component loading options (use PipelineComponent enum values or plain strings)
    skip_components: List[PipelineComponent] = PydanticField(
        default_factory=list,
        description=(
            "Components to skip loading. "
            "Accepts PipelineComponent enum values or equivalent strings "
            "(e.g., [PipelineComponent.TEXT_ENCODER, PipelineComponent.VAE])"
        ),
    )

    # Skip warmup inference after loading (useful for testing or fast startup)
    skip_warmup: bool = False

    # Path to a pre-computed fixed latent tensor (.pt file).
    # When set, the pipeline replaces sampled latents with this tensor at
    # request time (lazy-loaded on first inference). Used for MLPerf
    # deterministic generation across runs.
    fixed_latent_path: Optional[str] = PydanticField(
        None,
        description=(
            "Path to a pre-computed fixed latent tensor file (.pt). "
            "When set, the pipeline bypasses random latent sampling and "
            "uses this tensor for all real inference calls (warmup uses "
            "random latents to avoid shape-mismatch when the warmup grid "
            "differs from the latent shape)."
        ),
    )

    # Sub-configs (dict input for quant_config is coerced to QuantConfig in model_validator)
    quant_config: QuantConfig = PydanticField(default_factory=QuantConfig)
    compilation: CompilationConfig = PydanticField(default_factory=CompilationConfig)
    torch_compile: TorchCompileConfig = PydanticField(default_factory=TorchCompileConfig)
    cuda_graph: CudaGraphConfig = PydanticField(default_factory=CudaGraphConfig)
    pipeline: PipelineConfig = PydanticField(default_factory=PipelineConfig)
    attention: AttentionConfig = PydanticField(default_factory=AttentionConfig)
    parallel: ParallelConfig = PydanticField(default_factory=ParallelConfig)
    cache: Optional[CacheConfig] = None

    # Set by model_validator when quant_config is provided as a dict (ModelOpt format)
    dynamic_weight_quant: bool = False
    force_dynamic_quantization: bool = False

    @model_validator(mode="before")
    @classmethod
    def _parse_quant_config_dict(cls, data: Any) -> Any:
        """Parse user-facing VisualGenArgs.quant_config (dict or None) into QuantConfig and dynamic flags.

        User input is ModelOpt-format dict (e.g. {"quant_algo": "FP8", "dynamic": True}).
        We coerce it to QuantConfig + dynamic_weight_quant + force_dynamic_quantization so that
        from_pretrained() can copy them into DiffusionModelConfig (internal) without parsing again.
        """
        if not isinstance(data, dict):
            return data
        raw = data.get("quant_config")
        if raw is None:
            data = {**data, "quant_config": QuantConfig()}
            return data
        if not isinstance(raw, dict):
            return data
        qc, _, dwq, daq = DiffusionModelConfig.load_diffusion_quant_config(raw)
        data = {
            **data,
            "quant_config": qc,
            "dynamic_weight_quant": dwq,
            "force_dynamic_quantization": daq,
        }
        return data

    def to_mapping(self) -> Mapping:
        """Derive Mapping from ParallelConfig."""
        return self.parallel.to_mapping()

    @property
    def cache_backend(self) -> Optional[CacheBackendName]:
        return self.cache.cache_backend if self.cache is not None else None  # type: ignore[return-value]

    @property
    def teacache(self) -> Optional[TeaCacheConfig]:
        return self.cache if isinstance(self.cache, TeaCacheConfig) else None

    @property
    def cache_dit(self) -> Optional[CacheDiTConfig]:
        return self.cache if isinstance(self.cache, CacheDiTConfig) else None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump()

    @set_api_status("prototype")
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "VisualGenArgs":
        """Create from dictionary with automatic nested config parsing.

        Unknown fields cause a ValidationError (extra="forbid").
        """
        return cls(**config_dict)

    @set_api_status("prototype")
    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path], **overrides: Any) -> "VisualGenArgs":
        """Load configuration from a YAML file.

        Args:
            yaml_path: Path to the YAML configuration file.
            **overrides: Keyword arguments that override values from the YAML file.

        Returns:
            A validated VisualGenArgs instance.
        """
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f) or {}
        config_dict.update(overrides)
        return cls(**config_dict)


# =============================================================================
# Utilities
# =============================================================================


def discover_pipeline_components(checkpoint_path: Path) -> Dict[str, Path]:
    """
    Discover components from diffusers pipeline's model_index.json.

    Returns dict mapping component name to config.json path.
    """
    model_index_path = checkpoint_path / "model_index.json"
    if not model_index_path.exists():
        return {}

    with open(model_index_path) as f:
        model_index = json.load(f)

    components = {}
    for key, value in model_index.items():
        if key.startswith("_") or value is None:
            continue
        config_path = checkpoint_path / key / "config.json"
        if config_path.exists():
            components[key] = config_path

    return components


def create_attention_metadata_state() -> Dict[str, Any]:
    """Create model-scoped attention metadata state for TRTLLM visual-gen backend."""
    return {"metadata": None, "capacity": (0, 0)}


# =============================================================================
# DiffusionModelConfig - Internal configuration (merged/parsed)
# =============================================================================


class DiffusionModelConfig(BaseModel):
    """Internal ModelConfig for diffusion models.

    This is created by PipelineLoader from VisualGenArgs + checkpoint.
    Contains merged/parsed config from:
    - pretrained_config: From checkpoint/config.json
    - quant_config: From checkpoint or user quant config
    - Sub-configs: From VisualGenArgs (pipeline, attention, teacache)
    - visual_gen_mapping: Populated by setup_visual_gen_mapping() from ParallelConfig
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    pretrained_config: Optional[Any] = None
    mapping: Mapping = PydanticField(default_factory=Mapping)
    skip_create_weights_in_init: bool = False
    force_dynamic_quantization: bool = False
    allreduce_strategy: AllReduceStrategy = PydanticField(default=AllReduceStrategy.AUTO)
    extra_attrs: Dict = PydanticField(default_factory=dict)

    # Unified parallelism mapping (populated by setup_visual_gen_mapping)
    visual_gen_mapping: Optional[Any] = None  # VisualGenMapping (lazy import)

    # VAE parallelism (promoted from ParallelConfig for pipeline_loader)
    enable_parallel_vae: bool = True
    parallel_vae_split_dim: Literal["width", "height"] = "width"

    dynamic_weight_quant: bool = False

    # Sub-configs from VisualGenArgs (merged during from_pretrained)
    quant_config: QuantConfig = PydanticField(default_factory=QuantConfig)
    # Per-layer quant (from load_diffusion_quant_config layer_quant_config; None until mixed-precision parsing exists)
    quant_config_dict: Optional[Dict[str, QuantConfig]] = None
    compilation: CompilationConfig = PydanticField(default_factory=CompilationConfig)
    torch_compile: TorchCompileConfig = PydanticField(default_factory=TorchCompileConfig)
    cuda_graph: CudaGraphConfig = PydanticField(default_factory=CudaGraphConfig)
    pipeline: PipelineConfig = PydanticField(default_factory=PipelineConfig)
    attention: AttentionConfig = PydanticField(default_factory=AttentionConfig)
    attention_metadata_state: Optional[Dict[str, Any]] = None
    parallel: ParallelConfig = PydanticField(default_factory=ParallelConfig)
    cache: Optional[CacheConfig] = None

    @property
    def cache_backend(self) -> Optional[CacheBackendName]:
        return self.cache.cache_backend if self.cache is not None else None  # type: ignore[return-value]

    @property
    def teacache(self) -> Optional[TeaCacheConfig]:
        return self.cache if isinstance(self.cache, TeaCacheConfig) else None

    @property
    def cache_dit(self) -> Optional[CacheDiTConfig]:
        return self.cache if isinstance(self.cache, CacheDiTConfig) else None

    @property
    def torch_dtype(self) -> "torch.dtype":
        """Get the torch dtype of the model (default: bfloat16)."""
        return torch.bfloat16

    def get_quant_config(self, name: Optional[str] = None) -> QuantConfig:
        """Get quantization config for a layer or global. Resembles LLM ModelConfig.get_quant_config."""
        if name is None or self.quant_config_dict is None:
            return self.quant_config
        if name in self.quant_config_dict:
            return self.quant_config_dict[name]
        return self.quant_config

    @staticmethod
    def load_diffusion_quant_config(
        quant_config_dict: dict,
    ) -> Tuple[QuantConfig, Optional[Dict], bool, bool]:
        """
        Parse quantization config in ModelOpt format.

        Returns: (quant_config, layer_quant_config, dynamic_weight_quant, dynamic_activation_quant)
            - quant_config: Global QuantConfig
            - layer_quant_config: Per-layer config dict (None if not using mixed precision)
            - dynamic_weight_quant: Whether to quantize weights at load time
            - dynamic_activation_quant: Whether to quantize activations dynamically
        """
        quant_algo_str = quant_config_dict.get("quant_algo")
        quant_algo = None
        if quant_algo_str:
            algo_map = {
                "FP8": QuantAlgo.FP8,
                "FP8_BLOCK_SCALES": QuantAlgo.FP8_BLOCK_SCALES,
                "NVFP4": QuantAlgo.NVFP4,
                "W4A16_AWQ": QuantAlgo.W4A16_AWQ,
                "W4A8_AWQ": QuantAlgo.W4A8_AWQ,
                "W8A8_SQ_PER_CHANNEL": QuantAlgo.W8A8_SQ_PER_CHANNEL,
            }
            quant_algo = algo_map.get(quant_algo_str)
            if quant_algo is None:
                raise ValueError(f"Unknown quant_algo: {quant_algo_str}")

        # Parse group_size and dynamic flags from config_groups
        group_size = None
        dynamic_weight_quant = False
        dynamic_activation_quant = False
        for group_config in quant_config_dict.get("config_groups", {}).values():
            weights_config = group_config.get("weights", {})
            activations_config = group_config.get("input_activations", {})
            dynamic_weight_quant = weights_config.get("dynamic", False)
            dynamic_activation_quant = activations_config.get("dynamic", False)
            # Extract group_size from weights config (e.g., NVFP4: group_size=16)
            if group_size is None:
                group_size = weights_config.get("group_size")
            break

        # Set defaults based on quant_algo if group_size not specified
        if group_size is None:
            if quant_algo in (QuantAlgo.NVFP4,):
                group_size = 16  # NVFP4 default
            elif quant_algo == QuantAlgo.FP8_BLOCK_SCALES:
                group_size = 128  # FP8 blockwise default

        # Auto-enable dynamic weight quantization if quant_algo is specified
        # but no explicit config_groups setting is present.
        # This allows simple configs like {"quant_algo": "FP8"} to work.
        if quant_algo is not None and not quant_config_dict.get("config_groups"):
            dynamic_weight_quant = quant_config_dict.get("dynamic", True)
            # NVFP4 requires dynamic activation quantization when using dynamic mode
            # since input_scale is not calibrated
            if quant_algo == QuantAlgo.NVFP4 and dynamic_weight_quant:
                dynamic_activation_quant = True

        quant_config = QuantConfig(
            quant_algo=quant_algo,
            group_size=group_size,
            exclude_modules=quant_config_dict.get("ignore"),
        )

        # TODO: Per-layer config (None for now - future: parse mixed precision settings)
        layer_quant_config = None

        return quant_config, layer_quant_config, dynamic_weight_quant, dynamic_activation_quant

    @staticmethod
    def _convert_quantization_metadata(
        qmeta: Dict,
        tensor_keys: List[str],
    ) -> Dict:
        """
        TODO: Consider refactor this to be a utility functions.
        Convert per-layer ``_quantization_metadata`` to ModelOpt format.

        Some checkpoints (e.g. HuggingFace-quantized FP8) embed per-layer
        quantization info as::

            {"format_version": "1.0",
             "layers": {"model.diffusion_model.block.attn.to_q": {"format": "float8_e4m3fn"}, ...}}

        This converts it to the ModelOpt-compatible dict that
        :meth:`load_diffusion_quant_config` understands::

            {"quant_algo": "FP8",
             "config_groups": {"default": {"weights": {"dynamic": false}, ...}},
             "ignore": ["proj_in", "proj_out", ...]}
        """
        _FORMAT_TO_ALGO = {
            "float8_e4m3fn": "FP8",
        }

        layers = qmeta.get("layers", {})
        if not layers:
            return {}

        formats = {info.get("format") for info in layers.values()}
        if len(formats) != 1:
            logger.warning(f"_quantization_metadata has mixed formats {formats}; skipping")
            return {}

        fmt = formats.pop()
        quant_algo = _FORMAT_TO_ALGO.get(fmt)
        if quant_algo is None:
            logger.warning(f"_quantization_metadata format '{fmt}' is not supported; skipping")
            return {}

        quantized_layers = set(layers.keys())

        # Build ignore list: weight-bearing layers NOT in the quantized set.
        # Tensor keys ending with ".weight" (but not ".weight_scale") indicate
        # layers that own learnable weights.
        non_quantized = []
        for key in tensor_keys:
            if key.endswith(".weight") and not key.endswith("_scale.weight"):
                layer_name = key[: -len(".weight")]
                if layer_name not in quantized_layers:
                    non_quantized.append(layer_name)

        result = {
            "quant_algo": quant_algo,
            "config_groups": {
                "default": {
                    "weights": {"dynamic": False},
                    "input_activations": {"dynamic": False},
                }
            },
            "ignore": sorted(non_quantized),
        }
        logger.info(
            f"Converted _quantization_metadata: algo={quant_algo}, "
            f"{len(quantized_layers)} quantized layers, "
            f"{len(non_quantized)} excluded layers"
        )
        return result

    @classmethod
    def _try_load_safetensors_config(cls, checkpoint_path: Path) -> Optional[Dict]:
        """Try to read embedded config from a single-safetensors checkpoint.

        Accepts either a directory (globs for ``*.safetensors``) or a direct
        path to a ``.safetensors`` file.

        Returns the full config dict if found, ``None`` otherwise.
        """
        try:
            import safetensors.torch
        except ImportError:
            return None

        if checkpoint_path.is_file() and checkpoint_path.suffix == ".safetensors":
            sft_files = [checkpoint_path]
        else:
            sft_files = sorted(checkpoint_path.glob("*.safetensors"))

        if not sft_files:
            return None

        try:
            with safetensors.torch.safe_open(str(sft_files[0]), framework="pt") as f:
                meta = f.metadata()
                if meta and "config" in meta:
                    config = json.loads(meta["config"])
                    if "quantization_config" in meta:
                        config["quantization_config"] = json.loads(meta["quantization_config"])
                    elif "_quantization_metadata" in meta:
                        qmeta = json.loads(meta["_quantization_metadata"])
                        converted = cls._convert_quantization_metadata(qmeta, list(f.keys()))
                        if converted:
                            config["quantization_config"] = converted
                    return config
        except Exception:
            pass
        return None

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_dir: str,
        args: Optional["VisualGenArgs"] = None,
        **kwargs,
    ) -> "DiffusionModelConfig":
        """
        Load config from pretrained checkpoint.

        Called by PipelineLoader with VisualGenArgs:
            config = DiffusionModelConfig.from_pretrained(
                checkpoint_dir=args.checkpoint_path,
                args=args,
            )

        Supports two checkpoint formats:
        * **Diffusers directory layout** -- ``model_index.json`` with
          component sub-directories each containing ``config.json``.
        * **Single-safetensors** -- no ``model_index.json``; config embedded
          in the safetensors metadata header under a ``"config"`` key.  The
          transformer section is extracted as ``pretrained_config`` and the
          full dict is stored in ``extra_attrs["monolithic_safetensors_config"]``
          for use by component configurators.

        Args:
            checkpoint_dir: Path to checkpoint
            args: VisualGenArgs containing user config
                - (compilation, torch_compile, cuda_graph, pipeline, attention, parallel, teacache,
                   cache_backend, cache_dit)
            **kwargs: Additional config options (e.g., mapping)
        """
        kwargs.pop("trust_remote_code", None)

        # Extract sub-configs from args or use defaults
        compilation_cfg = args.compilation if args else CompilationConfig()
        torch_compile_cfg = args.torch_compile if args else TorchCompileConfig()
        cuda_graph_cfg = args.cuda_graph if args else CudaGraphConfig()
        pipeline_cfg = args.pipeline if args else PipelineConfig()
        attention_cfg = args.attention if args else AttentionConfig()
        parallel_cfg = args.parallel if args else ParallelConfig()
        cache_cfg = args.cache if args else None

        component = PipelineComponent.TRANSFORMER
        checkpoint_path = Path(checkpoint_dir)
        extra_attrs: Dict[str, Any] = {}

        # Propagate two-stage paths into extra_attrs for pipeline use
        if args and args.spatial_upsampler_path:
            extra_attrs["spatial_upsampler_path"] = args.spatial_upsampler_path
        if args and args.distilled_lora_path:
            extra_attrs["distilled_lora_path"] = args.distilled_lora_path
        # Propagate MLPerf-deterministic fixed latent path into extra_attrs.
        if args and args.fixed_latent_path:
            extra_attrs["fixed_latent_path"] = args.fixed_latent_path

        # Discover pipeline components (diffusers layout)
        components = discover_pipeline_components(checkpoint_path)

        if components:
            # ---------- Diffusers directory layout ----------
            if component not in components:
                raise ValueError(
                    f"Component '{component}' not found. Available: {list(components.keys())}"
                )
            config_path = components[component]
            if not config_path.exists():
                raise ValueError(f"Config not found at {config_path}")

            with open(config_path) as f:
                config_dict = json.load(f)
            pretrained_config = SimpleNamespace(**config_dict)

            # Ensure _name_or_path is set so TeaCache coefficient matching works.
            if not getattr(pretrained_config, "_name_or_path", None):
                pretrained_config._name_or_path = str(checkpoint_path)

            model_index_path = checkpoint_path / "model_index.json"
            if model_index_path.exists():
                with open(model_index_path) as f:
                    model_index = json.load(f)
                if "boundary_ratio" in model_index and "transformer_2" in model_index:
                    transformer_2_spec = model_index.get("transformer_2")
                    if transformer_2_spec and transformer_2_spec[0] is not None:
                        pretrained_config.boundary_ratio = model_index["boundary_ratio"]
                if "expand_timesteps" in model_index:
                    pretrained_config.expand_timesteps = bool(model_index["expand_timesteps"])
        else:
            # ---------- Single safetensors ----------
            native_config = cls._try_load_safetensors_config(checkpoint_path)

            if native_config is not None:
                transformer_dict = native_config.get("transformer", {})
                pretrained_config = SimpleNamespace(**transformer_dict)
                extra_attrs["monolithic_safetensors_config"] = native_config

                # quantization_config lives as a separate safetensors metadata
                # key, not inside the transformer section. Propagate it so
                # the quant-config resolution below can pick it up.
                if "quantization_config" in native_config:
                    qc = native_config["quantization_config"]
                    # ModelOpt prefixes module names with the wrapped model
                    # attribute (e.g. "velocity_model.proj_out"). Strip that
                    # wrapper prefix so the ignore list matches TRT-LLM names.
                    _MODELOPT_WRAPPER_PREFIXES = (
                        "model.diffusion_model.",
                        "velocity_model.",
                        "denoiser.",
                        "unet.",
                        "dit.",
                    )
                    if "ignore" in qc and qc["ignore"]:
                        cleaned = []
                        for entry in qc["ignore"]:
                            for wp in _MODELOPT_WRAPPER_PREFIXES:
                                if entry.startswith(wp):
                                    entry = entry[len(wp) :]
                                    break
                            cleaned.append(entry)
                        qc["ignore"] = cleaned
                    pretrained_config.quantization_config = qc
            else:
                raise ValueError(
                    f"Config not found at {checkpoint_dir}. "
                    "Expected model_index.json (diffusers) or "
                    "safetensors with embedded config metadata."
                )

        # Load sparse attention config.
        # Step 1: Load base config from YAML/checkpoint (formula + disabled_layers)
        # Step 2: Merge user-provided fields on top (target_sparsity, threshold)
        yaml_sparse = None

        # Try manual YAML path
        yaml_path = attention_cfg.sparse_config_path
        if yaml_path is not None:
            yaml_sparse = load_sparse_config_from_yaml(yaml_path)
            if yaml_sparse is not None:
                logger.info(f"Loaded sparse config from {yaml_path}")

        # Try auto-detect YAML in checkpoint
        if yaml_sparse is None:
            yaml_configs = auto_detect_sparse_yaml(str(checkpoint_path))
            if yaml_configs:
                first_key = next(iter(yaml_configs))
                yaml_sparse = yaml_configs[first_key]
                logger.info(
                    f"Auto-detected sparse config from {first_key}.yaml "
                    f"(formula: log_a={yaml_sparse.formula.log_a:.2f}, b={yaml_sparse.formula.b:.2f})"
                )

        # Try auto-detect from config.json
        if yaml_sparse is None:
            ckpt_dict = vars(pretrained_config) if pretrained_config else {}
            yaml_sparse = auto_detect_sparse_attention_config(ckpt_dict)
            if yaml_sparse is not None:
                logger.info(
                    f"Auto-detected sparse config from config.json "
                    f"(formula: log_a={yaml_sparse.formula.log_a:.2f}, b={yaml_sparse.formula.b:.2f})"
                )

        # Merge: YAML provides formula + disabled_layers, user provides
        # target_sparsity / threshold_scale_factor / additional overrides
        if yaml_sparse is not None:
            user_cfg = attention_cfg.sparse_attention_config
            if user_cfg is not None and isinstance(user_cfg, SkipSoftmaxConfig):
                # User provided some fields — merge on top of YAML
                merged = yaml_sparse.model_copy(
                    update={
                        k: v
                        for k, v in {
                            "threshold_scale_factor": user_cfg.threshold_scale_factor,
                            "target_sparsity": user_cfg.target_sparsity,
                            "formula": user_cfg.formula or yaml_sparse.formula,
                            "layer_overrides": user_cfg.layer_overrides
                            or yaml_sparse.layer_overrides,
                        }.items()
                        if v is not None
                    }
                )
                attention_cfg = attention_cfg.model_copy(update={"sparse_attention_config": merged})
            else:
                # No user config — use YAML as-is
                attention_cfg = attention_cfg.model_copy(
                    update={"sparse_attention_config": yaml_sparse}
                )

        # Resolve quant config
        if args and args.quant_config.quant_algo is not None:
            quant_config = args.quant_config
            quant_config_dict = (
                None  # VisualGenArgs has no per-layer dict; only from checkpoint parse
            )
            dynamic_weight_quant = args.dynamic_weight_quant
            dynamic_activation_quant = args.force_dynamic_quantization
        else:
            quant_config = QuantConfig()
            quant_config_dict = None
            dynamic_weight_quant = False
            dynamic_activation_quant = False
            quant_dict = getattr(pretrained_config, "quantization_config", None)
            if isinstance(quant_dict, dict):
                quant_config, quant_config_dict, dynamic_weight_quant, dynamic_activation_quant = (
                    cls.load_diffusion_quant_config(quant_dict)
                )

        # Enable tunable FP4 quantize for visual gen: larger activation
        # tensors (full image/video latents) amortize the AutoTuner overhead.
        if quant_config.quant_algo == QuantAlgo.NVFP4:
            from tensorrt_llm._torch.modules.linear import NVFP4LinearMethod

            NVFP4LinearMethod.use_tunable_quantize = True

        attention_metadata_state = (
            create_attention_metadata_state() if attention_cfg.backend == "TRTLLM" else None
        )

        return cls(
            pretrained_config=pretrained_config,
            quant_config=quant_config,
            quant_config_dict=quant_config_dict,
            dynamic_weight_quant=dynamic_weight_quant,
            force_dynamic_quantization=dynamic_activation_quant,
            # Sub-configs from VisualGenArgs
            compilation=compilation_cfg,
            torch_compile=torch_compile_cfg,
            cuda_graph=cuda_graph_cfg,
            pipeline=pipeline_cfg,
            attention=attention_cfg,
            attention_metadata_state=attention_metadata_state,
            parallel=parallel_cfg,
            cache=cache_cfg,
            skip_create_weights_in_init=True,
            extra_attrs=extra_attrs,
            # Promote VAE-parallelism knobs from ParallelConfig so pipeline_loader's
            # `if config.enable_parallel_vae` and pipeline.setup_parallel_vae's check
            # see the user-supplied value rather than the field default.
            enable_parallel_vae=parallel_cfg.enable_parallel_vae,
            parallel_vae_split_dim=parallel_cfg.parallel_vae_split_dim,
            **kwargs,
        )
