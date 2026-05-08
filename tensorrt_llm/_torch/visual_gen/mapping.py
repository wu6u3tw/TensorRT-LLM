"""Unified multi-dimensional communicator mesh for visual generation models.

VisualGenMapping subclasses DeviceMeshTopologyImpl and overrides build_mesh()
to create a single PyTorch DeviceMesh covering all parallelism axes
(CFG, TP, CP, Ulysses).  The resulting mesh is stored in the shared
DeviceMeshTopologyImpl.device_mesh class variable so that any Mapping object
constructed afterward (e.g. via to_llm_mapping()) can reuse the same
process groups.
"""

from __future__ import annotations

from typing import Optional

import torch.distributed as dist
from torch.distributed import ProcessGroup
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from tensorrt_llm._torch.device_mesh import DeviceMeshTopologyImpl, SingleProcessGroup
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

# Legacy mesh: single CP dimension (ring or unused placeholder size 1).
_DEVICE_MESH_DIM_ORDER_LEGACY = "cfg-tp-cp-ulysses"
# Attention2D mesh: split CP into a 2D tile (cp_row × cp_col) so row/col
# subgroups and seq flatten (cp_row, cp_col, ulysses) are well-defined.
_DEVICE_MESH_DIM_ORDER_ATTN2D = "cfg-tp-cp_row-cp_col-ulysses"


class VisualGenMapping(DeviceMeshTopologyImpl):
    """Multi-dimensional communicator mesh for visual generation models.

    Parallelism Strategy:
    - CFG Parallelism: Distributes positive/negative prompts across GPUs
    - CP (Context Parallelism): Ring attention or Attention2D (sequence sharding)
    - Ulysses Parallelism: Head sharding within each CFG group

    Parallelism Hierarchy:
        total_workers = cfg × sp
        sp (sequence parallelism) = cp × ulysses
        cp (context parallelism)  = ring           [ring attention, 1D ``cp`` mesh dim]
                                  | attn2d         [``cp_row`` × ``cp_col`` mesh dims]

        cfg:     Splits positive/negative CFG prompts across GPUs (independent streams).
        tp:      Tensor parallelism all-reduce within tp groups.
        sp:      Total sequence-axis parallel degree = cp × ulysses.
          cp:    Shards the sequence dimension across ranks.
            ring:   Passes KV blocks in a ring; single ``cp`` mesh dimension.
            attn2d: Two mesh dimensions ``cp_row`` and ``cp_col``; Q/KV gather along
                    row/col fibers; logical ``cp_rank = cp_row * col + cp_col``.
          ulysses: Shards heads via all-to-all (head-sharding, not sequence-sharding).

    Mesh schemas (build-time branch on Attention2D active or not):
    - **Legacy** (``attn2d_row_size * attn2d_col_size == 1``): dims
      ``cfg``, ``tp``, ``cp``, ``ulysses``;
      ``seq_mesh = flatten(cp, ulysses)``;
      ``seq_rank = cp_rank * ulysses_size + ulysses_rank``.
    - **Attention2D** (2D CP tile active): dims ``cfg``, ``tp``, ``cp_row``,
      ``cp_col``, ``ulysses``;
      ``seq_mesh = flatten(cp_row, cp_col, ulysses)``;
      ``seq_rank = (cp_row_rank * col + cp_col_rank) * ulysses_size + ulysses_rank``.

    Fixed inner ordering (implementation detail): Ulysses innermost for latency;
    CP next; TP; CFG outermost.

    Rank linearisation (single-node and multi-node)
    -----------------------------------------------
    ``init_device_mesh`` lays out **global ranks** `0 … world_size-1` in **row-major
    (C-style) order** over ``mesh_shape`` following ``mesh_dim_names`` **left to
    right**: the **leftmost** named dimension is the **slowest** varying index in
    the flat rank number; the **rightmost** is the **fastest** varying.

    **Legacy** mesh ``(cfg, tp, cp, ulysses)``::

        rank = ((i_cfg * T + i_tp) * C + i_cp) * U + i_uly

    with ``T = tp_size``, ``C = cp_size``, ``U = ulysses_size``.

    **Attention2D** mesh ``(cfg, tp, cp_row, cp_col, ulysses)``::

        rank = (((i_cfg * T + i_tp) * R + i_r) * Co + i_c) * U + i_uly

    with ``R = attn2d_row_size``, ``Co = attn2d_col_size``, ``U = ulysses_size``.

    **Sequence index** ``seq_rank`` matches flattening ``(cp_row, cp_col,
    ulysses)`` in that same row-major order — i.e. the same order as the
    three-dimensional submesh ``cp_row``, ``cp_col``, ``ulysses`` as built in
    ``VisualGenMapping.build_mesh``:

    .. code-block:: text

        seq_rank = (i_r * Co + i_c) * U + i_uly

    which is identical to ``cp_rank * U + i_uly`` with
    ``cp_rank = i_r * Co + i_c``.

    **Logical CP** row-major ``(i_r, i_c)`` matches the previous single-``cp``
    Attn2D layout where ``cp_rank`` counted across the tile in row-major order.

    Multi-node launchers (MPI, SLURM, etc.) assign **global** rank → host/GPU.
    That assignment is **outside** this module; for predictable NCCL topology,
    place ranks so that process groups you care about (e.g. ``ulysses_group``,
    ``seq_group``, Attention2D row/col fibers) land on the NVLink / node edges
    you want. Because **ulysses** is the **innermost** mesh dimension, ranks that
    share ``(cfg, tp, cp_row, cp_col)`` have **consecutive** global rank IDs
    (varying only in ``ulysses_rank``).

    Callers should use rank and process-group properties only, not raw mesh layout.
    """

    # Flattened sequence mesh, cached after build_mesh().  Shared
    # across instances via the class object.
    seq_mesh: Optional[DeviceMesh] = None

    def __init__(
        self,
        world_size: int,
        rank: int,
        cfg_size: int = 1,
        tp_size: int = 1,
        ring_size: int = 1,
        ulysses_size: int = 1,
        attn2d_row_size: int = 1,
        attn2d_col_size: int = 1,
    ):
        # cp_size unifies ring and Attention2D under one logical context-parallelism degree.
        attn2d_size = attn2d_row_size * attn2d_col_size
        if ring_size > 1 and attn2d_size > 1:
            raise ValueError(
                "Ring and Attention2D are mutually exclusive: both shard the sequence "
                f"dimension. Got ring_size={ring_size}, attn2d={attn2d_row_size}x{attn2d_col_size}."
            )

        self._use_attn2d_plane = attn2d_size > 1

        if self._use_attn2d_plane:
            cp_size = attn2d_size
            if tp_size > 1:
                raise NotImplementedError(
                    "Combining Attention2D and TP is not yet supported. "
                    "The row/col group construction does not account for TP ranks."
                )
        else:
            cp_size = ring_size

        product = cfg_size * tp_size * cp_size * ulysses_size
        if product != world_size:
            raise ValueError(
                f"cfg({cfg_size}) * tp({tp_size}) * cp({cp_size}) * "
                f"ulysses({ulysses_size}) = {product} != world_size({world_size})"
            )

        if self._use_attn2d_plane:
            dim_order = _DEVICE_MESH_DIM_ORDER_ATTN2D
            dims = dim_order.split("-")
            self._dim_sizes = {
                "cfg": cfg_size,
                "tp": tp_size,
                "cp_row": attn2d_row_size,
                "cp_col": attn2d_col_size,
                "ulysses": ulysses_size,
            }
        else:
            dim_order = _DEVICE_MESH_DIM_ORDER_LEGACY
            dims = dim_order.split("-")
            self._dim_sizes = {
                "cfg": cfg_size,
                "tp": tp_size,
                "cp": cp_size,
                "ulysses": ulysses_size,
            }

        self.world_size = world_size
        self._rank = rank
        self.cfg_size = cfg_size
        self.tp_size = tp_size
        self.ring_size = ring_size
        self.cp_size = cp_size
        self.ulysses_size = ulysses_size
        self.attn2d_row_size = attn2d_row_size
        self.attn2d_col_size = attn2d_col_size
        self._dim_names = tuple(dims)
        self._attn2d_row_group: Optional[ProcessGroup] = None
        self._attn2d_col_group: Optional[ProcessGroup] = None
        # Flattened ``cp_row`` × ``cp_col`` submesh (Attention2D logical CP tile).
        self._cp_plane_mesh_flat: Optional[DeviceMesh] = None

        if dist.is_initialized() and world_size > 1:
            self.build_mesh()

    # ------------------------------------------------------------------
    # Mesh construction
    # ------------------------------------------------------------------
    def build_mesh(self):
        cls = DeviceMeshTopologyImpl
        if cls.device_mesh is not None:
            return

        shape = tuple(self._dim_sizes[d] for d in self._dim_names)
        cls.device_mesh = init_device_mesh(
            "cuda",
            mesh_shape=shape,
            mesh_dim_names=self._dim_names,
        )
        logger.debug(
            f"VisualGenMapping.build_mesh: dims={self._dim_names}, "
            f"shape={shape}, mesh={cls.device_mesh}"
        )

        if self.seq_size > 1:
            if self._use_attn2d_plane:
                names = self._dim_names
                i_row = names.index("cp_row")
                i_col = names.index("cp_col")
                i_uly = names.index("ulysses")
                if not (i_row + 1 == i_col and i_col + 1 == i_uly):
                    raise RuntimeError(
                        "seq_mesh requires adjacent mesh dims cp_row, cp_col, ulysses; "
                        f"fix dim order (got {names!r})."
                    )
                VisualGenMapping.seq_mesh = cls.device_mesh["cp_row", "cp_col", "ulysses"]._flatten(
                    mesh_dim_name="seq"
                )
            else:
                cp_idx = self._dim_names.index("cp")
                uly_idx = self._dim_names.index("ulysses")
                if abs(cp_idx - uly_idx) != 1:
                    raise RuntimeError(
                        "seq_mesh requires cp and ulysses adjacent; "
                        f"fix dim order (got {self._dim_names!r})"
                    )
                VisualGenMapping.seq_mesh = cls.device_mesh["cp", "ulysses"]._flatten(
                    mesh_dim_name="seq"
                )

        if self._use_attn2d_plane:
            self._attach_attn2d_groups_from_device_mesh()

    def _attach_attn2d_groups_from_device_mesh(self) -> None:
        """Set Attention2D row/col process groups from the ``cp_row`` × ``cp_col`` submesh.

        Row group (Q all-gather + output reduce-scatter): fixed ``cp_row``, vary ``cp_col``.
        Col group (K/V all-gather): fixed ``cp_col``, vary ``cp_row``.
        """
        cls = DeviceMeshTopologyImpl
        assert cls.device_mesh is not None
        cp_sub: DeviceMesh = cls.device_mesh["cp_row", "cp_col"]
        self._cp_plane_mesh_flat = cp_sub._flatten(mesh_dim_name="cp")
        # Same row r, columns c \in [0, col): fiber along ``cp_col``.
        self._attn2d_row_group = cp_sub["cp_col"].get_group()
        # Same col c, rows r \in [0, row): fiber along ``cp_row``.
        self._attn2d_col_group = cp_sub["cp_row"].get_group()
        logger.debug(
            "VisualGenMapping._attach_attn2d_groups_from_device_mesh: "
            f"row_size={self.attn2d_row_size}, col_size={self.attn2d_col_size}"
        )

    # ------------------------------------------------------------------
    # Rank decomposition
    # ------------------------------------------------------------------
    def _local_rank(self, dim: str) -> int:
        cls = DeviceMeshTopologyImpl
        if cls.device_mesh is None:
            return 0
        return cls.device_mesh[dim].get_local_rank()

    @property
    def cfg_rank(self) -> int:
        return self._local_rank("cfg")

    @property
    def tp_rank(self) -> int:
        return self._local_rank("tp")

    @property
    def cp_row_rank(self) -> int:
        if not self._use_attn2d_plane:
            return 0
        return self._local_rank("cp_row")

    @property
    def cp_col_rank(self) -> int:
        if not self._use_attn2d_plane:
            return 0
        return self._local_rank("cp_col")

    @property
    def cp_rank(self) -> int:
        """Logical CP index within the context-parallel tile (row-major if Attention2D)."""
        if self._use_attn2d_plane:
            return self.cp_row_rank * self.attn2d_col_size + self.cp_col_rank
        return self._local_rank("cp")

    @property
    def ulysses_rank(self) -> int:
        return self._local_rank("ulysses")

    # Combined sequence-parallel dimension: cp × ulysses.
    @property
    def seq_size(self) -> int:
        return self.cp_size * self.ulysses_size

    @property
    def seq_rank(self) -> int:
        if self._use_attn2d_plane:
            return (
                self.cp_row_rank * self.attn2d_col_size + self.cp_col_rank
            ) * self.ulysses_size + self.ulysses_rank
        return self.cp_rank * self.ulysses_size + self.ulysses_rank

    @property
    def ring_rank(self) -> int:
        return self.cp_rank

    @property
    def attn2d_mesh_rank(self) -> int:
        """Rank within the Attention2D CP tile (same as logical ``cp_rank``)."""
        return self.cp_rank

    @property
    def is_cfg_conditional(self) -> bool:
        return self.cfg_rank == 0

    # ------------------------------------------------------------------
    # Process groups (None when size == 1 and mesh was not built)
    # ------------------------------------------------------------------
    def _group(self, dim: str) -> Optional[ProcessGroup]:
        cls = DeviceMeshTopologyImpl
        if cls.device_mesh is None:
            if self.world_size == 1:
                return SingleProcessGroup.get_group()
            return None
        return cls.device_mesh[dim].get_group()

    def _logical_cp_group(self) -> Optional[ProcessGroup]:
        """Full logical CP tile (all ``cp_row`` × ``cp_col`` ranks in one group)."""
        if not self._use_attn2d_plane:
            return self._group("cp")
        if self._cp_plane_mesh_flat is None:
            cls = DeviceMeshTopologyImpl
            if cls.device_mesh is None:
                if self.world_size == 1:
                    return SingleProcessGroup.get_group()
                return None
            cp_sub: DeviceMesh = cls.device_mesh["cp_row", "cp_col"]
            self._cp_plane_mesh_flat = cp_sub._flatten(mesh_dim_name="cp")
        return self._cp_plane_mesh_flat.get_group()

    @property
    def ulysses_group(self) -> Optional[ProcessGroup]:
        return self._group("ulysses")

    @property
    def cp_group(self) -> Optional[ProcessGroup]:
        return self._logical_cp_group()

    @property
    def tp_group_pg(self) -> Optional[ProcessGroup]:
        return self._group("tp")

    @property
    def cfg_group(self) -> Optional[ProcessGroup]:
        return self._group("cfg")

    @property
    def ring_group(self) -> Optional[ProcessGroup]:
        """Process group for RingAttention (1D ``cp`` mesh when ``ring_size > 1``)."""
        return self._logical_cp_group()

    @property
    def attn2d_row_group(self) -> Optional[ProcessGroup]:
        return self._attn2d_row_group

    @property
    def attn2d_col_group(self) -> Optional[ProcessGroup]:
        return self._attn2d_col_group

    @property
    def attn2d_mesh_group(self) -> Optional[ProcessGroup]:
        """Full logical CP group for Attention2D (same as ``cp_group``)."""
        return self._logical_cp_group()

    @property
    def seq_group(self) -> Optional[ProcessGroup]:
        """Process group spanning the flattened (CP × Ulysses) sequence mesh."""
        cls = DeviceMeshTopologyImpl
        if cls.device_mesh is None:
            if self.world_size == 1:
                return SingleProcessGroup.get_group()
            return None
        if self.seq_size == 1:
            return self._group("ulysses")
        if VisualGenMapping.seq_mesh is None:
            return None
        return VisualGenMapping.seq_mesh.get_group()

    # ------------------------------------------------------------------
    # Bridge to LLM Mapping (for Linear layers)
    # ------------------------------------------------------------------
    def to_llm_mapping(self) -> Mapping:
        """Return a ``Mapping`` whose TP group is backed by this mesh's TP dim."""
        return Mapping(
            world_size=self.tp_size,
            rank=self.tp_rank,
            tp_size=self.tp_size,
        )
