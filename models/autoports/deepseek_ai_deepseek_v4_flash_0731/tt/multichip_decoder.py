"""Four-device tensor-parallel DeepSeek-V4-Flash decoder.

The implementation keeps :class:`OptimizedDecoder` as its numerical and
per-device program baseline.  Layer inputs/outputs, mHC, routing, the single KV
head, and page metadata are replicated.  Query heads, grouped output rank, and
every active expert's intermediate dimension are split four ways; row-parallel
attention and MoE outputs are reduced once per sublayer.

Only the complete p300c QuietBox mesh is supported.  Keeping one explicit
topology avoids runtime fallback among unmeasured mesh configurations.
"""

from __future__ import annotations

import copy
import dataclasses
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule

from ..config import AttentionKind, DeepSeekV4FlashConfig
from .expert_pool import ExpertHostStore, ExpertPool
from .functional_decoder import (
    ATTENTION_MASK_VALUE,
    HCA_CACHE_GATHER_CHUNK_ROWS,
    HostLayerWeights,
    PrefillPlan,
    _round_up,
)
from .optimized_decoder import (
    DecoderCache,
    DecodeTrace,
    OptimizedDecoder,
    PagedTensor,
    _optimized_sparse_program,
    optimization_policy,
)

TARGET_MESH_SHAPE = (4, 1)
TP_DEGREE = 4
TP_AXIS = 0
CCL_NUM_LINKS = 2
INDEXER_SEQUENCE_AXIS = 1
CACHE_BFP8_TILE_BYTES = 1088
STOCK_PRECISION_POLICY = "stock"
QUIETBOX2_MIXED_BFP2_POLICY = "quietbox2_mixed_bfp2"
QUIETBOX2_SELECTIVE_BFP4_POLICY = "quietbox2_selective_bfp4"
QUIETBOX2_SELECTIVE_BFP4_LAYERS = frozenset({2, 27, 29, 32, 34, 36, 41, 42})
# Every routed projection BFLOAT4_B, only a fixed pool of experts resident per layer (expert_pool.py).
QUIETBOX2_PAGED_BFP4_POLICY = "quietbox2_paged_bfp4"
QUIETBOX2_PAGED_DEFAULT_POOL = 176
PAGED_MOE_CHUNK_TOKENS = 16  # at most 16 * top_k = 96 distinct experts per MoE call
# Paged prefill routes in the resident path's 32-token chunks, reads every chunk's picks back once per
# layer, and splits a chunk only when its distinct experts would not fit in the pool.
PAGED_PREFILL_ROUTE_TOKENS = 32
GROUPED_PREFILL_BUCKET = 16  # forward_grouped pads its expert-group count to this multiple
STREAMING_PREFILL_MAX_BLOCK = 512  # largest streaming prefill block prefill_forward_block accepts


@dataclass
class PrefillBlockInputs:
    """The per-block values a middle streaming prefill block (block_start >= sliding window, 128 full rows)
    reads, as device tensors, so the block's ops take no host integers (trace-ready): the host writes them before
    each block instead of the block computing / uploading them itself. int32 ROW_MAJOR."""

    prev_pos: ttnn.Tensor  # [1, W]: block_start - W .. block_start - 1 (W = sliding window), the previous local rows
    local_pages: ttnn.Tensor  # [1, S / page]: the local cache's page-table entries of the block's pages
    raw_pages: ttnn.Tensor | None  # [1, n]: compressor-state page-table entries of the raw fill
    entry: ttnn.Tensor | None  # HCA [1]: the block's compressed entry; CSA [1]: the boundary entry (block_start / 4)
    comp_pages: ttnn.Tensor | None  # CSA [1, 1]: compressed cache page of entries block_start / 4 .. + S / 4
    index_pages: ttnn.Tensor | None  # CSA [1, 1]: the indexer cache's page for the same entries
    bnd_prev_pos: ttnn.Tensor | None  # CSA [1, 4]: block_start - 4 .. block_start - 1 (boundary repair history)
    bnd_pos: ttnn.Tensor | None  # CSA [1]: block_start (the boundary entry's RoPE position)


SUPPORTED_PRECISION_POLICIES = (
    STOCK_PRECISION_POLICY,
    QUIETBOX2_MIXED_BFP2_POLICY,
    QUIETBOX2_SELECTIVE_BFP4_POLICY,
    QUIETBOX2_PAGED_BFP4_POLICY,
)
MOE_COMPUTE_PROFILES = ("lofi", "lofi_fp32", "hifi2", "hifi2_fp32")
DENSE_COMPUTE_PROFILES = ("lofi", "hifi2", "hifi4")


@dataclass(frozen=True)
class MultichipStrategy:
    """Static, testable contract for the only supported hardware path."""

    mesh_shape: tuple[int, int] = TARGET_MESH_SHAPE
    tp_degree: int = TP_DEGREE
    tp_axis: int = TP_AXIS
    ccl_num_links: int = CCL_NUM_LINKS
    topology: str = "ring"
    residual: str = "replicated"
    kv_cache: str = "replicated_single_kv_head"
    experts: str = "top6_intermediate_tensor_parallel"


MULTICHIP_STRATEGY = MultichipStrategy()


def tensor_parallel_shapes(
    config: DeepSeekV4FlashConfig | None = None,
) -> dict[str, tuple[tuple[int, ...], tuple[int, ...]]]:
    """Return global/local shapes for every material tensor-parallel axis."""

    c = DeepSeekV4FlashConfig() if config is None else config
    if c.num_attention_heads % TP_DEGREE or c.o_groups % TP_DEGREE or c.moe_intermediate_size % TP_DEGREE:
        raise ValueError("attention heads, output groups, and MoE intermediate width must divide TP exactly")
    return {
        "q_b_weight": ((c.q_lora_rank, c.attention_width), (c.q_lora_rank, c.attention_width // TP_DEGREE)),
        "query": (
            (1, c.num_attention_heads, 1, c.head_dim),
            (1, c.num_attention_heads // TP_DEGREE, 1, c.head_dim),
        ),
        "o_a_weight": (
            (1, c.o_groups, c.grouped_attention_width, c.o_lora_rank),
            (1, c.o_groups // TP_DEGREE, c.grouped_attention_width, c.o_lora_rank),
        ),
        "o_b_weight": (
            (c.o_groups * c.o_lora_rank, c.hidden_size),
            (c.o_groups * c.o_lora_rank // TP_DEGREE, c.hidden_size),
        ),
        "expert_gate_up_weight": (
            (c.num_experts, c.hidden_size, c.moe_intermediate_size),
            (c.num_experts, c.hidden_size, c.moe_intermediate_size // TP_DEGREE),
        ),
        "expert_gate_up_packed_weight": (
            (1, c.num_experts, c.hidden_size, 2 * c.moe_intermediate_size),
            (1, c.num_experts, c.hidden_size, 2 * c.moe_intermediate_size // TP_DEGREE),
        ),
        "expert_down_weight": (
            (c.num_experts, c.moe_intermediate_size, c.hidden_size),
            (c.num_experts, c.moe_intermediate_size // TP_DEGREE, c.hidden_size),
        ),
        "shared_gate_up_weight": (
            (c.hidden_size, c.moe_intermediate_size),
            (c.hidden_size, c.moe_intermediate_size // TP_DEGREE),
        ),
        "shared_gate_up_packed_weight": (
            (c.hidden_size, 2 * c.moe_intermediate_size),
            (c.hidden_size, 2 * c.moe_intermediate_size // TP_DEGREE),
        ),
        "shared_down_weight": (
            (c.moe_intermediate_size, c.hidden_size),
            (c.moe_intermediate_size // TP_DEGREE, c.hidden_size),
        ),
    }


def multichip_cache_shapes(
    config: DeepSeekV4FlashConfig,
    layer_idx: int,
    max_context: int,
) -> dict[str, tuple[int, int, int] | None]:
    """Return ``(physical pages, logical pages, width)`` for each cache bank.

    The compressed banks remain lossless and one-to-one.  Only state whose
    lifetime is bounded by the attention/compressor window aliases physical
    pages.  One extra physical page is required because a non-aligned window
    can straddle both end pages (128 live rows therefore touch five 32-row
    pages).
    """

    if max_context <= 0:
        raise ValueError(f"max_context must be positive, got {max_context}")
    kind = config.layer_kind(layer_idx)
    page = config.page_size
    logical_pages = math.ceil(max_context / page)

    def ring_pages(live_rows: int) -> int:
        return min(logical_pages, math.ceil(live_rows / page) + 1)

    shapes: dict[str, tuple[int, int, int] | None] = {
        "local": (ring_pages(config.sliding_window), logical_pages, config.head_dim),
        "compressed": None,
        "compressor": None,
        "indexer": None,
    }
    if not kind.compress_ratio:
        return shapes

    compressed_pages = math.ceil(math.ceil(max_context / kind.compress_ratio) / page)
    shapes["compressed"] = (compressed_pages, compressed_pages, config.head_dim)
    raw_rows = 2 * kind.compress_ratio if kind.attention is AttentionKind.CSA else kind.compress_ratio
    if kind.attention is AttentionKind.CSA:
        # Ratio-4 attention and indexer compressors both have a two-wide KV
        # projection and a matching two-wide score projection.
        compressor_width = 4 * config.head_dim + 4 * config.index_head_dim
        shapes["indexer"] = (compressed_pages + 1, compressed_pages + 1, config.index_head_dim)
    else:
        compressor_width = 2 * config.head_dim
    shapes["compressor"] = (ring_pages(raw_rows), logical_pages, compressor_width)
    return shapes


def multichip_cache_bytes(
    config: DeepSeekV4FlashConfig,
    layer_idx: int,
    max_context: int,
) -> int:
    """Conservative per-device BFP8 cache bytes, including device metadata."""

    total = 0
    for shape in multichip_cache_shapes(config, layer_idx, max_context).values():
        if shape is None:
            continue
        physical_pages, logical_pages, width = shape
        elements = physical_pages * config.page_size * width
        if elements % 1024:
            raise ValueError(f"cache tensor is not tile aligned: {shape}")
        total += elements // 1024 * CACHE_BFP8_TILE_BYTES
        # Both immutable page table and physical-row LUT are resident int32
        # row-major tensors.  Round each row to the 32-byte DRAM alignment.
        total += _round_up(logical_pages * 4, 32)
        total += _round_up(logical_pages * config.page_size * 4, 32)
    return total


def multichip_stack_cache_bytes(
    config: DeepSeekV4FlashConfig | None = None,
    max_context: int | None = None,
) -> int:
    """Exact planned cache allocation for all decoder layers on one rank."""

    c = DeepSeekV4FlashConfig() if config is None else config
    context = c.max_position_embeddings if max_context is None else max_context
    return sum(multichip_cache_bytes(c, layer_idx, context) for layer_idx in range(c.num_hidden_layers))


def _cyclic_permutation(
    permutation: Sequence[int] | None,
    *,
    logical_pages: int,
    physical_pages: int,
) -> tuple[int, ...] | None:
    """Normalize identity/reverse logical page permutations onto a page ring."""

    if permutation is None:
        return None
    values = tuple(int(value) for value in permutation)
    if len(values) == physical_pages and sorted(values) == list(range(physical_pages)):
        return values
    if len(values) != logical_pages or sorted(values) != list(range(logical_pages)):
        raise ValueError(
            "cyclic cache permutation must permute either every logical page "
            f"({logical_pages}) or every physical ring page ({physical_pages})"
        )
    residues = tuple(value % physical_pages for value in values)
    cycle = residues[:physical_pages]
    if sorted(cycle) != list(range(physical_pages)) or any(
        value != cycle[index % physical_pages] for index, value in enumerate(residues)
    ):
        raise ValueError("logical cache permutation is not representable by the bounded physical page ring")
    return cycle


def _contiguous_page_runs(physical_pages: Sequence[int]) -> tuple[tuple[int, int], ...]:
    """Return half-open runs that never cross a physical page discontinuity."""

    pages = tuple(int(page) for page in physical_pages)
    if not pages:
        return ()
    runs = []
    start = 0
    for index in range(1, len(pages)):
        if pages[index] != pages[index - 1] + 1:
            runs.append((start, index))
            start = index
    runs.append((start, len(pages)))
    return tuple(runs)


def _flat_shard_mapper(mesh_device: ttnn.MeshDevice, dim: int):
    """Map one tensor dimension over all four ranks of the logical 4x1 ring."""

    return ttnn.create_mesh_mapper(
        mesh_device,
        ttnn.MeshMapperConfig(
            placements=[ttnn.PlacementShard(dim)],
            mesh_shape_override=ttnn.MeshShape([TP_DEGREE]),
        ),
    )


class _PersistentDecodeAllReduce:
    """Caller-owned exact-shape decode all-reduce buffers and semaphore."""

    def __init__(self, mesh_device: ttnn.MeshDevice) -> None:
        grid = ttnn.num_cores_to_corerangeset(
            64,
            mesh_device.compute_with_storage_grid_size(),
            row_wise=True,
        )
        self.input_memory = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(grid, [32, 64], ttnn.ShardOrientation.ROW_MAJOR),
        )
        self.output_memory = self.input_memory
        intermediate_memory = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(grid, [32, 256], ttnn.ShardOrientation.ROW_MAJOR),
        )
        self.buffer = ttnn.from_torch(
            torch.zeros((1, 1, 1, 4 * 4096), dtype=torch.bfloat16),
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=intermediate_memory,
        )
        self.semaphore = ttnn.create_global_semaphore(mesh_device, grid, 0)
        self.mesh_device = mesh_device

    def __call__(self, tensor: ttnn.Tensor) -> ttnn.Tensor:
        sharded = ttnn.to_memory_config(tensor, self.input_memory)
        reduced = ttnn.experimental.all_reduce_async(
            sharded,
            self.buffer,
            cluster_axis=TP_AXIS,
            mesh_device=self.mesh_device,
            multi_device_global_semaphore=self.semaphore,
            dtype=ttnn.bfloat16,
            memory_config=self.output_memory,
            topology=ttnn.Topology.Ring,
            num_links=CCL_NUM_LINKS,
        )
        return ttnn.sharded_to_interleaved(reduced, ttnn.DRAM_MEMORY_CONFIG)


class _FusedDecodeMatmulAllReduce:
    """Exact O-B matmul+reduce-scatter, followed by the required replication gather.

    This is the fused-CCL placement A/B.  Its persistent buffers and semaphore
    ownership follow the production Blackhole MMRS contract; the all-gather is
    retained because the current public/inter-layer residual contract is
    replicated.
    """

    def __init__(self, mesh_device: ttnn.MeshDevice, *, k: int, n: int) -> None:
        grid = (8, 6)
        worker_cores = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid[0] - 1, grid[1] - 1))}
        )
        self.program = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=grid,
            in0_block_w=min(4, max(1, (k // 32) // grid[0])),
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=1,
            per_core_N=math.ceil(n / 32 / grid[0]),
            out_block_w=max(1, math.ceil(n / 32 / grid[0]) // 2),
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=False,
            allowed_worker_cores=worker_cores,
        )

        def buffer(width: int) -> ttnn.Tensor:
            return ttnn.from_torch(
                torch.zeros((1, 1, 1, width), dtype=torch.bfloat16),
                device=mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        self.intermediate = buffer(n)
        self.output = buffer(n // TP_DEGREE)
        semaphore_grid = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(
                        mesh_device.compute_with_storage_grid_size().x - 1,
                        mesh_device.compute_with_storage_grid_size().y - 1,
                    ),
                )
            }
        )
        self.rs_semaphores = [ttnn.create_global_semaphore(mesh_device, semaphore_grid, 0) for _ in range(3)]
        self.barrier = ttnn.create_global_semaphore(mesh_device, semaphore_grid, 0)
        self.num_links = int(os.environ.get("DSV4_MMRS_LINKS", "1"))
        if self.num_links not in {1, 2}:
            raise ValueError(f"MMRS link experiment must use one or two links, got {self.num_links}")

    def __call__(
        self,
        activation: ttnn.Tensor,
        weight: ttnn.Tensor,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
        *,
        gather_output: bool = True,
    ) -> ttnn.Tensor:
        _, fractured = ttnn.experimental.matmul_reduce_scatter_async(
            activation,
            weight,
            persistent_intermediate_buffer=self.intermediate,
            persistent_output_buffer=self.output,
            dim=3,
            multi_device_global_semaphore=self.rs_semaphores,
            reduce_scatter_core_grid_offset=(0, 6),
            barrier_semaphore=self.barrier,
            num_links=self.num_links,
            memory_config_rs=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Ring,
            subdevice_id=None,
            memory_config_mm=ttnn.DRAM_MEMORY_CONFIG,
            program_config=self.program,
            compute_kernel_config=compute_kernel_config,
        )
        if not gather_output:
            return fractured
        return ttnn.all_gather(
            fractured,
            dim=3,
            num_links=CCL_NUM_LINKS,
            topology=ttnn.Topology.Ring,
            cluster_axis=TP_AXIS,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )


def _all_reduce_partial(
    tensor: ttnn.Tensor,
    persistent_decode: _PersistentDecodeAllReduce | None = None,
    *,
    ccl_dtype: str = "bf16",
) -> ttnn.Tensor:
    """Turn a row-parallel partial into the replicated layer-boundary value."""

    if ccl_dtype not in {"bf16", "bfp8"}:
        raise ValueError(f"unknown multichip CCL dtype experiment {ccl_dtype!r}")
    if persistent_decode is not None and tensor.shape[2] == 1:
        if ccl_dtype != "bf16":
            raise ValueError("persistent decode CCL and reduced CCL dtype experiments must be isolated")
        return persistent_decode(tensor)
    partial = ttnn.typecast(tensor, ttnn.bfloat8_b) if ccl_dtype == "bfp8" else tensor
    reduced = ttnn.all_reduce(
        partial,
        num_links=CCL_NUM_LINKS,
        topology=ttnn.Topology.Ring,
        cluster_axis=TP_AXIS,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return ttnn.typecast(reduced, ttnn.bfloat16) if ccl_dtype == "bfp8" else reduced


def _pool_groups(token_picks: list[list[int]], pool_size: int) -> list[tuple[int, int]]:
    """Split one chunk's tokens into contiguous [lo, hi) runs whose distinct experts fit in the pool."""

    groups, lo, seen = [], 0, set()
    for index, experts in enumerate(token_picks):
        if len(seen | set(experts)) > pool_size:
            groups.append((lo, index))
            lo, seen = index, set()
        seen |= set(experts)
    groups.append((lo, len(token_picks)))
    return groups


def _grouped_call_ranges(
    token_picks: list[list[int]], pool_size: int, max_groups: int, align: int = 1
) -> list[tuple[int, int]]:
    """Split a block's tokens into contiguous [lo, hi) runs that one forward_grouped call can take: the run's
    distinct experts fit in the pool, and its 32-row groups (ceil(n / 32) per expert picked n times) do not exceed
    the sparse matmul's group count.

    ``align``: cut only at multiples of it, so the runs take a few fixed lengths (with 32: 32 / 64 / 96 / 128
    rows) instead of a routing-dependent one, whose first use compiles programs (behind live traces in a server). A
    single aligned span that does not fit on its own is split at token granularity as before."""

    if align > 1:
        ranges, lo = [], 0
        for start in range(0, len(token_picks), align):
            end = min(start + align, len(token_picks))
            if start > lo and len(_grouped_call_ranges(token_picks[lo:end], pool_size, max_groups)) > 1:
                ranges.append((lo, start))
                lo = start
            if len(_grouped_call_ranges(token_picks[lo:end], pool_size, max_groups)) > 1:
                ranges.extend(
                    (lo + a, lo + b) for a, b in _grouped_call_ranges(token_picks[lo:end], pool_size, max_groups)
                )
                lo = end
        if lo < len(token_picks):
            ranges.append((lo, len(token_picks)))
        return ranges

    ranges, lo, counts = [], 0, {}
    for index, experts in enumerate(token_picks):
        trial = dict(counts)
        for expert in experts:
            trial[expert] = trial.get(expert, 0) + 1
        groups = sum(-(-n // 32) for n in trial.values())
        if index > lo and (len(trial) > pool_size or groups > max_groups):
            ranges.append((lo, index))
            lo, trial = index, {}
            for expert in experts:
                trial[expert] = trial.get(expert, 0) + 1
        counts = trial
    ranges.append((lo, len(token_picks)))
    return ranges


class MultichipDeepSeekSparseMoE:
    """Active top-6 MoE with every selected expert split on its inner width."""

    # the verify MoE's token -> group gather and pick -> token combine as matmuls with one-hot
    # matrices (looked up on device from identity tables by the same ids) instead of untilizing the token / expert
    # rows for ttnn.embedding. Bit-identical (HiFi4, fp32 accumulation of x * 1.0): 43-layer M=6 verify trace
    # 207.7 -> 199.7 ms per block. Default on.
    verify_onehot = True
    # routed sparse-matmul geometry (0 = the env / built-in defaults: gate_up 8x2 / subblock 2, down
    # 8x4 / subblock 4, in0_block_w 16). Default gate_up 8x4 / subblock 1 and down 8x8 / subblock 2: bit-identical
    # (43-layer M=6 verify eager logits), verify block 165.9 -> 164.9 ms. in0_block_w 32 for gate_up gives 160.7 ms
    # but is not bitwise (K blocking), left off until accepted.
    routed_gate_grid_y = 4
    routed_gate_subblock = 1
    routed_down_grid_y = 8
    routed_down_subblock = 2
    routed_gate_block = 0  # in0_block_w (K tiles per block)
    routed_down_block = 0
    # shared-expert weight dtype: None keeps weights_dtype (BFLOAT4_B); BFLOAT8_B matches the checkpoint's FP8
    # precision at about +0.13 GiB per device per weight copy.
    shared_dtype = None
    # keep a resident interleaved copy of each DRAM-sharded shared weight for multi-row (prefill / verify) matmuls;
    # False (default) makes a one-layer transient copy per call instead: bit-identical logits, ~0.14 GiB per
    # device more free DRAM at BFP4
    keep_interleaved_shared = False

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        expert_w1: torch.Tensor,
        expert_w2: torch.Tensor,
        expert_w3: torch.Tensor,
        shared_w1: torch.Tensor,
        shared_w2: torch.Tensor,
        shared_w3: torch.Tensor,
        *,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        limit: float,
        weights_dtype: ttnn.DataType = ttnn.bfloat4_b,
        routed_gate_up_dtype: ttnn.DataType | None = None,
        compute_profile: str = "lofi",
        expert_pool_size: int | None = None,
        keep_torch_experts: bool = False,
        expert_store: ExpertHostStore | None = None,
    ) -> None:
        if intermediate_size % TP_DEGREE:
            raise ValueError(f"expert width {intermediate_size} must be divisible by TP={TP_DEGREE}")
        expected_gate = (num_experts, intermediate_size, hidden_size)
        expected_down = (num_experts, hidden_size, intermediate_size)
        if expert_store is not None:
            if expert_pool_size is None or expert_store.num_experts != num_experts:
                raise ValueError("a prebuilt expert store needs paging and one entry per routed expert")
        elif tuple(expert_w1.shape) != expected_gate or tuple(expert_w3.shape) != expected_gate:
            raise ValueError(f"routed gate/up shapes must both be {expected_gate}")
        elif tuple(expert_w2.shape) != expected_down:
            raise ValueError(f"routed down shape must be {expected_down}")

        self.mesh_device = mesh_device
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.local_intermediate_size = intermediate_size // TP_DEGREE
        self.num_experts = num_experts
        self.top_k = top_k
        self.limit = limit
        self.weights_dtype = weights_dtype
        self.routed_gate_up_dtype = weights_dtype if routed_gate_up_dtype is None else routed_gate_up_dtype
        if compute_profile not in MOE_COMPUTE_PROFILES:
            raise ValueError(f"unknown multichip MoE compute profile {compute_profile!r}")
        self.compute_profile = compute_profile
        if self.weights_dtype != ttnn.bfloat4_b:
            raise ValueError(f"routed down experts require BFLOAT4_B, got {self.weights_dtype}")
        if self.routed_gate_up_dtype not in {getattr(ttnn, "bfloat2_b", None), ttnn.bfloat4_b} - {None}:
            raise ValueError(
                "routed gate/up experts require BFLOAT2_B or BFLOAT4_B, " f"got {self.routed_gate_up_dtype}"
            )
        self.projection_experiment = os.environ.get("DSV4_MULTICHIP_MOE_PROJECTION", "packed")
        if self.projection_experiment not in {"packed", "separate"}:
            raise ValueError(f"unknown multichip MoE projection experiment {self.projection_experiment!r}")
        self._onehot_compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity={
                "lofi": ttnn.MathFidelity.LoFi,
                "lofi_fp32": ttnn.MathFidelity.LoFi,
                "hifi2": ttnn.MathFidelity.HiFi2,
                "hifi2_fp32": ttnn.MathFidelity.HiFi2,
            }[compute_profile],
            math_approx_mode=False,
            fp32_dest_acc_en=compute_profile.endswith("_fp32"),
            packer_l1_acc=True,
        )

        def upload(weight: torch.Tensor, shard_dim: int, dtype: ttnn.DataType) -> ttnn.Tensor:
            return ttnn.from_torch(
                weight.contiguous(),
                device=mesh_device,
                mesh_mapper=_flat_shard_mapper(mesh_device, shard_dim),
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        self.pool: ExpertPool | None = None
        if expert_pool_size is not None:
            # Host-paged routed experts: only a fixed pool is resident (expert_pool.py).
            if self.projection_experiment != "packed" or self.routed_gate_up_dtype != self.weights_dtype:
                raise ValueError("expert paging requires the packed projection with BFLOAT4_B routed weights")
            store = expert_store or ExpertHostStore.from_weights(
                mesh_device,
                expert_w1,
                expert_w2,
                expert_w3,
                tp_degree=TP_DEGREE,
                shard_mapper=_flat_shard_mapper,
                dtype=self.weights_dtype,
                keep_torch=keep_torch_experts,
            )
            self.pool = ExpertPool(
                mesh_device,
                store,
                expert_pool_size,
                gate_up_shape=(hidden_size, 2 * self.local_intermediate_size),
                down_shape=(self.local_intermediate_size, hidden_size),
                shard_mapper=_flat_shard_mapper,
            )
            self.routed_gate_up = self.pool.gate_up
            self.routed_gate = None
            self.routed_up = None
            self.routed_down = self.pool.down
        # sparse_matmul consumes [1,E,K,N].  Arrange gate/up as four adjacent
        # rank blocks before sharding so one fused projection gives each rank
        # [gate_r, up_r], rather than assigning whole gate/up halves to ranks.
        if self.pool is None:
            routed_gate = expert_w1.transpose(-2, -1)
            routed_up = expert_w3.transpose(-2, -1)
            routed_gate_chunks = routed_gate.chunk(TP_DEGREE, dim=-1)
            routed_up_chunks = routed_up.chunk(TP_DEGREE, dim=-1)
        if self.pool is None and self.projection_experiment == "packed":
            routed_gate_up = torch.cat(
                [
                    torch.cat((gate_chunk, up_chunk), dim=-1)
                    for gate_chunk, up_chunk in zip(routed_gate_chunks, routed_up_chunks, strict=True)
                ],
                dim=-1,
            )
            self.routed_gate_up = upload(routed_gate_up.unsqueeze(0), -1, self.routed_gate_up_dtype)
            self.routed_gate = None
            self.routed_up = None
        elif self.pool is None:
            self.routed_gate_up = None
            self.routed_gate = upload(routed_gate.unsqueeze(0), -1, self.routed_gate_up_dtype)
            self.routed_up = upload(routed_up.unsqueeze(0), -1, self.routed_gate_up_dtype)
        if self.pool is None:
            self.routed_down = upload(expert_w2.transpose(-2, -1).unsqueeze(0), -2, self.weights_dtype)
        shared_dtype = self.shared_dtype or self.weights_dtype
        if shared_dtype not in {ttnn.bfloat4_b, ttnn.bfloat8_b}:
            raise ValueError(f"shared experts require BFLOAT4_B or BFLOAT8_B, got {shared_dtype}")
        shared_gate = shared_w1.transpose(-2, -1)
        shared_up = shared_w3.transpose(-2, -1)
        shared_gate_chunks = shared_gate.chunk(TP_DEGREE, dim=-1)
        shared_up_chunks = shared_up.chunk(TP_DEGREE, dim=-1)
        if self.projection_experiment == "packed":
            shared_gate_up = torch.cat(
                [
                    torch.cat((gate_chunk, up_chunk), dim=-1)
                    for gate_chunk, up_chunk in zip(shared_gate_chunks, shared_up_chunks, strict=True)
                ],
                dim=-1,
            )
            self.shared_gate_up = upload(shared_gate_up, -1, shared_dtype)
            self.shared_gate = None
            self.shared_up = None
        else:
            self.shared_gate_up = None
            self.shared_gate = upload(shared_gate, -1, shared_dtype)
            self.shared_up = upload(shared_up, -1, shared_dtype)
        self.shared_down = upload(shared_w2.transpose(-2, -1), -2, shared_dtype)

        # routed sparse-matmul core grids: the class attributes (geometry sweep) win over the DSV4_ experiment
        # variables, which the qualified full-model build rejects
        gate_grid = (8, int(self.routed_gate_grid_y or os.environ.get("DSV4_ROUTED_GATE_GRID_Y", "2")))
        gate_block = int(self.routed_gate_block or os.environ.get("DSV4_ROUTED_GATE_BLOCK", "16"))
        gate_subblock_default = "2" if self.projection_experiment == "packed" else "1"
        self.gate_up_program = _optimized_sparse_program(
            k=hidden_size,
            n=(2 if self.projection_experiment == "packed" else 1) * self.local_intermediate_size,
            grid=gate_grid,
            in0_block_w=gate_block,
            out_subblock_w=int(
                self.routed_gate_subblock or os.environ.get("DSV4_ROUTED_GATE_SUBBLOCK", gate_subblock_default)
            ),
        )
        down_grid = (8, int(self.routed_down_grid_y or os.environ.get("DSV4_ROUTED_DOWN_GRID_Y", "4")))
        self.down_program = _optimized_sparse_program(
            k=self.local_intermediate_size,
            n=hidden_size,
            grid=down_grid,
            in0_block_w=int(self.routed_down_block or os.environ.get("DSV4_ROUTED_DOWN_BLOCK", "16")),
            out_subblock_w=int(self.routed_down_subblock or os.environ.get("DSV4_ROUTED_DOWN_SUBBLOCK", "4")),
        )
        self.sparsity = (
            self.pool.sparsity
            if self.pool is not None
            else ttnn.ones(
                [1, 1, 1, num_experts],
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        )
        # Eager callers (prefill, untraced decode) let forward() read the router picks back and fill the
        # pool itself; the split decode trace clears this and calls ensure_resident() between replays.
        self.pool_eager_ensure = True
        # paged prefill MoE grouped by expert (forward_grouped); False restores the per-token loop.
        self.prefill_grouped = True
        self._shared_dram_programs: dict[str, ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig] = {}
        self._shared_input_memories: dict[str, ttnn.MemoryConfig] = {}
        self._shared_interleaved_weights: dict[str, ttnn.Tensor] = {}
        self.use_shared_dram_sharding = False

    def configure_shared_dram_sharding(self) -> None:
        """Screen exact eight-bank shared-expert decode projections.

        The optimized baseline's 64-output-core helper cannot represent the
        TP-local 1024-wide gate/up result (its ``per_core_N`` becomes zero).
        This TP-specific form keeps the weight split exact across Blackhole's
        eight DRAM banks while separately choosing the legal output geometry.
        It is enabled only by the model-local topology A/B until measured.
        """

        dram_size = self.mesh_device.dram_grid_size()
        dram_cores = dram_size.x * dram_size.y
        if dram_cores != 8:
            raise ValueError(f"shared DRAM-sharded A/B requires eight DRAM banks, got {dram_cores}")
        dram_grid = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(dram_size.x - 1, dram_size.y - 1),
                )
            }
        )
        # The optimized multichip pass selected four wider K shards.  It keeps
        # the exact eight-bank weight distribution but halves worker fanout
        # and wins decode without changing the prefill path.
        shared_down_input_cores = int(os.environ.get("DSV4_SHARED_DOWN_INPUT_CORES", "4"))
        shared_down_block = int(os.environ.get("DSV4_SHARED_DOWN_BLOCK", "4"))
        gate_attributes = (
            [("shared_gate_up", dram_cores, 16, 1)]
            if self.projection_experiment == "packed"
            else [("shared_gate", dram_cores, 16, 1), ("shared_up", dram_cores, 16, 1)]
        )
        for attribute, input_cores, in0_block_w, per_core_n in (
            *gate_attributes,
            ("shared_down", shared_down_input_cores, shared_down_block, 2),
        ):
            weight = getattr(self, attribute)
            k, n = tuple(weight.shape)[-2:]
            if k % dram_cores or n % (32 * dram_cores):
                raise ValueError(f"{attribute} shape {(k, n)} cannot shard exactly over eight DRAM banks")
            if k % (32 * input_cores) or (k // (32 * input_cores)) % in0_block_w:
                raise ValueError(
                    f"{attribute} K tiles {k // 32} cannot use {input_cores} input cores and block {in0_block_w}"
                )
            weight_memory = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.DRAM,
                ttnn.ShardSpec(
                    dram_grid,
                    (k, n // dram_cores),
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            )
            sharded = ttnn.to_memory_config(weight, weight_memory)
            if self.keep_interleaved_shared:
                self._shared_interleaved_weights[attribute] = weight
            else:
                ttnn.deallocate(weight)
            setattr(self, attribute, sharded)
            self._shared_dram_programs[attribute] = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                in0_block_w=in0_block_w,
                per_core_M=1,
                per_core_N=per_core_n,
                fused_activation=None,
            )
            worker_grid = ttnn.num_cores_to_corerangeset(
                input_cores,
                self.mesh_device.compute_with_storage_grid_size(),
                row_wise=True,
            )
            self._shared_input_memories[attribute] = ttnn.create_sharded_memory_config_(
                shape=(32, k // input_cores),
                core_grid=worker_grid,
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )

    def _shared_linear(self, value: ttnn.Tensor, attribute: str) -> ttnn.Tensor:
        if not self.use_shared_dram_sharding or attribute not in self._shared_dram_programs:
            weight = self._shared_interleaved_weights.get(attribute)
            transient = weight is None and attribute in self._shared_dram_programs
            if transient:  # one-layer interleaved copy of the sharded weight (same values) instead of a resident one
                weight = ttnn.to_memory_config(getattr(self, attribute), ttnn.DRAM_MEMORY_CONFIG)
            elif weight is None:
                weight = getattr(self, attribute)
            output = ttnn.linear(
                value,
                weight,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                compute_kernel_config=self.compute_kernel_config,
            )
            if transient:
                ttnn.deallocate(weight)
            return output
        shape = tuple(value.shape)
        rows = math.prod(shape[:-1])
        k, n = tuple(getattr(self, attribute).shape)[-2:]
        flat = value if shape[:-2] == (1, 1) else ttnn.reshape(value, [1, 1, rows, k])
        sharded = ttnn.to_memory_config(flat, self._shared_input_memories[attribute])
        output = ttnn.linear(
            sharded,
            getattr(self, attribute),
            program_config=self._shared_dram_programs[attribute],
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute_kernel_config,
        )
        output = ttnn.sharded_to_interleaved(output, ttnn.L1_MEMORY_CONFIG)
        result_shape = (*shape[:-1], n)
        return output if tuple(output.shape) == result_shape else ttnn.reshape(output, result_shape)

    def _swiglu(self, gate: ttnn.Tensor, up: ttnn.Tensor) -> ttnn.Tensor:
        gate = ttnn.clamp(gate, min=None, max=self.limit, output_tensor=gate)
        up = ttnn.clamp(up, min=-self.limit, max=self.limit, output_tensor=up)
        return ttnn.mul(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])

    def _sparse_projection(
        self,
        token: ttnn.Tensor,
        weight: ttnn.Tensor,
        indices: ttnn.Tensor,
        program_config: ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig,
        *,
        input_a_sparse: bool = False,
    ) -> ttnn.Tensor:
        return ttnn.sparse_matmul(
            token,
            weight,
            sparsity=self.sparsity,
            indices=indices,
            is_input_a_sparse=input_a_sparse,
            is_input_b_sparse=True,
            program_config=program_config,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            output_tile=ttnn.Tile([32, 32]),
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute_kernel_config,
        )

    def ensure_resident(self, selected_experts: ttnn.Tensor) -> int:
        """Read router picks back (blocking) and load the missing experts into the pool."""

        picks = ttnn.to_torch((ttnn.get_device_tensors(selected_experts) or (selected_experts,))[0])
        return self.pool.ensure(picks.flatten().to(torch.int64).tolist())

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        selected_experts: ttnn.Tensor,
        routing_weights: ttnn.Tensor,
    ) -> ttnn.Tensor:
        seq = hidden_states.shape[2]
        if tuple(hidden_states.shape) != (1, 1, seq, self.hidden_size) or not 1 <= seq <= 32:
            raise ValueError(f"MoE input must be [1,1,S,{self.hidden_size}] with 1 <= S <= 32")
        expected_routing = (1, 1, seq, self.top_k)
        if tuple(selected_experts.shape) != expected_routing or tuple(routing_weights.shape) != expected_routing:
            raise ValueError(f"selected experts and weights must both be {expected_routing}")

        if self.pool is not None and self.pool_eager_ensure:
            self.ensure_resident(selected_experts)
        # Slot indices for every token at once, issued after this call's pool writes.
        slot_rows = self.pool.lookup(selected_experts) if self.pool is not None else None
        hidden_l1 = ttnn.to_memory_config(hidden_states, ttnn.L1_MEMORY_CONFIG)
        routed_tokens = []
        for token_idx in range(seq):
            token = ttnn.slice(
                hidden_l1,
                [0, 0, token_idx, 0],
                [1, 1, token_idx + 1, self.hidden_size],
            )
            if slot_rows is not None:
                indices = (
                    slot_rows
                    if seq == 1
                    else ttnn.slice(slot_rows, [0, 0, token_idx, 0], [1, 1, token_idx + 1, self.top_k])
                )
            else:
                indices = ttnn.slice(
                    selected_experts,
                    [0, 0, token_idx, 0],
                    [1, 1, token_idx + 1, self.top_k],
                )
                indices = ttnn.to_layout(
                    ttnn.reshape(indices, [1, 1, 1, self.top_k]),
                    ttnn.ROW_MAJOR_LAYOUT,
                )
            if self.projection_experiment == "packed":
                gate_up = self._sparse_projection(token, self.routed_gate_up, indices, self.gate_up_program)
                gate_up = ttnn.reshape(gate_up, (self.top_k, 1, 2 * self.local_intermediate_size))
                gate = ttnn.slice(gate_up, [0, 0, 0], [self.top_k, 1, self.local_intermediate_size])
                up = ttnn.slice(
                    gate_up,
                    [0, 0, self.local_intermediate_size],
                    [self.top_k, 1, 2 * self.local_intermediate_size],
                )
            else:
                gate = ttnn.reshape(
                    self._sparse_projection(token, self.routed_gate, indices, self.gate_up_program),
                    (self.top_k, 1, self.local_intermediate_size),
                )
                up = ttnn.reshape(
                    self._sparse_projection(token, self.routed_up, indices, self.gate_up_program),
                    (self.top_k, 1, self.local_intermediate_size),
                )
            activated = ttnn.reshape(
                self._swiglu(gate, up),
                (1, self.top_k, 1, self.local_intermediate_size),
            )
            routed = self._sparse_projection(
                activated,
                self.routed_down,
                indices,
                self.down_program,
                input_a_sparse=True,
            )
            routed = ttnn.reshape(routed, (self.top_k, 1, self.hidden_size))
            token_weights = ttnn.slice(
                routing_weights,
                [0, 0, token_idx, 0],
                [1, 1, token_idx + 1, self.top_k],
            )
            routed_tokens.append(
                ttnn.matmul(
                    ttnn.reshape(token_weights, (1, 1, 1, self.top_k)),
                    ttnn.reshape(routed, (1, 1, self.top_k, self.hidden_size)),
                    dtype=ttnn.bfloat16,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                    compute_kernel_config=self.compute_kernel_config,
                )
            )
        routed = routed_tokens[0] if len(routed_tokens) == 1 else ttnn.concat(routed_tokens, dim=2)
        # The decoder concatenates all 32-token chunk partials before issuing
        # the single sublayer collective.  Keeping the reduction out of this
        # method avoids 256 tiny CCLs at the advertised S=8192 prefill length.
        return ttnn.add(routed, self._shared_forward(hidden_l1, seq))

    def _shared_forward(self, hidden_l1: ttnn.Tensor, seq: int) -> ttnn.Tensor:
        if self.projection_experiment == "packed":
            shared_gate_up = self._shared_linear(hidden_l1, "shared_gate_up")
            shared_gate = ttnn.slice(
                shared_gate_up,
                [0, 0, 0, 0],
                [1, 1, seq, self.local_intermediate_size],
            )
            shared_up = ttnn.slice(
                shared_gate_up,
                [0, 0, 0, self.local_intermediate_size],
                [1, 1, seq, 2 * self.local_intermediate_size],
            )
        else:
            shared_gate = self._shared_linear(hidden_l1, "shared_gate")
            shared_up = self._shared_linear(hidden_l1, "shared_up")
        shared = self._swiglu(shared_gate, shared_up)
        return self._shared_linear(shared, "shared_down")

    def forward_grouped(
        self,
        hidden_states: ttnn.Tensor,
        picks: torch.Tensor,
        routing_weights: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """Paged prefill MoE grouped by expert instead of by token.

        ``picks`` [S, top_k] are the router's expert ids, already resident in the pool (the caller ran ensure()).
        The host turns them into a slot list and two row maps; on device the S tokens are gathered into one
        32-row group per active slot, each projection is one indexed sparse_matmul over all groups, and the
        rows are gathered back per (token, pick) for the same [1, top_k] x [top_k, H] weighted sum as forward().
        A token's dot products, and their K order, are the same as in forward(); only the op count changes
        (~15 dispatches per call instead of ~14 per token).
        """

        seq = hidden_states.shape[2]
        top_k = self.top_k
        if tuple(hidden_states.shape) != (1, 1, seq, self.hidden_size) or not 1 <= seq <= STREAMING_PREFILL_MAX_BLOCK:
            raise ValueError(
                f"MoE input must be [1,1,S,{self.hidden_size}] with 1 <= S <= {STREAMING_PREFILL_MAX_BLOCK}"
            )
        slots = [[self.pool.slot_of[int(e)] for e in row] for row in picks.tolist()]
        counts: dict[int, int] = {}
        for row in slots:
            for slot in row:
                counts[slot] = counts.get(slot, 0) + 1
        active = sorted(counts)
        if self.pool.null_slot in active:
            raise RuntimeError("forward_grouped needs every picked expert resident")
        if len(active) > self.pool.pool_size:
            raise RuntimeError("an MoE call may not touch more experts than the pool holds")
        # One 32-row group per active slot, or ceil(n / 32) consecutive groups for a slot picked by n > 32 tokens
        # (calls of more than 32 tokens); with S <= 32 this is exactly one group per slot, as before.
        group_slots: list[int] = []
        first_group: dict[int, int] = {}
        for slot in active:
            first_group[slot] = len(group_slots)
            group_slots += [slot] * -(-counts[slot] // 32)
        fill = dict.fromkeys(active, 0)
        gather = torch.zeros(len(group_slots) * 32, dtype=torch.int32)  # padding rows read token 0, never combined
        combine = torch.empty(seq * top_k, dtype=torch.int32)
        for token, row in enumerate(slots):
            for pick, slot in enumerate(row):
                k = fill[slot]
                index = (first_group[slot] + k // 32) * 32 + k % 32
                gather[index] = token
                combine[token * top_k + pick] = index
                fill[slot] = k + 1
        # Pad the group count to a multiple of 16 (capped at the pool's slot count while it fits) so a prefill
        # compiles at most a few dozen shapes instead of one per distinct expert count; padding groups use the null
        # slot and their rows are never combined.
        groups = -(-len(group_slots) // GROUPED_PREFILL_BUCKET) * GROUPED_PREFILL_BUCKET
        if len(group_slots) <= self.pool.pool_size + 1:
            groups = min(groups, self.pool.pool_size + 1)
        active = group_slots + [self.pool.null_slot] * (groups - len(group_slots))
        gather = torch.cat((gather, torch.zeros((groups - len(group_slots)) * 32, dtype=torch.int32)))
        # a merged prefill call (more than 32 tokens) is padded to whole 32-row tiles, so these calls compile a
        # few shapes (64 / 96 / 128 ... rows) instead of one per token count; padding tokens are zero rows whose picks
        # point at row 0, and the caller keeps only the first rows of the output.
        pad = (-seq) % 32 if seq > 32 else 0
        if pad:
            hidden_states = ttnn.pad(hidden_states, [(0, 0), (0, 0), (0, pad), (0, 0)], 0.0)
            routing_weights = ttnn.pad(routing_weights, [(0, 0), (0, 0), (0, pad), (0, 0)], 0.0)
            combine = torch.cat((combine, torch.zeros(pad * top_k, dtype=torch.int32)))
            seq += pad

        def upload(values: torch.Tensor, dtype: ttnn.DataType) -> ttnn.Tensor:
            return ttnn.from_torch(
                values,
                dtype=dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        slot_ids = upload(torch.tensor(active, dtype=torch.int32).reshape(1, 1, 1, groups), ttnn.uint16)
        gather_ids = upload(gather.reshape(1, -1), ttnn.uint32)
        combine_ids = upload(combine.reshape(1, -1), ttnn.uint32)

        table = ttnn.to_layout(ttnn.reshape(hidden_states, [seq, self.hidden_size]), ttnn.ROW_MAJOR_LAYOUT)
        grouped = ttnn.reshape(
            ttnn.embedding(gather_ids, table, layout=ttnn.TILE_LAYOUT), [1, groups, 32, self.hidden_size]
        )

        def project(value, weight, program_config):
            return ttnn.sparse_matmul(
                value,
                weight,
                sparsity=self.sparsity,
                indices=slot_ids,
                is_input_a_sparse=True,
                is_input_b_sparse=True,
                program_config=program_config,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                output_tile=ttnn.Tile([32, 32]),
                dtype=ttnn.bfloat16,
                compute_kernel_config=self.compute_kernel_config,
            )

        inter = self.local_intermediate_size
        gate_up = project(grouped, self.routed_gate_up, self.gate_up_program)
        gate = ttnn.slice(gate_up, [0, 0, 0, 0], [1, groups, 32, inter])
        up = ttnn.slice(gate_up, [0, 0, 0, inter], [1, groups, 32, 2 * inter])
        routed = project(self._swiglu(gate, up), self.routed_down, self.down_program)
        rows = ttnn.to_layout(ttnn.reshape(routed, [groups * 32, self.hidden_size]), ttnn.ROW_MAJOR_LAYOUT)
        per_pick = ttnn.reshape(
            ttnn.embedding(combine_ids, rows, layout=ttnn.TILE_LAYOUT), [1, seq, top_k, self.hidden_size]
        )
        combined = ttnn.matmul(
            ttnn.reshape(routing_weights, (1, seq, 1, top_k)),
            per_pick,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        hidden_l1 = ttnn.to_memory_config(hidden_states, ttnn.L1_MEMORY_CONFIG)
        # DRAM like forward()'s concatenated routed rows: the sublayer all-reduce then reuses the same cached
        # CCL program (and its L1_SMALL global semaphore) as the per-token path.
        return ttnn.add(
            ttnn.reshape(combined, (1, 1, seq, self.hidden_size)),
            self._shared_forward(hidden_l1, seq),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def make_verify_buffers(self, seq: int) -> dict:
        """Persistent index buffers for a traced M-token verify MoE (one set shared by every layer; the host
        rewrites them between trace segments). Fixed group count: every pick in its own group at worst (M * top_k),
        capped at the pool's slot count (+ the null slot); unused groups use the null slot."""

        groups = min(seq * self.top_k, self.pool.pool_size + 1)

        def zeros(shape, dtype):
            return ttnn.from_torch(
                torch.zeros(shape, dtype=torch.int32),
                dtype=dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        def eye(rows, width):
            return ttnn.from_torch(
                torch.eye(rows, width),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        return {
            "seq": seq,
            "groups": groups,
            "slot_ids": zeros((1, 1, 1, groups), ttnn.uint16),
            "gather_ids": zeros((1, groups * 32), ttnn.uint32),
            "combine_ids": zeros((1, seq * self.top_k), ttnn.uint32),
            # verify_onehot: identity tables; ttnn.embedding of the gather / combine ids in them gives the one-hot
            # matrices on device (the ids stay the only per-layer upload)
            "gather_eye": eye(seq, 32),
            "combine_eye": eye(groups * 32, groups * 32),
        }

    def fill_verify_buffers(self, picks: torch.Tensor, buffers: dict) -> None:
        """Host side between trace segments: load the picked experts, then write this layer's slot / gather /
        combine maps (same construction as forward_grouped) into the persistent buffers."""

        seq, top_k, groups = int(picks.shape[0]), self.top_k, buffers["groups"]
        self.pool.ensure(picks.flatten().to(torch.int64).tolist())
        slots = [[self.pool.slot_of[int(e)] for e in row] for row in picks.tolist()]
        active = sorted({slot for row in slots for slot in row})
        if self.pool.null_slot in active or len(active) > groups:
            raise RuntimeError("verify MoE needs every picked expert resident and at most `groups` of them")
        group_of = {slot: index for index, slot in enumerate(active)}
        fill = [0] * len(active)
        gather = torch.zeros(groups * 32, dtype=torch.int32)
        combine = torch.empty(seq * top_k, dtype=torch.int32)
        for token, row in enumerate(slots):
            for pick, slot in enumerate(row):
                group = group_of[slot]
                gather[group * 32 + fill[group]] = token
                combine[token * top_k + pick] = group * 32 + fill[group]
                fill[group] += 1
        active = active + [self.pool.null_slot] * (groups - len(active))

        def host(values: torch.Tensor, dtype: ttnn.DataType) -> ttnn.Tensor:
            return ttnn.from_torch(
                values,
                dtype=dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )

        ttnn.copy_host_to_device_tensor(
            host(torch.tensor(active, dtype=torch.int32).reshape(1, 1, 1, groups), ttnn.uint16), buffers["slot_ids"]
        )
        ttnn.copy_host_to_device_tensor(host(gather.reshape(1, -1), ttnn.uint32), buffers["gather_ids"])
        ttnn.copy_host_to_device_tensor(host(combine.reshape(1, -1), ttnn.uint32), buffers["combine_ids"])

    def forward_grouped_static(
        self,
        hidden_states: ttnn.Tensor,
        routing_weights: ttnn.Tensor,
        buffers: dict,
    ) -> ttnn.Tensor:
        """forward_grouped with its index maps taken from persistent buffers (fill_verify_buffers), so the device
        work has fixed shapes and no host uploads: traceable. Same ops and values as forward_grouped."""

        seq, top_k, groups = hidden_states.shape[2], self.top_k, buffers["groups"]
        if seq != buffers["seq"]:
            raise ValueError(f"verify buffers are for {buffers['seq']} tokens, got {seq}")
        if self.verify_onehot:
            # [groups * 32, 32] one-hot (columns >= seq zero) @ the tokens padded with zero rows to 32
            gather = ttnn.reshape(
                ttnn.embedding(buffers["gather_ids"], buffers["gather_eye"], layout=ttnn.TILE_LAYOUT),
                [1, 1, groups * 32, 32],
            )
            tokens = ttnn.pad(hidden_states, [(0, 0), (0, 0), (0, 32 - seq), (0, 0)], 0.0)
            grouped = ttnn.reshape(
                ttnn.matmul(gather, tokens, dtype=ttnn.bfloat16, compute_kernel_config=self._onehot_compute_config),
                [1, groups, 32, self.hidden_size],
            )
        else:
            table = ttnn.to_layout(ttnn.reshape(hidden_states, [seq, self.hidden_size]), ttnn.ROW_MAJOR_LAYOUT)
            grouped = ttnn.reshape(
                ttnn.embedding(buffers["gather_ids"], table, layout=ttnn.TILE_LAYOUT), [1, groups, 32, self.hidden_size]
            )

        def project(value, weight, program_config):
            return ttnn.sparse_matmul(
                value,
                weight,
                sparsity=self.sparsity,
                indices=buffers["slot_ids"],
                is_input_a_sparse=True,
                is_input_b_sparse=True,
                program_config=program_config,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                output_tile=ttnn.Tile([32, 32]),
                dtype=ttnn.bfloat16,
                compute_kernel_config=self.compute_kernel_config,
            )

        inter = self.local_intermediate_size
        gate_up = project(grouped, self.routed_gate_up, self.gate_up_program)
        gate = ttnn.slice(gate_up, [0, 0, 0, 0], [1, groups, 32, inter])
        up = ttnn.slice(gate_up, [0, 0, 0, inter], [1, groups, 32, 2 * inter])
        routed = project(self._swiglu(gate, up), self.routed_down, self.down_program)
        if self.verify_onehot:
            per_pick = ttnn.reshape(
                ttnn.matmul(
                    ttnn.reshape(
                        ttnn.embedding(buffers["combine_ids"], buffers["combine_eye"], layout=ttnn.TILE_LAYOUT),
                        [1, 1, seq * top_k, groups * 32],
                    ),
                    ttnn.reshape(routed, [1, 1, groups * 32, self.hidden_size]),
                    dtype=ttnn.bfloat16,
                    compute_kernel_config=self._onehot_compute_config,
                ),
                [1, seq, top_k, self.hidden_size],
            )
        else:
            rows = ttnn.to_layout(ttnn.reshape(routed, [groups * 32, self.hidden_size]), ttnn.ROW_MAJOR_LAYOUT)
            per_pick = ttnn.reshape(
                ttnn.embedding(buffers["combine_ids"], rows, layout=ttnn.TILE_LAYOUT), [1, seq, top_k, self.hidden_size]
            )
        combined = ttnn.matmul(
            ttnn.reshape(routing_weights, (1, seq, 1, top_k)),
            per_pick,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        hidden_l1 = ttnn.to_memory_config(hidden_states, ttnn.L1_MEMORY_CONFIG)
        return ttnn.add(
            ttnn.reshape(combined, (1, 1, seq, self.hidden_size)),
            self._shared_forward(hidden_l1, seq),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )


@dataclass
class DecodeRouteState:
    """Live tensors between the two halves of a split decode layer (kept alive by trace segments)."""

    residual: ttnn.Tensor
    post: ttnn.Tensor
    comb: ttnn.Tensor
    chunk: ttnn.Tensor
    selected: ttnn.Tensor
    weights: ttnn.Tensor


@dataclass
class VerifyRouteState:
    """Live tensors between the two halves of a split verify layer."""

    residual: ttnn.Tensor
    post: ttnn.Tensor
    comb: ttnn.Tensor
    ffn_input: ttnn.Tensor
    selected: ttnn.Tensor
    weights: ttnn.Tensor


INDEXER_TOPK_WIDE = 2048


class MultichipDecoder(OptimizedDecoder):
    """Optimized decoder with a fixed four-device tensor-parallel graph."""

    verify_sdpa = True  # non-CSA verify attention as one shared-bank SDPA (False: per-row gathered banks)
    verify_row_matmul = True  # verify per-row cache writes pick their row by one-hot matmul (exact)
    # the indexer top-k asks topk_large_indices for INDEXER_TOPK_WIDE and keeps the first index_topk. The op's
    # output is sorted by score, so that prefix is a top-k too, but a K' 2048 request runs its fused body (76 vs 142
    # us for 32832 scores, k 512 falls back to the classic one); ties at the boundary may resolve differently.
    indexer_topk_wide = False
    # the decode indexer top-k reads its search length (the valid entry count) from a device
    # tensor (topk_large_indices valid_length_tensor, trace-safe), so it scans only the filled prefix of the bank
    # instead of all rows, and the lt / where -inf mask over the whole row is dropped (columns past the length are
    # never read). Bank rows are in ordinal order (decode_index_ordinals is an arange), so the valid set is a prefix.
    # Not bitwise: just past top-k entries the bounded search orders tied scores differently (same set, same values,
    # so attention sums in another order). Default on (acceptance bar): determinism x30, 16-prompt decode
    # dKL +0.0015 (CI -0.004..+0.007), verify equal, stop rate 8/8; decode -5.6..-6.3 ms/step, verify -2.4 ms/block.
    indexer_topk_dynamic = True
    # With indexer_topk_dynamic, the decode keeps the lt / where -inf mask before the bounded top-k (as verify does).
    # The maskless decode measured not bitwise at short context on the full model (logits after 45 steps differ at
    # positions 2600 / 4096, equal at 16k / 32k) while verify (mask kept) is bitwise.
    indexer_topk_dynamic_mask = False
    verify_sharded_o = True  # verify O-A on the decode DRAM-sharded program (M <= 32)
    # mHC A2: decode / verify _hc_pre epilogue (split-K sum + bias, sigmoids, softmax, Sinkhorn) as one
    # program (OptimizedDecoder._hc_pre_epilogue_fused). Not bitwise (fp32 intermediates); off until accepted.
    hc_pre_fused_epilogue = False
    # Sinkhorn iterations of the fused epilogue: the official 20 (the op-by-op policy runs 11 to save time; in the
    # fused program the extra iterations cost ~2 ms / token and bring decode / verify closer to the official logits).
    hc_pre_fused_sinkhorn_iters = 20
    # mHC A3: with A2, the mHC map reads the residual directly (OptimizedDecoder._hc_map) and the RMS scale
    # moves into the epilogue, instead of flatten / typecast / RMSNorm / split-K re-layout + matmul. Not bitwise.
    hc_pre_map_fused = False
    # a compressor history's decode / verify pooling (mask, softmax, weighted sum, RMSNorm, RoPE,
    # Hadamard) as one program (OptimizedDecoder._comp_pool_fused). Not bitwise (fp32 intermediates); off until accepted.
    compressor_pool_fused = False
    # decode / verify mHC output map (comb^T @ residual + post * output) as one matmul per tile
    # (OptimizedDecoder._hc_post_fused). Needs A2's post / comb shapes. Not bitwise (one rounding instead of two).
    # Default on (2026-10-31, after the hc_post reader rewrite: M = 6 verify 110.7 -> 28.7 us per call, verify block
    # 168.8 -> 166.2 ms; part of the HCP set that passed the acceptance bar).
    hc_post_fused = True
    # prefill blocks (> 32 rows) run the mHC input map through A3 + the A2 epilogue (Sinkhorn iterations
    # prefill_hc_sinkhorn_iters, default the prefill policy's) and the output map through the hc_post kernel, instead
    # of the op-by-op path whose [1, S, n, H] reshapes and broadcast addcmul dominate it. Not bitwise; off until accepted.
    # Default on (acceptance bar, with prefill_hc_post_kernel + hc_post_fused): determinism x30, 16-prompt dKL
    # decode +0.0020 / verify +0.0012 (CIs span 0), stop rate 8/8, full TTFT 4k / 16k / 32k -10%, TPOT level or better.
    prefill_hc_fused = True
    prefill_hc_sinkhorn_iters = 0
    # the prefill output map through the hc_post kernel too (needs prefill_hc_fused's post / comb): 0.25 ms per
    # 128-row call (after the reader rewrite; 2.1 ms before) vs 0.95 ms for the broadcast addcmul. Default on (see above).
    prefill_hc_post_kernel = True
    # a prefill block's MoE in as few grouped calls as the expert pool allows instead of one call per 32-token
    # routing chunk (each call reads every active expert's weights): same per-token products and order. Bit-identical
    # (full model, long600 / 1100 / 2600 prefill + 8 decode steps); vLLM full-prefill TTFT 4k / 16k / 32k
    # 37.9 / 161.9 / 324.5 -> 32.0 / 138.5 / 277.2 s. Default on.
    prefill_moe_merge = True
    # the merged prefill MoE cuts its grouped calls only at 32-token boundaries (fixed call lengths; a
    # routing-dependent length compiled new programs behind live traces on most long prompts).
    prefill_moe_merge_aligned = True
    # Streaming prefill page-table ranges come from the host copy of the table instead of a device slice (same values;
    # a device slice compiled a program per new block position).
    streaming_host_page_tables = True
    # a prefill block's router as one call over all its rows instead of one per 32-token chunk.
    prefill_route_merge = False
    # a verify block's M rows go into the local / compressor caches with one program per cache
    # (_paged_update_rows) instead of M row-by-row paged_update_cache chains. Bit-identical (cache bytes and logits
    # equal over 30 / 40 blocks, window / CSA / HCA layers; 43-layer M=6 eager logits equal): 188.0 -> 180.8 ms per
    # verify block. Default on.
    verify_rows_update = True
    # pad a verify block's local key bank (window - 1 + M rows) to whole tiles with masked rows, so the bank
    # and mask concats with the compressed entries need no untilize / tilize. Not bitwise (SDPA key chunks shift).
    # Default on : verify 180.8 -> 175.1 ms per block, verify KL 0.0991 -> 0.0878 on the 4-prompt set.
    verify_local_pad = True
    # within one verify compressor pass, build each selection one-hot / position slot once and reuse it for
    # every table it picks from (identical ops on identical inputs: 43-layer M=6 eager logits equal, trace vs eager
    # 12/12; verify block 180.8 -> 177.5 ms). Default on.
    verify_select_memo = True
    # decode / verify router epilogue (sqrt-softplus, bias, top-k, weight normalization) as one program
    # (OptimizedDecoder._moe_route_fused). Not bitwise (fp32 scores before the top-k); off until accepted.
    moe_route_fused = False

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        layer_idx: int,
        host_weights: HostLayerWeights,
        *,
        config: DeepSeekV4FlashConfig | None = None,
        max_context: int | None = None,
        local_page_permutation: Sequence[int] | None = None,
        compressed_page_permutation: Sequence[int] | None = None,
        weights_dtype: ttnn.DataType = ttnn.bfloat16,
        max_batch_size: int = 1,
        optimization_profile: str | None = None,
        deployment_precision: str = STOCK_PRECISION_POLICY,
        moe_compute_profile: str = "lofi",
        dense_compute_profile: str = "lofi",
        shared_rope: tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor] | None = None,
        expert_pool_size: int | None = None,
        expert_pool_keep_torch: bool = False,
        attention_dtype: ttnn.DataType = ttnn.bfloat4_b,
    ) -> None:
        self.experimental_policy_overrides = tuple(sorted(name for name in os.environ if name.startswith("DSV4_")))
        profile = optimization_profile or os.environ.get("DSV4_OPT_PROFILE", "optimized")
        if profile != "optimized":
            raise ValueError(
                "multichip decoder is qualified only against the optimized single-chip profile, " f"got {profile!r}"
            )
        if weights_dtype != ttnn.bfloat16:
            raise ValueError("multichip decoder requires the optimized baseline's BF16 stable-weight upload")
        if deployment_precision not in SUPPORTED_PRECISION_POLICIES:
            raise ValueError(
                f"unknown multichip precision policy {deployment_precision!r}; "
                f"expected one of {SUPPORTED_PRECISION_POLICIES}"
            )
        self.deployment_precision = deployment_precision
        if deployment_precision == QUIETBOX2_PAGED_BFP4_POLICY:
            expert_pool_size = QUIETBOX2_PAGED_DEFAULT_POOL if expert_pool_size is None else int(expert_pool_size)
            # One MoE call must never need more distinct experts than the pool holds.
            top_k = (config or DeepSeekV4FlashConfig()).num_experts_per_tok
            self.moe_chunk_tokens = max(1, min(PAGED_MOE_CHUNK_TOKENS, expert_pool_size // top_k))
        elif expert_pool_size is not None:
            raise ValueError(f"expert_pool_size requires the {QUIETBOX2_PAGED_BFP4_POLICY!r} precision policy")
        self.expert_pool_size = expert_pool_size
        self.expert_pool_keep_torch = expert_pool_keep_torch
        # streaming prefill writes the whole raw compressor window (False reproduces the last-page-only fill).
        self.streaming_raw_full_window = True
        # attention projections (Q-A/KV/compressor/indexer row, Q-B, index-Q, O-A, O-B) default to
        # BFLOAT4_B; BFLOAT8_B keeps them near the official FP8 values at about twice the bytes.
        if attention_dtype not in {ttnn.bfloat4_b, ttnn.bfloat8_b}:
            raise ValueError(f"attention weights must be BFLOAT4_B or BFLOAT8_B, got {attention_dtype}")
        self.attention_dtype = attention_dtype
        if moe_compute_profile not in MOE_COMPUTE_PROFILES:
            raise ValueError(f"unknown multichip MoE compute profile {moe_compute_profile!r}")
        self.moe_compute_profile = moe_compute_profile
        if dense_compute_profile not in DENSE_COMPUTE_PROFILES:
            raise ValueError(f"unknown multichip dense compute profile {dense_compute_profile!r}")
        self.dense_compute_profile = dense_compute_profile
        self.optimization_policy = optimization_policy(profile)
        self._sinkhorn_iters = self.optimization_policy.sinkhorn_iters
        # A direct reduction of [1, S, 4, H] creates a 64 MiB tile-padded FP32
        # temporary at S=128.  Collapse the four lanes separately so repeated
        # prefills do not require a contiguous 8 MiB block in every DRAM bank.
        self.hc_collapse_strategy = "streamwise_fp32"

        # FunctionalDecoder's constructor intentionally gates on a 1x1 mesh.
        # Reproduce its stable initialization contract here and call the same
        # virtual upload/cache/MoE helpers, rather than weakening that baseline.
        LightweightModule.__init__(self)
        self.mesh_device = mesh_device
        self.config = DeepSeekV4FlashConfig() if config is None else config
        self.config.validate()
        self.layer_idx = layer_idx
        self.kind = self.config.layer_kind(layer_idx)
        if tuple(mesh_device.shape) != TARGET_MESH_SHAPE or mesh_device.get_num_devices() != TP_DEGREE:
            raise ValueError(
                f"multichip decoder requires the complete {TARGET_MESH_SHAPE} four-device mesh, "
                f"got shape={tuple(mesh_device.shape)} devices={mesh_device.get_num_devices()}"
            )
        if max_batch_size <= 0:
            raise ValueError(f"max_batch_size must be positive, got {max_batch_size}")
        self.max_batch_size = max_batch_size
        self.max_context = self.config.max_position_embeddings if max_context is None else max_context
        self.geometry = self.config.cache_geometry(layer_idx, self.max_context)
        self.weights_dtype = weights_dtype
        self.ccl_dtype = os.environ.get("DSV4_MULTICHIP_CCL_DTYPE", "bf16")
        if self.ccl_dtype not in {"bf16", "bfp8"}:
            raise ValueError(f"unknown multichip CCL dtype experiment {self.ccl_dtype!r}")
        self._mapper = ttnn.ReplicateTensorToMesh(mesh_device)
        self._prefix = f"layers.{layer_idx}"
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        # Hardware A/B selected the combined exact-shape program: two B=1
        # decode O-A rows plus eight-bank shared-expert projections.  Retain the
        # environment selector only to reproduce the rejected alternatives.
        self.topology_experiment = os.environ.get(
            "DSV4_MULTICHIP_TOPOLOGY_EXPERIMENT",
            "grouped_o_a_split_shared_dram",
        )
        if self.topology_experiment not in {
            "current",
            "shared_dram_sharded",
            "grouped_o_a_split",
            "grouped_o_a_split_shared_dram",
        }:
            raise ValueError(f"unknown multichip topology experiment {self.topology_experiment!r}")
        self.qr_projection_experiment = os.environ.get("DSV4_MULTICHIP_QR_PROJECTION", "separate")
        if self.qr_projection_experiment not in {"separate", "packed"}:
            raise ValueError(f"unknown multichip QR projection experiment {self.qr_projection_experiment!r}")
        self._tp_packed_qr_projection: ttnn.Tensor | None = None
        self._tp_packed_q_width = 0
        self._attention_decode_all_reduce: _PersistentDecodeAllReduce | None = None
        self._moe_decode_all_reduce: _PersistentDecodeAllReduce | None = None
        self.residual_layout = os.environ.get("DSV4_RESIDUAL_LAYOUT", "replicated")
        if self.residual_layout not in {"replicated", "fractured"}:
            raise ValueError(f"unknown residual layout experiment {self.residual_layout!r}")
        self.attention_ccl_placement = os.environ.get("DSV4_ATTENTION_CCL_PLACEMENT", "post_matmul_allreduce")
        if self.attention_ccl_placement not in {
            "post_matmul_allreduce",
            "fused_mmrs_replicated",
            "fused_mmrs_fractured",
        }:
            raise ValueError(f"unknown attention CCL placement experiment {self.attention_ccl_placement!r}")
        if (self.residual_layout == "fractured") != (self.attention_ccl_placement == "fused_mmrs_fractured"):
            raise ValueError("fractured residual and fused_mmrs_fractured placement must be selected together")
        self._attention_fused_mmrs: _FusedDecodeMatmulAllReduce | None = None
        self._o_b_input_cores_override = (
            int(os.environ["DSV4_O_B_INPUT_CORES"]) if "DSV4_O_B_INPUT_CORES" in os.environ else None
        )
        self._grouped_o_a_split_weights: tuple[ttnn.Tensor, ...] = ()
        self._grouped_o_a_split_input_memory: ttnn.MemoryConfig | None = None
        self._grouped_o_a_split_program: ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig | None = None

        self._upload_dense_weights(host_weights.tensors)
        if shared_rope is None:
            self._upload_rope()
        else:
            self.rope_cos, self.rope_sin, self.rope_trans = shared_rope
        self.cache = self._allocate_cache(local_page_permutation, compressed_page_permutation)
        self.moe = self._build_moe(host_weights)
        # router sqrt(softplus) on the fp32 logits instead of bf16 (FunctionalDecoder._moe_route).
        # False restores the old bf16 router, whose softplus flushes inputs below ~-5 to 0.
        self.router_fp32_softplus = True
        # decode-only cuts, each verified bit-identical (p7/p7_10_bitwise.py; FusedDecoder._attention_decode /
        # _decode_cache_rows). False restores the previous op sequence.
        # C1: identity / identity-ring page tables skip the row-LUT gather.
        self.decode_identity_rows = True
        # C2: HCA reads its whole identity compressed table by a TILE typecast instead of a row gather.
        self.decode_hca_full_table = True
        # C3: decode attention skips the identity embedding over the concatenated bank.
        self.decode_identity_gather = True
        # I1: the decode indexer masks / top-k / selected mask run on query row 0 only.
        self.decode_indexer_row0 = True
        # I3: the decode index-q RoPE + Hadamard run on [64 heads, 128] rows before the head permute. Not bitwise;
        # accepted on the criteria and made the default.
        self.decode_index_q_compact = True
        # I2: the decode index bank is the indexer tensor itself (its spare zero page is the guard), no concat.
        self.decode_index_bank_view = True
        # decode cache-row gathers from a BFP8 TILE cache read the selected rows' tiles directly
        # (FusedDecoder._tile_row_gather) instead of ttnn.embedding untilizing the whole table. Bit-identical
        # (p7/p7_10_c4b_unit.py, p7/p7_10_bitwise.py); TPOT -1.0 ms (trace).
        self.decode_tile_row_gather = True
        # decode Sinkhorn as one fused program (OptimizedDecoder._sinkhorn_fused). Not bitwise; accepted on
        # the criteria and made the default. False restores the op-by-op Sinkhorn.
        self.sinkhorn_fused_kernel = True
        # for prefill: the [1, S, n, n] Sinkhorn of a prefill block as one fused program (bit-identical to
        # the sum / add / div path, p7/p7_12_prefill_test.py with PT_SWITCH). False restores that path.
        self.sinkhorn_fused_prefill = True
        # streaming prefill CSA selection gathers each q chunk's rows in one embedding (identity tables).
        self.prefill_csa_gather_once = True
        # queries per streaming-prefill attention step; 128 = the whole block in one step.
        self.prefill_query_chunk = 32
        # decode mHC map split along K over this many cores (0 = the single-core ttnn.linear). Not bitwise;
        # accepted on the criteria and made the default.
        self.hc_pre_split_k = 16
        # streaming prefill untilizes the CSA compressed table once per block instead of per gather
        # (bit-identical, p7/p7_12_prefill_test.py). False restores the per-gather untilize.
        self.prefill_csa_untilize_once = True
        # mHC A1: decode lane collapse as one [1, n] x [n, H] matmul (OptimizedDecoder._hc_pre). Not
        # bitwise; accepted on the criteria and made the default (TPOT -8 ms).
        self.hc_collapse_matmul = True
        # mHC Sinkhorn sums as matmuls with a 4x4 ones block (OptimizedDecoder._hc_pre). The constant and
        # kernel config are created here, before any trace capture.
        self.sinkhorn_matmul = True
        self.sinkhorn_fused_eps = True
        ones = torch.zeros(1, 1, self.config.hc_mult, self.config.hc_mult)
        ones[..., : self.config.hc_mult, : self.config.hc_mult] = 1.0
        self._sinkhorn_ones = ttnn.from_torch(
            ones,
            dtype=ttnn.float32 if self.optimization_policy.sinkhorn_dtype == "fp32" else ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self._sinkhorn_kernel_config = ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        if self.topology_experiment in {"shared_dram_sharded", "grouped_o_a_split_shared_dram"}:
            self.moe.configure_shared_dram_sharding()
        self._decode_constants()
        self._batch_lanes: tuple[MultichipDecoder, ...] = ()
        if max_batch_size > 1:
            lanes = []
            for lane_idx in range(max_batch_size):
                lane = copy.copy(self)
                lane.max_batch_size = 1
                lane._batch_lanes = ()
                if lane_idx:
                    lane.cache = lane._allocate_cache(local_page_permutation, compressed_page_permutation)
                    lane._decode_constants()
                lanes.append(lane)
            self._batch_lanes = tuple(lanes)

        self.dense_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity={
                "lofi": ttnn.MathFidelity.LoFi,
                "hifi2": ttnn.MathFidelity.HiFi2,
                "hifi4": ttnn.MathFidelity.HiFi4,
            }[dense_compute_profile],
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        if self.attention_ccl_placement in {"fused_mmrs_replicated", "fused_mmrs_fractured"}:
            self._attention_fused_mmrs = _FusedDecodeMatmulAllReduce(
                mesh_device,
                k=self._dram_interleaved_weights[id(self.o_b)].shape[-2],
                n=self._dram_interleaved_weights[id(self.o_b)].shape[-1],
            )
        compact_guards: dict[int, ttnn.Tensor] = {}
        for decoder in (self, *self._batch_lanes):
            decoder.dense_compute_kernel_config = self.dense_compute_kernel_config
            decoder._sinkhorn_iters = self.optimization_policy.sinkhorn_iters
            if decoder.decode_index_guard is None or decoder.cache.indexer is None:
                continue
            guard_dtype = decoder.cache.indexer.tensor.dtype
            if decoder.decode_index_guard.dtype == guard_dtype:
                continue
            old_guard = decoder.decode_index_guard
            compact_guard = compact_guards.get(id(old_guard))
            if compact_guard is None:
                compact_guard = ttnn.typecast(old_guard, guard_dtype)
                compact_guards[id(old_guard)] = compact_guard
                ttnn.deallocate(old_guard)
            decoder.decode_index_guard = compact_guard

        ccl_mode = os.environ.get("DSV4_MULTICHIP_CCL_MODE", "default")
        if ccl_mode not in {"default", "persistent"}:
            raise ValueError(f"unknown multichip CCL mode experiment {ccl_mode!r}")
        if ccl_mode == "persistent":
            attention_ccl = _PersistentDecodeAllReduce(mesh_device)
            moe_ccl = _PersistentDecodeAllReduce(mesh_device)
            for decoder in (self, *self._batch_lanes):
                decoder._attention_decode_all_reduce = attention_ccl
                decoder._moe_decode_all_reduce = moe_ccl

    def _tp_tensor(
        self,
        tensor: torch.Tensor,
        *,
        shard_dim: int,
        dtype: ttnn.DataType,
    ) -> ttnn.Tensor:
        return ttnn.from_torch(
            tensor.contiguous(),
            device=self.mesh_device,
            mesh_mapper=_flat_shard_mapper(self.mesh_device, shard_dim),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _forget_optimized_weight(self, weight: ttnn.Tensor | None) -> None:
        if weight is None:
            return
        weight_id = id(weight)
        interleaved = self._dram_interleaved_weights.pop(weight_id, None)
        self._dram_linear_weight_ids.discard(weight_id)
        self._dram_linear_roles.pop(weight_id, None)
        self._optimized_dense_weight_ids.discard(weight_id)
        ttnn.deallocate(weight)
        if interleaved is not None and interleaved is not weight:
            ttnn.deallocate(interleaved)

    def _configure_tp_dram_linear(self, attribute: str, role: str) -> None:
        """Adapt the optimized decode DRAM program to a rank-local dense row."""

        weight = getattr(self, attribute)
        k, n = weight.shape[-2], weight.shape[-1]
        dram_grid = self.mesh_device.dram_grid_size()
        dram_cores = dram_grid.x
        grid = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_grid.x - 1, dram_grid.y - 1))}
        )
        shard_width = math.ceil(n / (32 * dram_cores)) * 32
        memory = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(grid, (k, shard_width), ttnn.ShardOrientation.ROW_MAJOR),
        )
        sharded = ttnn.to_memory_config(weight, memory)
        setattr(self, attribute, sharded)
        self._dram_linear_weight_ids.add(id(sharded))
        self._dram_interleaved_weights[id(sharded)] = weight
        self._dram_linear_roles[id(sharded)] = role
        self._optimized_dense_weight_ids.add(id(sharded))

    def _upload_dense_weights(self, state: Mapping[str, torch.Tensor]) -> None:
        """Keep optimized replicated projections and replace material TP rows."""

        super()._upload_dense_weights(state)
        p = self._prefix
        dense_dtype = self.attention_dtype

        if self._packed_attention_projection is not None:
            # The generic optimized path first uploads Q-A/KV as BF16 and then
            # device-typecasts those TILE tensors to BFP4 before concatenating
            # them.  Blackhole's device conversion truncates this format and is
            # observably different from the round-to-nearest host BFP packer
            # used by every material TP weight below.  Upload the complete
            # logical [K,N] row directly from host so Q-A/KV use the same
            # source-faithful conversion without changing bytes or topology.
            # Compressed layers append compressor and indexer projections to
            # this row, so retaining only the Q-A/KV prefix would leave their
            # precomputed slice ranges pointing beyond the replacement tensor.
            old_packed = self._packed_attention_projection
            old_packed_width = int(old_packed.shape[-1])
            self._forget_optimized_weight(old_packed)
            packed_host_projections = [
                state[f"{p}.attn.wq_a.weight"].transpose(0, 1),
                state[f"{p}.attn.wkv.weight"].transpose(0, 1),
            ]
            if self.kind.compress_ratio:
                packed_host_projections.extend(
                    (
                        state[f"{p}.attn.compressor.wkv.weight"].transpose(0, 1),
                        state[f"{p}.attn.compressor.wgate.weight"].transpose(0, 1),
                    )
                )
            if self.kind.attention is AttentionKind.CSA:
                index_scale = self.config.index_head_dim**-0.5 * self.config.index_n_heads**-0.5
                packed_host_projections.extend(
                    (
                        state[f"{p}.attn.indexer.compressor.wkv.weight"].transpose(0, 1),
                        state[f"{p}.attn.indexer.compressor.wgate.weight"].transpose(0, 1),
                        (state[f"{p}.attn.indexer.weights_proj.weight"].float() * index_scale).transpose(0, 1),
                    )
                )
            packed_host = torch.cat(packed_host_projections, dim=-1)
            if packed_host.shape[-1] != old_packed_width:
                raise RuntimeError(
                    f"host-packed attention width {packed_host.shape[-1]} != optimized width {old_packed_width}"
                )
            self._packed_attention_projection = self._to_tt(packed_host, dtype=dense_dtype)
            self._optimized_dense_weight_ids.add(id(self._packed_attention_projection))

        # CSA's optimized single-chip path packs Q-B and index-Q.  Q-B is
        # head-parallel here while the independent indexer stays replicated,
        # so the packed QR tensor is intentionally split at load time.
        if self._packed_qr_projection is not None:
            self._forget_optimized_weight(self._packed_qr_projection)
            self._packed_qr_projection = None
            self._packed_qr_projection_ranges = {}
        else:
            self._forget_optimized_weight(self.q_b)
        q_b_host = state[f"{p}.attn.wq_b.weight"].transpose(0, 1)
        if self.kind.attention is AttentionKind.CSA and self.qr_projection_experiment == "packed":
            # Every rank owns a different Q-B head slice but consumes the same
            # index-Q row.  Arrange four [local Q-B | replicated index-Q]
            # segments on host, then shard the segment axis so one matmul reads
            # QR once per rank and produces both projections.
            index_q_host = state[f"{p}.attn.indexer.wq_b.weight"].transpose(0, 1)
            q_b_chunks = torch.chunk(q_b_host, TP_DEGREE, dim=-1)
            self._tp_packed_q_width = q_b_chunks[0].shape[-1]
            packed_host = torch.cat(
                [torch.cat((q_b_chunk, index_q_host), dim=-1) for q_b_chunk in q_b_chunks],
                dim=-1,
            )
            self.q_b = None
            self.index_q = None
            self._tp_packed_qr_projection = self._tp_tensor(
                packed_host,
                shard_dim=-1,
                dtype=dense_dtype,
            )
            self._optimized_dense_weight_ids.add(id(self._tp_packed_qr_projection))
        else:
            self.q_b = self._tp_tensor(q_b_host, shard_dim=-1, dtype=dense_dtype)
            self._optimized_dense_weight_ids.add(id(self.q_b))
        if self.kind.attention is AttentionKind.CSA and self._tp_packed_qr_projection is None:
            # The old index_q object was deallocated when the packed QR weight
            # was formed.  Upload the folded single-chip value independently.
            self.index_q = self._weight(state[f"{p}.attn.indexer.wq_b.weight"], dtype=dense_dtype)
            self._optimized_dense_weight_ids.add(id(self.index_q))

        old_grouped = self.o_a
        grouped_interleaved = self._grouped_output_interleaved_weight
        self._forget_optimized_weight(old_grouped)
        if grouped_interleaved is not None and grouped_interleaved is not old_grouped:
            ttnn.deallocate(grouped_interleaved)
        self._grouped_output_interleaved_weight = None
        self._grouped_output_input_memory = None
        self._grouped_output_output_memory = None
        self._grouped_output_program = None
        wo_a = state[f"{p}.attn.wo_a.weight"].reshape(
            self.config.o_groups,
            self.config.o_lora_rank,
            self.config.grouped_attention_width,
        )
        self.o_a = self._tp_tensor(wo_a.transpose(-2, -1).unsqueeze(0), shard_dim=1, dtype=dense_dtype)
        self._optimized_dense_weight_ids.add(id(self.o_a))
        if self.topology_experiment in {"grouped_o_a_split", "grouped_o_a_split_shared_dram"}:
            self._configure_grouped_o_a_split(wo_a, dense_dtype)

        self._forget_optimized_weight(self.o_b)
        self.o_b = self._tp_tensor(
            state[f"{p}.attn.wo_b.weight"].transpose(0, 1),
            shard_dim=-2,
            dtype=dense_dtype,
        )
        self._optimized_dense_weight_ids.add(id(self.o_b))
        if self._tp_packed_qr_projection is None:
            self._configure_tp_dram_linear("q_b", "q_b")
        else:
            self._configure_tp_dram_linear("_tp_packed_qr_projection", "q_b")
        self._configure_tp_dram_linear("o_b", "o_b")

        ttnn.deallocate(self.attention_sink)
        self.attention_sink = self._tp_tensor(
            state[f"{p}.attn.attn_sink"].reshape(1, -1, 1, 1),
            shard_dim=1,
            dtype=ttnn.bfloat16,
        )

        router_dtype = os.environ.get("DSV4_MULTICHIP_ROUTER_DTYPE", "bf16")
        router_tt_dtype = {
            "fp32": None,
            "bf16": ttnn.bfloat16,
            "bfp8": ttnn.bfloat8_b,
        }.get(router_dtype)
        if router_dtype not in {"fp32", "bf16", "bfp8"}:
            raise ValueError(f"unknown multichip router dtype experiment {router_dtype!r}")
        if router_tt_dtype is not None:
            old_gate_weight = self.gate_weight
            self.gate_weight = ttnn.typecast(old_gate_weight, router_tt_dtype)
            ttnn.deallocate(old_gate_weight)

    def _configure_grouped_o_a_split(self, wo_a: torch.Tensor, dtype: ttnn.DataType) -> None:
        """Create two exact B=1 DRAM-sharded O-A rows per TP rank for A/B."""

        dram_size = self.mesh_device.dram_grid_size()
        dram_cores = dram_size.x * dram_size.y
        if dram_cores != 8:
            raise ValueError(f"grouped O-A split A/B requires eight DRAM banks, got {dram_cores}")
        dram_grid = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(dram_size.x - 1, dram_size.y - 1),
                )
            }
        )
        weights = []
        for local_group in range(self.config.o_groups // TP_DEGREE):
            # Contiguous TP chunks are pairs: rank r owns groups 2r and 2r+1.
            group_row = wo_a[local_group::2].transpose(-2, -1).unsqueeze(0)
            weight = self._tp_tensor(group_row, shard_dim=1, dtype=dtype)
            memory = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.DRAM,
                ttnn.ShardSpec(
                    dram_grid,
                    (weight.shape[-2], weight.shape[-1] // dram_cores),
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            )
            weights.append(ttnn.to_memory_config(weight, memory))
        self._grouped_o_a_split_weights = tuple(weights)
        worker_grid = ttnn.num_cores_to_corerangeset(
            dram_cores,
            self.mesh_device.compute_with_storage_grid_size(),
            row_wise=True,
        )
        self._grouped_o_a_split_input_memory = ttnn.create_sharded_memory_config_(
            shape=(32, self.config.grouped_attention_width // dram_cores),
            core_grid=worker_grid,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self._grouped_o_a_split_program = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=int(os.environ.get("DSV4_GROUPED_O_A_BLOCK", "16")),
            per_core_M=1,
            per_core_N=1,
            fused_activation=None,
        )

    def _build_moe(self, host: HostLayerWeights) -> MultichipDeepSeekSparseMoE:
        p = self._prefix
        use_bfp2_gate_up = self.deployment_precision == QUIETBOX2_MIXED_BFP2_POLICY or (
            self.deployment_precision == QUIETBOX2_SELECTIVE_BFP4_POLICY
            and self.layer_idx not in QUIETBOX2_SELECTIVE_BFP4_LAYERS
        )
        if use_bfp2_gate_up and not hasattr(ttnn, "bfloat2_b"):
            # BFLOAT2_B is a ndalyTT core extension that upstream tt-metal does not carry; only the old selective
            # (all-resident) deployment uses it, the default paged deployment keeps every routed expert in BFLOAT4_B.
            raise RuntimeError("BFLOAT2_B routed gate/up weights need a tt-metal build with ttnn.bfloat2_b")
        routed_gate_up_dtype = ttnn.bfloat2_b if use_bfp2_gate_up else ttnn.bfloat4_b
        return MultichipDeepSeekSparseMoE(
            self.mesh_device,
            host.expert_w1,
            host.expert_w2,
            host.expert_w3,
            host.tensors[f"{p}.ffn.shared_experts.w1.weight"],
            host.tensors[f"{p}.ffn.shared_experts.w2.weight"],
            host.tensors[f"{p}.ffn.shared_experts.w3.weight"],
            hidden_size=self.config.hidden_size,
            intermediate_size=self.config.moe_intermediate_size,
            num_experts=self.config.num_experts,
            top_k=self.config.num_experts_per_tok,
            limit=self.config.swiglu_limit,
            weights_dtype=ttnn.bfloat4_b,
            routed_gate_up_dtype=routed_gate_up_dtype,
            compute_profile=self.moe_compute_profile,
            expert_pool_size=self.expert_pool_size,
            keep_torch_experts=self.expert_pool_keep_torch,
            expert_store=host.expert_store,
        )

    def _allocate_cache(
        self,
        local_permutation: Sequence[int] | None,
        compressed_permutation: Sequence[int] | None,
    ) -> DecoderCache:
        """Allocate stack-feasible replicated cache banks.

        Local attention and raw compressor histories are bounded page rings;
        compressed attention/index banks remain full-context and one-to-one.
        The same layout is used on every TP rank because the model has one KV
        head and sharding it would turn each token into another collective.
        """

        shapes = multichip_cache_shapes(self.config, self.layer_idx, self.max_context)
        local_physical, local_logical, local_width = shapes["local"]
        local = self._page_tensor(
            local_physical,
            local_logical,
            local_width,
            cyclic=True,
            permutation=_cyclic_permutation(
                local_permutation,
                logical_pages=local_logical,
                physical_pages=local_physical,
            ),
        )
        compressed_shape = shapes["compressed"]
        if compressed_shape is None:
            return DecoderCache(self.max_context, local, None, None, None, None, None, None)

        compressed_physical, compressed_logical, compressed_width = compressed_shape
        compressed = self._page_tensor(
            compressed_physical,
            compressed_logical,
            compressed_width,
            cyclic=False,
            permutation=compressed_permutation,
        )
        compressor_shape = shapes["compressor"]
        if compressor_shape is None:
            raise RuntimeError("compressed attention cache is missing its compressor history")
        compressor_physical, compressor_logical, compressor_width = compressor_shape
        compressor = self._page_tensor(
            compressor_physical,
            compressor_logical,
            compressor_width,
            cyclic=True,
            permutation=None,
        )
        indexer_shape = shapes["indexer"]
        if indexer_shape is None:
            return DecoderCache(
                self.max_context,
                local,
                compressed,
                compressor,
                compressor,
                None,
                None,
                None,
            )

        indexer_physical, indexer_logical, indexer_width = indexer_shape
        # one zero page past the mapped ones. The page table maps only the logical pages, so nothing
        # writes it, and the whole tensor is the decode index bank plus its 32-row zero guard (rows stays the
        # logical row count everywhere else).
        indexer = self._page_tensor(
            indexer_physical + 1,
            indexer_logical,
            indexer_width,
            cyclic=False,
            permutation=None,
        )
        indexer = dataclasses.replace(indexer, rows=indexer_logical * self.config.page_size)
        return DecoderCache(
            self.max_context,
            local,
            compressed,
            compressor,
            compressor,
            indexer,
            compressor,
            compressor,
        )

    def _moe_forward(self, x: ttnn.Tensor, input_ids: ttnn.Tensor) -> ttnn.Tensor:
        """Reduce once after all rank-local active-expert chunks are joined."""

        self.moe.use_shared_dram_sharding = bool(self.moe._shared_dram_programs) and x.shape[2] == 1
        try:
            if self.moe.pool is not None and x.shape[2] > 1:
                partial = self._paged_moe_prefill(x, input_ids)
            else:
                partial = super()._moe_forward(x, input_ids)
            if getattr(self, "residual_layout", "replicated") == "fractured" and x.shape[2] == 1:
                return ttnn.reduce_scatter(
                    partial,
                    dim=3,
                    num_links=CCL_NUM_LINKS,
                    topology=ttnn.Topology.Ring,
                    cluster_axis=TP_AXIS,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            persistent = getattr(self, "_moe_decode_all_reduce", None)
            return _all_reduce_partial(partial, persistent, ccl_dtype=self.ccl_dtype)
        finally:
            self.moe.use_shared_dram_sharding = False

    def _paged_moe_prefill(self, x: ttnn.Tensor, input_ids: ttnn.Tensor) -> ttnn.Tensor:
        """Paged prefill MoE with one router readback per layer instead of one per chunk.

        Every 32-token chunk is routed on device first (the resident path's chunking, so router values are
        identical), the picks are read back in one blocking transfer, then each chunk's experts are loaded and
        its MoE issued without further host waits: a slot or table write is enqueued behind the MoE work that
        still reads the old contents and the dispatcher stalls a buffer write until earlier programs finish.
        """

        c = self.config
        pool = self.moe.pool
        seq = x.shape[2]
        routed = []
        # prefill_route_merge routes the whole block in one router call instead of one per 32 tokens
        route_tokens = seq if getattr(self, "prefill_route_merge", False) else PAGED_PREFILL_ROUTE_TOKENS
        for start in range(0, seq, route_tokens):
            end = min(seq, start + route_tokens)
            chunk = ttnn.slice(x, [0, 0, start, 0], [1, 1, end, c.hidden_size])
            ids = ttnn.slice(input_ids, [0, start], [1, end])
            selected, weights = self._moe_route(chunk, ids)
            routed.append((start, end, chunk, selected, weights))
        # Expert ids are < 256, exact in bfloat16; one concat and one blocking read for the whole layer.
        ids_bf16 = [ttnn.typecast(selected, ttnn.bfloat16) for _, _, _, selected, _ in routed]
        joined = ids_bf16[0] if len(ids_bf16) == 1 else ttnn.concat(ids_bf16, dim=2)
        host = ttnn.to_torch((ttnn.get_device_tensors(joined) or (joined,))[0]).reshape(-1, self.moe.top_k)
        picks = [host[start:end].to(torch.int64) for start, end, _, _, _ in routed]
        outputs = []
        eager = self.moe.pool_eager_ensure
        self.moe.pool_eager_ensure = False
        if getattr(self, "prefill_moe_merge", False) and self.moe.prefill_grouped and seq > PAGED_PREFILL_ROUTE_TOKENS:
            # the MoE over the whole block in as few grouped calls as the pool allows (the routing above stays
            # per 32-token chunk), so each active expert's weights are read once per call instead of once per chunk.
            all_picks = torch.cat(picks)
            all_weights = ttnn.concat([weights for _, _, _, _, weights in routed], dim=2)
            try:
                align = PAGED_PREFILL_ROUTE_TOKENS if getattr(self, "prefill_moe_merge_aligned", False) else 1
                for lo, hi in _grouped_call_ranges(all_picks.tolist(), pool.pool_size, pool.pool_size + 1, align):
                    group = x if (lo, hi) == (0, seq) else ttnn.slice(x, [0, 0, lo, 0], [1, 1, hi, c.hidden_size])
                    group_weights = (
                        all_weights
                        if (lo, hi) == (0, seq)
                        else ttnn.slice(all_weights, [0, 0, lo, 0], [1, 1, hi, self.moe.top_k])
                    )
                    pool.ensure(all_picks[lo:hi].flatten().tolist())
                    out = self.moe.forward_grouped(group, all_picks[lo:hi], group_weights)
                    outputs.append(ttnn.slice(out, [0, 0, 0, 0], [1, 1, hi - lo, c.hidden_size]))
            finally:
                self.moe.pool_eager_ensure = eager
            return outputs[0] if len(outputs) == 1 else ttnn.concat(outputs, dim=2)
        try:
            for (start, end, chunk, selected, weights), chunk_picks in zip(routed, picks, strict=True):
                for lo, hi in _pool_groups(chunk_picks.tolist(), pool.pool_size):
                    if (lo, hi) == (0, end - start):
                        group, group_selected, group_weights = chunk, selected, weights
                    else:
                        group = ttnn.slice(chunk, [0, 0, lo, 0], [1, 1, hi, c.hidden_size])
                        group_selected = ttnn.slice(selected, [0, 0, lo, 0], [1, 1, hi, self.moe.top_k])
                        group_weights = ttnn.slice(weights, [0, 0, lo, 0], [1, 1, hi, self.moe.top_k])
                    pool.ensure(chunk_picks[lo:hi].flatten().tolist())
                    if self.moe.prefill_grouped:
                        out = self.moe.forward_grouped(group, chunk_picks[lo:hi], group_weights)
                    else:
                        out = self.moe.forward(group, group_selected, group_weights)
                    outputs.append(ttnn.slice(out, [0, 0, 0, 0], [1, 1, hi - lo, c.hidden_size]))
        finally:
            self.moe.pool_eager_ensure = eager
        return outputs[0] if len(outputs) == 1 else ttnn.concat(outputs, dim=2)

    def decode_route(
        self,
        hidden_streams: ttnn.Tensor,
        input_ids: ttnn.Tensor,
        current_pos: ttnn.Tensor,
    ) -> "DecodeRouteState":
        """First half of a split decode layer: everything up to and including the router.

        Same ops as the replicated decode_forward up to the MoE; the host reads ``selected`` back and fills the
        expert pool before decode_finish() runs the MoE (expert paging, S mode).
        """

        if self.residual_layout != "replicated" or self._batch_lanes:
            raise ValueError("split decode is defined for the replicated single-lane decoder")
        expected = (1, 1, self.config.hc_mult, self.config.hidden_size)
        if tuple(hidden_streams.shape) != expected:
            raise ValueError(f"hidden_streams shape {tuple(hidden_streams.shape)} != {expected}")
        residual = hidden_streams
        post, comb, collapsed = self._hc_pre(residual, self.hc_attn_fn, self.hc_attn_base, self.hc_attn_scale)
        attention_input = self._optimized_rms_norm(collapsed, weight=self.attn_norm, epsilon=self.config.rms_norm_eps)
        attention = self._attention_decode(attention_input, current_pos)
        residual = self._hc_post(attention, residual, post, comb)
        post, comb, collapsed = self._hc_pre(residual, self.hc_ffn_fn, self.hc_ffn_base, self.hc_ffn_scale)
        ffn_input = self._optimized_rms_norm(
            collapsed,
            weight=self.ffn_norm,
            epsilon=self.config.rms_norm_eps,
            interleaved_memory=(
                ttnn.L1_MEMORY_CONFIG if self.optimization_policy.ffn_norm_l1 else ttnn.DRAM_MEMORY_CONFIG
            ),
        )
        chunk = ttnn.slice(ffn_input, [0, 0, 0, 0], [1, 1, 1, self.config.hidden_size])
        ids = ttnn.slice(input_ids, [0, 0], [1, 1])
        selected, weights = self._moe_route(chunk, ids)
        return DecodeRouteState(residual, post, comb, chunk, selected, weights)

    def decode_finish(self, state: "DecodeRouteState") -> ttnn.Tensor:
        """Second half of a split decode layer: MoE, its all-reduce and the FFN residual update."""

        self.moe.use_shared_dram_sharding = bool(self.moe._shared_dram_programs)
        try:
            out = self.moe.forward(state.chunk, state.selected, state.weights)
            partial = ttnn.slice(out, [0, 0, 0, 0], [1, 1, 1, self.config.hidden_size])
            persistent = getattr(self, "_moe_decode_all_reduce", None)
            moe = _all_reduce_partial(partial, persistent, ccl_dtype=self.ccl_dtype)
        finally:
            self.moe.use_shared_dram_sharding = False
        return self._hc_post(moe, state.residual, state.post, state.comb)

    def decode_forward(
        self,
        hidden_streams: ttnn.Tensor,
        input_ids: ttnn.Tensor,
        current_pos: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """Carry the residual-width shard through both mHC residual updates for A/B."""

        if self.residual_layout == "replicated":
            return super().decode_forward(hidden_streams, input_ids, current_pos)
        if self._batch_lanes:
            raise ValueError("fractured residual A/B is defined only for the measured single-lane decode")
        expected = (1, 1, self.config.hc_mult, self.config.hidden_size // TP_DEGREE)
        if tuple(hidden_streams.shape) != expected:
            raise ValueError(f"fractured hidden_streams shape {tuple(hidden_streams.shape)} != {expected}")

        def gather_residual(value: ttnn.Tensor) -> ttnn.Tensor:
            return ttnn.all_gather(
                value,
                dim=3,
                num_links=CCL_NUM_LINKS,
                topology=ttnn.Topology.Ring,
                cluster_axis=TP_AXIS,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        def fractured_hc_post(
            output: ttnn.Tensor,
            residual: ttnn.Tensor,
            post: ttnn.Tensor,
            comb: ttnn.Tensor,
        ) -> ttnn.Tensor:
            seq = residual.shape[1]
            mixed = ttnn.matmul(
                comb,
                residual,
                transpose_a=True,
                dtype=ttnn.bfloat16,
                compute_kernel_config=self.compute_kernel_config,
            )
            return ttnn.addcmul(
                mixed,
                ttnn.reshape(output, [1, seq, 1, self.config.hidden_size // TP_DEGREE]),
                ttnn.reshape(post, [1, seq, self.config.hc_mult, 1]),
            )

        residual = hidden_streams
        gathered = gather_residual(residual)
        post, comb, collapsed = self._hc_pre(gathered, self.hc_attn_fn, self.hc_attn_base, self.hc_attn_scale)
        attention = self._attention_decode(
            self._optimized_rms_norm(collapsed, weight=self.attn_norm, epsilon=self.config.rms_norm_eps),
            current_pos,
        )
        residual = fractured_hc_post(attention, residual, post, comb)

        gathered = gather_residual(residual)
        post, comb, collapsed = self._hc_pre(gathered, self.hc_ffn_fn, self.hc_ffn_base, self.hc_ffn_scale)
        moe = self._moe_forward(
            self._optimized_rms_norm(
                collapsed,
                weight=self.ffn_norm,
                epsilon=self.config.rms_norm_eps,
                interleaved_memory=(
                    ttnn.L1_MEMORY_CONFIG if self.optimization_policy.ffn_norm_l1 else ttnn.DRAM_MEMORY_CONFIG
                ),
            ),
            input_ids,
        )
        return fractured_hc_post(moe, residual, post, comb)

    def decode_forward_block(
        self,
        hidden_streams: ttnn.Tensor,
        input_ids: ttnn.Tensor,
        start: "int | ttnn.Tensor",
    ) -> ttnn.Tensor:
        """One decode pass over M = hidden_streams.shape[1] consecutive tokens at positions
        start .. start + M - 1, as a speculative-decoding verify block.

        Every cache this touches is position-indexed and each ring keeps more than its live window plus M - 1
        rows (local 160 >= 128 + 31), so rows past the accepted prefix need no rollback: their entries are masked
        for earlier rows (causal / valid-count masks) and rewritten when the real token reaches that position.
        Row r attends exactly as a single-token decode at position start + r after rows < r were decoded.
        ``start`` is the first row's position, a device int32 [1] tensor (traceable) or a host int (eager).
        """

        if self.residual_layout != "replicated" or self._batch_lanes:
            raise ValueError("block decode is defined for the replicated single-lane decoder")
        m = int(hidden_streams.shape[1])
        expected = (1, m, self.config.hc_mult, self.config.hidden_size)
        if tuple(hidden_streams.shape) != expected or not 1 <= m <= 32:
            raise ValueError(f"block decode takes [1, M <= 32, n, H] streams, got {tuple(hidden_streams.shape)}")
        if isinstance(start, int):
            if start + m > self.max_context:
                raise ValueError(f"block [{start}, {start + m}) exceeds context {self.max_context}")
            start = self._host_int(torch.tensor([start]))
        residual = hidden_streams
        self._verify_active = True
        post, comb, collapsed = self._hc_pre(residual, self.hc_attn_fn, self.hc_attn_base, self.hc_attn_scale)
        attention = self._attention_verify(
            self._optimized_rms_norm(collapsed, weight=self.attn_norm, epsilon=self.config.rms_norm_eps),
            start,
        )
        residual = self._hc_post(attention, residual, post, comb)
        post, comb, collapsed = self._hc_pre(residual, self.hc_ffn_fn, self.hc_ffn_base, self.hc_ffn_scale)
        self._verify_active = False
        moe = self._moe_forward(
            self._optimized_rms_norm(
                collapsed,
                weight=self.ffn_norm,
                epsilon=self.config.rms_norm_eps,
                interleaved_memory=(
                    ttnn.L1_MEMORY_CONFIG if self.optimization_policy.ffn_norm_l1 else ttnn.DRAM_MEMORY_CONFIG
                ),
            ),
            input_ids,
        )
        return self._hc_post(moe, residual, post, comb)

    def verify_route(
        self,
        hidden_streams: ttnn.Tensor,
        input_ids: ttnn.Tensor,
        start: ttnn.Tensor,
    ) -> VerifyRouteState:
        """First half of a split verify layer: decode_forward_block up to and including the router; the host
        reads ``selected`` back, loads the experts and fills the verify buffers before verify_finish runs."""

        residual = hidden_streams
        self._verify_active = True
        post, comb, collapsed = self._hc_pre(residual, self.hc_attn_fn, self.hc_attn_base, self.hc_attn_scale)
        attention = self._attention_verify(
            self._optimized_rms_norm(collapsed, weight=self.attn_norm, epsilon=self.config.rms_norm_eps),
            start,
        )
        residual = self._hc_post(attention, residual, post, comb)
        post, comb, collapsed = self._hc_pre(residual, self.hc_ffn_fn, self.hc_ffn_base, self.hc_ffn_scale)
        self._verify_active = False
        ffn_input = self._optimized_rms_norm(
            collapsed,
            weight=self.ffn_norm,
            epsilon=self.config.rms_norm_eps,
            interleaved_memory=(
                ttnn.L1_MEMORY_CONFIG if self.optimization_policy.ffn_norm_l1 else ttnn.DRAM_MEMORY_CONFIG
            ),
        )
        self._verify_active = True  # the router's decode / verify fast path covers the block rows too
        selected, weights = self._moe_route(ffn_input, input_ids)
        self._verify_active = False
        return VerifyRouteState(residual, post, comb, ffn_input, selected, weights)

    def verify_finish(self, state: VerifyRouteState, buffers: dict) -> ttnn.Tensor:
        """Second half of a split verify layer: grouped MoE from the persistent verify buffers, all-reduce, mHC."""

        partial = self.moe.forward_grouped_static(state.ffn_input, state.weights, buffers)
        moe = _all_reduce_partial(partial, None, ccl_dtype=self.ccl_dtype)
        return self._hc_post(moe, state.residual, state.post, state.comb)

    def _verify_row(self, value: ttnn.Tensor, r: int, m: int) -> ttnn.Tensor:
        """Row r of an [1, 1, M, W] BF16 TILE tensor as [1, 1, 1, W]: an exact one-hot matmul (verify_row_matmul)
        instead of a row slice inside a tile, which goes through untilize + tilize."""

        if not self.verify_row_matmul:
            return ttnn.slice(value, [0, 0, r, 0], [value.shape[0], value.shape[1], r + 1, value.shape[3]])
        return ttnn.matmul(
            self._verify_constants(m)["row_hot"][r],
            value,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute_kernel_config,
        )

    def _indexer_topk_call(self, logits: ttnn.Tensor, k: int, valid_length: int) -> ttnn.Tensor:
        if not self.indexer_topk_wide or valid_length < INDEXER_TOPK_WIDE:
            return ttnn.experimental.topk_large_indices(logits, k=k, valid_length=valid_length)
        wide = ttnn.experimental.topk_large_indices(logits, k=INDEXER_TOPK_WIDE, valid_length=valid_length)
        shape = [int(d) for d in wide.shape]
        return ttnn.slice(wide, [0] * len(shape), shape[:-1] + [k])

    def _host_int(self, values: torch.Tensor) -> ttnn.Tensor:
        return self._to_tt(values.to(torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)

    def _verify_constants(self, m: int) -> dict[str, ttnn.Tensor]:
        """Position-independent device constants of an M-row verify block (built once per M, before any capture)."""

        cache = self.__dict__.setdefault("_verify_constant_cache", {})
        key = (m, bool(self.verify_local_pad))
        if key in cache:
            return cache[key]
        c = self.config
        window = c.sliding_window
        rows = torch.arange(m)
        offsets = torch.arange(1 - window, m)  # local bank: start - (window - 1) .. start + M - 1
        if self.verify_local_pad:
            # pad the local bank to whole tiles with copies of offset 0 (a valid position, masked out below)
            # so the bank / mask concats with the compressed entries stay tile-aligned (no untilize + tilize)
            offsets = torch.cat(
                [offsets, torch.zeros(_round_up(offsets.numel(), 32) - offsets.numel(), dtype=offsets.dtype)]
            )
        n_local = offsets.numel()
        # row r sees bank offsets o with r - window < o <= r (and start + o >= 0, applied on device)
        window_valid = (offsets.reshape(1, -1) <= rows.reshape(-1, 1)) & (
            offsets.reshape(1, -1) > rows.reshape(-1, 1) - window
        )
        window_valid[:, window - 1 + m :] = False  # the padding columns (verify_local_pad)
        index_parts = [torch.arange(n_local).repeat(m, 1)]
        if self.kind.attention is AttentionKind.HCA:
            entries = self.decode_compressed_ordinals.shape[-1]
            index_parts.append(n_local + torch.arange(entries).repeat(m, 1))
        elif self.kind.attention is AttentionKind.CSA:
            index_parts.append(n_local + rows.reshape(-1, 1) * c.index_topk + torch.arange(c.index_topk))
        bf16 = lambda t: self._to_tt(t.to(torch.float32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)  # noqa: E731
        cache[key] = {
            "rows": self._host_int(rows.reshape(1, m)),
            "local_offsets": self._host_int(offsets.reshape(1, -1)),
            "window_valid": bf16(window_valid.reshape(m, 1, 1, n_local)),
            "indices": self._host_int(torch.cat(index_parts, dim=1)),
            "row_hot": [bf16(torch.eye(m)[r].reshape(1, 1, 1, m)) for r in range(m)],
        }
        if self.kind.compress_ratio:
            self._select_ordinals(m, self.kind.compress_ratio)  # verify APE bias rows
            self._select_ordinals(1, m)  # compressor candidate rows
        return cache[key]

    def _attention_verify(self, x: ttnn.Tensor, start: ttnn.Tensor) -> ttnn.Tensor:
        """Block-decode attention for decode_forward_block. x: [1, 1, M, H]; start: device int32 [1] (first row's
        position). Everything position-dependent is computed on device, so the block can be traced."""

        c = self.config
        m = int(x.shape[2])
        const = self._verify_constants(m)
        positions = ttnn.add(const["rows"], start)  # [1, M]
        row_pos = [start if r == 0 else ttnn.add(start, r) for r in range(m)]
        qr, q, kv, rope = self._project_q_kv(x, positions)
        projections = dict(getattr(self, "_fused_attention_projections", {}))
        index_q = getattr(self, "_fused_index_q_projection", None)
        index_w = getattr(self, "_fused_index_weight_projection", None)

        def mask_from(valid: ttnn.Tensor) -> ttnn.Tensor:
            mask = ttnn.typecast(ttnn.where(valid, 0.0, ATTENTION_MASK_VALUE), ttnn.bfloat16)
            return mask if mask.layout == ttnn.TILE_LAYOUT else ttnn.to_layout(mask, ttnn.TILE_LAYOUT)

        if self.verify_rows_update:
            self._paged_update_rows(self.cache.local, kv, start)
        else:
            for r in range(m):
                self._paged_update(self.cache.local, self._verify_row(kv, r, m), row_pos[r])

        logical = ttnn.add(start, const["local_offsets"])
        nonnegative = ttnn.to_layout(ttnn.typecast(ttnn.ge(logical, 0), ttnn.bfloat16), ttnn.TILE_LAYOUT)
        local_kv = self._decode_cache_rows(self.cache.local, ttnn.maximum(logical, 0))
        n_local = int(const["local_offsets"].shape[-1])
        parts = [local_kv]
        masks = [mask_from(ttnn.mul(const["window_valid"], ttnn.reshape(nonnegative, [1, 1, 1, n_local])))]

        if self.kind.compress_ratio:
            ratio = self.kind.compress_ratio
            self._compressor_verify(x, start, positions, row_pos, projections, m)
            # valid compressed entries per row: (start + r + 1) // ratio
            valid_entries = ttnn.div(ttnn.add(positions, 1), ratio, rounding_mode="floor")
            if self.kind.attention is AttentionKind.HCA:
                compressed = self.cache.compressed
                ordinals = self.decode_compressed_ordinals
                entries = ordinals.shape[-1]
                comp_parts = []
                for begin in range(0, entries, HCA_CACHE_GATHER_CHUNK_ROWS):
                    end = min(begin + HCA_CACHE_GATHER_CHUNK_ROWS, entries)
                    comp_parts.append(self._decode_cache_rows(compressed, ttnn.slice(ordinals, [0, begin], [1, end])))
                parts.append(comp_parts[0] if len(comp_parts) == 1 else ttnn.concat(comp_parts, dim=1))
                masks.append(
                    mask_from(
                        ttnn.lt(
                            ttnn.reshape(ordinals, [1, 1, 1, entries]),
                            ttnn.reshape(valid_entries, [m, 1, 1, 1]),
                        )
                    )
                )
            else:
                idx_q = ttnn.reshape(index_q, [1, m, c.index_n_heads, c.index_head_dim])
                idx_q = ttnn.permute(idx_q, (0, 2, 1, 3))
                idx_q = self._rope(idx_q, *rope)
                idx_q = ttnn.matmul(idx_q, self.index_hadamard)
                idx_w = index_w  # projection order [1, 1, S, Hi] (upstream indexer_score #55614)
                indexer_rows = self.cache.indexer.tensor.shape[0] * c.page_size
                index_bank = ttnn.reshape(self.cache.indexer.tensor, [1, 1, indexer_rows, c.index_head_dim])
                guard_rows = self.decode_index_guard.shape[2]
                if indexer_rows != self.cache.indexer.rows + guard_rows:
                    if indexer_rows != self.cache.indexer.rows:
                        index_bank = ttnn.slice(
                            index_bank, [0, 0, 0, 0], [1, 1, self.cache.indexer.rows, c.index_head_dim]
                        )
                    index_bank = ttnn.concat([index_bank, self.decode_index_guard], dim=2)
                length = None
                if self.indexer_topk_dynamic and not self.indexer_topk_wide:
                    # the block's last row sees the most entries (positions increase down the block)
                    last = ttnn.slice(ttnn.reshape(valid_entries, [m, 1]), [m - 1, 0], [m, 1])
                    length = ttnn.reshape(ttnn.to_layout(ttnn.typecast(last, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT), [1])
                topk, topk_mask = self._indexer_topk(
                    idx_q,
                    idx_w,
                    index_bank,
                    ttnn.reshape(valid_entries, [m, 1]),
                    ordinals=self.decode_index_ordinals,
                    length=length,
                )
                selected = ttnn.typecast(ttnn.eq(ttnn.reshape(topk_mask, [1, 1, m, c.index_topk]), 0.0), ttnn.int32)
                flat = ttnn.mul(ttnn.typecast(topk, ttnn.int32), selected)
                parts.append(self._decode_cache_rows(self.cache.compressed, ttnn.reshape(flat, [1, m * c.index_topk])))
                masks.append(
                    topk_mask if topk_mask.layout == ttnn.TILE_LAYOUT else ttnn.to_layout(topk_mask, ttnn.TILE_LAYOUT)
                )

        bank = parts[0] if len(parts) == 1 else ttnn.concat(parts, dim=1)
        if self.verify_sdpa and self.kind.attention is not AttentionKind.CSA:
            # One SDPA over the bank every row shares (the local window and all HCA entries), each row's visibility
            # in its mask: no per-row bank copies. CSA keeps the per-row gather: its rows read different top-k
            # blocks, and a stacked M x top-k shared bank measured slower (202 vs 198 ms per 4-token block).
            mask = masks[0] if len(masks) == 1 else ttnn.concat(masks, dim=-1)
            n = int(bank.shape[1])
            pad = _round_up(n, 128) - n
            keys = ttnn.reshape(bank, [1, 1, n, c.head_dim])
            if pad:
                keys = ttnn.pad(keys, [(0, 0), (0, 0), (0, pad), (0, 0)], 0.0)
                mask = ttnn.pad(mask, [(0, 0), (0, 0), (0, 0), (0, pad)], ATTENTION_MASK_VALUE)
            out = ttnn.transformer.scaled_dot_product_attention(
                q,
                keys,
                keys,
                attn_mask=ttnn.reshape(mask, [1, 1, m, n + pad]),
                is_causal=False,
                scale=c.head_dim**-0.5,
                attention_sink=self.attention_sink,
                program_config=ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=self.mesh_device.compute_with_storage_grid_size(),
                    q_chunk_size=32,
                    k_chunk_size=128,
                    exp_approx_mode=False,
                ),
                compute_kernel_config=self.compute_kernel_config,
            )
        else:
            mask = masks[0] if len(masks) == 1 else ttnn.concat(masks, dim=-1)
            out = self._gathered_attention(
                q,
                ttnn.reshape(bank, [bank.shape[1], c.head_dim]),
                const["indices"],
                mask,
            )
        out = self._rope(out, *rope, inverse=True)
        return self._grouped_output_projection(out, m, verify_rows=True)

    def _compressor_verify(
        self,
        x: ttnn.Tensor,
        start: ttnn.Tensor,
        positions: ttnn.Tensor,
        row_pos: list[ttnn.Tensor],
        projections: dict,
        m: int,
    ) -> None:
        """Compressor state for an M-row verify block, device-only.

        Every row appends its raw rows (the decode "none" chain), all M rows built at once (slot bias, score add and
        the history concat as in _update_compressor_decode) and written row by row. Then the pooling runs only on candidate rows:
        the rows whose position can close a ratio group, start + o + j * ratio with o = ratio - 1 - start % ratio,
        clamped to the last row. A clamped candidate that does not close its group writes a partial entry for a
        group no row can read yet (valid counts exclude it); the real completion rewrites it later. The candidate's
        projections are gathered by device index, and its raw row is rewritten with the same values before pooling.
        """

        c = self.config
        ratio = self.kind.compress_ratio
        saved_pool = self.__dict__.get("compressor_pool_enabled", None)

        def run(projection_rows: dict, position: ttnn.Tensor, pool: bool) -> None:
            self.compressor_pool_enabled = pool
            self._fused_attention_projections = projection_rows
            self._update_compressor_decode(
                x,
                position,
                self.comp_kv,
                self.comp_gate,
                self.comp_ape,
                self.comp_norm,
                self.cache.compressor_kv,
                self.cache.compressor_score,
                self.cache.compressed,
                c.head_dim,
                rotate_hadamard=False,
            )
            if self.kind.attention is AttentionKind.CSA:
                self._update_compressor_decode(
                    x,
                    position,
                    self.index_comp_kv,
                    self.index_comp_gate,
                    self.index_comp_ape,
                    self.index_comp_norm,
                    self.cache.indexer_compressor_kv,
                    self.cache.indexer_compressor_score,
                    self.cache.indexer,
                    c.index_head_dim,
                    rotate_hadamard=True,
                )

        def gathered(value: ttnn.Tensor, index: ttnn.Tensor) -> ttnn.Tensor:
            if self.compressor_select_matmul:
                return self._select_table_rows(value, index)
            table = ttnn.reshape(value, [m, value.shape[-1]])
            out = ttnn.embedding(ttnn.typecast(index, ttnn.uint32), table, layout=ttnn.TILE_LAYOUT)
            return ttnn.reshape(out, [1, 1, 1, value.shape[-1]])

        row_slots = ttnn.remainder(positions, ratio) if self.verify_select_memo else None

        def raw_rows(kv_weight: ttnn.Tensor, gate_weight: ttnn.Tensor, ape: ttnn.Tensor) -> list[ttnn.Tensor]:
            raw_kv = projections[(id(kv_weight), id(gate_weight))]
            score = projections[(id(gate_weight), id(kv_weight))]
            if self.compressor_select_matmul:
                bias = self._select_table_rows(
                    ape, row_slots if row_slots is not None else ttnn.remainder(positions, ratio)
                )
            else:
                slots = ttnn.typecast(ttnn.remainder(positions, ratio), ttnn.uint32)
                bias = ttnn.embedding(slots, ape, layout=ttnn.TILE_LAYOUT)
            return [raw_kv, ttnn.add(score, ttnn.reshape(bias, [1, 1, m, score.shape[-1]]))]

        if self.verify_select_memo:
            # one one-hot per selection index (and one slot per position) for every table it picks from
            self._select_hot_memo = {}
        try:
            histories = raw_rows(self.comp_kv, self.comp_gate, self.comp_ape)
            if self.kind.attention is AttentionKind.CSA:
                # one shared history tensor: [attention kv | attention score | indexer kv | indexer score]
                histories += raw_rows(self.index_comp_kv, self.index_comp_gate, self.index_comp_ape)
            rows = ttnn.concat(histories, dim=-1)
            if self.verify_rows_update:
                self._paged_update_rows(self.cache.compressor_kv, rows, start)
            else:
                for r in range(m):
                    self._paged_update(
                        self.cache.compressor_kv,
                        self._verify_row(rows, r, m),
                        row_pos[r],
                    )
            first = ttnn.mul(ttnn.add(ttnn.remainder(start, ratio), 1 - ratio), -1)  # ratio - 1 - start % ratio
            for j in range(math.ceil(m / ratio)):
                candidate = ttnn.minimum(ttnn.add(first, j * ratio), m - 1)
                index = ttnn.reshape(candidate, [1, 1])
                run({key: gathered(v, index) for key, v in projections.items()}, ttnn.add(start, candidate), True)
        finally:
            self.__dict__.pop("_select_hot_memo", None)
            if saved_pool is None:
                self.__dict__.pop("compressor_pool_enabled", None)
            else:
                self.compressor_pool_enabled = saved_pool
            self._fused_attention_projections = {}

    def _paged_update_rows(self, cache: PagedTensor, value: ttnn.Tensor, start: ttnn.Tensor) -> None:
        """Write rows 0 .. M - 1 of ``value`` [1, 1, M, W] (bf16 TILE) at positions start .. start + M - 1 of
        a paged cache in one program (kernels/rows_update/), instead of M row extractions + pads + L1 reshards +
        paged_update_cache calls. Each core takes a column chunk; the tile rows of positions start and start + M - 1
        are untilized, patched with every block row that falls in them and tilized back with paged_update_cache's
        own helpers, so the cache bytes match the row-by-row updates."""

        rows, width = int(value.shape[2]), int(value.shape[3])
        blocks, heads, block_size, cache_width = (int(d) for d in cache.tensor.shape)
        if rows > 32 or width != cache_width or width % 32 or block_size % 32 or value.dtype != ttnn.bfloat16:
            raise ValueError(f"multi-row cache write: value {tuple(value.shape)} vs cache {tuple(cache.tensor.shape)}")
        logical_rows = len(cache.host_page_table) * self.config.page_size
        modulo = cache.rows if cache.rows < logical_rows else 0
        wt = width // 32
        wc = max(d for d in range(1, 9) if wt % d == 0)
        page_table_bytes = int(cache.page_table.shape[-1]) * 4
        grid_x = self.mesh_device.compute_with_storage_grid_size().x
        coords = [(k % grid_x, k // grid_x) for k in range(wt // wc)]
        cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in coords])
        cache_page = {ttnn.bfloat8_b: 1088, ttnn.bfloat16: 2048}[cache.tensor.dtype]
        scratch = 64 + page_table_bytes
        cbs = [
            # 2 and 3 alias one buffer, as in paged_update_cache: the writer patches rows in place in the untilized
            # cache (2) and hands the same bytes to the tilize step as 3
            ttnn.CBDescriptor(
                total_size=wc * 2048,
                core_ranges=cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=2, data_format=ttnn.bfloat16, page_size=2048),
                    ttnn.CBFormatDescriptor(buffer_index=3, data_format=ttnn.bfloat16, page_size=2048),
                ],
            )
        ] + [
            ttnn.CBDescriptor(
                total_size=count * size,
                core_ranges=cores,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=size)],
            )
            for index, count, dtype, size in (
                (0, wc, cache.tensor.dtype, cache_page),
                (1, wc, ttnn.bfloat16, 2048),
                (4, wc, ttnn.bfloat16, 2048),
                (7, 1, ttnn.uint32, scratch),
                (8, 1, ttnn.uint32, scratch),
                (16, wc, cache.tensor.dtype, cache_page),
            )
        ]
        ct = [wt, wc, rows, block_size, heads, modulo, page_table_bytes]
        reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for k, (x, y) in enumerate(coords):
            reader_rt[x][y] = [
                value.buffer_address(),
                cache.tensor.buffer_address(),
                cache.page_table.buffer_address(),
                start.buffer_address(),
                k * wc,
            ]
            writer_rt[x][y] = [
                cache.tensor.buffer_address(),
                cache.page_table.buffer_address(),
                start.buffer_address(),
                k * wc,
            ]
            compute_rt[x][y] = []
        kernels = Path(__file__).resolve().parent / "kernels" / "rows_update"
        program = ttnn.ProgramDescriptor(
            kernels=[
                ttnn.KernelDescriptor(
                    kernel_source=str(kernels / "reader_rows_update.cpp"),
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=cores,
                    compile_time_args=ct
                    + ttnn.TensorAccessorArgs(value).get_compile_time_args()
                    + ttnn.TensorAccessorArgs(cache.tensor).get_compile_time_args()
                    + ttnn.TensorAccessorArgs(cache.page_table).get_compile_time_args()
                    + ttnn.TensorAccessorArgs(start).get_compile_time_args(),
                    runtime_args=reader_rt,
                    config=ttnn.ReaderConfigDescriptor(),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=str(kernels / "writer_rows_update.cpp"),
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=cores,
                    compile_time_args=ct
                    + ttnn.TensorAccessorArgs(cache.tensor).get_compile_time_args()
                    + ttnn.TensorAccessorArgs(cache.page_table).get_compile_time_args()
                    + ttnn.TensorAccessorArgs(start).get_compile_time_args(),
                    runtime_args=writer_rt,
                    config=ttnn.WriterConfigDescriptor(),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=str(kernels / "rows_update_compute.cpp"),
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=cores,
                    compile_time_args=[wc],
                    runtime_args=compute_rt,
                    config=ttnn.ComputeConfigDescriptor(),
                ),
            ],
            semaphores=[],
            cbs=cbs,
        )
        ttnn.generic_op([value, cache.tensor, cache.page_table, start], program)

    def _paged_update(self, cache: PagedTensor, value: ttnn.Tensor, position: ttnn.Tensor) -> None:
        """Update full caches normally and bounded histories through TTNN's ring mode."""

        logical_rows = len(cache.host_page_table) * self.config.page_size
        if cache.rows >= logical_rows:
            return super()._paged_update(cache, value, position)

        padded = ttnn.pad(value, [(0, 0), (0, 0), (0, 31), (0, 0)], 0.0)
        shard = ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
            [32, value.shape[-1]],
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        memory = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard)
        padded = ttnn.to_memory_config(padded, memory)
        ttnn.experimental.paged_update_cache(
            cache.tensor,
            padded,
            update_idxs_tensor=position,
            page_table=cache.page_table,
            cache_position_modulo=cache.rows,
        )

    def _project_q_kv(
        self,
        x: ttnn.Tensor,
        positions: ttnn.Tensor,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, tuple[ttnn.Tensor, ttnn.Tensor]]:
        """Run optimized shared projections and form 16 complete local Q heads."""

        c = self.config
        cos, sin = self._rope_values(positions)
        if self._packed_attention_projection is None:
            q_a = self._optimized_linear(x, self.q_a)
            kv_projected = self._optimized_linear(x, self.kv)
            self._fused_attention_projections = {}
            if self.kind.compress_ratio:
                self._fused_attention_projections.update(
                    {
                        (id(self.comp_kv), id(self.comp_gate)): self._optimized_linear(x, self.comp_kv),
                        (id(self.comp_gate), id(self.comp_kv)): self._optimized_linear(x, self.comp_gate),
                    }
                )
            if self.kind.attention is AttentionKind.CSA:
                self._fused_attention_projections.update(
                    {
                        (id(self.index_comp_kv), id(self.index_comp_gate)): self._optimized_linear(
                            x, self.index_comp_kv
                        ),
                        (id(self.index_comp_gate), id(self.index_comp_kv)): self._optimized_linear(
                            x, self.index_comp_gate
                        ),
                    }
                )
                self._fused_index_weight_projection = self._optimized_linear(x, self.index_weight)
        else:
            packed = self._optimized_linear(x, self._packed_attention_projection)
            q_a = self._packed_projection(packed, self.q_a)
            kv_projected = self._packed_projection(packed, self.kv)
            self._fused_attention_projections = {}
            if self.kind.compress_ratio:
                self._fused_attention_projections.update(
                    {
                        key: self._packed_projection(packed, projection)
                        for key, projection in (
                            ((id(self.comp_kv), id(self.comp_gate)), self.comp_kv),
                            ((id(self.comp_gate), id(self.comp_kv)), self.comp_gate),
                        )
                    }
                )
            if self.kind.attention is AttentionKind.CSA:
                self._fused_attention_projections.update(
                    {
                        (id(self.index_comp_kv), id(self.index_comp_gate)): self._packed_projection(
                            packed, self.index_comp_kv
                        ),
                        (id(self.index_comp_gate), id(self.index_comp_kv)): self._packed_projection(
                            packed, self.index_comp_gate
                        ),
                    }
                )
                self._fused_index_weight_projection = self._packed_projection(packed, self.index_weight)

        carry_q_norm = (
            self.optimization_policy.carry_q_norm_shard
            and x.shape[2] == 1
            and self.kind.attention is not AttentionKind.CSA
        )
        qr = self._optimized_rms_norm(
            q_a,
            weight=self.q_a_norm,
            epsilon=c.rms_norm_eps,
            core_cap=(self.optimization_policy.dram_attention_input_cores if carry_q_norm else None),
            return_sharded=carry_q_norm,
        )
        if self._tp_packed_qr_projection is None:
            q_projected = self._optimized_linear(qr, self.q_b)
            self._fused_index_q_projection = (
                self._optimized_linear(qr, self.index_q) if self.kind.attention is AttentionKind.CSA else None
            )
        else:
            packed_qr = self._optimized_linear(qr, self._tp_packed_qr_projection)
            q_projected = ttnn.slice(
                packed_qr,
                [0, 0, 0, 0],
                [packed_qr.shape[0], packed_qr.shape[1], packed_qr.shape[2], self._tp_packed_q_width],
            )
            self._fused_index_q_projection = ttnn.slice(
                packed_qr,
                [0, 0, 0, self._tp_packed_q_width],
                [packed_qr.shape[0], packed_qr.shape[1], packed_qr.shape[2], packed_qr.shape[-1]],
            )
        local_heads = c.num_attention_heads // TP_DEGREE
        q = ttnn.reshape(q_projected, [1, x.shape[2], local_heads, c.head_dim])
        q = ttnn.permute(q, (0, 2, 1, 3))
        q = self._optimized_rms_norm(q, epsilon=c.rms_norm_eps)
        q = self._rope(q, cos, sin)
        kv = self._optimized_rms_norm(kv_projected, weight=self.kv_norm, epsilon=c.rms_norm_eps)
        if getattr(self, "_decode_boundary_capture", None) is not None:
            self._capture_prefill_boundary("kv_rank", kv)
        kv = self._rope(ttnn.reshape(kv, [1, 1, x.shape[2], c.head_dim]), cos, sin)
        return qr, q, kv, (cos, sin)

    def _gathered_attention(
        self,
        q: ttnn.Tensor,
        bank: ttnn.Tensor,
        indices: ttnn.Tensor,
        mask: ttnn.Tensor,
        *,
        identity: bool = False,
    ) -> ttnn.Tensor:
        """Apply sink-aware attention independently to each rank's 16 heads.

        ``identity`` (, decode): the caller guarantees ``indices`` is 0..N-1 over ``bank``'s N rows, so
        when no padding is needed the gather is the bank itself and the embedding (with its full untilize of the
        bank) is skipped; the SDPA input is the same tensor values.
        """

        q_len = q.shape[2]
        padded_k = _round_up(indices.shape[-1], 128)
        if identity and q_len == 1 and padded_k == indices.shape[-1] == bank.shape[0]:
            mask = ttnn.to_layout(mask, ttnn.TILE_LAYOUT)
            gathered = ttnn.reshape(bank, [q_len, 1, indices.shape[-1], self.config.head_dim])
        else:
            if padded_k != indices.shape[-1]:
                padding = padded_k - indices.shape[-1]
                indices = ttnn.pad(indices, [(0, 0), (0, padding)], 0)
                mask = ttnn.pad(mask, [(0, 0), (0, 0), (0, 0), (0, padding)], -65504.0)
            mask = ttnn.to_layout(mask, ttnn.TILE_LAYOUT)
            gathered = ttnn.embedding(ttnn.typecast(indices, ttnn.uint32), bank, layout=ttnn.TILE_LAYOUT)
            gathered = ttnn.reshape(gathered, [q_len, 1, indices.shape[-1], self.config.head_dim])
        if q_len == 1:
            return ttnn.transformer.scaled_dot_product_attention(
                q,
                gathered,
                gathered,
                attn_mask=mask,
                is_causal=False,
                scale=self.config.head_dim**-0.5,
                attention_sink=self.attention_sink,
                program_config=ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=self.mesh_device.compute_with_storage_grid_size(),
                    q_chunk_size=32,
                    k_chunk_size=128,
                    exp_approx_mode=False,
                ),
                compute_kernel_config=self.compute_kernel_config,
            )

        q_batched = ttnn.permute(q, (0, 2, 1, 3))
        logits = ttnn.matmul(
            q_batched,
            ttnn.permute(gathered, (1, 0, 3, 2)),
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute_kernel_config,
        )
        logits = ttnn.add(ttnn.mul(logits, self.config.head_dim**-0.5), ttnn.permute(mask, (1, 0, 2, 3)))
        sink = ttnn.repeat(ttnn.permute(self.attention_sink, (0, 2, 1, 3)), [1, q_len, 1, 1])
        probabilities = ttnn.softmax(
            ttnn.concat([logits, sink], dim=-1),
            dim=-1,
            compute_kernel_config=self.compute_kernel_config,
            numeric_stable=True,
        )
        probabilities = ttnn.slice(
            probabilities,
            [0, 0, 0, 0],
            [1, q_len, self.config.num_attention_heads // TP_DEGREE, indices.shape[-1]],
        )
        out = ttnn.matmul(
            probabilities,
            ttnn.permute(gathered, (1, 0, 2, 3)),
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute_kernel_config,
        )
        return ttnn.permute(out, (0, 2, 1, 3))

    def _indexer_topk_decode(
        self,
        q: ttnn.Tensor,
        weights: ttnn.Tensor,
        index_bank: ttnn.Tensor,
        valid_entries: ttnn.Tensor,
        ordinals: ttnn.Tensor,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """_indexer_topk for one decode query, keeping only row 0 after the score.

        indexer_score_dsa needs 32 query rows (the caller pads q and weights), but the decode consumer only
        reads row 0, so the validity mask, top-k and selected mask run on that row alone. Row 0's scores,
        comparisons and its top-k (one row per core in topk_large_indices) are the same as before.
        Returns (indices, mask), both [1, 1, 1, index_topk], equal to row 0 of _indexer_topk's outputs.
        """

        # upstream indexer_score (#55614) takes the gate weights in projection order [B, 1, Sq, Hi]; a caller still
        # holding the old head-major [B, Hi, Sq, 1] layout is converted here.
        if weights.shape[1] != 1:
            weights = ttnn.permute(weights, (0, 3, 2, 1))
        cfg = ttnn.IndexerScoreProgramConfig(q_chunk_size=32, k_chunk_size=64, head_group_size=0)
        rows = index_bank.shape[2]
        topk = self.config.index_topk
        logits = ttnn.experimental.indexer_score_dsa(
            q,
            index_bank,
            weights,
            chunk_start_idx=rows - 32,
            program_config=cfg,
            seq_shard_axes=[INDEXER_SEQUENCE_AXIS],
        )
        logits = ttnn.slice(logits, [0, 0, 0, 0], [1, 1, 1, rows])
        limit = ttnn.reshape(valid_entries, [1, 1, 1, 1])
        if self.indexer_topk_dynamic and not self.indexer_topk_wide:
            length = ttnn.to_layout(ttnn.typecast(limit, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT)
            if self.indexer_topk_dynamic_mask:
                valid = ttnn.lt(ttnn.reshape(ordinals, [1, 1, 1, rows]), limit)
                logits = ttnn.where(valid, logits, float("-inf"))
            indices = ttnn.experimental.topk_large_indices(
                logits, k=topk, valid_length_tensor=ttnn.reshape(length, [1])
            )
        else:
            valid = ttnn.lt(ttnn.reshape(ordinals, [1, 1, 1, rows]), limit)
            logits = ttnn.where(valid, logits, float("-inf"))
            indices = self._indexer_topk_call(logits, topk, rows)
        selected_limits = ttnn.typecast(ttnn.repeat(limit, [1, 1, 1, topk]), ttnn.uint32)
        selected_mask = ttnn.where(ttnn.lt(indices, selected_limits), 0.0, ATTENTION_MASK_VALUE)
        selected_mask = ttnn.typecast(selected_mask, ttnn.bfloat16)
        return ttnn.slice(indices, [0, 0, 0, 0], [1, 1, 1, topk]), selected_mask

    def _indexer_topk(
        self,
        q: ttnn.Tensor,
        weights: ttnn.Tensor,
        index_bank: ttnn.Tensor,
        valid_entries: ttnn.Tensor,
        ordinals: ttnn.Tensor | None = None,
        length: ttnn.Tensor | None = None,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Score replicated CSA inputs without treating TP ranks as sequence ranks.

        length (indexer_topk_dynamic): a 1-element UINT32 device tensor >= every row's valid entry count; the top-k
        scans only that prefix of the bank (the per-row -inf mask still applies inside it).
        """

        # upstream indexer_score (#55614) takes the gate weights in projection order [B, 1, Sq, Hi]; a caller still
        # holding the old head-major [B, Hi, Sq, 1] layout is converted here.
        if weights.shape[1] != 1:
            weights = ttnn.permute(weights, (0, 3, 2, 1))
        cfg = ttnn.IndexerScoreProgramConfig(q_chunk_size=32, k_chunk_size=64, head_group_size=0)
        q_len = q.shape[2]
        rows = max(q_len, 32)  # query rows after padding (128-query prefill chunks)
        if q_len < 32:
            q = ttnn.pad(q, [(0, 0), (0, 0), (0, 32 - q_len), (0, 0)], 0.0)
            weights = ttnn.pad(weights, [(0, 0), (0, 0), (0, 32 - q_len), (0, 0)], 0.0)
            valid_entries = ttnn.pad(valid_entries, [(0, 32 - q_len), (0, 0)], 0)
        logits = ttnn.experimental.indexer_score_dsa(
            q,
            index_bank,
            weights,
            chunk_start_idx=index_bank.shape[2] - rows,
            program_config=cfg,
            # Axis 0 is tensor parallel, while axis 1 has extent one.  Naming
            # the latter as SP keeps every replicated TP rank at causal rank
            # zero instead of applying the op's default whole-mesh Sq stride.
            seq_shard_axes=[INDEXER_SEQUENCE_AXIS],
        )
        if ordinals is None:
            ordinals = ttnn.arange(0, index_bank.shape[2], 1, device=self.mesh_device, dtype=ttnn.int32)
            ordinals = ttnn.reshape(ordinals, [1, index_bank.shape[2]])
        ordinals = ttnn.reshape(ordinals, [1, 1, 1, index_bank.shape[2]])
        limits = ttnn.reshape(valid_entries, [1, 1, rows, 1])
        valid = ttnn.lt(ordinals, limits)
        logits = ttnn.where(valid, logits, float("-inf"))
        if length is not None:
            indices = ttnn.experimental.topk_large_indices(logits, k=self.config.index_topk, valid_length_tensor=length)
        else:
            indices = self._indexer_topk_call(logits, self.config.index_topk, index_bank.shape[2])
        selected_limits = ttnn.repeat(
            ttnn.reshape(valid_entries, [1, 1, rows, 1]),
            [1, 1, 1, self.config.index_topk],
        )
        selected_limits = ttnn.typecast(selected_limits, ttnn.uint32)
        selected_valid = ttnn.lt(indices, selected_limits)
        selected_mask = ttnn.where(selected_valid, 0.0, ATTENTION_MASK_VALUE)
        selected_mask = ttnn.typecast(selected_mask, ttnn.bfloat16)
        selected_mask = ttnn.permute(selected_mask, (2, 1, 0, 3))
        selected_mask = ttnn.slice(
            selected_mask,
            [0, 0, 0, 0],
            [q_len, 1, 1, self.config.index_topk],
        )
        return (
            ttnn.slice(indices, [0, 0, 0, 0], [1, 1, q_len, self.config.index_topk]),
            selected_mask,
        )

    def _attention_prefill(self, x: ttnn.Tensor, plan: PrefillPlan) -> ttnn.Tensor:
        """Preserve optimized paged prefill with a rank-local Q-head slice."""

        c = self.config
        qr, q, kv, rope = self._project_q_kv(x, plan.position_ids)
        self._capture_prefill_boundary("query_rank", qr)
        self._capture_prefill_boundary("query_heads", q)
        self._capture_prefill_boundary("kv_heads", kv)
        compressed = index_entries = index_bank = None
        if self.kind.compress_ratio and plan.seq_len >= self.kind.compress_ratio:
            compressed = self._compress_prefill(
                x,
                plan.position_ids,
                self.comp_kv,
                self.comp_gate,
                self.comp_ape,
                self.comp_norm,
                c.head_dim,
                rotate_hadamard=False,
            )
        if self.kind.attention is AttentionKind.CSA and plan.seq_len >= self.kind.compress_ratio:
            index_entries = self._compress_prefill(
                x,
                plan.position_ids,
                self.index_comp_kv,
                self.index_comp_gate,
                self.index_comp_ape,
                self.index_comp_norm,
                c.index_head_dim,
                rotate_hadamard=True,
            )
            score_rows = _round_up(max(c.index_topk, index_entries.shape[2]), 32)
            score_pad = score_rows - index_entries.shape[2] + 32
            guard = ttnn.zeros(
                [1, 1, score_pad, c.index_head_dim],
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
            )
            index_bank = ttnn.concat([index_entries, guard], dim=2)
        self._fill_prefill_cache(kv, compressed, index_entries, plan)
        if self.kind.compress_ratio:
            self._fill_raw_prefill_state(
                x,
                plan.position_ids,
                self.comp_kv,
                self.comp_gate,
                self.comp_ape,
                self.cache.compressor_kv,
                self.cache.compressor_score,
                plan,
            )
        if self.kind.attention is AttentionKind.CSA:
            self._fill_raw_prefill_state(
                x,
                plan.position_ids,
                self.index_comp_kv,
                self.index_comp_gate,
                self.index_comp_ape,
                self.cache.indexer_compressor_kv,
                self.cache.indexer_compressor_score,
                plan,
            )

        bank = ttnn.reshape(kv, [plan.seq_len, c.head_dim])
        if compressed is not None:
            bank = ttnn.concat([bank, ttnn.reshape(compressed, [compressed.shape[2], c.head_dim])], dim=0)
        outputs = []
        local_heads = c.num_attention_heads // TP_DEGREE
        for chunk in plan.chunks:
            q_chunk = ttnn.slice(q, [0, 0, chunk.start, 0], [1, local_heads, chunk.end, c.head_dim])
            indices = chunk.local_indices
            mask = chunk.local_mask
            if self.kind.attention is AttentionKind.HCA:
                if chunk.compressed_indices is not None:
                    indices = ttnn.concat([indices, chunk.compressed_indices], dim=-1)
                    mask = ttnn.concat([mask, chunk.compressed_mask], dim=-1)
            elif self.kind.attention is AttentionKind.CSA and index_bank is not None:
                if self._fused_index_q_projection is None:
                    raise RuntimeError("CSA index query escaped the fused QR projection")
                idx_q = ttnn.slice(
                    self._fused_index_q_projection,
                    [0, 0, chunk.start, 0],
                    [1, 1, chunk.end, c.index_n_heads * c.index_head_dim],
                )
                idx_q = ttnn.reshape(idx_q, [1, chunk.end - chunk.start, c.index_n_heads, c.index_head_dim])
                idx_q = ttnn.permute(idx_q, (0, 2, 1, 3))
                cos = ttnn.slice(rope[0], [0, 0, chunk.start, 0], [1, 1, chunk.end, c.rope_head_dim])
                sin = ttnn.slice(rope[1], [0, 0, chunk.start, 0], [1, 1, chunk.end, c.rope_head_dim])
                idx_q = self._rope(idx_q, cos, sin)
                idx_q = ttnn.matmul(idx_q, self.index_hadamard)
                idx_w = ttnn.slice(
                    self._fused_index_weight_projection,
                    [0, 0, chunk.start, 0],
                    [1, 1, chunk.end, c.index_n_heads],
                )
                topk, topk_mask = self._indexer_topk(idx_q, idx_w, index_bank, chunk.compressed_counts)
                topk = ttnn.reshape(topk, [chunk.end - chunk.start, c.index_topk])
                flat_topk_mask = ttnn.reshape(topk_mask, [chunk.end - chunk.start, c.index_topk])
                topk = ttnn.typecast(topk, ttnn.int32)
                topk_valid = ttnn.typecast(ttnn.eq(flat_topk_mask, 0.0), ttnn.int32)
                topk = ttnn.add(ttnn.mul(topk, topk_valid), plan.seq_len)
                indices = ttnn.concat([indices, topk], dim=-1)
                mask = ttnn.concat([mask, topk_mask], dim=-1)
            outputs.append(self._gathered_attention(q_chunk, bank, indices, mask))
        attention = outputs[0] if len(outputs) == 1 else ttnn.concat(outputs, dim=2)
        self._capture_prefill_boundary("attention_heads", attention)
        attention = self._rope(attention, *rope, inverse=True)
        self._capture_prefill_boundary("attention_heads_inverse_rope", attention)
        return self._grouped_output_projection(attention, plan.seq_len)

    def _page_table_range(self, cache: PagedTensor, first_page: int, last_page: int) -> ttnn.Tensor:
        """Logical pages [first_page, last_page) of ``cache``'s page table as a [1, n] int32 device tensor.

        With streaming_host_page_tables the range is uploaded from the host copy of the table (same values): slicing
        the device table compiles a slice program per distinct (start, length), so every new block position of a
        long prompt compiled programs (and, in a server, released and recaptured the traces)."""

        if self.streaming_host_page_tables:
            return self._to_tt(
                torch.tensor(cache.host_page_table[first_page:last_page], dtype=torch.int32).reshape(1, -1),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
        return ttnn.slice(cache.page_table, [0, first_page], [1, last_page])

    def _streaming_fill_page_aligned(
        self,
        cache: PagedTensor,
        value: ttnn.Tensor,
        logical_start: int,
    ) -> None:
        """Fill a page-aligned logical cache interval through its page table."""

        page_size = self.config.page_size
        row_count = int(value.shape[2])
        if logical_start % page_size or row_count % page_size:
            raise ValueError(f"streaming cache fill must be page aligned, got start={logical_start}, rows={row_count}")
        first_page = logical_start // page_size
        page_count = row_count // page_size
        physical_pages = cache.host_page_table[first_page : first_page + page_count]
        # Blackhole paged_fill_cache silently leaves stale pages when one call
        # crosses a cyclic physical wrap (for example [3, 4, 0, 1]).  Keep
        # each call monotonically contiguous while retaining page-sized bulk
        # fills and the exact logical page table.
        for run_start, run_end in _contiguous_page_runs(physical_pages):
            page_table = self._page_table_range(cache, first_page + run_start, first_page + run_end)
            run_value = (
                value
                if run_start == 0 and run_end == page_count
                else ttnn.slice(
                    value,
                    [0, 0, run_start * page_size, 0],
                    [value.shape[0], value.shape[1], run_end * page_size, value.shape[3]],
                )
            )
            self._paged_fill_cache(cache, run_value, page_table)
            if run_value is not value:
                ttnn.deallocate(run_value)

    def _streaming_raw_plan(
        self,
        input_ids: ttnn.Tensor,
        position_ids: ttnn.Tensor,
        block_start: int,
        valid_seq_len: int,
    ) -> PrefillPlan:
        """Describe the page containing the final real raw-compressor rows."""

        seq_len = int(input_ids.shape[-1])
        if not 1 <= valid_seq_len <= seq_len:
            raise ValueError(f"valid streaming length {valid_seq_len} outside [1, {seq_len}]")
        page_size = self.config.page_size
        if self.streaming_raw_full_window:
            # Decode completes a compressed entry from the cached raw rows of its whole window (ratio rows for
            # HCA, 2 * ratio for CSA), exactly like the non-streaming plan fills them; the earlier rows of
            # that window that precede this block were written by the previous block's last page.
            raw_rows = (
                2 * self.kind.compress_ratio if self.kind.attention is AttentionKind.CSA else self.kind.compress_ratio
            )
            fill_start = (max(0, valid_seq_len - raw_rows) // page_size) * page_size
            fill_length = _round_up(valid_seq_len, page_size) - fill_start
        else:
            fill_length = page_size
            fill_start = ((valid_seq_len - 1) // page_size) * page_size
        logical_start = block_start + fill_start
        if logical_start % page_size:
            raise ValueError(f"raw compressor fill start {logical_start} is not page aligned")
        first_page = logical_start // page_size
        page_count = fill_length // page_size
        compressor_page_table = self._page_table_range(self.cache.compressor_kv, first_page, first_page + page_count)
        local_page_table = self._page_table_range(
            self.cache.local, block_start // page_size, (block_start + seq_len) // page_size
        )
        return PrefillPlan(
            valid_seq_len,
            position_ids,
            input_ids,
            (),
            local_page_table,
            0,
            seq_len,
            None,
            compressor_page_table,
            fill_start,
            fill_length,
        )

    def _streaming_csa_boundary_entries(
        self,
        block_start: int,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor] | None:
        """Rebuild CSA's first entry from the previous and current four-token halves."""

        if self.kind.attention is not AttentionKind.CSA or block_start == 0:
            return None
        projections = getattr(self, "_fused_prefill_projections", {})
        try:
            attention_kv, attention_score = projections[(id(self.comp_kv), id(self.comp_gate))]
            index_kv, index_score = projections[(id(self.index_comp_kv), id(self.index_comp_gate))]
        except KeyError as exc:
            raise RuntimeError("CSA boundary repair escaped the fused prefill projections") from exc

        previous_positions = self._to_tt(
            torch.arange(block_start - 4, block_start, dtype=torch.int32).reshape(1, -1),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        previous = self._cache_rows(self.cache.compressor_kv, previous_positions)
        current = ttnn.concat(
            [
                ttnn.slice(attention_kv, [0, 0, 0, 0], [1, 1, 4, attention_kv.shape[-1]]),
                ttnn.slice(attention_score, [0, 0, 0, 0], [1, 1, 4, attention_score.shape[-1]]),
                ttnn.slice(index_kv, [0, 0, 0, 0], [1, 1, 4, index_kv.shape[-1]]),
                ttnn.slice(index_score, [0, 0, 0, 0], [1, 1, 4, index_score.shape[-1]]),
            ],
            dim=-1,
        )
        current = ttnn.reshape(current, [1, 4, current.shape[-1]])
        if current.dtype != previous.dtype:
            current = ttnn.typecast(current, previous.dtype)
        raw = ttnn.concat([previous, current], dim=1)
        entry_position = self._to_tt(
            torch.tensor([block_start], dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        cos, sin = self._rope_values(entry_position)

        def pool(
            offset: int,
            kv_width: int,
            score_width: int,
            width: int,
            norm: ttnn.Tensor,
            *,
            rotate_hadamard: bool,
        ) -> tuple[ttnn.Tensor, int]:
            history_kv = ttnn.slice(raw, [0, 0, offset], [1, 8, offset + kv_width])
            offset += kv_width
            history_score = ttnn.slice(raw, [0, 0, offset], [1, 8, offset + score_width])
            offset += score_width
            previous_kv = ttnn.slice(history_kv, [0, 0, 0], [1, 4, width])
            current_kv = ttnn.slice(history_kv, [0, 4, width], [1, 8, 2 * width])
            previous_score = ttnn.slice(history_score, [0, 0, 0], [1, 4, width])
            current_score = ttnn.slice(history_score, [0, 4, width], [1, 8, 2 * width])
            history_kv = ttnn.concat([previous_kv, current_kv], dim=1)
            history_score = ttnn.concat([previous_score, current_score], dim=1)
            pooled = ttnn.sum(
                ttnn.mul(history_kv, ttnn.softmax(history_score, dim=1)),
                dim=1,
                keepdim=True,
            )
            pooled = ttnn.rms_norm(
                ttnn.reshape(pooled, [1, 1, 1, width]),
                weight=norm,
                epsilon=self.config.rms_norm_eps,
            )
            pooled = self._rope(pooled, cos, sin)
            if rotate_hadamard:
                pooled = ttnn.matmul(pooled, self.index_hadamard)
            return pooled, offset

        attention_entry, offset = pool(
            0,
            self.comp_kv.shape[-1],
            self.comp_gate.shape[-1],
            self.config.head_dim,
            self.comp_norm,
            rotate_hadamard=False,
        )
        index_entry, offset = pool(
            offset,
            self.index_comp_kv.shape[-1],
            self.index_comp_gate.shape[-1],
            self.config.index_head_dim,
            self.index_comp_norm,
            rotate_hadamard=True,
        )
        if offset != raw.shape[-1]:
            raise RuntimeError(f"CSA boundary repair consumed {offset} of {raw.shape[-1]} packed columns")
        entry = self._to_tt(
            torch.tensor([block_start // self.kind.compress_ratio], dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        return attention_entry, index_entry, entry

    def _attention_from_streaming_rows(
        self,
        q: ttnn.Tensor,
        gathered: ttnn.Tensor,
        mask: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """Apply sink-aware attention to a query-specific gathered KV tensor."""

        q_len = int(q.shape[2])
        padded_k = _round_up(int(gathered.shape[2]), 128)
        if padded_k != gathered.shape[2]:
            padding = padded_k - gathered.shape[2]
            gathered = ttnn.pad(gathered, [(0, 0), (0, 0), (0, padding), (0, 0)], 0.0)
            mask = ttnn.pad(mask, [(0, 0), (0, 0), (0, 0), (0, padding)], ATTENTION_MASK_VALUE)
        mask = ttnn.to_layout(mask, ttnn.TILE_LAYOUT)
        q_batched = ttnn.permute(q, (0, 2, 1, 3))
        logits = ttnn.matmul(
            q_batched,
            ttnn.permute(gathered, (1, 0, 3, 2)),
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute_kernel_config,
        )
        logits = ttnn.add(
            ttnn.mul(logits, self.config.head_dim**-0.5),
            ttnn.permute(mask, (1, 0, 2, 3)),
        )
        sink = ttnn.repeat(ttnn.permute(self.attention_sink, (0, 2, 1, 3)), [1, q_len, 1, 1])
        probabilities = ttnn.softmax(
            ttnn.concat([logits, sink], dim=-1),
            dim=-1,
            compute_kernel_config=self.compute_kernel_config,
            numeric_stable=True,
        )
        probabilities = ttnn.slice(
            probabilities,
            [0, 0, 0, 0],
            [1, q_len, self.config.num_attention_heads // TP_DEGREE, padded_k],
        )
        output = ttnn.matmul(
            probabilities,
            ttnn.permute(gathered, (1, 0, 2, 3)),
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute_kernel_config,
        )
        return ttnn.permute(output, (0, 2, 1, 3))

    def _attention_streaming_block(
        self,
        x: ttnn.Tensor,
        input_ids: ttnn.Tensor,
        position_ids: ttnn.Tensor,
        block_start: int,
        valid_seq_len: int,
    ) -> ttnn.Tensor:
        """Run an exact 128-token block with bounded, cache-backed KV banks."""

        c = self.config
        seq_len = int(input_ids.shape[-1])
        if not 1 <= valid_seq_len <= seq_len:
            raise ValueError(f"valid streaming length {valid_seq_len} outside [1, {seq_len}]")
        _, q, kv, rope = self._project_q_kv(x, position_ids)
        streaming_capture = getattr(self, "_streaming_kv_capture", None)

        def capture_streaming_rows(name: str, value: ttnn.Tensor) -> None:
            if streaming_capture is None or block_start != streaming_capture[0]:
                return
            local_tensors = ttnn.get_device_tensors(value) or (value,)
            host = ttnn.to_torch(local_tensors[0])
            streaming_capture[1][name] = host.reshape(-1, c.head_dim).detach().float().cpu().clone()

        capture_streaming_rows("projected_bf16", kv)
        compressed = index_entries = None
        if self.kind.compress_ratio:
            compressed = self._compress_prefill(
                x,
                position_ids,
                self.comp_kv,
                self.comp_gate,
                self.comp_ape,
                self.comp_norm,
                c.head_dim,
                rotate_hadamard=False,
            )
        if self.kind.attention is AttentionKind.CSA:
            index_entries = self._compress_prefill(
                x,
                position_ids,
                self.index_comp_kv,
                self.index_comp_gate,
                self.index_comp_ape,
                self.index_comp_norm,
                c.index_head_dim,
                rotate_hadamard=True,
            )
        boundary_entries = self._streaming_csa_boundary_entries(block_start)

        # Save the previous 128-row tile-aligned window before the cyclic local
        # cache write can reuse their physical pages.  Current-block KV is
        # explicitly converted once to the same cache dtype used by the fill,
        # so attention and later decode observe the identical quantized rows.
        previous_count = min(block_start, c.sliding_window)
        previous_local = None
        if previous_count:
            previous_positions = self._to_tt(
                torch.arange(block_start - previous_count, block_start, dtype=torch.int32).reshape(1, -1),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            previous_local = self._cache_rows(self.cache.local, previous_positions)
            previous_local = ttnn.reshape(previous_local, [1, 1, previous_count, c.head_dim])
        cached_kv = (
            kv if kv.dtype == self.cache.local.tensor.dtype else ttnn.typecast(kv, self.cache.local.tensor.dtype)
        )
        capture_streaming_rows("cache_dtype", cached_kv)
        current_local = cached_kv
        if previous_local is not None and current_local.dtype != previous_local.dtype:
            current_local = ttnn.typecast(current_local, previous_local.dtype)
        local_bank = current_local if previous_local is None else ttnn.concat([previous_local, current_local], dim=2)
        local_bank = ttnn.reshape(local_bank, [previous_count + seq_len, c.head_dim])
        local_bank_start = block_start - previous_count
        local_fill_rows = _round_up(valid_seq_len, c.page_size)
        local_fill = ttnn.slice(
            cached_kv,
            [0, 0, 0, 0],
            [cached_kv.shape[0], cached_kv.shape[1], valid_seq_len, cached_kv.shape[3]],
        )
        if local_fill_rows != valid_seq_len:
            local_fill = ttnn.pad(
                local_fill,
                [(0, 0), (0, 0), (0, local_fill_rows - valid_seq_len), (0, 0)],
                0.0,
            )
        self._streaming_fill_page_aligned(self.cache.local, local_fill, block_start)
        if streaming_capture is not None and block_start == streaming_capture[0]:
            capture_positions = self._to_tt(
                torch.arange(block_start, block_start + seq_len, dtype=torch.int32).reshape(1, -1),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            readback = self._cache_rows(self.cache.local, capture_positions)
            capture_streaming_rows("readback", readback)
            ttnn.deallocate(readback)
            ttnn.deallocate(capture_positions)
        if self.kind.attention is AttentionKind.HCA:
            # One compressed entry per ratio rows: a 128-token block writes one, a 512-token block four.
            for index in range(seq_len // self.kind.compress_ratio):
                entry = self._to_tt(
                    torch.tensor([block_start // self.kind.compress_ratio + index], dtype=torch.int32),
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )
                value = (
                    compressed
                    if seq_len == self.kind.compress_ratio
                    else ttnn.slice(compressed, [0, 0, index, 0], [1, 1, index + 1, compressed.shape[-1]])
                )
                self._paged_update(self.cache.compressed, value, entry)
        elif self.kind.attention is AttentionKind.CSA:
            entry_start = block_start // self.kind.compress_ratio
            self._streaming_fill_page_aligned(self.cache.compressed, compressed, entry_start)
            self._streaming_fill_page_aligned(self.cache.indexer, index_entries, entry_start)

        if self.kind.compress_ratio:
            raw_plan = self._streaming_raw_plan(input_ids, position_ids, block_start, valid_seq_len)
            self._fill_raw_prefill_state(
                x,
                position_ids,
                self.comp_kv,
                self.comp_gate,
                self.comp_ape,
                self.cache.compressor_kv,
                self.cache.compressor_score,
                raw_plan,
            )
            if self.kind.attention is AttentionKind.CSA:
                self._fill_raw_prefill_state(
                    x,
                    position_ids,
                    self.index_comp_kv,
                    self.index_comp_gate,
                    self.index_comp_ape,
                    self.cache.indexer_compressor_kv,
                    self.cache.indexer_compressor_score,
                    raw_plan,
                )
        if boundary_entries is not None and valid_seq_len >= self.kind.compress_ratio:
            attention_entry, index_entry, entry = boundary_entries
            self._paged_update(self.cache.compressed, attention_entry, entry)
            self._paged_update(self.cache.indexer, index_entry, entry)

        hca_bank = index_bank = index_ordinals = None
        total_entries = 0
        if self.kind.attention is AttentionKind.HCA:
            total_entries = (block_start + seq_len) // self.kind.compress_ratio
            logical = self._to_tt(
                torch.arange(total_entries, dtype=torch.int32).reshape(1, -1),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            hca_bank = self._cache_rows(self.cache.compressed, logical)
        elif self.kind.attention is AttentionKind.CSA:
            total_entries = (block_start + seq_len) // self.kind.compress_ratio
            score_rows = _round_up(max(c.index_topk, total_entries), 32)
            stored_rows = min(score_rows, self.cache.indexer.rows)
            physical_index = ttnn.reshape(
                self.cache.indexer.tensor,
                [1, 1, self.cache.indexer.tensor.shape[0] * c.page_size, c.index_head_dim],
            )
            physical_index = ttnn.slice(
                physical_index,
                [0, 0, 0, 0],
                [1, 1, stored_rows, c.index_head_dim],
            )
            # Zero guard rows past the scored entries: indexer_score_dsa's causal start sits query_chunk rows before
            # the bank end, so the guard must hold at least one query chunk (lets the chunk be 128).
            query_chunk = int(getattr(self, "prefill_query_chunk", 32))
            guard_rows = score_rows - stored_rows + query_chunk
            if guard_rows <= self.decode_index_guard.shape[2]:
                guard = ttnn.slice(
                    self.decode_index_guard,
                    [0, 0, 0, 0],
                    [1, 1, guard_rows, c.index_head_dim],
                )
            else:
                guard = ttnn.zeros(
                    [1, 1, guard_rows, c.index_head_dim],
                    dtype=self.decode_index_guard.dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.mesh_device,
                )
            index_bank = ttnn.concat([physical_index, guard], dim=2)
            if score_rows + query_chunk <= self.decode_index_ordinals.shape[-1]:
                index_ordinals = ttnn.slice(
                    self.decode_index_ordinals,
                    [0, 0],
                    [1, score_rows + query_chunk],
                )
            else:
                index_ordinals = ttnn.reshape(
                    ttnn.arange(0, score_rows + query_chunk, 1, device=self.mesh_device, dtype=ttnn.int32),
                    [1, score_rows + query_chunk],
                )

        outputs = []
        local_heads = c.num_attention_heads // TP_DEGREE
        offsets = torch.arange(1 - c.sliding_window, 1, dtype=torch.int32).reshape(1, -1)
        csa_table = None
        if self.kind.attention is AttentionKind.CSA and getattr(self, "prefill_csa_untilize_once", False):
            # every q chunk below gathers 16 x 1024 selected rows from cache.compressed, and each
            # embedding untilized the whole 32768-row TILE table again. Nothing in the loop writes the table, so
            # it is converted once here; ttnn.embedding performs the same to_layout itself, so the rows match.
            compressed = self.cache.compressed
            csa_table = ttnn.to_layout(
                ttnn.reshape(compressed.tensor, [compressed.rows, compressed.width]),
                ttnn.ROW_MAJOR_LAYOUT,
            )
        qc = int(getattr(self, "prefill_query_chunk", 32))  # queries per attention step (32 or 128)
        if seq_len % qc:
            raise ValueError(f"prefill_query_chunk {qc} must divide the block of {seq_len}")
        for chunk_start in range(0, seq_len, qc):
            chunk_end = chunk_start + qc
            q_chunk = ttnn.slice(q, [0, 0, chunk_start, 0], [1, local_heads, chunk_end, c.head_dim])
            q_positions = torch.arange(
                block_start + chunk_start,
                block_start + chunk_end,
                dtype=torch.int32,
            ).reshape(-1, 1)
            local_indices = q_positions + offsets
            local_valid = local_indices >= 0
            local_indices = local_indices.clamp(min=0)
            local_indices = local_indices - local_bank_start
            local_indices_tt = self._to_tt(
                local_indices,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            local_rows = ttnn.embedding(
                ttnn.typecast(local_indices_tt, ttnn.uint32),
                local_bank,
                layout=ttnn.TILE_LAYOUT,
            )
            gathered = ttnn.reshape(local_rows, [qc, 1, c.sliding_window, c.head_dim])
            local_mask = torch.zeros_like(local_indices, dtype=torch.float32).masked_fill(
                ~local_valid,
                ATTENTION_MASK_VALUE,
            )
            mask = self._to_tt(
                local_mask.reshape(qc, 1, 1, c.sliding_window),
                dtype=ttnn.bfloat16,
            )

            if self.kind.attention is AttentionKind.HCA:
                repeated_hca = ttnn.repeat(
                    ttnn.reshape(hca_bank, [1, 1, total_entries, c.head_dim]),
                    [qc, 1, 1, 1],
                )
                counts = torch.div(
                    q_positions.squeeze(1) + 1,
                    self.kind.compress_ratio,
                    rounding_mode="floor",
                )
                ordinals = torch.arange(total_entries, dtype=torch.int32).reshape(1, -1)
                compressed_valid = ordinals < counts.reshape(-1, 1)
                compressed_mask = torch.zeros_like(compressed_valid, dtype=torch.float32).masked_fill(
                    ~compressed_valid,
                    ATTENTION_MASK_VALUE,
                )
                gathered = ttnn.concat([gathered, repeated_hca], dim=2)
                mask = ttnn.concat(
                    [
                        mask,
                        self._to_tt(
                            compressed_mask.reshape(qc, 1, 1, total_entries),
                            dtype=ttnn.bfloat16,
                        ),
                    ],
                    dim=-1,
                )
            elif self.kind.attention is AttentionKind.CSA:
                idx_q = ttnn.slice(
                    self._fused_index_q_projection,
                    [0, 0, chunk_start, 0],
                    [1, 1, chunk_end, c.index_n_heads * c.index_head_dim],
                )
                idx_q = ttnn.reshape(idx_q, [1, qc, c.index_n_heads, c.index_head_dim])
                idx_q = ttnn.permute(idx_q, (0, 2, 1, 3))
                cos = ttnn.slice(rope[0], [0, 0, chunk_start, 0], [1, 1, chunk_end, c.rope_head_dim])
                sin = ttnn.slice(rope[1], [0, 0, chunk_start, 0], [1, 1, chunk_end, c.rope_head_dim])
                idx_q = self._rope(idx_q, cos, sin)
                idx_q = ttnn.matmul(idx_q, self.index_hadamard)
                idx_w = ttnn.slice(
                    self._fused_index_weight_projection,
                    [0, 0, chunk_start, 0],
                    [1, 1, chunk_end, c.index_n_heads],
                )
                counts = torch.div(
                    q_positions.squeeze(1) + 1,
                    self.kind.compress_ratio,
                    rounding_mode="floor",
                )
                counts_tt = self._to_tt(
                    counts.reshape(-1, 1),
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )
                topk, topk_mask = self._indexer_topk(
                    idx_q,
                    idx_w,
                    index_bank,
                    counts_tt,
                    ordinals=index_ordinals,
                )
                topk = ttnn.reshape(topk, [qc, c.index_topk])
                flat_topk_mask = ttnn.reshape(topk_mask, [qc, c.index_topk])
                topk = ttnn.typecast(topk, ttnn.int32)
                topk_valid = ttnn.typecast(ttnn.eq(flat_topk_mask, 0.0), ttnn.int32)
                topk = ttnn.mul(topk, topk_valid)
                flat_topk = ttnn.reshape(topk, [1, qc * c.index_topk])
                if (
                    csa_table is not None
                    and getattr(self, "prefill_csa_gather_once", False)
                    and self._row_lut_kind(self.cache.compressed) == "identity"
                ):
                    # on an identity page table the logical rows are the physical rows, so the chunk's
                    # 32 x 512 selected rows come out of the row-major table in one embedding (same rows as the
                    # 16 LUT-mapped 1024-row gathers below, ~64 ops -> 2).
                    selected = ttnn.embedding(ttnn.typecast(flat_topk, ttnn.uint32), csa_table, layout=ttnn.TILE_LAYOUT)
                else:
                    selected_parts = []
                    for gather_start in range(0, flat_topk.shape[-1], HCA_CACHE_GATHER_CHUNK_ROWS):
                        gather_end = min(gather_start + HCA_CACHE_GATHER_CHUNK_ROWS, flat_topk.shape[-1])
                        selected_parts.append(
                            self._cache_rows(
                                self.cache.compressed,
                                ttnn.slice(flat_topk, [0, gather_start], [1, gather_end]),
                                table=csa_table,
                            )
                        )
                    selected = selected_parts[0] if len(selected_parts) == 1 else ttnn.concat(selected_parts, dim=1)
                selected = ttnn.reshape(selected, [qc, 1, c.index_topk, c.head_dim])
                gathered = ttnn.concat([gathered, selected], dim=2)
                mask = ttnn.concat([mask, topk_mask], dim=-1)
            outputs.append(self._attention_from_streaming_rows(q_chunk, gathered, mask))

        if csa_table is not None:
            ttnn.deallocate(csa_table)
        attention = outputs[0] if len(outputs) == 1 else ttnn.concat(outputs, dim=2)
        attention = self._rope(attention, *rope, inverse=True)
        return self._grouped_output_projection(attention, seq_len)

    def prefill_fixed_block_ok(self, block_start: int, seq_len: int, valid_seq_len: int) -> bool:
        """Whether a streaming block can take the fixed-shape path (a full 128-row block after the first
        sliding window: every row's local window and the CSA boundary history lie in earlier blocks)."""

        return (
            seq_len == 128
            and valid_seq_len == seq_len
            and block_start >= self.config.sliding_window
            and block_start % 128 == 0
            and self.streaming_raw_full_window
        )

    def _prefill_raw_fill_window(self, seq_len: int) -> tuple[int, int]:
        """(fill_start, fill_length) of a full block's raw compressor fill, as _streaming_raw_plan computes it."""

        raw_rows = (
            2 * self.kind.compress_ratio if self.kind.attention is AttentionKind.CSA else self.kind.compress_ratio
        )
        page = self.config.page_size
        fill_start = (max(0, seq_len - raw_rows) // page) * page
        return fill_start, _round_up(seq_len, page) - fill_start

    def prefill_block_host_inputs(self, block_start: int, seq_len: int) -> dict[str, torch.Tensor]:
        """Host values of PrefillBlockInputs for one middle block of this layer (int32)."""

        c = self.config
        page = c.page_size
        window = c.sliding_window
        i32 = torch.int32
        first = block_start // page
        out = {
            "prev_pos": torch.arange(block_start - window, block_start, dtype=i32).reshape(1, -1),
            "local_pages": torch.tensor(
                [self.cache.local.host_page_table[first + k] for k in range(seq_len // page)], dtype=i32
            ).reshape(1, -1),
        }
        if self.kind.compress_ratio:
            fill_start, fill_length = self._prefill_raw_fill_window(seq_len)
            raw_first = (block_start + fill_start) // page
            out["raw_pages"] = torch.tensor(
                [self.cache.compressor_kv.host_page_table[raw_first + k] for k in range(fill_length // page)],
                dtype=i32,
            ).reshape(1, -1)
        if self.kind.attention is AttentionKind.HCA:
            out["entry"] = torch.tensor([block_start // self.kind.compress_ratio], dtype=i32)
        elif self.kind.attention is AttentionKind.CSA:
            entry_start = block_start // self.kind.compress_ratio
            out["entry"] = torch.tensor([entry_start], dtype=i32)
            out["comp_pages"] = torch.tensor([[self.cache.compressed.host_page_table[entry_start // page]]], dtype=i32)
            out["index_pages"] = torch.tensor([[self.cache.indexer.host_page_table[entry_start // page]]], dtype=i32)
            out["bnd_prev_pos"] = torch.arange(block_start - 4, block_start, dtype=i32).reshape(1, -1)
            out["bnd_pos"] = torch.tensor([block_start], dtype=i32)
        return out

    def prefill_block_device_inputs(self, host: Mapping[str, torch.Tensor]) -> PrefillBlockInputs:
        """Upload prefill_block_host_inputs as fresh device tensors (eager use; a trace keeps persistent ones)."""

        def up(name: str) -> ttnn.Tensor | None:
            if name not in host:
                return None
            return self._to_tt(host[name], dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)

        return PrefillBlockInputs(
            prev_pos=up("prev_pos"),
            local_pages=up("local_pages"),
            raw_pages=up("raw_pages"),
            entry=up("entry"),
            comp_pages=up("comp_pages"),
            index_pages=up("index_pages"),
            bnd_prev_pos=up("bnd_prev_pos"),
            bnd_pos=up("bnd_pos"),
        )

    def _prefill_fixed_const(self, key: str, build) -> ttnn.Tensor:
        """Device constants of the fixed-shape prefill path, built once per decoder."""

        consts = self.__dict__.setdefault("_prefill_fixed_consts", {})
        if key not in consts:
            consts[key] = build()
        return consts[key]

    def _streaming_csa_boundary_entries_fixed(
        self, prev_pos: ttnn.Tensor, pos: ttnn.Tensor
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """_streaming_csa_boundary_entries with the block start as device tensors (block_start > 0 by construction):
        the history positions block_start - 4 .. block_start - 1 and the entry's RoPE position block_start."""

        projections = getattr(self, "_fused_prefill_projections", {})
        try:
            attention_kv, attention_score = projections[(id(self.comp_kv), id(self.comp_gate))]
            index_kv, index_score = projections[(id(self.index_comp_kv), id(self.index_comp_gate))]
        except KeyError as exc:
            raise RuntimeError("CSA boundary repair escaped the fused prefill projections") from exc
        previous = self._cache_rows(self.cache.compressor_kv, prev_pos)
        current = ttnn.concat(
            [
                ttnn.slice(attention_kv, [0, 0, 0, 0], [1, 1, 4, attention_kv.shape[-1]]),
                ttnn.slice(attention_score, [0, 0, 0, 0], [1, 1, 4, attention_score.shape[-1]]),
                ttnn.slice(index_kv, [0, 0, 0, 0], [1, 1, 4, index_kv.shape[-1]]),
                ttnn.slice(index_score, [0, 0, 0, 0], [1, 1, 4, index_score.shape[-1]]),
            ],
            dim=-1,
        )
        current = ttnn.reshape(current, [1, 4, current.shape[-1]])
        if current.dtype != previous.dtype:
            current = ttnn.typecast(current, previous.dtype)
        raw = ttnn.concat([previous, current], dim=1)
        cos, sin = self._rope_values(pos)

        def pool(offset, kv_width, score_width, width, norm, *, rotate_hadamard):
            history_kv = ttnn.slice(raw, [0, 0, offset], [1, 8, offset + kv_width])
            offset += kv_width
            history_score = ttnn.slice(raw, [0, 0, offset], [1, 8, offset + score_width])
            offset += score_width
            previous_kv = ttnn.slice(history_kv, [0, 0, 0], [1, 4, width])
            current_kv = ttnn.slice(history_kv, [0, 4, width], [1, 8, 2 * width])
            previous_score = ttnn.slice(history_score, [0, 0, 0], [1, 4, width])
            current_score = ttnn.slice(history_score, [0, 4, width], [1, 8, 2 * width])
            history_kv = ttnn.concat([previous_kv, current_kv], dim=1)
            history_score = ttnn.concat([previous_score, current_score], dim=1)
            pooled = ttnn.sum(ttnn.mul(history_kv, ttnn.softmax(history_score, dim=1)), dim=1, keepdim=True)
            pooled = ttnn.rms_norm(
                ttnn.reshape(pooled, [1, 1, 1, width]), weight=norm, epsilon=self.config.rms_norm_eps
            )
            pooled = self._rope(pooled, cos, sin)
            if rotate_hadamard:
                pooled = ttnn.matmul(pooled, self.index_hadamard)
            return pooled, offset

        attention_entry, offset = pool(
            0,
            self.comp_kv.shape[-1],
            self.comp_gate.shape[-1],
            self.config.head_dim,
            self.comp_norm,
            rotate_hadamard=False,
        )
        index_entry, offset = pool(
            offset,
            self.index_comp_kv.shape[-1],
            self.index_comp_gate.shape[-1],
            self.config.index_head_dim,
            self.index_comp_norm,
            rotate_hadamard=True,
        )
        if offset != raw.shape[-1]:
            raise RuntimeError(f"CSA boundary repair consumed {offset} of {raw.shape[-1]} packed columns")
        return attention_entry, index_entry

    def _attention_streaming_block_fixed(
        self,
        x: ttnn.Tensor,
        input_ids: ttnn.Tensor,
        position_ids: ttnn.Tensor,
        inputs: PrefillBlockInputs,
    ) -> ttnn.Tensor:
        """_attention_streaming_block for a middle block (prefill_fixed_block_ok) with no host integers: the
        block start only enters through ``position_ids`` and ``inputs``; the HCA bank and the CSA indexer bank are the
        caches' full capacity with per-row validity masks (masked rows get exact zero attention weight), the local
        window is always the previous W rows, and every cache write goes through device page-table entries."""

        c = self.config
        seq_len = int(input_ids.shape[-1])
        page = c.page_size
        window = c.sliding_window
        _, q, kv, rope = self._project_q_kv(x, position_ids)
        compressed = index_entries = None
        if self.kind.compress_ratio:
            compressed = self._compress_prefill(
                x,
                position_ids,
                self.comp_kv,
                self.comp_gate,
                self.comp_ape,
                self.comp_norm,
                c.head_dim,
                rotate_hadamard=False,
            )
        if self.kind.attention is AttentionKind.CSA:
            index_entries = self._compress_prefill(
                x,
                position_ids,
                self.index_comp_kv,
                self.index_comp_gate,
                self.index_comp_ape,
                self.index_comp_norm,
                c.index_head_dim,
                rotate_hadamard=True,
            )
        boundary_entries = None
        if self.kind.attention is AttentionKind.CSA:
            boundary_entries = self._streaming_csa_boundary_entries_fixed(inputs.bnd_prev_pos, inputs.bnd_pos)

        # previous window (always W rows here) before the cyclic local write can reuse their pages
        previous_local = self._cache_rows(self.cache.local, inputs.prev_pos)
        previous_local = ttnn.reshape(previous_local, [1, 1, window, c.head_dim])
        cached_kv = (
            kv if kv.dtype == self.cache.local.tensor.dtype else ttnn.typecast(kv, self.cache.local.tensor.dtype)
        )
        current_local = cached_kv
        if current_local.dtype != previous_local.dtype:
            current_local = ttnn.typecast(current_local, previous_local.dtype)
        local_bank = ttnn.concat([previous_local, current_local], dim=2)
        local_bank = ttnn.reshape(local_bank, [window + seq_len, c.head_dim])
        for k in range(seq_len // page):
            self._paged_fill_cache(
                self.cache.local,
                ttnn.slice(cached_kv, [0, 0, k * page, 0], [1, 1, (k + 1) * page, cached_kv.shape[3]]),
                ttnn.slice(inputs.local_pages, [0, k], [1, k + 1]),
            )
        if self.kind.attention is AttentionKind.HCA:
            if seq_len != self.kind.compress_ratio:
                raise ValueError("fixed-shape HCA prefill expects one compressed entry per block")
            self._paged_update(self.cache.compressed, compressed, inputs.entry)
        elif self.kind.attention is AttentionKind.CSA:
            self._paged_fill_cache(self.cache.compressed, compressed, inputs.comp_pages)
            self._paged_fill_cache(self.cache.indexer, index_entries, inputs.index_pages)

        if self.kind.compress_ratio:
            fill_start, fill_length = self._prefill_raw_fill_window(seq_len)
            raw_plan = PrefillPlan(
                seq_len,
                position_ids,
                input_ids,
                (),
                inputs.local_pages,
                0,
                seq_len,
                None,
                inputs.raw_pages,
                fill_start,
                fill_length,
            )
            self._fill_raw_prefill_state(
                x,
                position_ids,
                self.comp_kv,
                self.comp_gate,
                self.comp_ape,
                self.cache.compressor_kv,
                self.cache.compressor_score,
                raw_plan,
            )
            if self.kind.attention is AttentionKind.CSA:
                self._fill_raw_prefill_state(
                    x,
                    position_ids,
                    self.index_comp_kv,
                    self.index_comp_gate,
                    self.index_comp_ape,
                    self.cache.indexer_compressor_kv,
                    self.cache.indexer_compressor_score,
                    raw_plan,
                )
        if boundary_entries is not None:
            attention_entry, index_entry = boundary_entries
            self._paged_update(self.cache.compressed, attention_entry, inputs.entry)
            self._paged_update(self.cache.indexer, index_entry, inputs.entry)

        hca_bank = hca_ordinals = index_bank = index_ordinals = None
        if self.kind.attention is AttentionKind.HCA:
            capacity = self.cache.compressed.rows
            hca_ordinals = self._prefill_fixed_const(
                "hca_ordinals",
                lambda: ttnn.reshape(
                    ttnn.arange(0, capacity, 1, device=self.mesh_device, dtype=ttnn.int32), [1, capacity]
                ),
            )
            hca_bank = self._cache_rows(self.cache.compressed, hca_ordinals)
        elif self.kind.attention is AttentionKind.CSA:
            rows = self.cache.indexer.rows
            score_rows = _round_up(max(c.index_topk, rows), 32)
            stored_rows = min(score_rows, rows)
            physical_index = ttnn.reshape(
                self.cache.indexer.tensor, [1, 1, self.cache.indexer.tensor.shape[0] * page, c.index_head_dim]
            )
            physical_index = ttnn.slice(physical_index, [0, 0, 0, 0], [1, 1, stored_rows, c.index_head_dim])
            query_chunk = int(getattr(self, "prefill_query_chunk", 32))
            guard_rows = score_rows - stored_rows + query_chunk
            guard = ttnn.slice(self.decode_index_guard, [0, 0, 0, 0], [1, 1, guard_rows, c.index_head_dim])
            index_bank = ttnn.concat([physical_index, guard], dim=2)
            index_ordinals = ttnn.slice(self.decode_index_ordinals, [0, 0], [1, score_rows + query_chunk])

        outputs = []
        local_heads = c.num_attention_heads // TP_DEGREE
        csa_table = None
        if self.kind.attention is AttentionKind.CSA and getattr(self, "prefill_csa_untilize_once", False):
            compressed_cache = self.cache.compressed
            csa_table = ttnn.to_layout(
                ttnn.reshape(compressed_cache.tensor, [compressed_cache.rows, compressed_cache.width]),
                ttnn.ROW_MAJOR_LAYOUT,
            )
        qc = int(getattr(self, "prefill_query_chunk", 32))
        if seq_len % qc:
            raise ValueError(f"prefill_query_chunk {qc} must divide the block of {seq_len}")
        for chunk_start in range(0, seq_len, qc):
            chunk_end = chunk_start + qc
            q_chunk = ttnn.slice(q, [0, 0, chunk_start, 0], [1, local_heads, chunk_end, c.head_dim])
            # local bank row of query r's key at offset o (o = 1 - W .. 0): W + chunk_start + r + o, never negative
            local_indices = self._prefill_fixed_const(
                f"local_idx_{qc}_{chunk_start}",
                lambda: self._to_tt(
                    (
                        torch.arange(chunk_start, chunk_end, dtype=torch.int32).reshape(-1, 1)
                        + torch.arange(1 - window, 1, dtype=torch.int32).reshape(1, -1)
                        + window
                    ),
                    dtype=ttnn.uint32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                ),
            )
            local_rows = ttnn.embedding(local_indices, local_bank, layout=ttnn.TILE_LAYOUT)
            gathered = ttnn.reshape(local_rows, [qc, 1, window, c.head_dim])
            mask = self._prefill_fixed_const(
                f"local_mask_{qc}",
                lambda: self._to_tt(torch.zeros(qc, 1, 1, window), dtype=ttnn.bfloat16),
            )
            counts = None
            if self.kind.compress_ratio:
                # each query row's visible compressed entries: (position + 1) // ratio
                chunk_positions = ttnn.slice(position_ids, [0, chunk_start], [1, chunk_end])
                counts = ttnn.div(ttnn.add(chunk_positions, 1), self.kind.compress_ratio, rounding_mode="floor")
            if self.kind.attention is AttentionKind.HCA:
                capacity = self.cache.compressed.rows
                repeated_hca = ttnn.repeat(ttnn.reshape(hca_bank, [1, 1, capacity, c.head_dim]), [qc, 1, 1, 1])
                valid = ttnn.lt(ttnn.reshape(hca_ordinals, [1, 1, 1, capacity]), ttnn.reshape(counts, [qc, 1, 1, 1]))
                compressed_mask = ttnn.typecast(ttnn.where(valid, 0.0, ATTENTION_MASK_VALUE), ttnn.bfloat16)
                gathered = ttnn.concat([gathered, repeated_hca], dim=2)
                mask = ttnn.concat([mask, compressed_mask], dim=-1)
            elif self.kind.attention is AttentionKind.CSA:
                idx_q = ttnn.slice(
                    self._fused_index_q_projection,
                    [0, 0, chunk_start, 0],
                    [1, 1, chunk_end, c.index_n_heads * c.index_head_dim],
                )
                idx_q = ttnn.reshape(idx_q, [1, qc, c.index_n_heads, c.index_head_dim])
                idx_q = ttnn.permute(idx_q, (0, 2, 1, 3))
                cos = ttnn.slice(rope[0], [0, 0, chunk_start, 0], [1, 1, chunk_end, c.rope_head_dim])
                sin = ttnn.slice(rope[1], [0, 0, chunk_start, 0], [1, 1, chunk_end, c.rope_head_dim])
                idx_q = self._rope(idx_q, cos, sin)
                idx_q = ttnn.matmul(idx_q, self.index_hadamard)
                idx_w = ttnn.slice(
                    self._fused_index_weight_projection, [0, 0, chunk_start, 0], [1, 1, chunk_end, c.index_n_heads]
                )
                topk, topk_mask = self._indexer_topk(
                    idx_q, idx_w, index_bank, ttnn.reshape(counts, [qc, 1]), ordinals=index_ordinals
                )
                topk = ttnn.reshape(topk, [qc, c.index_topk])
                flat_topk_mask = ttnn.reshape(topk_mask, [qc, c.index_topk])
                topk = ttnn.typecast(topk, ttnn.int32)
                topk_valid = ttnn.typecast(ttnn.eq(flat_topk_mask, 0.0), ttnn.int32)
                topk = ttnn.mul(topk, topk_valid)
                flat_topk = ttnn.reshape(topk, [1, qc * c.index_topk])
                if (
                    csa_table is not None
                    and getattr(self, "prefill_csa_gather_once", False)
                    and self._row_lut_kind(self.cache.compressed) == "identity"
                ):
                    selected = ttnn.embedding(ttnn.typecast(flat_topk, ttnn.uint32), csa_table, layout=ttnn.TILE_LAYOUT)
                else:
                    selected_parts = []
                    for gather_start in range(0, flat_topk.shape[-1], HCA_CACHE_GATHER_CHUNK_ROWS):
                        gather_end = min(gather_start + HCA_CACHE_GATHER_CHUNK_ROWS, flat_topk.shape[-1])
                        selected_parts.append(
                            self._cache_rows(
                                self.cache.compressed,
                                ttnn.slice(flat_topk, [0, gather_start], [1, gather_end]),
                                table=csa_table,
                            )
                        )
                    selected = selected_parts[0] if len(selected_parts) == 1 else ttnn.concat(selected_parts, dim=1)
                selected = ttnn.reshape(selected, [qc, 1, c.index_topk, c.head_dim])
                gathered = ttnn.concat([gathered, selected], dim=2)
                mask = ttnn.concat([mask, topk_mask], dim=-1)
            outputs.append(self._attention_from_streaming_rows(q_chunk, gathered, mask))

        if csa_table is not None:
            ttnn.deallocate(csa_table)
        attention = outputs[0] if len(outputs) == 1 else ttnn.concat(outputs, dim=2)
        attention = self._rope(attention, *rope, inverse=True)
        return self._grouped_output_projection(attention, seq_len)

    def prefill_attention_segment(
        self,
        hidden_streams: ttnn.Tensor,
        input_ids: ttnn.Tensor,
        position_ids: ttnn.Tensor,
        fixed_inputs: PrefillBlockInputs,
    ) -> ttnn.Tensor:
        """a middle block's attention half (mHC input map, norm, fixed-shape attention, mHC output map): device
        tensors in, device tensors out, no host integers or readbacks, so it can be captured in a trace."""

        residual = hidden_streams
        post, comb, collapsed = self._hc_pre(residual, self.hc_attn_fn, self.hc_attn_base, self.hc_attn_scale)
        normed = self._optimized_rms_norm(collapsed, weight=self.attn_norm, epsilon=self.config.rms_norm_eps)
        attention = self._attention_streaming_block_fixed(normed, input_ids, position_ids, fixed_inputs)
        return self._hc_post(attention, residual, post, comb)

    def prefill_forward_block(
        self,
        hidden_streams: ttnn.Tensor,
        input_ids: ttnn.Tensor,
        position_ids: ttnn.Tensor,
        *,
        block_start: int | None = None,
        valid_seq_len: int | None = None,
        fixed_inputs: PrefillBlockInputs | None = None,
    ) -> ttnn.Tensor:
        """Batch exact cache-backed attention for aligned prompt blocks of 128 to 512 tokens (multiples of 128).

        ``fixed_inputs``: a middle block's per-block values as device tensors; the attention then runs the
        fixed-shape path (_attention_streaming_block_fixed)."""

        seq_len = int(input_ids.shape[-1])
        valid_seq_len = seq_len if valid_seq_len is None else int(valid_seq_len)
        if not 1 <= valid_seq_len <= seq_len:
            raise ValueError(f"valid streaming length {valid_seq_len} outside [1, {seq_len}]")
        if (
            block_start is None
            or block_start < 0
            or block_start % 128
            or seq_len % 128
            or not 128 <= seq_len <= STREAMING_PREFILL_MAX_BLOCK
            or tuple(position_ids.shape) != (1, seq_len)
            or self.residual_layout != "replicated"
        ):
            return super().prefill_forward_block(
                hidden_streams,
                input_ids,
                position_ids,
                block_start=block_start,
            )
        if block_start + seq_len > self.max_context:
            raise ValueError(
                f"streaming block [{block_start}, {block_start + seq_len}) exceeds context {self.max_context}"
            )
        expected = (1, seq_len, self.config.hc_mult, self.config.hidden_size)
        if tuple(hidden_streams.shape) != expected:
            raise ValueError(f"hidden_streams shape {tuple(hidden_streams.shape)} != {expected}")

        if fixed_inputs is not None:
            streams = self.prefill_attention_segment(hidden_streams, input_ids, position_ids, fixed_inputs)
        else:
            residual = hidden_streams
            post, comb, collapsed = self._hc_pre(residual, self.hc_attn_fn, self.hc_attn_base, self.hc_attn_scale)
            normed = self._optimized_rms_norm(collapsed, weight=self.attn_norm, epsilon=self.config.rms_norm_eps)
            attention = self._attention_streaming_block(normed, input_ids, position_ids, block_start, valid_seq_len)
            streams = self._hc_post(attention, residual, post, comb)
        return self.prefill_ffn_half(streams, input_ids)

    def prefill_ffn_half(self, streams: ttnn.Tensor, input_ids: ttnn.Tensor) -> ttnn.Tensor:
        """A streaming prefill block's FFN half (mHC input map, norm, MoE, mHC output map) after its attention half."""

        residual = streams
        post, comb, collapsed = self._hc_pre(residual, self.hc_ffn_fn, self.hc_ffn_base, self.hc_ffn_scale)
        moe = self._moe_forward(
            self._optimized_rms_norm(
                collapsed,
                weight=self.ffn_norm,
                epsilon=self.config.rms_norm_eps,
                interleaved_memory=(
                    ttnn.L1_MEMORY_CONFIG if self.optimization_policy.ffn_norm_l1 else ttnn.DRAM_MEMORY_CONFIG
                ),
            ),
            input_ids,
        )
        return self._hc_post(moe, residual, post, comb)

    def _grouped_output_projection(
        self, attention: ttnn.Tensor, seq_len: int, *, verify_rows: bool = False
    ) -> ttnn.Tensor:
        """Run local O groups and reduce the row-parallel O-B result.

        ``verify_rows``: an M <= 32 row verify block fills the same single row tile as decode, so it runs
        the decode DRAM-sharded O-A program instead of copying each sharded weight to interleaved per call."""

        c = self.config
        local_groups = c.o_groups // TP_DEGREE
        decode_program = seq_len == 1 or (verify_rows and seq_len <= 32 and self.verify_sharded_o)
        if self._grouped_o_a_split_weights:
            if decode_program:
                grouped = ttnn.permute(attention, (0, 2, 1, 3))
                grouped = ttnn.reshape(grouped, [1, seq_len, local_groups, c.grouped_attention_width])
                grouped = ttnn.permute(grouped, (0, 2, 1, 3))
                assert self._grouped_o_a_split_input_memory is not None
                assert self._grouped_o_a_split_program is not None
                input_memory = self._grouped_o_a_split_input_memory
                program = self._grouped_o_a_split_program
                group_inputs = [
                    ttnn.slice(
                        grouped,
                        [0, local_group, 0, 0],
                        [1, local_group + 1, seq_len, c.grouped_attention_width],
                    )
                    for local_group in range(local_groups)
                ]
            else:
                local_heads = c.num_attention_heads // TP_DEGREE
                heads_per_group = local_heads // local_groups
                input_memory = None
                program = None
                group_inputs = []
                for local_group in range(local_groups):
                    group = ttnn.slice(
                        attention,
                        [0, local_group * heads_per_group, 0, 0],
                        [1, (local_group + 1) * heads_per_group, seq_len, c.head_dim],
                    )
                    group = ttnn.permute(group, (0, 2, 1, 3))
                    group_inputs.append(ttnn.reshape(group, [1, 1, seq_len, c.grouped_attention_width]))
            outputs = []
            for group, weight in zip(group_inputs, self._grouped_o_a_split_weights, strict=True):
                if decode_program:
                    assert input_memory is not None and program is not None
                    group = ttnn.to_memory_config(group, input_memory)
                    output = ttnn.matmul(
                        group,
                        weight,
                        program_config=program,
                        memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                        dtype=ttnn.bfloat16,
                        compute_kernel_config=self.dense_compute_kernel_config,
                    )
                    outputs.append(ttnn.sharded_to_interleaved(output, ttnn.L1_MEMORY_CONFIG))
                else:
                    interleaved_weight = ttnn.to_memory_config(weight, ttnn.DRAM_MEMORY_CONFIG)
                    outputs.append(
                        ttnn.matmul(
                            group,
                            interleaved_weight,
                            dtype=ttnn.bfloat16,
                            compute_kernel_config=self.dense_compute_kernel_config,
                        )
                    )
                    ttnn.deallocate(interleaved_weight)
            projected = ttnn.concat(outputs, dim=1)
        else:
            grouped = ttnn.permute(attention, (0, 2, 1, 3))
            grouped = ttnn.reshape(grouped, [1, seq_len, local_groups, c.grouped_attention_width])
            grouped = ttnn.permute(grouped, (0, 2, 1, 3))
            projected = ttnn.matmul(
                grouped,
                self.o_a,
                dtype=ttnn.bfloat16,
                compute_kernel_config=self.dense_compute_kernel_config,
            )
        projected = ttnn.permute(projected, (0, 2, 1, 3))
        projected = ttnn.reshape(projected, [1, 1, seq_len, local_groups * c.o_lora_rank])
        self._capture_prefill_boundary("output_lora", projected)
        if self._attention_fused_mmrs is not None and seq_len == 1:
            return self._attention_fused_mmrs(
                projected,
                self._dram_interleaved_weights[id(self.o_b)],
                self.dense_compute_kernel_config,
                gather_output=self.residual_layout == "replicated",
            )
        partial = self._optimized_linear(projected, self.o_b)
        return _all_reduce_partial(
            partial,
            self._attention_decode_all_reduce,
            ccl_dtype=self.ccl_dtype,
        )


__all__ = [
    "CACHE_BFP8_TILE_BYTES",
    "CCL_NUM_LINKS",
    "DecodeTrace",
    "DENSE_COMPUTE_PROFILES",
    "INDEXER_SEQUENCE_AXIS",
    "MULTICHIP_STRATEGY",
    "MOE_COMPUTE_PROFILES",
    "MultichipDecoder",
    "MultichipDeepSeekSparseMoE",
    "MultichipStrategy",
    "PagedTensor",
    "QUIETBOX2_MIXED_BFP2_POLICY",
    "QUIETBOX2_SELECTIVE_BFP4_LAYERS",
    "QUIETBOX2_SELECTIVE_BFP4_POLICY",
    "STOCK_PRECISION_POLICY",
    "SUPPORTED_PRECISION_POLICIES",
    "TARGET_MESH_SHAPE",
    "TP_AXIS",
    "TP_DEGREE",
    "multichip_cache_bytes",
    "multichip_cache_shapes",
    "multichip_stack_cache_bytes",
    "tensor_parallel_shapes",
]
