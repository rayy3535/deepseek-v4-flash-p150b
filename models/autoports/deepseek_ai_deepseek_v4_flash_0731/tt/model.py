"""Full autoregressive TTNN model for DeepSeek-V4-Flash-0731.

The production path is deliberately narrow: a complete four-p300c tensor-
parallel mesh, the evidence-selected :class:`MultichipDecoder` for every
backbone layer, BF16 replicated residual streams between layers, BFP8 paged
caches, and a vocab-sharded terminal projection.  There is no layer streaming,
single-device, replicated-weight, or host-compute fallback in this module.

The stock-precision, 1M-context model cannot physically fit this target.
:func:`estimate_full_model_capacity` records that exact lower bound and
``from_checkpoint`` fails before loading stock weights.  QuietBox2 production
admission preserves all 43 layers, batch one, the exact 131,072-token benchmark
context, routed down/shared experts in BFP4_B, and two shared RoPE variants.
The vLLM-selected policy retains routed gate/up in BFP4_B for eight
sensitivity-selected layers and uses BFP2_B for the other 35.  The older
all-BFP2 policy remains explicit for reproducibility.  Reduced-layer
construction remains probe-only and is rejected by the production generator.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import torch
from loguru import logger

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.autoports.deepseek_ai_deepseek_v4_flash_0731.tt.vendored.embedding_1d import Embedding1D, Embedding1DConfig
from models.common.lazy_weight import LazyWeight
from models.autoports.deepseek_ai_deepseek_v4_flash_0731.tt.vendored.lm_head_1d import LMHead1D, LMHead1DConfig

from ..checkpoint import CheckpointIndex, LayerCheckpoint
from ..config import DeepSeekV4FlashConfig, physical_row_lut
from .expert_pool import ExpertHostStore
from .expert_pool import expert_cache_dir as expert_cache_dir_for
from .functional_decoder import BatchedPrefillPlan, PagedTensor, PrefillPlan
from .multichip_decoder import (
    CCL_NUM_LINKS,
    QUIETBOX2_MIXED_BFP2_POLICY,
    QUIETBOX2_PAGED_BFP4_POLICY,
    QUIETBOX2_PAGED_DEFAULT_POOL,
    QUIETBOX2_SELECTIVE_BFP4_LAYERS,
    QUIETBOX2_SELECTIVE_BFP4_POLICY,
    STOCK_PRECISION_POLICY,
    STREAMING_PREFILL_MAX_BLOCK,
    TARGET_MESH_SHAPE,
    TP_AXIS,
    TP_DEGREE,
    DecodeRouteState,
    MultichipDecoder,
    _flat_shard_mapper,
    multichip_cache_bytes,
)
from .optimized_decoder import streamwise_hc_collapse

# Keep this share of the cache filesystem free; below it a layer is packed but not written.
EXPERT_CACHE_MIN_FREE_FRACTION = 0.20


def _save_expert_cache(store: ExpertHostStore, layer_dir: Path, key: str) -> None:
    import shutil

    layer_bytes = sum(t.volume() for t in (*store.gate_up, *store.down)) * 9 // 16  # BFP4: 4.5 bits/value
    layer_dir.parent.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(layer_dir.parent)
    if usage.free - layer_bytes < EXPERT_CACHE_MIN_FREE_FRACTION * usage.total:
        logger.warning(
            f"expert cache: not writing {layer_dir}, {usage.free / 2**30:.0f} GiB free of {usage.total / 2**30:.0f} GiB"
        )
        return
    store.save(layer_dir, key)


# These are the selected-layout figures audited in
# doc/multichip_decoder/capacity_weights.csv.  They include the current
# per-layer RoPE residency and every accepted duplicate decode layout.
USABLE_P300C_DRAM_BYTES_PER_DEVICE = 33_910_295_552
FULL_DECODER_WEIGHT_BYTES_PER_DEVICE = 53_167_682_048
QUIETBOX2_CONTEXT = 131_072
STREAMING_PREFILL_BLOCK_SIZE = 128
# HCA and CSA workspaces change shape as the logical context grows. TTNN's
# program cache therefore gains context-specific entries even though the input
# block itself stays fixed at 128 tokens. Bound those entries before they can
# consume the selected policy's runtime DRAM margin. Clearing every 64 blocks
# limits one cache epoch to 8,192 prompt tokens.
STREAMING_PREFILL_PROGRAM_CACHE_BLOCKS = 64
QUIETBOX2_QUANTIZED_WEIGHTS_EXCLUDING_ROPE_PER_DEVICE = 30_082_144_768
QUIETBOX2_SHARED_ROPE_BYTES_PER_DEVICE = 67_110_912
QUIETBOX2_QUANTIZED_DECODER_WEIGHT_BYTES_PER_DEVICE = (
    QUIETBOX2_QUANTIZED_WEIGHTS_EXCLUDING_ROPE_PER_DEVICE + QUIETBOX2_SHARED_ROPE_BYTES_PER_DEVICE
)
QUIETBOX2_BFP4_UPGRADE_BYTES_PER_LAYER_PER_DEVICE = 268_435_456
QUIETBOX2_SELECTIVE_BFP4_DECODER_WEIGHT_BYTES_PER_DEVICE = (
    QUIETBOX2_QUANTIZED_DECODER_WEIGHT_BYTES_PER_DEVICE
    + len(QUIETBOX2_SELECTIVE_BFP4_LAYERS) * QUIETBOX2_BFP4_UPGRADE_BYTES_PER_LAYER_PER_DEVICE
)
# Paged policy: the mixed-BFP2 figure minus every resident routed expert (W1/W3 BFP2, W2 BFP4), plus a
# BFLOAT4_B pool of P + 1 slots per layer (the extra slot is the null slot).
QUIETBOX2_MIXED_ROUTED_BYTES_PER_LAYER_PER_DEVICE = 256 * (2 * 655_360 + 1_179_648)
PAGED_BFP4_EXPERT_BYTES_PER_DEVICE = 3 * 1_179_648


def paged_decoder_weight_bytes_per_device(pool_size: int, layers: int = 43) -> int:
    return (
        QUIETBOX2_QUANTIZED_DECODER_WEIGHT_BYTES_PER_DEVICE
        - layers * QUIETBOX2_MIXED_ROUTED_BYTES_PER_LAYER_PER_DEVICE
        + layers * (pool_size + 1) * PAGED_BFP4_EXPERT_BYTES_PER_DEVICE
    )


FULL_DECODE_CONSTANT_BYTES_PER_DEVICE = 45_545_632
EMBEDDING_BYTES_PER_DEVICE = 264_765_440
LM_HEAD_BYTES_PER_DEVICE = 264_765_440
FINAL_NORM_BYTES_PER_DEVICE = 262_144
HYPER_HEAD_BYTES_PER_DEVICE = 2_105_344
TERMINAL_WEIGHT_BYTES_PER_DEVICE = (
    EMBEDDING_BYTES_PER_DEVICE + LM_HEAD_BYTES_PER_DEVICE + FINAL_NORM_BYTES_PER_DEVICE + HYPER_HEAD_BYTES_PER_DEVICE
)

FULL_LAYER_INDICES = tuple(range(DeepSeekV4FlashConfig().num_hidden_layers))
REPRESENTATIVE_LAYER_INDICES = (0, 2, 3, 4)
LM_HEAD_LOCAL_SPLIT_COLUMNS = 8192
EXPERIMENTAL_POLICY_ENV_PREFIX = "DSV4_"


def _prefill_execution_length(prompt_length: int) -> int:
    """Round long prompts to the fixed streaming-block shape."""

    if not 1 <= prompt_length <= QUIETBOX2_CONTEXT:
        raise ValueError(f"prompt length {prompt_length} outside deployment context")
    if prompt_length <= STREAMING_PREFILL_BLOCK_SIZE:
        return prompt_length
    return (
        (prompt_length + STREAMING_PREFILL_BLOCK_SIZE - 1)
        // STREAMING_PREFILL_BLOCK_SIZE
        * STREAMING_PREFILL_BLOCK_SIZE
    )


def _streaming_blocks(execution_length: int, block_size: int) -> list[tuple[int, int]]:
    """[start, end) streaming blocks: block_size tokens while that many remain, then 128-token tails.

    execution_length is a multiple of 128 (_prefill_execution_length), so every block starts on a 128 boundary and
    no more padding is executed than with 128-token blocks.
    """

    if block_size % STREAMING_PREFILL_BLOCK_SIZE or block_size < STREAMING_PREFILL_BLOCK_SIZE:
        raise ValueError(f"streaming block size {block_size} must be a multiple of {STREAMING_PREFILL_BLOCK_SIZE}")
    blocks, start = [], 0
    while start < execution_length:
        size = block_size if execution_length - start >= block_size else STREAMING_PREFILL_BLOCK_SIZE
        blocks.append((start, min(execution_length, start + size)))
        start += size
    return blocks


def _should_clear_streaming_program_cache(completed_blocks: int, total_blocks: int) -> bool:
    """Return whether a nonterminal streaming-prefill cache epoch is complete."""

    if not 1 <= completed_blocks <= total_blocks:
        raise ValueError(f"completed blocks {completed_blocks} outside [1, {total_blocks}]")
    return completed_blocks < total_blocks and completed_blocks % STREAMING_PREFILL_PROGRAM_CACHE_BLOCKS == 0


def active_experimental_policy_overrides() -> tuple[str, ...]:
    """Return decoder experiment selectors that would change deployment policy."""

    return tuple(sorted(name for name in os.environ if name.startswith(EXPERIMENTAL_POLICY_ENV_PREFIX)))


def require_locked_deployment_policy() -> None:
    """Fail closed when an experiment selector could alter the qualified graph."""

    overrides = active_experimental_policy_overrides()
    if overrides:
        raise RuntimeError(
            "DeepSeek-V4-Flash full-model deployment requires the qualified no-override policy; "
            f"unset experimental environment variables: {', '.join(overrides)}"
        )


def require_selected_decoder_policy(
    layers: Sequence[MultichipDecoder],
    *,
    deployment_mode: str = STOCK_PRECISION_POLICY,
    moe_compute_profile: str = "lofi",
    dense_compute_profile: str = "lofi",
) -> None:
    """Validate immutable layer provenance before accepting a direct model build."""

    expected = {
        "optimization_profile": "optimized",
        "ccl_dtype": "bf16",
        "topology": "grouped_o_a_split_shared_dram",
        "qr_projection": "separate",
        "residual_layout": "replicated",
        "attention_ccl_placement": "post_matmul_allreduce",
        "moe_projection": "packed",
        "deployment_precision": deployment_mode,
        "moe_compute_profile": moe_compute_profile,
        "dense_compute_profile": dense_compute_profile,
    }
    failures = []
    for position, layer in enumerate(layers):
        overrides = tuple(getattr(layer, "experimental_policy_overrides", ()))
        actual = {
            "optimization_profile": getattr(getattr(layer, "optimization_policy", None), "name", None),
            "ccl_dtype": getattr(layer, "ccl_dtype", None),
            "topology": getattr(layer, "topology_experiment", None),
            "qr_projection": getattr(layer, "qr_projection_experiment", None),
            "residual_layout": getattr(layer, "residual_layout", None),
            "attention_ccl_placement": getattr(layer, "attention_ccl_placement", None),
            "moe_projection": getattr(getattr(layer, "moe", None), "projection_experiment", None),
            "deployment_precision": getattr(layer, "deployment_precision", None),
            "moe_compute_profile": getattr(layer, "moe_compute_profile", None),
            "dense_compute_profile": getattr(layer, "dense_compute_profile", None),
        }
        if overrides or actual != expected:
            failures.append(f"layer[{position}] overrides={overrides} policy={actual}")
    if failures:
        raise RuntimeError(
            "DeepSeek-V4-Flash full-model layers do not match the qualified immutable decoder policy: "
            + "; ".join(failures)
        )


class FullModelCapacityError(RuntimeError):
    """Raised when the selected all-resident full model cannot fit target DRAM."""


@dataclass(frozen=True)
class FullModelCapacity:
    """Exact selected-layout capacity estimate for one TP rank."""

    usable_bytes_per_device: int
    decoder_weight_bytes_per_device: int
    kv_cache_bytes_per_device: int
    decode_constant_bytes_per_device: int
    terminal_weight_bytes_per_device: int
    required_bytes_per_device: int
    over_budget_bytes_per_device: int
    max_context: int
    max_batch_size: int
    layer_count: int
    deployment_mode: str

    @property
    def fits(self) -> bool:
        return self.over_budget_bytes_per_device <= 0


def estimate_full_model_capacity(
    *,
    config: DeepSeekV4FlashConfig | None = None,
    max_context: int | None = None,
    max_batch_size: int = 1,
    deployment_mode: str = STOCK_PRECISION_POLICY,
    expert_pool_size: int | None = None,
) -> FullModelCapacity:
    """Return the selected full-stack lower bound without touching a device.

    Decoder weights and constants are the checked exact full-context values.
    Cache bytes are recomputed from the implementation's physical page geometry
    for the requested context and multiplied by independent batch lanes.
    Activations, trace buffers, sampler state, scratch, and allocator slack are
    intentionally excluded, so a failure is a hard physical impossibility.
    """

    c = DeepSeekV4FlashConfig() if config is None else config
    c.validate()
    quietbox_policies = {
        QUIETBOX2_MIXED_BFP2_POLICY,
        QUIETBOX2_SELECTIVE_BFP4_POLICY,
        QUIETBOX2_PAGED_BFP4_POLICY,
    }
    if deployment_mode not in {STOCK_PRECISION_POLICY, *quietbox_policies}:
        raise ValueError(f"unknown deployment mode {deployment_mode!r}")
    default_context = QUIETBOX2_CONTEXT if deployment_mode in quietbox_policies else c.max_position_embeddings
    context = default_context if max_context is None else int(max_context)
    if not 1 <= context <= c.max_position_embeddings:
        raise ValueError(f"max_context must be in [1, {c.max_position_embeddings}], got {context}")
    if max_batch_size <= 0:
        raise ValueError(f"max_batch_size must be positive, got {max_batch_size}")
    cache = sum(multichip_cache_bytes(c, layer_idx, context) for layer_idx in FULL_LAYER_INDICES)
    cache *= max_batch_size
    if deployment_mode == QUIETBOX2_MIXED_BFP2_POLICY:
        decoder_weights = QUIETBOX2_QUANTIZED_DECODER_WEIGHT_BYTES_PER_DEVICE
    elif deployment_mode == QUIETBOX2_SELECTIVE_BFP4_POLICY:
        decoder_weights = QUIETBOX2_SELECTIVE_BFP4_DECODER_WEIGHT_BYTES_PER_DEVICE
    elif deployment_mode == QUIETBOX2_PAGED_BFP4_POLICY:
        decoder_weights = paged_decoder_weight_bytes_per_device(
            QUIETBOX2_PAGED_DEFAULT_POOL if expert_pool_size is None else int(expert_pool_size)
        )
    else:
        decoder_weights = FULL_DECODER_WEIGHT_BYTES_PER_DEVICE
    required = decoder_weights + cache + FULL_DECODE_CONSTANT_BYTES_PER_DEVICE + TERMINAL_WEIGHT_BYTES_PER_DEVICE
    return FullModelCapacity(
        usable_bytes_per_device=USABLE_P300C_DRAM_BYTES_PER_DEVICE,
        decoder_weight_bytes_per_device=decoder_weights,
        kv_cache_bytes_per_device=cache,
        decode_constant_bytes_per_device=FULL_DECODE_CONSTANT_BYTES_PER_DEVICE,
        terminal_weight_bytes_per_device=TERMINAL_WEIGHT_BYTES_PER_DEVICE,
        required_bytes_per_device=required,
        over_budget_bytes_per_device=required - USABLE_P300C_DRAM_BYTES_PER_DEVICE,
        max_context=context,
        max_batch_size=max_batch_size,
        layer_count=c.num_hidden_layers,
        deployment_mode=deployment_mode,
    )


@dataclass(frozen=True)
class TerminalHostWeights:
    embedding: torch.Tensor
    hyper_head_fn: torch.Tensor
    hyper_head_base: torch.Tensor
    hyper_head_scale: torch.Tensor
    final_norm: torch.Tensor
    lm_head: torch.Tensor


def load_terminal_weights(checkpoint_root: str | Path, config: DeepSeekV4FlashConfig) -> TerminalHostWeights:
    """Read and validate the six non-decoder tensors in the base HF path."""

    index = CheckpointIndex.open(checkpoint_root)
    keys = ("embed.weight", "hc_head_fn", "hc_head_base", "hc_head_scale", "norm.weight", "head.weight")
    raw = index.read_many(keys)
    expected = {
        "embed.weight": (config.vocab_size, config.hidden_size),
        "hc_head_fn": (config.hc_mult, config.hc_width),
        "hc_head_base": (config.hc_mult,),
        "hc_head_scale": (1,),
        "norm.weight": (config.hidden_size,),
        "head.weight": (config.vocab_size, config.hidden_size),
    }
    mismatches = [
        f"{key}: {tuple(raw[key].shape)} != {shape}"
        for key, shape in expected.items()
        if tuple(raw[key].shape) != shape
    ]
    if mismatches:
        raise ValueError("invalid full-model checkpoint tensors: " + "; ".join(mismatches))
    return TerminalHostWeights(
        embedding=raw["embed.weight"].to(torch.bfloat16),
        hyper_head_fn=raw["hc_head_fn"].to(torch.float32),
        hyper_head_base=raw["hc_head_base"].to(torch.float32),
        hyper_head_scale=raw["hc_head_scale"].to(torch.float32),
        final_norm=raw["norm.weight"].to(torch.bfloat16),
        lm_head=raw["head.weight"].to(torch.bfloat16),
    )


@dataclass(frozen=True)
class PrefillLaneState:
    """Prompt state for one slot; long uploads are padded but execution is exact."""

    slot: int
    prompt_length: int
    token_ids: ttnn.Tensor
    layer_plans: tuple[PrefillPlan, ...]
    position_ids: ttnn.Tensor | None = None
    # session continuation (streaming lanes only): skip the blocks before resume_start (the caches already
    # hold them, after restore_ring_snapshot), and snapshot the rings before this prompt's last block.
    resume_start: int = 0
    snapshot_rings: bool = False
    # (streaming lanes): the padded token ids on the host; each block's token / position inputs are uploaded
    # from here instead of sliced out of the whole-prompt device tensors, whose shape (the prompt's padded length)
    # made every new prompt length compile two slice programs (behind live traces in a server).
    host_token_ids: torch.Tensor | None = None

    @property
    def uses_streaming_blocks(self) -> bool:
        return self.position_ids is not None


@dataclass(frozen=True)
class FullModelPrefillState:
    """Prepared mixed-prompt state; inactive slots have no lane entry."""

    lanes: tuple[PrefillLaneState, ...]
    max_batch_size: int

    @property
    def active_slots(self) -> tuple[int, ...]:
        return tuple(lane.slot for lane in self.lanes)


@dataclass
class FullModelDecodeTrace:
    trace_id: int
    logits: ttnn.Tensor

    def execute(self, mesh_device: ttnn.MeshDevice, *, blocking: bool = False) -> ttnn.Tensor:
        ttnn.execute_trace(mesh_device, self.trace_id, cq_id=0, blocking=blocking)
        return self.logits

    def release(self, mesh_device: ttnn.MeshDevice) -> None:
        ttnn.release_trace(mesh_device, self.trace_id)


@dataclass
class FullModelPagedDecodeTrace:
    """Decode trace split at every MoE so the host can fill expert pools between segments (S mode).

    Segment 0 is embedding + routing of layer 0, segment i finishes layer i-1 and routes layer i, the last
    segment finishes the last layer and computes logits.  Same execute/release/logits contract as
    FullModelDecodeTrace, so the generator replays it unchanged.
    """

    segments: tuple[int, ...]
    routes: tuple[DecodeRouteState, ...]
    decoders: tuple[MultichipDecoder, ...]
    logits: ttnn.Tensor
    # compressor variants: "none" pools no compressed entry and writes the shared logits. With a host
    # position set, execute() replays "none" unless that position completes a compressed entry, then advances
    # the position; without one it always replays the full graph.
    variants: dict = field(default_factory=dict)
    position: int | None = None
    ratios: tuple[int, int] = (4, 128)
    # mean-pooled outputs of model.decode_tap_layers ([1, 1, 1, H] each), shared by every variant
    taps: dict = field(default_factory=dict)

    def _chain(self) -> tuple[tuple[int, ...], tuple[DecodeRouteState, ...]]:
        if self.position is None or not self.variants:
            return self.segments, self.routes
        csa, _ = self.ratios
        key = "full" if (self.position + 1) % csa == 0 else "none"
        return self.variants.get(key, (self.segments, self.routes))

    def execute(self, mesh_device: ttnn.MeshDevice, *, blocking: bool = False) -> ttnn.Tensor:
        segments, routes = self._chain()
        for index, trace_id in enumerate(segments):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
            if index < len(routes) and self.decoders[index].moe.pool is not None:
                # Reading the router picks waits for this segment; slot and table writes are queued on
                # the same command queue ahead of the next segment.
                self.decoders[index].moe.ensure_resident(routes[index].selected)
        if self.position is not None:
            self.position += 1
        if blocking:
            ttnn.synchronize_device(mesh_device)
        return self.logits

    def release(self, mesh_device: ttnn.MeshDevice) -> None:
        released = set()
        for segments, _ in (*self.variants.values(), (self.segments, ())):
            for trace_id in segments:
                if trace_id not in released:
                    ttnn.release_trace(mesh_device, trace_id)
                    released.add(trace_id)
        for decoder in self.decoders:
            decoder.moe.pool_eager_ensure = True


@dataclass
class FullModelPagedVerifyTrace:
    """The M-token verify block captured like FullModelPagedDecodeTrace, one segment per MoE boundary.

    Inputs are persistent device tensors: ``token_ids`` [1, 1, 1, M] uint32 and ``start`` [1] int32 (first row's
    position). execute() writes them, replays the segments and, after each routing segment, reads that layer's
    picks back, loads the experts and fills the shared verify buffers before the next segment. Returns the
    vocab-sharded logits of all M rows; ``taps`` holds the mean-pooled outputs of the tapped layers.
    """

    m: int
    segments: tuple[int, ...]
    routes: tuple
    decoders: tuple[MultichipDecoder, ...]
    buffers: dict
    token_ids: ttnn.Tensor
    start: ttnn.Tensor
    logits: ttnn.Tensor
    taps: dict
    # per-device row max / first argmax of the logits, captured in the last segment (greedy=True)
    greedy: tuple[ttnn.Tensor, ttnn.Tensor] | None = None

    def execute(
        self,
        mesh_device: ttnn.MeshDevice,
        tokens: Sequence[int],
        start: int,
        *,
        blocking: bool = False,
        refill: bool = True,
        stats: dict | None = None,
    ) -> ttnn.Tensor:
        """``refill=False`` (profiling) replays without reading the picks back or touching the buffers / pools;
        ``stats`` accumulates the host time spent in readback + fill per call."""

        if len(tokens) != self.m:
            raise ValueError(f"verify trace takes {self.m} tokens, got {len(tokens)}")
        mapper = ttnn.ReplicateTensorToMesh(mesh_device)
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(
                torch.tensor(list(tokens), dtype=torch.int32).reshape(1, 1, 1, self.m),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=mapper,
            ),
            self.token_ids,
        )
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(
                torch.tensor([start], dtype=torch.int32),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=mapper,
            ),
            self.start,
        )
        for index, trace_id in enumerate(self.segments):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
            if refill and index < len(self.routes):
                t0 = time.perf_counter()
                selected = self.routes[index].selected
                picks = ttnn.to_torch((ttnn.get_device_tensors(selected) or (selected,))[0])
                t1 = time.perf_counter()
                self.decoders[index].moe.fill_verify_buffers(picks.reshape(self.m, -1).to(torch.int64), self.buffers)
                if stats is not None:
                    stats["readback_s"] = stats.get("readback_s", 0.0) + t1 - t0
                    stats["fill_s"] = stats.get("fill_s", 0.0) + time.perf_counter() - t1
        if blocking:
            ttnn.synchronize_device(mesh_device)
        return self.logits

    def release(self, mesh_device: ttnn.MeshDevice) -> None:
        for trace_id in self.segments:
            ttnn.release_trace(mesh_device, trace_id)


def _lm_head_lazy_splits(
    host_weight: torch.Tensor,
    *,
    mesh_device: ttnn.MeshDevice,
    config: DeepSeekV4FlashConfig,
) -> list[LazyWeight]:
    """Build common LMHead1D weights in rank-local contiguous vocab order."""

    if config.vocab_size % TP_DEGREE:
        raise ValueError("vocabulary must divide the fixed TP degree")
    local_width = config.vocab_size // TP_DEGREE
    sizes = [LM_HEAD_LOCAL_SPLIT_COLUMNS] * (local_width // LM_HEAD_LOCAL_SPLIT_COLUMNS)
    if sum(sizes) < local_width:
        sizes.append(local_width - sum(sizes))
    transposed = host_weight.transpose(0, 1).contiguous()
    result: list[LazyWeight] = []
    offset = 0
    mapper = ttnn.MeshMapperConfig(
        placements=[ttnn.PlacementShard(-1)],
        mesh_shape_override=ttnn.MeshShape([TP_DEGREE]),
    )
    for split_idx, width in enumerate(sizes):
        per_rank = [
            transposed[:, rank * local_width + offset : rank * local_width + offset + width]
            for rank in range(TP_DEGREE)
        ]
        combined = torch.cat(per_rank, dim=-1).unsqueeze(0).unsqueeze(0)
        result.append(
            LazyWeight(
                source=combined,
                dtype=ttnn.bfloat16,
                device=mesh_device,
                mesh_mapper_config=mapper,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_dir_weight_name=None,
            )
        )
        offset += width
    if offset != local_width:
        raise RuntimeError(f"LM-head split coverage {offset} != local vocabulary {local_width}")
    return result


class DeepSeekV4FlashModel(LightweightModule):
    """Full HF autoregressive path around selected four-device decoder layers."""

    # Short prompts (<= 128 tokens) run as one streaming block padded to 128 instead of the exact-length path, so
    # every prompt compiles the same prefill programs: a serving loop never meets a new prefill shape (whose compile
    # needs every trace released and, past the cached-shape limit, a program cache clear). Not bitwise with the exact
    # path. Default off.
    prefill_pad_short = False

    def __init__(
        self,
        *,
        mesh_device: ttnn.MeshDevice,
        config: DeepSeekV4FlashConfig,
        layers: Sequence[MultichipDecoder],
        layer_indices: Sequence[int],
        terminal: TerminalHostWeights,
        max_context: int,
        max_batch_size: int,
        reduced_probe: bool,
        deployment_mode: str = STOCK_PRECISION_POLICY,
        moe_compute_profile: str = "lofi",
        dense_compute_profile: str = "lofi",
    ) -> None:
        super().__init__()
        require_locked_deployment_policy()
        if (moe_compute_profile != "lofi" or dense_compute_profile != "lofi") and not reduced_probe:
            raise ValueError("non-default compute profiles are admitted only for explicit reduced probes")
        require_selected_decoder_policy(
            layers,
            deployment_mode=deployment_mode,
            moe_compute_profile=moe_compute_profile,
            dense_compute_profile=dense_compute_profile,
        )
        if tuple(mesh_device.shape) != TARGET_MESH_SHAPE or mesh_device.get_num_devices() != TP_DEGREE:
            raise ValueError(
                f"full model requires the selected {TARGET_MESH_SHAPE} mesh, got "
                f"shape={tuple(mesh_device.shape)} devices={mesh_device.get_num_devices()}"
            )
        self.mesh_device = mesh_device
        self.config = config
        self.layers = tuple(layers)
        self.layer_indices = tuple(int(index) for index in layer_indices)
        self.max_context = int(max_context)
        self.max_batch_size = int(max_batch_size)
        self.reduced_probe = bool(reduced_probe)
        self.deployment_mode = deployment_mode
        self.moe_compute_profile = moe_compute_profile
        self.dense_compute_profile = dense_compute_profile
        self.vocab_size = config.vocab_size
        self.padded_vocab_size = config.vocab_size
        self.num_devices = TP_DEGREE
        self.residual_layout = "replicated_bf16_hc4"
        self.cache_dtype = "bfloat8_b"
        self.activation_dtype = "bfloat16"
        self.ccl_dtype = "bfloat16"

        embed_source = terminal.embedding.unsqueeze(0).unsqueeze(0)
        embed_lazy = LazyWeight(
            source=embed_source,
            dtype=ttnn.bfloat16,
            device=mesh_device,
            mesh_mapper_config=ttnn.MeshMapperConfig(
                placements=[ttnn.PlacementShard(-1)],
                mesh_shape_override=ttnn.MeshShape([TP_DEGREE]),
            ),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_dir_weight_name=None,
        )
        self.embedding = Embedding1D.from_config(
            Embedding1DConfig(
                weights=embed_lazy,
                mesh_device=mesh_device,
                weights_dtype=ttnn.bfloat16,
                weights_memcfg=ttnn.DRAM_MEMORY_CONFIG,
                output_memcfg=ttnn.DRAM_MEMORY_CONFIG,
            )
        )
        self.lm_head = LMHead1D.from_config(
            LMHead1DConfig(
                output_weights=_lm_head_lazy_splits(terminal.lm_head, mesh_device=mesh_device, config=config),
                mesh_device=mesh_device,
                dim=config.hidden_size,
                max_batch_size=max_batch_size,
                lm_head_dtype=ttnn.bfloat16,
                output_memcfg=ttnn.DRAM_MEMORY_CONFIG,
            )
        )
        self.hyper_head_fn = ttnn.from_torch(
            terminal.hyper_head_fn.transpose(0, 1).contiguous(),
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.hyper_head_base = ttnn.from_torch(
            terminal.hyper_head_base.reshape(1, 1, 1, -1),
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.hyper_head_scale = ttnn.from_torch(
            terminal.hyper_head_scale.reshape(1, 1, 1, -1),
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.final_norm = ttnn.from_torch(
            terminal.final_norm.reshape(1, 1, 1, -1),
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.terminal_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_root: str | Path,
        mesh_device: ttnn.MeshDevice,
        *,
        config: DeepSeekV4FlashConfig | None = None,
        max_context: int | None = None,
        max_batch_size: int = 1,
        layer_indices: Sequence[int] | None = None,
        allow_reduced_probe: bool = False,
        deployment_mode: str = STOCK_PRECISION_POLICY,
        moe_compute_profile: str = "lofi",
        dense_compute_profile: str = "lofi",
        expert_pool_size: int | None = None,
        expert_pool_keep_torch: bool = False,
        expert_cache_dir: str | Path | None = None,
        attention_dtype: ttnn.DataType = ttnn.bfloat4_b,
        prefill_block_size: int = STREAMING_PREFILL_BLOCK_SIZE,
    ) -> "DeepSeekV4FlashModel":
        """Construct the all-resident model or an explicitly scoped probe.

        With expert paging, ``expert_cache_dir`` keeps each layer's packed BFP4 routed experts as tensorbin
        files: a layer whose manifest matches skips expert dequantization and packing, a missing or
        stale layer is packed from the checkpoint and written back.

        Production construction always means all 43 layers at the logical
        context fixed by the selected deployment policy.  It has no capacity
        bypass: failing the exact lower bound must never turn into implicit
        streaming or a smaller advertised model.
        """

        require_locked_deployment_policy()
        c = DeepSeekV4FlashConfig() if config is None else config
        c.validate()
        quietbox_policies = {
            QUIETBOX2_MIXED_BFP2_POLICY,
            QUIETBOX2_SELECTIVE_BFP4_POLICY,
            QUIETBOX2_PAGED_BFP4_POLICY,
        }
        if deployment_mode not in {STOCK_PRECISION_POLICY, *quietbox_policies}:
            raise ValueError(f"unknown deployment mode {deployment_mode!r}")
        default_context = QUIETBOX2_CONTEXT if deployment_mode in quietbox_policies else c.max_position_embeddings
        context = default_context if max_context is None else int(max_context)
        indices = FULL_LAYER_INDICES if layer_indices is None else tuple(int(i) for i in layer_indices)
        if not indices or len(set(indices)) != len(indices) or any(i not in FULL_LAYER_INDICES for i in indices):
            raise ValueError(f"invalid layer_indices {indices}")
        if deployment_mode in quietbox_policies:
            reduced = indices != FULL_LAYER_INDICES or context != QUIETBOX2_CONTEXT or max_batch_size != 1
            if reduced and not allow_reduced_probe:
                raise ValueError(
                    f"{deployment_mode} is qualified only for all 43 layers, "
                    f"context={QUIETBOX2_CONTEXT}, and max_batch_size=1"
                )
        else:
            reduced = indices != FULL_LAYER_INDICES or context != c.max_position_embeddings
        if (moe_compute_profile != "lofi" or dense_compute_profile != "lofi") and not (reduced and allow_reduced_probe):
            raise ValueError("non-default compute profiles are admitted only for explicit reduced probes")
        if deployment_mode == STOCK_PRECISION_POLICY and reduced and not allow_reduced_probe:
            raise ValueError(
                "reduced layers/context are permitted only for explicit representative-layer probes; "
                "production must preserve all 43 layers and the 1048576-token context"
            )
        if not reduced:
            capacity = estimate_full_model_capacity(
                config=c,
                max_context=context,
                max_batch_size=max_batch_size,
                deployment_mode=deployment_mode,
                expert_pool_size=expert_pool_size,
            )
            if not capacity.fits:
                raise FullModelCapacityError(
                    "selected all-resident TP4 full model has a hard p300c DRAM deficit: "
                    f"required={capacity.required_bytes_per_device} B/device, "
                    f"usable={capacity.usable_bytes_per_device} B/device, "
                    f"over={capacity.over_budget_bytes_per_device} B/device; this excludes activations, "
                    "sampler/trace buffers, scratch, and allocator slack"
                )

        terminal = load_terminal_weights(checkpoint_root, c)
        layers = []
        shared_rope: dict[bool, tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]] = {}
        for layer_idx in indices:
            checkpoint = LayerCheckpoint(checkpoint_root, layer_idx, c)
            store = None
            use_cache = (
                deployment_mode == QUIETBOX2_PAGED_BFP4_POLICY
                and expert_cache_dir is not None
                and not expert_pool_keep_torch
            )
            if use_cache:
                layer_dir, key = expert_cache_dir_for(
                    expert_cache_dir, checkpoint_root, layer_idx, tp_degree=TP_DEGREE, dtype=ttnn.bfloat4_b
                )
                if ExpertHostStore.cache_valid(layer_dir, key, c.num_experts):
                    store = ExpertHostStore.from_cache(layer_dir, dtype=ttnn.bfloat4_b, num_experts=c.num_experts)
            state, w1, w2, w3 = checkpoint.materialize(include_experts=store is None)
            if use_cache and store is None:
                store = ExpertHostStore.from_weights(
                    mesh_device, w1, w2, w3, tp_degree=TP_DEGREE, shard_mapper=_flat_shard_mapper, dtype=ttnn.bfloat4_b
                )
                _save_expert_cache(store, layer_dir, key)
                w1 = w2 = w3 = None
            from .functional_decoder import HostLayerWeights

            rope_kind = bool(c.layer_kind(layer_idx).compress_ratio)
            layer = MultichipDecoder(
                mesh_device,
                layer_idx,
                HostLayerWeights(state, w1, w2, w3, expert_store=store),
                config=c,
                max_context=context,
                max_batch_size=max_batch_size,
                weights_dtype=ttnn.bfloat16,
                optimization_profile="optimized",
                deployment_precision=deployment_mode,
                moe_compute_profile=moe_compute_profile,
                dense_compute_profile=dense_compute_profile,
                shared_rope=shared_rope.get(rope_kind),
                expert_pool_size=expert_pool_size,
                expert_pool_keep_torch=expert_pool_keep_torch,
                attention_dtype=attention_dtype,
            )
            layers.append(layer)
            shared_rope.setdefault(rope_kind, (layer.rope_cos, layer.rope_sin, layer.rope_trans))
            del checkpoint, state, w1, w2, w3, store
        model = cls(
            mesh_device=mesh_device,
            config=c,
            layers=layers,
            layer_indices=indices,
            terminal=terminal,
            max_context=context,
            max_batch_size=max_batch_size,
            reduced_probe=reduced,
            deployment_mode=deployment_mode,
            moe_compute_profile=moe_compute_profile,
            dense_compute_profile=dense_compute_profile,
        )
        # streaming prefill block (128 = the qualified fork behaviour; 512 amortizes per-block dispatch).
        _streaming_blocks(STREAMING_PREFILL_BLOCK_SIZE, int(prefill_block_size))
        if int(prefill_block_size) > STREAMING_PREFILL_MAX_BLOCK:
            raise ValueError(f"prefill_block_size {prefill_block_size} exceeds {STREAMING_PREFILL_MAX_BLOCK}")
        model.prefill_block_size = int(prefill_block_size)
        return model

    def expert_pool_stats(self) -> dict[str, int] | None:
        """Cumulative expert-pool counters summed over layers (slot 0), or None without paging."""

        pools = [self._decoder_lane(layer, 0).moe.pool for layer in self.layers]
        if any(pool is None for pool in pools):
            return None
        stats = {
            "ensure_calls": sum(pool.ensure_calls for pool in pools),
            "requested": sum(pool.requested for pool in pools),
            "loads": sum(pool.loads for pool in pools),
        }
        stats["hits"] = stats["requested"] - stats["loads"]
        return stats

    def _decoder_lane(self, layer: MultichipDecoder, slot: int) -> MultichipDecoder:
        if not 0 <= slot < self.max_batch_size:
            raise ValueError(f"slot {slot} outside [0, {self.max_batch_size})")
        return layer if not layer._batch_lanes else layer._batch_lanes[slot]

    def _upload_token_ids(self, token_ids: torch.Tensor) -> ttnn.Tensor:
        return ttnn.from_torch(
            token_ids.to(torch.uint32).reshape(1, 1, 1, -1),
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _upload_position_ids(self, prompt_length: int) -> ttnn.Tensor:
        return ttnn.from_torch(
            torch.arange(prompt_length, dtype=torch.int32).reshape(1, -1),
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def embed_tokens(self, token_ids: ttnn.Tensor) -> ttnn.Tensor:
        """Vocab-sharded lookup followed by the one required TP all-gather."""

        hidden_shard = self.embedding.forward(token_ids)
        hidden = ttnn.all_gather(
            hidden_shard,
            dim=-1,
            num_links=CCL_NUM_LINKS,
            topology=ttnn.Topology.Ring,
            cluster_axis=TP_AXIS,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        seq_len = hidden.shape[-2]
        hidden = ttnn.reshape(hidden, [1, seq_len, 1, self.config.hidden_size])
        return ttnn.repeat(hidden, [1, 1, self.config.hc_mult, 1])

    def _ring_tensors(self, slot: int) -> list[ttnn.Tensor]:
        """The per-layer page rings of one slot: local window and raw compressor history (its KV / score /
        indexer views share one tensor). Everything else is full-context and one-to-one."""

        seen, rings = set(), []
        for layer in self.layers:
            cache = self._decoder_lane(layer, slot).cache
            for paged in (
                cache.local,
                cache.compressor_kv,
                cache.compressor_score,
                cache.indexer_compressor_kv,
                cache.indexer_compressor_score,
            ):
                if paged is not None and id(paged.tensor) not in seen:
                    seen.add(id(paged.tensor))
                    rings.append(paged.tensor)
        # the DSpark drafter's context rings follow the same positions
        return rings + list(getattr(self, "extra_ring_tensors", ()))

    def preallocate_ring_snapshot(self, slot: int = 0) -> None:
        """Allocate the snapshot buffers now, before any trace exists, so a later first snapshot does
        not allocate long-lived buffers behind a live trace."""

        if getattr(self, "_ring_snapshot_buffers", None) is None:
            self._ring_snapshot_buffers = [ttnn.clone(ring) for ring in self._ring_tensors(slot)]

    def _snapshot_rings(self, slot: int, start: int) -> None:
        """Copy every ring as it stands before the block at ``start`` (buffers allocated once)."""

        rings = self._ring_tensors(slot)
        buffers = getattr(self, "_ring_snapshot_buffers", None)
        if buffers is None or len(buffers) != len(rings):
            buffers = [ttnn.clone(ring) for ring in rings]
            self._ring_snapshot_buffers = buffers
        else:
            for ring, buffer in zip(rings, buffers, strict=True):
                ttnn.copy(ring, buffer)
        self._ring_snapshot = (slot, int(start))

    def ring_snapshot_start(self, slot: int = 0) -> int | None:
        """Block start of the valid ring snapshot of ``slot``, or None."""

        snapshot = getattr(self, "_ring_snapshot", None)
        return snapshot[1] if snapshot is not None and snapshot[0] == slot else None

    def invalidate_ring_snapshot(self) -> None:
        self._ring_snapshot = None

    def restore_ring_snapshot(self, slot: int = 0) -> int:
        """Copy the snapshot back into the rings; returns its block start. The snapshot stays valid only until
        the caller's next prefill takes or invalidates one."""

        start = self.ring_snapshot_start(slot)
        if start is None:
            raise RuntimeError("no valid ring snapshot for this slot")
        for ring, buffer in zip(self._ring_tensors(slot), self._ring_snapshot_buffers, strict=True):
            ttnn.copy(buffer, ring)
        return start

    def prepare_prefill(
        self,
        token_ids: torch.Tensor,
        *,
        prompt_lengths: Sequence[int] | None = None,
        slot_ids: Sequence[int] | None = None,
        query_chunk_size: int = 32,
        resume_start: int = 0,
        snapshot_rings: bool = False,
    ) -> FullModelPrefillState:
        """Own padding removal and build exact plans for mixed prompt lengths.

        ``resume_start`` / ``snapshot_rings`` (, one streaming lane): see PrefillLaneState.
        """

        if token_ids.ndim == 1:
            token_ids = token_ids.reshape(1, -1)
        if token_ids.ndim != 2:
            raise ValueError(f"token_ids must be [batch, padded_seq], got {tuple(token_ids.shape)}")
        batch = token_ids.shape[0]
        lengths = tuple(
            int(value) for value in ([token_ids.shape[1]] * batch if prompt_lengths is None else prompt_lengths)
        )
        slots = tuple(int(value) for value in (range(batch) if slot_ids is None else slot_ids))
        if len(lengths) != batch or len(slots) != batch:
            raise ValueError("prompt_lengths and slot_ids must match token_ids batch")
        if len(set(slots)) != len(slots):
            raise ValueError("slot_ids must be unique")
        lanes = []
        for row, (length, slot) in enumerate(zip(lengths, slots)):
            if not 1 <= length <= token_ids.shape[1] or length > self.max_context:
                raise ValueError(f"prompt length {length} outside supplied/allocated bounds")
            exact = token_ids[row, :length].contiguous()
            streaming = length > STREAMING_PREFILL_BLOCK_SIZE or self.prefill_pad_short
            execution_length = (
                max(_prefill_execution_length(length), STREAMING_PREFILL_BLOCK_SIZE) if streaming else length
            )
            execution_tokens = torch.nn.functional.pad(exact, (0, execution_length - length))
            uploaded_ids = self._upload_token_ids(execution_tokens)
            if streaming:
                # A full-context PrefillPlan owns O(layers * sequence) device
                # index tensors and the decoder's one-shot graph owns
                # O(sequence) residual streams.  Stream one HCA-aligned block
                # through every layer instead.  Each layer then carries its
                # causal state only in the already-resident paged caches.
                if resume_start:
                    if batch != 1:
                        raise ValueError("a resumed prefill takes exactly one prompt")
                    if (
                        getattr(self, "prefill_block_size", STREAMING_PREFILL_BLOCK_SIZE)
                        != STREAMING_PREFILL_BLOCK_SIZE
                    ):
                        raise ValueError("a resumed prefill needs the default 128-token streaming blocks")
                    if resume_start % STREAMING_PREFILL_BLOCK_SIZE or not 0 < resume_start < length:
                        raise ValueError(f"resume_start {resume_start} must be a block start inside the prompt")
                lanes.append(
                    PrefillLaneState(
                        slot,
                        length,
                        uploaded_ids,
                        (),
                        self._upload_position_ids(execution_length),
                        resume_start=int(resume_start),
                        snapshot_rings=bool(snapshot_rings),
                        host_token_ids=execution_tokens,
                    )
                )
                continue
            if resume_start:
                raise ValueError("only streaming (> 128-token) prompts can resume")
            plans = []
            for layer in self.layers:
                plan = self._decoder_lane(layer, slot).prepare_prefill(
                    length,
                    exact,
                    query_chunk_size=query_chunk_size,
                )
                if isinstance(plan, BatchedPrefillPlan):
                    raise RuntimeError("fixed-slot prefill must resolve to a single cache lane")
                plans.append(plan)
            lanes.append(PrefillLaneState(slot, length, uploaded_ids, tuple(plans)))
        return FullModelPrefillState(tuple(lanes), self.max_batch_size)

    def _hyper_head(self, streams: ttnn.Tensor, params: tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor] | None = None):
        """mHC collapse of the final streams; ``params`` = (fn, scale, base) of another head (the DSpark drafter's)."""

        c = self.config
        fn, scale, base = params or (self.hyper_head_fn, self.hyper_head_scale, self.hyper_head_base)
        seq_len = streams.shape[1]
        flat = ttnn.reshape(streams, [1, 1, seq_len, c.hc_width])
        flat = ttnn.rms_norm(ttnn.typecast(flat, ttnn.float32), epsilon=c.rms_norm_eps)
        mixes = ttnn.linear(
            flat,
            fn,
            dtype=ttnn.float32,
            compute_kernel_config=self.terminal_compute_kernel_config,
        )
        pre_logits = ttnn.add(ttnn.mul(mixes, scale), base)
        pre = ttnn.add(
            pre_logits,
            c.hc_eps,
            input_tensor_a_activations=[ttnn.UnaryOpType.SIGMOID],
        )
        collapsed = streamwise_hc_collapse(
            streams,
            pre,
            seq=seq_len,
            hc_mult=c.hc_mult,
            hidden_size=c.hidden_size,
        )
        return ttnn.reshape(ttnn.typecast(collapsed, ttnn.bfloat16), [1, 1, seq_len, c.hidden_size])

    def terminal_logits(self, streams: ttnn.Tensor, *, last_token_only: bool) -> ttnn.Tensor:
        hidden = self._hyper_head(streams)
        if last_token_only:
            seq_len = hidden.shape[-2]
            hidden = ttnn.slice(hidden, [0, 0, seq_len - 1, 0], [1, 1, seq_len, self.config.hidden_size])
        hidden = ttnn.rms_norm(hidden, weight=self.final_norm, epsilon=self.config.rms_norm_eps)
        return self.lm_head.forward(hidden)

    _PREFILL_INPUT_FIELDS = (
        "prev_pos",
        "local_pages",
        "raw_pages",
        "entry",
        "comp_pages",
        "index_pages",
        "bnd_prev_pos",
        "bnd_pos",
    )

    def _prefill_attention_replay(
        self,
        layer_idx: int,
        decoder,
        hidden: ttnn.Tensor,
        input_ids: ttnn.Tensor,
        position_ids: ttnn.Tensor,
        block_start: int,
        seq_len: int,
    ) -> ttnn.Tensor:
        """Run a middle block's attention half (MultichipDecoder.prefill_attention_segment) through a per-layer
        trace captured on first use: the block's hidden streams, token / position ids and per-block inputs are
        written into the trace's persistent buffers, then the trace replays. Returns a copy of the trace output.
        A live trace must not see new program compilations: callers warm the shapes first and release the traces
        (release_prefill_traces) before anything else may compile."""

        traces = self.__dict__.setdefault("_prefill_traces", {})
        host_inputs = decoder.prefill_block_host_inputs(block_start, seq_len)
        entry = traces.get(layer_idx)
        if "ids" not in traces:
            traces["ids"], traces["pos"] = ttnn.clone(input_ids), ttnn.clone(position_ids)
        else:
            ttnn.copy(input_ids, traces["ids"])
            ttnn.copy(position_ids, traces["pos"])
        if entry is None:
            entry = {"hidden": ttnn.clone(hidden), "inputs": decoder.prefill_block_device_inputs(host_inputs)}
            entry["tid"] = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
            entry["out"] = decoder.prefill_attention_segment(
                entry["hidden"], traces["ids"], traces["pos"], entry["inputs"]
            )
            ttnn.end_trace_capture(self.mesh_device, entry["tid"], cq_id=0)
            traces[layer_idx] = entry
        else:
            ttnn.copy(hidden, entry["hidden"])
            mapper = ttnn.ReplicateTensorToMesh(self.mesh_device)
            for name in self._PREFILL_INPUT_FIELDS:
                dst = getattr(entry["inputs"], name)
                if dst is not None:
                    ttnn.copy_host_to_device_tensor(
                        ttnn.from_torch(
                            host_inputs[name], dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=mapper
                        ),
                        dst,
                    )
        ttnn.execute_trace(self.mesh_device, entry["tid"], cq_id=0, blocking=False)
        # the trace's output buffer: read by this block's FFN half before the next replay of this layer overwrites it
        return entry["out"]

    def release_prefill_traces(self) -> None:
        """Release every prefill attention trace (keep before any program may compile)."""

        traces = self.__dict__.pop("_prefill_traces", {})
        for key, entry in traces.items():
            if isinstance(key, int):
                ttnn.release_trace(self.mesh_device, entry["tid"])

    def prefill_forward(
        self,
        state: FullModelPrefillState,
        *,
        return_all_logits: bool = False,
        capture_last_token_by_layer: dict[int, torch.Tensor] | None = None,
        tap_layers: Sequence[int] = (),
        tap_callback=None,
        tap_from: int = 0,
    ) -> ttnn.Tensor:
        """Run exact unpadded prompts and return vocab-sharded logits by active slot.

        ``tap_callback(block_start, valid_rows, taps)``: for every prompt block that ends past
        ``tap_from``, called after the block with the mean-pooled outputs of ``tap_layers`` ([1, rows, 1, H])."""

        if capture_last_token_by_layer is not None and len(state.lanes) != 1:
            raise ValueError("diagnostic layer capture requires exactly one prefill lane")

        def capture(key: int, hidden: ttnn.Tensor) -> None:
            if capture_last_token_by_layer is None:
                return
            seq_len = hidden.shape[1]
            last_token = ttnn.slice(
                hidden,
                [0, seq_len - 1, 0, 0],
                [1, seq_len, hidden.shape[2], hidden.shape[3]],
            )
            local_tensors = ttnn.get_device_tensors(last_token) or (last_token,)
            host = ttnn.to_torch(local_tensors[0])
            capture_last_token_by_layer[key] = host[:, -1].detach().float().cpu().clone()
            ttnn.deallocate(last_token)

        logits_by_slot = {}
        for lane in state.lanes:
            if lane.uses_streaming_blocks:
                if capture_last_token_by_layer is not None:
                    raise ValueError("diagnostic layer capture requires a prompt of at most 128 tokens")
                if return_all_logits:
                    raise ValueError("streaming prefill retains only the terminal prompt block")
                hidden = None
                execution_length = int(lane.token_ids.shape[-1])
                total_blocks = execution_length // STREAMING_PREFILL_BLOCK_SIZE
                blocks = _streaming_blocks(
                    execution_length, getattr(self, "prefill_block_size", STREAMING_PREFILL_BLOCK_SIZE)
                )
                for start, end in blocks:
                    if end <= lane.resume_start:
                        continue  # already in the caches (restored ring snapshot + untouched banks)
                    if lane.snapshot_rings and (start, end) == blocks[-1] and start >= STREAMING_PREFILL_BLOCK_SIZE:
                        self._snapshot_rings(lane.slot, start)
                    valid_seq_len = min(
                        end - start,
                        max(0, lane.prompt_length - start),
                    )
                    if valid_seq_len <= 0:
                        raise RuntimeError("streaming prefill produced an empty padded block")
                    if lane.host_token_ids is not None:
                        token_block = self._upload_token_ids(lane.host_token_ids[start:end])
                        position_ids = ttnn.from_torch(
                            torch.arange(start, end, dtype=torch.int32).reshape(1, -1),
                            device=self.mesh_device,
                            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                            dtype=ttnn.int32,
                            layout=ttnn.ROW_MAJOR_LAYOUT,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        )
                    else:
                        token_block = ttnn.slice(
                            lane.token_ids,
                            [0, 0, 0, start],
                            [1, 1, 1, end],
                        )
                        position_ids = ttnn.slice(lane.position_ids, [0, start], [1, end])
                    input_ids = ttnn.reshape(token_block, [1, end - start])
                    hidden = self.embed_tokens(token_block)
                    block_taps = {} if tap_callback is not None and end > tap_from else None
                    for layer_idx, layer in zip(self.layer_indices, self.layers, strict=True):
                        previous = hidden
                        decoder = self._decoder_lane(layer, lane.slot)
                        fixed_ok = getattr(self, "prefill_fixed_blocks", False) and decoder.prefill_fixed_block_ok(
                            start, end - start, valid_seq_len
                        )
                        if fixed_ok and getattr(self, "prefill_trace_attention", False):
                            # the attention half replays this layer's captured trace; the MoE half stays eager
                            streams = self._prefill_attention_replay(
                                layer_idx, decoder, hidden, input_ids, position_ids, start, end - start
                            )
                            hidden = decoder.prefill_ffn_half(streams, input_ids)
                        else:
                            fixed_inputs = None
                            if fixed_ok:
                                # a middle block takes the fixed-shape (trace-ready) attention path
                                fixed_inputs = decoder.prefill_block_device_inputs(
                                    decoder.prefill_block_host_inputs(start, end - start)
                                )
                            hidden = decoder.prefill_forward_block(
                                hidden,
                                input_ids,
                                position_ids,
                                block_start=start,
                                valid_seq_len=valid_seq_len,
                                fixed_inputs=fixed_inputs,
                            )
                        ttnn.deallocate(previous)
                        if block_taps is not None and layer_idx in tap_layers:
                            block_taps[layer_idx] = ttnn.mean(hidden, dim=2, keepdim=True)
                    if block_taps is not None:
                        tap_callback(start, valid_seq_len, block_taps)
                    ttnn.deallocate(position_ids)
                    ttnn.deallocate(input_ids)
                    ttnn.deallocate(token_block)
                    if end != execution_length:
                        ttnn.deallocate(hidden)
                        hidden = None
                    completed_blocks = end // STREAMING_PREFILL_BLOCK_SIZE
                    if _should_clear_streaming_program_cache(completed_blocks, total_blocks):
                        # Every live request tensor from this block has been
                        # released, and decode capture has not started yet.
                        # Synchronize before invalidating cached programs so
                        # no queued command still references their resources.
                        ttnn.synchronize_device(self.mesh_device)
                        before = ttnn.get_memory_view(self.mesh_device, ttnn.BufferType.DRAM)
                        entries_before = self.mesh_device.num_program_cache_entries()
                        self.mesh_device.clear_program_cache()
                        after = ttnn.get_memory_view(self.mesh_device, ttnn.BufferType.DRAM)
                        logger.info(
                            "DeepSeek streaming prefill bounded program cache at {} tokens: "
                            "entries {} -> {}, DRAM free {:.3f} -> {:.3f} GiB",
                            end,
                            entries_before,
                            self.mesh_device.num_program_cache_entries(),
                            before.total_bytes_free_per_bank * before.num_banks / 2**30,
                            after.total_bytes_free_per_bank * after.num_banks / 2**30,
                        )
                if hidden is None:
                    raise RuntimeError("streaming prefill produced no terminal block")
                if lane.prompt_length != execution_length:
                    terminal_offset = lane.prompt_length - 1 - blocks[-1][0]
                    terminal_hidden = ttnn.slice(
                        hidden,
                        [0, terminal_offset, 0, 0],
                        [1, terminal_offset + 1, self.config.hc_mult, self.config.hidden_size],
                    )
                    ttnn.deallocate(hidden)
                    hidden = terminal_hidden
            else:
                hidden = self.embed_tokens(lane.token_ids)
                capture(-1, hidden)
                block_taps = {} if tap_callback is not None else None
                for layer_idx, layer, plan in zip(self.layer_indices, self.layers, lane.layer_plans, strict=True):
                    hidden = self._decoder_lane(layer, lane.slot).prefill_forward(hidden, plan)
                    capture(layer_idx, hidden)
                    if block_taps is not None and layer_idx in tap_layers:
                        block_taps[layer_idx] = ttnn.mean(hidden, dim=2, keepdim=True)
                if block_taps is not None:
                    tap_callback(0, lane.prompt_length, block_taps)
            logits_by_slot[lane.slot] = self.terminal_logits(hidden, last_token_only=not return_all_logits)
        if not logits_by_slot:
            raise ValueError("prefill has no active lanes")
        if return_all_logits:
            if len(logits_by_slot) > 1:
                raise ValueError("mixed-length all-logits output is ragged; request per-lane output")
            return next(iter(logits_by_slot.values()))

        # Last-token logits use fixed batch rows: row N always belongs to slot
        # N, and inactive rows are explicit zeros.  The generator pads this to
        # the common sampler's 32-row tile without changing slot identity.
        template = next(iter(logits_by_slot.values()))
        inactive = ttnn.zeros_like(template)
        rows = [logits_by_slot.get(slot, inactive) for slot in range(self.max_batch_size)]
        return rows[0] if len(rows) == 1 else ttnn.concat(rows, dim=2)

    def decode_forward(
        self,
        token_ids: ttnn.Tensor,
        current_pos: ttnn.Tensor,
        *,
        slot: int = 0,
        increment_position: bool = True,
        capture_last_token_by_layer: dict[int, torch.Tensor] | None = None,
    ) -> ttnn.Tensor:
        """One low-level decode step with explicit token, position, and cache slot."""

        def capture(key: int, hidden: ttnn.Tensor) -> None:
            if capture_last_token_by_layer is None:
                return
            local_tensors = ttnn.get_device_tensors(hidden) or (hidden,)
            host = ttnn.to_torch(local_tensors[0])
            capture_last_token_by_layer[key] = host[:, -1].detach().float().cpu().clone()

        decoder_token_ids, hidden = self._decode_prologue(token_ids, slot)
        capture(-1, hidden)
        for layer_idx, layer in zip(self.layer_indices, self.layers, strict=True):
            decoder = self._decoder_lane(layer, slot)
            hidden = decoder.decode_forward(hidden, decoder_token_ids, current_pos)
            capture(layer_idx, hidden)
        return self._decode_epilogue(hidden, current_pos, slot=slot, increment_position=increment_position)

    def _decode_prologue(self, token_ids: ttnn.Tensor, slot: int) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        # The common sampler's persistent output contract is [1, 1, 1, 32]
        # with fixed batch slots on the last axis.  Slice exactly one slot
        # before embedding; a scalar direct-decode input is valid only for
        # slot zero.
        if token_ids.shape[-1] > 1:
            token_ids = ttnn.slice(token_ids, [0, 0, 0, slot], [1, 1, 1, slot + 1])
        elif slot:
            raise ValueError("a scalar decode token input can address only slot zero")
        return ttnn.reshape(token_ids, [1, 1]), self.embed_tokens(token_ids)

    def _decode_epilogue(
        self, hidden: ttnn.Tensor, current_pos: ttnn.Tensor, *, slot: int, increment_position: bool
    ) -> ttnn.Tensor:
        logits = self.terminal_logits(hidden, last_token_only=True)
        if logits.shape[-2] != 1:
            raise RuntimeError(f"decode terminal path must produce one logical row, got {logits.shape[-2]}")
        logits = ttnn.pad(logits, [(0, 0), (0, 0), (slot, 31 - slot), (0, 0)], 0.0)
        if increment_position:
            ttnn.plus_one(current_pos, skip_negative_entries=True)
        return logits

    @property
    def expert_paging(self) -> bool:
        return any(self._decoder_lane(layer, 0).moe.pool is not None for layer in self.layers)

    def verify_forward_block(
        self,
        token_ids: Sequence[int],
        start: int,
        *,
        slot: int = 0,
        tap_layers: Sequence[int] = (),
    ) -> tuple[ttnn.Tensor, dict[int, ttnn.Tensor]]:
        """Eager: decode M = len(token_ids) consecutive tokens at positions start .. start + M - 1 in
        one pass (speculative-decoding verify). Returns the vocab-sharded logits of all M rows and, for each layer
        in ``tap_layers``, that layer's output streams mean-pooled over the mHC lanes ([1, M, 1, H], the DSpark
        draft input). Does not move any device position counter; the caller decides the next start."""

        m = len(token_ids)
        ids = ttnn.from_torch(
            torch.tensor(list(token_ids), dtype=torch.int32).reshape(1, 1, 1, m),
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        hidden = self.embed_tokens(ids)
        input_ids = ttnn.reshape(ids, [1, m])
        taps = {}
        for layer_idx, layer in zip(self.layer_indices, self.layers, strict=True):
            hidden = self._decoder_lane(layer, slot).decode_forward_block(hidden, input_ids, start)
            if layer_idx in tap_layers:
                taps[layer_idx] = ttnn.mean(hidden, dim=2, keepdim=True)
        return self.terminal_logits(hidden, last_token_only=False), taps

    def verify_buffers(self, m: int, slot: int = 0) -> dict:
        """The persistent verify-MoE index buffers for M-token blocks, shared by every layer."""

        cache = self.__dict__.setdefault("_verify_buffers", {})
        if m not in cache:
            cache[m] = self._decoder_lane(self.layers[0], slot).moe.make_verify_buffers(m)
        return cache[m]

    def verify_forward_split(
        self,
        token_ids: Sequence[int],
        start: int,
        *,
        slot: int = 0,
        tap_layers: Sequence[int] = (),
    ) -> tuple[ttnn.Tensor, dict[int, ttnn.Tensor]]:
        """(eager, split like the paged decode trace): the verify block layer by layer, each layer cut at
        its router; between the halves the host reads the picks back, loads the experts and fills the shared
        verify buffers. Same results as verify_forward_block; the device work per segment is what a trace replays."""

        m = len(token_ids)
        buffers = self.verify_buffers(m, slot)
        ids = ttnn.from_torch(
            torch.tensor(list(token_ids), dtype=torch.int32).reshape(1, 1, 1, m),
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        start_tensor = ttnn.from_torch(
            torch.tensor([start], dtype=torch.int32),
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        hidden = self.embed_tokens(ids)
        input_ids = ttnn.reshape(ids, [1, m])
        decoders = [self._decoder_lane(layer, slot) for layer in self.layers]
        taps = {}
        for layer_idx, decoder in zip(self.layer_indices, decoders, strict=True):
            route = decoder.verify_route(hidden, input_ids, start_tensor)
            picks = ttnn.to_torch((ttnn.get_device_tensors(route.selected) or (route.selected,))[0])
            decoder.moe.fill_verify_buffers(picks.reshape(m, -1).to(torch.int64), buffers)
            hidden = decoder.verify_finish(route, buffers)
            if layer_idx in tap_layers:
                taps[layer_idx] = ttnn.mean(hidden, dim=2, keepdim=True)
        return self.terminal_logits(hidden, last_token_only=False), taps

    def capture_paged_verify_trace(
        self,
        m: int,
        *,
        slot: int = 0,
        tap_layers: Sequence[int] = (),
        warm_tokens: Sequence[int] | None = None,
        warm_start: int = 0,
        greedy: bool = False,
        tap_hook=None,
    ) -> FullModelPagedVerifyTrace:
        """Capture the M-token verify block as one trace segment per MoE boundary.

        Compiles every program first with one eager split pass on the persistent inputs (trace capture cannot
        compile); that pass writes the caches at warm_start .. warm_start + M - 1, which the caller should treat as
        clobbered (a later prefill or verify rewrites them)."""

        mapper = ttnn.ReplicateTensorToMesh(self.mesh_device)
        tokens = list(warm_tokens) if warm_tokens is not None else [0] * m
        token_ids = ttnn.from_torch(
            torch.tensor(tokens, dtype=torch.int32).reshape(1, 1, 1, m),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            mesh_mapper=mapper,
        )
        start = ttnn.from_torch(
            torch.tensor([warm_start], dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            mesh_mapper=mapper,
        )
        buffers = self.verify_buffers(m, slot)
        decoders = tuple(self._decoder_lane(layer, slot) for layer in self.layers)
        input_ids = ttnn.reshape(token_ids, [1, m])

        collected: dict = {}  # taps of earlier segments (device tensors the replay rewrites in order)

        def body(index, hidden_or_route, *, warm):
            """Segment ``index``: finish layer index - 1 (if any) and route layer ``index`` (or the epilogue).
            ``tap_hook(taps, start)`` runs in the epilogue with every tapped layer's rows."""
            taps = {}
            if index == 0:
                collected.clear()
                hidden = self.embed_tokens(token_ids)
            else:
                hidden = decoders[index - 1].verify_finish(hidden_or_route, buffers)
                if self.layer_indices[index - 1] in tap_layers:
                    taps[self.layer_indices[index - 1]] = ttnn.mean(hidden, dim=2, keepdim=True)
                    collected.update(taps)
            if index == len(decoders):
                logits = self.terminal_logits(hidden, last_token_only=False)
                if tap_hook is not None:
                    tap_hook(dict(collected), start)
                return (logits, self._verify_greedy_reduce(logits, m) if greedy else None), taps
            return decoders[index].verify_route(hidden, input_ids, start), taps

        def host_fill(index, route):
            picks = ttnn.to_torch((ttnn.get_device_tensors(route.selected) or (route.selected,))[0])
            decoders[index].moe.fill_verify_buffers(picks.reshape(m, -1).to(torch.int64), buffers)

        # warm-up (compiles every program on these exact persistent tensors)
        carry = None
        for index in range(len(decoders) + 1):
            carry, _ = body(index, carry, warm=True)
            if index < len(decoders):
                host_fill(index, carry)
        ttnn.synchronize_device(self.mesh_device)

        segments, routes, taps = [], [], {}
        carry = None
        try:
            for index in range(len(decoders) + 1):
                trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
                try:
                    carry, segment_taps = body(index, carry, warm=False)
                finally:
                    ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
                segments.append(trace_id)
                taps.update(segment_taps)
                if index < len(decoders):
                    routes.append(carry)
        except BaseException:
            for trace_id in segments:
                ttnn.release_trace(self.mesh_device, trace_id)
            raise
        trace_view = ttnn.get_memory_view(self.mesh_device, ttnn.BufferType.TRACE)
        logger.info(
            "paged verify trace (M={}) captured: {} segments, trace region used {:.1f} MiB per device",
            m,
            len(segments),
            trace_view.total_bytes_allocated_per_bank * trace_view.num_banks / 2**20,
        )
        logits, reduced = carry
        return FullModelPagedVerifyTrace(
            m, tuple(segments), tuple(routes), decoders, buffers, token_ids, start, logits, taps, reduced
        )

    def decode_forward_split(
        self,
        token_ids: ttnn.Tensor,
        current_pos: ttnn.Tensor,
        *,
        slot: int = 0,
        increment_position: bool = True,
    ) -> ttnn.Tensor:
        """Untraced decode through the split layers, filling expert pools between halves (S mode)."""

        decoder_token_ids, hidden = self._decode_prologue(token_ids, slot)
        for layer in self.layers:
            decoder = self._decoder_lane(layer, slot)
            route = decoder.decode_route(hidden, decoder_token_ids, current_pos)
            if decoder.moe.pool is not None:
                decoder.moe.ensure_resident(route.selected)
            eager = decoder.moe.pool_eager_ensure
            decoder.moe.pool_eager_ensure = False
            try:
                hidden = decoder.decode_finish(route)
            finally:
                decoder.moe.pool_eager_ensure = eager
        return self._decode_epilogue(hidden, current_pos, slot=slot, increment_position=increment_position)

    def capture_paged_decode_trace(
        self,
        token_ids: ttnn.Tensor,
        current_pos: ttnn.Tensor,
        *,
        slot: int = 0,
    ) -> FullModelPagedDecodeTrace:
        """Capture one trace segment per MoE boundary; the host fills expert pools between segments."""

        decoders = tuple(self._decoder_lane(layer, slot) for layer in self.layers)
        for decoder in decoders:
            decoder.moe.pool_eager_ensure = False
        captured_all: list[int] = []

        tap_layers = tuple(getattr(self, "decode_tap_layers", ()))
        # called inside the last segment with the step's taps and position (the drafter's context write)
        tap_hook = getattr(self, "decode_tap_hook", None)

        def capture_chain(shared_logits: ttnn.Tensor | None, shared_taps: dict | None = None):
            segments: list[int] = []
            routes: list[DecodeRouteState] = []
            taps: dict = {}

            def finish(i: int, route: DecodeRouteState) -> ttnn.Tensor:
                """Finish layer i; a tapped layer also publishes its mean-pooled output."""

                hidden = decoders[i].decode_finish(route)
                layer_idx = self.layer_indices[i]
                if layer_idx in tap_layers:
                    tap = ttnn.mean(hidden, dim=2, keepdim=True)
                    if shared_taps is not None:
                        ttnn.copy(tap, shared_taps[layer_idx])
                        tap = shared_taps[layer_idx]
                    taps[layer_idx] = tap
                return hidden

            def segment(body):
                trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
                try:
                    result = body()
                except BaseException:
                    ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
                    ttnn.release_trace(self.mesh_device, trace_id)
                    for captured in (*captured_all, *segments):
                        ttnn.release_trace(self.mesh_device, captured)
                    for decoder in decoders:
                        decoder.moe.pool_eager_ensure = True
                        decoder.compressor_pool_enabled = True
                    raise
                ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
                segments.append(trace_id)
                return result

            def first() -> tuple[ttnn.Tensor, DecodeRouteState]:
                ids, hidden = self._decode_prologue(token_ids, slot)
                return ids, decoders[0].decode_route(hidden, ids, current_pos)

            decoder_token_ids, route = segment(first)
            routes.append(route)
            for index in range(1, len(decoders)):
                previous = routes[-1]
                routes.append(
                    segment(
                        lambda i=index, r=previous: decoders[i].decode_route(
                            finish(i - 1, r), decoder_token_ids, current_pos
                        )
                    )
                )
            last = routes[-1]

            def epilogue() -> ttnn.Tensor:
                hidden = finish(len(decoders) - 1, last)
                if tap_hook is not None:
                    tap_hook(taps, current_pos)  # before the position advances: the taps belong to current_pos
                logits = self._decode_epilogue(hidden, current_pos, slot=slot, increment_position=True)
                if shared_logits is None:
                    return logits
                # Every variant publishes into the full chain's logits, which the sampler trace reads.
                ttnn.copy(logits, shared_logits)
                return shared_logits

            logits = segment(epilogue)
            captured_all.extend(segments)
            trace_view = ttnn.get_memory_view(self.mesh_device, ttnn.BufferType.TRACE)
            logger.info(
                "paged decode trace chain captured: {} segments, trace region used {:.1f} / {:.1f} MiB per device",
                len(segments),
                trace_view.total_bytes_allocated_per_bank * trace_view.num_banks / 2**20,
                (trace_view.total_bytes_allocated_per_bank + trace_view.total_bytes_free_per_bank)
                * trace_view.num_banks
                / 2**20,
            )
            return tuple(segments), tuple(routes), logits, taps

        if tap_layers:
            # Compile the tap reductions (trace capture cannot compile): one eager split pass at the capture
            # position, whose cache writes the replays overwrite (the caller restores the position).
            for decoder in decoders:
                decoder.moe.pool_eager_ensure = True
            ids, hidden = self._decode_prologue(token_ids, slot)
            route = decoders[0].decode_route(hidden, ids, current_pos)
            warm_taps = {}
            for index in range(1, len(decoders) + 1):
                hidden = decoders[index - 1].decode_finish(route)
                if self.layer_indices[index - 1] in tap_layers:
                    # the reduction, and the copy a variant chain publishes it with
                    warm_taps[self.layer_indices[index - 1]] = ttnn.mean(hidden, dim=2, keepdim=True)
                    ttnn.copy(warm_taps[self.layer_indices[index - 1]], ttnn.mean(hidden, dim=2, keepdim=True))
                if index < len(decoders):
                    route = decoders[index].decode_route(hidden, ids, current_pos)
            if tap_hook is not None:
                tap_hook(warm_taps, current_pos)
            self._decode_epilogue(hidden, current_pos, slot=slot, increment_position=False)
            for decoder in decoders:
                decoder.moe.pool_eager_ensure = False
        segments, routes, logits, taps = capture_chain(None)
        variants = {"full": (segments, routes)}
        ratios = sorted({int(d.kind.compress_ratio) for d in decoders if d.kind.compress_ratio})
        if getattr(self, "decode_compressor_variants", True) and len(ratios) == 2:
            csa_ratio, hca_ratio = ratios
            try:
                # Two chains fit the 256 MiB trace region (~107 MiB each): "full" for every token that completes a
                # CSA entry (and so every HCA one), "none" for the other three tokens in four.
                for key, pooled in (("none", set()),):
                    for decoder in decoders:
                        decoder.compressor_pool_enabled = int(decoder.kind.compress_ratio) in pooled
                    # Warm the variant's programs eagerly first (trace capture cannot compile). Its cache writes
                    # land at the capture position and are overwritten by replay; the caller restores the position.
                    for decoder in decoders:
                        decoder.moe.pool_eager_ensure = True
                    ids, hidden = self._decode_prologue(token_ids, slot)
                    route = decoders[0].decode_route(hidden, ids, current_pos)
                    for index in range(1, len(decoders)):
                        route = decoders[index].decode_route(decoders[index - 1].decode_finish(route), ids, current_pos)
                    warm = self._decode_epilogue(
                        decoders[-1].decode_finish(route), current_pos, slot=slot, increment_position=True
                    )
                    ttnn.copy(warm, logits)
                    for decoder in decoders:
                        decoder.moe.pool_eager_ensure = False
                    variant_segments, variant_routes, _, _ = capture_chain(logits, taps)
                    variants[key] = (variant_segments, variant_routes)
            finally:
                for decoder in decoders:
                    decoder.compressor_pool_enabled = True
            return FullModelPagedDecodeTrace(
                segments, routes, decoders, logits, variants=variants, ratios=(csa_ratio, hca_ratio), taps=taps
            )
        return FullModelPagedDecodeTrace(segments, routes, decoders, logits, taps=taps)

    def capture_decode_trace(
        self,
        token_ids: ttnn.Tensor,
        current_pos: ttnn.Tensor,
        *,
        slot: int = 0,
    ) -> FullModelDecodeTrace | FullModelPagedDecodeTrace:
        """Capture the model graph only; sampling owns a separate trace."""

        if self.expert_paging:
            return self.capture_paged_decode_trace(token_ids, current_pos, slot=slot)
        trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        try:
            logits = self.decode_forward(token_ids, current_pos, slot=slot, increment_position=True)
        except BaseException:
            ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
            ttnn.release_trace(self.mesh_device, trace_id)
            raise
        ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
        return FullModelDecodeTrace(trace_id, logits)

    def cache_state(self) -> dict[int, dict[int, Mapping[str, PagedTensor | None]]]:
        """Expose explicit fixed-slot cache and page-table handles to serving."""

        result: dict[int, dict[int, Mapping[str, PagedTensor | None]]] = {}
        names = (
            "local",
            "compressed",
            "compressor_kv",
            "compressor_score",
            "indexer",
            "indexer_compressor_kv",
            "indexer_compressor_score",
        )
        for layer_idx, layer in zip(self.layer_indices, self.layers):
            per_slot = {}
            for slot in range(self.max_batch_size):
                cache = self._decoder_lane(layer, slot).cache
                per_slot[slot] = {name: getattr(cache, name) for name in names}
            result[layer_idx] = per_slot
        return result

    def update_page_tables(
        self,
        updates: Mapping[tuple[int, int, str], Sequence[int]],
    ) -> int:
        """Refresh persistent trace-visible page tables in place.

        Keys are ``(layer_idx, slot, cache_bank)``.  Aliased compressor banks
        are copied once.  The cache storage never moves, so captured model
        traces remain valid and observe the changed mapping on their next
        replay.  The return value is the exact number of host-to-device metadata
        copies (page table plus row LUT for every rank).
        """

        state = self.cache_state()
        copied: set[int] = set()
        writes = 0
        for (layer_idx, slot, bank), values in updates.items():
            try:
                paged = state[int(layer_idx)][int(slot)][str(bank)]
            except KeyError as exc:
                raise KeyError(f"unknown cache page-table target {(layer_idx, slot, bank)}") from exc
            if paged is None:
                raise ValueError(f"cache bank {(layer_idx, slot, bank)} is absent for this layer kind")
            identity = id(paged)
            table = tuple(int(value) for value in values)
            if len(table) != len(paged.host_page_table):
                raise ValueError(
                    f"page table {(layer_idx, slot, bank)} has {len(table)} entries, "
                    f"expected {len(paged.host_page_table)}"
                )
            physical_pages = paged.rows // self.config.page_size
            if any(value < 0 or value >= physical_pages for value in table):
                raise ValueError(
                    f"page table {(layer_idx, slot, bank)} references storage outside {physical_pages} pages"
                )
            if identity in copied:
                if table != tuple(paged.host_page_table):
                    raise ValueError("conflicting updates were supplied for aliased cache banks")
                continue
            host_page_table = ttnn.from_torch(
                torch.tensor(table, dtype=torch.int32).reshape(1, -1),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            host_row_lut = ttnn.from_torch(
                # MultichipDecoder inherits the optimized/fused cache path,
                # whose gather contract stores the LUT as [1, rows].  Keep
                # the host replacement identical to that persistent shape so
                # it can be copied in place without invalidating a trace.
                torch.tensor(physical_row_lut(table, self.config.page_size), dtype=torch.int32).reshape(1, -1),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            local_page_tables = ttnn.get_device_tensors(paged.page_table) or (paged.page_table,)
            local_row_luts = ttnn.get_device_tensors(paged.row_lut) or (paged.row_lut,)
            if len(local_page_tables) != len(local_row_luts):
                raise RuntimeError("replicated page-table and row-LUT rank counts differ")
            for local_page_table, local_row_lut in zip(local_page_tables, local_row_luts):
                ttnn.copy_host_to_device_tensor(host_page_table, local_page_table)
                ttnn.copy_host_to_device_tensor(host_row_lut, local_row_lut)
            paged.host_page_table = table
            copied.add(identity)
            writes += 2 * len(local_page_tables)
        return writes

    def verify_greedy_tokens(self, logits: ttnn.Tensor, m: int) -> list[int]:
        """The greedy token of each of the first ``m`` rows of vocab-sharded verify logits.

        Each device reduces its own contiguous vocab slice (max value and its first index), and the host picks the
        device holding the larger value, the lower device on a tie, so the result is the first global argmax."""

        best, index = self._verify_greedy_reduce(logits, m)
        return self.greedy_tokens_from_reduced(best, index, m)

    def _verify_greedy_reduce(self, logits: ttnn.Tensor, m: int) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Per-device max and first argmax of the first ``m`` rows (device ops; captured into a verify trace when
        it is built with greedy=True, so a replay allocates nothing)."""

        local = int(logits.shape[-1])
        if local * TP_DEGREE != self.vocab_size:
            raise ValueError(f"verify logits hold {local} columns per device; expected {self.vocab_size // TP_DEGREE}")
        rows = ttnn.slice(logits, [0, 0, 0, 0], [1, 1, m, local])
        return ttnn.max(rows, dim=-1), ttnn.argmax(ttnn.to_layout(rows, ttnn.ROW_MAJOR_LAYOUT), dim=-1)

    def greedy_tokens_from_reduced(self, best: ttnn.Tensor, index: ttnn.Tensor, m: int) -> list[int]:
        local = self.vocab_size // TP_DEGREE
        values = torch.stack(
            [ttnn.to_torch(t).float().reshape(-1)[:m] for t in ttnn.get_device_tensors(best)]
        )  # [devices, m]
        indices = torch.stack(
            [ttnn.to_torch(t).to(torch.int64).reshape(-1)[:m] for t in ttnn.get_device_tensors(index)]
        )
        device = values.argmax(dim=0)  # torch returns the first maximum
        return [int(device[r]) * local + int(indices[device[r], r]) for r in range(m)]

    def verify_logits_for_host(self, logits: ttnn.Tensor, m: int) -> torch.Tensor:
        """The first ``m`` rows of vocab-sharded verify logits as a float32 host [m, V] tensor (each device
        holds a contiguous vocab slice; read and concatenated in device order)."""

        parts = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(logits)]
        full = torch.cat(parts, dim=-1)
        return full.reshape(-1, full.shape[-1])[:m, : self.vocab_size].float()

    def gather_logits_for_host(self, logits: ttnn.Tensor) -> torch.Tensor:
        """Explicit compatibility boundary; never used by optimized token-out."""

        gathered = ttnn.all_gather(
            logits,
            dim=3,
            num_links=CCL_NUM_LINKS,
            topology=ttnn.Topology.Ring,
            cluster_axis=TP_AXIS,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return ttnn.to_torch(ttnn.untilize(gathered), mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0))[
            0, 0, :, : self.vocab_size
        ]


__all__ = [
    "active_experimental_policy_overrides",
    "DeepSeekV4FlashModel",
    "FullModelCapacity",
    "FullModelCapacityError",
    "FullModelDecodeTrace",
    "FullModelPrefillState",
    "QUIETBOX2_CONTEXT",
    "QUIETBOX2_SELECTIVE_BFP4_DECODER_WEIGHT_BYTES_PER_DEVICE",
    "QUIETBOX2_QUANTIZED_DECODER_WEIGHT_BYTES_PER_DEVICE",
    "REPRESENTATIVE_LAYER_INDICES",
    "estimate_full_model_capacity",
    "load_terminal_weights",
    "require_locked_deployment_policy",
    "require_selected_decoder_policy",
]
