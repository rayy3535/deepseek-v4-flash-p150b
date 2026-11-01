"""Per-device optimized decoder for DeepSeek-V4-Flash-0731.

The optimized implementation starts from :class:`FusedDecoder` and owns every
runtime override that differs from the fused-stage graph.  Candidate profiles
exist only to make real-weight A/B experiments reproducible; ``optimized`` is
the delivered default and never selects the functional decoder.
"""

from __future__ import annotations

import math
import os
import struct
from dataclasses import dataclass, replace
from pathlib import Path

import torch

import ttnn

from ..config import AttentionKind, RouterKind
from .functional_decoder import FunctionalDecoder
from .fused_decoder import (
    BatchedPrefillPlan,
    DecoderCache,
    DecodeTrace,
    FusedDecoder,
    FusedDeepSeekSparseMoE,
    HostLayerWeights,
    PagedTensor,
    PrefillChunk,
    PrefillPlan,
)
from .sparse_moe import DeepSeekSparseMoE


def streamwise_hc_collapse(
    streams: ttnn.Tensor,
    pre: ttnn.Tensor,
    *,
    seq: int,
    hc_mult: int,
    hidden_size: int,
) -> ttnn.Tensor:
    """Collapse mHC lanes in FP32 without a tile-padded 64 MiB temporary."""

    streams_by_lane = ttnn.permute(streams, (0, 2, 1, 3))
    pre_by_lane = ttnn.permute(pre, (0, 3, 2, 1))
    collapsed = None
    for lane in range(hc_mult):
        lane_streams = ttnn.slice(
            streams_by_lane,
            [0, lane, 0, 0],
            [1, lane + 1, seq, hidden_size],
        )
        lane_pre = ttnn.slice(pre_by_lane, [0, lane, 0, 0], [1, lane + 1, seq, 1])
        weighted = ttnn.mul(
            ttnn.typecast(lane_streams, ttnn.float32),
            lane_pre,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        collapsed = weighted if collapsed is None else ttnn.add(collapsed, weighted)
    if collapsed is None:
        raise RuntimeError("mHC collapse requires at least one residual lane")
    return collapsed


@dataclass(frozen=True)
class OptimizationPolicy:
    """Named cumulative runtime contract used by an optimized candidate."""

    name: str
    sinkhorn_iters: int
    moe_weight_dtype: str = "bfp8"
    moe_fidelity: str = "hifi4"
    moe_grid: tuple[int, int] | None = None
    moe_in0_block_w: int = 1
    moe_out_subblock_w: int = 1
    sharded_decode_norms: bool = False
    dram_sharded_dense: bool = False
    moe_l1_decode: bool = False
    sinkhorn_dtype: str = "fp32"
    dense_weight_dtype: str = "bf16"
    dense_fidelity: str = "hifi2"
    cache_dtype: str = "bf16"
    batch_sinkhorn_iters: int | None = None
    dram_attention_input_cores: int | None = None
    dram_output_input_cores: int | None = None
    dram_in0_block_w: int | None = None
    packed_attention_projections: bool = True
    packed_moe_gate_up: bool = True
    grouped_output_dram_sharded: bool = False
    grouped_output_in0_block_w: int | None = None
    pack_two_attention_projections: bool = False
    carry_q_norm_shard: bool = False
    explicit_prefill_dense_program: bool = False
    explicit_prefill_attention_programs: bool = False
    moe_l1_prefill: bool = False
    shared_dram_sharded: bool = False
    hc_norm_l1: bool = False
    ffn_norm_l1: bool = False
    dram_dense_min_n: int = 4096


_POLICIES = {
    # Exact graph control used to keep candidate comparisons honest.
    "fused_control": OptimizationPolicy("fused_control", sinkhorn_iters=20),
    # Focused convergence probes.  These change only the number of device-side
    # 4x4 Sinkhorn row/column normalizations in each mHC pre block.
    "sinkhorn_1": OptimizationPolicy("sinkhorn_1", sinkhorn_iters=1),
    "sinkhorn_2": OptimizationPolicy("sinkhorn_2", sinkhorn_iters=2),
    "sinkhorn_4": OptimizationPolicy("sinkhorn_4", sinkhorn_iters=4),
    "sinkhorn_8": OptimizationPolicy("sinkhorn_8", sinkhorn_iters=8),
    "sinkhorn_10": OptimizationPolicy("sinkhorn_10", sinkhorn_iters=10),
    "sinkhorn_11": OptimizationPolicy("sinkhorn_11", sinkhorn_iters=11),
    "sinkhorn_12": OptimizationPolicy("sinkhorn_12", sinkhorn_iters=12),
    "sinkhorn_16": OptimizationPolicy("sinkhorn_16", sinkhorn_iters=16),
    # Expert-matmul sweeps.  The BFP8 row isolates the larger-block geometry;
    # the BFP4 rows then cross the mandatory low-precision policy with the
    # material 64/32/16-core shard geometries.
    "moe_bfp8_hifi4_g64_b16_s2": OptimizationPolicy(
        "moe_bfp8_hifi4_g64_b16_s2", 20, moe_grid=(8, 8), moe_in0_block_w=16, moe_out_subblock_w=2
    ),
    "moe_bfp4_lofi_g64_b16_s2": OptimizationPolicy(
        "moe_bfp4_lofi_g64_b16_s2",
        20,
        moe_weight_dtype="bfp4",
        moe_fidelity="lofi",
        moe_grid=(8, 8),
        moe_in0_block_w=16,
        moe_out_subblock_w=2,
    ),
    "moe_bfp4_lofi_g32_b16_s4": OptimizationPolicy(
        "moe_bfp4_lofi_g32_b16_s4",
        20,
        moe_weight_dtype="bfp4",
        moe_fidelity="lofi",
        moe_grid=(8, 4),
        moe_in0_block_w=16,
        moe_out_subblock_w=4,
    ),
    "moe_bfp4_lofi_g16_b16_s8": OptimizationPolicy(
        "moe_bfp4_lofi_g16_b16_s8",
        20,
        moe_weight_dtype="bfp4",
        moe_fidelity="lofi",
        moe_grid=(4, 4),
        moe_in0_block_w=16,
        moe_out_subblock_w=8,
    ),
    "cumulative_sinkhorn_10_moe_g32": OptimizationPolicy(
        "cumulative_sinkhorn_10_moe_g32",
        10,
        moe_weight_dtype="bfp4",
        moe_fidelity="lofi",
        moe_grid=(8, 4),
        moe_in0_block_w=16,
        moe_out_subblock_w=4,
    ),
    "cumulative_sinkhorn_11_moe_g32": OptimizationPolicy(
        "cumulative_sinkhorn_11_moe_g32",
        11,
        moe_weight_dtype="bfp4",
        moe_fidelity="lofi",
        moe_grid=(8, 4),
        moe_in0_block_w=16,
        moe_out_subblock_w=4,
    ),
    "cumulative_sinkhorn_12_moe_g32": OptimizationPolicy(
        "cumulative_sinkhorn_12_moe_g32",
        12,
        moe_weight_dtype="bfp4",
        moe_fidelity="lofi",
        moe_grid=(8, 4),
        moe_in0_block_w=16,
        moe_out_subblock_w=4,
    ),
    "cumulative_sinkhorn_11_moe_g32_sharded_norms": OptimizationPolicy(
        "cumulative_sinkhorn_11_moe_g32_sharded_norms",
        11,
        moe_weight_dtype="bfp4",
        moe_fidelity="lofi",
        moe_grid=(8, 4),
        moe_in0_block_w=16,
        moe_out_subblock_w=4,
        sharded_decode_norms=True,
    ),
    "cumulative_dram_sharded_dense": OptimizationPolicy(
        "cumulative_dram_sharded_dense",
        11,
        moe_weight_dtype="bfp4",
        moe_fidelity="lofi",
        moe_grid=(8, 4),
        moe_in0_block_w=16,
        moe_out_subblock_w=4,
        sharded_decode_norms=True,
        dram_sharded_dense=True,
    ),
    "cumulative_moe_l1": OptimizationPolicy(
        "cumulative_moe_l1",
        11,
        moe_weight_dtype="bfp4",
        moe_fidelity="lofi",
        moe_grid=(8, 4),
        moe_in0_block_w=16,
        moe_out_subblock_w=4,
        sharded_decode_norms=True,
        dram_sharded_dense=True,
        moe_l1_decode=True,
    ),
    "cumulative_bf16_sinkhorn": OptimizationPolicy(
        "cumulative_bf16_sinkhorn",
        11,
        moe_weight_dtype="bfp4",
        moe_fidelity="lofi",
        moe_grid=(8, 4),
        moe_in0_block_w=16,
        moe_out_subblock_w=4,
        sharded_decode_norms=True,
        dram_sharded_dense=True,
        moe_l1_decode=True,
        sinkhorn_dtype="bf16",
    ),
    "cumulative_dense_bfp8": OptimizationPolicy(
        "cumulative_dense_bfp8",
        11,
        moe_weight_dtype="bfp4",
        moe_fidelity="lofi",
        moe_grid=(8, 4),
        moe_in0_block_w=16,
        moe_out_subblock_w=4,
        sharded_decode_norms=True,
        dram_sharded_dense=True,
        moe_l1_decode=True,
        sinkhorn_dtype="bf16",
        dense_weight_dtype="bfp8",
    ),
    "cumulative_dense_bfp8_no_l1_moe": OptimizationPolicy(
        "cumulative_dense_bfp8_no_l1_moe",
        11,
        moe_weight_dtype="bfp4",
        moe_fidelity="lofi",
        moe_grid=(8, 4),
        moe_in0_block_w=16,
        moe_out_subblock_w=4,
        sharded_decode_norms=True,
        dram_sharded_dense=True,
        sinkhorn_dtype="bf16",
        dense_weight_dtype="bfp8",
    ),
    "cumulative_bfp8_cache": OptimizationPolicy(
        "cumulative_bfp8_cache",
        11,
        moe_weight_dtype="bfp4",
        moe_fidelity="lofi",
        moe_grid=(8, 4),
        moe_in0_block_w=16,
        moe_out_subblock_w=4,
        sharded_decode_norms=True,
        dram_sharded_dense=True,
        moe_l1_decode=True,
        sinkhorn_dtype="bf16",
        dense_weight_dtype="bfp8",
        cache_dtype="bfp8",
        batch_sinkhorn_iters=12,
    ),
    "cumulative_bfp8_cache_sinkhorn12": OptimizationPolicy(
        "cumulative_bfp8_cache_sinkhorn12",
        12,
        moe_weight_dtype="bfp4",
        moe_fidelity="lofi",
        moe_grid=(8, 4),
        moe_in0_block_w=16,
        moe_out_subblock_w=4,
        sharded_decode_norms=True,
        dram_sharded_dense=True,
        moe_l1_decode=True,
        sinkhorn_dtype="bf16",
        dense_weight_dtype="bfp8",
        cache_dtype="bfp8",
    ),
    "cumulative_bfp8_cache_no_l1_moe": OptimizationPolicy(
        "cumulative_bfp8_cache_no_l1_moe",
        11,
        moe_weight_dtype="bfp4",
        moe_fidelity="lofi",
        moe_grid=(8, 4),
        moe_in0_block_w=16,
        moe_out_subblock_w=4,
        sharded_decode_norms=True,
        dram_sharded_dense=True,
        sinkhorn_dtype="bf16",
        dense_weight_dtype="bfp8",
        cache_dtype="bfp8",
    ),
    "cumulative_bfp8_cache_dense_lofi": OptimizationPolicy(
        "cumulative_bfp8_cache_dense_lofi",
        11,
        moe_weight_dtype="bfp4",
        moe_fidelity="lofi",
        moe_grid=(8, 4),
        moe_in0_block_w=16,
        moe_out_subblock_w=4,
        sharded_decode_norms=True,
        dram_sharded_dense=True,
        moe_l1_decode=True,
        sinkhorn_dtype="bf16",
        dense_weight_dtype="bfp8",
        dense_fidelity="lofi",
        cache_dtype="bfp8",
        batch_sinkhorn_iters=12,
    ),
    "cumulative_bfp8_cache_dense_lofi_c32": OptimizationPolicy(
        "cumulative_bfp8_cache_dense_lofi_c32",
        11,
        moe_weight_dtype="bfp4",
        moe_fidelity="lofi",
        moe_grid=(8, 4),
        moe_in0_block_w=16,
        moe_out_subblock_w=4,
        sharded_decode_norms=True,
        dram_sharded_dense=True,
        moe_l1_decode=True,
        sinkhorn_dtype="bf16",
        dense_weight_dtype="bfp8",
        dense_fidelity="lofi",
        cache_dtype="bfp8",
        batch_sinkhorn_iters=12,
        dram_attention_input_cores=32,
        dram_output_input_cores=32,
    ),
    "cumulative_bfp8_cache_dense_lofi_c16": OptimizationPolicy(
        "cumulative_bfp8_cache_dense_lofi_c16",
        11,
        moe_weight_dtype="bfp4",
        moe_fidelity="lofi",
        moe_grid=(8, 4),
        moe_in0_block_w=16,
        moe_out_subblock_w=4,
        sharded_decode_norms=True,
        dram_sharded_dense=True,
        moe_l1_decode=True,
        sinkhorn_dtype="bf16",
        dense_weight_dtype="bfp8",
        dense_fidelity="lofi",
        cache_dtype="bfp8",
        batch_sinkhorn_iters=12,
        dram_attention_input_cores=16,
        dram_output_input_cores=16,
    ),
    # Real-weight attention precision trial required by the optimization
    # checklist.  It keeps the winning topology/geometry fixed so that the
    # BFP4 result is directly comparable with the BFP8 candidate above.
    "cumulative_attention_bfp4_lofi_c16": OptimizationPolicy(
        "cumulative_attention_bfp4_lofi_c16",
        11,
        moe_weight_dtype="bfp4",
        moe_fidelity="lofi",
        moe_grid=(8, 4),
        moe_in0_block_w=16,
        moe_out_subblock_w=4,
        sharded_decode_norms=True,
        dram_sharded_dense=True,
        moe_l1_decode=True,
        sinkhorn_dtype="bf16",
        dense_weight_dtype="bfp4",
        dense_fidelity="lofi",
        cache_dtype="bfp8",
        batch_sinkhorn_iters=12,
        dram_attention_input_cores=16,
        dram_output_input_cores=16,
    ),
    # Precision-locked controls for the material DRAM-sharded dense geometry.
    # These repeat the 32/16/8-core sweep under the delivered BFP4/LoFi policy
    # instead of borrowing the earlier BFP8 geometry result.
    "cumulative_attention_bfp4_lofi_c32": OptimizationPolicy(
        "cumulative_attention_bfp4_lofi_c32",
        11,
        moe_weight_dtype="bfp4",
        moe_fidelity="lofi",
        moe_grid=(8, 4),
        moe_in0_block_w=16,
        moe_out_subblock_w=4,
        sharded_decode_norms=True,
        dram_sharded_dense=True,
        moe_l1_decode=True,
        sinkhorn_dtype="bf16",
        dense_weight_dtype="bfp4",
        dense_fidelity="lofi",
        cache_dtype="bfp8",
        batch_sinkhorn_iters=12,
        dram_attention_input_cores=32,
        dram_output_input_cores=32,
    ),
    "cumulative_attention_bfp4_lofi_c8_b16": OptimizationPolicy(
        "cumulative_attention_bfp4_lofi_c8_b16",
        11,
        moe_weight_dtype="bfp4",
        moe_fidelity="lofi",
        moe_grid=(8, 4),
        moe_in0_block_w=16,
        moe_out_subblock_w=4,
        sharded_decode_norms=True,
        dram_sharded_dense=True,
        moe_l1_decode=True,
        sinkhorn_dtype="bf16",
        dense_weight_dtype="bfp4",
        dense_fidelity="lofi",
        cache_dtype="bfp8",
        batch_sinkhorn_iters=12,
        dram_attention_input_cores=8,
        dram_output_input_cores=8,
        dram_in0_block_w=16,
    ),
    "cumulative_attention_bfp4_lofi_c8_b8": OptimizationPolicy(
        "cumulative_attention_bfp4_lofi_c8_b8",
        11,
        moe_weight_dtype="bfp4",
        moe_fidelity="lofi",
        moe_grid=(8, 4),
        moe_in0_block_w=16,
        moe_out_subblock_w=4,
        sharded_decode_norms=True,
        dram_sharded_dense=True,
        moe_l1_decode=True,
        sinkhorn_dtype="bf16",
        dense_weight_dtype="bfp4",
        dense_fidelity="lofi",
        cache_dtype="bfp8",
        batch_sinkhorn_iters=12,
        dram_attention_input_cores=8,
        dram_output_input_cores=8,
        dram_in0_block_w=8,
    ),
    "cumulative_attention_bfp4_lofi_c8_b1": OptimizationPolicy(
        "cumulative_attention_bfp4_lofi_c8_b1",
        11,
        moe_weight_dtype="bfp4",
        moe_fidelity="lofi",
        moe_grid=(8, 4),
        moe_in0_block_w=16,
        moe_out_subblock_w=4,
        sharded_decode_norms=True,
        dram_sharded_dense=True,
        moe_l1_decode=True,
        sinkhorn_dtype="bf16",
        dense_weight_dtype="bfp4",
        dense_fidelity="lofi",
        cache_dtype="bfp8",
        batch_sinkhorn_iters=12,
        dram_attention_input_cores=8,
        dram_output_input_cores=8,
        dram_in0_block_w=1,
    ),
    # The grouped O-A projection is a true [1,8,32,4096] x
    # [1,8,4096,1024] batched matmul.  These candidates use TTNN's specialized
    # batch-sharded DRAM program and sweep the material K-block sizes.
    "cumulative_bfp4_grouped_dram_b16": OptimizationPolicy(
        "cumulative_bfp4_grouped_dram_b16",
        11,
        moe_weight_dtype="bfp4",
        moe_fidelity="lofi",
        moe_grid=(8, 4),
        moe_in0_block_w=16,
        moe_out_subblock_w=4,
        sharded_decode_norms=True,
        dram_sharded_dense=True,
        moe_l1_decode=True,
        sinkhorn_dtype="bf16",
        dense_weight_dtype="bfp4",
        dense_fidelity="lofi",
        cache_dtype="bfp8",
        batch_sinkhorn_iters=12,
        dram_attention_input_cores=16,
        dram_output_input_cores=16,
        grouped_output_dram_sharded=True,
        grouped_output_in0_block_w=16,
    ),
    "cumulative_bfp4_grouped_dram_b32": OptimizationPolicy(
        "cumulative_bfp4_grouped_dram_b32",
        11,
        moe_weight_dtype="bfp4",
        moe_fidelity="lofi",
        moe_grid=(8, 4),
        moe_in0_block_w=16,
        moe_out_subblock_w=4,
        sharded_decode_norms=True,
        dram_sharded_dense=True,
        moe_l1_decode=True,
        sinkhorn_dtype="bf16",
        dense_weight_dtype="bfp4",
        dense_fidelity="lofi",
        cache_dtype="bfp8",
        batch_sinkhorn_iters=12,
        dram_attention_input_cores=16,
        dram_output_input_cores=16,
        grouped_output_dram_sharded=True,
        grouped_output_in0_block_w=32,
    ),
    "cumulative_bfp4_grouped_dram_b64": OptimizationPolicy(
        "cumulative_bfp4_grouped_dram_b64",
        11,
        moe_weight_dtype="bfp4",
        moe_fidelity="lofi",
        moe_grid=(8, 4),
        moe_in0_block_w=16,
        moe_out_subblock_w=4,
        sharded_decode_norms=True,
        dram_sharded_dense=True,
        moe_l1_decode=True,
        sinkhorn_dtype="bf16",
        dense_weight_dtype="bfp4",
        dense_fidelity="lofi",
        cache_dtype="bfp8",
        batch_sinkhorn_iters=12,
        dram_attention_input_cores=16,
        dram_output_input_cores=16,
        grouped_output_dram_sharded=True,
        grouped_output_in0_block_w=64,
    ),
    "cumulative_bfp4_grouped_dram_b128": OptimizationPolicy(
        "cumulative_bfp4_grouped_dram_b128",
        11,
        moe_weight_dtype="bfp4",
        moe_fidelity="lofi",
        moe_grid=(8, 4),
        moe_in0_block_w=16,
        moe_out_subblock_w=4,
        sharded_decode_norms=True,
        dram_sharded_dense=True,
        moe_l1_decode=True,
        sinkhorn_dtype="bf16",
        dense_weight_dtype="bfp4",
        dense_fidelity="lofi",
        cache_dtype="bfp8",
        batch_sinkhorn_iters=12,
        dram_attention_input_cores=16,
        dram_output_input_cores=16,
        grouped_output_dram_sharded=True,
        grouped_output_in0_block_w=128,
    ),
    "cumulative_split_attention_bfp4_lofi_c16": OptimizationPolicy(
        "cumulative_split_attention_bfp4_lofi_c16",
        11,
        moe_weight_dtype="bfp4",
        moe_fidelity="lofi",
        moe_grid=(8, 4),
        moe_in0_block_w=16,
        moe_out_subblock_w=4,
        sharded_decode_norms=True,
        dram_sharded_dense=True,
        moe_l1_decode=True,
        sinkhorn_dtype="bf16",
        dense_weight_dtype="bfp4",
        dense_fidelity="lofi",
        cache_dtype="bfp8",
        batch_sinkhorn_iters=12,
        dram_attention_input_cores=16,
        dram_output_input_cores=16,
        packed_attention_projections=False,
    ),
    "cumulative_split_moe_bfp4_lofi_c16": OptimizationPolicy(
        "cumulative_split_moe_bfp4_lofi_c16",
        11,
        moe_weight_dtype="bfp4",
        moe_fidelity="lofi",
        moe_grid=(8, 4),
        moe_in0_block_w=16,
        moe_out_subblock_w=4,
        sharded_decode_norms=True,
        dram_sharded_dense=True,
        moe_l1_decode=True,
        sinkhorn_dtype="bf16",
        dense_weight_dtype="bfp4",
        dense_fidelity="lofi",
        cache_dtype="bfp8",
        batch_sinkhorn_iters=12,
        dram_attention_input_cores=16,
        dram_output_input_cores=16,
        packed_moe_gate_up=False,
    ),
    "cumulative_bfp8_cache_dense_lofi_c8": OptimizationPolicy(
        "cumulative_bfp8_cache_dense_lofi_c8",
        11,
        moe_weight_dtype="bfp4",
        moe_fidelity="lofi",
        moe_grid=(8, 4),
        moe_in0_block_w=16,
        moe_out_subblock_w=4,
        sharded_decode_norms=True,
        dram_sharded_dense=True,
        moe_l1_decode=True,
        sinkhorn_dtype="bf16",
        dense_weight_dtype="bfp8",
        dense_fidelity="lofi",
        cache_dtype="bfp8",
        batch_sinkhorn_iters=12,
        dram_attention_input_cores=8,
        dram_output_input_cores=8,
        dram_in0_block_w=1,
    ),
    # Delivered winner after the review-remediation sweeps: BFP4/LoFi 8-core
    # block-16 dense rows, specialized grouped O-A, legal layer-0 packing,
    # phase-local Q-norm sharding, L1 prefill MoE, and DRAM-sharded shared
    # experts.  Every flag below has a matched real-weight control artifact.
    "optimized": OptimizationPolicy(
        "optimized",
        11,
        moe_weight_dtype="bfp4",
        moe_fidelity="lofi",
        moe_grid=(8, 4),
        moe_in0_block_w=16,
        moe_out_subblock_w=4,
        sharded_decode_norms=True,
        dram_sharded_dense=True,
        moe_l1_decode=True,
        sinkhorn_dtype="bf16",
        dense_weight_dtype="bfp4",
        dense_fidelity="lofi",
        cache_dtype="bfp8",
        batch_sinkhorn_iters=12,
        dram_attention_input_cores=8,
        dram_output_input_cores=8,
        dram_in0_block_w=16,
        grouped_output_dram_sharded=True,
        grouped_output_in0_block_w=16,
        pack_two_attention_projections=True,
        carry_q_norm_shard=True,
        moe_l1_prefill=True,
        shared_dram_sharded=True,
        dram_dense_min_n=2560,
    ),
}


# Review-remediation candidates build cumulatively from the measured grouped
# O-A winner.  ``replace`` keeps the precision/topology lock explicit while
# avoiding long copies that can accidentally drift in unrelated fields.
_POLICIES.update(
    {
        "cumulative_bfp4_c8_grouped_b16": replace(
            _POLICIES["cumulative_bfp4_grouped_dram_b16"],
            name="cumulative_bfp4_c8_grouped_b16",
            dram_attention_input_cores=8,
            dram_output_input_cores=8,
            dram_in0_block_w=16,
        ),
        "cumulative_bfp4_c8_grouped_b16_pack2": replace(
            _POLICIES["cumulative_bfp4_grouped_dram_b16"],
            name="cumulative_bfp4_c8_grouped_b16_pack2",
            dram_attention_input_cores=8,
            dram_output_input_cores=8,
            dram_in0_block_w=16,
            pack_two_attention_projections=True,
        ),
        "cumulative_bfp4_c8_grouped_b16_qcarry": replace(
            _POLICIES["cumulative_bfp4_grouped_dram_b16"],
            name="cumulative_bfp4_c8_grouped_b16_qcarry",
            dram_attention_input_cores=8,
            dram_output_input_cores=8,
            dram_in0_block_w=16,
            pack_two_attention_projections=True,
            carry_q_norm_shard=True,
        ),
        "cumulative_bfp4_c8_grouped_b16_prefill2d": replace(
            _POLICIES["cumulative_bfp4_grouped_dram_b16"],
            name="cumulative_bfp4_c8_grouped_b16_prefill2d",
            dram_attention_input_cores=8,
            dram_output_input_cores=8,
            dram_in0_block_w=16,
            pack_two_attention_projections=True,
            carry_q_norm_shard=True,
            explicit_prefill_dense_program=True,
            explicit_prefill_attention_programs=True,
        ),
        "cumulative_bfp4_c8_grouped_b16_prefill_moe_l1": replace(
            _POLICIES["cumulative_bfp4_grouped_dram_b16"],
            name="cumulative_bfp4_c8_grouped_b16_prefill_moe_l1",
            dram_attention_input_cores=8,
            dram_output_input_cores=8,
            dram_in0_block_w=16,
            pack_two_attention_projections=True,
            carry_q_norm_shard=True,
            moe_l1_prefill=True,
        ),
        "cumulative_bfp4_c8_grouped_b16_shared_dram": replace(
            _POLICIES["cumulative_bfp4_grouped_dram_b16"],
            name="cumulative_bfp4_c8_grouped_b16_shared_dram",
            dram_attention_input_cores=8,
            dram_output_input_cores=8,
            dram_in0_block_w=16,
            pack_two_attention_projections=True,
            carry_q_norm_shard=True,
            moe_l1_prefill=True,
            shared_dram_sharded=True,
        ),
        "cumulative_bfp4_c8_grouped_b16_l1_carry": replace(
            _POLICIES["cumulative_bfp4_grouped_dram_b16"],
            name="cumulative_bfp4_c8_grouped_b16_l1_carry",
            dram_attention_input_cores=8,
            dram_output_input_cores=8,
            dram_in0_block_w=16,
            pack_two_attention_projections=True,
            carry_q_norm_shard=True,
            hc_norm_l1=True,
            ffn_norm_l1=True,
        ),
        "cumulative_bfp4_c8_grouped_b16_layer3_n2560": replace(
            _POLICIES["cumulative_bfp4_grouped_dram_b16"],
            name="cumulative_bfp4_c8_grouped_b16_layer3_n2560",
            dram_attention_input_cores=8,
            dram_output_input_cores=8,
            dram_in0_block_w=16,
            pack_two_attention_projections=True,
            carry_q_norm_shard=True,
            dram_dense_min_n=2560,
        ),
    }
)
_POLICIES.update(
    {
        "cumulative_final_hc_l1": replace(
            _POLICIES["optimized"],
            name="cumulative_final_hc_l1",
            hc_norm_l1=True,
        ),
        "cumulative_final_ffn_l1": replace(
            _POLICIES["optimized"],
            name="cumulative_final_ffn_l1",
            ffn_norm_l1=True,
        ),
    }
)


def optimization_policy(name: str) -> OptimizationPolicy:
    try:
        return _POLICIES[name]
    except KeyError as exc:
        raise ValueError(f"unknown DeepSeek optimized-decoder policy {name!r}; choose {sorted(_POLICIES)}") from exc


def _optimized_sparse_program(
    *, k: int, n: int, grid: tuple[int, int], in0_block_w: int, out_subblock_w: int
) -> ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig:
    """Build a legal explicit decode sparse-matmul geometry."""

    grid_x, grid_y = grid
    cores = grid_x * grid_y
    k_tiles = math.ceil(k / 32)
    n_tiles = math.ceil(n / 32)
    if k_tiles % in0_block_w:
        raise ValueError(f"K tiles {k_tiles} must be divisible by in0_block_w {in0_block_w}")
    if n_tiles % cores:
        raise ValueError(f"N tiles {n_tiles} must be divisible by grid core count {cores}")
    per_core_n = n_tiles // cores
    if per_core_n % out_subblock_w:
        raise ValueError(f"per_core_N {per_core_n} must be divisible by subblock {out_subblock_w}")
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(grid_x, grid_y),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=out_subblock_w,
        out_block_h=1,
        out_block_w=out_subblock_w,
        per_core_M=1,
        per_core_N=per_core_n,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )


def _largest_divisor_at_most(value: int, cap: int) -> int:
    return max(divisor for divisor in range(1, min(value, cap) + 1) if value % divisor == 0)


def _optimized_prefill_program(
    *,
    m: int,
    k: int,
    n: int,
    max_grid: ttnn.CoreCoord,
) -> ttnn.MatmulMultiCoreReuseMultiCastProgramConfig:
    """Build an exact-divisor 2D program for logical non-aligned prefill."""

    m_tiles = math.ceil(m / 32)
    k_tiles = math.ceil(k / 32)
    n_tiles = math.ceil(n / 32)
    grid_x = _largest_divisor_at_most(n_tiles, max_grid.x)
    grid_y = _largest_divisor_at_most(m_tiles, max_grid.y)
    per_core_m = m_tiles // grid_y
    per_core_n = n_tiles // grid_x
    # Large-N prefill rows are dominated by the in1 double buffer.  A larger
    # K block can exceed Blackhole's 1.5 MiB/core even with BFP4 weights (for
    # example 64x1024x32768 at block 16 needs 2.80 MiB).  Block 2 is the
    # largest common divisor that remains legal across the material rows.
    in0_block_w = _largest_divisor_at_most(k_tiles, 2)
    out_subblock_w = _largest_divisor_at_most(per_core_n, 8)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(grid_x, grid_y),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )


class OptimizedDeepSeekSparseMoE(FusedDeepSeekSparseMoE):
    """Fused active-expert MoE with an optional decode-only L1 data path."""

    use_l1_decode = False
    use_l1_prefill = False

    def configure_shared_dram_sharding(self, *, input_cores: int, in0_block_w_cap: int) -> None:
        """Shard the two material shared-expert projections over DRAM banks."""

        self._shared_dram_programs = {}
        self._shared_input_memories = {}
        dram_size = self.mesh_device.dram_grid_size()
        dram_cores = dram_size.x * dram_size.y
        dram_grid = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_size.x - 1, dram_size.y - 1))}
        )
        max_workers = self.mesh_device.compute_with_storage_grid_size()
        for attribute in ("shared_gate_up", "shared_down"):
            weight = getattr(self, attribute)
            k, n = weight.shape[-2], weight.shape[-1]
            if n % (32 * dram_cores):
                raise ValueError(f"shared {attribute} width {n} must shard exactly over {dram_cores} DRAM banks")
            weight_memory = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.DRAM,
                ttnn.ShardSpec(
                    dram_grid,
                    (k, n // dram_cores),
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            )
            setattr(self, attribute, ttnn.to_memory_config(weight, weight_memory))
            k_tiles = k // 32
            cores = max(divisor for divisor in range(1, input_cores + 1) if k_tiles % divisor == 0)
            tiles_per_core = k_tiles // cores
            block = _largest_divisor_at_most(tiles_per_core, in0_block_w_cap)
            self._shared_dram_programs[attribute] = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                in0_block_w=block,
                per_core_M=1,
                per_core_N=(n // 32) // 64,
                fused_activation=None,
            )
            core_grid = ttnn.num_cores_to_corerangeset(cores, max_workers, row_wise=True)
            self._shared_input_memories[attribute] = ttnn.create_sharded_memory_config_(
                shape=(32, k // cores),
                core_grid=core_grid,
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )

    def _shared_linear(self, x: ttnn.Tensor, attribute: str) -> ttnn.Tensor:
        programs = getattr(self, "_shared_dram_programs", {})
        if attribute not in programs:
            return ttnn.linear(
                x,
                getattr(self, attribute),
                dtype=ttnn.bfloat16,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                compute_kernel_config=self.compute_kernel_config,
            )
        shape = tuple(x.shape)
        rows = math.prod(shape[:-1])
        k, n = tuple(getattr(self, attribute).shape)[-2:]
        flat = x if shape[:-2] == (1, 1) else ttnn.reshape(x, [1, 1, rows, k])
        sharded = ttnn.to_memory_config(flat, self._shared_input_memories[attribute])
        output = ttnn.linear(
            sharded,
            getattr(self, attribute),
            program_config=programs[attribute],
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute_kernel_config,
        )
        output = ttnn.sharded_to_interleaved(output, ttnn.L1_MEMORY_CONFIG)
        result_shape = (*shape[:-1], n)
        return output if tuple(output.shape) == result_shape else ttnn.reshape(output, result_shape)

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        selected_experts: ttnn.Tensor,
        routing_weights: ttnn.Tensor,
    ) -> ttnn.Tensor:
        seq = hidden_states.shape[2]
        if seq > 1 and self.use_l1_prefill:
            return self._forward_l1_prefill(hidden_states, selected_experts, routing_weights)
        if not self.use_l1_decode or seq != 1:
            return super().forward(hidden_states, selected_experts, routing_weights)

        if tuple(hidden_states.shape) != (1, 1, seq, self.hidden_size):
            raise ValueError(f"MoE input must be [1,1,S,{self.hidden_size}]")
        expected_routing = (1, 1, seq, self.top_k)
        if tuple(selected_experts.shape) != expected_routing or tuple(routing_weights.shape) != expected_routing:
            raise ValueError(f"selected experts and weights must both be {expected_routing}")

        hidden_l1 = ttnn.to_memory_config(hidden_states, ttnn.L1_MEMORY_CONFIG)
        indices = ttnn.to_layout(
            ttnn.reshape(selected_experts, [1, 1, 1, self.top_k]),
            ttnn.ROW_MAJOR_LAYOUT,
        )
        common = {
            "sparsity": self.sparsity,
            "indices": indices,
            # Indexed/gather mode loops exactly over the six UINT16 ids in
            # ``indices``. TTNN rejects ``nnz`` together with ``indices``;
            # nnz is only the non-indexed sparsity-scan contract.
            "is_input_b_sparse": True,
            "memory_config": ttnn.L1_MEMORY_CONFIG,
            "output_tile": ttnn.Tile([32, 32]),
            "dtype": ttnn.bfloat16,
            "compute_kernel_config": self.compute_kernel_config,
        }
        gate_up = ttnn.sparse_matmul(
            hidden_l1,
            self.routed_gate_up,
            program_config=self.gate_up_program,
            **common,
        )
        gate_up = ttnn.reshape(gate_up, (self.top_k, 1, 2 * self.intermediate_size))
        gate = ttnn.slice(gate_up, [0, 0, 0], [self.top_k, 1, self.intermediate_size])
        up = ttnn.slice(
            gate_up,
            [0, 0, self.intermediate_size],
            [self.top_k, 1, 2 * self.intermediate_size],
        )
        activated = ttnn.reshape(self._swiglu(gate, up), (1, self.top_k, 1, self.intermediate_size))
        routed = ttnn.sparse_matmul(
            activated,
            self.routed_down,
            program_config=self.down_program,
            is_input_a_sparse=True,
            **common,
        )
        routed = ttnn.reshape(routed, (self.top_k, 1, self.hidden_size))
        routed = ttnn.matmul(
            ttnn.reshape(routing_weights, (1, 1, 1, self.top_k)),
            ttnn.reshape(routed, (1, 1, self.top_k, self.hidden_size)),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )

        shared_gate_up = self._shared_linear(hidden_l1, "shared_gate_up")
        shared_gate = ttnn.slice(shared_gate_up, [0, 0, 0, 0], [1, 1, 1, self.intermediate_size])
        shared_up = ttnn.slice(
            shared_gate_up,
            [0, 0, 0, self.intermediate_size],
            [1, 1, 1, 2 * self.intermediate_size],
        )
        shared = self._swiglu(shared_gate, shared_up)
        shared = self._shared_linear(shared, "shared_down")
        return ttnn.add(routed, shared)

    def _forward_l1_prefill(
        self,
        hidden_states: ttnn.Tensor,
        selected_experts: ttnn.Tensor,
        routing_weights: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """Carry one <=32-token MoE chunk through the indexed L1 graph."""

        seq = hidden_states.shape[2]
        if tuple(hidden_states.shape) != (1, 1, seq, self.hidden_size) or not 1 < seq <= 32:
            raise ValueError(f"prefill MoE input must be [1,1,S,{self.hidden_size}] with 1 < S <= 32")
        expected_routing = (1, 1, seq, self.top_k)
        if tuple(selected_experts.shape) != expected_routing or tuple(routing_weights.shape) != expected_routing:
            raise ValueError(f"selected experts and weights must both be {expected_routing}")
        hidden_l1 = ttnn.to_memory_config(hidden_states, ttnn.L1_MEMORY_CONFIG)
        common = {
            "sparsity": self.sparsity,
            "is_input_b_sparse": True,
            "memory_config": ttnn.L1_MEMORY_CONFIG,
            "output_tile": ttnn.Tile([32, 32]),
            "dtype": ttnn.bfloat16,
            "compute_kernel_config": self.compute_kernel_config,
        }
        routed_tokens = []
        for token_idx in range(seq):
            token = ttnn.slice(
                hidden_l1,
                [0, 0, token_idx, 0],
                [1, 1, token_idx + 1, self.hidden_size],
            )
            indices = ttnn.slice(
                selected_experts,
                [0, 0, token_idx, 0],
                [1, 1, token_idx + 1, self.top_k],
            )
            indices = ttnn.to_layout(
                ttnn.reshape(indices, [1, 1, 1, self.top_k]),
                ttnn.ROW_MAJOR_LAYOUT,
            )
            gate_up = ttnn.sparse_matmul(
                token,
                self.routed_gate_up,
                indices=indices,
                program_config=self.gate_up_program,
                **common,
            )
            gate_up = ttnn.reshape(gate_up, (self.top_k, 1, 2 * self.intermediate_size))
            gate = ttnn.slice(gate_up, [0, 0, 0], [self.top_k, 1, self.intermediate_size])
            up = ttnn.slice(
                gate_up,
                [0, 0, self.intermediate_size],
                [self.top_k, 1, 2 * self.intermediate_size],
            )
            activated = ttnn.reshape(self._swiglu(gate, up), (1, self.top_k, 1, self.intermediate_size))
            routed = ttnn.sparse_matmul(
                activated,
                self.routed_down,
                indices=indices,
                program_config=self.down_program,
                is_input_a_sparse=True,
                **common,
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
        shared_gate_up = self._shared_linear(hidden_l1, "shared_gate_up")
        shared_gate = ttnn.slice(shared_gate_up, [0, 0, 0, 0], [1, 1, seq, self.intermediate_size])
        shared_up = ttnn.slice(
            shared_gate_up,
            [0, 0, 0, self.intermediate_size],
            [1, 1, seq, 2 * self.intermediate_size],
        )
        shared = self._swiglu(shared_gate, shared_up)
        shared = self._shared_linear(shared, "shared_down")
        return ttnn.add(routed, shared)


class OptimizedSeparateDeepSeekSparseMoE(DeepSeekSparseMoE):
    """Equally tuned legal split gate/up candidate for packed-MoE A/B."""

    use_l1_decode = False

    def _swiglu(self, gate: ttnn.Tensor, up: ttnn.Tensor) -> ttnn.Tensor:
        gate = ttnn.clamp(gate, min=None, max=self.limit, output_tensor=gate)
        up = ttnn.clamp(up, min=-self.limit, max=self.limit, output_tensor=up)
        return ttnn.mul(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        selected_experts: ttnn.Tensor,
        routing_weights: ttnn.Tensor,
    ) -> ttnn.Tensor:
        if not self.use_l1_decode or hidden_states.shape[2] != 1:
            return super().forward(hidden_states, selected_experts, routing_weights)

        hidden_l1 = ttnn.to_memory_config(hidden_states, ttnn.L1_MEMORY_CONFIG)
        indices = ttnn.to_layout(
            ttnn.reshape(selected_experts, [1, 1, 1, self.top_k]),
            ttnn.ROW_MAJOR_LAYOUT,
        )
        common = {
            "sparsity": self.sparsity,
            "indices": indices,
            # Indexed mode derives its exact loop count from this top-k index
            # stick and must not also receive ``nnz``.
            "is_input_b_sparse": True,
            "memory_config": ttnn.L1_MEMORY_CONFIG,
            "output_tile": ttnn.Tile([32, 32]),
            "dtype": ttnn.bfloat16,
            "compute_kernel_config": self.compute_kernel_config,
        }
        gate = ttnn.sparse_matmul(hidden_l1, self.weights.gate, program_config=self.gate_up_program, **common)
        up = ttnn.sparse_matmul(hidden_l1, self.weights.up, program_config=self.gate_up_program, **common)
        gate = ttnn.reshape(gate, (self.top_k, 1, self.intermediate_size))
        up = ttnn.reshape(up, (self.top_k, 1, self.intermediate_size))
        activated = ttnn.reshape(self._swiglu(gate, up), (1, self.top_k, 1, self.intermediate_size))
        routed = ttnn.sparse_matmul(
            activated,
            self.weights.down,
            program_config=self.down_program,
            is_input_a_sparse=True,
            **common,
        )
        routed = ttnn.reshape(routed, (self.top_k, 1, self.hidden_size))
        routed = ttnn.matmul(
            ttnn.reshape(routing_weights, (1, 1, 1, self.top_k)),
            ttnn.reshape(routed, (1, 1, self.top_k, self.hidden_size)),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        shared_gate = ttnn.linear(
            hidden_l1,
            self.weights.shared_gate,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        shared_up = ttnn.linear(
            hidden_l1,
            self.weights.shared_up,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        shared = self._swiglu(shared_gate, shared_up)
        shared = ttnn.linear(
            shared,
            self.weights.shared_down,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        return ttnn.add(routed, shared)


class OptimizedDecoder(FusedDecoder):
    """Fused decoder plus evidence-selected per-device optimizations."""

    def __init__(self, *args, optimization_profile: str | None = None, **kwargs) -> None:
        profile = optimization_profile or os.environ.get("DSV4_OPT_PROFILE", "optimized")
        self.optimization_policy = optimization_policy(profile)
        self._sinkhorn_iters = self.optimization_policy.sinkhorn_iters
        super().__init__(*args, **kwargs)
        self.dense_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(),
            math_fidelity={"hifi2": ttnn.MathFidelity.HiFi2, "lofi": ttnn.MathFidelity.LoFi}[
                self.optimization_policy.dense_fidelity
            ],
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        compact_guards: dict[int, ttnn.Tensor] = {}
        for decoder in (self, *self._batch_lanes):
            decoder.dense_compute_kernel_config = self.dense_compute_kernel_config
            # Capacity is storage, not a numerical-policy input.  Batched
            # entrypoints select the iteration count from the active logical
            # batch before constructing/capturing that device graph.
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

    def _set_active_batch_sinkhorn(self, batch_size: int) -> None:
        iterations = (
            self.optimization_policy.batch_sinkhorn_iters
            if batch_size > 1 and self.optimization_policy.batch_sinkhorn_iters is not None
            else self.optimization_policy.sinkhorn_iters
        )
        for lane in self._batch_lanes[:batch_size]:
            lane._sinkhorn_iters = iterations

    def _page_tensor(self, *args, **kwargs) -> PagedTensor:
        """Allocate paged cache at the selected evidence-backed precision."""

        paged = super()._page_tensor(*args, **kwargs)
        if self.optimization_policy.cache_dtype == "bfp8":
            compact = ttnn.typecast(paged.tensor, ttnn.bfloat8_b)
            ttnn.deallocate(paged.tensor)
            paged.tensor = compact
        return paged

    @staticmethod
    def _paged_fill_cache(target: PagedTensor, value: ttnn.Tensor, page_table: ttnn.Tensor) -> None:
        """Honor paged-fill's raw-copy dtype contract without changing decode updates."""

        fill = value if value.dtype == target.tensor.dtype else ttnn.typecast(value, target.tensor.dtype)
        ttnn.experimental.paged_fill_cache(target.tensor, fill, page_table, batch_idx=0)

    def _fill_prefill_cache(
        self,
        kv: ttnn.Tensor,
        compressed: ttnn.Tensor | None,
        indexer: ttnn.Tensor | None,
        plan: PrefillPlan,
    ) -> None:
        """Fill every optimized cache with an exactly matching tile dtype."""

        local = ttnn.slice(
            kv,
            [0, 0, plan.local_fill_start, 0],
            [1, 1, min(plan.seq_len, plan.local_fill_start + plan.local_fill_length), self.config.head_dim],
        )
        missing = plan.local_fill_length - local.shape[2]
        if missing:
            local = ttnn.pad(local, [(0, 0), (0, 0), (0, missing), (0, 0)], 0.0)
        self._paged_fill_cache(self.cache.local, local, plan.local_fill_page_table)
        if compressed is not None:
            padded = math.ceil(compressed.shape[2] / self.config.page_size) * self.config.page_size
            fill = ttnn.pad(compressed, [(0, 0), (0, 0), (0, padded - compressed.shape[2]), (0, 0)], 0.0)
            self._paged_fill_cache(self.cache.compressed, fill, plan.compressed_fill_page_table)
        if indexer is not None:
            padded = math.ceil(indexer.shape[2] / self.config.page_size) * self.config.page_size
            fill = ttnn.pad(indexer, [(0, 0), (0, 0), (0, padded - indexer.shape[2]), (0, 0)], 0.0)
            index_pt = ttnn.slice(self.cache.indexer.page_table, [0, 0], [1, padded // self.config.page_size])
            self._paged_fill_cache(self.cache.indexer, fill, index_pt)

    def _fill_raw_prefill_state(
        self,
        x: ttnn.Tensor,
        positions: ttnn.Tensor,
        kv_weight: ttnn.Tensor,
        gate_weight: ttnn.Tensor,
        ape: ttnn.Tensor,
        kv_cache: PagedTensor,
        score_cache: PagedTensor,
        plan: PrefillPlan,
    ) -> None:
        """Persist packed compressor state with the cache's exact fill dtype."""

        cache = getattr(self, "_fused_prefill_projections", {})
        projected = cache.pop((id(kv_weight), id(gate_weight)), None)
        if projected is None:
            slot = ttnn.remainder(positions, self.kind.compress_ratio)
            bias = ttnn.embedding(ttnn.typecast(slot, ttnn.uint32), ape, layout=ttnn.TILE_LAYOUT)
            bias = ttnn.reshape(bias, [1, 1, plan.seq_len, kv_weight.shape[-1]])
            packed_cache = getattr(self, "_fused_attention_projections", {})
            try:
                raw_kv = packed_cache.pop((id(kv_weight), id(gate_weight)))
                raw_score_projection = packed_cache.pop((id(gate_weight), id(kv_weight)))
            except KeyError as exc:
                raise RuntimeError("raw compressor state escaped the fused shared-LHS graph") from exc
            raw_score = ttnn.add(raw_score_projection, bias)
        else:
            raw_kv, raw_score = projected

        if kv_cache is not score_cache:
            raise RuntimeError("compressor histories escaped the fused paged-cache allocation")
        if self.kind.attention is AttentionKind.CSA and kv_weight is self.comp_kv:
            self._fused_prefill_history_pending = (raw_kv, raw_score, kv_cache)
            return
        if self.kind.attention is AttentionKind.CSA:
            try:
                attention_kv, attention_score, attention_cache = self._fused_prefill_history_pending
            except AttributeError as exc:
                raise RuntimeError("CSA compressor histories escaped their fused prefill update") from exc
            del self._fused_prefill_history_pending
            if attention_cache is not kv_cache:
                raise RuntimeError("CSA compressor histories escaped the shared paged tensor")
            packed = ttnn.concat([attention_kv, attention_score, raw_kv, raw_score], dim=-1)
        else:
            packed = ttnn.concat([raw_kv, raw_score], dim=-1)

        tail = ttnn.slice(
            packed,
            [0, 0, plan.compressor_fill_start, 0],
            [1, 1, plan.seq_len, packed.shape[-1]],
        )
        missing = plan.compressor_fill_length - tail.shape[2]
        if missing:
            tail = ttnn.pad(tail, [(0, 0), (0, 0), (0, missing), (0, 0)], 0.0)
        self._paged_fill_cache(kv_cache, tail, plan.compressor_fill_page_table)

    def _upload_dense_weights(self, state) -> None:
        """Select packed/split projections, then tune material decode weights."""

        policy = self.optimization_policy
        if self.optimization_policy.packed_attention_projections:
            super()._upload_dense_weights(state)
            if policy.pack_two_attention_projections and self._packed_attention_projection is None:
                projections = (self.q_a, self.kv)
                start = 0
                ranges: dict[int, tuple[int, int]] = {}
                for projection in projections:
                    end = start + projection.shape[-1]
                    ranges[id(projection)] = (start, end)
                    start = end
                self._packed_attention_projection = ttnn.concat(projections, dim=-1)
                self._packed_projection_ranges = ranges
                for projection in projections:
                    ttnn.deallocate(projection)
        else:
            # Preserve the fused stage's scale folding while retaining every
            # original projection for the legal same-input split A/B.
            FunctionalDecoder._upload_dense_weights(self, state)
            p = self._prefix

            def fold_hc_scale(fn_suffix: str, scale_suffix: str, attr: str, scale_attr: str) -> None:
                scale = state[f"{p}.{scale_suffix}"].float().reshape(-1)
                expanded = torch.cat(
                    [
                        scale[0].expand(self.config.hc_mult),
                        scale[1].expand(self.config.hc_mult),
                        scale[2].expand(self.config.hc_mix_width - 2 * self.config.hc_mult),
                    ]
                )
                folded = state[f"{p}.{fn_suffix}"].float() * expanded[:, None]
                ttnn.deallocate(getattr(self, attr))
                ttnn.deallocate(getattr(self, scale_attr))
                setattr(self, attr, self._weight(folded, dtype=ttnn.float32))
                setattr(self, scale_attr, None)

            fold_hc_scale("hc_attn_fn", "hc_attn_scale", "hc_attn_fn", "hc_attn_scale")
            fold_hc_scale("hc_ffn_fn", "hc_ffn_scale", "hc_ffn_fn", "hc_ffn_scale")
            if self.kind.attention is AttentionKind.CSA:
                index_scale = self.config.index_head_dim**-0.5 * self.config.index_n_heads**-0.5
                ttnn.deallocate(self.index_weight)
                self.index_weight = self._weight(state[f"{p}.attn.indexer.weights_proj.weight"].float() * index_scale)
            self._packed_attention_projection = None
            self._packed_projection_ranges = {}
            self._packed_qr_projection = None
            self._packed_qr_projection_ranges = {}
        self._dram_linear_weight_ids: set[int] = set()
        self._dram_interleaved_weights: dict[int, ttnn.Tensor] = {}
        self._dram_linear_roles: dict[int, str] = {}
        self._optimized_dense_weight_ids: set[int] = set()
        self._grouped_output_interleaved_weight: ttnn.Tensor | None = None
        self._grouped_output_input_memory: ttnn.MemoryConfig | None = None
        self._grouped_output_output_memory: ttnn.MemoryConfig | None = None
        self._grouped_output_program: ttnn.MatmulMultiCoreReuseMultiCastBatchedDRAMShardedProgramConfig | None = None
        dense_attributes: list[str] = []
        if self._packed_attention_projection is not None:
            dense_attributes.append("_packed_attention_projection")
        else:
            dense_attributes.extend(("q_a", "kv"))
            if self.kind.compress_ratio:
                dense_attributes.extend(("comp_kv", "comp_gate"))
            if self.kind.attention is AttentionKind.CSA:
                dense_attributes.extend(("index_comp_kv", "index_comp_gate", "index_weight"))
        dense_attributes.append("_packed_qr_projection" if self._packed_qr_projection is not None else "q_b")
        if self._packed_qr_projection is None and self.kind.attention is AttentionKind.CSA:
            dense_attributes.append("index_q")
        dense_attributes.extend(("o_a", "o_b"))
        dense_dtype = {
            "bfp8": ttnn.bfloat8_b,
            "bfp4": ttnn.bfloat4_b,
        }.get(self.optimization_policy.dense_weight_dtype)
        if dense_dtype is not None:
            for attribute in dense_attributes:
                weight = getattr(self, attribute)
                quantized = ttnn.typecast(weight, dense_dtype)
                ttnn.deallocate(weight)
                setattr(self, attribute, quantized)
                self._optimized_dense_weight_ids.add(id(quantized))
        if not self.optimization_policy.dram_sharded_dense:
            return
        attributes = (
            ["_packed_attention_projection"]
            if self._packed_attention_projection is not None
            else [attribute for attribute in dense_attributes if attribute not in ("o_a", "o_b")]
        )
        attributes.append("_packed_qr_projection" if self._packed_qr_projection is not None else "q_b")
        attributes.append("o_b")
        for attribute in attributes:
            weight = getattr(self, attribute, None)
            if (
                weight is None
                or id(weight) in self._dram_linear_weight_ids
                or weight.shape[-1] < policy.dram_dense_min_n
            ):
                continue
            k, n = weight.shape[-2], weight.shape[-1]
            dram_grid = self.mesh_device.dram_grid_size()
            dram_cores = dram_grid.x
            grid = ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_grid.x - 1, dram_grid.y - 1))}
            )
            shard_width = math.ceil(n / (32 * dram_cores)) * 32
            memory_config = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.DRAM,
                ttnn.ShardSpec(grid, (k, shard_width), ttnn.ShardOrientation.ROW_MAJOR),
            )
            optimized_dense = id(weight) in self._optimized_dense_weight_ids
            sharded = ttnn.to_memory_config(weight, memory_config)
            setattr(self, attribute, sharded)
            self._dram_linear_weight_ids.add(id(sharded))
            self._dram_interleaved_weights[id(sharded)] = weight
            self._dram_linear_roles[id(sharded)] = attribute
            if optimized_dense:
                self._optimized_dense_weight_ids.add(id(sharded))
        if policy.grouped_output_dram_sharded:
            grouped_weight = self.o_a
            grouped_batch, grouped_k, grouped_n = tuple(grouped_weight.shape)[-3:]
            worker_cores = self.mesh_device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
            num_banks = len(worker_cores)
            if grouped_batch % num_banks:
                raise ValueError(f"grouped O-A batch {grouped_batch} must divide evenly over {num_banks} DRAM banks")
            batches_per_core = grouped_batch // num_banks
            worker_grid = ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(core.x, core.y), ttnn.CoreCoord(core.x, core.y))
                    for core in worker_cores
                ]
            )
            dram_grid_size = self.mesh_device.dram_grid_size()
            dram_grid = ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(
                        ttnn.CoreCoord(0, 0),
                        ttnn.CoreCoord(dram_grid_size.x - 1, dram_grid_size.y - 1),
                    )
                }
            )
            grouped_weight_memory = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.DRAM,
                ttnn.ShardSpec(
                    dram_grid,
                    (batches_per_core * grouped_k, grouped_n),
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            )
            self._grouped_output_interleaved_weight = grouped_weight
            grouped_weight = ttnn.to_memory_config(grouped_weight, grouped_weight_memory)
            self.o_a = grouped_weight
            self._optimized_dense_weight_ids.add(id(grouped_weight))
            self._grouped_output_input_memory = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    worker_grid,
                    (batches_per_core * 32, grouped_k),
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            )
            self._grouped_output_output_memory = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    worker_grid,
                    (batches_per_core * 32, grouped_n),
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            )
            grouped_block = policy.grouped_output_in0_block_w
            if grouped_block is None or (grouped_k // 32) % grouped_block:
                raise ValueError(f"grouped O-A K tiles {grouped_k // 32} must be divisible by block {grouped_block}")
            self._grouped_output_program = ttnn.MatmulMultiCoreReuseMultiCastBatchedDRAMShardedProgramConfig(
                in0_block_w=grouped_block,
                per_core_M=1,
                per_core_N=grouped_n // 32,
                fused_activation=None,
            )

    def _optimized_linear(self, x: ttnn.Tensor, weight: ttnn.Tensor) -> ttnn.Tensor:
        """Run selected large projections with L1 activations/DRAM-sharded weights."""

        compute = self.dense_compute_kernel_config if id(weight) in self._optimized_dense_weight_ids else None
        shape = tuple(x.shape)
        rows = math.prod(shape[:-1])
        padded_rows = math.ceil(rows / 32) * 32
        if (
            self.optimization_policy.explicit_prefill_dense_program
            and padded_rows > 32
            and id(weight) in self._optimized_dense_weight_ids
        ):
            interleaved_weight = self._dram_interleaved_weights.get(id(weight), weight)
            program = _optimized_prefill_program(
                m=padded_rows,
                k=interleaved_weight.shape[-2],
                n=interleaved_weight.shape[-1],
                max_grid=self.mesh_device.compute_with_storage_grid_size(),
            )
            return ttnn.linear(
                x,
                interleaved_weight,
                program_config=program,
                compute_kernel_config=compute,
            )
        if id(weight) not in self._dram_linear_weight_ids:
            return ttnn.linear(x, weight, compute_kernel_config=compute)
        if padded_rows != 32:
            return ttnn.linear(x, self._dram_interleaved_weights[id(weight)], compute_kernel_config=compute)
        k, n = weight.shape[-2], weight.shape[-1]
        k_tiles = k // 32
        padded_n_tiles = math.ceil(n / (32 * self.mesh_device.dram_grid_size().x)) * self.mesh_device.dram_grid_size().x
        max_cores = (
            self.mesh_device.compute_with_storage_grid_size().x * self.mesh_device.compute_with_storage_grid_size().y
        )
        role = self._dram_linear_roles[id(weight)]
        requested_input_cores = (
            self.optimization_policy.dram_output_input_cores
            if role == "o_b"
            else self.optimization_policy.dram_attention_input_cores
        )
        o_b_input_cores_override = getattr(self, "_o_b_input_cores_override", None) if role == "o_b" else None
        if o_b_input_cores_override is not None:
            requested_input_cores = o_b_input_cores_override
        elif self.kind.attention is AttentionKind.CSA:
            requested_input_cores = max(requested_input_cores or 0, 16)
        input_core_cap = min(64, max_cores, requested_input_cores or max_cores)
        input_cores = max(count for count in range(1, input_core_cap + 1) if k_tiles % count == 0)
        output_cores = max(count for count in range(1, min(64, max_cores) + 1) if padded_n_tiles % count == 0)
        tiles_per_input_core = k_tiles // input_cores
        # Logical prefill rows 2..32 share the one-tile M geometry with decode
        # but retain more live attention state.  Keep their K block at two to
        # avoid colliding with layer-kind-specific persistent L1 allocations;
        # the measured decode row retains the precision-locked block-16 winner.
        in0_block_w_cap = (
            2
            if rows > 1
            else (8 if self.kind.attention is AttentionKind.CSA else (self.optimization_policy.dram_in0_block_w or 8))
        )
        in0_block_w = max(
            divisor
            for divisor in range(1, min(in0_block_w_cap, tiles_per_input_core) + 1)
            if tiles_per_input_core % divisor == 0
        )
        program_config = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=in0_block_w,
            per_core_M=padded_rows // 32,
            per_core_N=padded_n_tiles // output_cores,
            fused_activation=None,
        )
        grid = ttnn.num_cores_to_corerangeset(
            input_cores,
            self.mesh_device.compute_with_storage_grid_size(),
            row_wise=True,
        )
        input_memory = ttnn.create_sharded_memory_config_(
            shape=(padded_rows, k // input_cores),
            core_grid=grid,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        flat = x if shape[:-2] == (1, 1) else ttnn.reshape(x, [1, 1, rows, k])
        sharded_x = (
            flat
            if flat.is_sharded() and flat.memory_config() == input_memory
            else ttnn.to_memory_config(flat, input_memory)
        )
        sharded_output = ttnn.linear(
            sharded_x,
            weight,
            program_config=program_config,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            compute_kernel_config=compute,
        )
        interleaved = ttnn.sharded_to_interleaved(sharded_output, ttnn.DRAM_MEMORY_CONFIG)
        result_shape = (*shape[:-1], n)
        return interleaved if tuple(interleaved.shape) == result_shape else ttnn.reshape(interleaved, result_shape)

    def _build_moe(self, host: HostLayerWeights) -> FusedDeepSeekSparseMoE | DeepSeekSparseMoE:
        """Build the active-expert graph with the selected measured policy."""

        p = self._prefix
        policy = self.optimization_policy
        dtype = {"bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}[policy.moe_weight_dtype]
        moe_class = OptimizedDeepSeekSparseMoE if policy.packed_moe_gate_up else OptimizedSeparateDeepSeekSparseMoE
        moe = moe_class(
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
            weights_dtype=dtype,
        )
        if policy.moe_grid is not None:
            gate_up_width = (
                2 * self.config.moe_intermediate_size
                if policy.packed_moe_gate_up
                else self.config.moe_intermediate_size
            )
            gate_up_subblock = policy.moe_out_subblock_w if policy.packed_moe_gate_up else 2
            moe.gate_up_program = _optimized_sparse_program(
                k=self.config.hidden_size,
                n=gate_up_width,
                grid=policy.moe_grid,
                in0_block_w=policy.moe_in0_block_w,
                out_subblock_w=gate_up_subblock,
            )
            moe.down_program = _optimized_sparse_program(
                k=self.config.moe_intermediate_size,
                n=self.config.hidden_size,
                grid=policy.moe_grid,
                in0_block_w=policy.moe_in0_block_w,
                out_subblock_w=policy.moe_out_subblock_w,
            )
        fidelity = {"hifi4": ttnn.MathFidelity.HiFi4, "lofi": ttnn.MathFidelity.LoFi}[policy.moe_fidelity]
        moe.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(),
            math_fidelity=fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=policy.moe_fidelity == "hifi4",
            packer_l1_acc=True,
        )
        moe.use_l1_decode = policy.moe_l1_decode
        if isinstance(moe, OptimizedDeepSeekSparseMoE):
            moe.use_l1_prefill = policy.moe_l1_prefill
            if policy.shared_dram_sharded:
                moe.configure_shared_dram_sharding(
                    input_cores=policy.dram_attention_input_cores or 8,
                    in0_block_w_cap=policy.dram_in0_block_w or 16,
                )
        return moe

    def _optimized_rms_norm(
        self,
        x: ttnn.Tensor,
        *,
        weight: ttnn.Tensor | None = None,
        epsilon: float,
        core_cap: int | None = None,
        return_sharded: bool = False,
        interleaved_memory: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config: object | None = None,
    ) -> ttnn.Tensor:
        """Use a bounded width-sharded L1 norm for small decode-like rows."""

        if not self.optimization_policy.sharded_decode_norms:
            return ttnn.rms_norm(x, weight=weight, epsilon=epsilon)
        shape = tuple(x.shape)
        width = shape[-1]
        rows = math.prod(shape[:-1])
        padded_rows = math.ceil(rows / 32) * 32
        width_tiles = width // 32
        if width % 32 or padded_rows > 64:
            kwargs = {}
            if compute_kernel_config is not None:
                kwargs["compute_kernel_config"] = compute_kernel_config
            return ttnn.rms_norm(x, weight=weight, epsilon=epsilon, **kwargs)
        max_norm_cores = min(64, core_cap or 64)
        cores = max(core_count for core_count in range(1, max_norm_cores + 1) if width_tiles % core_count == 0)
        grid_x = min(8, cores)
        while cores % grid_x:
            grid_x -= 1
        grid_y = cores // grid_x
        block_w = width_tiles // cores
        subblock_w = min(4, block_w)
        while block_w % subblock_w:
            subblock_w -= 1
        memory_config = ttnn.create_sharded_memory_config(
            shape=(padded_rows, width // cores),
            core_grid=ttnn.CoreGrid(x=grid_x, y=grid_y),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=[grid_x, grid_y],
            subblock_w=subblock_w,
            block_h=padded_rows // 32,
            block_w=block_w,
            inplace=False,
        )
        flat = x if shape[:-2] == (1, 1) else ttnn.reshape(x, [1, 1, rows, width])
        sharded = ttnn.to_memory_config(flat, memory_config)
        norm_kwargs = {}
        if compute_kernel_config is not None:
            norm_kwargs["compute_kernel_config"] = compute_kernel_config
        normed = ttnn.rms_norm(
            sharded,
            weight=weight,
            epsilon=epsilon,
            program_config=program_config,
            memory_config=memory_config,
            **norm_kwargs,
        )
        if return_sharded:
            return normed if tuple(normed.shape) == shape else ttnn.reshape(normed, shape)
        interleaved = ttnn.sharded_to_interleaved(normed, interleaved_memory)
        return interleaved if tuple(interleaved.shape) == shape else ttnn.reshape(interleaved, shape)

    def _project_q_kv(
        self,
        x: ttnn.Tensor,
        positions: ttnn.Tensor,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, tuple[ttnn.Tensor, ttnn.Tensor]]:
        """Keep fused packed projections and tune their decode-local norms."""

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
            if self.kind.attention.value == "compressed_sparse_attention":
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
        # CSA consumes additional full-grid Hadamard/indexer programs after
        # projection.  Retaining QR in L1 overlaps their static CB region on
        # the measured layer-2/4 geometry, so the carry is confined to the
        # compatible local/HCA phases where it is a measured no-regression.
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
        if self._packed_qr_projection is None:
            q_projected = self._optimized_linear(qr, self.q_b)
            self._fused_index_q_projection = (
                self._optimized_linear(qr, self.index_q) if self.kind.attention is AttentionKind.CSA else None
            )
        else:
            packed_qr = self._optimized_linear(qr, self._packed_qr_projection)
            q_projected = self._packed_qr_projection_slice(packed_qr, self.q_b)
            self._fused_index_q_projection = self._packed_qr_projection_slice(packed_qr, self.index_q)
        q = ttnn.reshape(q_projected, [1, x.shape[2], c.num_attention_heads, c.head_dim])
        q = ttnn.permute(q, (0, 2, 1, 3))
        q = self._optimized_rms_norm(q, epsilon=c.rms_norm_eps)
        q = self._rope(q, cos, sin)
        kv = self._optimized_rms_norm(kv_projected, weight=self.kv_norm, epsilon=c.rms_norm_eps)
        kv = self._rope(ttnn.reshape(kv, [1, 1, x.shape[2], c.head_dim]), cos, sin)
        return qr, q, kv, (cos, sin)

    def _gathered_attention(
        self,
        q: ttnn.Tensor,
        bank: ttnn.Tensor,
        indices: ttnn.Tensor,
        mask: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """Use explicit 2D programs for prefill's per-query gathered QK/PV."""

        q_len = q.shape[2]
        if q_len == 1 or not self.optimization_policy.explicit_prefill_attention_programs:
            return super()._gathered_attention(q, bank, indices, mask)
        padded_k = math.ceil(indices.shape[-1] / 128) * 128
        if padded_k != indices.shape[-1]:
            padding = padded_k - indices.shape[-1]
            indices = ttnn.pad(indices, [(0, 0), (0, padding)], 0)
            mask = ttnn.pad(mask, [(0, 0), (0, 0), (0, 0), (0, padding)], -65504.0)
        mask = ttnn.to_layout(mask, ttnn.TILE_LAYOUT)
        gathered = ttnn.embedding(
            ttnn.typecast(indices, ttnn.uint32),
            bank,
            layout=ttnn.TILE_LAYOUT,
        )
        gathered = ttnn.reshape(gathered, [q_len, 1, indices.shape[-1], self.config.head_dim])
        q_batched = ttnn.permute(q, (0, 2, 1, 3))
        gathered_t = ttnn.permute(gathered, (1, 0, 3, 2))
        qk_program = _optimized_prefill_program(
            m=self.config.num_attention_heads,
            k=self.config.head_dim,
            n=indices.shape[-1],
            max_grid=self.mesh_device.compute_with_storage_grid_size(),
        )
        logits = ttnn.matmul(
            q_batched,
            gathered_t,
            dtype=ttnn.bfloat16,
            program_config=qk_program,
            compute_kernel_config=self.compute_kernel_config,
        )
        mask = ttnn.permute(mask, (1, 0, 2, 3))
        logits = ttnn.add(ttnn.mul(logits, self.config.head_dim**-0.5), mask)
        sink = ttnn.permute(self.attention_sink, (0, 2, 1, 3))
        sink = ttnn.repeat(sink, [1, q_len, 1, 1])
        probabilities = ttnn.softmax(
            ttnn.concat([logits, sink], dim=-1),
            dim=-1,
            compute_kernel_config=self.compute_kernel_config,
            numeric_stable=True,
        )
        probabilities = ttnn.slice(
            probabilities,
            [0, 0, 0, 0],
            [1, q_len, self.config.num_attention_heads, indices.shape[-1]],
        )
        pv_program = _optimized_prefill_program(
            m=self.config.num_attention_heads,
            k=indices.shape[-1],
            n=self.config.head_dim,
            max_grid=self.mesh_device.compute_with_storage_grid_size(),
        )
        out = ttnn.matmul(
            probabilities,
            ttnn.permute(gathered, (1, 0, 2, 3)),
            dtype=ttnn.bfloat16,
            program_config=pv_program,
            compute_kernel_config=self.compute_kernel_config,
        )
        return ttnn.permute(out, (0, 2, 1, 3))

    def _grouped_output_projection(self, attention: ttnn.Tensor, seq_len: int) -> ttnn.Tensor:
        """Keep the legal grouped first projection and tune the large O-B row."""

        c = self.config
        grouped = ttnn.permute(attention, (0, 2, 1, 3))
        grouped = ttnn.reshape(grouped, [1, seq_len, c.o_groups, c.grouped_attention_width])
        grouped = ttnn.permute(grouped, (0, 2, 1, 3))
        if seq_len == 1 and self._grouped_output_program is not None:
            grouped = ttnn.to_memory_config(grouped, self._grouped_output_input_memory)
            projected = ttnn.matmul(
                grouped,
                self.o_a,
                program_config=self._grouped_output_program,
                memory_config=self._grouped_output_output_memory,
                dtype=ttnn.bfloat16,
                compute_kernel_config=self.dense_compute_kernel_config,
            )
            projected = ttnn.sharded_to_interleaved(projected, ttnn.DRAM_MEMORY_CONFIG)
        else:
            grouped_weight = (
                self._grouped_output_interleaved_weight
                if self._grouped_output_interleaved_weight is not None
                else self.o_a
            )
            prefill_program = (
                _optimized_prefill_program(
                    m=math.ceil(seq_len / 32) * 32,
                    k=grouped_weight.shape[-2],
                    n=grouped_weight.shape[-1],
                    max_grid=self.mesh_device.compute_with_storage_grid_size(),
                )
                if self.optimization_policy.explicit_prefill_attention_programs and seq_len > 1
                else None
            )
            projected = ttnn.matmul(
                grouped,
                grouped_weight,
                program_config=prefill_program,
                compute_kernel_config=(
                    self.dense_compute_kernel_config if id(grouped_weight) in self._optimized_dense_weight_ids else None
                ),
            )
        projected = ttnn.permute(projected, (0, 2, 1, 3))
        projected = ttnn.reshape(projected, [1, 1, seq_len, c.o_groups * c.o_lora_rank])
        return self._optimized_linear(projected, self.o_b)

    def prefill_forward(
        self,
        hidden_streams: ttnn.Tensor,
        plan: PrefillPlan | BatchedPrefillPlan,
    ) -> ttnn.Tensor:
        """Preserve fused prefill while applying the selected residual norms."""

        if self._batch_lanes:
            if isinstance(plan, BatchedPrefillPlan):
                self._set_active_batch_sinkhorn(plan.batch_size)
            return super().prefill_forward(hidden_streams, plan)
        if not isinstance(plan, PrefillPlan):
            raise TypeError("a single-lane decoder requires PrefillPlan")
        expected = (1, plan.seq_len, self.config.hc_mult, self.config.hidden_size)
        if tuple(hidden_streams.shape) != expected:
            raise ValueError(f"hidden_streams shape {tuple(hidden_streams.shape)} != {expected}")
        residual = hidden_streams
        post, comb, collapsed = self._hc_pre(residual, self.hc_attn_fn, self.hc_attn_base, self.hc_attn_scale)
        self._capture_prefill_boundary("attention_post", post)
        self._capture_prefill_boundary("attention_comb", ttnn.permute(comb, (0, 2, 1, 3)))
        self._capture_prefill_boundary("attention_collapsed", collapsed)
        attention_input = self._optimized_rms_norm(
            collapsed,
            weight=self.attn_norm,
            epsilon=self.config.rms_norm_eps,
        )
        self._capture_prefill_boundary("attention_input", attention_input)
        attention = self._attention_prefill(attention_input, plan)
        self._capture_prefill_boundary("attention_output", attention)
        streams = self._hc_post(attention, residual, post, comb)
        self._capture_prefill_boundary("attention_streams", ttnn.permute(streams, (0, 2, 1, 3)))
        residual = streams
        post, comb, collapsed = self._hc_pre(residual, self.hc_ffn_fn, self.hc_ffn_base, self.hc_ffn_scale)
        self._capture_prefill_boundary("ffn_post", post)
        self._capture_prefill_boundary("ffn_comb", ttnn.permute(comb, (0, 2, 1, 3)))
        self._capture_prefill_boundary("ffn_collapsed", collapsed)
        ffn_input = self._optimized_rms_norm(
            collapsed,
            weight=self.ffn_norm,
            epsilon=self.config.rms_norm_eps,
            interleaved_memory=(
                ttnn.L1_MEMORY_CONFIG if self.optimization_policy.ffn_norm_l1 else ttnn.DRAM_MEMORY_CONFIG
            ),
        )
        self._capture_prefill_boundary("ffn_input", ffn_input)
        moe = self._moe_forward(
            ffn_input,
            plan.input_ids,
        )
        self._capture_prefill_boundary("ffn_output", moe)
        streams = self._hc_post(moe, residual, post, comb)
        self._capture_prefill_boundary("layer_output", ttnn.permute(streams, (0, 2, 1, 3)))
        return streams

    def _capture_prefill_boundary(self, name: str, value: ttnn.Tensor) -> None:
        """Copy one reduced-probe last-token boundary without retaining device buffers."""

        target = getattr(self, "_decode_boundary_capture", None)
        if target is None:
            target = getattr(self, "_prefill_boundary_capture", None)
        if target is None:
            return
        if len(value.shape) != 4 or value.shape[0] != 1:
            raise ValueError(f"diagnostic boundary {name} has unsupported shape {tuple(value.shape)}")
        seq_len = int(value.shape[2])
        last_token = ttnn.slice(
            value,
            [0, 0, seq_len - 1, 0],
            [1, value.shape[1], seq_len, value.shape[3]],
        )
        local_tensors = ttnn.get_device_tensors(last_token) or (last_token,)
        host = ttnn.to_torch(local_tensors[0])
        target[name] = host[0, :, -1].detach().float().cpu().clone()
        # A full-range one-token slice may alias ``value`` instead of owning a
        # distinct buffer.  Deallocating that diagnostic view invalidates the
        # live decode tensor consumed by the next boundary.
        if seq_len > 1:
            ttnn.deallocate(last_token)

    def prefill_forward_block(
        self,
        hidden_streams: ttnn.Tensor,
        input_ids: ttnn.Tensor,
        position_ids: ttnn.Tensor,
        *,
        block_start: int | None = None,
    ) -> ttnn.Tensor:
        """Bound prefill activation residency while preserving exact cache updates.

        Hyper-connections and MoE retain the 128-token prefill shape. Attention
        advances each position through the already-qualified decode cache
        update/read path, so HCA and CSA compression state remains exact across
        block boundaries without retaining a sequence-wide KV bank.
        """

        if self._batch_lanes:
            raise TypeError("streaming prefill blocks require a fixed cache lane")
        seq_len = int(input_ids.shape[-1])
        expected = (1, seq_len, self.config.hc_mult, self.config.hidden_size)
        if tuple(hidden_streams.shape) != expected:
            raise ValueError(f"hidden_streams shape {tuple(hidden_streams.shape)} != {expected}")
        if tuple(position_ids.shape) == (1, seq_len):
            flat_position_ids = ttnn.reshape(position_ids, [seq_len])
        elif tuple(position_ids.shape) == (seq_len,):
            flat_position_ids = position_ids
        else:
            raise ValueError(f"position_ids shape {tuple(position_ids.shape)} must be {(1, seq_len)} or {(seq_len,)}")
        if block_start is not None and block_start < 0:
            raise ValueError(f"block_start must be non-negative, got {block_start}")

        residual = hidden_streams
        post, comb, collapsed = self._hc_pre(residual, self.hc_attn_fn, self.hc_attn_base, self.hc_attn_scale)
        attention_input = self._optimized_rms_norm(
            collapsed,
            weight=self.attn_norm,
            epsilon=self.config.rms_norm_eps,
        )
        attention_rows = []
        for offset in range(seq_len):
            row = ttnn.slice(
                attention_input,
                [0, 0, offset, 0],
                [1, 1, offset + 1, self.config.hidden_size],
            )
            position = ttnn.slice(flat_position_ids, [offset], [offset + 1])
            attention_rows.append(self._attention_decode(row, position))
        attention = attention_rows[0] if len(attention_rows) == 1 else ttnn.concat(attention_rows, dim=2)
        streams = self._hc_post(attention, residual, post, comb)
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

    def decode_forward(
        self,
        hidden_streams: ttnn.Tensor,
        input_ids: ttnn.Tensor,
        current_pos: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """Preserve fused paged decode while applying all decode-local norms."""

        if self._batch_lanes:
            self._set_active_batch_sinkhorn(hidden_streams.shape[0])
            return super().decode_forward(hidden_streams, input_ids, current_pos)
        expected = (1, 1, self.config.hc_mult, self.config.hidden_size)
        if tuple(hidden_streams.shape) != expected:
            raise ValueError(f"hidden_streams shape {tuple(hidden_streams.shape)} != {expected}")
        residual = hidden_streams
        post, comb, collapsed = self._hc_pre(residual, self.hc_attn_fn, self.hc_attn_base, self.hc_attn_scale)
        self._capture_prefill_boundary("attention_post", post)
        self._capture_prefill_boundary("attention_comb", ttnn.permute(comb, (0, 2, 1, 3)))
        self._capture_prefill_boundary("attention_collapsed", collapsed)
        attention_input = self._optimized_rms_norm(
            collapsed,
            weight=self.attn_norm,
            epsilon=self.config.rms_norm_eps,
        )
        self._capture_prefill_boundary("attention_input", attention_input)
        attention = self._attention_decode(attention_input, current_pos)
        self._capture_prefill_boundary("attention_output", attention)
        streams = self._hc_post(attention, residual, post, comb)
        self._capture_prefill_boundary("attention_streams", ttnn.permute(streams, (0, 2, 1, 3)))
        residual = streams
        post, comb, collapsed = self._hc_pre(residual, self.hc_ffn_fn, self.hc_ffn_base, self.hc_ffn_scale)
        self._capture_prefill_boundary("ffn_post", post)
        self._capture_prefill_boundary("ffn_comb", ttnn.permute(comb, (0, 2, 1, 3)))
        self._capture_prefill_boundary("ffn_collapsed", collapsed)
        ffn_input = self._optimized_rms_norm(
            collapsed,
            weight=self.ffn_norm,
            epsilon=self.config.rms_norm_eps,
            interleaved_memory=(
                ttnn.L1_MEMORY_CONFIG if self.optimization_policy.ffn_norm_l1 else ttnn.DRAM_MEMORY_CONFIG
            ),
        )
        self._capture_prefill_boundary("ffn_input", ffn_input)
        moe = self._moe_forward(
            ffn_input,
            input_ids,
        )
        self._capture_prefill_boundary("ffn_output", moe)
        streams = self._hc_post(moe, residual, post, comb)
        self._capture_prefill_boundary("layer_output", ttnn.permute(streams, (0, 2, 1, 3)))
        return streams

    def _sinkhorn_by_matmul(self, comb: ttnn.Tensor, seq: int) -> ttnn.Tensor:
        """Sinkhorn on [1, S, n, n] with row / column sums as matmuls against an n x n ones block.

        Each ttnn.sum on the tile-padded n x n matrix first rewrites the tile padding (a FillPad op per sum, ~20
        per call). Here the padding is zeroed once; comb @ J broadcasts row sums and J @ comb column sums with zero
        padding in and out, and 0 / (0 + eps) keeps it zero, so no further fills are needed.
        """

        c = self.config
        ones = self._sinkhorn_ones
        left = ones if seq == 1 else ttnn.repeat(ones, ttnn.Shape([1, seq, 1, 1]))
        config = self._sinkhorn_kernel_config
        comb = ttnn.fill_implicit_tile_padding(comb, 0.0)
        if getattr(self, "sinkhorn_fused_kernel", False) and seq == 1 and getattr(self, "sinkhorn_fused_eps", False):
            decode_iters = (
                getattr(self, "sinkhorn_decode_iters", None)
                or getattr(self, "sinkhorn_iters_override", None)
                or self._sinkhorn_iters
            )
            return self._sinkhorn_fused(comb, decode_iters)

        def normalize(value: ttnn.Tensor, rows: bool) -> ttnn.Tensor:
            sums = (
                ttnn.matmul(value, ones, dtype=value.dtype, compute_kernel_config=config)
                if rows
                else ttnn.matmul(left, value, dtype=value.dtype, compute_kernel_config=config)
            )
            if fused_eps:
                # eps rides on the divisor as an SFPU pre-activation: one op instead of add + div. Not
                # bit-identical to the separate add; sinkhorn_fused_eps=False restores it.
                return ttnn.div(value, sums, input_tensor_b_activations=[eps_add])
            return ttnn.div(value, ttnn.add(sums, c.hc_eps))

        fused_eps = getattr(self, "sinkhorn_fused_eps", False)
        eps_add = ttnn.UnaryWithParam(ttnn.UnaryOpType.ADD_UNARY_SFPU, float(c.hc_eps))
        comb = normalize(comb, rows=False)
        # experiment: sinkhorn_decode_iters overrides the policy's iteration count for decode only.
        iters = (
            getattr(self, "sinkhorn_decode_iters", None)
            or getattr(self, "sinkhorn_iters_override", None)
            or self._sinkhorn_iters
        )
        for _ in range(int(iters) - 1):
            comb = normalize(comb, rows=True)
            comb = normalize(comb, rows=False)
        return comb

    def _sinkhorn_fused(self, comb: ttnn.Tensor, iters: int) -> ttnn.Tensor:
        """The Sinkhorn of [1, S, n, n] combination tiles (one per token) as one program (kernels/sinkhorn/)
        instead of ~43 (decode) or ~60 (prefill) ops: matmul-with-ones sums and eps divisions per normalization,
        intermediates packed to the tile's data format as the separate ops do. Tiles are split over cores. Not
        bit-identical to the op-by-op versions (the division runs with the matmul's fp32 DST)."""

        if len(comb.shape) != 4 or comb.shape[0] != 1:
            raise ValueError(f"fused Sinkhorn takes [1, S, n, n] tiles, got {tuple(comb.shape)}")
        ones = self._sinkhorn_ones
        num_tiles = int(comb.shape[1])
        out = ttnn.allocate_tensor_on_device(
            comb.shape, comb.dtype, ttnn.TILE_LAYOUT, self.mesh_device, ttnn.DRAM_MEMORY_CONFIG
        )
        grid = self.mesh_device.compute_with_storage_grid_size()
        all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
        _, cores, group_1, group_2, per_1, per_2 = ttnn.split_work_to_cores(all_cores, num_tiles)
        page = {ttnn.bfloat16: 2048, ttnn.float32: 4096}[comb.dtype]
        cbs = [
            ttnn.CBDescriptor(
                total_size=pages * page,
                core_ranges=cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=index, data_format=comb.dtype, page_size=page)
                ],
            )
            for index, pages in ((0, 2), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (16, 2))
        ]
        reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        start = 0
        for group, count in ((group_1, per_1), (group_2, per_2)):
            for core_range in group.ranges():
                for y in range(core_range.start.y, core_range.end.y + 1):
                    for x in range(core_range.start.x, core_range.end.x + 1):
                        reader_rt[x][y] = [comb.buffer_address(), ones.buffer_address(), start, count]
                        writer_rt[x][y] = [out.buffer_address(), count, start]
                        compute_rt[x][y] = [count]
                        start += count
        kernels = Path(__file__).resolve().parent / "kernels" / "sinkhorn"
        program = ttnn.ProgramDescriptor(
            kernels=[
                ttnn.KernelDescriptor(
                    kernel_source=str(kernels / "reader_sinkhorn.cpp"),
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=cores,
                    compile_time_args=ttnn.TensorAccessorArgs(comb).get_compile_time_args()
                    + ttnn.TensorAccessorArgs(ones).get_compile_time_args(),
                    runtime_args=reader_rt,
                    config=ttnn.ReaderConfigDescriptor(),
                ),
                ttnn.KernelDescriptor(
                    kernel_source="ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/"
                    "writer_unary_interleaved_start_id.cpp",
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=cores,
                    compile_time_args=[16] + ttnn.TensorAccessorArgs(out).get_compile_time_args(),
                    runtime_args=writer_rt,
                    config=ttnn.WriterConfigDescriptor(),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=str(kernels / "sinkhorn_compute.cpp"),
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=cores,
                    compile_time_args=[
                        int(iters),
                        struct.unpack("<I", struct.pack("<f", float(self.config.hc_eps)))[0],
                    ],
                    runtime_args=compute_rt,
                    config=ttnn.ComputeConfigDescriptor(
                        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True
                    ),
                ),
            ],
            semaphores=[],
            cbs=cbs,
        )
        return ttnn.generic_op([comb, ones, out], program)

    def _moe_route(self, chunk: ttnn.Tensor, ids: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Decode / verify rows of a learned router take the fused epilogue (_moe_route_fused)."""

        rows = chunk.shape[2]
        if (
            getattr(self, "moe_route_fused", False)
            and self.kind.router is not RouterKind.HASH
            and getattr(self, "router_fp32_softplus", False)
            and (rows == 1 or getattr(self, "_verify_active", False))
            and rows <= 32
        ):
            logits = ttnn.linear(
                ttnn.typecast(chunk, ttnn.float32),
                self.gate_weight,
                dtype=ttnn.float32,
                compute_kernel_config=self.compute_kernel_config,
            )
            return self._moe_route_fused(logits, rows)
        return super()._moe_route(chunk, ids)

    def _moe_route_fused(self, logits: ttnn.Tensor, rows: int) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """The router after the gate matmul as one program (kernels/moe_route/) instead of ~25 ops
        (softplus, sqrt, typecasts, bias add, topk, gather, sum, div, mul, FillPads): scores = sqrt(softplus(logits))
        and selection = scores + bias in fp32, top-k on the data-movement core, weights = picked scores / sum * scale.
        Returns selected (uint16) and weights (bf16), both [1, 1, rows, top_k] tiles as _moe_route does. The op-by-op
        router rounds the scores to bf16 first, so this is not bit-identical."""

        c = self.config
        experts, top_k = int(logits.shape[-1]), c.num_experts_per_tok
        if experts % 32 or rows > 32:
            raise ValueError(f"fused router takes [1, 1, rows <= 32, 32n] logits, got {tuple(logits.shape)}")
        expert_tiles = experts // 32
        packed = math.ceil(rows * expert_tiles / 32)
        device = self.mesh_device
        dram = ttnn.DRAM_MEMORY_CONFIG
        shape = ttnn.Shape([1, 1, rows, top_k])
        selected = ttnn.allocate_tensor_on_device(shape, ttnn.uint16, ttnn.TILE_LAYOUT, device, dram)
        weights = ttnn.allocate_tensor_on_device(shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, device, dram)
        core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
        f32, half = 4096, 2048
        cbs = [
            ttnn.CBDescriptor(
                total_size=pages * size,
                core_ranges=core,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=size)],
            )
            for index, pages, dtype, size in (
                (0, expert_tiles, ttnn.float32, f32),
                (1, expert_tiles, ttnn.float32, f32),
                (2, packed, ttnn.float32, f32),
                (3, packed, ttnn.float32, f32),
                (4, 1, ttnn.float32, f32),
                (6, 1, ttnn.float32, f32),
                (7, packed, ttnn.float32, f32),
                (8, packed, ttnn.float32, f32),
                (16, 1, ttnn.bfloat16, half),
                (17, 1, ttnn.uint16, half),
            )
        ]
        unpack_modes = [ttnn.UnpackToDestMode.Default] * 64
        for index in (2, 3):  # only ever copied to DST: unpack straight into it at fp32
            unpack_modes[index] = ttnn.UnpackToDestMode.UnpackToDestFp32
        compute_config = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True
        )
        compute_config.unpack_to_dest_mode = unpack_modes
        reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        reader_rt[0][0] = [logits.buffer_address(), self.gate_bias.buffer_address(), rows]
        writer_rt[0][0] = [selected.buffer_address(), weights.buffer_address(), rows]
        compute_rt[0][0] = [packed]
        kernels = Path(__file__).resolve().parent / "kernels" / "moe_route"
        program = ttnn.ProgramDescriptor(
            kernels=[
                ttnn.KernelDescriptor(
                    kernel_source=str(kernels / "reader_moe_route.cpp"),
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=core,
                    compile_time_args=[expert_tiles, top_k]
                    + ttnn.TensorAccessorArgs(logits).get_compile_time_args()
                    + ttnn.TensorAccessorArgs(self.gate_bias).get_compile_time_args(),
                    runtime_args=reader_rt,
                    config=ttnn.ReaderConfigDescriptor(),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=str(kernels / "writer_moe_route.cpp"),
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=core,
                    compile_time_args=[expert_tiles, top_k]
                    + ttnn.TensorAccessorArgs(selected).get_compile_time_args()
                    + ttnn.TensorAccessorArgs(weights).get_compile_time_args(),
                    runtime_args=writer_rt,
                    config=ttnn.WriterConfigDescriptor(),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=str(kernels / "moe_route_compute.cpp"),
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=core,
                    compile_time_args=[struct.unpack("<I", struct.pack("<f", float(c.routed_scaling_factor)))[0]],
                    runtime_args=compute_rt,
                    config=compute_config,
                ),
            ],
            semaphores=[],
            cbs=cbs,
        )
        ttnn.generic_op([logits, self.gate_bias, selected, weights], program)
        return selected, weights

    def _hc_post_fused(
        self, output: ttnn.Tensor, residual: ttnn.Tensor, post: ttnn.Tensor, comb: ttnn.Tensor
    ) -> ttnn.Tensor:
        """The mHC output map as one program (kernels/hc_post/): per residual tile t of block row m,
        L_m @ R_t with L_m = [comb_m^T | post_m] and R_t the residual tile with the sublayer output's row in row n,
        i.e. comb^T @ residual + post * output with fp32 accumulation and one bf16 rounding (the op-by-op matmul +
        addcmul rounds twice, so not bit-identical). ``post`` is A2's [1, M, n, 1] column, ``comb`` [1, M, n, n]."""

        c = self.config
        rows, n, hidden = int(residual.shape[1]), c.hc_mult, c.hidden_size
        hidden_tiles = hidden // 32
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, rows, n, hidden]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.mesh_device, ttnn.DRAM_MEMORY_CONFIG
        )
        items = rows * hidden_tiles
        grid = self.mesh_device.compute_with_storage_grid_size()
        cores_n = min(64, items)
        per = math.ceil(items / cores_n)
        cores_n = math.ceil(items / per)
        coords = [(k % grid.x, k // grid.x) for k in range(cores_n)]
        core_set = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in coords])
        cbs = [
            ttnn.CBDescriptor(
                total_size=count * 2048,
                core_ranges=core_set,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=index, data_format=ttnn.bfloat16, page_size=2048)
                ],
            )
            for index, count in ((0, 2), (1, 2), (2, 1), (16, 2))
        ]
        reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for k, (x, y) in enumerate(coords):
            first, count = k * per, min(per, items - k * per)
            reader_rt[x][y] = [
                residual.buffer_address(),
                output.buffer_address(),
                comb.buffer_address(),
                post.buffer_address(),
                first,
                count,
            ]
            writer_rt[x][y] = [out.buffer_address(), count, first]
            compute_rt[x][y] = [first, count]
        kernels = Path(__file__).resolve().parent / "kernels" / "hc_post"
        program = ttnn.ProgramDescriptor(
            kernels=[
                ttnn.KernelDescriptor(
                    kernel_source=str(kernels / getattr(self, "hc_post_reader", "reader_hc_post.cpp")),
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=core_set,
                    compile_time_args=[n, hidden_tiles]
                    + ttnn.TensorAccessorArgs(residual).get_compile_time_args()
                    + ttnn.TensorAccessorArgs(output).get_compile_time_args()
                    + ttnn.TensorAccessorArgs(comb).get_compile_time_args()
                    + ttnn.TensorAccessorArgs(post).get_compile_time_args(),
                    runtime_args=reader_rt,
                    config=ttnn.ReaderConfigDescriptor(),
                ),
                ttnn.KernelDescriptor(
                    kernel_source="ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/"
                    "writer_unary_interleaved_start_id.cpp",
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=core_set,
                    compile_time_args=[16] + ttnn.TensorAccessorArgs(out).get_compile_time_args(),
                    runtime_args=writer_rt,
                    config=ttnn.WriterConfigDescriptor(),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=str(kernels / "hc_post_compute.cpp"),
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=core_set,
                    compile_time_args=[hidden_tiles],
                    runtime_args=compute_rt,
                    config=ttnn.ComputeConfigDescriptor(
                        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True
                    ),
                ),
            ],
            semaphores=[],
            cbs=cbs,
        )
        ttnn.generic_op([residual, output, comb, post, out], program)
        return out

    def _comp_pool_fused(
        self,
        raw: ttnn.Tensor,
        current_pos: ttnn.Tensor,
        kv_offset: int,
        score_offset: int,
        width: int,
        norm: ttnn.Tensor,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        rotate: bool,
    ) -> ttnn.Tensor:
        """One compressor history's decode pooling as two programs (kernels/comp_pool/), one core
        per output column tile each, instead of ~40 ops. Stage 1: the column's score / kv rows of ``raw`` (columns
        from kv_offset / score_offset; ratio 4: previous / current half), rows before the sequence start masked,
        column softmax over the rows, the weighted kv sum and its sum of squares. Stage 2: RMSNorm with ``norm`` over
        the whole row, RoPE on the trailing rope_head_dim columns and, for the indexer history, the Hadamard rotation.
        Returns [1, 1, 1, width] bf16 as the op-by-op chain does. Intermediates fp32, so not bit-identical."""

        c = self.config
        ratio = self.kind.compress_ratio
        rows, raw_width = int(raw.shape[1]), int(raw.shape[2])
        if kv_offset % 32 or score_offset % 32 or width % 32 or raw_width % 32:
            raise ValueError("fused compressor pooling needs tile-aligned history columns")
        wt, row_tiles, rope_tiles = width // 32, math.ceil(rows / 32), c.rope_head_dim // 32
        consts = self.__dict__.setdefault("_comp_pool_consts", {})
        if "trans" not in consts:
            consts["trans"] = ttnn.typecast(self.rope_trans, ttnn.float32)
        if rotate and "had" not in consts:
            consts["had"] = ttnn.typecast(self.index_hadamard, ttnn.float32)
        trans, had = consts["trans"], consts.get("had") if rotate else consts["trans"]
        device, dram = self.mesh_device, ttnn.DRAM_MEMORY_CONFIG
        stage1 = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, 32, 2 * width]), ttnn.float32, ttnn.TILE_LAYOUT, device, dram
        )
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, 1, width]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, dram
        )
        grid_x = device.compute_with_storage_grid_size().x
        coords = [(k % grid_x, k // grid_x) for k in range(wt)]
        cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in coords])
        f32, half = 4096, 2048
        bits = lambda value: struct.unpack("<I", struct.pack("<f", float(value)))[0]  # noqa: E731
        kernels = Path(__file__).resolve().parent / "kernels" / "comp_pool"
        hifi4 = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, dst_full_sync_en=True
        )

        def cbs(spec):
            return [
                ttnn.CBDescriptor(
                    total_size=count * size,
                    core_ranges=cores,
                    format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=size)],
                )
                for index, count, dtype, size in spec
            ]

        # stage 1: one column tile per core
        r1, w1, k1 = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for col, (x, y) in enumerate(coords):
            r1[x][y] = [raw.buffer_address(), current_pos.buffer_address(), col]
            w1[x][y] = [stage1.buffer_address(), col]
            k1[x][y] = []
        program1 = ttnn.ProgramDescriptor(
            kernels=[
                ttnn.KernelDescriptor(
                    kernel_source=str(kernels / "reader_pool_cols.cpp"),
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=cores,
                    compile_time_args=[
                        rows,
                        raw_width // 32,
                        kv_offset // 32,
                        score_offset // 32,
                        wt,
                        ratio,
                        int(ratio == 4),
                    ]
                    + ttnn.TensorAccessorArgs(raw).get_compile_time_args()
                    + ttnn.TensorAccessorArgs(current_pos).get_compile_time_args(),
                    runtime_args=r1,
                    config=ttnn.ReaderConfigDescriptor(),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=str(kernels / "writer_pool_cols.cpp"),
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=cores,
                    compile_time_args=[wt] + ttnn.TensorAccessorArgs(stage1).get_compile_time_args(),
                    runtime_args=w1,
                    config=ttnn.WriterConfigDescriptor(),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=str(kernels / "pool_cols_compute.cpp"),
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=cores,
                    compile_time_args=[row_tiles],
                    runtime_args=k1,
                    config=hifi4,
                ),
            ],
            semaphores=[],
            cbs=cbs(
                (
                    (0, row_tiles, ttnn.bfloat16, half),
                    (1, row_tiles, ttnn.bfloat16, half),
                    (2, 1, ttnn.bfloat16, half),
                    (3, 1, ttnn.float32, f32),
                    (4, 1, ttnn.float32, f32),
                    (5, 1, ttnn.float32, f32),
                    (9, 1, ttnn.bfloat16, half),
                    (10, 1, ttnn.uint32, 64),
                    (16, 2, ttnn.float32, f32),
                    (17, 1, ttnn.float32, f32),
                    (18, row_tiles, ttnn.float32, f32),
                    (19, row_tiles, ttnn.float32, f32),
                    (20, 1, ttnn.float32, f32),
                    (21, 1, ttnn.float32, f32),
                )
            ),
        )
        ttnn.generic_op([raw, current_pos, stage1], program1)

        # stage 2: one output column tile per core
        need = wt if rotate else 1
        r2, w2, k2 = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for col, (x, y) in enumerate(coords):
            r2[x][y] = [
                stage1.buffer_address(),
                norm.buffer_address(),
                cos.buffer_address(),
                sin.buffer_address(),
                trans.buffer_address(),
                had.buffer_address(),
                col,
            ]
            w2[x][y] = [out.buffer_address(), 1, col]
            k2[x][y] = [col]
        program2 = ttnn.ProgramDescriptor(
            kernels=[
                ttnn.KernelDescriptor(
                    kernel_source=str(kernels / "reader_pool_finish.cpp"),
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=cores,
                    compile_time_args=[wt, rope_tiles, int(rotate)]
                    + ttnn.TensorAccessorArgs(stage1).get_compile_time_args()
                    + ttnn.TensorAccessorArgs(norm).get_compile_time_args()
                    + ttnn.TensorAccessorArgs(cos).get_compile_time_args()
                    + ttnn.TensorAccessorArgs(sin).get_compile_time_args()
                    + ttnn.TensorAccessorArgs(trans).get_compile_time_args()
                    + ttnn.TensorAccessorArgs(had).get_compile_time_args(),
                    runtime_args=r2,
                    config=ttnn.ReaderConfigDescriptor(),
                ),
                ttnn.KernelDescriptor(
                    kernel_source="ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/"
                    "writer_unary_interleaved_start_id.cpp",
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=cores,
                    compile_time_args=[16] + ttnn.TensorAccessorArgs(out).get_compile_time_args(),
                    runtime_args=w2,
                    config=ttnn.WriterConfigDescriptor(),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=str(kernels / "pool_finish_compute.cpp"),
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=cores,
                    compile_time_args=[wt, rope_tiles, int(rotate), bits(1.0 / width), bits(c.rms_norm_eps)],
                    runtime_args=k2,
                    config=hifi4,
                ),
            ],
            semaphores=[],
            cbs=cbs(
                (
                    (0, wt, ttnn.float32, f32),
                    (1, need, ttnn.float32, f32),
                    (2, need, ttnn.float32, f32),
                    (3, need, ttnn.float32, f32),
                    (4, need, ttnn.float32, f32),
                    (5, 1, ttnn.float32, f32),
                    (6, wt if rotate else 1, ttnn.float32, f32),
                    (7, 1, ttnn.bfloat16, half),
                    (16, 1, ttnn.bfloat16, half),
                    (17, 1, ttnn.float32, f32),
                    (18, 1, ttnn.float32, f32),
                    (19, need, ttnn.float32, f32),
                )
            ),
        )
        ttnn.generic_op([stage1, norm, cos, sin, trans, had, out], program2)
        return out

    def _hc_map(self, streams: ttnn.Tensor, fn: ttnn.Tensor, rows: int, cores: int = 16) -> ttnn.Tensor:
        """mHC A3: x @ fn straight from the [1, rows, n, H] residual (kernels/hc_map/), split over ``cores``
        along the hidden width, with each block row's lane sums of squares (Gram diagonal) for the RMS norm, instead
        of flattening it to [1, 1, rows, n H], typecasting, RMS-normalizing and re-laying it out for the split-K
        matmul. Returns the partials [1, 1, cores * rows * (n + 1) * 32, 32] fp32 (core c, row m, tile d at page
        (c * rows + m) * (n + 1) + d) for _hc_pre_epilogue_fused(map_cores=cores)."""

        c = self.config
        n, hidden = c.hc_mult, c.hidden_size
        hidden_tiles = hidden // 32
        if hidden_tiles % cores:
            raise ValueError(f"{cores} map cores do not divide {hidden_tiles} hidden tiles")
        per_core = hidden_tiles // cores
        pages = rows * (n + 1)
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, cores * pages * 32, 32]),
            ttnn.float32,
            ttnn.TILE_LAYOUT,
            self.mesh_device,
            ttnn.DRAM_MEMORY_CONFIG,
        )
        grid_x = self.mesh_device.compute_with_storage_grid_size().x
        coords = [(k % grid_x, k // grid_x) for k in range(cores)]
        core_set = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in coords])
        cbs = [
            ttnn.CBDescriptor(
                total_size=count * size,
                core_ranges=core_set,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=size)],
            )
            for index, count, dtype, size in (
                (0, 2, streams.dtype, 2048),
                (1, n * per_core, ttnn.float32, 4096),
                (16, n + 1, ttnn.float32, 4096),
            )
        ]
        reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for k, (x, y) in enumerate(coords):
            reader_rt[x][y] = [streams.buffer_address(), fn.buffer_address(), k * per_core, per_core, rows]
            writer_rt[x][y] = [out.buffer_address(), pages, k * pages]
            compute_rt[x][y] = [per_core, rows]
        kernels = Path(__file__).resolve().parent / "kernels" / "hc_map"
        program = ttnn.ProgramDescriptor(
            kernels=[
                ttnn.KernelDescriptor(
                    kernel_source=str(kernels / "reader_hc_map.cpp"),
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=core_set,
                    compile_time_args=[n, hidden_tiles]
                    + ttnn.TensorAccessorArgs(streams).get_compile_time_args()
                    + ttnn.TensorAccessorArgs(fn).get_compile_time_args(),
                    runtime_args=reader_rt,
                    config=ttnn.ReaderConfigDescriptor(),
                ),
                ttnn.KernelDescriptor(
                    kernel_source="ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/"
                    "writer_unary_interleaved_start_id.cpp",
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=core_set,
                    compile_time_args=[16] + ttnn.TensorAccessorArgs(out).get_compile_time_args(),
                    runtime_args=writer_rt,
                    config=ttnn.WriterConfigDescriptor(),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=str(kernels / "hc_map_compute.cpp"),
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=core_set,
                    compile_time_args=[n],
                    runtime_args=compute_rt,
                    config=ttnn.ComputeConfigDescriptor(
                        math_fidelity=ttnn.MathFidelity.HiFi4,
                        math_approx_mode=False,
                        fp32_dest_acc_en=True,
                        dst_full_sync_en=True,  # n + 1 fp32 accumulators
                    ),
                ),
            ],
            semaphores=[],
            cbs=cbs,
        )
        ttnn.generic_op([streams, fn, out], program)
        return out

    def _hc_pre_epilogue_fused(
        self, partial: ttnn.Tensor, base: ttnn.Tensor, rows: int, map_cores: int = 0, iters: int | None = None
    ) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        """mHC A2: everything _hc_pre does between the split-K matmul and the collapse, as one program
        (kernels/hc_pre/): partial sums + bias, pre = sigmoid + eps, post = 2 sigmoid, comb = softmax + eps and the
        Sinkhorn, instead of ~20 ops (slices, reshapes, FillPads, typecasts, softmax, the Sinkhorn program).
        ``partial`` is the [1, split, rows, n(n + 2)] fp32 output of the split-K matmul. Returns, per block row m,
        pre [1, rows, 1, n] fp32 (zero K padding, ready for the collapse matmul), post [1, rows, n, 1] bf16 (the
        column _hc_post broadcasts) and comb [1, rows, n, n] bf16. Intermediates stay fp32, so the result is not
        bit-identical with the op-by-op path."""

        c = self.config
        n = c.hc_mult
        if map_cores:
            # mHC A3: ``partial`` is _hc_map's output, each map core's n + 1 tiles per block row
            split = map_cores * (n + 1)
        else:
            split = int(partial.shape[1])
            if rows > 32 or tuple(partial.shape)[2:] != (rows, n * (n + 2)):
                raise ValueError(
                    f"fused mHC epilogue takes [1, split, rows <= 32, {n * (n + 2)}], got {tuple(partial.shape)}"
                )
        device = self.mesh_device
        dram = ttnn.DRAM_MEMORY_CONFIG
        pre = ttnn.allocate_tensor_on_device(ttnn.Shape([1, rows, 1, n]), ttnn.float32, ttnn.TILE_LAYOUT, device, dram)
        post = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, rows, n, 1]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, dram
        )
        comb = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, rows, n, n]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, dram
        )
        # One block row per core (verify blocks); every core sums the partials itself. A prefill block with more rows
        # than the grid has cores runs as several programs of at most one row per core (a run of several rows
        # on one core gave the core's later rows its first row's result).
        grid = self.mesh_device.compute_with_storage_grid_size()
        grid_x = grid.x
        cores_per_program = grid.x * grid.y
        for chunk_start in range(0, rows, cores_per_program):
            spans = [(m, 1) for m in range(chunk_start, min(rows, chunk_start + cores_per_program))]
            coords = [(k % grid_x, k // grid_x) for k in range(len(spans))]
            core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in coords])
            f32, bf16 = 4096, 2048
            cbs = [
                ttnn.CBDescriptor(
                    total_size=pages * size,
                    core_ranges=core,
                    format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=size)],
                )
                for index, pages, dtype, size in (
                    (0, split, ttnn.float32, f32),
                    *((index, 1, ttnn.float32, f32) for index in range(1, 12)),
                    (12, n + 1, ttnn.float32, f32),
                    *((index, 1, ttnn.float32, f32) for index in range(13, 16)),
                    (16, 1, ttnn.float32, f32),
                    (17, 1, ttnn.bfloat16, bf16),
                    (18, 1, ttnn.bfloat16, bf16),
                )
            ]
            # Inputs only ever copied to DST unpack straight into it at fp32; the matmul operands (J, ping / pong)
            # go through the source registers.
            unpack_modes = [ttnn.UnpackToDestMode.Default] * 64
            for index in (0, 1, 3, 4, 5, 6, 7):
                unpack_modes[index] = ttnn.UnpackToDestMode.UnpackToDestFp32
            compute_config = ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True
            )
            compute_config.unpack_to_dest_mode = unpack_modes
            reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
            for (m, count), (x, y) in zip(spans, coords):
                reader_rt[x][y] = [partial.buffer_address(), base.buffer_address(), m, count, rows]
                writer_rt[x][y] = [pre.buffer_address(), post.buffer_address(), comb.buffer_address(), m, count]
                compute_rt[x][y] = [count]
            iters = (
                iters
                or getattr(self, "hc_pre_fused_sinkhorn_iters", None)
                or getattr(self, "sinkhorn_decode_iters", None)
                or getattr(self, "sinkhorn_iters_override", None)
                or self._sinkhorn_iters
            )
            kernels = Path(__file__).resolve().parent / "kernels" / "hc_pre"
            program = ttnn.ProgramDescriptor(
                kernels=[
                    ttnn.KernelDescriptor(
                        kernel_source=str(kernels / "reader_hc_pre.cpp"),
                        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                        core_ranges=core,
                        compile_time_args=[n, split, int(bool(map_cores)), map_cores]
                        + ttnn.TensorAccessorArgs(partial).get_compile_time_args()
                        + ttnn.TensorAccessorArgs(base).get_compile_time_args(),
                        runtime_args=reader_rt,
                        config=ttnn.ReaderConfigDescriptor(),
                    ),
                    ttnn.KernelDescriptor(
                        kernel_source=str(kernels / "writer_hc_pre.cpp"),
                        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                        core_ranges=core,
                        compile_time_args=ttnn.TensorAccessorArgs(pre).get_compile_time_args()
                        + ttnn.TensorAccessorArgs(post).get_compile_time_args()
                        + ttnn.TensorAccessorArgs(comb).get_compile_time_args(),
                        runtime_args=writer_rt,
                        config=ttnn.WriterConfigDescriptor(),
                    ),
                    ttnn.KernelDescriptor(
                        kernel_source=str(kernels / "hc_pre_compute.cpp"),
                        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                        core_ranges=core,
                        compile_time_args=[
                            int(iters),
                            struct.unpack("<I", struct.pack("<f", float(c.hc_eps)))[0],
                            split,
                            int(bool(map_cores)),
                            map_cores,
                            struct.unpack("<I", struct.pack("<f", 1.0 / c.hc_width))[0],
                            struct.unpack("<I", struct.pack("<f", float(c.rms_norm_eps)))[0],
                        ],
                        runtime_args=compute_rt,
                        config=compute_config,
                    ),
                ],
                semaphores=[],
                cbs=cbs,
            )
            ttnn.generic_op([partial, base, pre, post, comb], program)
        return pre, post, comb

    def _hc_linear_split_k(
        self, normed: ttnn.Tensor, fn: ttnn.Tensor, base: ttnn.Tensor, split_k: int, *, partials: bool = False
    ) -> ttnn.Tensor:
        """, decode: the mHC map [1, 16384] x [16384, 24] has a single output tile, so ttnn.linear runs it on
        one core (~130 us, twice per layer). Splitting K into ``split_k`` batches gives one output tile per batch
        (one core each); the fp32 partial sums are then added and the bias applied. Same fp32 accumulation, but
        in a different order, so the result is not bit-identical."""

        k = normed.shape[-1]
        rows = normed.shape[2]
        if k % (split_k * ttnn.TILE_SIZE):
            raise ValueError(f"hc_pre_split_k={split_k} must divide the mHC width {k} into whole tiles")
        k_tiles = k // split_k // ttnn.TILE_SIZE
        if rows == 1:
            lhs = ttnn.reshape(normed, [1, split_k, 1, k // split_k])
        else:
            # verify rows: [1, 1, M, K] -> [1, split, M, K / split]
            lhs = ttnn.permute(ttnn.reshape(normed, [1, rows, split_k, k // split_k]), (0, 2, 1, 3))
        partial = ttnn.matmul(
            lhs,
            ttnn.reshape(fn, [1, split_k, k // split_k, fn.shape[-1]]),
            dtype=ttnn.float32,
            compute_kernel_config=self.compute_kernel_config,
            # The automatic 1D config keeps the batches on one core (fuse_batch off); the batched reuse factory
            # gives each batch's single output tile its own core.
            program_config=ttnn.MatmulMultiCoreReuseProgramConfig(
                compute_with_storage_grid_size=self.mesh_device.compute_with_storage_grid_size(),
                in0_block_w=math.gcd(k_tiles, 8),
                out_subblock_h=1,
                out_subblock_w=1,
                per_core_M=1,
                per_core_N=1,
            ),
        )
        if partials:
            return partial
        raw = ttnn.sum(partial, dim=1, keepdim=True, compute_kernel_config=self.compute_kernel_config)
        return ttnn.add(raw, base)

    def _hc_pre(
        self,
        streams: ttnn.Tensor,
        fn: ttnn.Tensor,
        base: ttnn.Tensor,
        _scale: ttnn.Tensor | None,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        """Run the mHC input map with a measured Sinkhorn convergence policy."""

        c = self.config
        seq = streams.shape[1]
        if (
            getattr(self, "prefill_hc_fused", False)
            and seq > 32
            and not getattr(self, "_verify_active", False)
            and not streams.is_sharded()
            and not fn.is_sharded()
            and streams.dtype == ttnn.bfloat16
            and fn.dtype == ttnn.float32
        ):
            # a prefill block through the A3 map (straight from the [1, S, n, H] residual, no [1, 1, S, n H]
            # flatten / typecast / RMSNorm re-layout) and the A2 epilogue with the prefill Sinkhorn policy, then the
            # A1 collapse matmul. Not bit-identical with the op-by-op prefill path (fp32 intermediates).
            map_cores = 16
            iters = int(getattr(self, "prefill_hc_sinkhorn_iters", 0) or self._sinkhorn_iters)
            pre, post, comb = self._hc_pre_epilogue_fused(
                self._hc_map(streams, fn, seq, map_cores), base, seq, map_cores=map_cores, iters=iters
            )
            collapsed = ttnn.matmul(pre, streams, dtype=ttnn.bfloat16, compute_kernel_config=self.compute_kernel_config)
            return post, comb, ttnn.reshape(collapsed, [1, 1, seq, c.hidden_size])
        if (
            getattr(self, "hc_pre_map_fused", False)
            and getattr(self, "hc_pre_fused_epilogue", False)
            and getattr(self, "hc_collapse_matmul", False)
            and (seq == 1 or getattr(self, "_verify_active", False))
            and seq <= 32
            and not streams.is_sharded()
            and not fn.is_sharded()
            and streams.dtype == ttnn.bfloat16  # kernels/hc_map/ reads 2 KiB residual pages
            and fn.dtype == ttnn.float32
        ):
            # mHC A3: map straight from the residual + the fused epilogue (RMS scale inside), then A1's collapse
            map_cores = 16
            pre, post, comb = self._hc_pre_epilogue_fused(
                self._hc_map(streams, fn, seq, map_cores), base, seq, map_cores=map_cores
            )
            collapsed = ttnn.matmul(pre, streams, dtype=ttnn.bfloat16, compute_kernel_config=self.compute_kernel_config)
            return post, comb, ttnn.reshape(collapsed, [1, 1, seq, c.hidden_size])
        flat = ttnn.reshape(streams, [1, 1, seq, c.hc_width])
        normed = self._optimized_rms_norm(
            ttnn.typecast(flat, ttnn.float32),
            epsilon=c.rms_norm_eps,
            interleaved_memory=(
                ttnn.L1_MEMORY_CONFIG if self.optimization_policy.hc_norm_l1 else ttnn.DRAM_MEMORY_CONFIG
            ),
            # mHC normalizes a four-stream FP32 row.  Blackhole's default and
            # BF16-destination RMSNorm paths under-scale this boundary by about
            # 20%; use the decoder's exact FP32 accumulation only here.
            compute_kernel_config=self.compute_kernel_config,
        )
        split_k = int(getattr(self, "hc_pre_split_k", 0) or 0)
        # the decode-only fast paths also serve a speculative verify block (a few rows, decode numerics).
        rows_fast = seq == 1 or bool(getattr(self, "_verify_active", False))
        if (
            split_k > 1
            and rows_fast
            and getattr(self, "hc_pre_fused_epilogue", False)
            and getattr(self, "hc_collapse_matmul", False)
            and seq <= 32
        ):
            # mHC A2: split-K partials -> pre / post / comb in one program, then the A1 collapse matmul.
            pre, post, comb = self._hc_pre_epilogue_fused(
                self._hc_linear_split_k(normed, fn, base, split_k, partials=True), base, seq
            )
            collapsed = ttnn.matmul(pre, streams, dtype=ttnn.bfloat16, compute_kernel_config=self.compute_kernel_config)
            return post, comb, ttnn.reshape(collapsed, [1, 1, seq, c.hidden_size])
        if split_k > 1 and rows_fast:
            raw = self._hc_linear_split_k(normed, fn, base, split_k)
        else:
            raw = ttnn.linear(
                normed,
                fn,
                bias=base,
                dtype=ttnn.float32,
                compute_kernel_config=self.compute_kernel_config,
            )
        pre = ttnn.add(
            ttnn.slice(raw, [0, 0, 0, 0], [1, 1, seq, c.hc_mult]),
            c.hc_eps,
            input_tensor_a_activations=[ttnn.UnaryOpType.SIGMOID],
        )
        post = ttnn.mul(
            ttnn.slice(raw, [0, 0, 0, c.hc_mult], [1, 1, seq, 2 * c.hc_mult]),
            2.0,
            input_tensor_a_activations=[ttnn.UnaryOpType.SIGMOID],
        )
        comb = ttnn.add(
            ttnn.softmax(
                ttnn.reshape(
                    ttnn.slice(raw, [0, 0, 0, 2 * c.hc_mult], [1, 1, seq, c.hc_mix_width]),
                    [1, seq, c.hc_mult, c.hc_mult],
                ),
                dim=-1,
            ),
            c.hc_eps,
        )
        if self.optimization_policy.sinkhorn_dtype == "bf16":
            comb = ttnn.typecast(comb, ttnn.bfloat16)
        if getattr(self, "sinkhorn_matmul", False) and seq == 1:
            # Decode only: for prefill rows the per-token matmuls cost more than the padded sums they replace.
            comb = self._sinkhorn_by_matmul(comb, seq)
        elif getattr(self, "sinkhorn_fused_prefill", False) and seq > 1:
            # for prefill: one program for every token's tile instead of ~60 ops (prefill is dispatch-bound).
            comb = self._sinkhorn_fused(
                ttnn.fill_implicit_tile_padding(comb, 0.0),
                int(getattr(self, "sinkhorn_iters_override", None) or self._sinkhorn_iters),
            )
        else:
            comb = ttnn.div(comb, ttnn.add(ttnn.sum(comb, dim=-2, keepdim=True), c.hc_eps))
            # / experiment: sinkhorn_iters_override replaces the policy's iteration count everywhere.
            for _ in range(int(getattr(self, "sinkhorn_iters_override", None) or self._sinkhorn_iters) - 1):
                comb = ttnn.div(comb, ttnn.add(ttnn.sum(comb, dim=-1, keepdim=True), c.hc_eps))
                comb = ttnn.div(comb, ttnn.add(ttnn.sum(comb, dim=-2, keepdim=True), c.hc_eps))
        if getattr(self, "hc_collapse_matmul", False) and rows_fast:
            # Decode: the lane collapse sum_l pre[l] * streams[l] as one [1, n] x [n, H] matmul (fp32 accumulation)
            # instead of ~25 permute / slice / typecast / mul / add ops. The K padding of pre is zeroed as _hc_post's
            # comb is; not bit-identical (the matmul unpacks pre with fewer mantissa bits than the fp32 eltwise mul).
            collapsed = ttnn.matmul(
                ttnn.fill_implicit_tile_padding(ttnn.reshape(pre, [1, seq, 1, c.hc_mult]), 0.0),
                streams,
                dtype=ttnn.bfloat16,
                compute_kernel_config=self.compute_kernel_config,
            )
            return (
                ttnn.typecast(post, ttnn.bfloat16),
                ttnn.typecast(comb, ttnn.bfloat16),
                ttnn.reshape(collapsed, [1, 1, seq, c.hidden_size]),
            )
        if getattr(self, "hc_collapse_strategy", "reduce") == "streamwise_fp32":
            collapsed = streamwise_hc_collapse(
                streams,
                pre,
                seq=seq,
                hc_mult=c.hc_mult,
                hidden_size=c.hidden_size,
            )
        else:
            streams_fp32 = ttnn.typecast(streams, ttnn.float32)
            collapsed = ttnn.sum(
                ttnn.mul(streams_fp32, ttnn.reshape(pre, [1, seq, c.hc_mult, 1])),
                dim=2,
            )
        collapsed = ttnn.typecast(collapsed, ttnn.bfloat16)
        return (
            ttnn.typecast(post, ttnn.bfloat16),
            ttnn.typecast(comb, ttnn.bfloat16),
            ttnn.reshape(collapsed, [1, 1, seq, c.hidden_size]),
        )


__all__ = [
    "BatchedPrefillPlan",
    "DecodeTrace",
    "DecoderCache",
    "FusedDeepSeekSparseMoE",
    "HostLayerWeights",
    "OptimizationPolicy",
    "OptimizedDecoder",
    "PagedTensor",
    "streamwise_hc_collapse",
    "PrefillChunk",
    "PrefillPlan",
    "optimization_policy",
]
