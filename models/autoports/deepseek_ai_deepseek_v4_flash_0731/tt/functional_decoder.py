"""Single-chip TTNN functional decoder for DeepSeek-V4-Flash-0731.

The public runtime contract is intentionally small:

* :meth:`FunctionalDecoder.from_state_dict` converts source-format weights once.
* :meth:`FunctionalDecoder.prepare_prefill` uploads all host-built shape artifacts.
* :meth:`FunctionalDecoder.prefill_forward` consumes only TTNN tensors.
* :meth:`FunctionalDecoder.decode_forward` consumes only TTNN tensors and is trace-safe.

Hidden streams stay `[B, S, 4, 4096]`.  Attention maintains a physical paged cache,
including permuted page tables, while the sink-aware attention kernel consumes rows
gathered from those physical pages.  This is necessary for V4: sparse_sdpa has no
attention-sink input, and omitting the sink changes the softmax denominator.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule

from ..checkpoint import LayerCheckpoint
from ..config import AttentionKind, DeepSeekV4FlashConfig, RouterKind, make_page_table, physical_row_lut
from .sparse_moe import DeepSeekSparseMoE

ATTENTION_MASK_VALUE = -65504.0
HCA_CACHE_GATHER_CHUNK_ROWS = 1024


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _yarn_cos_sin(
    config: DeepSeekV4FlashConfig,
    compressed: bool,
    max_context: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Official inference/model.py YaRN frequencies in interleaved pair layout."""

    dim = config.rope_head_dim
    base = config.compress_rope_theta if compressed else config.rope_theta
    original = config.original_max_position_embeddings if compressed else 0
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original:

        def correction(rotations: float) -> float:
            return dim * math.log(original / (rotations * 2 * math.pi)) / (2 * math.log(base))

        low = max(math.floor(correction(config.beta_fast)), 0)
        high = min(math.ceil(correction(config.beta_slow)), dim - 1)
        if low == high:
            high += 0.001
        ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low)).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / config.rope_factor * (1 - smooth) + freqs * smooth
    phases = torch.outer(torch.arange(max_context, dtype=torch.float32), freqs)
    cos = torch.cos(phases).repeat_interleave(2, dim=-1).to(torch.bfloat16)
    sin = torch.sin(phases).repeat_interleave(2, dim=-1).to(torch.bfloat16)
    return cos, sin


def _rotation_matrix() -> torch.Tensor:
    matrix = torch.zeros(1, 1, 32, 32, dtype=torch.bfloat16)
    matrix[..., torch.arange(0, 32, 2), torch.arange(1, 32, 2)] = 1
    matrix[..., torch.arange(1, 32, 2), torch.arange(0, 32, 2)] = -1
    return matrix


def _hadamard(size: int) -> torch.Tensor:
    if size <= 0 or size & (size - 1):
        raise ValueError(f"Hadamard width must be a positive power of two, got {size}")
    matrix = torch.tensor([[1.0]], dtype=torch.float32)
    while matrix.shape[0] < size:
        matrix = torch.cat((torch.cat((matrix, matrix), 1), torch.cat((matrix, -matrix), 1)), 0)
    return (matrix * size**-0.5).to(torch.bfloat16)


@dataclass
class HostLayerWeights:
    tensors: dict[str, torch.Tensor]
    expert_w1: torch.Tensor | None
    expert_w2: torch.Tensor | None
    expert_w3: torch.Tensor | None
    # Packed routed experts from the cache; the expert_w* tensors are then None.
    expert_store: Any = None


@dataclass
class PagedTensor:
    tensor: ttnn.Tensor
    page_table: ttnn.Tensor
    row_lut: ttnn.Tensor
    host_page_table: tuple[int, ...]
    rows: int
    width: int


@dataclass
class DecoderCache:
    max_context: int
    local: PagedTensor
    compressed: PagedTensor | None
    compressor_kv: PagedTensor | None
    compressor_score: PagedTensor | None
    indexer: PagedTensor | None
    indexer_compressor_kv: PagedTensor | None
    indexer_compressor_score: PagedTensor | None


@dataclass(frozen=True)
class PrefillChunk:
    start: int
    end: int
    local_indices: ttnn.Tensor
    local_mask: ttnn.Tensor
    compressed_indices: ttnn.Tensor | None
    compressed_mask: ttnn.Tensor | None
    compressed_counts: ttnn.Tensor | None


@dataclass(frozen=True)
class PrefillPlan:
    seq_len: int
    position_ids: ttnn.Tensor
    input_ids: ttnn.Tensor
    chunks: tuple[PrefillChunk, ...]
    local_fill_page_table: ttnn.Tensor
    local_fill_start: int
    local_fill_length: int
    compressed_fill_page_table: ttnn.Tensor | None
    compressor_fill_page_table: ttnn.Tensor | None
    compressor_fill_start: int
    compressor_fill_length: int


@dataclass(frozen=True)
class BatchedPrefillPlan:
    """Setup-time collection of independent per-user paged-cache plans."""

    seq_len: int
    batch_size: int
    lanes: tuple[PrefillPlan, ...]


@dataclass
class DecodeTrace:
    trace_id: int
    output: ttnn.Tensor

    def execute(self, mesh_device: ttnn.MeshDevice, *, blocking: bool = False) -> ttnn.Tensor:
        ttnn.execute_trace(mesh_device, self.trace_id, cq_id=0, blocking=blocking)
        return self.output

    def release(self, mesh_device: ttnn.MeshDevice) -> None:
        ttnn.release_trace(mesh_device, self.trace_id)


class FunctionalDecoder(LightweightModule):
    """One complete V4-Flash backbone decoder layer on a 1x1 mesh."""

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
    ):
        super().__init__()
        self.mesh_device = mesh_device
        self.config = DeepSeekV4FlashConfig() if config is None else config
        self.config.validate()
        self.layer_idx = layer_idx
        self.kind = self.config.layer_kind(layer_idx)
        if max_batch_size <= 0:
            raise ValueError(f"max_batch_size must be positive, got {max_batch_size}")
        self.max_batch_size = max_batch_size
        if tuple(mesh_device.shape) != (1, 1):
            raise ValueError(f"functional decoder requires a 1x1 mesh, got {tuple(mesh_device.shape)}")
        self.max_context = self.config.max_position_embeddings if max_context is None else max_context
        self.geometry = self.config.cache_geometry(layer_idx, self.max_context)
        self.weights_dtype = weights_dtype
        self._mapper = ttnn.ReplicateTensorToMesh(mesh_device)
        self._prefix = f"layers.{layer_idx}"
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        self._upload_dense_weights(host_weights.tensors)
        self._upload_rope()
        self.cache = self._allocate_cache(local_page_permutation, compressed_page_permutation)
        self.moe = self._build_moe(host_weights)
        self._decode_constants()
        self._batch_lanes: tuple[FunctionalDecoder, ...] = ()
        if max_batch_size > 1:
            # Each user needs independent mutable cache storage, but decoder
            # weights are immutable and can be shared.  Shallow lane views keep
            # one uploaded copy of every weight while allocating private paged
            # cache/constants for lanes 1..N-1.  Runtime dispatch is still all
            # TTNN and the lane loop is captured as one decode trace.
            lanes = []
            for lane_idx in range(max_batch_size):
                lane = copy.copy(self)
                lane.max_batch_size = 1
                lane._batch_lanes = ()
                if lane_idx:
                    lane.cache = lane._allocate_cache(
                        local_page_permutation,
                        compressed_page_permutation,
                    )
                    lane._decode_constants()
                lanes.append(lane)
            self._batch_lanes = tuple(lanes)

    @classmethod
    def from_state_dict(
        cls,
        state_dict: Mapping[str, torch.Tensor],
        *,
        hf_config: object | None = None,
        layer_idx: int,
        mesh_device: ttnn.MeshDevice,
        max_context: int | None = None,
        **kwargs,
    ) -> "FunctionalDecoder":
        """Construct from dequantized official source-format tensors.

        Expert tensors may be supplied either as 256 individual source keys or as
        stacked convenience keys ``expert_w{1,2,3}`` with shape `[256,O,K]`.
        """

        config = DeepSeekV4FlashConfig() if hf_config is None else DeepSeekV4FlashConfig.from_hf_config(hf_config)
        prefix = f"layers.{layer_idx}.ffn.experts"

        def stack(projection: str) -> torch.Tensor:
            convenience = f"expert_{projection}"
            if convenience in state_dict:
                result = state_dict[convenience]
            else:
                result = torch.stack(
                    [state_dict[f"{prefix}.{expert}.{projection}.weight"] for expert in range(config.num_experts)]
                )
            if result.shape[0] != config.num_experts:
                raise ValueError(f"{projection} has {result.shape[0]} experts, expected {config.num_experts}")
            return result

        experts = {projection: stack(projection) for projection in ("w1", "w2", "w3")}
        non_expert = {
            key: tensor
            for key, tensor in state_dict.items()
            if not key.startswith(prefix + ".") and not key.startswith("expert_w")
        }
        host = HostLayerWeights(non_expert, experts["w1"], experts["w2"], experts["w3"])
        return cls(mesh_device, layer_idx, host, config=config, max_context=max_context, **kwargs)

    @classmethod
    def from_checkpoint(
        cls,
        mesh_device: ttnn.MeshDevice,
        checkpoint_root: str | Path,
        layer_idx: int,
        *,
        config: DeepSeekV4FlashConfig | None = None,
        max_context: int | None = None,
        **kwargs,
    ) -> "FunctionalDecoder":
        """Load and dequantize one real official layer, including every expert."""

        config = DeepSeekV4FlashConfig() if config is None else config
        checkpoint = LayerCheckpoint(checkpoint_root, layer_idx, config)
        tensors, w1, w2, w3 = checkpoint.materialize()
        return cls(
            mesh_device,
            layer_idx,
            HostLayerWeights(tensors, w1, w2, w3),
            config=config,
            max_context=max_context,
            **kwargs,
        )

    def _to_tt(
        self,
        tensor: torch.Tensor,
        *,
        dtype: ttnn.DataType | None = None,
        layout: ttnn.Layout = ttnn.TILE_LAYOUT,
        memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
    ) -> ttnn.Tensor:
        return ttnn.from_torch(
            tensor.contiguous(),
            device=self.mesh_device,
            mesh_mapper=self._mapper,
            dtype=self.weights_dtype if dtype is None else dtype,
            layout=layout,
            memory_config=memory_config,
        )

    def _weight(self, tensor: torch.Tensor, *, dtype: ttnn.DataType | None = None) -> ttnn.Tensor:
        if tensor.ndim != 2:
            raise ValueError(f"linear weight must be rank 2, got {tensor.shape}")
        return self._to_tt(tensor.transpose(0, 1), dtype=dtype)

    def _upload_dense_weights(self, state: Mapping[str, torch.Tensor]) -> None:
        p = self._prefix

        def get(suffix: str) -> torch.Tensor:
            key = f"{p}.{suffix}"
            try:
                return state[key]
            except KeyError as exc:
                raise KeyError(f"missing required layer tensor {key}") from exc

        self.hc_attn_fn = self._weight(get("hc_attn_fn"), dtype=ttnn.float32)
        self.hc_attn_base = self._to_tt(
            get("hc_attn_base").reshape(1, 1, 1, -1),
            dtype=ttnn.float32,
        )
        self.hc_attn_scale = self._to_tt(
            get("hc_attn_scale").reshape(1, 1, 1, -1),
            dtype=ttnn.float32,
        )
        self.hc_ffn_fn = self._weight(get("hc_ffn_fn"), dtype=ttnn.float32)
        self.hc_ffn_base = self._to_tt(
            get("hc_ffn_base").reshape(1, 1, 1, -1),
            dtype=ttnn.float32,
        )
        self.hc_ffn_scale = self._to_tt(
            get("hc_ffn_scale").reshape(1, 1, 1, -1),
            dtype=ttnn.float32,
        )
        self.attn_norm = self._to_tt(get("attn_norm.weight").reshape(1, 1, 1, -1))
        self.ffn_norm = self._to_tt(get("ffn_norm.weight").reshape(1, 1, 1, -1))
        self.q_a = self._weight(get("attn.wq_a.weight"))
        self.q_a_norm = self._to_tt(get("attn.q_norm.weight").reshape(1, 1, 1, -1))
        self.q_b = self._weight(get("attn.wq_b.weight"))
        self.kv = self._weight(get("attn.wkv.weight"))
        self.kv_norm = self._to_tt(get("attn.kv_norm.weight").reshape(1, 1, 1, -1))
        wo_a = get("attn.wo_a.weight").reshape(
            self.config.o_groups, self.config.o_lora_rank, self.config.grouped_attention_width
        )
        self.o_a = self._to_tt(wo_a.transpose(-2, -1).unsqueeze(0))
        self.o_b = self._weight(get("attn.wo_b.weight"))
        self.attention_sink = self._to_tt(get("attn.attn_sink").reshape(1, -1, 1, 1))
        self.gate_weight = self._weight(get("ffn.gate.weight"), dtype=ttnn.float32)
        if self.kind.router is RouterKind.HASH:
            self.gate_bias = None
            self.tid2eid = self._to_tt(
                get("ffn.gate.tid2eid").to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
            )
        else:
            self.gate_bias = self._to_tt(
                get("ffn.gate.bias").reshape(1, 1, 1, -1),
                dtype=ttnn.float32,
            )
            self.tid2eid = None

        if self.kind.compress_ratio:
            self.comp_kv = self._weight(get("attn.compressor.wkv.weight"))
            self.comp_gate = self._weight(get("attn.compressor.wgate.weight"))
            self.comp_ape = self._to_tt(get("attn.compressor.ape"))
            self.comp_norm = self._to_tt(get("attn.compressor.norm.weight").reshape(1, 1, 1, -1))
        else:
            self.comp_kv = self.comp_gate = self.comp_ape = self.comp_norm = None

        if self.kind.attention is AttentionKind.CSA:
            self.index_q = self._weight(get("attn.indexer.wq_b.weight"))
            self.index_weight = self._weight(get("attn.indexer.weights_proj.weight"))
            self.index_comp_kv = self._weight(get("attn.indexer.compressor.wkv.weight"))
            self.index_comp_gate = self._weight(get("attn.indexer.compressor.wgate.weight"))
            self.index_comp_ape = self._to_tt(get("attn.indexer.compressor.ape"))
            self.index_comp_norm = self._to_tt(get("attn.indexer.compressor.norm.weight").reshape(1, 1, 1, -1))
            self.index_hadamard = self._to_tt(_hadamard(self.config.index_head_dim))
        else:
            self.index_q = self.index_weight = None
            self.index_comp_kv = self.index_comp_gate = self.index_comp_ape = self.index_comp_norm = None
            self.index_hadamard = None

    def _upload_rope(self) -> None:
        cos, sin = _yarn_cos_sin(
            self.config,
            compressed=bool(self.kind.compress_ratio),
            max_context=self.max_context,
        )
        self.rope_cos = self._to_tt(cos, layout=ttnn.ROW_MAJOR_LAYOUT)
        self.rope_sin = self._to_tt(sin, layout=ttnn.ROW_MAJOR_LAYOUT)
        self.rope_trans = self._to_tt(_rotation_matrix())

    def _page_tensor(
        self,
        physical_pages: int,
        logical_pages: int,
        width: int,
        *,
        cyclic: bool,
        permutation: Sequence[int] | None,
    ) -> PagedTensor:
        table = make_page_table(logical_pages, physical_pages, permutation, cyclic=cyclic)
        tensor = ttnn.zeros(
            [physical_pages, 1, self.config.page_size, width],
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        page_table = self._to_tt(
            torch.tensor(table, dtype=torch.int32).reshape(1, -1),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        lut = self._to_tt(
            torch.tensor(physical_row_lut(table, self.config.page_size), dtype=torch.int32).reshape(-1, 1),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        return PagedTensor(tensor, page_table, lut, table, physical_pages * self.config.page_size, width)

    def _allocate_cache(
        self,
        local_permutation: Sequence[int] | None,
        compressed_permutation: Sequence[int] | None,
    ) -> DecoderCache:
        c = self.config
        logical_pages = math.ceil(self.max_context / c.page_size)
        local = self._page_tensor(
            logical_pages,
            logical_pages,
            c.head_dim,
            cyclic=False,
            permutation=local_permutation,
        )
        if not self.kind.compress_ratio:
            return DecoderCache(self.max_context, local, None, None, None, None, None, None)

        compressed_logical = math.ceil(math.ceil(self.max_context / self.kind.compress_ratio) / c.page_size)
        compressed = self._page_tensor(
            compressed_logical,
            compressed_logical,
            c.head_dim,
            cyclic=False,
            permutation=compressed_permutation,
        )
        # Keep the projected compressor history fully paged.  It costs more DRAM
        # than a custom sub-page ring, but preserves non-aligned prompt tails
        # exactly with stock paged_fill_cache/paged_update_cache.
        raw_pages = logical_pages
        compressor_kv = self._page_tensor(
            raw_pages, logical_pages, self.comp_kv.shape[-1], cyclic=False, permutation=None
        )
        compressor_score = self._page_tensor(
            raw_pages, logical_pages, self.comp_gate.shape[-1], cyclic=False, permutation=None
        )
        if self.kind.attention is not AttentionKind.CSA:
            return DecoderCache(
                self.max_context,
                local,
                compressed,
                compressor_kv,
                compressor_score,
                None,
                None,
                None,
            )

        # One extra tile is an all-visible guard band for indexer_score_dsa's
        # tile-aligned query window.  It never participates in top-k.
        index_pages = compressed_logical + 1
        indexer = self._page_tensor(index_pages, index_pages, c.index_head_dim, cyclic=False, permutation=None)
        index_raw_pages = logical_pages
        index_kv = self._page_tensor(
            index_raw_pages, logical_pages, self.index_comp_kv.shape[-1], cyclic=False, permutation=None
        )
        index_score = self._page_tensor(
            index_raw_pages, logical_pages, self.index_comp_gate.shape[-1], cyclic=False, permutation=None
        )
        return DecoderCache(
            self.max_context,
            local,
            compressed,
            compressor_kv,
            compressor_score,
            indexer,
            index_kv,
            index_score,
        )

    def _build_moe(self, host: HostLayerWeights) -> DeepSeekSparseMoE:
        p = self._prefix
        return DeepSeekSparseMoE(
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
        )

    def _decode_constants(self) -> None:
        c = self.config
        local_offsets = torch.arange(1 - c.sliding_window, 1, dtype=torch.int32).reshape(1, -1)
        self.decode_local_offsets = self._to_tt(local_offsets, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        self.decode_local_ordinals = self._to_tt(
            torch.arange(c.sliding_window, dtype=torch.int32).reshape(1, -1),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        if self.kind.compress_ratio:
            max_entries = math.ceil(self.max_context / self.kind.compress_ratio)
            self.decode_compressed_ordinals = self._to_tt(
                torch.arange(max_entries, dtype=torch.int32).reshape(1, -1),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            raw = 2 * self.kind.compress_ratio if self.kind.attention is AttentionKind.CSA else self.kind.compress_ratio
            self.decode_comp_offsets = self._to_tt(
                torch.arange(1 - raw, 1, dtype=torch.int32).reshape(1, -1),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            if self.kind.attention is AttentionKind.CSA:
                score_rows = _round_up(
                    max(self.config.index_topk, self.cache.indexer.rows),
                    32,
                )
                guard_rows = score_rows - self.cache.indexer.rows + 32
                self.decode_index_guard = ttnn.zeros(
                    [1, 1, guard_rows, self.config.index_head_dim],
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.mesh_device,
                )
                self.decode_index_ordinals = ttnn.reshape(
                    ttnn.arange(
                        0,
                        self.cache.indexer.rows + guard_rows,
                        1,
                        device=self.mesh_device,
                        dtype=ttnn.int32,
                    ),
                    [1, self.cache.indexer.rows + guard_rows],
                )
                extra_count = self.config.index_topk
            else:
                self.decode_index_guard = None
                self.decode_index_ordinals = None
                extra_count = math.ceil(self.max_context / self.kind.compress_ratio)
            self.decode_attention_extra_indices = ttnn.reshape(
                ttnn.arange(
                    self.config.sliding_window,
                    self.config.sliding_window + extra_count,
                    1,
                    device=self.mesh_device,
                    dtype=ttnn.int32,
                ),
                [1, extra_count],
            )
        else:
            self.decode_compressed_ordinals = self.decode_comp_offsets = None
            self.decode_index_guard = None
            self.decode_index_ordinals = None
            self.decode_attention_extra_indices = None

    def prepare_prefill(
        self,
        seq_len: int,
        input_ids: torch.Tensor,
        *,
        query_chunk_size: int = 32,
    ) -> PrefillPlan | BatchedPrefillPlan:
        """Upload all host-created prefill indices before entering the runtime pass."""

        if self._batch_lanes:
            if input_ids.ndim == 1:
                input_ids = input_ids.reshape(1, -1)
            if input_ids.ndim != 2 or input_ids.shape[1] != seq_len:
                raise ValueError(f"batched input_ids shape {tuple(input_ids.shape)} must be [batch, {seq_len}]")
            batch_size = input_ids.shape[0]
            if not 1 <= batch_size <= self.max_batch_size:
                raise ValueError(f"batch size {batch_size} outside configured capacity {self.max_batch_size}")
            return BatchedPrefillPlan(
                seq_len,
                batch_size,
                tuple(
                    self._batch_lanes[lane].prepare_prefill(
                        seq_len,
                        input_ids[lane],
                        query_chunk_size=query_chunk_size,
                    )
                    for lane in range(batch_size)
                ),
            )

        if not 1 <= seq_len <= self.max_context:
            raise ValueError(f"seq_len {seq_len} outside cache capacity {self.max_context}")
        if query_chunk_size <= 0 or query_chunk_size % 32:
            raise ValueError("query_chunk_size must be a positive multiple of 32")
        if input_ids.numel() != seq_len:
            raise ValueError(f"input_ids has {input_ids.numel()} tokens, expected {seq_len}")
        positions = torch.arange(seq_len, dtype=torch.int32).reshape(1, -1)
        chunks = []
        for start in range(0, seq_len, query_chunk_size):
            end = min(seq_len, start + query_chunk_size)
            qpos = torch.arange(start, end, dtype=torch.int32).reshape(-1, 1)
            offsets = torch.arange(1 - self.config.sliding_window, 1, dtype=torch.int32).reshape(1, -1)
            indices = qpos + offsets
            valid = indices >= 0
            indices = indices.clamp(min=0)
            mask = torch.zeros_like(indices, dtype=torch.float32).masked_fill(~valid, ATTENTION_MASK_VALUE)
            compressed_indices = compressed_mask = compressed_counts = None
            if self.kind.attention is AttentionKind.HCA:
                counts = torch.div(qpos.squeeze(1) + 1, self.kind.compress_ratio, rounding_mode="floor")
                max_count = int(counts.max()) if counts.numel() else 0
                if max_count:
                    comp = torch.arange(max_count, dtype=torch.int32).expand(end - start, -1)
                    comp_valid = comp < counts.reshape(-1, 1)
                    compressed_indices = self._to_tt(comp + seq_len, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
                    compressed_mask = self._to_tt(
                        torch.zeros_like(comp, dtype=torch.float32)
                        .masked_fill(~comp_valid, ATTENTION_MASK_VALUE)
                        .reshape(end - start, 1, 1, -1),
                        dtype=ttnn.bfloat16,
                    )
            elif self.kind.attention is AttentionKind.CSA:
                counts = torch.div(qpos.squeeze(1) + 1, self.kind.compress_ratio, rounding_mode="floor")
                compressed_counts = self._to_tt(
                    counts.reshape(-1, 1).to(torch.int32),
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )
            chunks.append(
                PrefillChunk(
                    start,
                    end,
                    self._to_tt(indices, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT),
                    self._to_tt(mask.reshape(end - start, 1, 1, -1), dtype=ttnn.bfloat16),
                    compressed_indices,
                    compressed_mask,
                    compressed_counts,
                )
            )

        last_page = math.ceil(seq_len / self.config.page_size)
        first_page = max(0, (seq_len - self.config.sliding_window) // self.config.page_size)
        local_pages = self.cache.local.host_page_table[first_page:last_page]
        local_pt = self._to_tt(
            torch.tensor(local_pages, dtype=torch.int32).reshape(1, -1),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        comp_pt = None
        if self.cache.compressed is not None:
            entries = seq_len // self.kind.compress_ratio
            pages = math.ceil(entries / self.config.page_size)
            comp_pt = self._to_tt(
                torch.tensor(self.cache.compressed.host_page_table[:pages], dtype=torch.int32).reshape(1, -1),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
        compressor_pt = None
        compressor_start = compressor_length = 0
        if self.cache.compressor_kv is not None:
            raw_rows = (
                2 * self.kind.compress_ratio if self.kind.attention is AttentionKind.CSA else self.kind.compress_ratio
            )
            compressor_first_page = max(0, (seq_len - raw_rows) // self.config.page_size)
            raw_table = self.cache.compressor_kv.host_page_table[compressor_first_page:last_page]
            compressor_pt = self._to_tt(
                torch.tensor(raw_table, dtype=torch.int32).reshape(1, -1),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            compressor_start = compressor_first_page * self.config.page_size
            compressor_length = (last_page - compressor_first_page) * self.config.page_size
        return PrefillPlan(
            seq_len,
            self._to_tt(positions, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT),
            self._to_tt(input_ids.reshape(1, -1).to(torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT),
            tuple(chunks),
            local_pt,
            first_page * self.config.page_size,
            (last_page - first_page) * self.config.page_size,
            comp_pt,
            compressor_pt,
            compressor_start,
            compressor_length,
        )

    def _rope_values(self, positions: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        indices = ttnn.typecast(positions, ttnn.uint32)
        cos = ttnn.embedding(indices, self.rope_cos, layout=ttnn.TILE_LAYOUT)
        sin = ttnn.embedding(indices, self.rope_sin, layout=ttnn.TILE_LAYOUT)
        count = positions.shape[-1]
        return (
            ttnn.reshape(cos, [1, 1, count, self.config.rope_head_dim]),
            ttnn.reshape(sin, [1, 1, count, self.config.rope_head_dim]),
        )

    def _rope(self, x: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor, *, inverse: bool = False) -> ttnn.Tensor:
        c = self.config
        nope_width = x.shape[-1] - c.rope_head_dim
        nope = ttnn.slice(x, [0, 0, 0, 0], [x.shape[0], x.shape[1], x.shape[2], nope_width])
        rope = ttnn.slice(x, [0, 0, 0, nope_width], list(x.shape))
        # rotary_embedding_llama requires height-sharded inputs. Keep this
        # functional path interleaved by applying its exact row-vector rotation
        # matrix independently to each 32-wide tile of the 64-wide RoPE slice.
        rotated_parts = []
        for start in range(0, c.rope_head_dim, 32):
            end = start + 32
            value = ttnn.slice(rope, [0, 0, 0, start], [rope.shape[0], rope.shape[1], rope.shape[2], end])
            cos_part = ttnn.slice(cos, [0, 0, 0, start], [cos.shape[0], cos.shape[1], cos.shape[2], end])
            sin_part = ttnn.slice(sin, [0, 0, 0, start], [sin.shape[0], sin.shape[1], sin.shape[2], end])
            if inverse:
                sin_part = ttnn.neg(sin_part)
            rotated = ttnn.matmul(value, self.rope_trans)
            rotated_parts.append(ttnn.add(ttnn.mul(value, cos_part), ttnn.mul(rotated, sin_part)))
        rope = ttnn.concat(rotated_parts, dim=-1)
        return ttnn.concat([nope, rope], dim=-1)

    def _hc_pre(
        self,
        streams: ttnn.Tensor,
        fn: ttnn.Tensor,
        base: ttnn.Tensor,
        scale: ttnn.Tensor,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        c = self.config
        seq = streams.shape[1]
        flat = ttnn.reshape(streams, [1, 1, seq, c.hc_width])
        normed = ttnn.rms_norm(ttnn.typecast(flat, ttnn.float32), epsilon=c.rms_norm_eps)
        pre_scale = ttnn.slice(scale, [0, 0, 0, 0], [1, 1, 1, 1])
        post_scale = ttnn.slice(scale, [0, 0, 0, 1], [1, 1, 1, 2])
        comb_scale = ttnn.slice(scale, [0, 0, 0, 2], [1, 1, 1, 3])
        # Base was added above only after scaling would be correct; undo and apply
        # each learned scale before its matching base slice.
        pre_b = ttnn.slice(base, [0, 0, 0, 0], [1, 1, 1, c.hc_mult])
        post_b = ttnn.slice(base, [0, 0, 0, c.hc_mult], [1, 1, 1, 2 * c.hc_mult])
        comb_b = ttnn.slice(base, [0, 0, 0, 2 * c.hc_mult], [1, 1, 1, c.hc_mix_width])
        raw = ttnn.linear(
            normed,
            fn,
            dtype=ttnn.float32,
            compute_kernel_config=self.compute_kernel_config,
        )
        pre = ttnn.add(
            ttnn.sigmoid(ttnn.add(ttnn.mul(ttnn.slice(raw, [0, 0, 0, 0], [1, 1, seq, c.hc_mult]), pre_scale), pre_b)),
            c.hc_eps,
        )
        post = ttnn.mul(
            ttnn.sigmoid(
                ttnn.add(
                    ttnn.mul(
                        ttnn.slice(raw, [0, 0, 0, c.hc_mult], [1, 1, seq, 2 * c.hc_mult]),
                        post_scale,
                    ),
                    post_b,
                )
            ),
            2.0,
        )
        comb = ttnn.add(
            ttnn.softmax(
                ttnn.reshape(
                    ttnn.add(
                        ttnn.mul(
                            ttnn.slice(raw, [0, 0, 0, 2 * c.hc_mult], [1, 1, seq, c.hc_mix_width]),
                            comb_scale,
                        ),
                        comb_b,
                    ),
                    [1, seq, c.hc_mult, c.hc_mult],
                ),
                dim=-1,
            ),
            c.hc_eps,
        )
        comb = ttnn.div(comb, ttnn.add(ttnn.sum(comb, dim=-2, keepdim=True), c.hc_eps))
        for _ in range(c.hc_sinkhorn_iters - 1):
            comb = ttnn.div(comb, ttnn.add(ttnn.sum(comb, dim=-1, keepdim=True), c.hc_eps))
            comb = ttnn.div(comb, ttnn.add(ttnn.sum(comb, dim=-2, keepdim=True), c.hc_eps))
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

    def _hc_post(
        self,
        output: ttnn.Tensor,
        residual: ttnn.Tensor,
        post: ttnn.Tensor,
        comb: ttnn.Tensor,
    ) -> ttnn.Tensor:
        seq = residual.shape[1]
        placed = ttnn.mul(
            ttnn.reshape(output, [1, seq, 1, self.config.hidden_size]),
            ttnn.reshape(post, [1, seq, self.config.hc_mult, 1]),
        )
        mixed = ttnn.matmul(
            ttnn.permute(comb, (0, 1, 3, 2)),
            residual,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute_kernel_config,
        )
        return ttnn.add(placed, mixed)

    def _project_q_kv(
        self, x: ttnn.Tensor, positions: ttnn.Tensor
    ) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, tuple[ttnn.Tensor, ttnn.Tensor]]:
        c = self.config
        cos, sin = self._rope_values(positions)
        qr = ttnn.rms_norm(ttnn.linear(x, self.q_a), weight=self.q_a_norm, epsilon=c.rms_norm_eps)
        q = ttnn.reshape(ttnn.linear(qr, self.q_b), [1, x.shape[2], c.num_attention_heads, c.head_dim])
        q = ttnn.permute(q, (0, 2, 1, 3))
        q = ttnn.rms_norm(q, epsilon=c.rms_norm_eps)
        q = self._rope(q, cos, sin)
        kv = ttnn.rms_norm(ttnn.linear(x, self.kv), weight=self.kv_norm, epsilon=c.rms_norm_eps)
        kv = self._rope(ttnn.reshape(kv, [1, 1, x.shape[2], c.head_dim]), cos, sin)
        return qr, q, kv, (cos, sin)

    def _compress_prefill(
        self,
        x: ttnn.Tensor,
        positions: ttnn.Tensor,
        kv_weight: ttnn.Tensor,
        gate_weight: ttnn.Tensor,
        ape: ttnn.Tensor,
        norm: ttnn.Tensor,
        width: int,
        *,
        rotate_hadamard: bool,
    ) -> ttnn.Tensor:
        ratio = self.kind.compress_ratio
        cutoff = (x.shape[2] // ratio) * ratio
        if cutoff == 0:
            raise ValueError(f"prefill length {x.shape[2]} produces no complete ratio-{ratio} entry")
        source = ttnn.slice(x, [0, 0, 0, 0], [1, 1, cutoff, self.config.hidden_size])
        groups = cutoff // ratio
        kv = ttnn.reshape(ttnn.linear(source, kv_weight), [1, groups, ratio, -1])
        score = ttnn.add(ttnn.reshape(ttnn.linear(source, gate_weight), [1, groups, ratio, -1]), ape)
        if ratio == 4:
            # [previous-window first half | current-window second half].
            prev_kv = ttnn.slice(kv, [0, 0, 0, 0], [1, groups, ratio, width])
            cur_kv = ttnn.slice(kv, [0, 0, 0, width], [1, groups, ratio, 2 * width])
            prev_score = ttnn.slice(score, [0, 0, 0, 0], [1, groups, ratio, width])
            cur_score = ttnn.slice(score, [0, 0, 0, width], [1, groups, ratio, 2 * width])
            zero = ttnn.zeros_like(ttnn.slice(prev_kv, [0, 0, 0, 0], [1, 1, ratio, width]))
            neginf = ttnn.full_like(zero, float("-inf"))
            prev_kv = (
                ttnn.concat([zero, ttnn.slice(prev_kv, [0, 0, 0, 0], [1, groups - 1, ratio, width])], dim=1)
                if groups > 1
                else zero
            )
            prev_score = (
                ttnn.concat([neginf, ttnn.slice(prev_score, [0, 0, 0, 0], [1, groups - 1, ratio, width])], dim=1)
                if groups > 1
                else neginf
            )
            kv = ttnn.concat([prev_kv, cur_kv], dim=2)
            score = ttnn.concat([prev_score, cur_score], dim=2)
        pooled = ttnn.sum(ttnn.mul(kv, ttnn.softmax(score, dim=2)), dim=2, keepdim=True)
        pooled = ttnn.rms_norm(pooled, weight=norm, epsilon=self.config.rms_norm_eps)
        compressed_positions = ttnn.slice(positions, [0, 0], [1, cutoff], [1, ratio])
        cos, sin = self._rope_values(compressed_positions)
        pooled = self._rope(ttnn.reshape(pooled, [1, 1, groups, width]), cos, sin)
        if rotate_hadamard:
            pooled = ttnn.matmul(pooled, self.index_hadamard)
        return pooled

    def _fill_prefill_cache(
        self,
        kv: ttnn.Tensor,
        compressed: ttnn.Tensor | None,
        indexer: ttnn.Tensor | None,
        plan: PrefillPlan,
    ) -> None:
        local = ttnn.slice(
            kv,
            [0, 0, plan.local_fill_start, 0],
            [1, 1, min(plan.seq_len, plan.local_fill_start + plan.local_fill_length), self.config.head_dim],
        )
        missing = plan.local_fill_length - local.shape[2]
        if missing:
            local = ttnn.pad(local, [(0, 0), (0, 0), (0, missing), (0, 0)], 0.0)
        ttnn.experimental.paged_fill_cache(self.cache.local.tensor, local, plan.local_fill_page_table, batch_idx=0)
        if compressed is not None:
            padded = _round_up(compressed.shape[2], self.config.page_size)
            fill = ttnn.pad(compressed, [(0, 0), (0, 0), (0, padded - compressed.shape[2]), (0, 0)], 0.0)
            ttnn.experimental.paged_fill_cache(
                self.cache.compressed.tensor, fill, plan.compressed_fill_page_table, batch_idx=0
            )
        if indexer is not None:
            padded = _round_up(indexer.shape[2], self.config.page_size)
            fill = ttnn.pad(indexer, [(0, 0), (0, 0), (0, padded - indexer.shape[2]), (0, 0)], 0.0)
            index_pt = ttnn.slice(self.cache.indexer.page_table, [0, 0], [1, padded // self.config.page_size])
            ttnn.experimental.paged_fill_cache(self.cache.indexer.tensor, fill, index_pt, batch_idx=0)

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
        """Persist the exact projected tail needed by the first decode step."""

        slot = ttnn.remainder(positions, self.kind.compress_ratio)
        bias = ttnn.embedding(ttnn.typecast(slot, ttnn.uint32), ape, layout=ttnn.TILE_LAYOUT)
        bias = ttnn.reshape(bias, [1, 1, plan.seq_len, kv_weight.shape[-1]])
        raw_kv = ttnn.linear(x, kv_weight)
        raw_score = ttnn.add(ttnn.linear(x, gate_weight), bias)

        def fill(cache: PagedTensor, value: ttnn.Tensor) -> None:
            tail = ttnn.slice(
                value,
                [0, 0, plan.compressor_fill_start, 0],
                [1, 1, plan.seq_len, value.shape[-1]],
            )
            missing = plan.compressor_fill_length - tail.shape[2]
            if missing:
                tail = ttnn.pad(tail, [(0, 0), (0, 0), (0, missing), (0, 0)], 0.0)
            ttnn.experimental.paged_fill_cache(
                cache.tensor,
                tail,
                plan.compressor_fill_page_table,
                batch_idx=0,
            )

        fill(kv_cache, raw_kv)
        fill(score_cache, raw_score)

    def _indexer_topk(
        self,
        q: ttnn.Tensor,
        weights: ttnn.Tensor,
        index_bank: ttnn.Tensor,
        valid_entries: ttnn.Tensor,
        ordinals: ttnn.Tensor | None = None,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        # index_bank includes a 32-row dummy guard band.
        # upstream indexer_score (#55614) takes the gate weights in projection order [B, 1, Sq, Hi]; a caller still
        # holding the old head-major [B, Hi, Sq, 1] layout is converted here.
        if weights.shape[1] != 1:
            weights = ttnn.permute(weights, (0, 3, 2, 1))
        cfg = ttnn.IndexerScoreProgramConfig(q_chunk_size=32, k_chunk_size=64, head_group_size=0)
        q_len = q.shape[2]
        if q_len < 32:
            q = ttnn.pad(q, [(0, 0), (0, 0), (0, 32 - q_len), (0, 0)], 0.0)
            weights = ttnn.pad(weights, [(0, 0), (0, 0), (0, 32 - q_len), (0, 0)], 0.0)
            valid_entries = ttnn.pad(valid_entries, [(0, 32 - q_len), (0, 0)], 0)
        logits = ttnn.experimental.indexer_score_dsa(
            q,
            index_bank,
            weights,
            chunk_start_idx=index_bank.shape[2] - 32,
            program_config=cfg,
        )
        if ordinals is None:
            ordinals = ttnn.arange(0, index_bank.shape[2], 1, device=self.mesh_device, dtype=ttnn.int32)
            ordinals = ttnn.reshape(ordinals, [1, index_bank.shape[2]])
        # Compare explicit rank-4 row/column axes.  A rank-2 singleton compare
        # left finite guard-band logits on BH, while materializing 32 copies of
        # a full-context ordinal vector exceeds L1.  Rank-4 broadcasting keeps
        # the axes unambiguous without allocating the 32 x context expansion.
        ordinals = ttnn.reshape(ordinals, [1, 1, 1, index_bank.shape[2]])
        limits = ttnn.reshape(valid_entries, [1, 1, 32, 1])
        valid = ttnn.lt(ordinals, limits)
        logits = ttnn.where(valid, logits, float("-inf"))
        indices = ttnn.experimental.topk_large_indices(
            logits, k=self.config.index_topk, valid_length=index_bank.shape[2]
        )
        selected_limits = ttnn.repeat(
            ttnn.reshape(valid_entries, [1, 1, 32, 1]),
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

    def _gathered_attention(
        self,
        q: ttnn.Tensor,
        bank: ttnn.Tensor,
        indices: ttnn.Tensor,
        mask: ttnn.Tensor,
    ) -> ttnn.Tensor:
        q_len = q.shape[2]
        padded_k = _round_up(indices.shape[-1], 128)
        if padded_k != indices.shape[-1]:
            padding = padded_k - indices.shape[-1]
            indices = ttnn.pad(indices, [(0, 0), (0, padding)], 0)
            mask = ttnn.pad(mask, [(0, 0), (0, 0), (0, 0), (0, padding)], ATTENTION_MASK_VALUE)
        mask = ttnn.to_layout(mask, ttnn.TILE_LAYOUT)
        gathered = ttnn.embedding(ttnn.typecast(indices, ttnn.uint32), bank, layout=ttnn.TILE_LAYOUT)
        gathered = ttnn.reshape(gathered, [q_len, 1, indices.shape[-1], self.config.head_dim])
        # Batch by query position and put all heads on the matmul M axis.  This
        # shares that query's gathered one-head KV matrix without materializing
        # 64 copies of it.
        q_batched = ttnn.permute(q, (0, 2, 1, 3))
        gathered_t = ttnn.permute(gathered, (1, 0, 3, 2))
        logits = ttnn.matmul(
            q_batched,
            gathered_t,
            dtype=ttnn.bfloat16,
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
        out = ttnn.matmul(
            probabilities,
            ttnn.permute(gathered, (1, 0, 2, 3)),
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute_kernel_config,
        )
        return ttnn.permute(out, (0, 2, 1, 3))

    def _grouped_output_projection(self, attention: ttnn.Tensor, seq_len: int) -> ttnn.Tensor:
        """Apply V4's block-diagonal low-rank output projection.

        ``attention`` is the inverse-RoPE result in ``[1, heads, S, D]``
        order.  Keeping this boundary explicit also makes the attention core
        independently testable from the two output projections.
        """

        c = self.config
        grouped = ttnn.permute(attention, (0, 2, 1, 3))
        grouped = ttnn.reshape(grouped, [1, seq_len, c.o_groups, c.grouped_attention_width])
        grouped = ttnn.permute(grouped, (0, 2, 1, 3))
        projected = ttnn.matmul(grouped, self.o_a)
        projected = ttnn.permute(projected, (0, 2, 1, 3))
        projected = ttnn.reshape(projected, [1, 1, seq_len, c.o_groups * c.o_lora_rank])
        return ttnn.linear(projected, self.o_b)

    def _attention_prefill(
        self,
        x: ttnn.Tensor,
        plan: PrefillPlan,
    ) -> ttnn.Tensor:
        c = self.config
        qr, q, kv, rope = self._project_q_kv(x, plan.position_ids)
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
        for chunk in plan.chunks:
            q_chunk = ttnn.slice(q, [0, 0, chunk.start, 0], [1, c.num_attention_heads, chunk.end, c.head_dim])
            indices = chunk.local_indices
            mask = chunk.local_mask
            if self.kind.attention is AttentionKind.HCA:
                if chunk.compressed_indices is not None:
                    indices = ttnn.concat([indices, chunk.compressed_indices], dim=-1)
                    mask = ttnn.concat([mask, chunk.compressed_mask], dim=-1)
            elif self.kind.attention is AttentionKind.CSA and index_bank is not None:
                idx_q = ttnn.matmul(
                    ttnn.slice(qr, [0, 0, chunk.start, 0], [1, 1, chunk.end, c.q_lora_rank]),
                    self.index_q,
                )
                idx_q = ttnn.reshape(idx_q, [1, chunk.end - chunk.start, c.index_n_heads, c.index_head_dim])
                idx_q = ttnn.permute(idx_q, (0, 2, 1, 3))
                cos = ttnn.slice(rope[0], [0, 0, chunk.start, 0], [1, 1, chunk.end, c.rope_head_dim])
                sin = ttnn.slice(rope[1], [0, 0, chunk.start, 0], [1, 1, chunk.end, c.rope_head_dim])
                idx_q = self._rope(idx_q, cos, sin)
                idx_q = ttnn.matmul(idx_q, self.index_hadamard)
                idx_w = ttnn.linear(
                    ttnn.slice(x, [0, 0, chunk.start, 0], [1, 1, chunk.end, c.hidden_size]),
                    self.index_weight,
                )
                idx_w = ttnn.mul(idx_w, c.index_head_dim**-0.5 * c.index_n_heads**-0.5)
                idx_w = ttnn.permute(idx_w, (0, 3, 2, 1))
                topk, topk_mask = self._indexer_topk(idx_q, idx_w, index_bank, chunk.compressed_counts)
                topk = ttnn.reshape(topk, [chunk.end - chunk.start, c.index_topk])
                flat_topk_mask = ttnn.reshape(topk_mask, [chunk.end - chunk.start, c.index_topk])
                # topk_large_indices returns an unsigned tensor.  Mixed-width
                # unsigned multiplication did not reliably clear masked guard
                # indices on Blackhole, allowing out-of-bank reads before the
                # attention mask was applied.  Sanitize in signed int32.
                topk = ttnn.typecast(topk, ttnn.int32)
                topk_valid = ttnn.typecast(ttnn.eq(flat_topk_mask, 0.0), ttnn.int32)
                topk = ttnn.mul(topk, topk_valid)
                topk = ttnn.add(topk, plan.seq_len)
                indices = ttnn.concat([indices, topk], dim=-1)
                mask = ttnn.concat([mask, topk_mask], dim=-1)
            outputs.append(self._gathered_attention(q_chunk, bank, indices, mask))
        attention = outputs[0] if len(outputs) == 1 else ttnn.concat(outputs, dim=2)
        attention = self._rope(attention, *rope, inverse=True)
        return self._grouped_output_projection(attention, plan.seq_len)

    def _moe_route(self, chunk: ttnn.Tensor, ids: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Router for one MoE chunk: top-k expert ids [1,1,S,k] (uint16) and normalized weights."""

        c = self.config
        rows = chunk.shape[2]
        logits = ttnn.linear(
            ttnn.typecast(chunk, ttnn.float32),
            self.gate_weight,
            dtype=ttnn.float32,
            compute_kernel_config=self.compute_kernel_config,
        )
        if getattr(self, "router_fp32_softplus", False):
            # bf16 ttnn.softplus flushes every input below ~-5 to exactly 0, so most experts' scores
            # vanish (selection falls back to the bias alone) and an all-negative row normalises 0/0 to inf.
            # The fp32 op keeps them (max rel err ~2e-4); scores stay bf16 downstream as before.
            scores = ttnn.typecast(ttnn.sqrt(ttnn.softplus(logits)), ttnn.bfloat16)
        else:
            logits = ttnn.typecast(logits, ttnn.bfloat16)
            scores = ttnn.sqrt(ttnn.softplus(logits))
        if self.kind.router is RouterKind.HASH:
            selected = ttnn.embedding(ttnn.typecast(ids, ttnn.uint32), self.tid2eid, layout=ttnn.TILE_LAYOUT)
            selected = ttnn.typecast(selected, ttnn.uint16)
        else:
            selection_scores = ttnn.add(ttnn.typecast(scores, ttnn.float32), self.gate_bias)
            _, selected = ttnn.topk(selection_scores, c.num_experts_per_tok, dim=-1)
            selected = ttnn.typecast(selected, ttnn.uint16)
        selected = ttnn.reshape(selected, [1, 1, rows, c.num_experts_per_tok])
        weights = ttnn.gather(scores, dim=-1, index=selected)
        weights = ttnn.mul(ttnn.div(weights, ttnn.sum(weights, dim=-1, keepdim=True)), c.routed_scaling_factor)
        return selected, weights

    def _moe_forward(self, x: ttnn.Tensor, input_ids: ttnn.Tensor) -> ttnn.Tensor:
        c = self.config
        seq = x.shape[2]
        outputs = []
        chunk_tokens = getattr(self, "moe_chunk_tokens", 32)
        for start in range(0, seq, chunk_tokens):
            end = min(seq, start + chunk_tokens)
            chunk = ttnn.slice(x, [0, 0, start, 0], [1, 1, end, c.hidden_size])
            ids = ttnn.slice(input_ids, [0, start], [1, end])
            selected, weights = self._moe_route(chunk, ids)
            out = self.moe.forward(chunk, selected, weights)
            outputs.append(ttnn.slice(out, [0, 0, 0, 0], [1, 1, end - start, c.hidden_size]))
        return outputs[0] if len(outputs) == 1 else ttnn.concat(outputs, dim=2)

    def prefill_forward(
        self,
        hidden_streams: ttnn.Tensor,
        plan: PrefillPlan | BatchedPrefillPlan,
    ) -> ttnn.Tensor:
        """Paged prefill.  No torch conversion or host/device transfer occurs here."""

        if self._batch_lanes:
            if not isinstance(plan, BatchedPrefillPlan):
                raise TypeError("a batched decoder requires BatchedPrefillPlan")
            expected = (
                plan.batch_size,
                plan.seq_len,
                self.config.hc_mult,
                self.config.hidden_size,
            )
            if tuple(hidden_streams.shape) != expected:
                raise ValueError(f"hidden_streams shape {tuple(hidden_streams.shape)} != {expected}")
            outputs = []
            for lane_idx, lane_plan in enumerate(plan.lanes):
                lane_hidden = ttnn.slice(
                    hidden_streams,
                    [lane_idx, 0, 0, 0],
                    [lane_idx + 1, plan.seq_len, self.config.hc_mult, self.config.hidden_size],
                )
                outputs.append(self._batch_lanes[lane_idx].prefill_forward(lane_hidden, lane_plan))
            return outputs[0] if len(outputs) == 1 else ttnn.concat(outputs, dim=0)

        expected = (1, plan.seq_len, self.config.hc_mult, self.config.hidden_size)
        if tuple(hidden_streams.shape) != expected:
            raise ValueError(f"hidden_streams shape {tuple(hidden_streams.shape)} != {expected}")
        residual = hidden_streams
        post, comb, collapsed = self._hc_pre(residual, self.hc_attn_fn, self.hc_attn_base, self.hc_attn_scale)
        attention = self._attention_prefill(
            ttnn.rms_norm(collapsed, weight=self.attn_norm, epsilon=self.config.rms_norm_eps), plan
        )
        streams = self._hc_post(attention, residual, post, comb)
        residual = streams
        post, comb, collapsed = self._hc_pre(residual, self.hc_ffn_fn, self.hc_ffn_base, self.hc_ffn_scale)
        moe = self._moe_forward(
            ttnn.rms_norm(collapsed, weight=self.ffn_norm, epsilon=self.config.rms_norm_eps), plan.input_ids
        )
        return self._hc_post(moe, residual, post, comb)

    def _cache_rows(self, cache: PagedTensor, logical: ttnn.Tensor) -> ttnn.Tensor:
        logical_u32 = ttnn.typecast(logical, ttnn.uint32)
        page = ttnn.div(logical, self.config.page_size, rounding_mode="floor")
        page = ttnn.typecast(page, ttnn.uint32)
        physical_page = ttnn.gather(cache.page_table, dim=-1, index=page)
        physical_page = ttnn.typecast(physical_page, ttnn.uint32)
        rows = ttnn.add(
            ttnn.mul(physical_page, self.config.page_size),
            ttnn.remainder(logical_u32, self.config.page_size),
        )
        rows = ttnn.typecast(rows, ttnn.uint32)
        flat = ttnn.reshape(cache.tensor, [cache.rows, cache.width])
        return ttnn.embedding(rows, flat, layout=ttnn.TILE_LAYOUT)

    def _paged_update(self, cache: PagedTensor, value: ttnn.Tensor, position: ttnn.Tensor) -> None:
        """Format one-head decode data for paged_update_cache's canonical BH layout."""

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
        )

    def _update_compressor_decode(
        self,
        x: ttnn.Tensor,
        current_pos: ttnn.Tensor,
        kv_weight: ttnn.Tensor,
        gate_weight: ttnn.Tensor,
        ape: ttnn.Tensor,
        norm: ttnn.Tensor,
        kv_state: PagedTensor,
        score_state: PagedTensor,
        output_cache: PagedTensor,
        width: int,
        *,
        rotate_hadamard: bool,
    ) -> ttnn.Tensor:
        ratio = self.kind.compress_ratio
        slot = ttnn.remainder(current_pos, ratio)
        bias = ttnn.embedding(ttnn.typecast(slot, ttnn.uint32), ape, layout=ttnn.TILE_LAYOUT)
        raw_kv = ttnn.linear(x, kv_weight)
        raw_score = ttnn.add(ttnn.linear(x, gate_weight), bias)
        self._paged_update(kv_state, raw_kv, current_pos)
        self._paged_update(score_state, raw_score, current_pos)
        logical = ttnn.add(current_pos, self.decode_comp_offsets)
        valid = ttnn.ge(logical, 0)
        logical = ttnn.maximum(logical, 0)
        raw_kv = self._cache_rows(kv_state, logical)
        raw_score = self._cache_rows(score_state, logical)
        raw_score = ttnn.where(ttnn.reshape(valid, [1, -1, 1]), raw_score, float("-inf"))
        if ratio == 4:
            prev_kv = ttnn.slice(raw_kv, [0, 0, 0], [1, ratio, width])
            cur_kv = ttnn.slice(raw_kv, [0, ratio, width], [1, 2 * ratio, 2 * width])
            prev_score = ttnn.slice(raw_score, [0, 0, 0], [1, ratio, width])
            cur_score = ttnn.slice(raw_score, [0, ratio, width], [1, 2 * ratio, 2 * width])
            raw_kv = ttnn.concat([prev_kv, cur_kv], dim=1)
            raw_score = ttnn.concat([prev_score, cur_score], dim=1)
        pooled = ttnn.sum(ttnn.mul(raw_kv, ttnn.softmax(raw_score, dim=1)), dim=1, keepdim=True)
        pooled = ttnn.rms_norm(ttnn.reshape(pooled, [1, 1, 1, width]), weight=norm, epsilon=self.config.rms_norm_eps)
        entry_pos = ttnn.mul(ttnn.div(current_pos, ratio, rounding_mode="floor"), ratio)
        cos, sin = self._rope_values(entry_pos)
        pooled = self._rope(pooled, cos, sin)
        if rotate_hadamard:
            pooled = ttnn.matmul(pooled, self.index_hadamard)
        entry = ttnn.div(current_pos, ratio, rounding_mode="floor")
        self._paged_update(output_cache, pooled, entry)
        return pooled

    def _attention_decode(
        self,
        x: ttnn.Tensor,
        current_pos: ttnn.Tensor,
    ) -> ttnn.Tensor:
        c = self.config
        qr, q, kv, rope = self._project_q_kv(x, current_pos)
        capture_boundary = getattr(self, "_capture_prefill_boundary", None)
        if capture_boundary is not None:
            capture_boundary("query_rank", qr)
            capture_boundary("query_heads", q)
            capture_boundary("kv_heads", kv)
        self._paged_update(self.cache.local, kv, current_pos)
        logical = ttnn.add(current_pos, self.decode_local_offsets)
        local_valid = ttnn.ge(logical, 0)
        logical = ttnn.maximum(logical, 0)
        local_kv = self._cache_rows(self.cache.local, logical)
        parts = [local_kv]
        masks = [ttnn.typecast(ttnn.where(local_valid, 0.0, ATTENTION_MASK_VALUE), ttnn.bfloat16)]

        if self.kind.compress_ratio:
            self._update_compressor_decode(
                x,
                current_pos,
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
            valid_entries = ttnn.div(ttnn.add(current_pos, 1), self.kind.compress_ratio, rounding_mode="floor")
            if self.kind.attention is AttentionKind.HCA:
                valid = ttnn.lt(self.decode_compressed_ordinals, valid_entries)
                logical_comp = self.decode_compressed_ordinals
                # The 8,192-index row-major page-table gather selects
                # RmSingleRowMultiCore on Blackhole and produces wrong physical
                # rows.  Keep each lookup on the verified bounded path, then
                # compose the same logical KV bank on device.
                comp_parts = []
                for start in range(0, logical_comp.shape[-1], HCA_CACHE_GATHER_CHUNK_ROWS):
                    end = min(start + HCA_CACHE_GATHER_CHUNK_ROWS, logical_comp.shape[-1])
                    comp_parts.append(
                        self._cache_rows(
                            self.cache.compressed,
                            ttnn.slice(logical_comp, [0, start], [1, end]),
                        )
                    )
                comp_kv = comp_parts[0] if len(comp_parts) == 1 else ttnn.concat(comp_parts, dim=1)
                parts.append(comp_kv)
                masks.append(ttnn.typecast(ttnn.where(valid, 0.0, ATTENTION_MASK_VALUE), ttnn.bfloat16))
            else:
                self._update_compressor_decode(
                    x,
                    current_pos,
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
                idx_q = ttnn.reshape(ttnn.linear(qr, self.index_q), [1, 1, c.index_n_heads, c.index_head_dim])
                idx_q = ttnn.permute(idx_q, (0, 2, 1, 3))
                idx_q = self._rope(idx_q, *rope)
                idx_q = ttnn.matmul(idx_q, self.index_hadamard)
                idx_w = ttnn.mul(
                    ttnn.permute(ttnn.linear(x, self.index_weight), (0, 3, 2, 1)),
                    c.index_head_dim**-0.5 * c.index_n_heads**-0.5,
                )
                # indexer_score requires Sq=32.  Pad 31 inert rows; the guard
                # band makes every real compressed key visible to row zero.
                idx_q = ttnn.pad(idx_q, [(0, 0), (0, 0), (0, 31), (0, 0)], 0.0)
                idx_w = ttnn.pad(idx_w, [(0, 0), (0, 0), (0, 31), (0, 0)], 0.0)
                index_bank = ttnn.reshape(
                    self.cache.indexer.tensor,
                    [1, 1, self.cache.indexer.rows, c.index_head_dim],
                )
                index_bank = ttnn.concat([index_bank, self.decode_index_guard], dim=2)
                valid32 = ttnn.repeat(ttnn.reshape(valid_entries, [1, 1]), [32, 1])
                topk, topk_mask = self._indexer_topk(
                    idx_q,
                    idx_w,
                    index_bank,
                    valid32,
                    ordinals=self.decode_index_ordinals,
                )
                topk = ttnn.slice(topk, [0, 0, 0, 0], [1, 1, 1, c.index_topk])
                topk_mask = ttnn.slice(topk_mask, [0, 0, 0, 0], [1, 1, 1, c.index_topk])
                topk = ttnn.reshape(topk, [1, c.index_topk])
                topk = ttnn.typecast(topk, ttnn.int32)
                topk_valid = ttnn.typecast(
                    ttnn.eq(ttnn.reshape(topk_mask, [1, c.index_topk]), 0.0),
                    ttnn.int32,
                )
                topk = ttnn.mul(topk, topk_valid)
                parts.append(self._cache_rows(self.cache.compressed, topk))
                masks.append(ttnn.reshape(topk_mask, [1, c.index_topk]))

        bank = parts[0] if len(parts) == 1 else ttnn.concat(parts, dim=1)
        mask = masks[0] if len(masks) == 1 else ttnn.concat(masks, dim=-1)
        indices = self.decode_local_ordinals
        if bank.shape[1] > c.sliding_window:
            extra = ttnn.slice(
                self.decode_attention_extra_indices,
                [0, 0],
                [1, bank.shape[1] - c.sliding_window],
            )
            indices = ttnn.concat([indices, extra], dim=-1)
        out = self._gathered_attention(
            q, ttnn.reshape(bank, [bank.shape[1], c.head_dim]), indices, ttnn.reshape(mask, [1, 1, 1, -1])
        )
        if capture_boundary is not None:
            capture_boundary("attention_heads", out)
        out = self._rope(out, *rope, inverse=True)
        if capture_boundary is not None:
            capture_boundary("attention_heads_inverse_rope", out)
        return self._grouped_output_projection(out, 1)

    def decode_forward(
        self,
        hidden_streams: ttnn.Tensor,
        input_ids: ttnn.Tensor,
        current_pos: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """One paged decode step.  All changing values are device tensors."""

        if self._batch_lanes:
            batch_size = hidden_streams.shape[0]
            if not 1 <= batch_size <= self.max_batch_size:
                raise ValueError(f"decode batch size {batch_size} outside configured capacity {self.max_batch_size}")
            expected = (batch_size, 1, self.config.hc_mult, self.config.hidden_size)
            if tuple(hidden_streams.shape) != expected:
                raise ValueError(f"hidden_streams shape {tuple(hidden_streams.shape)} != {expected}")
            if tuple(input_ids.shape) != (batch_size, 1):
                raise ValueError(f"input_ids shape {tuple(input_ids.shape)} != {(batch_size, 1)}")
            if tuple(current_pos.shape) not in ((batch_size,), (batch_size, 1)):
                raise ValueError(
                    f"current_pos shape {tuple(current_pos.shape)} must be [{batch_size}] or [{batch_size}, 1]"
                )
            outputs = []
            for lane_idx in range(batch_size):
                lane_hidden = ttnn.slice(
                    hidden_streams,
                    [lane_idx, 0, 0, 0],
                    [lane_idx + 1, 1, self.config.hc_mult, self.config.hidden_size],
                )
                lane_ids = ttnn.slice(input_ids, [lane_idx, 0], [lane_idx + 1, 1])
                lane_pos = (
                    ttnn.slice(current_pos, [lane_idx], [lane_idx + 1])
                    if len(current_pos.shape) == 1
                    else ttnn.reshape(
                        ttnn.slice(current_pos, [lane_idx, 0], [lane_idx + 1, 1]),
                        [1],
                    )
                )
                outputs.append(self._batch_lanes[lane_idx].decode_forward(lane_hidden, lane_ids, lane_pos))
            return outputs[0] if len(outputs) == 1 else ttnn.concat(outputs, dim=0)

        expected = (1, 1, self.config.hc_mult, self.config.hidden_size)
        if tuple(hidden_streams.shape) != expected:
            raise ValueError(f"hidden_streams shape {tuple(hidden_streams.shape)} != {expected}")
        residual = hidden_streams
        post, comb, collapsed = self._hc_pre(residual, self.hc_attn_fn, self.hc_attn_base, self.hc_attn_scale)
        attention = self._attention_decode(
            ttnn.rms_norm(collapsed, weight=self.attn_norm, epsilon=self.config.rms_norm_eps), current_pos
        )
        streams = self._hc_post(attention, residual, post, comb)
        residual = streams
        post, comb, collapsed = self._hc_pre(residual, self.hc_ffn_fn, self.hc_ffn_base, self.hc_ffn_scale)
        moe = self._moe_forward(
            ttnn.rms_norm(collapsed, weight=self.ffn_norm, epsilon=self.config.rms_norm_eps), input_ids
        )
        return self._hc_post(moe, residual, post, comb)

    def capture_decode_trace(
        self,
        hidden_streams: ttnn.Tensor,
        input_ids: ttnn.Tensor,
        current_pos: ttnn.Tensor,
    ) -> DecodeTrace:
        """Warm once, capture once, then return a replayable trace handle."""

        self.decode_forward(hidden_streams, input_ids, current_pos)
        ttnn.synchronize_device(self.mesh_device)
        trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        try:
            output = self.decode_forward(hidden_streams, input_ids, current_pos)
        except BaseException:
            # Never strand the device in trace-capture mode when validation of
            # a newly supported op fails.
            ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
            ttnn.release_trace(self.mesh_device, trace_id)
            raise
        ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
        return DecodeTrace(trace_id, output)
