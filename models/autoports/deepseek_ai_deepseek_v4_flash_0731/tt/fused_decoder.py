"""Graph-fused single-chip decoder for DeepSeek-V4-Flash-0731.

The functional decoder remains the correctness oracle and supplies the stable
construction/public API.  This class changes the runtime graph explicitly; it
does not select or fall back to the functional implementation at runtime.
Every override in this module is kept only after a real-device PCC and latency
A/B against :class:`FunctionalDecoder`.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Sequence

import torch
from loguru import logger

import ttnn

from ..config import make_page_table, physical_row_lut
from .functional_decoder import (
    ATTENTION_MASK_VALUE,
    HCA_CACHE_GATHER_CHUNK_ROWS,
    AttentionKind,
    BatchedPrefillPlan,
    DecoderCache,
    DecodeTrace,
    FunctionalDecoder,
    HostLayerWeights,
    PagedTensor,
    PrefillChunk,
    PrefillPlan,
    _round_up,
)
from .sparse_moe import DeepSeekSparseMoE, _program_config

# TTNN gather sizes its circular buffers from the entire source tensor, not
# merely the selected indices.  A 1,048,576-entry int32 row LUT needs 4.31 MiB
# on one Blackhole core, above its 1.5 MiB L1.  The 262K-entry ceiling leaves
# room for gather overhead; larger caches use the exact device-side page-table
# arithmetic below while ordinary performance shapes retain the fused LUT op.
MAX_ROW_LUT_GATHER_ROWS = 262_144


class FusedDeepSeekSparseMoE(DeepSeekSparseMoE):
    """Single-chip MoE with packed routed and shared gate/up projections."""

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
        weights_dtype: ttnn.DataType = ttnn.bfloat8_b,
    ) -> None:
        self.mesh_device = mesh_device
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.limit = limit

        expected_gate = (num_experts, intermediate_size, hidden_size)
        expected_down = (num_experts, hidden_size, intermediate_size)
        if tuple(expert_w1.shape) != expected_gate or tuple(expert_w3.shape) != expected_gate:
            raise ValueError(
                f"routed gate/up shapes must be {expected_gate}, got "
                f"{tuple(expert_w1.shape)} and {tuple(expert_w3.shape)}"
            )
        if tuple(expert_w2.shape) != expected_down:
            raise ValueError(f"routed down shape must be {expected_down}, got {tuple(expert_w2.shape)}")

        mapper = ttnn.ReplicateTensorToMesh(mesh_device)

        def upload(name: str, weight: torch.Tensor) -> ttnn.Tensor:
            start = time.perf_counter()
            logger.info(
                f"Uploading fused DeepSeek sparse-MoE {name}: shape={tuple(weight.shape)} dtype={weights_dtype}"
            )
            staged = weight.contiguous() if weights_dtype == ttnn.bfloat16 else weight
            result = ttnn.from_torch(
                staged,
                device=mesh_device,
                mesh_mapper=mapper,
                dtype=weights_dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            logger.info(f"Uploaded fused DeepSeek sparse-MoE {name} in {time.perf_counter() - start:.2f}s")
            return result

        routed_gate_up_host = torch.cat([expert_w1, expert_w3], dim=1)
        self.routed_gate_up = upload(
            "routed gate+up",
            routed_gate_up_host.transpose(-2, -1).unsqueeze(0),
        )
        del routed_gate_up_host
        self.routed_down = upload("routed down", expert_w2.transpose(-2, -1).unsqueeze(0))
        shared_gate_up_host = torch.cat([shared_w1, shared_w3], dim=0)
        self.shared_gate_up = upload("shared gate+up", shared_gate_up_host.transpose(-2, -1))
        del shared_gate_up_host
        self.shared_down = upload("shared down", shared_w2.transpose(-2, -1))
        self.gate_up_program = _program_config(32, 2 * self.intermediate_size)
        self.down_program = _program_config(32, hidden_size)
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.sparsity = ttnn.ones(
            [1, 1, 1, num_experts],
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _swiglu(self, gate: ttnn.Tensor, up: ttnn.Tensor) -> ttnn.Tensor:
        """Clamp both halves, then fuse SiLU into their binary multiply."""

        gate = ttnn.clamp(gate, min=None, max=self.limit, output_tensor=gate)
        up = ttnn.clamp(up, min=-self.limit, max=self.limit, output_tensor=up)
        return ttnn.mul(
            gate,
            up,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
        )

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        selected_experts: ttnn.Tensor,
        routing_weights: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """Run routed and shared SwiGLU with one gate/up matmul apiece."""

        seq = hidden_states.shape[2]
        if tuple(hidden_states.shape) != (1, 1, seq, self.hidden_size) or not 1 <= seq <= 32:
            raise ValueError(f"MoE input must be [1,1,S,{self.hidden_size}] with 1 <= S <= 32")
        expected_routing = (1, 1, seq, self.top_k)
        if tuple(selected_experts.shape) != expected_routing or tuple(routing_weights.shape) != expected_routing:
            raise ValueError(
                f"selected experts and weights must both be {expected_routing}, got "
                f"{tuple(selected_experts.shape)} and {tuple(routing_weights.shape)}"
            )
        common = {
            "sparsity": self.sparsity,
            "is_input_b_sparse": True,
            "memory_config": ttnn.DRAM_MEMORY_CONFIG,
            "output_tile": ttnn.Tile([32, 32]),
            "dtype": ttnn.bfloat16,
            "compute_kernel_config": self.compute_kernel_config,
        }
        routed_tokens = []
        for token_idx in range(seq):
            token = ttnn.slice(
                hidden_states,
                [0, 0, token_idx, 0],
                [1, 1, token_idx + 1, self.hidden_size],
            )
            indices = ttnn.slice(
                selected_experts,
                [0, 0, token_idx, 0],
                [1, 1, token_idx + 1, self.top_k],
            )
            indices = ttnn.to_layout(ttnn.reshape(indices, [1, 1, 1, self.top_k]), ttnn.ROW_MAJOR_LAYOUT)
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
            activated = ttnn.reshape(
                self._swiglu(gate, up),
                (1, self.top_k, 1, self.intermediate_size),
            )
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
                    compute_kernel_config=self.compute_kernel_config,
                )
            )
        routed = routed_tokens[0] if len(routed_tokens) == 1 else ttnn.concat(routed_tokens, dim=2)

        shared_gate_up = ttnn.linear(
            hidden_states,
            self.shared_gate_up,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute_kernel_config,
        )
        shared_gate = ttnn.slice(
            shared_gate_up,
            [0, 0, 0, 0],
            [1, 1, seq, self.intermediate_size],
        )
        shared_up = ttnn.slice(
            shared_gate_up,
            [0, 0, 0, self.intermediate_size],
            [1, 1, seq, 2 * self.intermediate_size],
        )
        shared = self._swiglu(shared_gate, shared_up)
        shared = ttnn.linear(
            shared,
            self.shared_down,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute_kernel_config,
        )
        return ttnn.add(routed, shared)


class FusedDecoder(FunctionalDecoder):
    """Functional-equivalent decoder with verified faster TTNN subgraphs."""

    # small-table row lookups (compressor APE bias, verify candidate rows) as a one-hot matmul instead of
    # ttnn.embedding, which untilizes the whole TILE table and tilizes the selected rows (~55 us for 4 x 1024).
    compressor_select_matmul = True
    # the CSA overlap history (previous group's first half + current group's second half) as a row select
    # between two column slices instead of 4-row slices and a concat (each an untilize / tilize round trip).
    compressor_overlap_where = True

    def _select_ordinals(self, k: int, n: int) -> ttnn.Tensor:
        """The [1, 1, k, n] BF16 constant 1..n per row for _select_table_rows (built before any trace capture:
        decode sizes in _decode_constants, verify sizes in _verify_constants)."""

        ordinals = self.__dict__.setdefault("_select_ordinal_cache", {})
        if (k, n) not in ordinals:
            ordinals[(k, n)] = self._to_tt(
                torch.arange(1, n + 1, dtype=torch.float32).repeat(k, 1).reshape(1, 1, k, n), dtype=ttnn.bfloat16
            )
        return ordinals[(k, n)]

    def _overlap_prev_rows(self, width: int) -> ttnn.Tensor:
        """[1, 8, width] BF16 condition for the CSA overlap history: 1 on rows 0..3 (built in _decode_constants)."""

        rows = self.__dict__.setdefault("_overlap_prev_row_cache", {})
        if width not in rows:
            ratio = self.kind.compress_ratio
            mask = (torch.arange(2 * ratio) < ratio).to(torch.float32).reshape(1, -1, 1).expand(1, 2 * ratio, width)
            rows[width] = self._to_tt(mask, dtype=ttnn.bfloat16)
        return rows[width]

    def _decode_constants(self) -> None:
        super()._decode_constants()
        if self.kind.compress_ratio:
            self._select_ordinals(1, self.kind.compress_ratio)
        if self.kind.attention is AttentionKind.CSA and self.kind.compress_ratio == 4:
            self._overlap_prev_rows(self.config.head_dim)
            self._overlap_prev_rows(self.config.index_head_dim)

    def _select_table_rows(self, table: ttnn.Tensor, index: ttnn.Tensor) -> ttnn.Tensor:
        """Rows ``index`` (int32, k values < n) of a BF16 TILE table [.., n, W] as a [1, 1, k, W] TILE tensor.

        One-hot [k, n] x table: every output is a single table value times 1.0 accumulated in fp32, so the rows are
        exact. The one-hot compares index + 1 with 1..n, so the tile padding columns (0) never match.
        """

        n, width = int(table.shape[-2]), int(table.shape[-1])
        k = math.prod(int(d) for d in index.shape)
        memo = self.__dict__.get("_select_hot_memo")  # set while one verify compressor pass runs
        hot = None if memo is None else memo.get((id(index), k, n))
        if hot is None:
            wanted = ttnn.typecast(ttnn.reshape(ttnn.add(index, 1), [1, 1, k, 1]), ttnn.bfloat16)
            hot = ttnn.eq(ttnn.to_layout(wanted, ttnn.TILE_LAYOUT), self._select_ordinals(k, n))
            if memo is not None:
                memo[(id(index), k, n)] = hot
                memo[("keep", id(index))] = index  # keep the index alive so its id is not reused
        return ttnn.matmul(
            hot,
            ttnn.reshape(table, [1, 1, n, width]),
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute_kernel_config,
        )

    def _build_moe(self, host: HostLayerWeights) -> FusedDeepSeekSparseMoE:
        p = self._prefix
        return FusedDeepSeekSparseMoE(
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

    def _allocate_cache(
        self,
        local_permutation: Sequence[int] | None,
        compressed_permutation: Sequence[int] | None,
    ) -> DecoderCache:
        """Allocate compressor KV/score histories in shared paged tensors."""

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
        if self.kind.attention is not AttentionKind.CSA:
            compressor = self._page_tensor(
                logical_pages,
                logical_pages,
                self.comp_kv.shape[-1] + self.comp_gate.shape[-1],
                cyclic=False,
                permutation=None,
            )
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

        compressor = self._page_tensor(
            logical_pages,
            logical_pages,
            self.comp_kv.shape[-1]
            + self.comp_gate.shape[-1]
            + self.index_comp_kv.shape[-1]
            + self.index_comp_gate.shape[-1],
            cyclic=False,
            permutation=None,
        )
        index_pages = compressed_logical + 1
        indexer = self._page_tensor(
            index_pages,
            index_pages,
            c.index_head_dim,
            cyclic=False,
            permutation=None,
        )
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

    def _upload_dense_weights(self, state: dict[str, torch.Tensor]) -> None:
        """Upload the stable weights, then pack shared attention projections."""

        FunctionalDecoder._upload_dense_weights(self, state)
        p = self._prefix

        def fold_hc_scale(fn_suffix: str, scale_suffix: str, attr: str, scale_attr: str) -> None:
            scale = state[f"{p}.{scale_suffix}"].float().reshape(-1)
            if scale.numel() != 3:
                raise ValueError(f"mHC scale must have three values, got {tuple(scale.shape)}")
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
        projections = [self.q_a, self.kv]
        if self.kind.compress_ratio:
            projections.extend((self.comp_kv, self.comp_gate))
        if self.kind.attention.value == "compressed_sparse_attention":
            projections.extend((self.index_comp_kv, self.index_comp_gate, self.index_weight))
        if len(projections) < 3:
            self._packed_attention_projection = None
            self._packed_projection_ranges = {}
            self._packed_qr_projection = None
            self._packed_qr_projection_ranges = {}
            return
        ranges: dict[int, tuple[int, int]] = {}
        start = 0
        for projection in projections:
            end = start + projection.shape[-1]
            ranges[id(projection)] = (start, end)
            start = end
        self._packed_attention_projection = ttnn.concat(projections, dim=-1)
        self._packed_projection_ranges = ranges
        for projection in projections:
            ttnn.deallocate(projection)
        if self.kind.attention is AttentionKind.CSA:
            qr_projections = (self.q_b, self.index_q)
            self._packed_qr_projection_ranges = {
                id(self.q_b): (0, self.q_b.shape[-1]),
                id(self.index_q): (self.q_b.shape[-1], self.q_b.shape[-1] + self.index_q.shape[-1]),
            }
            self._packed_qr_projection = ttnn.concat(qr_projections, dim=-1)
            for projection in qr_projections:
                ttnn.deallocate(projection)
        else:
            self._packed_qr_projection = None
            self._packed_qr_projection_ranges = {}

    def _packed_projection(
        self,
        projected: ttnn.Tensor,
        weight: ttnn.Tensor,
    ) -> ttnn.Tensor:
        start, end = self._packed_projection_ranges[id(weight)]
        return ttnn.slice(
            projected,
            [0, 0, 0, start],
            [projected.shape[0], projected.shape[1], projected.shape[2], end],
        )

    def _packed_qr_projection_slice(
        self,
        projected: ttnn.Tensor,
        weight: ttnn.Tensor,
    ) -> ttnn.Tensor:
        start, end = self._packed_qr_projection_ranges[id(weight)]
        return ttnn.slice(
            projected,
            [0, 0, 0, start],
            [projected.shape[0], projected.shape[1], projected.shape[2], end],
        )

    def _page_tensor(
        self,
        physical_pages: int,
        logical_pages: int,
        width: int,
        *,
        cyclic: bool,
        permutation: Sequence[int] | None,
    ) -> PagedTensor:
        """Allocate a cache with a lossless int32 logical-row lookup table.

        The functional path stores the LUT as ``[rows, 1]`` but computes page
        division, page-table gather, remainder, multiply, and add at runtime.
        The fused path stores the same immutable mapping as ``[1, rows]`` so a
        single int32 gather replaces that arithmetic without using an
        inexact BF16 embedding table.
        """

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
        row_lut = self._to_tt(
            torch.tensor(physical_row_lut(table, self.config.page_size), dtype=torch.int32).reshape(1, -1),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        return PagedTensor(
            tensor,
            page_table,
            row_lut,
            table,
            physical_pages * self.config.page_size,
            width,
        )

    def _cache_rows(
        self,
        cache: PagedTensor,
        logical: ttnn.Tensor,
        *,
        table: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        """Gather physical cache rows through the pre-expanded int32 LUT.

        ``table``: the cache already converted to a row-major [rows, width] table by the caller, reused
        across several gathers instead of letting each embedding untilize the whole TILE cache again.
        """

        flat = ttnn.reshape(cache.tensor, [cache.rows, cache.width]) if table is None else table
        return ttnn.embedding(self._physical_rows(cache, logical), flat, layout=ttnn.TILE_LAYOUT)

    def _physical_rows(self, cache: PagedTensor, logical: ttnn.Tensor) -> ttnn.Tensor:
        """uint32 physical cache rows of the logical rows, through the pre-expanded int32 LUT."""

        logical_u32 = ttnn.typecast(logical, ttnn.uint32)
        if cache.row_lut.shape[-1] <= MAX_ROW_LUT_GATHER_ROWS:
            physical_rows = ttnn.gather(cache.row_lut, dim=-1, index=logical_u32)
        else:
            page = ttnn.typecast(
                ttnn.div(logical, self.config.page_size, rounding_mode="floor"),
                ttnn.uint32,
            )
            physical_page = ttnn.typecast(
                ttnn.gather(cache.page_table, dim=-1, index=page),
                ttnn.uint32,
            )
            physical_rows = ttnn.add(
                ttnn.mul(physical_page, self.config.page_size),
                ttnn.remainder(logical_u32, self.config.page_size),
            )
            physical_rows = ttnn.typecast(physical_rows, ttnn.uint32)
        return ttnn.typecast(physical_rows, ttnn.uint32)

    def _row_lut_kind(self, cache: PagedTensor) -> str | None:
        """'identity' (physical row == logical row), 'ring' (== logical row mod cache.rows) or None.

        Decided on the host from the page table the LUT was expanded from; cached per (tensor, table) so a
        replaced table is re-examined.
        """

        key = (id(cache.tensor), cache.host_page_table)
        kinds = self.__dict__.setdefault("_row_lut_kinds", {})
        if key not in kinds:
            lut = physical_row_lut(cache.host_page_table, self.config.page_size)
            if len(lut) == cache.rows and all(row == i for i, row in enumerate(lut)):
                kinds[key] = "identity"
            elif all(row == i % cache.rows for i, row in enumerate(lut)):
                kinds[key] = "ring"
            else:
                kinds[key] = None
        return kinds[key]

    def _decode_cache_rows(self, cache: PagedTensor, logical: ttnn.Tensor) -> ttnn.Tensor:
        """Decode-path _cache_rows. (decode_identity_rows): with an identity or identity-ring page
        table the physical row is the logical row (mod the ring), so the LUT gather is skipped; the embedding
        reads the same rows, so the result is bit-identical."""

        kind = self._row_lut_kind(cache) if getattr(self, "decode_identity_rows", False) else None
        tile_gather = getattr(self, "decode_tile_row_gather", False) and self._tile_row_gather_fits(cache, logical)
        if kind is None and not tile_gather:
            return self._cache_rows(cache, logical)
        if kind is None:
            rows = self._physical_rows(cache, logical)
        else:
            rows = ttnn.typecast(logical, ttnn.uint32)
            if kind == "ring":
                rows = ttnn.typecast(ttnn.remainder(rows, cache.rows), ttnn.uint32)
        if tile_gather:
            return self._tile_row_gather(cache, rows)
        flat = ttnn.reshape(cache.tensor, [cache.rows, cache.width])
        return ttnn.embedding(rows, flat, layout=ttnn.TILE_LAYOUT)

    @staticmethod
    def _tile_row_gather_fits(cache: PagedTensor, logical: ttnn.Tensor) -> bool:
        """The tile-row gather applies to a BFP8_b TILE DRAM-interleaved cache (no padding: its tiles are the [rows, width]
        table's tiles in row-major order) and a [1, N] row request with N a multiple of 32."""

        tensor = cache.tensor
        return (
            tensor.dtype == ttnn.bfloat8_b
            and tensor.layout == ttnn.TILE_LAYOUT
            and tensor.memory_config() == ttnn.DRAM_MEMORY_CONFIG
            and list(tensor.padded_shape) == list(tensor.shape)
            and cache.rows % 32 == 0
            and cache.width % 32 == 0
            and len(logical.shape) == 2
            and logical.shape[0] == 1
            and logical.shape[1] % 32 == 0
        )

    def _tile_row_gather(self, cache: PagedTensor, rows: ttnn.Tensor) -> ttnn.Tensor:
        """decode_tile_row_gather: the selected rows of the BFP8 TILE cache as a [1, N, width] BF16
        TILE tensor, read straight from the cache tiles (kernels/row_gather/) instead of ttnn.embedding, which
        untilizes the whole table (32768 x 512 for CSA, ~130 us per call) to read N rows. The gathered BFP8 tiles
        are unpacked and packed to BF16 like the untilize does, so the result is meant to be bit-identical."""

        table = cache.tensor
        n = int(rows.shape[1])
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, n, cache.width]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.mesh_device, ttnn.DRAM_MEMORY_CONFIG
        )
        width_tiles = cache.width // 32
        num_tiles = (n // 32) * width_tiles
        grid = self.mesh_device.compute_with_storage_grid_size()
        all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
        _, cores, group_1, group_2, per_1, per_2 = ttnn.split_work_to_cores(all_cores, num_tiles)
        cbs = [
            ttnn.CBDescriptor(
                total_size=pages * page,
                core_ranges=cores,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page)],
            )
            for index, dtype, page, pages in (
                (0, ttnn.bfloat8_b, 1088, 2),  # assembled BFP8 tiles
                (1, ttnn.uint32, 128, 1),  # this row block's 32 indices
                (2, ttnn.uint32, 32 * 3 * 64, 1),  # staging: 3 x 64 B DRAM chunks per row
                (16, ttnn.bfloat16, 2048, 2),
            )
        ]
        reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        start = 0
        for group, count in ((group_1, per_1), (group_2, per_2)):
            for core_range in group.ranges():
                for y in range(core_range.start.y, core_range.end.y + 1):
                    for x in range(core_range.start.x, core_range.end.x + 1):
                        reader_rt[x][y] = [table.buffer_address(), rows.buffer_address(), start, count]
                        writer_rt[x][y] = [out.buffer_address(), count, start]
                        compute_rt[x][y] = [count]
                        start += count
        kernels = Path(__file__).resolve().parent / "kernels" / "row_gather"
        program = ttnn.ProgramDescriptor(
            kernels=[
                ttnn.KernelDescriptor(
                    kernel_source=str(kernels / "reader_row_gather_bfp8.cpp"),
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=cores,
                    compile_time_args=[width_tiles, int(rows.layout == ttnn.TILE_LAYOUT)]
                    + ttnn.TensorAccessorArgs(table).get_compile_time_args()
                    + ttnn.TensorAccessorArgs(rows).get_compile_time_args(),
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
                    kernel_source=str(kernels / "row_gather_unpack.cpp"),
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=cores,
                    runtime_args=compute_rt,
                    config=ttnn.ComputeConfigDescriptor(),
                ),
            ],
            semaphores=[],
            cbs=cbs,
        )
        return ttnn.generic_op([table, rows, out], program)

    def _gathered_attention(
        self,
        q: ttnn.Tensor,
        bank: ttnn.Tensor,
        indices: ttnn.Tensor,
        mask: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """Run decode QK/softmax/PV as one sink-aware SDPA operation.

        Prefill has a different gathered KV bank for every query position, a
        geometry stock SDPA cannot express because its attention-sink contract
        requires batch one.  It therefore retains the explicit numerically
        equivalent graph.  Decode has one query position and maps exactly to
        the dedicated sink-aware kernel.
        """

        q_len = q.shape[2]
        padded_k = _round_up(indices.shape[-1], 128)
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

        # Each prefill query owns different gathered K/V rows.  Treat query
        # positions as the matmul batch while retaining 64 heads on M.
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

    def _rope(
        self,
        x: ttnn.Tensor,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        *,
        inverse: bool = False,
    ) -> ttnn.Tensor:
        """Apply the 64-wide interleaved RoPE slice with one dedicated op."""

        nope_width = x.shape[-1] - self.config.rope_head_dim
        nope = ttnn.slice(
            x,
            [0, 0, 0, 0],
            [x.shape[0], x.shape[1], x.shape[2], nope_width],
        )
        rope = ttnn.slice(
            x,
            [0, 0, 0, nope_width],
            list(x.shape),
        )
        if inverse:
            sin = ttnn.neg(sin)
        rope = ttnn.experimental.rotary_embedding_llama(
            rope,
            cos,
            sin,
            self.rope_trans,
            is_decode_mode=False,
            compute_kernel_config=self.compute_kernel_config,
        )
        return ttnn.concat([nope, rope], dim=-1)

    def _hc_pre(
        self,
        streams: ttnn.Tensor,
        fn: ttnn.Tensor,
        base: ttnn.Tensor,
        _scale: ttnn.Tensor | None,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        """Run mHC modulation with scale and bias folded into its linear."""

        c = self.config
        seq = streams.shape[1]
        flat = ttnn.reshape(streams, [1, 1, seq, c.hc_width])
        normed = ttnn.rms_norm(ttnn.typecast(flat, ttnn.float32), epsilon=c.rms_norm_eps)
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
        """Fuse residual addition with per-stream output placement."""

        seq = residual.shape[1]
        if (
            (
                getattr(self, "hc_post_fused", False)
                and (seq == 1 or getattr(self, "_verify_active", False))
                and seq <= 32
                # a prefill block (with the fused _hc_pre's post / comb) through the same one-matmul kernel
                or getattr(self, "prefill_hc_post_kernel", False)
                and seq > 32
                and not getattr(self, "_verify_active", False)
            )
            and tuple(comb.shape) == (1, seq, self.config.hc_mult, self.config.hc_mult)
            and tuple(post.shape) == (1, seq, self.config.hc_mult, 1)
            and not any(t.is_sharded() for t in (output, residual, post, comb))
            and all(t.dtype == ttnn.bfloat16 for t in (output, residual, post, comb))  # 2 KiB pages
        ):
            # comb^T @ residual + post * output as one matmul per tile (OptimizedDecoder)
            return self._hc_post_fused(output, residual, post, comb)
        mixed = ttnn.matmul(
            comb,
            residual,
            transpose_a=True,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute_kernel_config,
        )
        return ttnn.addcmul(
            mixed,
            ttnn.reshape(output, [1, seq, 1, self.config.hidden_size]),
            ttnn.reshape(post, [1, seq, self.config.hc_mult, 1]),
        )

    def _project_q_kv(
        self,
        x: ttnn.Tensor,
        positions: ttnn.Tensor,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, tuple[ttnn.Tensor, ttnn.Tensor]]:
        """Project every shared-LHS attention input with one packed matmul."""

        c = self.config
        cos, sin = self._rope_values(positions)
        if self._packed_attention_projection is None:
            q_a = ttnn.linear(x, self.q_a)
            kv_projected = ttnn.linear(x, self.kv)
            self._fused_attention_projections = {}
        else:
            packed = ttnn.linear(x, self._packed_attention_projection)
            q_a = self._packed_projection(packed, self.q_a)
            kv_projected = self._packed_projection(packed, self.kv)
            self._fused_attention_projections = {
                key: self._packed_projection(packed, projection)
                for key, projection in (
                    ((id(self.comp_kv), id(self.comp_gate)), self.comp_kv),
                    ((id(self.comp_gate), id(self.comp_kv)), self.comp_gate),
                )
            }
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
        qr = ttnn.rms_norm(q_a, weight=self.q_a_norm, epsilon=c.rms_norm_eps)
        if self._packed_qr_projection is None:
            q_projected = ttnn.linear(qr, self.q_b)
            self._fused_index_q_projection = None
        else:
            packed_qr = ttnn.linear(qr, self._packed_qr_projection)
            q_projected = self._packed_qr_projection_slice(packed_qr, self.q_b)
            self._fused_index_q_projection = self._packed_qr_projection_slice(packed_qr, self.index_q)
        q = ttnn.reshape(
            q_projected,
            [1, x.shape[2], c.num_attention_heads, c.head_dim],
        )
        q = ttnn.permute(q, (0, 2, 1, 3))
        q = ttnn.rms_norm(q, epsilon=c.rms_norm_eps)
        q = self._rope(q, cos, sin)
        kv = ttnn.rms_norm(
            kv_projected,
            weight=self.kv_norm,
            epsilon=c.rms_norm_eps,
        )
        kv = self._rope(
            ttnn.reshape(kv, [1, 1, x.shape[2], c.head_dim]),
            cos,
            sin,
        )
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
        """Project compressor KV/gates once for pooling and persisted state."""

        ratio = self.kind.compress_ratio
        cutoff = (x.shape[2] // ratio) * ratio
        if cutoff == 0:
            raise ValueError(f"prefill length {x.shape[2]} produces no complete ratio-{ratio} entry")

        packed_cache = getattr(self, "_fused_attention_projections", {})
        try:
            raw_kv = packed_cache.pop((id(kv_weight), id(gate_weight)))
            raw_score_projection = packed_cache.pop((id(gate_weight), id(kv_weight)))
        except KeyError as exc:
            raise RuntimeError("compressor projection escaped the fused shared-LHS graph") from exc
        slot = ttnn.remainder(positions, ratio)
        bias = ttnn.embedding(
            ttnn.typecast(slot, ttnn.uint32),
            ape,
            layout=ttnn.TILE_LAYOUT,
        )
        bias = ttnn.reshape(bias, [1, 1, x.shape[2], gate_weight.shape[-1]])
        raw_score = ttnn.add(raw_score_projection, bias)
        cache = getattr(self, "_fused_prefill_projections", None)
        if cache is None:
            cache = {}
            self._fused_prefill_projections = cache
        cache[(id(kv_weight), id(gate_weight))] = (raw_kv, raw_score)

        groups = cutoff // ratio
        kv = ttnn.reshape(
            ttnn.slice(raw_kv, [0, 0, 0, 0], [1, 1, cutoff, raw_kv.shape[-1]]),
            [1, groups, ratio, -1],
        )
        score = ttnn.reshape(
            ttnn.slice(raw_score, [0, 0, 0, 0], [1, 1, cutoff, raw_score.shape[-1]]),
            [1, groups, ratio, -1],
        )
        if ratio == 4:
            prev_kv = ttnn.slice(kv, [0, 0, 0, 0], [1, groups, ratio, width])
            cur_kv = ttnn.slice(kv, [0, 0, 0, width], [1, groups, ratio, 2 * width])
            prev_score = ttnn.slice(score, [0, 0, 0, 0], [1, groups, ratio, width])
            cur_score = ttnn.slice(score, [0, 0, 0, width], [1, groups, ratio, 2 * width])
            zero = ttnn.zeros_like(ttnn.slice(prev_kv, [0, 0, 0, 0], [1, 1, ratio, width]))
            neginf = ttnn.full_like(zero, float("-inf"))
            prev_kv = (
                ttnn.concat(
                    [zero, ttnn.slice(prev_kv, [0, 0, 0, 0], [1, groups - 1, ratio, width])],
                    dim=1,
                )
                if groups > 1
                else zero
            )
            prev_score = (
                ttnn.concat(
                    [neginf, ttnn.slice(prev_score, [0, 0, 0, 0], [1, groups - 1, ratio, width])],
                    dim=1,
                )
                if groups > 1
                else neginf
            )
            kv = ttnn.concat([prev_kv, cur_kv], dim=2)
            score = ttnn.concat([prev_score, cur_score], dim=2)
        pooled = ttnn.sum(
            ttnn.mul(kv, ttnn.softmax(score, dim=2)),
            dim=2,
            keepdim=True,
        )
        pooled = ttnn.rms_norm(
            pooled,
            weight=norm,
            epsilon=self.config.rms_norm_eps,
        )
        compressed_positions = ttnn.slice(positions, [0, 0], [1, cutoff], [1, ratio])
        cos, sin = self._rope_values(compressed_positions)
        pooled = self._rope(
            ttnn.reshape(pooled, [1, 1, groups, width]),
            cos,
            sin,
        )
        if rotate_hadamard:
            pooled = ttnn.matmul(pooled, self.index_hadamard)
        return pooled

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
        """Persist cached compressor projections without duplicate matmuls."""

        cache = getattr(self, "_fused_prefill_projections", {})
        projected = cache.pop((id(kv_weight), id(gate_weight)), None)
        if projected is None:
            # Prompts shorter than the compressor ratio have no pooled entry,
            # hence no earlier projections to reuse.
            slot = ttnn.remainder(positions, self.kind.compress_ratio)
            bias = ttnn.embedding(
                ttnn.typecast(slot, ttnn.uint32),
                ape,
                layout=ttnn.TILE_LAYOUT,
            )
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

        def fill(target: PagedTensor, value: ttnn.Tensor) -> None:
            tail = ttnn.slice(
                value,
                [0, 0, plan.compressor_fill_start, 0],
                [1, 1, plan.seq_len, value.shape[-1]],
            )
            missing = plan.compressor_fill_length - tail.shape[2]
            if missing:
                tail = ttnn.pad(
                    tail,
                    [(0, 0), (0, 0), (0, missing), (0, 0)],
                    0.0,
                )
            ttnn.experimental.paged_fill_cache(
                target.tensor,
                tail,
                plan.compressor_fill_page_table,
                batch_idx=0,
            )

        fill(kv_cache, packed)

    def _attention_prefill(
        self,
        x: ttnn.Tensor,
        plan: PrefillPlan,
    ) -> ttnn.Tensor:
        """Prefill attention with packed CSA query and index projections."""

        c = self.config
        _, q, kv, rope = self._project_q_kv(x, plan.position_ids)
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
                idx_w = ttnn.permute(idx_w, (0, 3, 2, 1))
                topk, topk_mask = self._indexer_topk(idx_q, idx_w, index_bank, chunk.compressed_counts)
                topk = ttnn.reshape(topk, [chunk.end - chunk.start, c.index_topk])
                flat_topk_mask = ttnn.reshape(topk_mask, [chunk.end - chunk.start, c.index_topk])
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
        """Decode compressor using the packed attention-input projection."""

        ratio = self.kind.compress_ratio
        memo = self.__dict__.get("_select_hot_memo")
        slot = None if memo is None else memo.get(("slot", id(current_pos), ratio))
        if slot is None:
            slot = ttnn.remainder(current_pos, ratio)
            if memo is not None:
                memo[("slot", id(current_pos), ratio)] = slot
                memo[("keep", id(current_pos))] = current_pos
        if self.compressor_select_matmul:
            bias = self._select_table_rows(ape, slot)
        else:
            bias = ttnn.embedding(
                ttnn.typecast(slot, ttnn.uint32),
                ape,
                layout=ttnn.TILE_LAYOUT,
            )
        packed_cache = getattr(self, "_fused_attention_projections", {})
        try:
            raw_kv = packed_cache.pop((id(kv_weight), id(gate_weight)))
            raw_score_projection = packed_cache.pop((id(gate_weight), id(kv_weight)))
        except KeyError as exc:
            raise RuntimeError("decode compressor escaped the fused shared-LHS graph") from exc
        raw_score = ttnn.add(raw_score_projection, bias)
        if kv_state is not score_state:
            raise RuntimeError("decode compressor escaped the fused paged-cache allocation")

        if self.kind.attention is AttentionKind.CSA and not rotate_hadamard:
            self._fused_decode_history_pending = (
                raw_kv,
                raw_score,
                kv_weight.shape[-1],
                gate_weight.shape[-1],
                norm,
                kv_state,
                output_cache,
                width,
            )
            return raw_kv

        histories: list[tuple[ttnn.Tensor, ttnn.Tensor, int, int, ttnn.Tensor, PagedTensor, int, bool]]
        if self.kind.attention is AttentionKind.CSA:
            try:
                (
                    attention_kv,
                    attention_score,
                    attention_kv_width,
                    attention_score_width,
                    attention_norm,
                    attention_state,
                    attention_output,
                    attention_width,
                ) = self._fused_decode_history_pending
            except AttributeError as exc:
                raise RuntimeError("CSA compressor histories escaped their fused decode update") from exc
            del self._fused_decode_history_pending
            if attention_state is not kv_state:
                raise RuntimeError("CSA compressor histories escaped the shared paged tensor")
            self._paged_update(
                kv_state,
                ttnn.concat([attention_kv, attention_score, raw_kv, raw_score], dim=-1),
                current_pos,
            )
            histories = [
                (
                    attention_kv,
                    attention_score,
                    attention_kv_width,
                    attention_score_width,
                    attention_norm,
                    attention_output,
                    attention_width,
                    False,
                ),
                (
                    raw_kv,
                    raw_score,
                    kv_weight.shape[-1],
                    gate_weight.shape[-1],
                    norm,
                    output_cache,
                    width,
                    True,
                ),
            ]
        else:
            self._paged_update(kv_state, ttnn.concat([raw_kv, raw_score], dim=-1), current_pos)
            histories = [
                (
                    raw_kv,
                    raw_score,
                    kv_weight.shape[-1],
                    gate_weight.shape[-1],
                    norm,
                    output_cache,
                    width,
                    rotate_hadamard,
                )
            ]

        if not getattr(self, "compressor_pool_enabled", True):
            # this token completes no compressed entry, so the pooled (partial) entry is never read —
            # decode masks entries >= (pos + 1) // ratio. Only the raw state above is needed.
            return raw_kv
        logical = ttnn.add(current_pos, self.decode_comp_offsets)
        valid = ttnn.ge(logical, 0)
        logical = ttnn.maximum(logical, 0)
        raw = self._decode_cache_rows(kv_state, logical)
        entry_pos = ttnn.mul(
            ttnn.div(current_pos, ratio, rounding_mode="floor"),
            ratio,
        )
        cos, sin = self._rope_values(entry_pos)
        entry = ttnn.div(current_pos, ratio, rounding_mode="floor")
        offset = 0
        outputs = []
        for _, _, kv_width, score_width, history_norm, history_output, history_width, rotate in histories:
            if getattr(self, "compressor_pool_fused", False):
                # the whole pooling below as one program (OptimizedDecoder._comp_pool_fused)
                pooled = self._comp_pool_fused(
                    raw, current_pos, offset, offset + kv_width, history_width, history_norm, cos, sin, rotate
                )
                offset += kv_width + score_width
                self._paged_update(history_output, pooled, entry)
                outputs.append(pooled)
                continue
            history_kv = ttnn.slice(raw, [0, 0, offset], [1, raw.shape[1], offset + kv_width])
            offset += kv_width
            history_score = ttnn.slice(raw, [0, 0, offset], [1, raw.shape[1], offset + score_width])
            offset += score_width
            # ttnn.where with an int32 row-major [1, rows, 1] condition broadcast against the TILE scores reads
            # stale device memory and intermittently masks whole valid rows to -inf (history dependent); give
            # it a bf16 condition with the scores' own shape and layout instead.
            valid_rows = ttnn.repeat(
                ttnn.typecast(ttnn.reshape(valid, [1, -1, 1]), ttnn.bfloat16),
                [1, 1, history_score.shape[-1]],
            )
            if valid_rows.layout != history_score.layout:
                valid_rows = ttnn.to_layout(valid_rows, history_score.layout)
            history_score = ttnn.where(valid_rows, history_score, float("-inf"))
            if ratio == 4 and self.compressor_overlap_where:
                # rows 0..3 take the previous group's half (columns 0..W), rows 4..7 the current group's (W..2W):
                # tile-aligned column slices and a row select instead of 4-row slices + concat through row-major
                prev_rows = self._overlap_prev_rows(history_width)
                history_kv = ttnn.where(
                    prev_rows,
                    ttnn.slice(history_kv, [0, 0, 0], [1, 2 * ratio, history_width]),
                    ttnn.slice(history_kv, [0, 0, history_width], [1, 2 * ratio, 2 * history_width]),
                )
                history_score = ttnn.where(
                    prev_rows,
                    ttnn.slice(history_score, [0, 0, 0], [1, 2 * ratio, history_width]),
                    ttnn.slice(history_score, [0, 0, history_width], [1, 2 * ratio, 2 * history_width]),
                )
            elif ratio == 4:
                prev_kv = ttnn.slice(history_kv, [0, 0, 0], [1, ratio, history_width])
                cur_kv = ttnn.slice(
                    history_kv,
                    [0, ratio, history_width],
                    [1, 2 * ratio, 2 * history_width],
                )
                prev_score = ttnn.slice(history_score, [0, 0, 0], [1, ratio, history_width])
                cur_score = ttnn.slice(
                    history_score,
                    [0, ratio, history_width],
                    [1, 2 * ratio, 2 * history_width],
                )
                history_kv = ttnn.concat([prev_kv, cur_kv], dim=1)
                history_score = ttnn.concat([prev_score, cur_score], dim=1)
            pooled = ttnn.sum(
                ttnn.mul(history_kv, ttnn.softmax(history_score, dim=1)),
                dim=1,
                keepdim=True,
            )
            pooled = ttnn.rms_norm(
                ttnn.reshape(pooled, [1, 1, 1, history_width]),
                weight=history_norm,
                epsilon=self.config.rms_norm_eps,
            )
            pooled = self._rope(pooled, cos, sin)
            if rotate:
                pooled = ttnn.matmul(pooled, self.index_hadamard)
            self._paged_update(history_output, pooled, entry)
            outputs.append(pooled)
        return outputs[-1]

    def _attention_decode(
        self,
        x: ttnn.Tensor,
        current_pos: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """Decode attention with every CSA shared-input projection fused."""

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
        local_kv = self._decode_cache_rows(self.cache.local, logical)
        decode_boundary_target = getattr(self, "_decode_boundary_capture", None)
        if decode_boundary_target is not None:
            local_tensors = ttnn.get_device_tensors(local_kv) or (local_kv,)
            decode_boundary_target["local_kv_bank"] = ttnn.to_torch(local_tensors[0])[0].detach().float().cpu().clone()
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
                compressed = self.cache.compressed
                if (
                    getattr(self, "decode_hca_full_table", False)
                    and logical_comp.shape[-1] == compressed.rows
                    and self._row_lut_kind(compressed) == "identity"
                ):
                    # every entry in order on an identity table is the whole table; a TILE typecast
                    # decodes the same BFP8 tiles the untilize + embedding did, without the row gather.
                    comp_kv = ttnn.typecast(
                        ttnn.reshape(compressed.tensor, [1, compressed.rows, compressed.width]),
                        ttnn.bfloat16,
                    )
                else:
                    comp_parts = []
                    for start in range(0, logical_comp.shape[-1], HCA_CACHE_GATHER_CHUNK_ROWS):
                        end = min(start + HCA_CACHE_GATHER_CHUNK_ROWS, logical_comp.shape[-1])
                        comp_parts.append(
                            self._decode_cache_rows(
                                compressed,
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
                if self._fused_index_q_projection is None:
                    raise RuntimeError("CSA index query escaped the fused QR projection")
                idx_q = ttnn.reshape(
                    self._fused_index_q_projection,
                    [1, 1, c.index_n_heads, c.index_head_dim],
                )
                if getattr(self, "decode_index_q_compact", False):
                    # RoPE and the Hadamard rotation on the 64 heads as 64 rows of one [64, 128] matrix
                    # (2 tile rows) instead of 64 one-row tiles; every head row sees the same position's cos /
                    # sin, and each output element is computed the same way. Heads move to dim 1 afterwards.
                    cos, sin = rope
                    idx_q = self._rope(
                        idx_q,
                        ttnn.repeat(cos, ttnn.Shape([1, 1, c.index_n_heads, 1])),
                        ttnn.repeat(sin, ttnn.Shape([1, 1, c.index_n_heads, 1])),
                    )
                    idx_q = ttnn.matmul(idx_q, self.index_hadamard)
                    idx_q = ttnn.permute(idx_q, (0, 2, 1, 3))
                else:
                    idx_q = ttnn.permute(idx_q, (0, 2, 1, 3))
                    idx_q = self._rope(idx_q, *rope)
                    idx_q = ttnn.matmul(idx_q, self.index_hadamard)
                idx_w = ttnn.permute(self._fused_index_weight_projection, (0, 3, 2, 1))
                idx_q = ttnn.pad(idx_q, [(0, 0), (0, 0), (0, 31), (0, 0)], 0.0)
                idx_w = ttnn.pad(idx_w, [(0, 0), (0, 0), (0, 31), (0, 0)], 0.0)
                indexer_rows = self.cache.indexer.tensor.shape[0] * c.page_size  # mapped pages + any spare
                index_bank = ttnn.reshape(self.cache.indexer.tensor, [1, 1, indexer_rows, c.index_head_dim])
                guard_rows = self.decode_index_guard.shape[2]
                if not (
                    getattr(self, "decode_index_bank_view", False)
                    and indexer_rows == self.cache.indexer.rows + guard_rows
                ):
                    # off: concatenate the guard to the mapped rows (a full copy of the bank per token).
                    if indexer_rows != self.cache.indexer.rows:
                        index_bank = ttnn.slice(
                            index_bank, [0, 0, 0, 0], [1, 1, self.cache.indexer.rows, c.index_head_dim]
                        )
                    index_bank = ttnn.concat([index_bank, self.decode_index_guard], dim=2)
                if getattr(self, "decode_indexer_row0", False):
                    topk, topk_mask = self._indexer_topk_decode(
                        idx_q,
                        idx_w,
                        index_bank,
                        valid_entries,
                        self.decode_index_ordinals,
                    )
                else:
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
                parts.append(self._decode_cache_rows(self.cache.compressed, topk))
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
        # indices are 0..N-1 (local ordinals then the consecutive extra indices).
        identity = {"identity": True} if getattr(self, "decode_identity_gather", False) else {}
        out = self._gathered_attention(
            q,
            ttnn.reshape(bank, [bank.shape[1], c.head_dim]),
            indices,
            ttnn.reshape(mask, [1, 1, 1, -1]),
            **identity,
        )
        if capture_boundary is not None:
            capture_boundary("attention_heads", out)
        out = self._rope(out, *rope, inverse=True)
        if capture_boundary is not None:
            capture_boundary("attention_heads_inverse_rope", out)
        return self._grouped_output_projection(out, 1)


__all__ = [
    "BatchedPrefillPlan",
    "DecodeTrace",
    "DecoderCache",
    "FusedDecoder",
    "FusedDeepSeekSparseMoE",
    "HostLayerWeights",
    "PagedTensor",
    "PrefillChunk",
    "PrefillPlan",
]
