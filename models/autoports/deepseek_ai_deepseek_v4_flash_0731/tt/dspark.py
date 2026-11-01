"""The DSpark draft model of the 0731 checkpoint (mtp.0 / mtp.1 / mtp.2) on the TP4 mesh.

Semantics follow the official inference/model.py (forward_spec, DSparkAttention):
  * context: for every committed position p, main_x[p] = main_norm(main_proj(concat of the main model's layer
    40 / 41 / 42 outputs at p, each the mean of its 4 mHC streams)); each draft layer writes
    kv_norm(wkv(main_x[p])) + RoPE(p) into its own 128-window ring;
  * a block anchored at position P (context = positions P - 128 .. P - 1) is [anchor, noise x (K - 1)] at
    positions P .. P + K - 1, embedded with the main embedding; each draft layer is an ordinary sliding-window,
    learned-router Block whose queries see the context ring plus the block's own K keys (bidirectional);
  * head: mtp.2's mHC head and norm, the main LM head, then sequentially logits[i] += markov_w2(markov_w1[prev])
    with prev = the anchor, then each argmax; draft i is the guess for position P + 1 + i.
The draft layers reuse MultichipDecoder (same weights layout as a sliding-window main layer) with the attention
replaced, the M-row verify machinery (verify_route / verify_finish, grouped MoE from a small expert pool) and the
main model's embedding and LM head.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Mapping, Sequence

import torch
from safetensors import safe_open

import ttnn

from ..checkpoint import CheckpointIndex, LayerCheckpoint, dequantize_block_fp8
from ..config import AttentionKind, DeepSeekV4FlashConfig, LayerKind, RouterKind, required_checkpoint_shapes
from .expert_pool import ExpertHostStore
from .expert_pool import expert_cache_dir as expert_cache_dir_for
from .functional_decoder import ATTENTION_MASK_VALUE, HostLayerWeights, _round_up
from .multichip_decoder import CCL_NUM_LINKS, TP_AXIS, TP_DEGREE, MultichipDecoder

DRAFT_LAYERS = 3


@dataclasses.dataclass(frozen=True)
class _DraftLayerConfig(DeepSeekV4FlashConfig):
    """The main config with every layer index read as a sliding-window, learned-router layer (the draft layers)."""

    def layer_kind(self, layer_idx: int) -> LayerKind:
        return LayerKind(attention=AttentionKind.SLIDING, router=RouterKind.LEARNED, compress_ratio=0)


@dataclasses.dataclass(frozen=True)
class _AliasIndex(CheckpointIndex):
    """A checkpoint index whose ``layers.0.*`` keys read the tensors of one ``mtp.<i>.*`` layer."""

    alias: Mapping[str, str] = dataclasses.field(default_factory=dict)

    def read(self, key: str) -> torch.Tensor:
        with safe_open(self.root / self.weight_map[key], framework="pt", device="cpu") as f:
            return f.get_tensor(self.alias.get(key, key))

    def read_many(self, keys: Sequence[str]) -> dict[str, torch.Tensor]:
        by_shard: dict[str, list[str]] = {}
        for key in keys:
            by_shard.setdefault(self.weight_map[key], []).append(key)
        result = {}
        for shard, shard_keys in by_shard.items():
            with safe_open(self.root / shard, framework="pt", device="cpu") as f:
                result.update({key: f.get_tensor(self.alias.get(key, key)) for key in shard_keys})
        return result


def _draft_checkpoint(root: Path, draft_idx: int, config: DeepSeekV4FlashConfig) -> LayerCheckpoint:
    base = CheckpointIndex.open(root)
    prefix = f"mtp.{draft_idx}."
    alias = {"layers.0." + key[len(prefix) :]: key for key in base.weight_map if key.startswith(prefix)}
    checkpoint = LayerCheckpoint.__new__(LayerCheckpoint)
    checkpoint.config = config
    checkpoint.layer_idx = 0
    checkpoint.index = _AliasIndex(
        root=base.root, weight_map={k: base.weight_map[v] for k, v in alias.items()}, alias=alias
    )
    checkpoint.expected = required_checkpoint_shapes(config, 0)
    return checkpoint


class DSparkLayer(MultichipDecoder):
    """One draft layer: a sliding-window learned-router decoder whose block attention reads the context ring
    (written from main_x by write_context) plus the block's own keys, bidirectionally."""

    def _dspark_constants(self, m: int) -> dict[str, ttnn.Tensor]:
        cache = self.__dict__.setdefault("_dspark_constant_cache", {})
        if m not in cache:
            window = self.config.sliding_window
            cache[m] = {
                "context_offsets": self._host_int(torch.arange(-window, 0).reshape(1, -1)),
                "block_mask": self._to_tt(torch.zeros(1, 1, m, m), dtype=ttnn.bfloat16),
            }
        return cache[m]

    def write_context(self, main_x: ttnn.Tensor, start: ttnn.Tensor) -> None:
        """Write kv_norm(wkv(main_x)) + RoPE for the M rows of ``main_x`` [1, 1, M, H] at positions start ..
        start + M - 1 of the ring (rows past the committed ones are overwritten before any block reads them)."""

        m = int(main_x.shape[2])
        const = self._verify_constants(m)
        positions = ttnn.add(const["rows"], start)
        _, _, kv, _ = self._project_q_kv(main_x, positions)
        for r in range(m):
            self._paged_update(self.cache.local, self._verify_row(kv, r, m), start if r == 0 else ttnn.add(start, r))

    def _attention_verify(self, x: ttnn.Tensor, start: ttnn.Tensor) -> ttnn.Tensor:
        """DSpark block attention: x [1, 1, K, H] at positions start .. start + K - 1; keys = the ring's context
        positions start - 128 .. start - 1 (those >= 0) and the block's K keys, all visible to every row."""

        c = self.config
        m = int(x.shape[2])
        const = self._verify_constants(m)
        dconst = self._dspark_constants(m)
        positions = ttnn.add(const["rows"], start)
        qr, q, kv, rope = self._project_q_kv(x, positions)
        logical = ttnn.add(start, dconst["context_offsets"])
        valid = ttnn.ge(logical, 0)
        context = self._decode_cache_rows(self.cache.local, ttnn.maximum(logical, 0))  # [1, W, D]
        window = int(dconst["context_offsets"].shape[-1])
        context_mask = ttnn.typecast(ttnn.where(valid, 0.0, ATTENTION_MASK_VALUE), ttnn.bfloat16)
        context_mask = ttnn.to_layout(ttnn.reshape(context_mask, [1, 1, 1, window]), ttnn.TILE_LAYOUT)
        mask = ttnn.concat([ttnn.repeat(context_mask, [1, 1, m, 1]), dconst["block_mask"]], dim=-1)
        keys = ttnn.concat(
            [ttnn.reshape(context, [1, 1, window, c.head_dim]), ttnn.reshape(kv, [1, 1, m, c.head_dim])], dim=2
        )
        n = window + m
        pad = _round_up(n, 128) - n
        if pad:
            keys = ttnn.pad(keys, [(0, 0), (0, 0), (0, pad), (0, 0)], 0.0)
            mask = ttnn.pad(mask, [(0, 0), (0, 0), (0, 0), (0, pad)], ATTENTION_MASK_VALUE)
        out = ttnn.transformer.scaled_dot_product_attention(
            q,
            keys,
            keys,
            attn_mask=mask,
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
        out = self._rope(out, *rope, inverse=True)
        return self._grouped_output_projection(out, m, verify_rows=True)


class DSparkDrafter:
    """The three draft layers plus main_proj / main_norm, mtp.2's head and the Markov head, bound to a loaded
    DeepSeekV4FlashModel (whose embedding, LM head and RoPE tables it shares)."""

    def __init__(
        self,
        model,
        checkpoint_root: str | Path,
        *,
        block: int,
        expert_pool_size: int = 32,
        expert_cache_dir: str | Path | None = None,
        attention_dtype: ttnn.DataType = ttnn.bfloat8_b,
    ) -> None:
        root = Path(checkpoint_root)
        spec = json.loads((root / "inference/config.json").read_text())
        if spec["n_mtp_layers"] != DRAFT_LAYERS or list(spec["dspark_target_layer_ids"]) != [40, 41, 42]:
            raise ValueError(f"unexpected DSpark config {spec}")
        self.model = model
        self.mesh_device = model.mesh_device
        self.block = int(block)
        self.noise_token = int(spec["dspark_noise_token_id"])
        self.target_layers = tuple(spec["dspark_target_layer_ids"])
        main_config = model.config
        config = _DraftLayerConfig(**{f.name: getattr(main_config, f.name) for f in dataclasses.fields(main_config)})
        rope_source = model.layers[model.layer_indices.index(0)] if 0 in model.layer_indices else None
        self.layers: list[DSparkLayer] = []
        for draft_idx in range(DRAFT_LAYERS):
            checkpoint = _draft_checkpoint(root, draft_idx, config)
            store = None
            if expert_cache_dir is not None:
                # keyed as layer 43 + i (the draft layers follow the 43 main layers in the official numbering)
                layer_dir, key = expert_cache_dir_for(
                    expert_cache_dir, root, 43 + draft_idx, tp_degree=TP_DEGREE, dtype=ttnn.bfloat4_b
                )
                if ExpertHostStore.cache_valid(layer_dir, key, config.num_experts):
                    store = ExpertHostStore.from_cache(layer_dir, dtype=ttnn.bfloat4_b, num_experts=config.num_experts)
            state, w1, w2, w3 = checkpoint.materialize(include_experts=store is None)
            if expert_cache_dir is not None and store is None:
                from .model import _flat_shard_mapper, _save_expert_cache

                store = ExpertHostStore.from_weights(
                    self.mesh_device,
                    w1,
                    w2,
                    w3,
                    tp_degree=TP_DEGREE,
                    shard_mapper=_flat_shard_mapper,
                    dtype=ttnn.bfloat4_b,
                )
                _save_expert_cache(store, layer_dir, key)
                w1 = w2 = w3 = None
            layer = DSparkLayer(
                self.mesh_device,
                0,
                HostLayerWeights(state, w1, w2, w3, expert_store=store),
                config=config,
                max_context=model.max_context,
                max_batch_size=1,
                weights_dtype=ttnn.bfloat16,
                optimization_profile="optimized",
                deployment_precision=model.deployment_mode,
                moe_compute_profile=model.moe_compute_profile,
                dense_compute_profile=model.dense_compute_profile,
                shared_rope=(
                    (rope_source.rope_cos, rope_source.rope_sin, rope_source.rope_trans) if rope_source else None
                ),
                expert_pool_size=expert_pool_size,
                attention_dtype=attention_dtype,
            )
            self.layers.append(layer)
            del checkpoint, state, w1, w2, w3, store
        self.buffers = self.layers[0].moe.make_verify_buffers(self.block)
        self._zero_rows = self._tt(torch.zeros(1, self.CONTEXT_CHUNK, 1, main_config.hidden_size), dtype=ttnn.bfloat16)
        self._load_heads(root)

    def _tt(self, tensor: torch.Tensor, *, dtype, layout=ttnn.TILE_LAYOUT, mapper=None) -> ttnn.Tensor:
        return ttnn.from_torch(
            tensor.contiguous(),
            device=self.mesh_device,
            mesh_mapper=mapper or ttnn.ReplicateTensorToMesh(self.mesh_device),
            dtype=dtype,
            layout=layout,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _load_heads(self, root: Path) -> None:
        index = CheckpointIndex.open(root)
        last = f"mtp.{DRAFT_LAYERS - 1}."
        raw = index.read_many(
            [
                "mtp.0.main_proj.weight",
                "mtp.0.main_proj.scale",
                "mtp.0.main_norm.weight",
                last + "norm.weight",
                last + "hc_head_fn",
                last + "hc_head_base",
                last + "hc_head_scale",
                last + "markov_head.markov_w1.weight",
                last + "markov_head.markov_w2.weight",
            ]
        )
        shard_last = ttnn.ShardTensorToMesh(self.mesh_device, dim=-1)
        main_proj = dequantize_block_fp8(raw["mtp.0.main_proj.weight"], raw["mtp.0.main_proj.scale"])  # [4096, 12288]
        self.main_proj = self._tt(main_proj.transpose(0, 1).to(torch.bfloat16), dtype=ttnn.bfloat8_b, mapper=shard_last)
        self.main_norm = self._tt(
            raw["mtp.0.main_norm.weight"].reshape(1, 1, 1, -1).to(torch.bfloat16), dtype=ttnn.bfloat16
        )
        self.final_norm = self._tt(
            raw[last + "norm.weight"].reshape(1, 1, 1, -1).to(torch.bfloat16), dtype=ttnn.bfloat16
        )
        self.head_params = (
            self._tt(raw[last + "hc_head_fn"].float().transpose(0, 1), dtype=ttnn.float32),
            self._tt(raw[last + "hc_head_scale"].float().reshape(1, 1, 1, -1), dtype=ttnn.float32),
            self._tt(raw[last + "hc_head_base"].float().reshape(1, 1, 1, -1), dtype=ttnn.float32),
        )
        # markov_w1: [V, rank] embedding rows (replicated, row-major for ttnn.embedding); markov_w2: [V, rank]
        # head, vocab-sharded like the LM head so each device adds the bias to its own logits slice.
        self.markov_w1 = self._tt(
            raw[last + "markov_head.markov_w1.weight"].to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        w2 = raw[last + "markov_head.markov_w2.weight"].float()  # [V, rank]
        self.markov_w2 = self._tt(
            w2.transpose(0, 1).to(torch.bfloat16).reshape(1, 1, w2.shape[1], w2.shape[0]),
            dtype=ttnn.bfloat16,
            mapper=shard_last,
        )

    def _position(self, position: int) -> ttnn.Tensor:
        return ttnn.reshape(self._int([position]), [1])

    def _int(self, values: Sequence[int], dtype=ttnn.int32) -> ttnn.Tensor:
        return self._tt(
            torch.tensor(list(values), dtype=torch.int32).reshape(1, -1), dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT
        )

    CONTEXT_CHUNK = 32

    def warm_compile(self, verify_rows: int, at: int) -> None:
        """Compile every drafter program and build every lazily created device constant (context writes of 1 /
        CONTEXT_CHUNK / verify_rows rows, the K-row block, the Markov head) with no trace alive: a program
        compiled or a long-lived buffer allocated behind a live trace can land on memory a replay rewrites.
        The garbage context goes to positions at .. at + 31, which the caller picks past every committed position
        a block can read (they are rewritten before any block reads them)."""

        self.release_trace()
        hidden = self.model.config.hidden_size
        for rows in sorted({1, self.CONTEXT_CHUNK, int(verify_rows)}):
            zeros = self._tt(torch.zeros(1, rows, 1, hidden), dtype=ttnn.bfloat16)
            self.write_context({layer: zeros for layer in self.target_layers}, at)
        self.draft(0, at)
        ttnn.synchronize_device(self.mesh_device)

    def register_rings(self) -> None:
        """Snapshot / restore the context rings with the main model's (session continuation)."""

        self.model.extra_ring_tensors = [layer.cache.local.tensor for layer in self.layers]

    # ------------------------------------------------------------------------------------------------ context
    def write_context_rows(self, taps: Mapping[int, ttnn.Tensor], start: int, first: int, count: int) -> None:
        """Write rows first .. first + count - 1 of [1, rows, 1, H] taps as positions start + first .. in chunks
        of CONTEXT_CHUNK rows (the last chunk zero-padded: its spare rows land at most 31 positions past the
        written ones, outside every window that reads the ring before they are rewritten)."""

        chunk = self.CONTEXT_CHUNK
        total = int(next(iter(taps.values())).shape[1])
        for offset in range(first, first + count, chunk):
            end = min(offset + chunk, total)
            part = {}
            for layer, tap in taps.items():
                piece = ttnn.slice(tap, [0, offset, 0, 0], [1, end, 1, int(tap.shape[-1])])
                if end - offset < chunk:
                    filler = ttnn.slice(
                        self._zero_rows, [0, 0, 0, 0], [1, chunk - (end - offset), 1, int(tap.shape[-1])]
                    )
                    piece = ttnn.concat([piece, filler], dim=1)
                part[layer] = piece
            self.write_context(part, start + offset)

    def main_x(self, taps: Mapping[int, ttnn.Tensor]) -> ttnn.Tensor:
        """main_norm(main_proj(concat of the tapped layers' [1, M, 1, H] means)) -> [1, 1, M, H] replicated."""

        hidden = ttnn.concat([taps[layer] for layer in self.target_layers], dim=-1)  # [1, M, 1, 3H]
        m = int(hidden.shape[1])
        hidden = ttnn.reshape(hidden, [1, 1, m, int(hidden.shape[-1])])
        partial = ttnn.matmul(
            hidden, self.main_proj, dtype=ttnn.bfloat16, compute_kernel_config=self.layers[0].compute_kernel_config
        )
        full = ttnn.all_gather(
            partial,
            dim=3,
            num_links=CCL_NUM_LINKS,
            topology=ttnn.Topology.Ring,
            cluster_axis=TP_AXIS,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return ttnn.rms_norm(full, weight=self.main_norm, epsilon=self.model.config.rms_norm_eps)

    def write_context(self, taps: Mapping[int, ttnn.Tensor], start: int | ttnn.Tensor) -> None:
        """The tapped rows of a main-model pass that started at ``start`` become context positions start .. ."""

        start_dev = self._position(start) if isinstance(start, int) else start
        main_x = self.main_x(taps)
        for layer in self.layers:
            layer.write_context(main_x, start_dev)

    # ------------------------------------------------------------------------------------------------ trace
    def capture_trace(self) -> None:
        """Capture the draft block like the verify trace: one segment per MoE boundary (embed + route 0 | finish
        i + route i + 1 | finish 2 + head + LM head); the host fills the expert buffers between segments. Every
        program must already be compiled (warm_compile)."""

        k, n = self.block, len(self.layers)
        mapper = ttnn.ReplicateTensorToMesh(self.mesh_device)
        token_ids = ttnn.from_torch(
            torch.tensor([0] + [self.noise_token] * (k - 1), dtype=torch.int32).reshape(1, 1, 1, k),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            mesh_mapper=mapper,
        )
        start = ttnn.from_torch(
            torch.tensor([1], dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            mesh_mapper=mapper,
        )
        input_ids = ttnn.reshape(token_ids, [1, k])

        def body(index, carry):
            hidden = (
                self.model.embed_tokens(token_ids)
                if index == 0
                else self.layers[index - 1].verify_finish(carry, self.buffers)
            )
            if index == n:
                head = self.model._hyper_head(hidden, self.head_params)
                head = ttnn.rms_norm(head, weight=self.final_norm, epsilon=self.model.config.rms_norm_eps)
                return self.model.lm_head.forward(head)
            return self.layers[index].verify_route(hidden, input_ids, start)

        carry = None
        for index in range(n + 1):  # warm: compiles every program on these persistent inputs
            carry = body(index, carry)
            if index < n:
                self._fill(index, carry)
        ttnn.synchronize_device(self.mesh_device)
        segments, routes, carry = [], [], None
        try:
            for index in range(n + 1):
                trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
                try:
                    carry = body(index, carry)
                finally:
                    ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
                segments.append(trace_id)
                if index < n:
                    routes.append(carry)
        except BaseException:
            for trace_id in segments:
                ttnn.release_trace(self.mesh_device, trace_id)
            raise
        self._trace = {"segments": segments, "routes": routes, "token_ids": token_ids, "start": start, "logits": carry}

    def release_trace(self) -> None:
        trace = getattr(self, "_trace", None)
        self._trace = None
        if trace is not None:
            for trace_id in trace["segments"]:
                ttnn.release_trace(self.mesh_device, trace_id)

    def _fill(self, index: int, route) -> None:
        picks = ttnn.to_torch((ttnn.get_device_tensors(route.selected) or (route.selected,))[0])
        self.layers[index].moe.fill_verify_buffers(picks.reshape(self.block, -1).to(torch.int64), self.buffers)

    def _trace_logits(self, anchor: int, start: int) -> ttnn.Tensor:
        trace, k = self._trace, self.block
        mapper = ttnn.ReplicateTensorToMesh(self.mesh_device)
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(
                torch.tensor([anchor] + [self.noise_token] * (k - 1), dtype=torch.int32).reshape(1, 1, 1, k),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=mapper,
            ),
            trace["token_ids"],
        )
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(
                torch.tensor([start], dtype=torch.int32),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=mapper,
            ),
            trace["start"],
        )
        for index, trace_id in enumerate(trace["segments"]):
            ttnn.execute_trace(self.mesh_device, trace_id, cq_id=0, blocking=False)
            if index < len(trace["routes"]):
                self._fill(index, trace["routes"][index])
        return trace["logits"]

    # ------------------------------------------------------------------------------------------------ draft
    def block_logits(self, anchor: int, start: int) -> ttnn.Tensor:
        """Eager draft block at position ``start`` (the anchor's position): vocab-sharded logits [1, 1, K, V/4]."""

        if getattr(self, "_trace", None) is not None:
            return self._trace_logits(anchor, start)
        k = self.block
        ids = self._tt(
            torch.tensor([anchor] + [self.noise_token] * (k - 1), dtype=torch.int32).reshape(1, 1, 1, k),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        start_dev = self._position(start)
        hidden = self.model.embed_tokens(ids)
        input_ids = ttnn.reshape(ids, [1, k])
        for layer in self.layers:
            route = layer.verify_route(hidden, input_ids, start_dev)
            picks = ttnn.to_torch((ttnn.get_device_tensors(route.selected) or (route.selected,))[0])
            layer.moe.fill_verify_buffers(picks.reshape(k, -1).to(torch.int64), self.buffers)
            hidden = layer.verify_finish(route, self.buffers)
        head = self.model._hyper_head(hidden, self.head_params)
        head = ttnn.rms_norm(head, weight=self.final_norm, epsilon=self.model.config.rms_norm_eps)
        return self.model.lm_head.forward(head)

    def markov_drafts(self, logits: ttnn.Tensor, anchor: int) -> list[int]:
        """Sequential Markov bias + greedy pick per row (eager): K drafts."""

        drafts, prev = [], anchor
        local = self.model.vocab_size // TP_DEGREE
        for i in range(self.block):
            row = ttnn.slice(logits, [0, 0, i, 0], [1, 1, i + 1, local])
            emb = ttnn.embedding(self._int([prev], dtype=ttnn.uint32), self.markov_w1, layout=ttnn.TILE_LAYOUT)
            bias = ttnn.matmul(
                ttnn.reshape(emb, [1, 1, 1, int(emb.shape[-1])]),
                self.markov_w2,
                dtype=ttnn.bfloat16,
                compute_kernel_config=self.layers[0].compute_kernel_config,
            )
            prev = self.model.verify_greedy_tokens(ttnn.add(row, bias), 1)[0]
            drafts.append(prev)
        return drafts

    def draft(self, anchor: int, start: int) -> list[int]:
        timing = getattr(self, "timing", None)
        if timing is None:
            return self.markov_drafts(self.block_logits(anchor, start), anchor)
        import time

        t0 = time.perf_counter()
        logits = self.block_logits(anchor, start)
        ttnn.synchronize_device(self.mesh_device)
        t1 = time.perf_counter()
        drafts = self.markov_drafts(logits, anchor)
        t2 = time.perf_counter()
        timing["block_s"] = timing.get("block_s", 0.0) + t1 - t0
        timing["markov_s"] = timing.get("markov_s", 0.0) + t2 - t1
        timing["drafts"] = timing.get("drafts", 0) + 1
        return drafts
