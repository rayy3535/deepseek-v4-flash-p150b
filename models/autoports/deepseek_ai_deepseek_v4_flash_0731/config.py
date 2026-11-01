"""Exact model and paged-cache geometry for DeepSeek-V4-Flash-0731.

This module deliberately has no torch or TTNN dependency.  It is the common source of
truth used by checkpoint conversion, the device implementation, tests, and capacity
evidence.  Reduced toy dimensions are rejected: functional tests may shorten a prompt,
but they always exercise the checkpoint's physical widths.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence


class AttentionKind(str, Enum):
    SLIDING = "sliding_window"
    CSA = "compressed_sparse_attention"
    HCA = "heavily_compressed_attention"


class RouterKind(str, Enum):
    HASH = "hash_moe"
    LEARNED = "learned_moe"


@dataclass(frozen=True)
class LayerKind:
    attention: AttentionKind
    router: RouterKind
    compress_ratio: int


@dataclass(frozen=True)
class CacheGeometry:
    """Physical cache allocation for one decoder layer and one user."""

    page_size: int
    max_context: int
    local_pages: int
    compressed_pages: int
    compressor_state_pages: int
    indexer_pages: int

    @property
    def local_rows(self) -> int:
        return self.local_pages * self.page_size

    @property
    def compressed_rows(self) -> int:
        return self.compressed_pages * self.page_size

    @property
    def indexer_rows(self) -> int:
        return self.indexer_pages * self.page_size


@dataclass(frozen=True)
class DeepSeekV4FlashConfig:
    """The published 0731 checkpoint geometry.

    ``from_json`` validates every shape-affecting field that the upstream inference
    implementation consumes.  This makes accidentally running a toy decoder or a V4-Pro
    config fail at construction rather than pass a misleading PCC test.
    """

    model_name: str = "deepseek-ai/DeepSeek-V4-Flash-0731"
    architectures: tuple[str, ...] = ("DeepseekV4ForCausalLM",)
    hidden_size: int = 4096
    num_hidden_layers: int = 43
    num_attention_heads: int = 64
    num_key_value_heads: int = 1
    head_dim: int = 512
    rope_head_dim: int = 64
    q_lora_rank: int = 1024
    o_lora_rank: int = 1024
    o_groups: int = 8
    sliding_window: int = 128
    max_position_embeddings: int = 1_048_576
    original_max_position_embeddings: int = 65_536
    rope_theta: float = 10_000.0
    compress_rope_theta: float = 160_000.0
    rope_factor: float = 16.0
    beta_fast: float = 32.0
    beta_slow: float = 1.0
    rms_norm_eps: float = 1e-6
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    num_experts: int = 256
    num_experts_per_tok: int = 6
    num_shared_experts: int = 1
    moe_intermediate_size: int = 2048
    routed_scaling_factor: float = 1.5
    swiglu_limit: float = 10.0
    scoring_func: str = "sqrtsoftplus"
    n_hash_layers: int = 3
    vocab_size: int = 129_280
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 512
    page_size: int = 32

    # The checkpoint schedule is 0,0,4 then alternating 128,4, ending in 4.
    compress_ratios: tuple[int, ...] = (
        0,
        0,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
    )

    @property
    def nope_head_dim(self) -> int:
        return self.head_dim - self.rope_head_dim

    @property
    def hc_width(self) -> int:
        return self.hc_mult * self.hidden_size

    @property
    def hc_mix_width(self) -> int:
        return (2 + self.hc_mult) * self.hc_mult

    @property
    def attention_width(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def grouped_attention_width(self) -> int:
        return self.attention_width // self.o_groups

    def layer_kind(self, layer_idx: int) -> LayerKind:
        if not 0 <= layer_idx < self.num_hidden_layers:
            raise IndexError(f"layer_idx {layer_idx} outside [0, {self.num_hidden_layers})")
        ratio = self.compress_ratios[layer_idx]
        attention = AttentionKind.SLIDING if ratio == 0 else AttentionKind.CSA if ratio == 4 else AttentionKind.HCA
        router = RouterKind.HASH if layer_idx < self.n_hash_layers else RouterKind.LEARNED
        return LayerKind(attention=attention, router=router, compress_ratio=ratio)

    def cache_geometry(self, layer_idx: int, max_context: int | None = None) -> CacheGeometry:
        max_context = self.max_position_embeddings if max_context is None else max_context
        if not 1 <= max_context <= self.max_position_embeddings:
            raise ValueError(f"max_context must be in [1, {self.max_position_embeddings}], got {max_context}")
        ratio = self.layer_kind(layer_idx).compress_ratio
        compressed = math.ceil(max_context / ratio) if ratio else 0
        indexer = compressed if ratio == 4 else 0
        return CacheGeometry(
            page_size=self.page_size,
            max_context=max_context,
            # The baseline keeps the source KV and compressor projections in
            # non-aliasing paged storage.  This is larger than a custom
            # sub-page ring, but makes non-page-aligned prefill tails exact with
            # stock paged_fill_cache/update_cache operations.
            local_pages=math.ceil(max_context / self.page_size),
            compressed_pages=math.ceil(compressed / self.page_size),
            compressor_state_pages=math.ceil(max_context / self.page_size) if ratio else 0,
            indexer_pages=math.ceil(indexer / self.page_size),
        )

    def validate(self) -> None:
        expected = DeepSeekV4FlashConfig()
        fields = (
            "architectures",
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "rope_head_dim",
            "q_lora_rank",
            "o_lora_rank",
            "o_groups",
            "sliding_window",
            "max_position_embeddings",
            "original_max_position_embeddings",
            "rope_theta",
            "compress_rope_theta",
            "rope_factor",
            "beta_fast",
            "beta_slow",
            "rms_norm_eps",
            "hc_mult",
            "hc_sinkhorn_iters",
            "hc_eps",
            "num_experts",
            "num_experts_per_tok",
            "num_shared_experts",
            "moe_intermediate_size",
            "routed_scaling_factor",
            "swiglu_limit",
            "scoring_func",
            "n_hash_layers",
            "vocab_size",
            "index_n_heads",
            "index_head_dim",
            "index_topk",
            "compress_ratios",
        )
        mismatches = [
            f"{name}={getattr(self, name)!r} (expected {getattr(expected, name)!r})"
            for name in fields
            if getattr(self, name) != getattr(expected, name)
        ]
        if mismatches:
            raise ValueError("not the DeepSeek-V4-Flash-0731 physical config: " + "; ".join(mismatches))
        if self.nope_head_dim != 448:
            raise ValueError(f"invalid nope head width {self.nope_head_dim}, expected 448")
        if self.compress_ratios != expected.compress_ratios:
            raise ValueError("compression schedule mismatch")

    @classmethod
    def from_json(cls, path: str | Path) -> "DeepSeekV4FlashConfig":
        with Path(path).open() as f:
            raw = json.load(f)

        # The official inference config uses these names.  The local HF-compatible
        # config uses the names on the right in the fallback alternatives.
        def value(*names: str, default: Any = None) -> Any:
            for name in names:
                if name in raw:
                    return raw[name]
            return default

        cfg = cls(
            architectures=tuple(value("architectures", default=("DeepseekV4ForCausalLM",))),
            hidden_size=value("dim", "hidden_size"),
            num_hidden_layers=value("n_layers", "num_hidden_layers"),
            num_attention_heads=value("n_heads", "num_attention_heads"),
            num_key_value_heads=value("num_key_value_heads", default=1),
            head_dim=value("head_dim"),
            rope_head_dim=value("rope_head_dim", "qk_rope_head_dim"),
            q_lora_rank=value("q_lora_rank"),
            o_lora_rank=value("o_lora_rank"),
            o_groups=value("o_groups"),
            sliding_window=value("window_size", "sliding_window"),
            max_position_embeddings=value("max_seq_len", "max_position_embeddings"),
            original_max_position_embeddings=value(
                "original_seq_len",
                "original_max_position_embeddings",
                default=value("rope_scaling", default={}).get("original_max_position_embeddings", 65_536),
            ),
            rope_theta=value("rope_theta", default=10_000.0),
            compress_rope_theta=value("compress_rope_theta", default=160_000.0),
            rope_factor=value("rope_factor", default=value("rope_scaling", default={}).get("factor", 16.0)),
            beta_fast=value("beta_fast", default=value("rope_scaling", default={}).get("beta_fast", 32.0)),
            beta_slow=value("beta_slow", default=value("rope_scaling", default={}).get("beta_slow", 1.0)),
            rms_norm_eps=value("norm_eps", "rms_norm_eps"),
            hc_mult=value("hc_mult"),
            hc_sinkhorn_iters=value("hc_sinkhorn_iters"),
            hc_eps=value("hc_eps"),
            num_experts=value("n_routed_experts", "num_local_experts"),
            num_experts_per_tok=value("n_activated_experts", "num_experts_per_tok"),
            num_shared_experts=value("n_shared_experts", "num_shared_experts"),
            moe_intermediate_size=value("moe_inter_dim", "moe_intermediate_size", "intermediate_size"),
            routed_scaling_factor=value("route_scale", "routed_scaling_factor"),
            swiglu_limit=value("swiglu_limit"),
            scoring_func=value("scoring_func"),
            n_hash_layers=value("n_hash_layers", "num_hash_layers", default=3),
            vocab_size=value("vocab_size"),
            index_n_heads=value("index_n_heads", default=64),
            index_head_dim=value("index_head_dim", default=128),
            index_topk=value("index_topk", default=512),
            # The JSON also carries three DSpARK/MTP slots after the 43 backbone
            # layers.  A functional decoder represents the backbone only.
            compress_ratios=tuple(value("compress_ratios"))[: value("n_layers", "num_hidden_layers")],
        )
        cfg.validate()
        return cfg

    @classmethod
    def from_hf_config(cls, hf_config: object) -> "DeepSeekV4FlashConfig":
        """Validate an HF-compatible config and return the fixed physical contract.

        The generated decoder intentionally supports one checkpoint, not a family
        of smaller lookalikes.  Returning the canonical dataclass after validating
        the external config keeps all downstream cache and weight geometry strict.
        """

        if isinstance(hf_config, cls):
            hf_config.validate()
            return hf_config
        expected = cls()

        def read(*names: str, default: Any = None) -> Any:
            for name in names:
                if isinstance(hf_config, Mapping) and name in hf_config:
                    return hf_config[name]
                if hasattr(hf_config, name):
                    return getattr(hf_config, name)
            return default

        rope_scaling = read("rope_scaling", default={}) or {}
        compress_rope_scaling = rope_scaling.get("compress", rope_scaling)
        architectures = read("architectures", default=expected.architectures)
        if architectures is None:
            architectures = expected.architectures

        checks = {
            "architectures": tuple(architectures),
            "hidden_size": read("hidden_size", "dim"),
            "num_hidden_layers": read("num_hidden_layers", "n_layers"),
            "num_attention_heads": read("num_attention_heads", "n_heads"),
            "num_key_value_heads": read("num_key_value_heads", default=1),
            "head_dim": read("head_dim"),
            "q_lora_rank": read("q_lora_rank"),
            "o_lora_rank": read("o_lora_rank"),
            "o_groups": read("o_groups"),
            "sliding_window": read("sliding_window", "window_size"),
            "max_position_embeddings": read("max_position_embeddings", "max_seq_len"),
            "original_max_position_embeddings": read(
                "original_max_position_embeddings",
                "original_seq_len",
                default=compress_rope_scaling.get("original_max_position_embeddings"),
            ),
            "rope_theta": read("rope_theta"),
            "compress_rope_theta": read("compress_rope_theta", default=expected.compress_rope_theta),
            "rope_factor": read(
                "rope_factor",
                default=compress_rope_scaling.get("factor"),
            ),
            "beta_fast": read(
                "beta_fast",
                default=compress_rope_scaling.get("beta_fast"),
            ),
            "beta_slow": read(
                "beta_slow",
                default=compress_rope_scaling.get("beta_slow"),
            ),
            "rms_norm_eps": read("rms_norm_eps", "norm_eps"),
            "hc_mult": read("hc_mult"),
            "hc_sinkhorn_iters": read("hc_sinkhorn_iters"),
            "hc_eps": read("hc_eps", default=expected.hc_eps),
            "num_experts": read("n_routed_experts", "num_local_experts"),
            "num_experts_per_tok": read("num_experts_per_tok", "n_activated_experts"),
            "num_shared_experts": read("n_shared_experts", "num_shared_experts"),
            "moe_intermediate_size": read("moe_intermediate_size", "moe_inter_dim", "intermediate_size"),
            "routed_scaling_factor": read("routed_scaling_factor", "route_scale"),
            "swiglu_limit": read("swiglu_limit"),
            "scoring_func": read("scoring_func"),
            "vocab_size": read("vocab_size"),
            "index_n_heads": read("index_n_heads", default=64),
            "index_head_dim": read("index_head_dim", default=128),
            "index_topk": read("index_topk", default=512),
        }
        rope_head_dim = read("rope_head_dim", "qk_rope_head_dim")
        if rope_head_dim is None:
            factor = read("partial_rotary_factor")
            rope_head_dim = round(expected.head_dim * factor) if factor is not None else None
        checks["rope_head_dim"] = rope_head_dim
        mismatches = [
            f"{name}={actual!r} (expected {getattr(expected, name)!r})"
            for name, actual in checks.items()
            if actual != getattr(expected, name)
        ]

        layer_types = read("layer_types")
        if layer_types is not None:
            ratio_by_type = {
                "sliding_attention": 0,
                "sliding_window": 0,
                "compressed_sparse_attention": 4,
                "heavily_compressed_attention": 128,
            }
            try:
                ratios = tuple(ratio_by_type[name] for name in layer_types)
            except KeyError as exc:
                raise ValueError(f"unknown DeepSeek-V4 layer type {exc.args[0]!r}") from exc
            if ratios != expected.compress_ratios:
                mismatches.append("layer_types do not match the 43-layer 0731 compression schedule")
        else:
            ratios = read("compress_ratios")
            if ratios is not None and tuple(ratios)[: expected.num_hidden_layers] != expected.compress_ratios:
                mismatches.append("compress_ratios do not match the 43-layer 0731 schedule")

        mlp_types = read("mlp_layer_types")
        if mlp_types is not None:
            expected_mlp = ("hash_moe",) * expected.n_hash_layers + ("moe",) * (
                expected.num_hidden_layers - expected.n_hash_layers
            )
            if tuple(mlp_types) != expected_mlp:
                mismatches.append("mlp_layer_types do not match the 3 hash + 40 learned schedule")
        if mismatches:
            raise ValueError("not the DeepSeek-V4-Flash-0731 HF config: " + "; ".join(mismatches))
        return expected


def make_page_table(
    logical_pages: int,
    physical_pages: int | None = None,
    permutation: Sequence[int] | None = None,
    *,
    cyclic: bool = False,
) -> tuple[int, ...]:
    """Build and validate a logical-page to physical-page mapping.

    ``cyclic=True`` is used only for the bounded 128-token local cache.  Compressed
    pages are one-to-one so a full-context decode never aliases old entries.
    """

    if logical_pages < 0:
        raise ValueError("logical_pages must be non-negative")
    physical_pages = logical_pages if physical_pages is None else physical_pages
    if physical_pages <= 0 and logical_pages:
        raise ValueError("physical_pages must be positive")
    if permutation is None:
        permutation = tuple(range(physical_pages))
    else:
        permutation = tuple(int(x) for x in permutation)
    if sorted(permutation) != list(range(physical_pages)):
        raise ValueError(f"permutation must contain each physical page exactly once: {permutation}")
    if not cyclic and logical_pages > physical_pages:
        raise ValueError("non-cyclic page table cannot alias logical pages")
    return tuple(permutation[i % physical_pages] for i in range(logical_pages))


def physical_row_lut(page_table: Sequence[int], page_size: int) -> tuple[int, ...]:
    """Expand a page table into the row-address LUT consumed by device gathers."""

    if page_size <= 0:
        raise ValueError("page_size must be positive")
    return tuple(physical * page_size + offset for physical in page_table for offset in range(page_size))


def advertised_prefill_input_bytes(config: DeepSeekV4FlashConfig, element_bytes: int = 2) -> int:
    """Bytes for the decoder input alone at advertised context (before any weights/scratch)."""

    return config.max_position_embeddings * config.hc_mult * config.hidden_size * element_bytes


def required_checkpoint_shapes(config: DeepSeekV4FlashConfig, layer_idx: int) -> Mapping[str, tuple[int, ...]]:
    """Logical (dequantized) shapes for all non-expert tensors in one source layer."""

    p = f"layers.{layer_idx}"
    shapes: dict[str, tuple[int, ...]] = {
        f"{p}.hc_attn_base": (config.hc_mix_width,),
        f"{p}.hc_attn_fn": (config.hc_mix_width, config.hc_width),
        f"{p}.hc_attn_scale": (3,),
        f"{p}.hc_ffn_base": (config.hc_mix_width,),
        f"{p}.hc_ffn_fn": (config.hc_mix_width, config.hc_width),
        f"{p}.hc_ffn_scale": (3,),
        f"{p}.attn.attn_sink": (config.num_attention_heads,),
        f"{p}.attn.wq_a.weight": (config.q_lora_rank, config.hidden_size),
        f"{p}.attn.wq_b.weight": (config.attention_width, config.q_lora_rank),
        f"{p}.attn.q_norm.weight": (config.q_lora_rank,),
        f"{p}.attn.wkv.weight": (config.head_dim, config.hidden_size),
        f"{p}.attn.kv_norm.weight": (config.head_dim,),
        f"{p}.attn.wo_a.weight": (config.o_groups * config.o_lora_rank, config.grouped_attention_width),
        f"{p}.attn.wo_b.weight": (config.hidden_size, config.o_groups * config.o_lora_rank),
        f"{p}.attn_norm.weight": (config.hidden_size,),
        f"{p}.ffn_norm.weight": (config.hidden_size,),
        f"{p}.ffn.gate.weight": (config.num_experts, config.hidden_size),
        f"{p}.ffn.shared_experts.w1.weight": (config.moe_intermediate_size, config.hidden_size),
        f"{p}.ffn.shared_experts.w3.weight": (config.moe_intermediate_size, config.hidden_size),
        f"{p}.ffn.shared_experts.w2.weight": (config.hidden_size, config.moe_intermediate_size),
    }
    kind = config.layer_kind(layer_idx)
    if kind.router is RouterKind.HASH:
        shapes[f"{p}.ffn.gate.tid2eid"] = (config.vocab_size, config.num_experts_per_tok)
    else:
        shapes[f"{p}.ffn.gate.bias"] = (config.num_experts,)
    if kind.compress_ratio:
        coff = 2 if kind.compress_ratio == 4 else 1
        shapes.update(
            {
                f"{p}.attn.compressor.ape": (kind.compress_ratio, coff * config.head_dim),
                f"{p}.attn.compressor.wkv.weight": (coff * config.head_dim, config.hidden_size),
                f"{p}.attn.compressor.wgate.weight": (coff * config.head_dim, config.hidden_size),
                f"{p}.attn.compressor.norm.weight": (config.head_dim,),
            }
        )
    if kind.attention is AttentionKind.CSA:
        shapes.update(
            {
                f"{p}.attn.indexer.wq_b.weight": (
                    config.index_n_heads * config.index_head_dim,
                    config.q_lora_rank,
                ),
                f"{p}.attn.indexer.weights_proj.weight": (config.index_n_heads, config.hidden_size),
                f"{p}.attn.indexer.compressor.ape": (
                    kind.compress_ratio,
                    2 * config.index_head_dim,
                ),
                f"{p}.attn.indexer.compressor.wkv.weight": (2 * config.index_head_dim, config.hidden_size),
                f"{p}.attn.indexer.compressor.wgate.weight": (2 * config.index_head_dim, config.hidden_size),
                f"{p}.attn.indexer.compressor.norm.weight": (config.index_head_dim,),
            }
        )
    return shapes
