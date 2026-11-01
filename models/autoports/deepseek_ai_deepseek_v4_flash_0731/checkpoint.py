"""Lazy source-checkpoint loading and exact FP8/FP4 dequantization.

The 0731 repository stores attention/dense matrices as FP8 E4M3 with E8M0
128x128 block scales.  Routed experts are packed FP4 E2M1 (two nibbles per byte)
with a scale per 32 logical input elements.  Conversion happens outside every
runtime pass; the functional decoder never calls torch or transfers weights back.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import torch
from safetensors import safe_open

from .config import DeepSeekV4FlashConfig, required_checkpoint_shapes

FP4_E2M1_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def dequantize_block_fp8(weight: torch.Tensor, scale: torch.Tensor, block: int = 128) -> torch.Tensor:
    """Dequantize `[O,K]` E4M3 data with `[ceil(O/B),ceil(K/B)]` E8M0 scales."""

    if weight.ndim != 2 or scale.ndim != 2:
        raise ValueError(f"FP8 weight/scale must both be matrices, got {weight.shape} and {scale.shape}")
    out, inp = weight.shape
    expected = ((out + block - 1) // block, (inp + block - 1) // block)
    if tuple(scale.shape) != expected:
        raise ValueError(f"FP8 scale shape {tuple(scale.shape)} != expected {expected} for {tuple(weight.shape)}")
    expanded = scale.float().repeat_interleave(block, 0).repeat_interleave(block, 1)[:out, :inp]
    return weight.float() * expanded


def unpack_fp4_e2m1(packed: torch.Tensor) -> torch.Tensor:
    """Unpack `[O,K/2]` bytes to `[O,K]` in source nibble order."""

    if packed.ndim != 2:
        raise ValueError(f"packed FP4 weight must be a matrix, got {packed.shape}")
    raw = packed.view(torch.uint8)
    low = raw & 0x0F
    high = raw >> 4
    values = torch.stack((FP4_E2M1_TABLE[low.long()], FP4_E2M1_TABLE[high.long()]), dim=-1)
    return values.flatten(-2)


def dequantize_block_fp4(packed: torch.Tensor, scale: torch.Tensor, block: int = 32) -> torch.Tensor:
    """Dequantize packed E2M1 with one E8M0 scale per logical K block."""

    values = unpack_fp4_e2m1(packed)
    out, inp = values.shape
    expected = (out, (inp + block - 1) // block)
    if tuple(scale.shape) != expected:
        raise ValueError(f"FP4 scale shape {tuple(scale.shape)} != expected {expected} for logical {(out, inp)}")
    expanded = scale.float().repeat_interleave(block, 1)[:, :inp]
    return values * expanded


@dataclass(frozen=True)
class CheckpointIndex:
    root: Path
    weight_map: Mapping[str, str]

    @classmethod
    def open(cls, root: str | Path) -> "CheckpointIndex":
        root = Path(root)
        index_path = root / "model.safetensors.index.json"
        if not index_path.is_file():
            raise FileNotFoundError(index_path)
        with index_path.open() as f:
            raw = json.load(f)
        return cls(root=root, weight_map=raw["weight_map"])

    def keys_for_layer(self, layer_idx: int) -> Iterator[str]:
        prefix = f"layers.{layer_idx}."
        return (key for key in self.weight_map if key.startswith(prefix))

    def read(self, key: str) -> torch.Tensor:
        try:
            shard = self.weight_map[key]
        except KeyError as exc:
            raise KeyError(f"checkpoint has no tensor {key!r}") from exc
        with safe_open(self.root / shard, framework="pt", device="cpu") as f:
            return f.get_tensor(key)

    def read_many(self, keys: Sequence[str]) -> dict[str, torch.Tensor]:
        """Read selected tensors while opening each safetensors shard only once."""

        by_shard: dict[str, list[str]] = {}
        for key in keys:
            try:
                shard = self.weight_map[key]
            except KeyError as exc:
                raise KeyError(f"checkpoint has no tensor {key!r}") from exc
            by_shard.setdefault(shard, []).append(key)
        result: dict[str, torch.Tensor] = {}
        for shard, shard_keys in by_shard.items():
            with safe_open(self.root / shard, framework="pt", device="cpu") as f:
                result.update({key: f.get_tensor(key) for key in shard_keys})
        return result


class LayerCheckpoint:
    """Lazy logical tensor view of one official source-format layer."""

    def __init__(self, root: str | Path, layer_idx: int, config: DeepSeekV4FlashConfig | None = None):
        self.config = DeepSeekV4FlashConfig() if config is None else config
        self.config.validate()
        self.layer_idx = layer_idx
        self.index = CheckpointIndex.open(root)
        self.expected = required_checkpoint_shapes(self.config, layer_idx)

    def _scale_key(self, key: str) -> str:
        return key.removesuffix(".weight") + ".scale"

    def _dequantize(
        self,
        key: str,
        raw: torch.Tensor,
        scale: torch.Tensor | None,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if scale is not None:
            expected = self.expected.get(key)
            if expected is not None and tuple(raw.shape) != expected:
                if raw.ndim == 2 and raw.shape[0] == expected[0] and raw.shape[1] * 2 == expected[1]:
                    raw = dequantize_block_fp4(raw, scale)
                else:
                    raise ValueError(
                        f"stored shape {tuple(raw.shape)} for {key} cannot represent logical shape {expected}"
                    )
            else:
                raw = dequantize_block_fp8(raw, scale)
        logical = self.expected.get(key)
        if logical is not None and tuple(raw.shape) != logical:
            raise ValueError(f"logical shape {tuple(raw.shape)} for {key} != required {logical}")
        return raw.to(dtype=dtype)

    def tensor(self, key: str, *, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        """Read one tensor, dequantizing it when the adjacent scale tensor exists."""

        scale_key = self._scale_key(key)
        raw = self.index.read(key)
        scale = self.index.read(scale_key) if key.endswith(".weight") and scale_key in self.index.weight_map else None
        return self._dequantize(key, raw, scale, dtype=dtype)

    def expert(self, expert_idx: int, projection: str, *, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        if not 0 <= expert_idx < self.config.num_experts:
            raise IndexError(expert_idx)
        if projection not in ("w1", "w2", "w3"):
            raise ValueError(f"unknown expert projection {projection!r}")
        key = f"layers.{self.layer_idx}.ffn.experts.{expert_idx}.{projection}.weight"
        raw = self.index.read(key)
        scale = self.index.read(self._scale_key(key))
        tensor = dequantize_block_fp4(raw, scale)
        expected = (
            (self.config.hidden_size, self.config.moe_intermediate_size)
            if projection == "w2"
            else (self.config.moe_intermediate_size, self.config.hidden_size)
        )
        if tuple(tensor.shape) != expected:
            raise ValueError(f"logical shape {tuple(tensor.shape)} for {key} != required {expected}")
        return tensor.to(dtype=dtype)

    def validate_manifest(self) -> None:
        missing = sorted(set(self.expected) - set(self.index.weight_map))
        missing_experts = []
        p = f"layers.{self.layer_idx}.ffn.experts"
        for expert in range(self.config.num_experts):
            for proj in ("w1", "w2", "w3"):
                key = f"{p}.{expert}.{proj}.weight"
                if key not in self.index.weight_map or self._scale_key(key) not in self.index.weight_map:
                    missing_experts.append(key)
        if missing or missing_experts:
            raise ValueError(
                f"incomplete layer {self.layer_idx}: missing non-expert={missing}, "
                f"missing expert tensors={missing_experts[:8]}{'...' if len(missing_experts) > 8 else ''}"
            )

    def non_expert_state_dict(self, *, dtype: torch.dtype = torch.bfloat16) -> dict[str, torch.Tensor]:
        """Materialize the validated non-expert layer tensors only."""

        self.validate_manifest()
        return {key: self.tensor(key, dtype=dtype) for key in self.expected}

    def materialize(
        self, *, dtype: torch.dtype = torch.bfloat16, include_experts: bool = True
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Materialize a complete layer with one file-open per source shard.

        Returns the source-format non-expert mapping followed by stacked routed
        expert ``w1``, ``w2``, and ``w3`` tensors (``None`` when
        ``include_experts`` is false, e.g. when an expert cache supplies them).
        """

        self.validate_manifest()
        expert_keys = {
            projection: [
                f"layers.{self.layer_idx}.ffn.experts.{expert}.{projection}.weight"
                for expert in range(self.config.num_experts)
            ]
            for projection in ("w1", "w2", "w3")
        }
        if not include_experts:
            expert_keys = {projection: [] for projection in expert_keys}
        value_keys = list(self.expected)
        for keys in expert_keys.values():
            value_keys.extend(keys)
        scale_keys = [
            self._scale_key(key)
            for key in value_keys
            if key.endswith(".weight") and self._scale_key(key) in self.index.weight_map
        ]
        raw = self.index.read_many(value_keys + scale_keys)

        non_expert = {
            key: self._dequantize(key, raw[key], raw.get(self._scale_key(key)), dtype=dtype) for key in self.expected
        }
        experts: dict[str, torch.Tensor] = {}
        for projection, keys in expert_keys.items():
            tensors = [dequantize_block_fp4(raw[key], raw[self._scale_key(key)]).to(dtype=dtype) for key in keys]
            expected = (
                (self.config.hidden_size, self.config.moe_intermediate_size)
                if projection == "w2"
                else (self.config.moe_intermediate_size, self.config.hidden_size)
            )
            if any(tuple(tensor.shape) != expected for tensor in tensors):
                raise ValueError(f"one or more {projection} experts do not have required shape {expected}")
            experts[projection] = torch.stack(tensors) if tensors else None
        return non_expert, experts["w1"], experts["w2"], experts["w3"]
