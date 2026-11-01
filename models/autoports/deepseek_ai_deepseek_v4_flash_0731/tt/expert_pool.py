"""Host-paged routed experts: a fixed-address device pool plus an expert->slot table.

Each MoE layer keeps ``pool_size`` routed experts resident on device in two pool tensors shaped like the
resident weights they replace (``[1, P + 1, K, N]``, BFLOAT4_B, TILE, DRAM interleaved).  The last slot is a
null slot whose sparsity entry is zero.  A replicated bf16 table maps every expert to its slot; the MoE turns
router picks into slot indices with ``ttnn.embedding`` (the same lookup the hash router uses), so a captured
trace only ever sees fixed buffers.  The host owns the LRU state and fills slots between trace replays.

All experts live on the host as per-expert BFLOAT4_B tensors already split over the tensor-parallel ranks,
packed exactly like the resident weights, so a slot write never repacks on the device.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from pathlib import Path

import torch

import ttnn

NULL_SLOT_SPARSITY = 0.0
# Bump when pack_expert or the host tensor format changes; old caches are then rebuilt.
EXPERT_CACHE_FORMAT = "routed-expert-bfp4-tp-packed-v1"
_MANIFEST = "manifest.json"


def expert_cache_key(checkpoint_root: str | Path, layer_idx: int, *, tp_degree: int, dtype: ttnn.DataType) -> str:
    """Identity of one layer's packed experts: checkpoint index, layer, packing format, TP degree, dtype."""

    index = Path(checkpoint_root) / "model.safetensors.index.json"
    digest = hashlib.sha256(index.read_bytes())
    digest.update(f"|layer={layer_idx}|{EXPERT_CACHE_FORMAT}|tp={tp_degree}|dtype={dtype}".encode())
    return digest.hexdigest()


def expert_cache_dir(
    cache_root: str | Path, checkpoint_root: str | Path, layer_idx: int, *, tp_degree: int, dtype: ttnn.DataType
) -> tuple[Path, str]:
    key = expert_cache_key(checkpoint_root, layer_idx, tp_degree=tp_degree, dtype=dtype)
    return Path(cache_root) / f"layer_{layer_idx:02d}", key


class ExpertHostStore:
    """Per-expert host tensors for one layer, sharded over the four TP ranks."""

    def __init__(
        self,
        gate_up: list[ttnn.Tensor],
        down: list[ttnn.Tensor],
        *,
        dtype: ttnn.DataType,
        torch_gate_up: list[torch.Tensor] | None = None,
        torch_down: list[torch.Tensor] | None = None,
    ) -> None:
        if len(gate_up) != len(down):
            raise ValueError("gate/up and down expert lists differ in length")
        self.num_experts = len(gate_up)
        self.dtype = dtype
        self.gate_up = gate_up
        self.down = down
        # Only the placeholder slot writer needs the source values; it is test-only.
        self.torch_gate_up = torch_gate_up
        self.torch_down = torch_down

    @classmethod
    def from_weights(
        cls,
        mesh_device: ttnn.MeshDevice,
        expert_w1: torch.Tensor,
        expert_w2: torch.Tensor,
        expert_w3: torch.Tensor,
        *,
        tp_degree: int,
        shard_mapper,
        dtype: ttnn.DataType = ttnn.bfloat4_b,
        keep_torch: bool = False,
    ) -> "ExpertHostStore":
        gate_up, down = [], []
        torch_gate_up: list[torch.Tensor] | None = [] if keep_torch else None
        torch_down: list[torch.Tensor] | None = [] if keep_torch else None
        for expert in range(expert_w1.shape[0]):
            gu, dn = pack_expert(expert_w1[expert], expert_w2[expert], expert_w3[expert], tp_degree)
            gate_up.append(_host_tensor(gu, shard_mapper(mesh_device, -1), dtype))
            down.append(_host_tensor(dn, shard_mapper(mesh_device, -2), dtype))
            if keep_torch:
                torch_gate_up.append(gu)
                torch_down.append(dn)
        return cls(gate_up, down, dtype=dtype, torch_gate_up=torch_gate_up, torch_down=torch_down)

    @staticmethod
    def cache_valid(layer_dir: Path, key: str, num_experts: int) -> bool:
        try:
            manifest = json.loads((layer_dir / _MANIFEST).read_text())
        except (OSError, ValueError):
            return False
        return manifest.get("key") == key and manifest.get("num_experts") == num_experts

    @classmethod
    def from_cache(cls, layer_dir: Path, *, dtype: ttnn.DataType, num_experts: int) -> "ExpertHostStore":
        gate_up = [ttnn.load_tensor(layer_dir / f"{e:03d}.gu.tensorbin") for e in range(num_experts)]
        down = [ttnn.load_tensor(layer_dir / f"{e:03d}.dn.tensorbin") for e in range(num_experts)]
        for tensor in (*gate_up, *down):
            if tensor.dtype != dtype:
                raise ValueError(f"cached expert dtype {tensor.dtype} != {dtype}")
        return cls(gate_up, down, dtype=dtype)

    def save(self, layer_dir: Path, key: str) -> None:
        """Write every expert, then the manifest; a directory without a matching manifest is never used."""

        layer_dir.mkdir(parents=True, exist_ok=True)
        (layer_dir / _MANIFEST).unlink(missing_ok=True)
        for expert in range(self.num_experts):
            ttnn.dump_tensor(layer_dir / f"{expert:03d}.gu.tensorbin", self.gate_up[expert])
            ttnn.dump_tensor(layer_dir / f"{expert:03d}.dn.tensorbin", self.down[expert])
        manifest = {
            "key": key,
            "num_experts": self.num_experts,
            "format": EXPERT_CACHE_FORMAT,
            "written": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        (layer_dir / _MANIFEST).write_text(json.dumps(manifest))


def pack_expert(
    w1: torch.Tensor, w2: torch.Tensor, w3: torch.Tensor, tp_degree: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """One expert in the resident layout: gate/up as adjacent per-rank blocks, down transposed."""

    gate_chunks = w1.transpose(-2, -1).chunk(tp_degree, dim=-1)
    up_chunks = w3.transpose(-2, -1).chunk(tp_degree, dim=-1)
    gate_up = torch.cat([torch.cat((g, u), dim=-1) for g, u in zip(gate_chunks, up_chunks, strict=True)], dim=-1)
    return gate_up.reshape(1, 1, *gate_up.shape).contiguous(), w2.transpose(-2, -1).reshape(1, 1, *w2.shape[::-1])


def _host_tensor(weight: torch.Tensor, mapper, dtype: ttnn.DataType) -> ttnn.Tensor:
    return ttnn.from_torch(weight.contiguous(), mesh_mapper=mapper, dtype=dtype, layout=ttnn.TILE_LAYOUT)


class ExpertPool:
    """Device pool, slot table and host LRU for one layer."""

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        store: ExpertHostStore,
        pool_size: int,
        *,
        gate_up_shape: Sequence[int],
        down_shape: Sequence[int],
        shard_mapper,
        initial_experts: Sequence[int] | None = None,
    ) -> None:
        if not 6 <= pool_size <= store.num_experts:
            raise ValueError(f"pool size {pool_size} outside [6, {store.num_experts}]")
        self.mesh_device = mesh_device
        self.store = store
        self.pool_size = pool_size
        self.null_slot = pool_size
        self._shard_mapper = shard_mapper
        self._gate_up_shape = tuple(gate_up_shape)
        self._down_shape = tuple(down_shape)
        slots = pool_size + 1
        self.gate_up = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, slots, *gate_up_shape]), store.dtype, ttnn.TILE_LAYOUT, mesh_device, ttnn.DRAM_MEMORY_CONFIG
        )
        self.down = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, slots, *down_shape]), store.dtype, ttnn.TILE_LAYOUT, mesh_device, ttnn.DRAM_MEMORY_CONFIG
        )
        mask = torch.ones(1, 1, 1, slots, dtype=torch.bfloat16)
        mask[..., self.null_slot] = NULL_SLOT_SPARSITY
        self.sparsity = ttnn.from_torch(
            mask,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.slot_of = [self.null_slot] * store.num_experts
        self.expert_in = [-1] * pool_size
        self._lru: OrderedDict[int, None] = OrderedDict()  # slots, least recent first
        self.table = ttnn.from_torch(
            self._table_torch(),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.loads = 0
        self.table_writes = 0
        # Hit-rate counters: ensure() calls and distinct experts asked for; hits = requested - loads.
        self.ensure_calls = 0
        self.requested = 0
        first = list(initial_experts) if initial_experts is not None else list(range(pool_size))
        if len(first) > pool_size or len(set(first)) != len(first):
            raise ValueError("initial experts must be unique and fit in the pool")
        self._place(first)

    def _table_torch(self) -> torch.Tensor:
        return torch.tensor(self.slot_of, dtype=torch.bfloat16).reshape(-1, 1)

    def _write_table(self) -> None:
        host = ttnn.from_torch(
            self._table_torch(),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        ttnn.copy_host_to_device_tensor(host, self.table)
        self.table_writes += 1

    def _place(self, experts: Sequence[int]) -> None:
        for slot, expert in enumerate(experts):
            self._assign(slot, expert)
            self._lru[slot] = None
        for slot in range(len(experts), self.pool_size):
            self._lru[slot] = None
            self._lru.move_to_end(slot, last=False)
        self._write_slots(list(range(len(experts))))
        self._write_table()

    def _assign(self, slot: int, expert: int) -> None:
        old = self.expert_in[slot]
        if old >= 0:
            self.slot_of[old] = self.null_slot
        self.expert_in[slot] = expert
        self.slot_of[expert] = slot

    def ensure(self, experts: Iterable[int]) -> int:
        """Make every listed expert resident; never evicts a listed expert. Returns the number loaded."""

        wanted = list(dict.fromkeys(int(e) for e in experts))
        if len(wanted) > self.pool_size:
            raise ValueError(f"{len(wanted)} experts requested at once, pool holds {self.pool_size}")
        pinned = set(wanted)
        self.ensure_calls += 1
        self.requested += len(wanted)
        written = []
        for expert in wanted:
            slot = self.slot_of[expert]
            if slot == self.null_slot:
                slot = next(s for s in self._lru if self.expert_in[s] not in pinned)
                self._assign(slot, expert)
                written.append(slot)
            self._lru.move_to_end(slot)
        if written:
            self._write_slots(written)
            self._write_table()
            self.loads += len(written)
        return len(written)

    def _write_slots(self, slots: Sequence[int]) -> None:
        writer = getattr(ttnn, "copy_host_to_device_tensor_region", None)
        if writer is not None:
            for slot in slots:
                expert = self.expert_in[slot]
                writer(self.store.gate_up[expert], self.gate_up, slot_offset_bytes(self.gate_up, slot))
                writer(self.store.down[expert], self.down, slot_offset_bytes(self.down, slot))
            return
        self._rewrite_whole_pool()

    def _rewrite_whole_pool(self) -> None:
        """Placeholder until a region write is bound: re-upload both pool tensors. Test-only."""

        if self.store.torch_gate_up is None:
            raise RuntimeError("no region writer bound and the host store kept no source values")
        for pool, sources, shape, dim in (
            (self.gate_up, self.store.torch_gate_up, self._gate_up_shape, -1),
            (self.down, self.store.torch_down, self._down_shape, -2),
        ):
            full_shape = list(sources[0].shape[-2:])
            image = torch.zeros(1, self.pool_size + 1, *full_shape, dtype=torch.bfloat16)
            for slot, expert in enumerate(self.expert_in):
                if expert >= 0:
                    image[0, slot] = sources[expert][0, 0]
            host = ttnn.from_torch(
                image,
                mesh_mapper=self._shard_mapper(self.mesh_device, dim),
                dtype=self.store.dtype,
                layout=ttnn.TILE_LAYOUT,
            )
            ttnn.copy_host_to_device_tensor(host, pool)

    def lookup(self, selected: ttnn.Tensor) -> ttnn.Tensor:
        """Router picks (uint16, [1, 1, S, top_k]) -> slot indices [1, 1, S, top_k], uint16 row-major.

        One embedding for all S tokens of an MoE call: per-token lookups cost ~9 dispatches per token.
        """

        rows, top_k = selected.shape[-2], selected.shape[-1]
        flat = ttnn.reshape(ttnn.typecast(selected, ttnn.uint32), [1, rows * top_k])
        slots = ttnn.embedding(flat, self.table, layout=ttnn.TILE_LAYOUT)
        slots = ttnn.reshape(ttnn.slice(slots, [0, 0, 0], [1, rows * top_k, 1]), [1, 1, rows, top_k])
        return ttnn.to_layout(ttnn.typecast(slots, ttnn.uint16), ttnn.ROW_MAJOR_LAYOUT)


def slot_offset_bytes(pool: ttnn.Tensor, slot: int) -> int:
    """Byte offset of a slot in an interleaved TILE pool [1, S, K, N]: slots are contiguous runs of tile pages."""

    shape = pool.padded_shape
    pages_per_slot = (shape[-2] // 32) * (shape[-1] // 32)
    if pool.buffer_num_pages() != shape[-3] * pages_per_slot:
        raise RuntimeError("pool buffer is not one page per tile")
    return slot * pages_per_slot * pool.buffer_page_size()
