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
import os
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

    # eviction: "lru" (least recently used) or "lfu" (fewest requests so far, ties by recency). On the recorded
    # routing at batch 8 / pool 176, LFU loads ~15% fewer experts per decode step.
    eviction = "lru"
    # load experts as large pages: each expert's packed tiles (the host store bytes, pinned once) are DMA'd into a
    # staging buffer with 9216 B pages and moved into their pool slots on the device (kernels/expert_repack/);
    # ~1.8x the host -> device rate of the per-tile (576 B page) writes. Needs the IOMMU in DMA mode for pinning.
    large_page_loads = False
    # per-tile loads: all of a call's experts in one batched region write per pool (same bytes, same order). Off:
    # no measurable gain over one write per expert (batch-8 steps within the run-to-run spread).
    batched_tile_writes = False
    # per-tile loads read straight from pinned host memory (needs the IOMMU in DMA mode; falls back to a copy)
    tile_writes_pin = False

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
        # the device slot table lags the host one: set by ensure(write_table=False) (the static verify / batch MoE
        # takes its slot ids from the host, not the table), written by the next ensure() that writes the table
        self._table_dirty = False
        # large-page loads: the last chunk of the latest ensure(defer_last_repack=True), left in staging for the
        # caller's repack (_LargePageLoader.repack_from, before any of those slots is read)
        self.deferred_slots: list[int] = []
        self.pending_slots: list[int] = []  # assigned by ensure(load=False), written by load_pending()
        # Hit-rate counters: ensure() calls and distinct experts asked for; hits = requested - loads.
        self.ensure_calls = 0
        self.requested = 0
        first = list(initial_experts) if initial_experts is not None else list(range(pool_size))
        if len(first) > pool_size or len(set(first)) != len(first):
            raise ValueError("initial experts must be unique and fit in the pool")
        self._place(first)
        self._zero_null_slot()

    def _zero_null_slot(self) -> None:
        """Zero weights in the null slot: the static MoE computes its unused groups on it, and their rows enter the
        one-hot combine multiplied by 0, which is exact only for finite values (an allocation holds stale bytes)."""
        writer = getattr(ttnn, "copy_host_to_device_tensor_region", None)
        if writer is None:
            return
        for pool, shape in ((self.gate_up, self._gate_up_shape), (self.down, self._down_shape)):
            zeros = ttnn.from_torch(
                torch.zeros(1, 1, *shape),
                dtype=self.store.dtype,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
            writer(zeros, pool, slot_offset_bytes(pool, self.null_slot))

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

    def ensure(
        self, experts: Iterable[int], *, write_table: bool = True, defer_last_repack: bool = False, load: bool = True
    ) -> int:
        """Make every listed expert resident; never evicts a listed expert. Returns the number loaded.

        ``write_table=False`` leaves the device slot table stale (written by the next ensure() that writes it);
        ``defer_last_repack`` (large-page loads) leaves the last chunk in staging, its slots in ``deferred_slots``.
        ``load=False`` only assigns the slots (``pending_slots``); load_pending() writes them later."""

        wanted = list(dict.fromkeys(int(e) for e in experts))
        if len(wanted) > self.pool_size:
            raise ValueError(f"{len(wanted)} experts requested at once, pool holds {self.pool_size}")
        pinned = set(wanted)
        self.ensure_calls += 1
        self.requested += len(wanted)
        lfu = self.eviction == "lfu"
        if lfu:
            counts = self.__dict__.setdefault("_request_counts", {})
            for expert in wanted:
                counts[expert] = counts.get(expert, 0) + 1
        written = []
        for expert in wanted:
            slot = self.slot_of[expert]
            if slot == self.null_slot:
                if lfu:  # fewest requests; min() keeps the first (least recent) of equal counts
                    slot = min(
                        (s for s in self._lru if self.expert_in[s] not in pinned),
                        key=lambda s: counts.get(self.expert_in[s], 0),
                    )
                else:
                    slot = next(s for s in self._lru if self.expert_in[s] not in pinned)
                self._assign(slot, expert)
                written.append(slot)
            self._lru.move_to_end(slot)
        self.deferred_slots = []
        if self.pending_slots:
            raise RuntimeError("ensure() with loads still pending (call load_pending first)")
        if written:
            if load:
                self.deferred_slots = self._write_slots(written, defer_last_repack=defer_last_repack)
            else:
                self.pending_slots = list(written)
            self.loads += len(written)
            self._table_dirty = True
        if write_table and self._table_dirty:
            self._write_table()
            self._table_dirty = False
        return len(written)

    def load_pending(self, cq_id: int = 0) -> None:
        """Write the slots ensure(load=False) assigned, on command queue ``cq_id``. Large-page loads only DMA to
        staging (one chunk, <= MAX_BATCH); the caller repacks them (_LargePageLoader.repack_from)."""
        slots, self.pending_slots = self.pending_slots, []
        if slots:
            self._write_slots(slots, defer_last_repack=True, cq_id=cq_id)

    def _write_slots(self, slots: Sequence[int], *, defer_last_repack: bool = False, cq_id: int = 0) -> list[int]:
        """Returns the slots whose repack was deferred (large-page loads, ``defer_last_repack``), else []."""
        writer = getattr(ttnn, "copy_host_to_device_tensor_region", None)
        if writer is not None and self.large_page_loads:
            return _LargePageLoader.get(self.mesh_device).load(
                self, slots, defer_last_repack=defer_last_repack, cq_id=cq_id
            )
        batch_writer = _regions_writer()
        if batch_writer is not None and self.batched_tile_writes:  # one queue submission per pool
            for pool, store in ((self.gate_up, self.store.gate_up), (self.down, self.store.down)):
                batch_writer(
                    [store[self.expert_in[slot]] for slot in slots],
                    pool,
                    [slot_offset_bytes(pool, slot) for slot in slots],
                    cq_id=ttnn.QueueId(cq_id),
                    pin=self.tile_writes_pin,
                )
            return []
        if writer is not None:
            if cq_id:
                raise ValueError("loads on another command queue need the batched region writer")
            for slot in slots:
                expert = self.expert_in[slot]
                writer(self.store.gate_up[expert], self.gate_up, slot_offset_bytes(self.gate_up, slot))
                writer(self.store.down[expert], self.down, slot_offset_bytes(self.down, slot))
            return []
        self._rewrite_whole_pool()
        return []

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


def _regions_writer():
    """copy_host_to_device_tensor_regions (several host tensors into one device tensor, one queue submission): only
    bound in the C++ module, not re-exported at the ttnn top level."""
    return getattr(ttnn, "copy_host_to_device_tensor_regions", None) or getattr(
        ttnn._ttnn.operations.core, "copy_host_to_device_tensor_regions", None
    )


def slot_offset_bytes(pool: ttnn.Tensor, slot: int) -> int:
    """Byte offset of a slot in an interleaved TILE pool [1, S, K, N]: slots are contiguous runs of tile pages."""

    shape = pool.padded_shape
    pages_per_slot = (shape[-2] // 32) * (shape[-1] // 32)
    if pool.buffer_num_pages() != shape[-3] * pages_per_slot:
        raise RuntimeError("pool buffer is not one page per tile")
    return slot * pages_per_slot * pool.buffer_page_size()


class _LargePageLoader:
    """Shared staging buffers (one per pool shape, all layers) and the device repack program."""

    TILES_PER_PAGE = 16
    MAX_BATCH = 16
    CORES = int(os.environ.get("LP_REPACK_CORES", "16"))
    DEPTH = int(os.environ.get("LP_REPACK_DEPTH", "16"))
    _instances: dict = {}

    @classmethod
    def get(cls, mesh_device) -> "_LargePageLoader":
        key = id(mesh_device)
        if key not in cls._instances:
            cls._instances[key] = cls(mesh_device)
        return cls._instances[key]

    def __init__(self, mesh_device) -> None:
        self.mesh_device = mesh_device
        self.staging: dict = {}

    def _staging(self, pool: ttnn.Tensor) -> tuple[ttnn.Tensor, int, int]:
        shape = tuple(pool.padded_shape)
        if shape not in self.staging:
            tiles = (shape[-2] // 32) * (shape[-1] // 32)  # per slot, per card
            if tiles % self.TILES_PER_PAGE:
                raise ValueError(f"expert of {tiles} tiles is not a whole number of {self.TILES_PER_PAGE}-tile pages")
            pages = tiles // self.TILES_PER_PAGE
            page_words = self.TILES_PER_PAGE * pool.buffer_page_size() // 4
            devices = self.mesh_device.get_num_devices()
            tensor = ttnn.from_torch(
                torch.zeros(1, 1, self.MAX_BATCH * pages, devices * page_words, dtype=torch.int32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.mesh_device,
                mesh_mapper=ttnn.create_mesh_mapper(
                    self.mesh_device,
                    ttnn.MeshMapperConfig(
                        placements=[ttnn.PlacementShard(-1)], mesh_shape_override=ttnn.MeshShape([devices])
                    ),
                ),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self.staging[shape] = (tensor, tiles, pages)
        return self.staging[shape]

    PARAMS_WORDS = 32  # [n, slot 0 .. slot MAX_BATCH - 1], padded to 128 B

    def _params(self) -> ttnn.Tensor:
        if getattr(self, "params", None) is None:
            self.params = ttnn.from_torch(
                torch.zeros(1, self.PARAMS_WORDS, dtype=torch.int32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        return self.params

    def _program(self, staging: ttnn.Tensor, pool: ttnn.Tensor, tiles: int, params: ttnn.Tensor | None = None):
        """One static program per (staging, pool, params): every runtime argument is constant. ``params``: a uint32
        row-major device tensor whose first PARAMS_WORDS words are [n, slot 0, ...] (default: the loader's own)."""

        params = self._params() if params is None else params
        key = (id(staging), id(pool), id(params))
        programs = self.__dict__.setdefault("programs", {})
        if key not in programs:
            grid = self.mesh_device.compute_with_storage_grid_size()
            cores_n = min(self.CORES, grid.x * grid.y)
            coords = [(k % grid.x, k // grid.x) for k in range(cores_n)]
            cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in coords])
            tile_bytes = pool.buffer_page_size()
            rt = ttnn.RuntimeArgs()
            for k, (x, y) in enumerate(coords):
                rt[x][y] = [staging.buffer_address(), pool.buffer_address(), params.buffer_address(), k, cores_n]
            kernel = Path(__file__).resolve().parent / "kernels" / "expert_repack" / "repack_tiles.cpp"
            program = ttnn.ProgramDescriptor(
                kernels=[
                    ttnn.KernelDescriptor(
                        kernel_source=str(kernel),
                        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                        core_ranges=cores,
                        compile_time_args=[tile_bytes, tiles, self.TILES_PER_PAGE, self.DEPTH, self.PARAMS_WORDS * 4]
                        + ttnn.TensorAccessorArgs(staging).get_compile_time_args()
                        + ttnn.TensorAccessorArgs(pool).get_compile_time_args()
                        + ttnn.TensorAccessorArgs(params).get_compile_time_args(),
                        runtime_args=rt,
                        config=ttnn.ReaderConfigDescriptor(),
                    )
                ],
                semaphores=[],
                cbs=[
                    ttnn.CBDescriptor(
                        total_size=self.DEPTH * tile_bytes,
                        core_ranges=cores,
                        format_descriptors=[
                            ttnn.CBFormatDescriptor(buffer_index=0, data_format=pool.dtype, page_size=tile_bytes)
                        ],
                    ),
                    ttnn.CBDescriptor(
                        total_size=self.PARAMS_WORDS * 4,
                        core_ranges=cores,
                        format_descriptors=[
                            ttnn.CBFormatDescriptor(
                                buffer_index=1, data_format=ttnn.uint32, page_size=self.PARAMS_WORDS * 4
                            )
                        ],
                    ),
                ],
            )
            programs[key] = program
        return programs[key]

    def _write_params(self, slots: Sequence[int]) -> None:
        values = torch.zeros(1, self.PARAMS_WORDS, dtype=torch.int32)
        values[0, 0] = len(slots)
        values[0, 1 : 1 + len(slots)] = torch.tensor(list(slots), dtype=torch.int32)
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(
                values,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            ),
            self._params(),
        )

    def _repack(self, staging: ttnn.Tensor, pool: ttnn.Tensor, tiles: int, params: ttnn.Tensor | None = None) -> None:
        params = self._params() if params is None else params
        ttnn.generic_op([staging, pool, params], self._program(staging, pool, tiles, params))

    def repack_from(self, pool_owner: "ExpertPool", params: ttnn.Tensor) -> None:
        """Device side of a deferred chunk (traceable): repack both pools, slots read from ``params`` on device."""
        for pool in (pool_owner.gate_up, pool_owner.down):
            staging, tiles, _ = self._staging(pool)
            self._repack(staging, pool, tiles, params)

    def prepin(self, pool_owner: "ExpertPool") -> None:
        """Pin every expert of this pool's host store once (a pinned write into staging slot 0); later loads then
        hit the pin cache (set TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES above the store size, default 4 GiB)."""

        writer = ttnn.copy_host_to_device_tensor_region
        for pool, store in ((pool_owner.gate_up, pool_owner.store.gate_up), (pool_owner.down, pool_owner.store.down)):
            staging, _, _ = self._staging(pool)
            for host in store:
                writer(host, staging, 0, pin=True, raw_bytes=True)

    def load(
        self, pool_owner: "ExpertPool", slots: Sequence[int], *, defer_last_repack: bool = False, cq_id: int = 0
    ) -> list[int]:
        """DMA the slots' experts to staging and repack them, MAX_BATCH at a time. ``defer_last_repack``: the last
        chunk stays in staging unrepacked and its slots are returned (the caller repacks before reading them)."""
        writer = ttnn.copy_host_to_device_tensor_region
        deferred: list[int] = []
        for start in range(0, len(slots), self.MAX_BATCH):
            chunk = list(slots[start : start + self.MAX_BATCH])
            last = start + self.MAX_BATCH >= len(slots)
            repack_now = not (defer_last_repack and last)
            if repack_now and cq_id:
                raise ValueError("loads on another command queue repack in the caller: one chunk, deferred")
            if repack_now:
                self._write_params(chunk)  # one params write for both pools (same slots)
            else:
                deferred = chunk
            for pool, store in (
                (pool_owner.gate_up, pool_owner.store.gate_up),
                (pool_owner.down, pool_owner.store.down),
            ):
                staging, tiles, pages = self._staging(pool)
                page_bytes = staging.buffer_page_size()
                hosts = [store[pool_owner.expert_in[slot]] for slot in chunk]
                offsets = [i * pages * page_bytes for i in range(len(chunk))]
                batch_writer = _regions_writer()
                if batch_writer is not None:  # every missing expert of the layer in one queue submission
                    batch_writer(hosts, staging, offsets, cq_id=ttnn.QueueId(cq_id), pin=True, raw_bytes=True)
                else:
                    for host, offset in zip(hosts, offsets):
                        writer(host, staging, offset, pin=True, raw_bytes=True)
                if repack_now:
                    self._repack(staging, pool, tiles)
        return deferred
