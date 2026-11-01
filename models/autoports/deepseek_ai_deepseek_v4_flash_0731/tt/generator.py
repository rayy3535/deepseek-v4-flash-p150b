"""Metal-readiness generator for the DeepSeek-V4-Flash TTNN full model.

The optimized mode uses two traces: one for model decode (including embedding,
all 43 decoder layers, terminal projection, and the on-device position update)
and one for the common sampler.  The sampler writes its result directly into
the persistent model token input.  Python schedules trace replays and may read
the small sampled-token tensor for presentation, but never computes or feeds
the next token in optimized mode.

``sampling_mode="host"`` is an explicit compatibility boundary for accuracy
tests.  It gathers logits and performs host argmax, and is intentionally never
used for token-out performance claims.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, Mapping, Sequence

import torch

import ttnn
from models.common.tt_ccl import get_tt_ccl
from models.common.sampling import SamplingGenerator, SamplingParams, broadcast_sampling_params, format_sampling_params

from .model import DeepSeekV4FlashModel, FullModelDecodeTrace, FullModelPrefillState
from .multichip_decoder import CCL_NUM_LINKS, TP_AXIS

SamplingMode = Literal["device", "host"]
_UNCHANGED_PAGE_TABLE = object()


def _page_table_version_changed(requested: object, current: object | None) -> bool:
    return requested is not _UNCHANGED_PAGE_TABLE and requested != current


@dataclass
class RuntimeAudit:
    """Counters that make host boundaries and page-table traffic auditable."""

    model_trace_replays: int = 0
    sampler_trace_replays: int = 0
    sampler_eager_steps: int = 0
    sampler_state_initializations: int = 0
    sampler_rng_advance_calls: int = 0
    page_table_copy_events: int = 0
    page_table_unchanged_replays: int = 0
    host_logit_reads: int = 0
    host_token_reads: int = 0
    host_token_feedback_writes: int = 0
    cache_resets: int = 0


@dataclass
class DecodeState:
    token_buffer: ttnn.Tensor
    current_position: ttnn.Tensor
    model_trace: FullModelDecodeTrace
    slot: int
    sampling_mode: SamplingMode
    sampler_traced: bool
    page_table_version: object | None
    initial_token_host: torch.Tensor
    initial_position_host: torch.Tensor


def _host_mesh_tensor(
    value: torch.Tensor,
    *,
    mesh_device: ttnn.MeshDevice,
    dtype: ttnn.DataType,
    layout: ttnn.Layout = ttnn.ROW_MAJOR_LAYOUT,
) -> ttnn.Tensor:
    return ttnn.from_torch(
        value,
        device=None,
        dtype=dtype,
        layout=layout,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _device_tensor_first_rank(value: ttnn.Tensor) -> torch.Tensor:
    local = ttnn.get_device_tensors(value)
    return ttnn.to_torch(local[0] if local else value)


class Generator:
    """Standard low-level prefill/decode plus autoregressive token-out wrapper."""

    def __init__(
        self,
        model: DeepSeekV4FlashModel,
        *,
        tokenizer=None,
        default_sampling_mode: SamplingMode = "device",
        allow_reduced_probe: bool = False,
    ) -> None:
        if default_sampling_mode not in ("device", "host"):
            raise ValueError(f"unknown sampling mode {default_sampling_mode!r}")
        if model.max_batch_size > 32:
            raise ValueError("the common split sampler supports at most 32 fixed slots")
        if model.reduced_probe and not allow_reduced_probe:
            raise ValueError("production Generator rejects reduced-layer/context probe models")
        self.model = model
        self.mesh_device = model.mesh_device
        self.tokenizer = tokenizer
        self.default_sampling_mode = default_sampling_mode
        self.audit = RuntimeAudit()
        self.decode_state: DecodeState | None = None
        self._sampling_params = SamplingParams(temperature=0.0, top_k=1, top_p=0.0, seed=None)

        # Chosen common path: SamplingGenerator/TTSampling.  Force-argmax is
        # disabled so k=1 remains the semantic local-top-k(32) + candidate
        # gather path and never gathers the full vocabulary.
        sampler_args = SimpleNamespace(
            vocab_size=model.vocab_size,
            padded_vocab_size=model.padded_vocab_size,
            max_batch_size=model.max_batch_size,
            max_top_k=32,
            cluster_shape=list(self.mesh_device.shape),
            sampling_all_gather_axis=TP_AXIS,
            sampling_dp=1,
            pad_logits_to_power_of_2=True,
            use_topk_logprobs=False,
            model_config={
                "SAMPLING_AG_CONFIG": {
                    "allow_force_argmax": False,
                    "num_links": CCL_NUM_LINKS,
                    "chunks_per_sync": 10,
                    "num_workers_per_link": 2,
                    "topology": ttnn.Topology.Ring,
                }
            },
        )
        self.sampling = SamplingGenerator(
            args=sampler_args,
            mesh_device=self.mesh_device,
            tt_ccl=get_tt_ccl(self.mesh_device),
        )
        self.set_sampling_params(self._sampling_params)
        if self.sampling.tt_sampling.force_argmax_sampling:
            raise RuntimeError("canonical split greedy unexpectedly selected the full-vocab argmax path")

    def set_sampling_params(self, params: SamplingParams) -> None:
        formatted = format_sampling_params(params, self.sampling.tt_sampling.max_batch_size)
        self.sampling.reset_sampling_params(formatted)
        self._sampling_params = params
        if (
            params.temperature == 0.0
            and params.top_k == 1
            and params.top_p == 0.0
            and self.sampling.tt_sampling.force_argmax_sampling
        ):
            raise RuntimeError("greedy sampling must use split local top-k, not full-vocabulary force-argmax")

    def _apply_prefill_sampling_state(self, prompt_token_ids: torch.Tensor, *, slot: int) -> None:
        """Admit one fixed-slot request into the common sampler lifecycle."""

        if prompt_token_ids.ndim != 1:
            raise ValueError("sampling prompt state requires one unpadded token sequence")
        if not 0 <= slot < self.model.max_batch_size:
            raise ValueError(f"slot {slot} outside [0, {self.model.max_batch_size})")
        width = self.sampling.tt_sampling.max_batch_size
        per_request = format_sampling_params(
            broadcast_sampling_params(self._sampling_params, slot, slot_len=width),
            width,
        )
        prompt_rows = torch.full((width, prompt_token_ids.numel()), -1, dtype=torch.int64)
        prompt_rows[slot] = prompt_token_ids.to(torch.int64)
        self.sampling.apply_prefill_state(
            sampling_params=per_request,
            prompt_tokens=prompt_rows,
            empty_slots=[slot],
            replicate_seeds=False,
        )
        self.audit.sampler_state_initializations += 1

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
        return self.model.prepare_prefill(
            token_ids,
            prompt_lengths=prompt_lengths,
            slot_ids=slot_ids,
            query_chunk_size=query_chunk_size,
            resume_start=resume_start,
            snapshot_rings=snapshot_rings,
        )

    def prefill_forward(self, state: FullModelPrefillState, *, return_all_logits: bool = False, **taps) -> ttnn.Tensor:
        """``taps``: tap_layers / tap_callback / tap_from, passed through."""

        return self.model.prefill_forward(state, return_all_logits=return_all_logits, **taps)

    @staticmethod
    def _sampling_logits(logits: ttnn.Tensor) -> ttnn.Tensor:
        rows = logits.shape[-2]
        if rows > 32:
            raise ValueError(f"sampling accepts at most one 32-row decode tile, got {rows}")
        return logits if rows == 32 else ttnn.pad(logits, [(0, 0), (0, 0), (0, 32 - rows), (0, 0)], 0.0)

    def _new_token_buffer(self, token: int, *, slot: int) -> tuple[ttnn.Tensor, torch.Tensor]:
        if not 0 <= slot < self.model.max_batch_size:
            raise ValueError(f"slot {slot} outside [0, {self.model.max_batch_size})")
        host = torch.zeros((1, 1, 1, 32), dtype=torch.uint32)
        host[0, 0, 0, slot] = int(token)
        device = _host_mesh_tensor(host, mesh_device=self.mesh_device, dtype=ttnn.uint32).to(
            self.mesh_device, ttnn.DRAM_MEMORY_CONFIG
        )
        return device, host

    def _new_position_buffer(self, position: int) -> tuple[ttnn.Tensor, torch.Tensor]:
        host = torch.tensor([int(position)], dtype=torch.int32)
        device = _host_mesh_tensor(host, mesh_device=self.mesh_device, dtype=ttnn.int32).to(
            self.mesh_device, ttnn.DRAM_MEMORY_CONFIG
        )
        return device, host

    @staticmethod
    def _copy_host(host: torch.Tensor, device: ttnn.Tensor, *, mesh_device: ttnn.MeshDevice, dtype) -> None:
        ttnn.copy_host_to_device_tensor(
            _host_mesh_tensor(host, mesh_device=mesh_device, dtype=dtype),
            device,
        )

    def sample_prefill(
        self,
        logits: ttnn.Tensor,
        *,
        prompt_token_ids: torch.Tensor | None = None,
        slot: int = 0,
        sampling_mode: SamplingMode | None = None,
        token_buffer: ttnn.Tensor | None = None,
    ) -> tuple[int, ttnn.Tensor]:
        """Sample the first token; device mode does not cross a logits boundary.

        ``token_buffer``: write the token into this persistent buffer instead of a new one, so no
        long-lived buffer is allocated behind a live trace."""

        mode = self.default_sampling_mode if sampling_mode is None else sampling_mode
        if token_buffer is None:
            token_buffer, _ = self._new_token_buffer(0, slot=slot)
        if mode == "device":
            if prompt_token_ids is None:
                raise ValueError("device prefill sampling requires prompt_token_ids for RNG and penalty state")
            self._apply_prefill_sampling_state(prompt_token_ids, slot=slot)
            sampled = self.sampling.sample(
                self._sampling_logits(logits),
                enable_trace=False,
                tt_out_tok=token_buffer,
            )
            self.audit.sampler_eager_steps += 1
            if isinstance(sampled, tuple):
                sampled = sampled[0]
            token = int(_device_tensor_first_rank(token_buffer).reshape(-1)[slot].item())
            self.audit.host_token_reads += 1
            return token, token_buffer
        if mode != "host":
            raise ValueError(f"unknown sampling mode {mode!r}")
        host_logits = self.model.gather_logits_for_host(logits)
        self.audit.host_logit_reads += 1
        token = int(torch.argmax(host_logits[0]).item())
        host = torch.zeros((1, 1, 1, 32), dtype=torch.uint32)
        host[0, 0, 0, slot] = token
        self._copy_host(host, token_buffer, mesh_device=self.mesh_device, dtype=ttnn.uint32)
        self.audit.host_token_feedback_writes += 1
        return token, token_buffer

    def prepare_decode_trace(
        self,
        token_buffer: ttnn.Tensor,
        *,
        current_position: int,
        slot: int = 0,
        sampling_mode: SamplingMode | None = None,
        page_table_version: object | None = None,
        position_buffer: ttnn.Tensor | None = None,
    ) -> DecodeState:
        """Compile, restore persistent inputs, then capture model and sampler.

        ``position_buffer``: a persistent int32 [1] position input allocated before any trace (kept
        by the caller when the trace is released, like a persistent token buffer)."""

        if self.decode_state is not None:
            self.release_decode_trace()
        mode = self.default_sampling_mode if sampling_mode is None else sampling_mode
        if mode not in ("device", "host"):
            raise ValueError(f"unknown sampling mode {mode!r}")
        if position_buffer is None:
            position_device, position_host = self._new_position_buffer(current_position)
        else:
            position_device, position_host = position_buffer, torch.tensor([int(current_position)], dtype=torch.int32)
            self._copy_host(position_host, position_device, mesh_device=self.mesh_device, dtype=ttnn.int32)
        token_host = _device_tensor_first_rank(token_buffer).to(torch.uint32)

        compile_logits = self.model.decode_forward(token_buffer, position_device, slot=slot)
        sampler_traced = mode == "device" and not self.sampling.seed_manager.has_active_request_seed()
        if mode == "device":
            self.sampling.precompile(logits=compile_logits, tt_out_tok=token_buffer)

        # Compile and sampler precompile mutate position/token/cache state.  The
        # same cache position is safely overwritten on the first replay; restore
        # the two persistent trace inputs so capture/replay starts coherently.
        self._copy_host(token_host, token_buffer, mesh_device=self.mesh_device, dtype=ttnn.uint32)
        self._copy_host(position_host, position_device, mesh_device=self.mesh_device, dtype=ttnn.int32)
        model_trace = self.model.capture_decode_trace(token_buffer, position_device, slot=slot)
        if sampler_traced:
            self.sampling.capture_trace(model_trace.logits, tt_out_tok=token_buffer, skip_precompile=True)
        # Capture executes both graphs once.  Do not let those construction
        # executions consume an autoregressive step: restore the original
        # sampled token and absolute position after both trace captures.  Cache
        # writes at that position are idempotently overwritten by replay.
        self._copy_host(token_host, token_buffer, mesh_device=self.mesh_device, dtype=ttnn.uint32)
        self._copy_host(position_host, position_device, mesh_device=self.mesh_device, dtype=ttnn.int32)
        if hasattr(model_trace, "position"):
            # The paged trace picks its per-token compressor variant from the host-side position.
            model_trace.position = int(current_position)
        state = DecodeState(
            token_buffer=token_buffer,
            current_position=position_device,
            model_trace=model_trace,
            slot=slot,
            sampling_mode=mode,
            sampler_traced=sampler_traced,
            page_table_version=page_table_version,
            initial_token_host=token_host,
            initial_position_host=position_host,
        )
        self.decode_state = state
        return state

    def decode_trace_step(
        self,
        *,
        page_table_version: object = _UNCHANGED_PAGE_TABLE,
        page_table_updates: Mapping[tuple[int, int, str], Sequence[int]] | None = None,
        read_token: bool = True,
    ) -> int | None:
        """Replay model then sampler; token feedback and position update stay on device."""

        state = self.decode_state
        if state is None:
            raise RuntimeError("call prepare_decode_trace before decode_trace_step")
        changed = _page_table_version_changed(page_table_version, state.page_table_version)
        if changed:
            if page_table_updates is None:
                raise ValueError("a changed page_table_version requires explicit page_table_updates")
            writes = self.model.update_page_tables(page_table_updates)
            self.audit.page_table_copy_events += writes
            state.page_table_version = page_table_version
        else:
            if page_table_updates:
                raise ValueError("page_table_updates require a changed page_table_version")
            self.audit.page_table_unchanged_replays += 1

        logits = state.model_trace.execute(self.mesh_device, blocking=False)
        self.audit.model_trace_replays += 1
        if state.sampling_mode == "device":
            self.sampling.seed_manager.get_new_values([state.slot])
            self.audit.sampler_rng_advance_calls += 1
            sampled = self.sampling.sample(
                logits,
                enable_trace=state.sampler_traced,
                tt_out_tok=state.token_buffer,
            )
            if isinstance(sampled, tuple):
                sampled = sampled[0]
            if state.sampler_traced:
                self.audit.sampler_trace_replays += 1
            else:
                self.audit.sampler_eager_steps += 1
        else:
            host_logits = self.model.gather_logits_for_host(logits)
            self.audit.host_logit_reads += 1
            token = int(torch.argmax(host_logits[state.slot]).item())
            host = torch.zeros((1, 1, 1, 32), dtype=torch.uint32)
            host[0, 0, 0, state.slot] = token
            self._copy_host(host, state.token_buffer, mesh_device=self.mesh_device, dtype=ttnn.uint32)
            self.audit.host_token_feedback_writes += 1
        if not read_token:
            return None
        token = int(_device_tensor_first_rank(state.token_buffer).reshape(-1)[state.slot].item())
        self.audit.host_token_reads += 1
        return token

    def resync_decode_inputs(self, token: int, position: int, *, token_buffer: ttnn.Tensor | None = None) -> None:
        """Continue the decode trace from a token and position committed outside it (a verify block).

        Writes the token into the persistent token buffer (the captured state's, or ``token_buffer`` before a
        capture) and, once a trace exists, the absolute position into its position input and host chain selector.
        """

        state = self.decode_state
        buffer = state.token_buffer if state is not None else token_buffer
        if buffer is None:
            raise RuntimeError("resync_decode_inputs needs a decode state or a token buffer")
        slot = state.slot if state is not None else 0
        host = torch.zeros((1, 1, 1, 32), dtype=torch.uint32)
        host[0, 0, 0, slot] = int(token)
        self._copy_host(host, buffer, mesh_device=self.mesh_device, dtype=ttnn.uint32)
        if state is None:
            return
        self._copy_host(
            torch.tensor([int(position)], dtype=torch.int32),
            state.current_position,
            mesh_device=self.mesh_device,
            dtype=ttnn.int32,
        )
        if hasattr(state.model_trace, "position"):
            state.model_trace.position = int(position)

    def sampler_trace_key(self) -> tuple:
        """The sampler state a captured decode step depends on: a decode state captured under another key must
        be recaptured, not replayed."""

        sampling = self.sampling
        return (
            sampling.tt_sampling.force_argmax_sampling,
            sampling._penalties_active,
            getattr(sampling, "_log_probs_active", False),
            sampling.seed_manager.has_active_request_seed(),
        )

    def release_decode_trace(self, *, keep_token_buffer: bool = False) -> None:
        """``keep_token_buffer``: the token and position buffers are owned by the caller (persistent ones) and stay
        allocated."""

        if self.decode_state is None:
            return
        state = self.decode_state
        # Drop the public state before releasing its trace-owned endpoints so
        # no later request can observe half-released persistent inputs.
        self.decode_state = None
        state.model_trace.release(self.mesh_device)
        self.sampling.reset_trace()
        ttnn.synchronize_device(self.mesh_device)
        seen: set[int] = {id(state.token_buffer), id(state.current_position)} if keep_token_buffer else set()
        for tensor in (state.model_trace.logits, state.current_position, state.token_buffer):
            if id(tensor) in seen:
                continue
            seen.add(id(tensor))
            ttnn.deallocate(tensor, force=True)

    def reset(self) -> None:
        """Release traces and zero all mutable cache storage without rebuilding weights."""

        self.release_decode_trace()
        seen: set[int] = set()
        for per_layer in self.model.cache_state().values():
            for per_slot in per_layer.values():
                for paged in per_slot.values():
                    if paged is None or id(paged.tensor) in seen:
                        continue
                    seen.add(id(paged.tensor))
                    ttnn.fill(
                        paged.tensor,
                        0.0,
                        memory_config=paged.tensor.memory_config(),
                        output_tensor=paged.tensor,
                    )
        self.audit.cache_resets += 1

    def generate_token_ids(
        self,
        prompt_token_ids: torch.Tensor,
        *,
        max_new_tokens: int = 100,
        sampling_mode: SamplingMode | None = None,
        slot: int = 0,
        next_input: torch.Tensor | None = None,
        enable_trace: bool = True,
    ) -> list[int]:
        """Autoregressive generation with optional readiness teacher forcing.

        ``next_input`` is an explicit compatibility/testing input.  Predictions
        still come from the configured token-out sampler, then the reference
        token is copied into the persistent input before the next step.  It is
        never enabled for optimized free-running measurements.
        """

        if max_new_tokens <= 0:
            return []
        if prompt_token_ids.ndim != 1:
            raise ValueError("generate_token_ids accepts one unpadded prompt")
        state = self.prepare_prefill(prompt_token_ids, prompt_lengths=[prompt_token_ids.numel()], slot_ids=[slot])
        logits = self.prefill_forward(state)
        first, token_buffer = self.sample_prefill(
            logits,
            prompt_token_ids=prompt_token_ids,
            slot=slot,
            sampling_mode=sampling_mode,
        )
        result = [first]
        if max_new_tokens == 1:
            return result
        teacher = None if next_input is None else next_input.reshape(-1).to(torch.uint32)

        def teacher_force(index: int) -> None:
            if teacher is None:
                return
            if index >= teacher.numel():
                raise ValueError(f"next_input has {teacher.numel()} tokens, need at least {index + 1}")
            host = _device_tensor_first_rank(token_buffer).to(torch.uint32)
            host.reshape(-1)[slot] = teacher[index]
            self._copy_host(host, token_buffer, mesh_device=self.mesh_device, dtype=ttnn.uint32)
            self.audit.host_token_feedback_writes += 1

        teacher_force(0)
        if enable_trace:
            self.prepare_decode_trace(
                token_buffer,
                current_position=prompt_token_ids.numel(),
                slot=slot,
                sampling_mode=sampling_mode,
            )
            for step in range(max_new_tokens - 1):
                token = self.decode_trace_step()
                assert token is not None
                result.append(token)
                teacher_force(step + 1)
        else:
            mode = self.default_sampling_mode if sampling_mode is None else sampling_mode
            current_pos, _ = self._new_position_buffer(prompt_token_ids.numel())
            for step in range(max_new_tokens - 1):
                step_logits = self.model.decode_forward(token_buffer, current_pos, slot=slot)
                if mode == "device":
                    self.sampling.seed_manager.get_new_values([slot])
                    self.audit.sampler_rng_advance_calls += 1
                    self.sampling.sample(step_logits, enable_trace=False, tt_out_tok=token_buffer)
                    self.audit.sampler_eager_steps += 1
                elif mode == "host":
                    host_logits = self.model.gather_logits_for_host(step_logits)
                    self.audit.host_logit_reads += 1
                    host = _device_tensor_first_rank(token_buffer).to(torch.uint32)
                    host.reshape(-1)[slot] = int(torch.argmax(host_logits[slot]).item())
                    self._copy_host(host, token_buffer, mesh_device=self.mesh_device, dtype=ttnn.uint32)
                    self.audit.host_token_feedback_writes += 1
                else:
                    raise ValueError(f"unknown sampling mode {mode!r}")
                token = int(_device_tensor_first_rank(token_buffer).reshape(-1)[slot].item())
                self.audit.host_token_reads += 1
                result.append(token)
                teacher_force(step + 1)
        return result

    def generate(
        self,
        prompt_token_ids: str | torch.Tensor,
        *,
        max_new_tokens: int = 100,
        next_input: torch.Tensor | None = None,
        enable_trace: bool = True,
        sampling_mode: SamplingMode | None = None,
    ) -> str | list[int]:
        """Thin deterministic loop over low-level prefill/decode methods."""

        return_text = isinstance(prompt_token_ids, str)
        if return_text:
            if self.tokenizer is None:
                raise ValueError("string generation requires a tokenizer")
            encoded = self.tokenizer(prompt_token_ids, return_tensors="pt")["input_ids"][0]
        else:
            encoded = prompt_token_ids
        generated = self.generate_token_ids(
            encoded,
            max_new_tokens=max_new_tokens,
            sampling_mode=sampling_mode,
            next_input=next_input,
            enable_trace=enable_trace,
        )
        return self.tokenizer.decode(generated, skip_special_tokens=True) if return_text else generated


def build_generator(model_dir: str | Path, mesh_device: ttnn.MeshDevice, **kwargs) -> Generator:
    """Build the standard Metal-readiness generator from an official checkpoint."""

    tokenizer = kwargs.pop("tokenizer", None)
    default_sampling_mode = kwargs.pop("default_sampling_mode", "device")
    allow_reduced_probe = kwargs.pop("allow_reduced_probe_generator", False)
    model = DeepSeekV4FlashModel.from_checkpoint(model_dir, mesh_device, **kwargs)
    return Generator(
        model,
        tokenizer=tokenizer,
        default_sampling_mode=default_sampling_mode,
        allow_reduced_probe=allow_reduced_probe,
    )


__all__ = ["DecodeState", "Generator", "RuntimeAudit", "SamplingMode", "build_generator"]
