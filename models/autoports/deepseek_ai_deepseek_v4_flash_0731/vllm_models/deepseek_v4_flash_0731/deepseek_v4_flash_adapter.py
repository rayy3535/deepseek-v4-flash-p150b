"""vLLM TT plugin bridge for the qualified QuietBox2 deployment.

The plugin discovers this bundle through EXTRA_MODELS_DIR and registers
TTDeepseekV4ForCausalLM without requiring a vllm-tt-plugin source change.
The underlying model owns its heterogeneous paged caches; vLLM owns request
admission and supplies tokens/positions, but its uniform block table is not a
device-cache allocation contract for this model.

This first production policy is deliberately batch-one and on-device-sampling
only. Those constraints match the capacity/quality qualification and prevent
the bridge from silently advertising an untested scheduling mode.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheSpec

import ttnn
from models.autoports.deepseek_ai_deepseek_v4_flash_0731.tt.generator import Generator
from models.autoports.deepseek_ai_deepseek_v4_flash_0731.tt.model import (
    FULL_LAYER_INDICES,
    QUIETBOX2_CONTEXT,
    STREAMING_PREFILL_BLOCK_SIZE,
    STREAMING_PREFILL_PROGRAM_CACHE_BLOCKS,
    DeepSeekV4FlashModel,
)
from models.autoports.deepseek_ai_deepseek_v4_flash_0731.tt.multichip_decoder import (
    QUIETBOX2_PAGED_BFP4_POLICY,
    QUIETBOX2_SELECTIVE_BFP4_POLICY,
    TARGET_MESH_SHAPE,
    MultichipDecoder,
    MultichipDeepSeekSparseMoE,
)

CHECKPOINT_ENV = "DEEPSEEK_V4_FLASH_HF_MODEL"
# Expert paging (all routed experts BFP4, a pool of this many per layer on device; default DEFAULT_EXPERT_POOL).
# "off" keeps the all-resident selective BFP2/BFP4 deployment (needs a build with ttnn.bfloat2_b). Not a DSV4_* variable: those are refused as policy overrides.
EXPERT_POOL_ENV = "DEEPSEEK_V4_FLASH_EXPERT_POOL"
# Directory of the packed routed-expert cache used with paging (default DEFAULT_EXPERT_CACHE); empty means no cache.
EXPERT_CACHE_ENV = "DEEPSEEK_V4_FLASH_EXPERT_CACHE"
# Attention projection weights: bfp8 (default) or bfp4.
ATTENTION_DTYPE_ENV = "DEEPSEEK_V4_FLASH_ATTENTION_DTYPE"
# Shared-expert weights: bfp8 (default, the checkpoint's FP8 precision) or bfp4.
SHARED_DTYPE_ENV = "DEEPSEEK_V4_FLASH_SHARED_DTYPE"
# Log expert-pool hit rate after every prefill and every N decode steps; unset or 0 means off.
POOL_STATS_ENV = "DEEPSEEK_V4_FLASH_POOL_STATS"
# Streaming prefill block in tokens: 128 (fork default) up to 512, a multiple of 128.
PREFILL_BLOCK_ENV = "DEEPSEEK_V4_FLASH_PREFILL_BLOCK"
# Session continuation: 1 = when a new prompt shares the previous prompt's tokens up to that prompt's last
# 128-token block start, restore the ring snapshot taken there and prefill only from it. Default 1; 0 turns it off.
SESSION_REUSE_ENV = "DEEPSEEK_V4_FLASH_SESSION_REUSE"
# 1 = keep the decode (and verify) traces across requests instead of recapturing them per request.
# Long-lived buffers (token buffer, ring snapshot) are allocated before any capture; a request continues the
# captured decode by rewriting its token / position inputs. Default 1 (always on with the DSpark drafter); 0 turns it off.
PERSISTENT_TRACES_ENV = "DEEPSEEK_V4_FLASH_PERSISTENT_TRACES"
_ATTENTION_DTYPES = {"bfp4": ttnn.bfloat4_b, "bfp8": ttnn.bfloat8_b}
SUPPORTED_OPTIMIZATIONS = (None, "performance", "accuracy")
PROGRAM_CACHE_PROMPT_SHAPES = 1
# longest draft a verify block serves (the block is 1 + K rows, one captured verify trace).
MAX_SPEC_DRAFTS = 7
# speculative decoding mode, set by scripts/run_dsv4_server.sh with the matching --speculative-config:
# "ngram:<K>" (host drafts, verify only) or "dspark:<K>" (the checkpoint's DSpark drafter on device, K drafts).
SPEC_ENV = "DEEPSEEK_V4_FLASH_SPEC"
# Routed-expert pool per DSpark draft layer (default 32).
DSPARK_POOL_ENV = "DEEPSEEK_V4_FLASH_DSPARK_POOL"
# mHC A2: 1 = decode / verify run the fused mHC input-map epilogue with the official 20 Sinkhorn iterations
# (MultichipDecoder.hc_pre_fused_epilogue). Default 1; 0 turns it off.
MHC_A2_ENV = "DEEPSEEK_V4_FLASH_MHC_A2"
# 1 = the DSpark draft layers run A2 too (default 0: only the main model, see initialize_vllm_model).
MHC_A2_DRAFTER_ENV = "DEEPSEEK_V4_FLASH_MHC_A2_DRAFTER"
# comma-separated MultichipDecoder fusion switches turned on for the main model only (the DSpark draft layers
# keep them off). Default DEFAULT_FUSIONS; "none" turns them all off.
FUSIONS_ENV = "DEEPSEEK_V4_FLASH_FUSIONS"
# indexer plan B (MultichipDecoder.indexer_topk_dynamic): "1" / "0" overrides the class default, unset keeps it.
INDEXER_DYNAMIC_ENV = "DEEPSEEK_V4_FLASH_INDEXER_DYNAMIC"
# (MultichipDecoder.verify_local_pad, default on): "1" / "0" overrides the class default, unset keeps it.
VERIFY_LOCAL_PAD_ENV = "DEEPSEEK_V4_FLASH_VERIFY_LOCAL_PAD"
# (MultichipDecoder.prefill_moe_merge): "1" / "0" overrides the class default, unset keeps it.
PREFILL_MOE_MERGE_ENV = "DEEPSEEK_V4_FLASH_PREFILL_MOE_MERGE"
# DeepSeekV4FlashModel.prefill_pad_short: default 1, "0" turns it off. On, a prompt of at
# most 128 tokens runs as one padded 128-token streaming block, so new short prompt lengths never compile prefill
# programs (which released every trace and, past PROGRAM_CACHE_PROMPT_SHAPES, cleared the program cache).
PREFILL_PAD_SHORT_ENV = "DEEPSEEK_V4_FLASH_PREFILL_PAD_SHORT"
# Trace guard (default 1; "0" turns it off): after a prefill that added program cache entries, release every trace before
# any replay (programs compiled behind live traces may have device buffers on a trace's freed intermediates).
TRACE_GUARD_ENV = "DEEPSEEK_V4_FLASH_TRACE_GUARD"
# with PREFILL_PAD_SHORT, "1" (default) compiles the padded prefill for every short length (1..128) at
# warmup, before any trace exists, so short prompts rarely trip the trace guard; "0" skips it.
PREFILL_WARM_ENV = "DEEPSEEK_V4_FLASH_PREFILL_WARM"
# (MultichipDecoder.prefill_hc_fused + prefill_hc_post_kernel): "1" / "0" overrides both, unset keeps them.
PREFILL_HC_ENV = "DEEPSEEK_V4_FLASH_PREFILL_HC"
# Adaptive speculation (2026-10-31): "1" turns it on. With DSpark, a round (drafts + verify + the step's overhead) costs
# about 3 plain decode steps, so speculating only pays while the drafts are mostly accepted (English / thinking text
# averages 4.2-4.5 tokens a round, Chinese chat without thinking 2.0-2.7, slower than plain decode). The adapter keeps an
# exponential average of the tokens each verify round commits and declines to draft (plain decode) while it is below
# DEEPSEEK_V4_FLASH_SPEC_ADAPT_MIN (default 3.0); while declining it probes one round every
# DEEPSEEK_V4_FLASH_SPEC_ADAPT_PROBE decode steps (default 32). The decisions use only the accepted counts, never
# timing, so a greedy request's output stays deterministic.
SPEC_ADAPT_ENV = "DEEPSEEK_V4_FLASH_SPEC_ADAPT"
SPEC_ADAPT_MIN_ENV = "DEEPSEEK_V4_FLASH_SPEC_ADAPT_MIN"
SPEC_ADAPT_PROBE_ENV = "DEEPSEEK_V4_FLASH_SPEC_ADAPT_PROBE"
SPEC_ADAPT_SMOOTHING = 0.3  # weight of the newest round in the average
_FUSION_SWITCHES = ("hc_pre_map_fused", "compressor_pool_fused", "moe_route_fused", "hc_post_fused")
# Tuned defaults for 4x Blackhole p150b (README); every one can be overridden by its environment variable.
DEFAULT_EXPERT_POOL = 176
DEFAULT_EXPERT_CACHE = "~/.cache/deepseek_v4_flash/experts_bfp4"
DEFAULT_FUSIONS = "hc_pre_map_fused,compressor_pool_fused,hc_post_fused"


def _single_value(value: Any, *, name: str) -> int:
    tensor = torch.as_tensor(value).reshape(-1)
    if tensor.numel() != 1:
        raise ValueError(f"{name} requires one active request, got {tensor.numel()}")
    return int(tensor[0].item())


def _prefill_start_position(value: Any) -> int:
    """Return the plugin's one start position for the qualified batch-one request."""

    if value is None:
        return 0
    return _single_value(value, name="prefill start_pos")


def _deployment() -> tuple[str, dict[str, Any]]:
    attention = os.environ.get(ATTENTION_DTYPE_ENV, "bfp8").strip().lower() or "bfp8"
    if attention not in _ATTENTION_DTYPES:
        raise ValueError(f"{ATTENTION_DTYPE_ENV} must be one of {sorted(_ATTENTION_DTYPES)}, got {attention!r}")
    extra: dict[str, Any] = {"attention_dtype": _ATTENTION_DTYPES[attention]}
    shared = os.environ.get(SHARED_DTYPE_ENV, "bfp8").strip().lower() or "bfp8"
    if shared not in _ATTENTION_DTYPES:
        raise ValueError(f"{SHARED_DTYPE_ENV} must be one of {sorted(_ATTENTION_DTYPES)}, got {shared!r}")
    MultichipDeepSeekSparseMoE.shared_dtype = _ATTENTION_DTYPES[shared]
    block = os.environ.get(PREFILL_BLOCK_ENV, "").strip()
    if block:
        extra["prefill_block_size"] = int(block)
    value = os.environ.get(EXPERT_POOL_ENV, str(DEFAULT_EXPERT_POOL)).strip().lower()
    if value in ("", "off", "0"):
        return QUIETBOX2_SELECTIVE_BFP4_POLICY, extra
    paging: dict[str, Any] = {"expert_pool_size": int(value), **extra}
    cache = os.environ.get(EXPERT_CACHE_ENV, DEFAULT_EXPERT_CACHE).strip()  # set but empty: no cache
    if cache:
        paging["expert_cache_dir"] = Path(cache).expanduser()
    return QUIETBOX2_PAGED_BFP4_POLICY, paging


def _spec_setting() -> tuple[str, int] | None:
    """(method, K) of DEEPSEEK_V4_FLASH_SPEC=ngram:<K> / dspark:<K>, or None when speculation is off."""

    value = os.environ.get(SPEC_ENV, "").strip().lower()
    if not value:
        return None
    method, _, k = value.partition(":")
    if method not in ("ngram", "dspark") or not k.isdigit() or not 1 <= int(k) <= MAX_SPEC_DRAFTS:
        raise ValueError(f"{SPEC_ENV}={value}: expected ngram:<K> or dspark:<K>, K in [1, {MAX_SPEC_DRAFTS}]")
    return method, int(k)


def _dspark_block() -> int | None:
    """K of DEEPSEEK_V4_FLASH_SPEC=dspark:<K>, or None when the device drafter is not configured."""

    setting = _spec_setting()
    return setting[1] if setting is not None and setting[0] == "dspark" else None


def _checkpoint_path(hf_config: Any) -> Path:
    configured = os.environ.get(CHECKPOINT_ENV)
    candidate = configured or getattr(hf_config, "_name_or_path", None)
    if not candidate:
        raise ValueError(f"set {CHECKPOINT_ENV} to the local DeepSeek-V4-Flash-0731 checkpoint")
    path = Path(candidate).expanduser().resolve()
    if not (path / "config.json").is_file():
        raise ValueError(f"DeepSeek checkpoint is missing config.json: {path}")
    return path


def _host_only_sampling_controls(params: Any) -> list[str]:
    """Request controls that make the TT runner fall back to host sampling, which this adapter does not support
    (_require_device_sampling would raise inside the engine core and take the whole server down)."""

    controls = []
    if params.logprobs is not None:
        controls.append("logprobs")
    if params.prompt_logprobs is not None:
        controls.append("prompt_logprobs")
    if getattr(params, "logprob_token_ids", None) is not None:
        controls.append("logprob_token_ids")
    if params.structured_outputs is not None:
        controls.append("structured outputs / response_format")
    if params.min_tokens:
        controls.append("min_tokens")
    if params.min_p:
        controls.append("min_p")
    if params.logit_bias:
        controls.append("logit_bias")
    if params.allowed_token_ids:
        controls.append("allowed_token_ids")
    if params.bad_words:
        controls.append("bad_words")
    if getattr(params, "thinking_token_budget", None) is not None:
        controls.append("thinking_token_budget")
    return controls


def _install_request_guard() -> None:
    """Reject host-sampling requests in the API server with a 400 instead of letting them reach the engine core.

    The plugin resolves this adapter's capabilities in TTPlatform.check_and_update_config, so the API server process
    imports this module before it validates any request; the guard wraps TTPlatform.validate_request there.
    """

    try:
        from vllm_tt_plugin.platform import TTPlatform
    except ImportError:  # offline use without the plugin
        return
    if getattr(TTPlatform, "_dsv4_request_guard", False):
        return
    original = TTPlatform.validate_request.__func__

    def validate_request(cls, processed_inputs, params):
        original(cls, processed_inputs, params)
        from vllm.exceptions import VLLMValidationError
        from vllm.sampling_params import SamplingParams

        if isinstance(params, SamplingParams):
            controls = _host_only_sampling_controls(params)
            if controls:
                raise VLLMValidationError(
                    "DeepSeek-V4-Flash on TT samples on device only; unsupported request controls: "
                    + ", ".join(controls)
                )

    TTPlatform.validate_request = classmethod(validate_request)
    TTPlatform._dsv4_request_guard = True


_install_request_guard()


class DeepseekV4ForCausalLM:
    """Batch-one vLLM facade around the all-resident mixed-BFP2 generator."""

    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_async_decode": False,
        "supports_sample_on_device": True,
        # greedy speculative decoding with host-proposed (ngram) drafts; only the verify side lives here.
        "supports_spec_decode": True,
        # device_propose: the DSpark drafter (custom_class); a host ngram launch needs nothing from it.
        "spec_requirements": ["device_propose"],
    }

    def __init__(self, model: DeepSeekV4FlashModel, drafter: Any = None) -> None:
        self.model = model
        # the DSpark drafter (tt/dspark.py) when DEEPSEEK_V4_FLASH_SPEC=dspark:<K>; its context follows every
        # computed position (prefill tail, decode and verify traces write it).
        self.drafter = drafter
        self.generator = Generator(model, default_sampling_mode="device")
        self._token_buffer: ttnn.Tensor | None = None
        self._active_position: int | None = None
        self._has_request = False
        self._kv_cache_contract: dict[str, Any] | None = None
        self._program_cache_prompt_lengths: set[int] = set()
        # DEEPSEEK_V4_FLASH_POOL_STATS=<N>: log pool counters after each prefill and every N decode steps.
        self._pool_stats_every = int(os.environ.get(POOL_STATS_ENV, "0") or 0)
        self._pool_stats_last: dict[str, int] = {}
        self._decode_steps_since_stats = 0
        # DEEPSEEK_V4_FLASH_SESSION_REUSE=1: previous prompt, to find the shared prefix of the next one.
        self._session_reuse = os.environ.get(SESSION_REUSE_ENV, "1").strip() == "1"
        self._previous_prompt: torch.Tensor | None = None
        # one captured verify trace per block width, and the last verify block (start, width) while the
        # committed position after it is still unknown (the next call's start_pos says how much was accepted).
        self._verify_traces: dict[int, Any] = {}
        self._last_verify: tuple[int, int] | None = None
        # persistent traces. The token buffer and the ring snapshot are allocated here, before any
        # trace; _resync_pending marks a decode state that must be pointed at the new request's first token.
        # The drafter keeps a block trace for the server's lifetime, so with it every trace input is persistent.
        self._persistent = os.environ.get(PERSISTENT_TRACES_ENV, "1").strip() == "1" or drafter is not None
        self._persistent_token_buffer: ttnn.Tensor | None = None
        self._decode_key: tuple | None = None
        self._resync_pending = False
        self._persistent_position_buffer: ttnn.Tensor | None = None
        self._spec = _spec_setting()
        self._adapt = os.environ.get(SPEC_ADAPT_ENV, "0").strip() == "1"
        self._trace_guard = os.environ.get(TRACE_GUARD_ENV, "1").strip() == "1"
        self._adapt_min = float(os.environ.get(SPEC_ADAPT_MIN_ENV, "3.0") or 3.0)
        self._adapt_probe = int(os.environ.get(SPEC_ADAPT_PROBE_ENV, "32") or 32)
        self._adapt_reset()
        if self._spec is not None:
            self._persistent = True  # speculation keeps its traces for the server's lifetime
        if self._persistent:
            self._persistent_token_buffer, _ = self.generator._new_token_buffer(0, slot=0)
            self._persistent_position_buffer, _ = self.generator._new_position_buffer(0)
            if self._session_reuse:
                self.model.preallocate_ring_snapshot(0)

    def _adapt_reset(self) -> None:
        """Adaptive speculation state for a new request: start out speculating, as if every draft were accepted."""

        self._adapt_avg = float(self._spec[1] + 1) if self._spec is not None else 0.0
        self._adapt_last_spec = False
        self._adapt_plain_steps = 0
        self._adapt_rounds = self._adapt_declines = 0

    def _adapt_wants_drafts(self, accepted: int) -> bool:
        """Update the average with the step that just ended (``accepted`` tokens committed) and decide this step."""

        if self._adapt_last_spec:
            self._adapt_avg += SPEC_ADAPT_SMOOTHING * (accepted - self._adapt_avg)
        else:
            self._adapt_plain_steps += 1
        if self._adapt_avg >= self._adapt_min or self._adapt_plain_steps >= self._adapt_probe:
            self._adapt_plain_steps = 0
            self._adapt_last_spec = True
            self._adapt_rounds += 1
            return True
        self._adapt_last_spec = False
        self._adapt_declines += 1
        return False

    @classmethod
    def spec_plan(cls, vllm_config: Any, max_num_seqs: int, requested_k: int):
        """argmax-only verify of a 1 + K block for the single request; narrow decode stays the traced
        single-token decode, so sampled requests decode normally inside a speculating launch."""

        from vllm_tt_plugin.spec_decode import SpecPlan, SpecReject

        if int(max_num_seqs) != 1:
            return SpecReject(reason="DeepSeek-V4-Flash serves one request", supported_k=())
        if int(requested_k) < 1:
            return SpecReject(reason=f"requested_k={requested_k}", supported_k=tuple(range(1, MAX_SPEC_DRAFTS + 1)))
        dspark = _dspark_block()
        if dspark is not None and int(requested_k) != dspark:
            return SpecReject(reason=f"the DSpark drafter is built for K={dspark} ({SPEC_ENV})", supported_k=(dspark,))
        return SpecPlan(
            effective_k=min(int(requested_k), MAX_SPEC_DRAFTS),
            lanes_per_request=1,
            extra_bytes_per_seq=0,
            extra_bytes_per_token=0,
            # argmax_ids for all-greedy steps (a few bytes back); logits once a row samples, for the plugin's
            # lossless rejection sampling on the host
            accept_modes=("argmax_ids", "logits"),
            drafter_state="internal",
            supports_narrow_decode=True,
        )

    @classmethod
    def get_max_tokens_all_users(
        cls,
        *,
        max_model_len: int | None = None,
        max_num_seqs: int | None = None,
        tt_data_parallel: int = 1,
        **_: Any,
    ) -> int:
        if max_model_len is not None and int(max_model_len) != QUIETBOX2_CONTEXT:
            raise ValueError(
                "DeepSeek-V4-Flash QuietBox2 vLLM deployment requires "
                f"max_model_len={QUIETBOX2_CONTEXT}, got {max_model_len}"
            )
        if max_num_seqs is not None and int(max_num_seqs) != 1:
            raise ValueError("DeepSeek-V4-Flash QuietBox2 vLLM deployment requires max_num_seqs=1")
        if int(tt_data_parallel) != 1:
            raise ValueError("DeepSeek-V4-Flash uses the full mesh as TP4; vLLM DP must be 1")
        return QUIETBOX2_CONTEXT

    @classmethod
    def get_kv_cache_spec(cls, vllm_config: Any) -> dict[str, KVCacheSpec]:
        """Publish only the scheduler's logical block-table contract.

        DeepSeek V4 mixes sliding, compressed, and heavily compressed
        attention while the TT model owns those heterogeneous physical cache
        banks.  vLLM still needs one logical block table for batch-one request
        admission.  A uniform sentinel spec supplies that table without asking
        the plugin's generic MLA path to infer or allocate duplicate caches.
        The shape is intentionally scheduler-only; ``allocate_kv_cache`` below
        records it and returns the already-resident model-owned cache contract.
        """

        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        parallel_config = vllm_config.parallel_config
        return {
            "model_owned_cache": FullAttentionSpec(
                block_size=int(cache_config.block_size),
                num_kv_heads=int(model_config.get_num_kv_heads(parallel_config)),
                head_size=int(model_config.get_head_size()),
                dtype=model_config.dtype,
                sliding_window=None,
            )
        }

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config: Any,
        mesh_device: ttnn.MeshDevice,
        max_batch_size: int,
        max_seq_len: int,
        tt_data_parallel: int = 1,
        optimizations: str | None = None,
        **_: Any,
    ) -> "DeepseekV4ForCausalLM":
        if tuple(mesh_device.shape) != TARGET_MESH_SHAPE or mesh_device.get_num_devices() != 4:
            raise ValueError(
                "DeepSeek-V4-Flash requires the complete 4x1 QuietBox2 mesh, "
                f"got shape={tuple(mesh_device.shape)} devices={mesh_device.get_num_devices()}"
            )
        if int(max_batch_size) != 1:
            raise ValueError(f"qualified deployment requires max_batch_size=1, got {max_batch_size}")
        if int(max_seq_len) != QUIETBOX2_CONTEXT:
            raise ValueError(f"qualified deployment requires max_seq_len={QUIETBOX2_CONTEXT}, got {max_seq_len}")
        if int(tt_data_parallel) != 1:
            raise ValueError("qualified deployment uses internal TP4 and vLLM data parallel size 1")
        if optimizations not in SUPPORTED_OPTIMIZATIONS:
            raise ValueError(f"unsupported optimizations={optimizations!r}; expected one of {SUPPORTED_OPTIMIZATIONS}")
        deployment_mode, paging = _deployment()
        MultichipDecoder.hc_pre_fused_epilogue = os.environ.get(MHC_A2_ENV, "1").strip() == "1"
        fusions_setting = os.environ.get(FUSIONS_ENV, DEFAULT_FUSIONS).strip()
        fusions = (
            set()
            if fusions_setting == "none"
            else {name.strip() for name in fusions_setting.split(",") if name.strip()}
        )
        if fusions - set(_FUSION_SWITCHES):
            raise ValueError(f"{FUSIONS_ENV}: unknown switches {sorted(fusions - set(_FUSION_SWITCHES))}")
        for name in _FUSION_SWITCHES:
            setattr(MultichipDecoder, name, name in fusions)
        indexer_dynamic = os.environ.get(INDEXER_DYNAMIC_ENV, "").strip()
        if indexer_dynamic:
            MultichipDecoder.indexer_topk_dynamic = indexer_dynamic == "1"
        local_pad = os.environ.get(VERIFY_LOCAL_PAD_ENV, "").strip()
        if local_pad:
            MultichipDecoder.verify_local_pad = local_pad == "1"
        moe_merge = os.environ.get(PREFILL_MOE_MERGE_ENV, "").strip()
        if moe_merge:
            MultichipDecoder.prefill_moe_merge = moe_merge == "1"
        pad_short = os.environ.get(PREFILL_PAD_SHORT_ENV, "1").strip()
        if pad_short:
            DeepSeekV4FlashModel.prefill_pad_short = pad_short == "1"
        prefill_hc = os.environ.get(PREFILL_HC_ENV, "").strip()
        if prefill_hc:
            MultichipDecoder.prefill_hc_fused = MultichipDecoder.prefill_hc_post_kernel = prefill_hc == "1"
        model = DeepSeekV4FlashModel.from_checkpoint(
            _checkpoint_path(hf_config),
            mesh_device,
            max_context=QUIETBOX2_CONTEXT,
            max_batch_size=1,
            layer_indices=FULL_LAYER_INDICES,
            deployment_mode=deployment_mode,
            **paging,
        )
        drafter = None
        block = _dspark_block()
        if block is not None:
            from models.autoports.deepseek_ai_deepseek_v4_flash_0731.tt.dspark import DSparkDrafter, DSparkLayer

            # stays off in the draft layers unless asked for: with it the drafts got worse (offline,
            # 6 prompts: mean acceptance 4.38 off / 3.96 everywhere / 4.17 main model only).
            DSparkLayer.hc_pre_fused_epilogue = os.environ.get(MHC_A2_DRAFTER_ENV, "0").strip() == "1"
            for name in _FUSION_SWITCHES:
                setattr(DSparkLayer, name, False)
            # 's prefill mHC switches are MultichipDecoder defaults too; the draft layers keep the op-by-op path
            # (with them on, the drafter's long context blocks took the fused input map and acceptance fell 4.3 -> ~2.2)
            DSparkLayer.prefill_hc_fused = DSparkLayer.prefill_hc_post_kernel = False

            cache = paging.get("expert_cache_dir")
            drafter = DSparkDrafter(
                model,
                _checkpoint_path(hf_config),
                block=block,
                expert_pool_size=int(os.environ.get(DSPARK_POOL_ENV, "32") or 32),
                expert_cache_dir=None if cache is None else Path(f"{cache}_mtp"),
                attention_dtype=paging.get("attention_dtype", ttnn.bfloat4_b),
            )
            model.decode_tap_layers = drafter.target_layers
            # The decode trace writes its position's context itself (last segment, before the position advances).
            model.decode_tap_hook = drafter.write_context
            drafter.register_rings()  # before the persistent ring snapshot is allocated
        return cls(model, drafter)

    @property
    def cache_path(self) -> None:
        # The model is constructed from local official safetensors and owns the
        # resulting TT tensors; no second vLLM weight-cache path is used.
        return None

    def allocate_kv_cache(self, kv_cache_shape: Any, dtype: Any, num_layers: int):
        """Accept vLLM's scheduler contract without allocating duplicate cache.

        The model already owns all local/compressed/indexer cache banks at the
        exact 131,072-token deployment context. The returned object is opaque
        to the plugin and is handed back to this bridge on every call.
        """

        if int(num_layers) != len(FULL_LAYER_INDICES):
            raise ValueError(f"vLLM requested {num_layers} cache layers; expected {len(FULL_LAYER_INDICES)}")
        self._kv_cache_contract = {
            "scheduler_shape": tuple(int(value) for value in kv_cache_shape),
            "scheduler_dtype": dtype,
            "model_owned": True,
            "max_context": QUIETBOX2_CONTEXT,
        }
        return self._kv_cache_contract

    def warmup_model_prefill(self, **_: Any) -> None:
        # The first real prompt compiles its exact ISL shape. Generic plugin
        # warmup would duplicate a full 43-layer pass and cannot cover all ISLs.
        # with padded short prompts the prefill programs still depend on the valid row count, and a prefill
        # that compiles releases every trace (trace guard). Compile all 128 counts once here, before any trace exists.
        if not self.model.prefill_pad_short or os.environ.get(PREFILL_WARM_ENV, "1").strip() != "1":
            return None
        if getattr(self, "_prefill_warmed", False):
            return None
        self._prefill_warmed = True
        start = time.perf_counter()
        programs = self.model.mesh_device.num_program_cache_entries()
        generator = torch.Generator().manual_seed(0)
        for length in range(1, STREAMING_PREFILL_BLOCK_SIZE + 1):
            prompt = torch.randint(1000, 100000, (length,), generator=generator)
            state = self.generator.prepare_prefill(prompt, prompt_lengths=[length], slot_ids=[0])
            taps = {}
            if self.drafter is not None:
                taps = {
                    "tap_layers": self.drafter.target_layers,
                    "tap_callback": lambda block_start, rows, block_taps: self.drafter.write_context_rows(
                        block_taps, block_start, 0, rows
                    ),
                    "tap_from": 0,
                }
            ttnn.deallocate(self.generator.prefill_forward(state, **taps))
        ttnn.synchronize_device(self.model.mesh_device)
        print(
            f"[prefill-warm] lengths 1..{STREAMING_PREFILL_BLOCK_SIZE}: "
            f"{self.model.mesh_device.num_program_cache_entries() - programs} programs, "
            f"{time.perf_counter() - start:.0f} s",
            flush=True,
        )
        return None

    def warmup_model_decode(self, **_: Any) -> None:
        # Decode capture is initialized from the first sampled prefill token so
        # token/position/sampler state all have production addresses.
        return None

    @staticmethod
    def _require_device_sampling(sampling_params: Any) -> None:
        if sampling_params is None:
            raise ValueError(
                "DeepSeek-V4-Flash vLLM bridge requires a device-sampling-compatible request; "
                "structured output, min_tokens, and host-only logits processors are unsupported"
            )

    @staticmethod
    def _validate_slot(values: Any, *, name: str) -> None:
        if values is None:
            return
        values = torch.as_tensor(values).reshape(-1)
        if values.numel() != 1 or int(values[0].item()) != 0:
            raise ValueError(f"{name} must identify the only qualified slot 0")

    def _reset_for_new_request(self) -> None:
        """Release request-local traces without zeroing the model-owned cache.

        A fresh prefill overwrites every KV position that the new request can
        read, so zeroing the full 131K cache here only adds request-boundary
        latency. Program-cache lifetime is managed separately because its
        all-gather semaphores have a different, bounded resource contract.
        """

        if not self._persistent:
            self.generator.release_decode_trace()
            self._release_verify_traces()
        self._token_buffer = None
        self._active_position = None
        self._has_request = False
        self._last_verify = None
        self._resync_pending = False

    def _release_decode_trace(self) -> None:
        self.generator.release_decode_trace(keep_token_buffer=self._persistent)
        if not self._persistent:
            self._token_buffer = None

    def _drop_traces(self) -> None:
        """Release every trace (keeping the persistent inputs) before anything compiles programs or clears the
        program cache: a program compiled behind a live trace (its CCL resources are allocated then), or a
        cached program freed under one, leaves memory the trace's replay uses inconsistent, which hung the
        device. The traces are captured again by the next decode / verify."""

        self._release_decode_trace()
        self._release_verify_traces()
        if self.drafter is not None:
            self.drafter.release_trace()

    def _warm_spec(self, position: int) -> None:
        """Speculation: (re)build every trace with every compile done while no trace is alive.

        A program compiled behind a live trace (its CCL resources are allocated then) or a long-lived buffer
        allocated behind one can land on memory the trace's replay rewrites; that hung the device twice. So:
        release every trace, compile the verify block (a throw-away capture) and the drafter, capture the decode
        trace (compiling its own programs as the only trace), then the drafter's block trace and the verify trace,
        which compile nothing. The cache
        rows written on the way are at ``position`` .. + 32 (an active request's current decode position, whose
        rows its next step rewrites, or an unused position) and rewritten before anything reads them."""

        method, k = self._spec
        self._drop_traces()

        def capture_verify():
            return self.model.capture_paged_verify_trace(
                k + 1,
                warm_tokens=[0] * (k + 1),
                warm_start=position,
                greedy=True,
                tap_layers=() if self.drafter is None else self.drafter.target_layers,
                tap_hook=None if self.drafter is None else self.drafter.write_context,
            )

        # 1. compile: the verify block (its capture compiles everything in its eager warm pass, then it goes),
        #    the drafter (eager), the decode step (first trace kept: its warm passes compile with nothing alive)
        capture_verify().release(self.model.mesh_device)
        if self.drafter is not None:
            self.drafter.warm_compile(verify_rows=k + 1, at=position + 1)
        token = self._token_buffer if self._token_buffer is not None else self._persistent_token_buffer
        self._token_buffer = token
        self.generator.prepare_decode_trace(
            token,
            current_position=position,
            slot=0,
            sampling_mode="device",
            page_table_version="model-owned-identity",
            position_buffer=self._persistent_position_buffer,
        )
        self._decode_key = self.generator.sampler_trace_key()
        # 2. capture the rest: nothing left to compile
        if self.drafter is not None:
            self.drafter.capture_trace()
        self._verify_traces[k + 1] = capture_verify()
        self._resync_pending = True  # the decode trace points at the warm-up position

    def _release_verify_traces(self) -> None:
        """Verify traces live no longer than the request: a prefill allocates long-lived buffers (ring snapshots)
        that must not land on a trace's freed intermediates, which a later replay would overwrite."""

        for trace in self._verify_traces.values():
            trace.release(self.model.mesh_device)
        self._verify_traces.clear()

    def _reserve_program_cache_for_prompt(self, prompt_len: int) -> None:
        """Bound shape-specific all-gather semaphores without long-prompt clears.

        Each distinct exact prefill length retains one 2 KiB-aligned global
        semaphore allocation while its program remains cached. Long prompts
        execute only fixed 128-token blocks, so all qualified benchmark and
        Terminal-Bench requests reuse shape 128 in the server's working 2 KiB
        L1_SMALL region. Exact short prompts clear only when switching shapes.
        """

        prefill_shape = (
            STREAMING_PREFILL_BLOCK_SIZE
            if self.model.prefill_pad_short
            else min(prompt_len, STREAMING_PREFILL_BLOCK_SIZE)
        )
        if prefill_shape in self._program_cache_prompt_lengths:
            return
        # A new shape compiles prefill programs: no trace may be alive then (see _drop_traces).
        self._drop_traces()
        if len(self._program_cache_prompt_lengths) >= PROGRAM_CACHE_PROMPT_SHAPES:
            self.model.mesh_device.clear_program_cache()
            self._program_cache_prompt_lengths.clear()
        self._program_cache_prompt_lengths.add(prefill_shape)

    def prefill_forward(
        self,
        *,
        tokens: torch.Tensor,
        prompt_lens: list[int],
        sampling_params: Any = None,
        start_pos: Any = None,
        empty_slots: Any = None,
        **_: Any,
    ) -> torch.Tensor:
        self._require_device_sampling(sampling_params)
        if tokens.ndim != 2 or tokens.shape[0] != 1:
            raise ValueError(f"prefill requires tokens shaped [1, S], got {tuple(tokens.shape)}")
        if len(prompt_lens) != 1:
            raise ValueError(f"prefill requires one prompt length, got {prompt_lens}")
        prompt_len = int(prompt_lens[0])
        if not 0 < prompt_len <= QUIETBOX2_CONTEXT:
            raise ValueError(f"prompt length {prompt_len} is outside the deployment context")
        prefill_start = _prefill_start_position(start_pos)
        self._validate_slot(empty_slots, name="empty_slots")

        # vllm-tt-plugin 0.1.0 classifies a normal running decode as prefill:
        # its continuation test compares num_computed_tokens with the complete
        # prompt-plus-output history, which is always one token ahead. Preserve
        # the plugin checkout unchanged, but recognize only that exact
        # sequential one-token shape and route it into the already-qualified
        # traced decode path. A real chunk/prefix continuation remains closed.
        if prefill_start != 0:
            if not self._has_request or self._active_position is None:
                raise ValueError("nonzero prefill start requires an active request")
            if prefill_start != self._active_position:
                raise ValueError(
                    f"non-sequential prefill continuation {prefill_start}; expected {self._active_position}"
                )
            if prompt_len != prefill_start + 1:
                raise ValueError(
                    "prefix caching/chunked prefill is not supported; "
                    "only the plugin's one-token sequential decode continuation is accepted"
                )
            return self.decode_forward(
                tokens=tokens[0, prefill_start:prompt_len],
                start_pos=torch.tensor([prefill_start], dtype=torch.int32),
                sampling_params=sampling_params,
            )

        if self._has_request:
            self._reset_for_new_request()
        self.generator.set_sampling_params(sampling_params)
        self._reserve_program_cache_for_prompt(prompt_len)
        if prompt_len > STREAMING_PREFILL_BLOCK_SIZE * STREAMING_PREFILL_PROGRAM_CACHE_BLOCKS:
            self._drop_traces()  # the streaming prefill clears the program cache between its block epochs
        prompt = tokens[0, :prompt_len].to(torch.int64)
        resume = self._session_resume_start(prompt)
        # Whatever happens below, the current snapshot belongs to the previous prompt only: a new one is taken by
        # this prefill (streaming prompts), and a failed prefill must not leave a stale one behind.
        self.model.invalidate_ring_snapshot()
        self._previous_prompt = None
        state = self.generator.prepare_prefill(
            prompt,
            prompt_lengths=[prompt_len],
            slot_ids=[0],
            resume_start=resume,
            snapshot_rings=self._session_reuse,
        )
        taps = {}
        if self.drafter is not None:
            # The drafter's context needs the prompt's last 128 positions (DSpark window); a resumed prefill finds
            # the ones before resume_start in the restored snapshot (the drafter rings are snapshotted too).
            tail = max(0, prompt_len - self.model.config.sliding_window)

            def write_prompt_context(block_start: int, rows: int, block_taps: dict) -> None:
                first = max(tail - block_start, 0)
                if first < rows:
                    self.drafter.write_context_rows(block_taps, block_start, first, rows - first)

            taps = {"tap_layers": self.drafter.target_layers, "tap_callback": write_prompt_context, "tap_from": tail}
        mesh = self.model.mesh_device
        programs = mesh.num_program_cache_entries()
        logits = self.generator.prefill_forward(state, **taps)
        if self._trace_guard and mesh.num_program_cache_entries() != programs:
            # The prefill compiled programs behind live traces (a new block shape or MoE grouping): their device
            # buffers may sit on a trace's freed intermediates, which the next replay would overwrite (a later run
            # of such a program hung the device). Release the traces before any replay; the next decode /
            # verify captures them again around the new allocations.
            print(f"[trace-guard] prefill len={prompt_len} compiled programs: traces released", flush=True)
            self._drop_traces()
        if self._session_reuse:
            self._previous_prompt = prompt.clone()
            # One line per prefill so session-continuation hits can be counted from the server log.
            print(f"[session-reuse] prefill len={prompt_len} resumed_from={resume}", flush=True)
        self._log_pool_stats(f"prefill len={prompt_len}{f' resumed_from={resume}' if resume else ''}")
        token, self._token_buffer = self.generator.sample_prefill(
            logits,
            prompt_token_ids=prompt,
            slot=0,
            sampling_mode="device",
            token_buffer=self._persistent_token_buffer,
        )
        self._active_position = prompt_len
        self._has_request = True
        if self._adapt and self._adapt_rounds + self._adapt_declines:
            print(
                f"[spec-adapt] previous request: {self._adapt_rounds} speculative rounds, {self._adapt_declines} plain "
                f"decode steps, final average {self._adapt_avg:.2f} tokens / round",
                flush=True,
            )
        self._adapt_reset()
        # A decode state kept from an earlier request still points at that request's last position.
        self._resync_pending = self.generator.decode_state is not None
        return torch.tensor([token], dtype=torch.int64)

    def decode_forward(
        self,
        *,
        tokens: torch.Tensor,
        start_pos: Any,
        sampling_params: Any = None,
        slot_remap: Any = None,
        read_from_device: bool = True,
        num_valid_drafts: Any = None,
        accepted_counts: Any = None,
        spec_mode: str | None = None,
        **_: Any,
    ):
        del read_from_device
        if not self._has_request:
            raise RuntimeError("decode requires a successful prefill for the active request")
        self._validate_slot(slot_remap, name="slot_remap")
        if spec_mode is not None:
            return self._verify_forward(tokens, start_pos, num_valid_drafts, accepted_counts, spec_mode)
        self._require_device_sampling(sampling_params)
        if tokens.numel() != 1:
            raise ValueError(f"decode requires one token, got shape {tuple(tokens.shape)}")
        position = _single_value(start_pos, name="start_pos")
        if self.generator.decode_state is not None and self._decode_key != self.generator.sampler_trace_key():
            # The kept decode step was captured with another sampler configuration (penalties, seeds, ...).
            if self._spec is not None:
                self._warm_spec(position)  # new sampler programs compile with no trace alive
            else:
                self._release_decode_trace()
                self._resync_pending = False
        if self._last_verify is not None or self._resync_pending:
            # Continue the decode trace from the token committed last: by a verify, or by a new request's prefill.
            if self._last_verify is not None:
                self._check_after_verify(position)
            token = _single_value(tokens, name="decode token")
            if self._token_buffer is None:
                self._token_buffer, _ = self.generator._new_token_buffer(token, slot=0)
            else:
                self.generator.resync_decode_inputs(token, position, token_buffer=self._token_buffer)
            self._active_position = position
            self._last_verify = None
            self._resync_pending = False
        if position != self._active_position:
            raise ValueError(f"non-sequential decode position {position}; expected {self._active_position}")
        if self.generator.decode_state is None and self._spec is not None:
            self._warm_spec(position)  # every trace, compiled with none alive (rows at position.. rewritten later)
            self._resync_pending = False
            self.generator.resync_decode_inputs(
                _single_value(tokens, name="decode token"), position, token_buffer=self._token_buffer
            )
        if self.generator.decode_state is None:
            self.generator.prepare_decode_trace(
                self._token_buffer,
                current_position=position,
                slot=0,
                sampling_mode="device",
                page_table_version="model-owned-identity",
                position_buffer=self._persistent_position_buffer,
            )
            self._decode_key = self.generator.sampler_trace_key()
        token = self.generator.decode_trace_step(read_token=True)
        assert token is not None
        self._active_position = position + 1
        self._decode_steps_since_stats += 1
        if self._pool_stats_every and self._decode_steps_since_stats >= self._pool_stats_every:
            self._log_pool_stats(f"decode pos={self._active_position}")
        return torch.tensor([token], dtype=torch.int64)

    def _check_after_verify(self, position: int) -> None:
        start, width = self._last_verify
        if not start < position <= start + width:
            raise ValueError(f"position {position} after the verify block at {start} of width {width}")

    def _verify_forward(self, tokens: Any, start_pos: Any, num_valid_drafts: Any, accepted_counts: Any, spec_mode):
        """/ S5: verify of the [1, 1 + K] candidate block; returns the argmax at every column, or (spec_mode
        "logits", a sampled row in the step) every column's full-vocabulary logits for the host's rejection sampling.

        Column 0 is the last committed token at start_pos[0, 0]; the block's rows run at consecutive positions.
        Padded draft columns (past num_valid_drafts) are computed with token 0 and never read by the accept walk.
        Every row writes its K/V and compressor state at its position: rejected rows are overwritten later before
        anything reads them (the verify path keeps no per-candidate state to roll back).
        """

        from vllm_tt_plugin.spec_decode import VerifyOutput, check_spec_side_tensors

        if spec_mode not in ("argmax_ids", "logits"):
            raise ValueError(f"DeepSeek-V4-Flash verify serves argmax_ids and logits, got spec_mode={spec_mode!r}")
        block = torch.as_tensor(tokens).reshape(1, -1).to(torch.int64)
        positions = torch.as_tensor(start_pos).reshape(1, -1)
        width = block.shape[1]
        check_spec_side_tensors(
            torch.as_tensor(num_valid_drafts), torch.as_tensor(accepted_counts), rows=1, num_drafts=width - 1
        )
        start = int(positions[0, 0])
        if self._last_verify is not None:
            self._check_after_verify(start)
        elif start != self._active_position:
            raise ValueError(f"verify block starts at {start}; expected {self._active_position}")
        valid = int(torch.as_tensor(num_valid_drafts).reshape(-1)[0])
        ids = [int(t) if j <= valid else 0 for j, t in enumerate(block[0].tolist())]
        rows = min(width, QUIETBOX2_CONTEXT - start)
        if rows < valid + 1:
            raise ValueError(f"verify columns 0..{valid} at {start} run past the {QUIETBOX2_CONTEXT}-token context")
        if rows < width:
            # The padded tail would run past the context: verify only the real columns, eagerly.
            logits, _ = self.model.verify_forward_split(ids[:rows], start)
            if spec_mode == "argmax_ids":
                greedy = self.model.verify_greedy_tokens(logits, rows) + [0] * (width - rows)
        else:
            if width not in self._verify_traces and self._spec is not None and width == self._spec[1] + 1:
                self._warm_spec(start)  # every trace, compiled with none alive (rows from start are rewritten)
            trace = self._verify_traces.get(width)
            if trace is None:
                if self.generator.decode_state is not None:
                    # The verify capture allocates long-lived inputs and constants; none may sit on the decode
                    # trace's freed intermediates. Drop the decode trace (and, unless persistent, its token
                    # buffer): the next ordinary decode recaptures it after this capture.
                    self._release_decode_trace()
                trace = self.model.capture_paged_verify_trace(
                    width,
                    warm_tokens=ids,
                    warm_start=start,
                    greedy=True,
                    tap_layers=() if self.drafter is None else self.drafter.target_layers,
                    # every row's context, written inside the trace (rejected rows are rewritten later)
                    tap_hook=None if self.drafter is None else self.drafter.write_context,
                )
                self._verify_traces[width] = trace
            logits = trace.execute(self.model.mesh_device, ids, start)
            if spec_mode == "argmax_ids":
                # The row reductions are part of the trace: a replay allocates no device buffers.
                greedy = self.model.greedy_tokens_from_reduced(*trace.greedy, width)
        self._last_verify = (start, width)
        self._active_position = None
        if spec_mode == "logits":
            host = self.model.verify_logits_for_host(logits, rows)
            if rows < width:  # columns past the context: never read by the walk
                host = torch.cat([host, host.new_zeros(width - rows, host.shape[-1])])
            return VerifyOutput(spec_mode="logits", logits=host.unsqueeze(0))
        return VerifyOutput(spec_mode="argmax_ids", argmax_ids=torch.tensor([greedy], dtype=torch.int32))

    def propose_draft_tokens(self, num_drafts, committed_tokens, committed_positions, accepted_counts, hidden=None):
        """(custom_class): DSpark drafts continuing the last committed token of the step.

        The context holds every position before it (written by the prefill, decode and verify that computed
        them), so the block is anchored at that token's position. The drafts are greedy (point masses); a sampled
        request's verify asks for logits and the host's rejection sampling keeps its distribution exact."""

        from vllm_tt_plugin.spec_decode import PLACEHOLDER_TOKEN_ID, DraftOutput

        k = int(num_drafts)
        decline = DraftOutput(
            draft_token_ids=torch.full((1, k), PLACEHOLDER_TOKEN_ID, dtype=torch.int32),
            num_valid=torch.zeros(1, dtype=torch.int32),
        )
        if self.drafter is None or not self._has_request:
            return decline
        if k != self.drafter.block:
            raise ValueError(
                f"propose_draft_tokens asked for {k} drafts; the DSpark drafter drafts {self.drafter.block}"
            )
        last = int(torch.as_tensor(accepted_counts).reshape(-1)[0]) - 1
        anchor = int(torch.as_tensor(committed_tokens).reshape(1, -1)[0, last])
        position = int(torch.as_tensor(committed_positions).reshape(1, -1)[0, last])
        if position + k >= QUIETBOX2_CONTEXT:
            return decline
        if self._adapt and not self._adapt_wants_drafts(last + 1):
            return decline  # plain decode this step
        drafts = self.drafter.draft(anchor, position)
        return DraftOutput(
            draft_token_ids=torch.tensor([drafts], dtype=torch.int32), num_valid=torch.tensor([k], dtype=torch.int32)
        )

    def _session_resume_start(self, prompt: torch.Tensor) -> int:
        """Block start to resume this prompt from, or 0 for a full prefill.

        The previous request's prefill snapshotted the rings at its last block start S, and nothing at positions
        below S has been written since (decode writes from that prompt's end onward). So a prompt that shares
        the previous prompt's first S tokens and extends past S can restore the snapshot and prefill from S.
        Thinking-mode templates drop the previous reasoning, so the shared prefix ends inside the previous
        prompt; the snapshot is therefore taken before the last block of the prompt, not at its end.
        """

        if not self._session_reuse or self._previous_prompt is None:
            return 0
        if getattr(self.model, "prefill_block_size", STREAMING_PREFILL_BLOCK_SIZE) != STREAMING_PREFILL_BLOCK_SIZE:
            return 0  # resuming needs the default 128-token blocks on both sides
        start = self.model.ring_snapshot_start(0)
        if start is None or prompt.numel() <= max(start, STREAMING_PREFILL_BLOCK_SIZE):
            return 0
        previous = self._previous_prompt
        if previous.numel() < start or not torch.equal(prompt[:start], previous[:start]):
            return 0
        self.model.restore_ring_snapshot(0)
        return start

    def _log_pool_stats(self, phase: str) -> None:
        if not self._pool_stats_every:
            return
        stats = self.model.expert_pool_stats()
        if stats is None:
            return
        last, self._pool_stats_last = self._pool_stats_last, stats
        delta = {key: value - last.get(key, 0) for key, value in stats.items()}
        hit_rate = delta["hits"] / delta["requested"] if delta["requested"] else 1.0
        print(
            f"[expert-pool] {phase} steps={self._decode_steps_since_stats} requested={delta['requested']} "
            f"loads={delta['loads']} hit_rate={hit_rate:.4f} ensure_calls={delta['ensure_calls']} "
            f"total_loads={stats['loads']}",
            flush=True,
        )
        self._decode_steps_since_stats = 0

    def read_decode_output(self, tt_out: torch.Tensor, async_read: bool = False):
        if not isinstance(tt_out, torch.Tensor):
            raise TypeError(f"expected host token tensor, got {type(tt_out)}")
        return (tt_out, []) if async_read else tt_out
