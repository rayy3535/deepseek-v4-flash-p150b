#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# vLLM server for DeepSeek-V4-Flash-0731 on 4x Blackhole p150b (QSFP-DD ring, MeshShape(4, 1)).
# The tuned settings are the adapter's defaults; export any DEEPSEEK_V4_FLASH_* variable (see ../README.md) to
# override one. Required: DEEPSEEK_MODEL (local checkpoint directory) and a tt-metal build with python_env and
# vllm-tt-plugin installed into it.
set -euo pipefail

TT_METAL_ROOT=${TT_METAL_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}
: "${DEEPSEEK_MODEL:?set DEEPSEEK_MODEL to the DeepSeek-V4-Flash-0731 checkpoint directory}"
PORT=${PORT:-8010}

export TT_METAL_HOME=$TT_METAL_ROOT TT_METAL_ROOT DEEPSEEK_MODEL
export MESH_DEVICE='(4,1)'
export PYTHONPATH="$TT_METAL_ROOT/build_Release"
export EXTRA_MODELS_DIR="$TT_METAL_ROOT/models/autoports/deepseek_ai_deepseek_v4_flash_0731/vllm_models"
export VLLM_PLUGINS=tt,tt_model_registry
export DEEPSEEK_V4_FLASH_HF_MODEL="$DEEPSEEK_MODEL"
# Large-page expert loads pin every routed expert's host bytes once (~156 GB with the BFP4 cache): the pin cache must
# hold them all (default 4 GiB).
export TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=${TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES:-214748364800}
unset TT_MESH_GRAPH_DESC_PATH  # let tt-metal discover the 4-card ring

# Speculative decoding: dspark:<K> (the checkpoint's DSpark drafter on device), ngram:<K>, or empty for plain decode.
# The adapter and vLLM's --speculative-config must agree, so it is set here for both.
SPEC=${DEEPSEEK_V4_FLASH_SPEC-dspark:5}
TRACE_REGION_SIZE=268435456
L1_SMALL_SIZE=2048
SPEC_ARGS=()
if [[ -n "$SPEC" ]]; then
  [[ "$SPEC" =~ ^(ngram|dspark):([1-7])$ ]] || { echo "DEEPSEEK_V4_FLASH_SPEC must be ngram:<1-7> or dspark:<1-7>, got $SPEC" >&2; exit 2; }
  TRACE_REGION_SIZE=402653184  # decode chains + verify + drafter traces
  L1_SMALL_SIZE=8192           # the verify block's CCL semaphores
  if [[ "${BASH_REMATCH[1]}" == ngram ]]; then
    SPEC_ARGS=(--speculative-config "{\"method\":\"ngram\",\"num_speculative_tokens\":${BASH_REMATCH[2]},\"prompt_lookup_min\":2,\"prompt_lookup_max\":5}")
  else
    SPEC_ARGS=(--speculative-config "{\"method\":\"custom_class\",\"model\":\"vllm_tt_plugin.model_owned_drafter\",\"num_speculative_tokens\":${BASH_REMATCH[2]}}")
  fi
fi
export DEEPSEEK_V4_FLASH_SPEC="$SPEC"

# Device sampling and traces only (the adapter rejects host-sampling requests); a 1D ring over the four cards.
printf -v TT_ADDITIONAL_CONFIG \
  '{"tt":{"sample_on_device_mode":"all","trace_mode":"all","trace_region_size":%s,"l1_small_size":%s,"fabric_config":"FABRIC_1D_RING"}}' \
  "$TRACE_REGION_SIZE" "$L1_SMALL_SIZE"

cd "$TT_METAL_ROOT"
exec python_env/bin/vllm serve "$DEEPSEEK_MODEL" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --served-model-name deepseek-ai/DeepSeek-V4-Flash-0731 \
  --tokenizer "$DEEPSEEK_MODEL" \
  --tokenizer-mode deepseek_v4 \
  --reasoning-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --tool-call-parser deepseek_v4 \
  --trust-remote-code \
  --dtype bfloat16 \
  --max-model-len 131072 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 131072 \
  --block-size 128 \
  --additional-config "$TT_ADDITIONAL_CONFIG" \
  "${SPEC_ARGS[@]}"
