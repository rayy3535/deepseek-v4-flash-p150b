#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# vLLM server for DeepSeek-V4-Flash-0731 on 4x Blackhole p150b (QSFP-DD ring, MeshShape(4, 1)) with the tuned
# defaults described in ../README.md. Every default below can be overridden by exporting the variable first.
#
# Required: DEEPSEEK_MODEL (local checkpoint directory); a tt-metal build with python_env and vllm-tt-plugin
# installed into it. Reset the cards right before starting (one multi-card open per reset):
#   timeout 180 tt-smi -r && models/autoports/deepseek_ai_deepseek_v4_flash_0731/vllm_models/run_server_p150b_x4.sh
set -euo pipefail

TT_METAL_ROOT=${TT_METAL_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}
: "${DEEPSEEK_MODEL:?set DEEPSEEK_MODEL to the DeepSeek-V4-Flash-0731 checkpoint directory}"
PORT=${PORT:-8010}
DEEPSEEK_L1_SMALL_SIZE=${DEEPSEEK_L1_SMALL_SIZE:-2048}

export TT_METAL_HOME=$TT_METAL_ROOT TT_METAL_ROOT DEEPSEEK_MODEL
export MESH_DEVICE='(4,1)'
export TT_METAL_DISABLE_FABRIC_TWO_ERISC=1
export PYTHONPATH="$TT_METAL_ROOT/build_Release"
export EXTRA_MODELS_DIR="$TT_METAL_ROOT/models/autoports/deepseek_ai_deepseek_v4_flash_0731/vllm_models"
export VLLM_PLUGINS=tt,tt_model_registry
export DEEPSEEK_V4_FLASH_HF_MODEL="$DEEPSEEK_MODEL"
unset TT_MESH_GRAPH_DESC_PATH

# Expert paging: every routed expert in BFP4, a pool of this many experts per layer resident on device ("off" keeps
# the all-resident selective BFP2/BFP4 deployment, which needs a build with ttnn.bfloat2_b).
export DEEPSEEK_V4_FLASH_EXPERT_POOL=${DEEPSEEK_V4_FLASH_EXPERT_POOL:-176}
# Packed BFP4 routed experts are cached here (about 156 GB) so a paged load skips dequantize + pack; stale layers
# are rebuilt automatically. Empty disables the cache.
export DEEPSEEK_V4_FLASH_EXPERT_CACHE=${DEEPSEEK_V4_FLASH_EXPERT_CACHE-$HOME/.cache/deepseek_v4_flash/experts_bfp4}
# Attention projection weights: bfp8 (+1.36 GiB per device over bfp4; fixes greedy stopping on some prompts).
export DEEPSEEK_V4_FLASH_ATTENTION_DTYPE=${DEEPSEEK_V4_FLASH_ATTENTION_DTYPE:-bfp8}
# Session continuation: a prompt that extends the previous one resumes from a ring snapshot (bitwise).
export DEEPSEEK_V4_FLASH_SESSION_REUSE=${DEEPSEEK_V4_FLASH_SESSION_REUSE:-1}
# Keep the decode / verify traces across requests (bitwise).
export DEEPSEEK_V4_FLASH_PERSISTENT_TRACES=${DEEPSEEK_V4_FLASH_PERSISTENT_TRACES:-1}
# Fused mHC input-map epilogue with the official 20 Sinkhorn iterations (main model).
export DEEPSEEK_V4_FLASH_MHC_A2=${DEEPSEEK_V4_FLASH_MHC_A2:-1}
# Fusion switches for the main model (comma list; "none" for none).
export DEEPSEEK_V4_FLASH_FUSIONS=${DEEPSEEK_V4_FLASH_FUSIONS-hc_pre_map_fused,compressor_pool_fused,hc_post_fused}
if [ "$DEEPSEEK_V4_FLASH_FUSIONS" = none ]; then DEEPSEEK_V4_FLASH_FUSIONS=; fi
# Verify-block local bank padded to whole tiles.
export DEEPSEEK_V4_FLASH_VERIFY_LOCAL_PAD=${DEEPSEEK_V4_FLASH_VERIFY_LOCAL_PAD:-1}
# Indexer top-k scans only the filled prefix of the indexer bank (search length from a device tensor).
export DEEPSEEK_V4_FLASH_INDEXER_DYNAMIC=${DEEPSEEK_V4_FLASH_INDEXER_DYNAMIC:-1}
# A prefill block's MoE in as few grouped calls as the expert pool allows (bitwise).
export DEEPSEEK_V4_FLASH_PREFILL_MOE_MERGE=${DEEPSEEK_V4_FLASH_PREFILL_MOE_MERGE:-1}
# Prefill mHC input map fused and output map through the hc_post kernel.
export DEEPSEEK_V4_FLASH_PREFILL_HC=${DEEPSEEK_V4_FLASH_PREFILL_HC:-1}
# Release every trace after a prefill that compiled new programs: a program compiled behind live traces can have
# device buffers on a trace's freed intermediates, which the next replay overwrites (bitwise).
export DEEPSEEK_V4_FLASH_TRACE_GUARD=${DEEPSEEK_V4_FLASH_TRACE_GUARD:-1}
# Prompts of at most 128 tokens run as one padded 128-token block; every short length is compiled at startup.
export DEEPSEEK_V4_FLASH_PREFILL_PAD_SHORT=${DEEPSEEK_V4_FLASH_PREFILL_PAD_SHORT:-1}

# Speculative decoding: dspark:<K> (the checkpoint's DSpark drafter on device) or ngram:<K>; empty disables it.
SPEC=${DEEPSEEK_V4_FLASH_SPEC-dspark:5}
TRACE_REGION_SIZE=268435456
SPEC_ARGS=()
if [[ -n "$SPEC" ]]; then
  [[ "$SPEC" =~ ^(ngram|dspark):([1-7])$ ]] || { echo "DEEPSEEK_V4_FLASH_SPEC must be ngram:<1-7> or dspark:<1-7>, got $SPEC" >&2; exit 2; }
  TRACE_REGION_SIZE=402653184
  DEEPSEEK_L1_SMALL_SIZE=8192
  export DEEPSEEK_V4_FLASH_SPEC="$SPEC"
  if [[ "${BASH_REMATCH[1]}" == ngram ]]; then
    SPEC_ARGS=(--speculative-config "{\"method\":\"ngram\",\"num_speculative_tokens\":${BASH_REMATCH[2]},\"prompt_lookup_min\":2,\"prompt_lookup_max\":5}")
  else
    SPEC_ARGS=(--speculative-config "{\"method\":\"custom_class\",\"model\":\"vllm_tt_plugin.model_owned_drafter\",\"num_speculative_tokens\":${BASH_REMATCH[2]}}")
  fi
else
  export DEEPSEEK_V4_FLASH_SPEC=
fi
# The plugin warmup reaches the adapter's warmup_model_prefill, which compiles the padded short prefill before any
# trace exists; without pad-short it does nothing, so it stays off then.
MODEL_WARMUP=false
[[ "$DEEPSEEK_V4_FLASH_PREFILL_PAD_SHORT" == 1 ]] && MODEL_WARMUP=true
printf -v TT_ADDITIONAL_CONFIG \
  '{"tt":{"sample_on_device_mode":"all","trace_mode":"all","enable_model_warmup":%s,"trace_region_size":%s,"l1_small_size":%s,"fabric_config":"FABRIC_1D_RING","fabric_reliability_mode":"RELAXED_INIT"}}' \
  "$MODEL_WARMUP" "$TRACE_REGION_SIZE" "$DEEPSEEK_L1_SMALL_SIZE"

env | grep -E '^DEEPSEEK_V4_FLASH_' | sort
cd "$TT_METAL_ROOT"
exec python_env/bin/vllm serve "$DEEPSEEK_MODEL" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --served-model-name deepseek-ai/DeepSeek-V4-Flash-0731 \
  --tokenizer "$DEEPSEEK_MODEL" \
  --tokenizer-mode deepseek_v4 \
  --reasoning-parser deepseek_v4 \
  --trust-remote-code \
  --dtype bfloat16 \
  --max-model-len 131072 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 131072 \
  --block-size 128 \
  --additional-config "$TT_ADDITIONAL_CONFIG" \
  "${SPEC_ARGS[@]}"
