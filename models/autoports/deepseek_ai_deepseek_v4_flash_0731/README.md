# DeepSeek-V4-Flash-0731 on 4x Blackhole p150b (TTNN + vLLM)

This directory runs [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
(all 43 layers, all 256 routed experts, 131,072-token context, batch 1) on four Tenstorrent Blackhole p150b cards
behind a vLLM OpenAI-compatible server. It builds on [ndalyTT's TTNN autoport](https://github.com/ndalyTT/tt-metal) for QuietBox2 (4x p300c) and adds what
the p150b box needed and what made it fast:

- **Host-paged routed experts in BFP4.** Every routed expert is stored in BFP4 (no BFP2); a pool of 176 experts per
  layer is resident on device and the rest are paged in from host memory per prefill block / decode step. A packed
  expert cache on disk (about 156 GB) makes later loads take about two minutes.
- **DSpark speculative decoding** with the checkpoint's own 3-layer drafter (`mtp.*`) on device, 5 drafts per step,
  lossless for greedy and sampled requests (vLLM rejection sampling on the verify logits).
- **Fused kernels** for the mHC input map (with the official 20 Sinkhorn iterations), the mHC output map, the
  compressor pooling and the cache-row updates, plus a merged prefill MoE, padded short-prompt prefill, session
  continuation (a prompt that extends the previous one only prefills the new blocks) and persistent traces.
- A **trace guard**: when a prefill compiles new programs while decode / verify traces are alive, every trace is
  released and recaptured before the next replay, so the new programs' device buffers cannot be overwritten by a
  replay (this was a reproducible server hang on long prompts of new lengths).

## Hardware and software

| Item | Version used |
| --- | --- |
| Cards | 4x Blackhole p150b, QSFP-DD ring 1-2-3-4-1 with two cables per hop (`p150_x4` mesh, `MeshShape(4, 1)`) |
| Firmware | fw bundle 19.17.1 (Ethernet FW 1.13.1), flashed with tt-flash 4.2.0 |
| KMD | tt-kmd 2.11.0 |
| Host | Ubuntu 22.04, kernel 6.8, 249 GB RAM (the expert pool pages from host memory) |
| Python | 3.12 |
| tt-metal | this branch (upstream main plus one core patch, see below) |
| vLLM | 0.30.0 with [vllm-tt-plugin](https://github.com/tenstorrent/vllm-tt-plugin) main |
| Checkpoint | `deepseek-ai/DeepSeek-V4-Flash-0731` revision `7872f01b1d1fe23eabc4c98b48bffcef5a386062` |

The only change outside this directory is `ttnn.copy_host_to_device_tensor_region` (a write of a host tensor into a
byte range of a device tensor), which the expert pool uses to fill one slot without rewriting the whole pool.

## Setup

```bash
# 1. tt-metal (this branch) with its own venv
git clone -b deepseek-v4-flash-p150b https://github.com/rayy3535/tt-metal.git && cd tt-metal
git submodule update --init --recursive
./create_venv.sh --python-version 3.12 && source python_env/bin/activate
./build_metal.sh --build-type Release

# 2. vLLM 0.30.0 + the TT plugin into the same venv
git clone https://github.com/tenstorrent/vllm-tt-plugin.git ../vllm-tt-plugin
(cd ../vllm-tt-plugin && source docs/install-vllm-tt.sh)

# 3. weights
huggingface-cli download deepseek-ai/DeepSeek-V4-Flash-0731 --revision 7872f01b1d1fe23eabc4c98b48bffcef5a386062 \
    --local-dir /path/to/DeepSeek-V4-Flash-0731
```

## Run

```bash
export DEEPSEEK_MODEL=/path/to/DeepSeek-V4-Flash-0731
models/autoports/deepseek_ai_deepseek_v4_flash_0731/vllm_models/run_server_p150b_x4.sh
```

- The fabric runs with the upstream defaults (two-ERISC routers, strict link initialization). With firmware 19.17.1
  the 4-card mesh opens repeatedly without a reset; after a hang or a crashed process, `timeout 180 tt-smi -r` first.
- The first start builds the packed expert cache (`DEEPSEEK_V4_FLASH_EXPERT_CACHE`, default
  `~/.cache/deepseek_v4_flash/experts_bfp4`, about 156 GB) and takes much longer; later starts take a few minutes
  (about 3 of them compiling the short-prompt prefill programs).
- The server listens on `127.0.0.1:8010` and serves `deepseek-ai/DeepSeek-V4-Flash-0731`. Thinking is controlled by
  `chat_template_kwargs: {"thinking": true}`.
- Only device sampling is supported: requests with `logprobs`, `min_tokens`, structured outputs (including a forced
  `tool_choice`), `min_p`, `logit_bias` or logits processors are rejected with an error.

Tools: `tools/chat_client.py` (interactive streaming chat with speed readout), `tools/realtext_speed.py` (fixed
request set, ms/token per group), `tools/stress_requests.py` (N random requests with a hang deadline; `long` for
150..900-token prompts).

## Coding agents

The server parses tool calls (`--enable-auto-tool-choice --tool-call-parser deepseek_v4`), so OpenAI-compatible
agents work against it. Example [opencode](https://github.com/anomalyco/opencode) configuration
(`opencode.json`):

```json
{
  "provider": {
    "dsv4": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "DeepSeek-V4-Flash (local)",
      "options": { "baseURL": "http://127.0.0.1:8010/v1", "apiKey": "local" },
      "models": {
        "deepseek-ai/DeepSeek-V4-Flash-0731": {
          "tool_call": true,
          "reasoning": true,
          "limit": { "context": 131072, "output": 16384 }
        }
      }
    }
  },
  "model": "dsv4/deepseek-ai/DeepSeek-V4-Flash-0731"
}
```

The first request of a session prefills the agent's system prompt and tool definitions (often 10-20k tokens, tens of
seconds); later turns only prefill what was added (session continuation).

## Switches

All are environment variables read by the adapter. The defaults below are the adapter's own (tuned for this box), so
nothing needs to be set; `run_server_p150b_x4.sh` only sets `DEEPSEEK_V4_FLASH_SPEC` (it must match vLLM's
`--speculative-config`).

| Variable | Default | Meaning |
| --- | --- | --- |
| `DEEPSEEK_V4_FLASH_SPEC` | `dspark:5` | `dspark:<K>` (on-device DSpark drafter), `ngram:<K>`, or empty for plain decode |
| `DEEPSEEK_V4_FLASH_EXPERT_POOL` | `176` | routed experts resident per layer (all BFP4, host-paged); `off` = all-resident selective BFP2/BFP4 (needs a build with BFP2) |
| `DEEPSEEK_V4_FLASH_EXPERT_CACHE` | `~/.cache/deepseek_v4_flash/experts_bfp4` | packed expert cache directory; empty disables it |
| `DEEPSEEK_V4_FLASH_ATTENTION_DTYPE` | `bfp8` | attention projection weights (`bfp4` saves 1.36 GiB per device) |
| `DEEPSEEK_V4_FLASH_SHARED_DTYPE` | `bfp8` | shared-expert weights (`bfp4` saves 0.13 GiB per device) |
| `DEEPSEEK_V4_FLASH_SESSION_REUSE` | `1` | resume a prompt that extends the previous one from a ring snapshot (bitwise) |
| `DEEPSEEK_V4_FLASH_PERSISTENT_TRACES` | `1` | keep decode / verify traces across requests (bitwise) |
| `DEEPSEEK_V4_FLASH_MHC_A2` | `1` | fused mHC input-map epilogue with 20 Sinkhorn iterations |
| `DEEPSEEK_V4_FLASH_FUSIONS` | `hc_pre_map_fused,compressor_pool_fused,hc_post_fused` | fused kernels for the main model (`none` disables) |
| `DEEPSEEK_V4_FLASH_VERIFY_LOCAL_PAD` | `1` | verify-block local bank padded to whole tiles |
| `DEEPSEEK_V4_FLASH_INDEXER_DYNAMIC` | `1` | indexer top-k scans only the filled prefix of the bank |
| `DEEPSEEK_V4_FLASH_PREFILL_MOE_MERGE` | `1` | prefill MoE over a whole block in as few grouped calls as the pool allows (bitwise) |
| `DEEPSEEK_V4_FLASH_PREFILL_HC` | `1` | fused prefill mHC input / output maps |
| `DEEPSEEK_V4_FLASH_PREFILL_PAD_SHORT` | `1` | prompts of at most 128 tokens run as one padded 128-token block, compiled at startup |
| `DEEPSEEK_V4_FLASH_TRACE_GUARD` | `1` | release all traces after a prefill that compiled new programs (bitwise) |
| `DEEPSEEK_V4_FLASH_SPEC_ADAPT` | `0` | decline drafts while the accepted-tokens average is low (measured slower; off) |

Switches marked bitwise give identical outputs on and off; the others were accepted on determinism (30 runs
bit-identical), a 16-prompt accuracy set against the official CPU model (paired KL confidence interval not worse),
greedy and sampled stop rates and serving speed.

## Measured (this branch, firmware 19.17.1, defaults)

Real-text serving speed, `tools/realtext_speed.py`-style request sets, temperature 0, ms per output token after the
first:

| Request type | ms / token | TTFT |
| --- | --- | --- |
| English, thinking off | 47.5 | 0.9 s |
| English, thinking on | 41.7 | 1.1 s |
| Short questions (answers of a few tokens) | 69.6 | 0.8 s |
| Chinese, thinking off | 65.2 | 0.8 s |

Speculation helps most on English and reasoning text (DSpark accepts about 4-4.5 tokens per step there); on Chinese
chat it is about break-even with plain decode.

Full prefill (session continuation off), single request, then decode with the defaults:

| Prompt tokens | TTFT | TPOT |
| --- | --- | --- |
| 4,096 | 26.7 s | 30.8 ms |
| 16,384 | 108.5 s | 40.2 ms |
| 32,768 | 220.4 s | 39.9 ms |

With session continuation on (the default), a follow-up turn only prefills the blocks after the shared prefix.

Accuracy against the official CPU implementation (16 prompts x 109 teacher-forced steps): decode top-1 1615/1744,
mean KL 0.092; verify top-1 1618/1744, mean KL 0.094.

Stability: 150 long (150-900 tokens) and 150 short random requests with no hang; determinism 30/30 bit-identical.

## Keeping up with upstream

The branch is upstream `main` plus two commits: the core region-write patch and this directory. To move to a newer
upstream:

```bash
git remote add upstream https://github.com/tenstorrent/tt-metal.git   # once
git fetch upstream && git rebase upstream/main     # usually conflict-free: one small core patch
git submodule update --init --recursive
./build_metal.sh --build-type Release
```

then check the server with `tools/realtext_speed.py` and a `tools/stress_requests.py` run (short and long) before
switching over. Upstream API
removals show up as Python import errors or as JIT compile errors of the kernels under `tt/kernels/`.

## Layout

- `tt/`: model (`model.py`, `multichip_decoder.py`, `dspark.py`, `expert_pool.py`, ...) and custom kernels (`kernels/`)
- `vllm_models/`: vLLM adapter and `run_server_p150b_x4.sh`
- `tools/`: `chat_client.py`, `realtext_speed.py`, `stress_requests.py`
- `config.py`, `checkpoint.py`: model configuration and checkpoint reader

Only what serving needs is included; ndalyTT's bring-up tests, CPU references and quantization studies are in the
original repository.

## Credits

Based on ndalyTT's DeepSeek-V4-Flash TTNN autoport for QuietBox2 ([ndalyTT/tt-metal](https://github.com/ndalyTT/tt-metal),
commit `d788873210`; hand-off notes in [ndalyTT/deepseek-v4-flash-quietbox2](https://github.com/ndalyTT/deepseek-v4-flash-quietbox2)):
functional through optimized multichip decoders, quantization study, full model and vLLM adapter. File copyright
and license headers are kept as in the original. The p150b deployment, expert paging, DSpark speculation and the
performance work were done on top of it.
