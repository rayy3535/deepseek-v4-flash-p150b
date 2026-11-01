# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Serving stress test against a local DeepSeek-V4-Flash vLLM server: N chat requests with random prompts
(random lengths, thinking on / off), small max_tokens, streamed with a per-request deadline. A request that
produces nothing within the deadline is reported as HANG and the run stops (the server needs a reset then).
Run: python tools/stress_requests.py <N> <out.json> [long]   ("long": ~150..900-token prompts; default 1..~100)
"""

import json
import random
import socket
import sys
import time
import urllib.request

URL = "http://127.0.0.1:8010/v1/chat/completions"
MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
WORDS_EN = ("the quick brown fox jumps over a lazy dog while explaining why prime numbers matter in cryptography "
            "and how compilers turn loops into vector instructions for modern processors").split()
DEADLINE_S = 180


LONG = False  # argv[3] == "long": prompts of roughly 150..900 tokens (streaming blocks, traces kept by default)


def prompt(rng):
    lo, hi = (150, 850) if LONG else (1, 90)
    return " ".join(rng.choice(WORDS_EN) for _ in range(rng.randint(lo, hi))) + "?"


def ask(text, thinking, max_tokens):
    body = {"model": MODEL, "messages": [{"role": "user", "content": text}], "stream": True, "temperature": 0.6,
            "max_tokens": max_tokens, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"thinking": thinking}}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    t0, usage = time.perf_counter(), None
    with opener.open(req, timeout=DEADLINE_S) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if line == "data: [DONE]":
                break
            if line.startswith("data:"):
                usage = json.loads(line[5:]).get("usage") or usage
            if time.perf_counter() - t0 > DEADLINE_S:
                raise TimeoutError("deadline")
    return usage, time.perf_counter() - t0


def main():
    global LONG
    n, out = int(sys.argv[1]), sys.argv[2]
    LONG = len(sys.argv) > 3 and sys.argv[3] == "long"
    rng = random.Random(1)
    rows = []
    for i in range(n):
        text, thinking, max_tokens = prompt(rng), rng.random() < 0.3, rng.choice((16, 64, 128))
        try:
            usage, secs = ask(text, thinking, max_tokens)
        except (TimeoutError, socket.timeout, OSError) as exc:
            print(f"HANG at request {i}: {exc!r} prompt chars {len(text)} thinking {thinking}", flush=True)
            rows.append({"i": i, "hang": True})
            break
        rows.append({"i": i, "prompt_tokens": usage["prompt_tokens"], "completion_tokens": usage["completion_tokens"],
                     "seconds": round(secs, 2)})
        if i % 10 == 9:
            done = rows[-10:]
            print(f"{i + 1}/{n} ok, last 10: mean {sum(r['seconds'] for r in done) / 10:.1f} s/request", flush=True)
        json.dump(rows, open(out, "w"))
    json.dump(rows, open(out, "w"))
    print(f"STRESS_DONE {len([r for r in rows if not r.get('hang')])} ok, hang {any(r.get('hang') for r in rows)}",
          flush=True)


if __name__ == "__main__":
    main()
