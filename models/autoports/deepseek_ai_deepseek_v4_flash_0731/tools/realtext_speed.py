# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Real-text serving speed against a local DeepSeek-V4-Flash vLLM server: a fixed set of chat requests, greedy
(temperature 0, so every server arm generates the same text), streamed; per request the prompt tokens, TTFT, completion
tokens and decode ms/token (first token excluded); a summary per group (en / qa_short / en_think).
Run: python tools/realtext_speed.py <out.json>     (the server at 127.0.0.1:8010)
"""

import json
import sys
import time
import urllib.request

URL = "http://127.0.0.1:8010/v1/chat/completions"
MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
REQUESTS = [
    ("en", False, 512, "Hello! Briefly introduce yourself."),
    ("en", False, 1024, "Write a Python function that parses an ISO 8601 date string without using datetime, with tests."),
    ("en", False, 1024, "Explain how a hash map works, including collision handling and resizing."),
    ("qa_short", False, 128, "What is the capital of France?"),
    ("qa_short", False, 128, "How many meters are in a kilometer?"),
    ("qa_short", False, 128, "What is the boiling point of water at sea level in Celsius?"),
    ("qa_short", False, 128, "Who wrote Pride and Prejudice?"),
    ("en_think", True, 2048, "How many positive integers below 1000 are divisible by 3 or 5 but not by 15?"),
    ("en_think", True, 2048, "A train leaves at 9:40 and arrives at 13:05. How long is the trip in minutes? Explain."),
]


def ask(text, thinking, max_tokens):
    body = {"model": MODEL, "messages": [{"role": "user", "content": text}], "stream": True, "temperature": 0,
            "max_tokens": max_tokens, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"thinking": thinking}}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    t0, first, usage, out = time.perf_counter(), None, None, []
    with opener.open(req, timeout=3600) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if line == "data: [DONE]":
                break
            if not line.startswith("data:"):
                continue
            data = json.loads(line[5:])
            usage = data.get("usage") or usage
            for choice in data.get("choices", []):
                delta = choice.get("delta", {})
                piece = (delta.get("reasoning") or "") + (delta.get("content") or "")
                if piece:
                    first = first or time.perf_counter()
                    out.append(piece)
    end = time.perf_counter()
    n = usage["completion_tokens"]
    return {"prompt_tokens": usage["prompt_tokens"], "completion_tokens": n, "ttft_s": round(first - t0, 3),
            "ms_per_token": round((end - first) * 1e3 / max(n - 1, 1), 1), "total_s": round(end - t0, 2),
            "text_head": "".join(out)[:80]}


def main():
    ask("Hi", False, 16)  # warm-up (first-request trace captures), not counted
    rows = []
    for group, thinking, max_tokens, text in REQUESTS:
        r = {"group": group, **ask(text, thinking, max_tokens)}
        rows.append(r)
        print(f"{group:9s} prompt {r['prompt_tokens']:4d} ttft {r['ttft_s']:6.2f} s  {r['completion_tokens']:5d} tok "
              f"{r['ms_per_token']:6.1f} ms/token  total {r['total_s']:6.1f} s", flush=True)
    summary = {}
    for group in dict.fromkeys(r["group"] for r in rows):
        g = [r for r in rows if r["group"] == group]
        tokens = sum(r["completion_tokens"] - 1 for r in g)
        decode = sum(r["total_s"] - r["ttft_s"] for r in g)
        summary[group] = {"ms_per_token": round(decode * 1e3 / tokens, 1),
                          "ttft_mean_s": round(sum(r["ttft_s"] for r in g) / len(g), 2),
                          "total_s": round(sum(r["total_s"] for r in g), 1)}
        print(f"SUMMARY {group}: {summary[group]}", flush=True)
    json.dump({"rows": rows, "summary": summary}, open(sys.argv[1], "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
