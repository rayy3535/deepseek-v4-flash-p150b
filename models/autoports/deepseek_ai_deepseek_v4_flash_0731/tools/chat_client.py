# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Interactive streaming chat against a local DeepSeek-V4-Flash vLLM server (see vllm_models/run_server_p150b_x4.sh).

Prints tokens as they arrive, then the time to first token, decode speed (tokens/s and ms/token) and the
total. Multi-turn: the conversation is kept, so later turns reuse the previous prompt (session continuation).
Commands: /new (forget the history), /think (toggle thinking, default off), /q (quit).
Line editing (readline): Backspace / Delete, Left / Right, Ctrl-A / Ctrl-E (start / end), Ctrl-U / Ctrl-K (delete to
start / end), Ctrl-W (delete word), Up / Down (previous inputs, kept in ~/.deepseek_v4_flash_chat_history), Ctrl-R (search).
Run: python3 tools/chat_client.py   [--url http://127.0.0.1:8010] [--temp 0.6] [--max N]  (default max_tokens 4096, 16384 with thinking)
"""

import argparse
import atexit
import json
import os
import time
import urllib.request

try:
    import readline  # noqa: F401  (importing it gives input() line editing and history)
except ImportError:  # e.g. a Python without GNU readline: plain input()
    readline = None

HISTORY = os.path.expanduser("~/.deepseek_v4_flash_chat_history")

URL = "http://127.0.0.1:8010/v1/chat/completions"  # --url overrides the server part
MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"


def ask(messages, temp, max_tokens, thinking):
    body = {"model": MODEL, "messages": messages, "stream": True, "temperature": temp, "max_tokens": max_tokens,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"thinking": thinking}}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # local server: never via the proxy
    t0, first, chunks, text, usage, finish = time.perf_counter(), None, 0, [], None, None
    with opener.open(req, timeout=3600) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            data = json.loads(line[5:])
            usage = data.get("usage") or usage
            for choice in data.get("choices", []):
                finish = choice.get("finish_reason") or finish
                delta = choice.get("delta", {})
                thought = delta.get("reasoning") or delta.get("reasoning_content") or ""
                answer = delta.get("content") or ""
                if thought or answer:
                    if first is None:
                        first = time.perf_counter()
                    chunks += 1
                    text.append(answer)
                    if thought:
                        print(f"\033[2m{thought}\033[0m", end="", flush=True)  # thinking, dimmed
                    print(answer, end="", flush=True)
    end = time.perf_counter()
    n = (usage or {}).get("completion_tokens") or chunks
    prompt = (usage or {}).get("prompt_tokens")
    print()
    if finish == "length":
        print(f"\033[33m[stopped at max_tokens = {max_tokens}: the answer was cut off; raise it with --max]\033[0m")
    if first is not None and n > 1:
        dec = end - first
        print(f"\033[2m[prompt {prompt} tok · TTFT {first - t0:.2f} s · {n} tok in {dec:.1f} s = "
              f"{(n - 1) / dec:.1f} tok/s ({dec * 1e3 / (n - 1):.0f} ms/token) · total {end - t0:.1f} s]\033[0m")
    return "".join(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8010", help="server base URL")
    ap.add_argument("--temp", type=float, default=0.6)
    ap.add_argument("--max", type=int, default=0, help="max_tokens (default 4096, or 16384 with thinking on)")
    args = ap.parse_args()
    global URL
    URL = args.url.rstrip("/") + "/v1/chat/completions"
    messages, thinking = [], False
    if readline is not None:
        try:
            readline.read_history_file(HISTORY)
        except OSError:
            pass
        readline.set_history_length(1000)
        atexit.register(readline.write_history_file, HISTORY)
    print("DeepSeek-V4-Flash on 4x Blackhole. /new resets, /think toggles thinking, /q quits.")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q in ("/q", "/quit"):
            break
        if q == "/new":
            messages = []
            print("(history cleared)")
            continue
        if q == "/think":
            thinking = not thinking
            print(f"(thinking {'on' if thinking else 'off'})")
            continue
        if not q:
            continue
        messages.append({"role": "user", "content": q})
        try:
            messages.append({"role": "assistant", "content": ask(messages, args.temp, args.max or (16384 if thinking else 4096), thinking)})
        except KeyboardInterrupt:
            messages.pop()
            print("\n(interrupted)")
        except Exception as exc:  # noqa: BLE001
            messages.pop()
            print(f"(request failed: {exc}; is the server up? curl --noproxy '*' <url>/health)")


if __name__ == "__main__":
    main()
