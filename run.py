#!/usr/bin/env python3
"""Together AI quality-parity harness.

  python run.py check                       # ping every endpoint
  python run.py quality --endpoint reference
  python run.py quality --endpoint together
  python run.py speed   --endpoint reference
  python run.py speed   --endpoint together
  python run.py report                      # -> results/RESULTS.md + chart

Runs are resumable: finished items are skipped on re-run.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from parity.client import Client, Endpoint
from parity.tasks import GRADERS, LOADERS

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def load_cfg(path: str, profile: str | None) -> tuple[dict, dict]:
    cfg = yaml.safe_load(open(path))
    prof = profile or cfg["profile"]
    cfg["profile"] = prof
    return cfg, cfg["profiles"][prof]["endpoints"]


def endpoint(eps: dict, name: str) -> Endpoint:
    if name not in eps:
        raise SystemExit(f"unknown endpoint {name!r}; have {list(eps)}")
    return Endpoint.from_config(name, eps[name])


def read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def write_meta(cfg: dict, ep: Endpoint, kind: str, info: dict):
    p = RESULTS / cfg["profile"] / "meta.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    meta = json.loads(p.read_text()) if p.exists() else {}
    meta.setdefault("runs", []).append({
        "kind": kind, "endpoint": ep.name, "base_url": ep.base_url, "model": ep.model,
        "server_info": info, "utc": datetime.now(timezone.utc).isoformat(),
        "client_python": sys.version.split()[0], "client_platform": platform.platform(),
    })
    p.write_text(json.dumps(meta, indent=2))


# ------------------------------------------------------------------ quality
async def cmd_quality(args, cfg, eps):
    q = cfg["quality"]
    ep = endpoint(eps, args.endpoint)
    client = Client(ep)
    write_meta(cfg, ep, "quality", await client.server_info())
    tasks = args.tasks.split(",")
    try:
        for rep in range(1, (args.repeats or q["repeats"]) + 1):
            for task in tasks:
                n = args.n or q[f"{task}_n"]
                items = LOADERS[task](n, q["seed"])
                out = RESULTS / cfg["profile"] / "quality" / ep.name / f"run{rep}" / f"{task}.jsonl"
                out.parent.mkdir(parents=True, exist_ok=True)
                done = {r["id"] for r in read_jsonl(out) if not r.get("error")}
                todo = [it for it in items if it.id not in done]
                print(f"[{ep.name}] run{rep} {task}: {len(todo)}/{len(items)} to do", flush=True)
                if not todo:
                    continue
                await _run_items(client, todo, task, q, out)
                rows = {r["id"]: r for r in read_jsonl(out)}       # dedupe retries
                acc = sum(rows[i.id]["correct"] for i in items if i.id in rows) / len(items)
                print(f"[{ep.name}] run{rep} {task}: accuracy {acc:.3f}", flush=True)
    finally:
        await client.aclose()


async def _run_items(client: Client, items, task, q, out: Path):
    grade = GRADERS[task]
    max_tokens = q[f"max_tokens_{task}"]
    lock = asyncio.Lock()
    n_done = 0

    async def one(it):
        nonlocal n_done
        c = await client.chat(it.messages, max_tokens=max_tokens,
                              temperature=q["temperature"], seed=q["seed"])
        g = await asyncio.to_thread(grade, it, c.text) if not c.error else {"pred": None, "correct": False}
        row = {"id": it.id, **g, **c.to_dict()}
        async with lock:
            with open(out, "a") as f:
                f.write(json.dumps(row) + "\n")
            n_done += 1
            if n_done % 50 == 0 or n_done == len(items):
                print(f"   {n_done}/{len(items)}", flush=True)

    await asyncio.gather(*(one(it) for it in items))


# ------------------------------------------------------------------ speed
def _filler(target_tokens: int, nonce: int) -> str:
    """Natural-ish text of ~target_tokens tokens (≈4 chars/token for Llama 3 BPE).
    The body is identical across endpoints/reps (seeded by length) so both sides do
    the same work; a unique nonce on the FIRST line defeats prefix caching, so every
    request pays full prefill."""
    src = (ROOT / "data" / "gsm8k_test.jsonl").read_text().splitlines()
    rng = random.Random(target_tokens)
    rng.shuffle(src)
    text, chars = [], 0
    for line in src:
        q = json.loads(line)["question"]
        text.append(q)
        chars += len(q) + 1
        if chars >= target_tokens * 4:
            break
    return f"[request {nonce}]\n" + "\n".join(text)[: target_tokens * 4]


async def cmd_speed(args, cfg, eps):
    s = cfg["speed"]
    ep = endpoint(eps, args.endpoint)
    ep.concurrency = 1                       # latency at batch-of-one; sequential
    client = Client(ep)
    write_meta(cfg, ep, "speed", await client.server_info())
    out = RESULTS / cfg["profile"] / "speed" / f"{ep.name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    nonce = time.time_ns()
    try:
        for target in s["prompt_tokens"]:
            for rep in range(s["warmup"] + s["reps"]):
                nonce += 1
                msg = [{"role": "user", "content": _filler(target, nonce) +
                        "\n\nWrite a long, detailed commentary on the problems above."}]
                c = await client.chat(msg, max_tokens=s["max_tokens"], temperature=0.0)
                if rep < s["warmup"]:
                    continue
                row = {"target_prompt_tokens": target, "rep": rep - s["warmup"], **c.to_dict()}
                with open(out, "a") as f:
                    f.write(json.dumps(row) + "\n")
                tps = f"{c.decode_tok_s:.1f}" if c.decode_tok_s else "-"
                ttft = f"{c.ttft_s:.3f}" if c.ttft_s else "-"
                print(f"[{ep.name}] ~{target} tok rep{rep - s['warmup']}: ttft {ttft}s, "
                      f"decode {tps} tok/s {('ERR ' + c.error) if c.error else ''}", flush=True)
    finally:
        await client.aclose()


# ------------------------------------------------------------------ check
async def cmd_check(args, cfg, eps):
    for name in eps:
        try:
            ep = endpoint(eps, name)
        except SystemExit as e:
            print(f"[{name}] SKIP: {e}")
            continue
        client = Client(ep, retries=0)
        c = await client.chat([{"role": "user", "content": "Reply with the single word: ok"}],
                              max_tokens=5)
        await client.aclose()
        status = f"ERROR {c.error}" if c.error else f"{c.text!r} ttft={c.ttft_s:.3f}s usage={c.completion_tokens}"
        print(f"[{name}] {ep.model} @ {ep.base_url}: {status}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--profile", help="config profile (default from config.yaml)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    pq = sub.add_parser("quality")
    pq.add_argument("--endpoint", required=True)
    pq.add_argument("--tasks", default="gsm8k,humaneval")
    pq.add_argument("--n", type=int, help="override item count (smoke runs)")
    pq.add_argument("--repeats", type=int)
    ps = sub.add_parser("speed")
    ps.add_argument("--endpoint", required=True)
    pr = sub.add_parser("report")
    pr.add_argument("--ref", default="reference")
    pr.add_argument("--test", default="together", help="any endpoint name, e.g. a hyperscaler")
    args = ap.parse_args()
    cfg, eps = load_cfg(args.config, args.profile)

    if args.cmd == "report":
        from parity.report import build_report
        print(build_report(cfg, RESULTS / cfg["profile"], args.ref, args.test))
        return
    fn = {"quality": cmd_quality, "speed": cmd_speed, "check": cmd_check}[args.cmd]
    asyncio.run(fn(args, cfg, eps))


if __name__ == "__main__":
    main()
