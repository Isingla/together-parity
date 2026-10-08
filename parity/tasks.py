"""Benchmark items + graders. Data files are vendored in data/ and pinned by sha256."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
SHA256 = {
    "gsm8k_test.jsonl": "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14",
    "HumanEval.jsonl.gz": "b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef",
}


@dataclass
class Item:
    task: str
    id: str
    messages: list[dict]
    gold: dict            # whatever the grader needs


def _verify(name: str) -> Path:
    p = DATA / name
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    if h != SHA256[name]:
        raise SystemExit(f"{name}: sha256 mismatch ({h}); data file changed")
    return p


def _subset(items: list[Item], n: int, seed: int) -> list[Item]:
    """Deterministic subset; n >= len returns everything in original order."""
    if n >= len(items):
        return items
    idx = sorted(random.Random(seed).sample(range(len(items)), n))
    return [items[i] for i in idx]


# ---------------------------------------------------------------- GSM8K
GSM8K_TEMPLATE = (
    "Solve the following math problem step by step. On the last line, write the final "
    "answer as 'The answer is: <number>'.\n\n{question}"
)


def load_gsm8k(n: int, seed: int) -> list[Item]:
    items = []
    with open(_verify("gsm8k_test.jsonl"), encoding="utf-8") as f:
        for i, line in enumerate(f):
            ex = json.loads(line)
            gold = normalize_number(ex["answer"].split("####")[-1])
            items.append(Item("gsm8k", f"gsm8k/{i}",
                              [{"role": "user", "content": GSM8K_TEMPLATE.format(question=ex["question"])}],
                              {"answer": gold}))
    return _subset(items, n, seed)


_NUM = r"-?\$?\s?[\d,]*\.?\d+"


def normalize_number(s: str | None) -> str | None:
    if s is None:
        return None
    s = s.strip().replace(",", "").replace("$", "").replace(" ", "").rstrip(".")
    try:
        v = float(s)
    except ValueError:
        return None
    return str(int(v)) if v == int(v) else f"{v:g}"


def extract_gsm8k(text: str) -> str | None:
    m = re.findall(r"answer is:?\s*\**\s*(" + _NUM + ")", text, flags=re.I)
    if m:
        return normalize_number(m[-1])
    nums = re.findall(_NUM, text)
    return normalize_number(nums[-1]) if nums else None


def grade_gsm8k(item: Item, text: str) -> dict:
    pred = extract_gsm8k(text)
    return {"pred": pred, "correct": pred is not None and pred == item.gold["answer"]}


# ---------------------------------------------------------------- HumanEval
HE_TEMPLATE = (
    "Complete the following Python function. Reply with the complete function "
    "(signature included, plus any imports it needs) in a single ```python code block "
    "and no explanation.\n\n```python\n{prompt}```"
)


def load_humaneval(n: int, seed: int) -> list[Item]:
    items = []
    with gzip.open(_verify("HumanEval.jsonl.gz"), "rt", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            items.append(Item("humaneval", ex["task_id"],
                              [{"role": "user", "content": HE_TEMPLATE.format(prompt=ex["prompt"])}],
                              {"prompt": ex["prompt"], "test": ex["test"],
                               "entry_point": ex["entry_point"]}))
    return _subset(items, n, seed)


def extract_code(text: str) -> str:
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.S)
    if blocks:
        return max(blocks, key=len)
    return text


def build_program(item: Item, code: str) -> str:
    g = item.gold
    # keep the prompt's imports (typing etc.) even if the model dropped them
    header = "\n".join(l for l in g["prompt"].splitlines() if l.startswith(("import ", "from ")))
    if re.search(rf"def\s+{re.escape(g['entry_point'])}\s*\(", code):
        body = header + "\n\n" + code
    else:                                 # model returned only the body
        body = g["prompt"] + code
    return f"{body}\n\n{g['test']}\n\ncheck({g['entry_point']})\n"


def run_program(program: str, timeout: float = 10.0) -> tuple[bool, str]:
    """Runs model-written code in a subprocess. Untrusted: run the harness in a
    container/VM if you care (see README)."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "prog.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(program)
        try:
            env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
            r = subprocess.run([sys.executable, "-I", path], cwd=d, capture_output=True,
                               text=True, encoding="utf-8", errors="replace",
                               timeout=timeout, env=env)
        except subprocess.TimeoutExpired:
            return False, "timeout"
        return r.returncode == 0, (r.stderr or "")[-300:]


def grade_humaneval(item: Item, text: str) -> dict:
    code = extract_code(text)
    ok, err = run_program(build_program(item, code))
    return {"pred": hashlib.sha1(code.strip().encode()).hexdigest()[:12],
            "correct": ok, "err": None if ok else err}


LOADERS = {"gsm8k": load_gsm8k, "humaneval": load_humaneval}
GRADERS = {"gsm8k": grade_gsm8k, "humaneval": grade_humaneval}
