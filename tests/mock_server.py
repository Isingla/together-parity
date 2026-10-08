"""Fake OpenAI-compatible streaming server for end-to-end tests (no GPU, no API key).

python tests/mock_server.py --port 9001 --acc 0.85 --flip-seed 0 --tok-s 40 --ttft 0.05
"""
import argparse
import gzip
import hashlib
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
GSM = {}
for line in open(DATA / "gsm8k_test.jsonl"):
    ex = json.loads(line)
    GSM[ex["question"]] = ex["answer"].split("####")[-1].strip().replace(",", "")
HE = {}
with gzip.open(DATA / "HumanEval.jsonl.gz", "rt") as f:
    for line in f:
        ex = json.loads(line)
        HE[ex["prompt"]] = ex


def u(key: str, salt: str) -> float:
    return int(hashlib.sha256(f"{salt}|{key}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def answer(prompt: str, a) -> str:
    for q, gold in GSM.items():
        if q in prompt:
            # base correctness shared by both servers; `flip` perturbs a few items
            ok = u(q, "base") < a.acc
            if u(q, f"flip{a.flip_seed}") < a.flip:
                ok = not ok
            ans = gold if ok else str(int(float(gold)) + 1)
            return f"Let me work through it step by step.\nSo the total is {ans}.\nThe answer is: {ans}"
    for p, ex in HE.items():
        if p in prompt:
            ok = u(p, "base") < a.acc
            if u(p, f"flip{a.flip_seed}") < a.flip:
                ok = not ok
            body = ex["canonical_solution"] if ok else "    return None\n"
            return f"```python\n{p}{body}```"
    return "ok " + "word " * 300


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        body = json.dumps({"version": "mock-0", "data": [{"id": A.model}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        prompt = req["messages"][-1]["content"]
        text = answer(prompt, A)
        words = text.split(" ")[: req["max_tokens"]]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        time.sleep(A.ttft + len(prompt) / 4 / A.prefill)
        for i, w in enumerate(words):
            piece = w if i == 0 else " " + w
            chunk = {"choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]}
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()
            time.sleep(1 / A.tok_s)
        fin = {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
        usage = {"choices": [], "usage": {"prompt_tokens": len(prompt) // 4,
                                          "completion_tokens": len(words)}}
        for c in (fin, usage):
            self.wfile.write(f"data: {json.dumps(c)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", default="mock")
    ap.add_argument("--acc", type=float, default=0.85)
    ap.add_argument("--flip", type=float, default=0.02)
    ap.add_argument("--flip-seed", type=int, default=0)
    ap.add_argument("--tok-s", type=float, default=200)
    ap.add_argument("--ttft", type=float, default=0.02)
    ap.add_argument("--prefill", type=float, default=50000, help="prompt tok/s")
    A = ap.parse_args()
    ThreadingHTTPServer(("127.0.0.1", A.port), H).serve_forever()
