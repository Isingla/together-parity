"""OpenAI-compatible streaming client that records timing per request."""
from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, asdict

import httpx


@dataclass
class Endpoint:
    name: str
    base_url: str
    model: str
    api_key: str | None = None
    concurrency: int = 8

    @classmethod
    def from_config(cls, name: str, cfg: dict) -> "Endpoint":
        base = os.path.expandvars(cfg["base_url"])
        if "${" in base:
            raise SystemExit(f"[{name}] base_url env var not set: {cfg['base_url']}")
        key_env = cfg.get("api_key_env")
        return cls(name=name, base_url=base.rstrip("/"), model=cfg["model"],
                   api_key=os.environ.get(key_env) if key_env else None,
                   concurrency=int(cfg.get("concurrency", 8)))


@dataclass
class Completion:
    text: str
    ttft_s: float | None          # time to first content token
    e2e_s: float                  # total request time
    prompt_tokens: int | None
    completion_tokens: int | None
    finish_reason: str | None
    n_chunks: int                 # content-bearing chunks (fallback token count)
    error: str | None = None

    @property
    def decode_tok_s(self) -> float | None:
        """Tokens/sec after the first token (excludes prefill + network RTT)."""
        n = self.completion_tokens or self.n_chunks
        if self.ttft_s is None or n is None or n < 2:
            return None
        dt = self.e2e_s - self.ttft_s
        return (n - 1) / dt if dt > 0 else None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["decode_tok_s"] = self.decode_tok_s
        return d


class Client:
    def __init__(self, ep: Endpoint, timeout: float = 600.0, retries: int = 4):
        self.ep = ep
        self.retries = retries
        headers = {"Content-Type": "application/json"}
        if ep.api_key:
            headers["Authorization"] = f"Bearer {ep.api_key}"
        self._http = httpx.AsyncClient(base_url=ep.base_url, headers=headers,
                                       timeout=httpx.Timeout(timeout, connect=30.0))
        self.sem = asyncio.Semaphore(ep.concurrency)

    async def aclose(self):
        await self._http.aclose()

    async def server_info(self) -> dict:
        """Best-effort provenance: vLLM exposes /version; everyone exposes /models."""
        info = {}
        root = self.ep.base_url.rsplit("/v1", 1)[0]
        for path in (f"{root}/version", "/models"):
            try:
                r = await self._http.get(path)
                if r.status_code == 200:
                    info[path] = r.json()
            except Exception:  # noqa: BLE001 - provenance is optional
                pass
        return info

    async def chat(self, messages: list[dict], *, max_tokens: int, temperature: float = 0.0,
                   seed: int | None = None) -> Completion:
        body = {
            "model": self.ep.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 1.0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if seed is not None:
            body["seed"] = seed
        last_err = None
        async with self.sem:
            for attempt in range(self.retries + 1):
                try:
                    return await self._stream(body)
                except FatalStatus as e:
                    last_err = str(e)
                    break
                except (httpx.HTTPError, RetryableStatus, json.JSONDecodeError) as e:
                    last_err = f"{type(e).__name__}: {e}"
                    await asyncio.sleep(min(60, 2 ** attempt))
        return Completion("", None, 0.0, None, None, None, 0, error=last_err)

    async def _stream(self, body: dict) -> Completion:
        parts: list[str] = []
        ttft = None
        usage = {}
        finish = None
        n_chunks = 0
        t0 = time.perf_counter()
        async with self._http.stream("POST", "/chat/completions", json=body) as r:
            if r.status_code in (429, 500, 502, 503, 504):
                await r.aread()
                raise RetryableStatus(f"HTTP {r.status_code}: {r.text[:200]}")
            if r.status_code != 200:
                await r.aread()
                raise FatalStatus(f"HTTP {r.status_code}: {r.text[:300]}")
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for ch in chunk.get("choices") or []:
                    delta = (ch.get("delta") or {}).get("content")
                    if delta:
                        if ttft is None:
                            ttft = time.perf_counter() - t0
                        parts.append(delta)
                        n_chunks += 1
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
        e2e = time.perf_counter() - t0
        return Completion("".join(parts), ttft, e2e, usage.get("prompt_tokens"),
                          usage.get("completion_tokens"), finish, n_chunks)


class RetryableStatus(Exception):
    pass


class FatalStatus(Exception):
    pass
