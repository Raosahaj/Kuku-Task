"""Model adapter: one function, any provider. Swap with --model provider:name.

  gemini:gemini-2.5-flash        (GEMINI_API_KEY, free tier)
  groq:llama-3.3-70b-versatile   (GROQ_API_KEY, free tier)
  ollama:qwen2.5:14b             (local, free, OLLAMA_HOST optional)
  openai:gpt-4o-mini             (OPENAI_API_KEY; OPENAI_BASE_URL for any
                                  OpenAI-compatible endpoint, e.g. one Kuku provides)
  anthropic:claude-sonnet-4-5    (ANTHROPIC_API_KEY)
  mock                           (offline, deterministic; for tests only)

Plain HTTP via `requests` so there are no SDK version issues. Every response is
cached on disk by (model, prompt) hash, so re-runs are free and reproducible.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path

import requests

CACHE = Path(os.environ.get("PROMO_CACHE", ".cache/llm"))
# Longest provider-requested pause we will sit through before giving up (seconds).
# Raise it (e.g. PROMO_MAX_WAIT=3000) to let a long run wait out a rolling token budget.
MAX_WAIT = float(os.environ.get("PROMO_MAX_WAIT", "300"))


class LLM:
    def __init__(self, spec: str, temperature: float = 0.2, use_cache: bool = True,
                 keys: dict | None = None):
        self.provider, _, self.model = spec.partition(":")
        self.spec, self.temperature, self.use_cache = spec, temperature, use_cache
        self.keys = keys or {}   # per-run API keys (web app); falls back to the environment
        self.calls = self.cache_hits = self.in_tokens = self.out_tokens = 0
        self.max_wait = MAX_WAIT
        self.last_spec = spec

    # ------------------------------------------------------------ public ---
    def json(self, system: str, user: str, retries: int = 2):
        """Return parsed JSON. Retries once with the parse error if the model
        returns malformed JSON."""
        msg = user
        for attempt in range(retries + 1):
            raw = self.complete(system, msg)
            try:
                return extract_json(raw)
            except ValueError as e:
                if attempt == retries:
                    raise
                msg = (user + f"\n\nYour previous reply was not valid JSON ({e}). "
                       "Reply with ONLY the JSON object.")

    def complete(self, system: str, user: str) -> str:
        path = self._path(system, user)
        if self.use_cache and path.exists():
            self.cache_hits += 1
            return json.loads(path.read_text(encoding="utf-8"))["text"]
        text, tin, tout = self._with_backoff(system, user)
        self.calls += 1
        self.in_tokens += tin
        self.out_tokens += tout
        CACHE.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"model": self.spec, "text": text, "in": tin, "out": tout}), encoding="utf-8")
        return text

    def cached(self, system: str, user: str) -> bool:
        return self.use_cache and self._path(system, user).exists()

    def _path(self, system: str, user: str) -> Path:
        key = hashlib.sha256(f"{self.spec}|{self.temperature}|{system}|{user}".encode()).hexdigest()
        return CACHE / f"{key}.json"

    def usage(self) -> dict:
        return {"model": self.spec, "api_calls": self.calls, "cache_hits": self.cache_hits,
                "input_tokens": self.in_tokens, "output_tokens": self.out_tokens}

    def usages(self) -> list[dict]:
        return [self.usage()]

    def _key(self, name: str) -> str:
        v = self.keys.get(name) or os.environ.get(name)
        if not v:
            raise MissingKey(f"Set {name} to use {self.spec} (see README).")
        return v


    # ----------------------------------------------------------- internal ---
    def _with_backoff(self, system, user, tries=8):
        delay = 5.0
        for i in range(tries):
            try:
                return self._call(system, user)
            except requests.HTTPError as e:
                code = e.response.status_code if e.response is not None else 0
                if code in (429, 500, 502, 503, 504) and i < tries - 1:
                    wait = _server_wait(e.response)
                    if wait is not None and wait > self.max_wait:   # a long quota window: fail fast
                        raise QuotaExhausted(
                            f"{self.spec}: provider quota exhausted, resets in {wait / 3600:.1f}h. "
                            "Use another --model or wait.") from e
                    time.sleep(wait + 1 if wait is not None else delay)  # free tiers rate-limit hard
                    delay = min(delay * 2, 60)
                    continue
                raise

    def _call(self, system: str, user: str) -> tuple[str, int, int]:
        p, m, T = self.provider, self.model, self.temperature
        if p == "mock":
            return _mock(user), len(system + user) // 4, 200
        if p == "gemini":
            # key goes in a header, never the URL: URLs end up in error messages and logs
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent"
            body = {"systemInstruction": {"parts": [{"text": system}]},
                    "contents": [{"role": "user", "parts": [{"text": user}]}],
                    "generationConfig": {"temperature": T, "responseMimeType": "application/json"}}
            r = _post(url, body, {"x-goog-api-key": self._key("GEMINI_API_KEY")})
            cands = r.get("candidates") or []
            parts = (cands[0].get("content") or {}).get("parts") if cands else None
            if not parts:   # happens intermittently on preview models; say why instead of KeyError
                why = (cands[0].get("finishReason") if cands else None) or r.get("promptFeedback")
                raise ValueError(f"gemini returned no text (reason: {why})")
            text = "".join(pt.get("text", "") for pt in parts)
            u = r.get("usageMetadata", {})
            return text, u.get("promptTokenCount", 0), u.get("candidatesTokenCount", 0)
        if p == "anthropic":
            r = _post("https://api.anthropic.com/v1/messages",
                      {"model": m, "max_tokens": 4096, "temperature": T, "system": system,
                       "messages": [{"role": "user", "content": user}]},
                      {"x-api-key": self._key("ANTHROPIC_API_KEY"), "anthropic-version": "2023-06-01"})
            u = r.get("usage", {})
            return r["content"][0]["text"], u.get("input_tokens", 0), u.get("output_tokens", 0)
        if p == "ollama":
            host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
            r = _post(f"{host}/api/chat", {"model": m, "stream": False, "format": "json",
                                           "options": {"temperature": T, "num_ctx": 16384},
                                           "messages": [{"role": "system", "content": system},
                                                        {"role": "user", "content": user}]},
                      timeout=600)
            return r["message"]["content"], r.get("prompt_eval_count", 0), r.get("eval_count", 0)
        if p in ("openai", "groq"):
            base = {"groq": "https://api.groq.com/openai/v1"}.get(
                p, os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
            key = self._key("GROQ_API_KEY" if p == "groq" else "OPENAI_API_KEY")
            r = _post(f"{base}/chat/completions",
                      {"model": m, "temperature": T, "response_format": {"type": "json_object"},
                       "messages": [{"role": "system", "content": system},
                                    {"role": "user", "content": user}]},
                      {"Authorization": f"Bearer {key}"})
            u = r.get("usage", {})
            return (r["choices"][0]["message"]["content"],
                    u.get("prompt_tokens", 0), u.get("completion_tokens", 0))
        raise ValueError(f"unknown provider {p!r}; see promo_finder/models.py")


class FallbackLLM(LLM):
    """--model "a,b,c": use a; when its quota runs out, move to b, then c.
    A reply already cached for any model in the chain is reused first, so a resumed
    run replays finished work for free. `last_spec` records who answered each call."""

    FALLBACK_WAIT = 60.0   # in a chain, don't sit out long limits: move on instead

    def __init__(self, spec: str, temperature: float = 0.2, use_cache: bool = True,
                 keys: dict | None = None):
        self.spec, self.temperature, self.use_cache = spec, temperature, use_cache
        self.chain = [LLM(s.strip(), temperature, use_cache, keys) for s in spec.split(",") if s.strip()]
        for m in self.chain[:-1]:
            m.max_wait = min(m.max_wait, self.FALLBACK_WAIT)
        self.exhausted: set[str] = set()
        self.last_spec = self.chain[0].spec

    def complete(self, system: str, user: str) -> str:
        for m in self.chain:                       # free replay of finished work
            if m.cached(system, user):
                self.last_spec = m.spec
                return m.complete(system, user)
        err = None
        for m in self.chain:
            if m.spec in self.exhausted:
                continue
            try:
                text = m.complete(system, user)
                self.last_spec = m.spec
                return text
            except (QuotaExhausted, MissingKey, requests.HTTPError) as e:   # no quota/key, retired, down
                self.exhausted.add(m.spec)
                err = e
        raise err or QuotaExhausted(f"all models in {self.spec} are out of quota")

    def usages(self) -> list[dict]:
        return [m.usage() for m in self.chain]


def make_llm(spec: str, **kw) -> LLM:
    return FallbackLLM(spec, **kw) if "," in spec else LLM(spec, **kw)


class QuotaExhausted(RuntimeError):
    pass


def _server_wait(resp) -> float | None:
    """Seconds the provider asks us to wait: Retry-After header (Groq/OpenAI) or
    Google's RetryInfo.retryDelay ("41346s"). None if it doesn't say."""
    if resp is None:
        return None
    ra = resp.headers.get("retry-after")
    if ra:
        try:
            return float(ra)
        except ValueError:
            pass
    try:
        for d in resp.json().get("error", {}).get("details", []):
            if "retryDelay" in d:
                return float(str(d["retryDelay"]).rstrip("s"))
    except (ValueError, AttributeError):
        pass
    return None


class MissingKey(RuntimeError):
    pass


def _post(url, body, headers=None, timeout=180):
    r = requests.post(url, json=body, headers={"Content-Type": "application/json", **(headers or {})},
                      timeout=timeout)
    r.raise_for_status()
    return r.json()


def extract_json(text: str):
    """Tolerate ```json fences and leading chatter."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"(\{.*\}|\[.*\])", t, re.S)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError as e:
                raise ValueError(str(e))
        raise ValueError("no JSON found")


def _mock(user: str) -> str:
    """Deterministic stand-in so the pipeline can be tested with no network.
    It picks plausible beat IDs from the prompt; it is NOT a quality signal."""
    ids = re.findall(r"^(E\d{2}-\d{3}) \[(\d\d:\d\d\.\d\d)-(\d\d:\d\d\.\d\d)\] (?!cam)", user, re.M)
    if "TASK: SHOW BIBLE" in user:
        return json.dumps({"premise": "mock premise", "characters": [], "twists": []})
    if "TASK: JUDGE" in user:
        n = len(re.findall(r"^PROMO \d+", user, re.M))
        return json.dumps({"scores": [{"id": i, "hook": 3, "cliffhanger": 3, "standalone": 3,
                                       "spoiler": 1, "note": "mock"} for i in range(n)]})
    if "TASK: MONTAGE" in user:
        return json.dumps({"promos": []})
    if len(ids) < 12:
        return json.dumps({"promos": []})
    def sec(s):
        mm, rest = s.split(":")
        return int(mm) * 60 + float(rest)
    a = ids[2][0]
    b = next((i for i, s, e in ids if sec(e) - sec(ids[2][1]) >= 28), ids[-1][0])
    return json.dumps({"promos": [{"segments": [{"first_beat": a, "last_beat": b}],
                                   "hook_beat": a, "hook": "mock hook",
                                   "cliffhanger_beat": b, "cliffhanger": "mock cliff",
                                   "why": "mock", "context_needed": "", "spoiler_risk": "low"}]})
