"""The one AI client: a free OpenRouter text model, with Groq as the single backup. Text only (no images).

    OPENROUTER_API_KEY + OPENROUTER_MODEL (must end in ":free"; default nvidia/nemotron-3-ultra-550b-a55b:free)
    GROQ_API_KEY + GROQ_MODEL (backup; default qwen/qwen3.8-27b)

Both are OpenAI-compatible, so the official ``openai`` SDK talks to both. If both fail (limits, network,
no key) ``complete`` raises AIBusy and the caller shows its "AI busy" result. One retry per provider, only
for a rate limit whose "retry-after" is at most RETRY_AFTER_MAX_S seconds; no budget files, no caches. Keys are
never logged.

Also here: the helpers every AI prompt uses (untrusted text in delimited blocks, loose JSON parsing).
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

log = logging.getLogger("common.llm")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_OPENROUTER_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
DEFAULT_GROQ_MODEL = "qwen/qwen3.8-27b"
TIMEOUT_S = 30.0
# A per-minute limit (HTTP 429) usually clears in seconds; the provider says how long in "retry-after".
# One wait-and-retry per provider when that wait is this short; longer limits (daily quota) fail fast.
RETRY_AFTER_MAX_S = 10.0
AI_BUSY_NOTE = "AI busy: details not extracted. Try again in a few minutes."


class AIBusy(Exception):
    """Every provider failed (limit, network, no key). ``attempts`` = [(provider, reason)] without secrets."""

    def __init__(self, attempts: List[tuple]) -> None:
        self.attempts = attempts
        super().__init__("; ".join(f"{p}: {r}" for p, r in attempts) or "no AI provider configured")


@dataclass
class Provider:
    name: str
    model: str
    client: Any = field(repr=False)
    extra_body: Dict[str, Any] = field(default_factory=dict)


def _client(base_url: str, api_key: str, headers: Optional[Dict[str, str]] = None):
    from openai import OpenAI
    return OpenAI(base_url=base_url, api_key=api_key, timeout=TIMEOUT_S, max_retries=0, default_headers=headers)


def providers_from_env(env: Optional[Dict[str, str]] = None) -> List[Provider]:
    env = os.environ if env is None else env
    out: List[Provider] = []
    key = env.get("OPENROUTER_API_KEY", "").strip()
    model = env.get("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL).strip()
    if key and model.endswith(":free"):        # free models only, never a paid one
        out.append(Provider("OpenRouter", model, _client(
            OPENROUTER_BASE_URL, key, {"HTTP-Referer": env.get("OPENROUTER_REFERER", "http://localhost"), "X-Title": "Vision AI"}),
            {"reasoning": {"enabled": False}}))   # reasoning models otherwise answer with nothing
    elif key:
        log.warning("OPENROUTER_MODEL is not a ':free' model; OpenRouter is not used")
    key = env.get("GROQ_API_KEY", "").strip()
    if key:
        out.append(Provider("Groq", env.get("GROQ_MODEL", DEFAULT_GROQ_MODEL).strip(), _client(GROQ_BASE_URL, key)))
    return out


class LLM:
    def __init__(self, providers: Optional[List[Provider]] = None) -> None:
        self.providers = providers_from_env() if providers is None else providers

    @classmethod
    def with_client(cls, client: Any, model: str = "test-model", name: str = "test") -> "LLM":
        """One injected client (tests)."""
        return cls([Provider(name, model, client)])

    @property
    def available(self) -> bool:
        return bool(self.providers)

    def describe(self) -> List[Dict[str, str]]:
        return [{"provider": p.name, "model": p.model} for p in self.providers]

    def complete(self, system: str, user: str, max_tokens: int = 500, json_mode: bool = False) -> Dict[str, Any]:
        """{"text", "provider", "model", "ms"} from the first provider that answers; AIBusy if none does."""
        attempts = []
        for p in self.providers:
            request: Dict[str, Any] = dict(model=p.model, temperature=0.1, max_tokens=max_tokens,
                                           messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
            if json_mode:
                request["response_format"] = {"type": "json_object"}
            if p.extra_body:
                request["extra_body"] = p.extra_body
            started = time.perf_counter()
            try:
                try:
                    response = p.client.chat.completions.create(**request)
                except Exception as exc:
                    wait = _short_retry_after(exc)
                    if wait is None:
                        raise
                    log.info("AI %s (%s) rate limited; retrying in %.1f s", p.name, p.model, wait)
                    time.sleep(wait)
                    response = p.client.chat.completions.create(**request)
                text = (response.choices[0].message.content or "").strip()
            except Exception as exc:   # rate limit, timeout, provider error: try the backup
                attempts.append((p.name, type(exc).__name__))
                log.info("AI %s (%s) failed: %s", p.name, p.model, type(exc).__name__)
                continue
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
            if not text:
                attempts.append((p.name, "empty answer"))
                continue
            ms = int((time.perf_counter() - started) * 1000)
            log.info("AI text from %s (%s) in %d ms", p.name, p.model, ms)
            return {"text": text, "provider": p.name, "model": p.model, "ms": ms}
        raise AIBusy(attempts)


def _short_retry_after(exc: Exception) -> Optional[float]:
    """Seconds to wait when ``exc`` is an HTTP 429 whose retry-after is short; None otherwise."""
    response = getattr(exc, "response", None)
    if getattr(exc, "status_code", None) != 429 and getattr(response, "status_code", None) != 429:
        return None
    headers = getattr(response, "headers", None) or {}
    try:
        wait = float(headers.get("retry-after"))
    except (TypeError, ValueError):
        return None
    return wait if 0 <= wait <= RETRY_AFTER_MAX_S else None


_DEFAULT: Optional[LLM] = None


def default_llm() -> LLM:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = LLM()
    return _DEFAULT


# ---- prompt helpers ------------------------------------------------------------------------------
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_FENCES = re.compile(r"`{3,}|<{3,}|>{3,}")


def sanitize_untrusted(text: Any, max_chars: int = 6000) -> str:
    """Strip control characters and delimiter look-alikes; cap the length."""
    value = _FENCES.sub(" ", _CONTROL.sub(" ", "" if text is None else str(text)))
    return value[:max_chars] + " [truncated]" if len(value) > max_chars else value


def data_block(label: str, text: Any, max_chars: int = 6000) -> str:
    """Wrap untrusted content (OCR text, a user's question) so the model treats it as data."""
    tag = re.sub(r"[^A-Z_]", "", label.upper()) or "DATA"
    return f"<<<{tag}\n{sanitize_untrusted(text, max_chars)}\n{tag}>>>"


def parse_json_loose(text: str) -> Optional[Dict[str, Any]]:
    """A JSON object from an AI reply: the whole text, the first {...} span or a ```json block."""
    if not text:
        return None
    candidates = [text.strip()]
    m = re.search(r"```json\s*(.*?)\s*```", text, re.S | re.I)
    if m:
        candidates.append(m.group(1))
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        candidates.append(m.group(0))
    for c in candidates:
        try:
            value = json.loads(c)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(value, dict):
            return value
    return None


def _numbers(text: str) -> List[str]:
    return re.findall(r"\d+(?:[.,]\d+)?", text or "")


def numbers_supported(answer: str, facts_text: str) -> bool:
    """Guard against invented numbers: every number in an AI answer must occur in the facts it was given
    (12.5 == 12,5; 12 == 12.0)."""
    known = set()
    for n in _numbers(facts_text):
        try:
            v = float(n.replace(",", "."))
        except ValueError:
            continue
        known.update((round(v, 3), round(v)))
    for n in _numbers(answer):
        try:
            v = float(n.replace(",", "."))
        except ValueError:
            return False
        if round(v, 3) not in known and round(v) not in known:
            return False
    return True
