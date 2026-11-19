"""Shared by the eval scripts: an AI client that counts real requests and (for repeat test runs only) keeps
AI answers in a local cache folder, so re-running an eval doesn't use up the free AI allowance.
The app itself has no AI cache."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

import app.config  # noqa: F401  loads backend/.env (the AI keys), like the app does
from model.common.llm import LLM


class EvalLLM(LLM):
    def __init__(self, cache_dir: Optional[Path] = None, inner: Optional[LLM] = None) -> None:
        self.inner = inner if inner is not None else LLM()
        super().__init__(self.inner.providers)
        self.cache_dir = cache_dir
        self.requests = 0       # real AI requests made in this run
        self.cached = 0         # answers taken from the eval cache

    def complete(self, system: str, user: str, max_tokens: int = 500, json_mode: bool = False) -> Dict[str, Any]:
        key = hashlib.sha256(json.dumps([system, user, max_tokens, json_mode]).encode()).hexdigest()
        path = self.cache_dir / f"{key}.json" if self.cache_dir else None
        if path and path.exists():
            self.cached += 1
            return json.loads(path.read_text())
        self.requests += 1
        out = self.inner.complete(system, user, max_tokens, json_mode)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(out))
        return out
