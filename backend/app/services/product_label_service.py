"""ProductLabelService: runs the label pipeline and keeps each result for an hour (for the explanation)."""
from __future__ import annotations

import logging
import re
import sys
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, Optional

from ..config import BACKEND_ROOT

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from model.product_label.pipeline import ProductLabelPipeline  # noqa: E402

log = logging.getLogger("backend.product_label")

SESSION_TTL_S = 3600
MAX_SESSIONS = 200
MAX_NAME_CHARS = 100


def clean_product_name(name: Optional[str]) -> Optional[str]:
    """The product name the user typed: control characters removed, spaces collapsed, at most 100 characters."""
    value = re.sub(r"\s+", " ", re.sub(r"[\x00-\x1f\x7f]", " ", name or "")).strip()[:MAX_NAME_CHARS]
    return value or None


class ProductLabelService:
    def __init__(self, pipeline: Optional[ProductLabelPipeline] = None) -> None:
        self.pipeline = pipeline or ProductLabelPipeline()
        self._sessions: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._lock = threading.Lock()
        self._run_lock = threading.Lock()

    @property
    def ready(self) -> bool:
        return self.pipeline.ocr.available

    @property
    def status(self) -> str:
        return "ready" if self.ready else "unavailable"

    def interpret(self, image, product_name: Optional[str] = None) -> Dict[str, Any]:
        with self._run_lock:
            result = self.pipeline.run(image)
        # Typed by the user: a title for the result and context for the explanation.
        result["product_name"] = clean_product_name(product_name)
        if result["status"] == "ok":
            with self._lock:
                self._sessions[result["session_id"]] = {"created": time.time(), "result": result}
                self._prune()
        return result

    def _get(self, session_id: str) -> Dict[str, Any]:
        with self._lock:
            self._prune()
            entry = self._sessions.get(session_id)
        if entry is None:
            raise KeyError(session_id)
        return entry

    def explanation(self, session_id: str) -> Dict[str, Any]:
        entry = self._get(session_id)
        with self._lock:
            lock = entry.setdefault("explain_lock", threading.Lock())
        with lock:   # generated once per session
            self.pipeline.add_explanation(entry["result"])
        return entry["result"]


    def _prune(self) -> None:
        now = time.time()
        for sid in [s for s, e in self._sessions.items() if now - e["created"] > SESSION_TTL_S]:
            self._sessions.pop(sid, None)
        while len(self._sessions) > MAX_SESSIONS:
            self._sessions.popitem(last=False)
