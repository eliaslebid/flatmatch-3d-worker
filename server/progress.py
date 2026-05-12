"""Per-scan progress emitter.

Mounts a tqdm monkey-patch so the streaming-inference bar inside
``model.inference_streaming`` writes per-frame progress to a JSON file
next to the scan output. The FastAPI status endpoint reads it.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Optional


_state = threading.local()


def set_path(path: Optional[Path]) -> None:
    _state.path = Path(path) if path else None


def set_phase(phase: str, **extra) -> None:
    _state.phase = phase
    _write({"phase": phase, **extra})


def _write(payload: dict) -> None:
    path = getattr(_state, "path", None)
    if path is None:
        return
    try:
        path.write_text(json.dumps(payload))
    except Exception:
        pass


# --- tqdm patch: intercept the "Streaming inference" bar only -----------------
from tqdm.std import tqdm as _stdtqdm  # noqa: E402

_orig_update = _stdtqdm.update


def _patched_update(self, n=1):
    result = _orig_update(self, n)
    desc = getattr(self, "desc", "") or ""
    if "Streaming inference" in desc and getattr(self, "total", None):
        _write(
            {
                "phase": getattr(_state, "phase", "inferring"),
                "current": int(self.n),
                "total": int(self.total),
            }
        )
    return result


_stdtqdm.update = _patched_update
