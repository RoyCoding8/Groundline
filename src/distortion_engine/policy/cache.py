from __future__ import annotations

import json
from collections.abc import Iterator, MutableMapping
from pathlib import Path
from threading import RLock

from distortion_engine.events.store import canonical_json
from distortion_engine.policy.models import PolicyDecision


class FileDecisionCache(MutableMapping[str, PolicyDecision]):
    """Content-addressed model decisions that make repeated runs exact."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = RLock()
        self._values: dict[str, PolicyDecision]
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            self._values = {key: PolicyDecision.model_validate(value) for key, value in raw.items()}
        else:
            self._values = {}

    def __getitem__(self, key: str) -> PolicyDecision:
        with self._lock:
            return self._values[key]

    def __setitem__(self, key: str, value: PolicyDecision) -> None:
        with self._lock:
            self._values[key] = value
            self._write()

    def __delitem__(self, key: str) -> None:
        with self._lock:
            del self._values[key]
            self._write()

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            canonical_json(
                {
                    cache_key: decision.model_dump(mode="json")
                    for cache_key, decision in self._values.items()
                }
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.path)

    def __iter__(self) -> Iterator[str]:
        with self._lock:
            return iter(tuple(self._values))

    def __len__(self) -> int:
        with self._lock:
            return len(self._values)
