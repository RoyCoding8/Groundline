import hashlib
import json
from pathlib import Path
from typing import Any

from groundline.events.models import DecisionLedgerEntry, Event, RunManifest


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def hash_lines(lines: list[str]) -> str:
    data = "".join(f"{line}\n" for line in lines).encode()
    return hashlib.sha256(data).hexdigest()


class FileEventStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def finalize(
        self,
        *,
        run_id: str,
        seed: int,
        request_hash: str,
        policy: str,
        policy_fingerprint: str,
        engine_fingerprint: str,
        events: tuple[Event, ...],
        decisions: tuple[DecisionLedgerEntry, ...],
        metrics: dict[str, Any],
        request: dict[str, Any],
    ) -> tuple[Path, RunManifest]:
        run_directory = self.root / run_id
        run_directory.mkdir(parents=True, exist_ok=True)
        event_lines = [canonical_json(event.model_dump(mode="json")) for event in events]
        decision_lines = [
            canonical_json(decision.model_dump(mode="json")) for decision in decisions
        ]
        manifest = RunManifest(
            run_id=run_id,
            seed=seed,
            request_hash=request_hash,
            event_hash=hash_lines(event_lines),
            decision_hash=hash_lines(decision_lines),
            metrics_hash=canonical_hash(metrics),
            event_count=len(events),
            decision_count=len(decisions),
            finalized=True,
            policy=policy,
            policy_fingerprint=policy_fingerprint,
            engine_fingerprint=engine_fingerprint,
        )
        manifest_path = run_directory / "manifest.json"
        if manifest_path.exists():
            from groundline.events.artifacts import (
                ArtifactCorruptError,
                verify_run_artifacts,
            )

            try:
                existing = verify_run_artifacts(run_directory).manifest
            except ArtifactCorruptError as error:
                raise ValueError(f"finalized run is immutable: {run_id}") from error
            if existing == manifest:
                return run_directory, existing
            raise ValueError(f"finalized run is immutable: {run_id}")

        self._write_atomic(
            run_directory / "events.jsonl",
            "".join(f"{line}\n" for line in event_lines).encode(),
        )
        self._write_atomic(
            run_directory / "decisions.jsonl",
            "".join(f"{line}\n" for line in decision_lines).encode(),
        )
        self._write_atomic(
            run_directory / "metrics.json",
            (canonical_json(metrics) + "\n").encode(),
        )
        self._write_atomic(
            run_directory / "request.json",
            (canonical_json(request) + "\n").encode(),
        )
        self._write_atomic(
            manifest_path,
            (canonical_json(manifest.model_dump(mode="json")) + "\n").encode(),
        )

        from groundline.events.artifacts import verify_run_artifacts

        verified = verify_run_artifacts(run_directory)
        return run_directory, verified.manifest

    @staticmethod
    def _write_atomic(path: Path, data: bytes) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(data)
        temporary.replace(path)
