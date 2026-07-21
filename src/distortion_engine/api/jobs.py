from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import time
import uuid
from contextlib import suppress
from pathlib import Path

from distortion_engine.api.models import JobStatus, LaunchExperiment
from distortion_engine.events.artifacts import verify_run_artifacts
from distortion_engine.events.store import FileEventStore, canonical_json
from distortion_engine.experiments.runner import ExperimentRunner
from distortion_engine.policy.factory import build_policy
from distortion_engine.policy.models import AgentPolicy


class ExperimentLaunchConflict(RuntimeError):
    """Raised when an experiment name belongs to a different launch."""


class ExperimentJobManager:
    lease_seconds = 30.0

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.database = self.root / "jobs.sqlite3"
        self.owner = uuid.uuid4().hex
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._initialize()

    def conflicts(self, launch: LaunchExperiment) -> bool:
        payload = canonical_json(launch.model_dump(mode="json"))
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT 1 FROM jobs
                WHERE experiment = ? AND launch_json != ?
                LIMIT 1
                """,
                (launch.experiment.name, payload),
            ).fetchone()
        return row is not None

    async def launch(self, launch: LaunchExperiment) -> JobStatus:
        payload = canonical_json(launch.model_dump(mode="json"))
        digest = hashlib.sha256(payload.encode()).hexdigest()
        job_id = f"job-{digest[:16]}"
        reporting_spans = {
            treatment.reporting_span for treatment in launch.experiment.treatments.values()
        }
        total = len(launch.experiment.seeds) * (
            len(launch.experiment.treatments) + len(reporting_spans)
        )
        now = time.time()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            conflict = connection.execute(
                """
                SELECT 1 FROM jobs
                WHERE experiment = ? AND launch_json != ?
                LIMIT 1
                """,
                (launch.experiment.name, payload),
            ).fetchone()
            if conflict is not None:
                raise ExperimentLaunchConflict(launch.experiment.name)
            existing = connection.execute(
                "SELECT status FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            if existing is None:
                connection.execute(
                    """
                    INSERT INTO jobs (
                        job_id, experiment, status, launch_json, completed_runs,
                        failed_runs, total_runs, error, created_at, updated_at,
                        lease_owner, lease_expires_at, result_path
                    ) VALUES (?, ?, 'queued', ?, 0, 0, ?, NULL, ?, ?, NULL, NULL, NULL)
                    """,
                    (job_id, launch.experiment.name, payload, total, now, now),
                )
            elif existing["status"] == "failed":
                connection.execute(
                    """
                    UPDATE jobs
                    SET status = 'queued', completed_runs = 0, failed_runs = 0,
                        error = NULL, updated_at = ?, lease_owner = NULL,
                        lease_expires_at = NULL, result_path = NULL
                    WHERE job_id = ?
                    """,
                    (now, job_id),
                )
            connection.commit()
        self._schedule(job_id)
        return self.snapshot(job_id)

    async def recover(self) -> None:
        now = time.time()
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT job_id FROM jobs
                WHERE status = 'queued'
                   OR (status = 'running' AND lease_expires_at <= ?)
                """,
                (now,),
            ).fetchall()
        for row in rows:
            self._schedule(str(row["job_id"]))

    def snapshot(self, job_id: str) -> JobStatus:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT job_id, experiment, status, completed_runs,
                       failed_runs, total_runs, error
                FROM jobs WHERE job_id = ?
                """,
                (job_id,),
            ).fetchone()
        if row is None:
            raise KeyError(job_id)
        job = JobStatus.model_validate(dict(row))
        if job.status not in {"queued", "running"}:
            return job
        completed, failed = self._state_progress(job.experiment)
        if completed == job.completed_runs and failed == job.failed_runs:
            return job
        self._update_progress(job_id, completed, failed)
        return job.model_copy(update={"completed_runs": completed, "failed_runs": failed})

    def _schedule(self, job_id: str) -> None:
        current = self._tasks.get(job_id)
        if current is not None and not current.done():
            return
        self._tasks[job_id] = asyncio.create_task(self._execute(job_id))

    async def _execute(self, job_id: str) -> None:
        if not self._claim(job_id):
            self._tasks.pop(job_id, None)
            return
        heartbeat = asyncio.create_task(self._heartbeat(job_id))
        try:
            launch = self._launch_request(job_id)
            policy = self._policy(launch)
            result = await ExperimentRunner().run(
                launch.experiment, policy, FileEventStore(self.root)
            )
            self._verify_experiment(result.state_path)
            status_value = "completed" if result.failed_runs == 0 else "failed"
            error = None if result.failed_runs == 0 else "one or more runs failed"
            self._finish(
                job_id,
                status=status_value,
                completed_runs=result.resumed_runs + result.executed_runs - result.failed_runs,
                failed_runs=result.failed_runs,
                error=error,
                result_path=str(result.analysis_path.resolve()),
            )
        except Exception as error:
            self._finish(
                job_id,
                status="failed",
                completed_runs=self.snapshot(job_id).completed_runs,
                failed_runs=max(self.snapshot(job_id).failed_runs, 1),
                error=f"{type(error).__name__}: {error}",
                result_path=None,
            )
        finally:
            heartbeat.cancel()
            with suppress(asyncio.CancelledError):
                await heartbeat
            self._tasks.pop(job_id, None)

    def _claim(self, job_id: str) -> bool:
        now = time.time()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT status, lease_expires_at FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            claimable = row is not None and (
                row["status"] == "queued"
                or (row["status"] == "running" and float(row["lease_expires_at"] or 0) <= now)
            )
            if claimable:
                connection.execute(
                    """
                    UPDATE jobs
                    SET status = 'running', lease_owner = ?, lease_expires_at = ?,
                        updated_at = ?, error = NULL
                    WHERE job_id = ?
                    """,
                    (self.owner, now + self.lease_seconds, now, job_id),
                )
            connection.commit()
        return claimable

    async def _heartbeat(self, job_id: str) -> None:
        while True:
            await asyncio.sleep(self.lease_seconds / 3)
            now = time.time()
            with self._connect() as connection:
                connection.execute(
                    """
                    UPDATE jobs
                    SET lease_expires_at = ?, updated_at = ?
                    WHERE job_id = ? AND status = 'running' AND lease_owner = ?
                    """,
                    (now + self.lease_seconds, now, job_id, self.owner),
                )

    def _launch_request(self, job_id: str) -> LaunchExperiment:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT launch_json FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        if row is None:
            raise KeyError(job_id)
        return LaunchExperiment.model_validate_json(str(row["launch_json"]))

    def _finish(
        self,
        job_id: str,
        *,
        status: str,
        completed_runs: int,
        failed_runs: int,
        error: str | None,
        result_path: str | None,
    ) -> None:
        now = time.time()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE jobs
                SET status = ?, completed_runs = ?, failed_runs = ?, error = ?,
                    result_path = ?, updated_at = ?, lease_owner = NULL,
                    lease_expires_at = NULL
                WHERE job_id = ? AND lease_owner = ?
                """,
                (
                    status,
                    completed_runs,
                    failed_runs,
                    error,
                    result_path,
                    now,
                    job_id,
                    self.owner,
                ),
            )

    def _state_progress(self, experiment: str) -> tuple[int, int]:
        path = self.root / "experiments" / experiment / "experiment-state.json"
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return 0, 0
        records = list(state.get("runs", {}).values())
        completed = sum(record.get("status") == "completed" for record in records)
        failed = sum(record.get("status") == "failed" for record in records)
        return completed, failed

    def _update_progress(self, job_id: str, completed: int, failed: int) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE jobs SET completed_runs = ?, failed_runs = ?, updated_at = ?
                WHERE job_id = ? AND status IN ('queued', 'running')
                """,
                (completed, failed, time.time(), job_id),
            )

    @staticmethod
    def _verify_experiment(state_path: Path) -> None:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        for record in state.get("runs", {}).values():
            if record.get("status") == "completed":
                verify_run_artifacts(Path(record["run_directory"]))

    def _policy(self, launch: LaunchExperiment) -> AgentPolicy:
        return build_policy(
            launch.policy,
            model=launch.model or None,
            artifacts=self.root,
        )

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    experiment TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (
                        status IN ('queued', 'running', 'completed', 'failed')
                    ),
                    launch_json TEXT NOT NULL,
                    completed_runs INTEGER NOT NULL,
                    failed_runs INTEGER NOT NULL,
                    total_runs INTEGER NOT NULL,
                    error TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    lease_owner TEXT,
                    lease_expires_at REAL,
                    result_path TEXT
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection
