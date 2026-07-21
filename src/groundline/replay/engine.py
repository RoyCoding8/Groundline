from pathlib import Path
from tempfile import TemporaryDirectory

from pydantic import BaseModel, ConfigDict

from groundline.events.artifacts import verify_run_artifacts
from groundline.events.models import DecisionLedgerEntry
from groundline.events.store import FileEventStore
from groundline.policy.models import AgentContext, PolicyDecision
from groundline.simulation.runner import SimulationRunner


class ReplayResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    equivalent: bool
    network_calls: int
    event_hash: str
    decision_hash: str
    reconstructed_event_count: int


class LedgerPolicy:
    def __init__(self, name: str, entries: tuple[DecisionLedgerEntry, ...]) -> None:
        self.name = name
        self.entries = entries
        self.cursor = 0

    async def decide(self, context: AgentContext) -> PolicyDecision:
        if self.cursor >= len(self.entries):
            raise ValueError("decision ledger exhausted during replay")
        entry = self.entries[self.cursor]
        self.cursor += 1
        if entry.agent_id != context.agent.id or entry.tick != context.tick:
            raise ValueError(
                f"decision ledger mismatch: expected {entry.agent_id}@{entry.tick}, "
                f"received {context.agent.id}@{context.tick}"
            )
        return PolicyDecision.model_validate(entry.decision)


class ReplayEngine:
    async def replay(self, run_directory: Path) -> ReplayResult:
        artifacts = verify_run_artifacts(run_directory)
        with TemporaryDirectory(prefix="distortion-replay-") as temporary:
            reconstructed = await SimulationRunner().run(
                artifacts.request,
                LedgerPolicy(artifacts.manifest.policy, artifacts.decisions),
                FileEventStore(Path(temporary)),
            )
            reconstructed_artifacts = verify_run_artifacts(reconstructed.run_directory)
        equivalent = (
            reconstructed_artifacts.manifest.event_hash == artifacts.manifest.event_hash
            and reconstructed_artifacts.manifest.decision_hash == artifacts.manifest.decision_hash
            and reconstructed_artifacts.manifest.metrics_hash == artifacts.manifest.metrics_hash
            and reconstructed_artifacts.manifest.event_count == artifacts.manifest.event_count
            and reconstructed_artifacts.manifest.decision_count == artifacts.manifest.decision_count
        )
        return ReplayResult(
            equivalent=equivalent,
            network_calls=0,
            event_hash=artifacts.manifest.event_hash,
            decision_hash=artifacts.manifest.decision_hash,
            reconstructed_event_count=reconstructed_artifacts.manifest.event_count,
        )
