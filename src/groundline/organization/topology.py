import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict

from groundline.organization.models import OrganizationConfig

type ReportingSpan = Literal["narrow", "wide"]

_ALGORITHM: Literal["root-manager-span-v1"] = "root-manager-span-v1"


class ReportingEdgeChange(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    agent_id: str
    previous_manager_id: str | None
    effective_manager_id: str


class ReportingTopology(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    algorithm: Literal["root-manager-span-v1"] = _ALGORITHM
    requested_span: ReportingSpan
    requested_fingerprint: str
    effective_fingerprint: str
    transform_fingerprint: str
    edge_changes: tuple[ReportingEdgeChange, ...]


class ReportingSpanTransform(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    organization: OrganizationConfig
    topology: ReportingTopology

    @property
    def effective_fingerprint(self) -> str:
        return self.topology.effective_fingerprint

    @property
    def edge_changes(self) -> tuple[ReportingEdgeChange, ...]:
        return self.topology.edge_changes


def _fingerprint(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _organization_fingerprint(organization: OrganizationConfig) -> str:
    return _fingerprint(organization.model_dump(mode="json"))


def transform_reporting_span(
    organization: OrganizationConfig,
    span: ReportingSpan,
) -> ReportingSpanTransform:
    managers = sorted(
        agent.id
        for agent in organization.agents
        if agent.role == "manager" and agent.manager_id == organization.root_id
    )
    if len(managers) < 2:
        raise ValueError("reporting span requires at least two root-level managers")
    contributors = sorted(agent.id for agent in organization.agents if agent.role == "contributor")
    if len(contributors) < 2:
        raise ValueError("reporting span requires at least two contributors")

    if span == "narrow":
        assignments = {
            contributor_id: managers[index % len(managers)]
            for index, contributor_id in enumerate(contributors)
        }
    else:
        assignments = {contributor_id: managers[0] for contributor_id in contributors}

    effective = OrganizationConfig(
        agents=tuple(
            agent.model_copy(update={"manager_id": assignments[agent.id]})
            if agent.id in assignments
            else agent
            for agent in organization.agents
        )
    )
    changes = tuple(
        ReportingEdgeChange(
            agent_id=contributor_id,
            previous_manager_id=organization.by_id[contributor_id].manager_id,
            effective_manager_id=assignments[contributor_id],
        )
        for contributor_id in contributors
        if organization.by_id[contributor_id].manager_id != assignments[contributor_id]
    )
    requested_fingerprint = _organization_fingerprint(organization)
    effective_fingerprint = _organization_fingerprint(effective)
    transform_fingerprint = _fingerprint(
        {
            "algorithm": _ALGORITHM,
            "requested_span": span,
            "requested_fingerprint": requested_fingerprint,
            "effective_fingerprint": effective_fingerprint,
            "edge_changes": [change.model_dump(mode="json") for change in changes],
        }
    )
    return ReportingSpanTransform(
        organization=effective,
        topology=ReportingTopology(
            requested_span=span,
            requested_fingerprint=requested_fingerprint,
            effective_fingerprint=effective_fingerprint,
            transform_fingerprint=transform_fingerprint,
            edge_changes=changes,
        ),
    )
