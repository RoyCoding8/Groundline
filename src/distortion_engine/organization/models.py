from collections import defaultdict
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

type AgentRole = Literal["contributor", "manager", "executive"]


class AgentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$")
    manager_id: str | None = None
    role: AgentRole
    department: str = Field(min_length=1)
    skills: dict[str, float] = Field(default_factory=dict)
    traits: dict[str, float] = Field(default_factory=dict)
    utility_weights: dict[str, float] = Field(default_factory=dict)


class OrganizationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    agents: tuple[AgentConfig, ...]

    @model_validator(mode="after")
    def validate_rooted_tree(self) -> "OrganizationConfig":
        ids = [agent.id for agent in self.agents]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("organization must be a rooted tree with unique agent ids")
        roots = [agent.id for agent in self.agents if agent.manager_id is None]
        if len(roots) != 1:
            raise ValueError("organization must be a rooted tree with exactly one root")
        known = set(ids)
        if any(agent.manager_id not in known for agent in self.agents if agent.manager_id):
            raise ValueError("organization must be a rooted tree with known managers")
        root = roots[0]
        for agent in self.agents:
            seen: set[str] = set()
            current = agent.id
            while current != root:
                if current in seen:
                    raise ValueError("organization must be a rooted tree without cycles")
                seen.add(current)
                manager = self.by_id[current].manager_id
                if manager is None:
                    raise ValueError("organization must be a connected rooted tree")
                current = manager
        return self

    @property
    def by_id(self) -> dict[str, AgentConfig]:
        return {agent.id: agent for agent in self.agents}

    @property
    def root_id(self) -> str:
        return next(agent.id for agent in self.agents if agent.manager_id is None)

    @property
    def children(self) -> dict[str, tuple[str, ...]]:
        values: dict[str, list[str]] = defaultdict(list)
        for agent in self.agents:
            if agent.manager_id is not None:
                values[agent.manager_id].append(agent.id)
        return {key: tuple(sorted(value)) for key, value in values.items()}

    @property
    def depths(self) -> dict[str, int]:
        result = {self.root_id: 0}
        queue = [self.root_id]
        while queue:
            parent = queue.pop(0)
            for child in self.children.get(parent, ()):
                result[child] = result[parent] + 1
                queue.append(child)
        return result

    @property
    def spans(self) -> dict[str, int]:
        return {agent.id: len(self.children.get(agent.id, ())) for agent in self.agents}

    @property
    def reporting_order(self) -> tuple[tuple[str, ...], ...]:
        depths = self.depths
        return tuple(
            tuple(sorted(agent_id for agent_id, value in depths.items() if value == depth))
            for depth in range(max(depths.values()), -1, -1)
        )
