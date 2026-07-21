from pydantic import BaseModel, ConfigDict, Field

from distortion_engine.world.models import OperationalHealth


class Report(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    agent_id: str
    department: str
    depth: int = Field(ge=0)
    tick: int = Field(ge=0)
    scope: tuple[str, ...]
    health: OperationalHealth
    confidence: float = Field(ge=0, le=1)
    escalate: bool
    resource_request: float = Field(default=0, ge=0)
    explanation: str = Field(max_length=2000)
