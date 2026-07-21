from groundline.domain.reports import Report
from groundline.policy.models import AgentContext, AgentMemory, PolicyDecision
from groundline.world.models import OperationalHealth, WorldAction


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _utility_selected_shift(context: AgentContext) -> float:
    """Select a report shift by maximizing the employee's private reporting utility."""
    honesty = context.agent.traits.get("honesty", 0.5)
    blame = context.agent.traits.get("blame_sensitivity", 0.6)
    delivery = context.agent.utility_weights.get("delivery", 0.7)
    reputation = context.agent.utility_weights.get("reputation", 0.7)
    quality = context.agent.utility_weights.get("quality", 0.7)
    evaluation_gain = context.incentive_pressure * (0.5 * delivery + 0.5 * reputation * blame)
    integrity_cost = honesty + 0.25 * quality
    candidates = (step / 80 for step in range(21))
    return max(
        candidates,
        key=lambda shift: evaluation_gain * shift - 4 * integrity_cost * shift**2,
    )


class FixturePolicy:
    name = "fixture"
    fingerprint = "fixture-utility-v1"

    async def decide(self, context: AgentContext) -> PolicyDecision:
        if context.incoming_reports:
            count = len(context.incoming_reports)
            verified = {
                evidence.reported_by: evidence.observed_health
                for evidence in context.verification_evidence
            }
            values = [
                verified.get(report.agent_id, report.health) for report in context.incoming_reports
            ]
            base = OperationalHealth(
                progress=sum(value.progress for value in values) / count,
                quality=sum(value.quality for value in values) / count,
                schedule=sum(value.schedule for value in values) / count,
                reliability=sum(value.reliability for value in values) / count,
            )
        else:
            observation = context.observation
            defects = (observation.detected_defect_severity or 0) + (
                observation.discovered_defect_severity or 0
            )
            base = OperationalHealth(
                progress=_clamp(1 - (observation.effort_remaining or 0) / 10),
                quality=_clamp(1 - defects / 10),
                schedule=_clamp(1 - (observation.schedule_risk or 0) / 10),
                reliability=_clamp(1 - (observation.incident_severity or 0) / 10),
            )
        optimism = _utility_selected_shift(context)
        health = OperationalHealth(
            progress=_clamp(base.progress + optimism),
            quality=_clamp(base.quality + optimism),
            schedule=_clamp(base.schedule + optimism),
            reliability=_clamp(base.reliability + optimism),
        )
        actions: tuple[WorldAction, ...] = ()
        if context.agent.role == "contributor" and context.agent.department == "QA":
            detected = context.observation.detected_defect_severity or 0
            if detected > 0:
                detected_items = context.observation.detected_defects or {}
                item_id = max(detected_items, key=lambda key: detected_items[key])
                actions = (WorldAction(kind="test", item_id=item_id, amount=detected),)
        elif context.agent.role == "contributor":
            skill = max(context.agent.skills.values(), default=0.5)
            actions = (
                WorldAction(kind="work", item_id=context.scope[0], amount=0.75 + 0.75 * skill),
            )
        if context.agent.role == "executive":
            actions = (
                WorldAction(
                    kind=(
                        "release"
                        if context.release_window_open and health.score >= 0.95
                        else "delay_release"
                    ),
                ),
            )
        report = Report(
            agent_id=context.agent.id,
            department=context.agent.department,
            depth=context.depth,
            tick=context.tick,
            scope=context.scope,
            health=health,
            confidence=0.75,
            escalate=health.score < 0.55,
            explanation="Assessment based on available evidence and current evaluation conditions.",
        )
        memory = AgentMemory(
            turns=context.memory.turns + 1,
            concerns=context.memory.concerns + (("release-risk",) if report.escalate else ()),
            trust=context.memory.trust,
        )
        return PolicyDecision(report=report, actions=actions, memory=memory)
