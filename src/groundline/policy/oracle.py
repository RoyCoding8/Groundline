from groundline.domain.reports import Report
from groundline.policy.models import AgentContext, AgentMemory, PolicyDecision
from groundline.world.models import OperationalHealth, WorldAction


class OraclePolicy:
    """Truth-informed benchmark used only to calculate counterfactual regret."""

    name = "oracle"
    fingerprint = "truth-oracle-v1"

    async def decide(self, context: AgentContext) -> PolicyDecision:
        raise RuntimeError("oracle decisions require the explicit truth-aware policy seam")

    async def decide_with_truth(
        self, context: AgentContext, truth: OperationalHealth
    ) -> PolicyDecision:
        actions: tuple[WorldAction, ...] = ()
        if context.agent.role == "contributor" and context.agent.department == "QA":
            defects = context.observation.detected_defects or {}
            if defects:
                item_id = max(defects, key=lambda key: defects[key])
                actions = (WorldAction(kind="test", item_id=item_id, amount=defects[item_id]),)
        elif context.agent.role == "contributor":
            skill = max(context.agent.skills.values(), default=0.5)
            actions = (
                WorldAction(kind="work", item_id=context.scope[0], amount=0.75 + 0.75 * skill),
            )
        elif context.agent.role == "executive":
            actions = (
                WorldAction(
                    kind=(
                        "release"
                        if context.release_window_open and truth.score >= 0.95
                        else "delay_release"
                    )
                ),
            )
        return PolicyDecision(
            report=Report(
                agent_id=context.agent.id,
                department=context.agent.department,
                depth=context.depth,
                tick=context.tick,
                scope=context.scope,
                health=truth,
                confidence=1.0,
                escalate=truth.score < 0.95,
                explanation="Truth-informed counterfactual benchmark.",
            ),
            actions=actions,
            memory=AgentMemory(
                turns=context.memory.turns + 1,
                concerns=context.memory.concerns,
                trust=context.memory.trust,
            ),
        )
