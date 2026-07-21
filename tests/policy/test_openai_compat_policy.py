from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

import httpx
import pytest

from distortion_engine.observation.engine import Observation
from distortion_engine.organization.models import AgentConfig
from distortion_engine.policy.cache import FileDecisionCache
from distortion_engine.policy.models import AgentContext
from distortion_engine.policy.openai_compat_policy import (
    INSTRUCTIONS,
    DecisionCacheMiss,
    OpenAICompatPolicy,
    OpenAICompatPolicyError,
    _Choice,
    _Message,
    _OpenAIResponse,
)
from distortion_engine.world.models import OperationalHealth


@pytest.fixture(autouse=True)
def _zero_retry_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    # Default the retry backoff to 0 for the suite so the retry-path tests
    # don't sleep. Dedicated backoff tests opt back in by setting the env.
    monkeypatch.setenv("DISTORTION_RETRY_BACKOFF_SECONDS", "0")


# ---------------------------------------------------------------------------
# Adapter-independent helpers
# ---------------------------------------------------------------------------


def _make_context(
    *,
    agent_id: str = "worker",
    role: str = "contributor",
    department: str = "Engineering",
    tick: int = 1,
    evidence: tuple[str, ...] = ("world:1",),
) -> AgentContext:
    return AgentContext(
        agent=AgentConfig(id=agent_id, role=role, department=department),
        depth=1,
        tick=tick,
        scope=("api",),
        observation=Observation(
            agent_id=agent_id,
            tick=tick,
            scope=("api",),
            evidence_refs=evidence,
        ),
        incentive_pressure=0.5,
        attention_budget=0,
    )


_HEALTH = OperationalHealth(
    progress=0.5,
    quality=0.4,
    schedule=0.3,
    reliability=0.2,
)


def _build_response_content(
    *,
    health: OperationalHealth = _HEALTH,
    confidence: float = 0.7,
    escalate: bool = False,
    explanation: str = "Current evidence indicates release risk.",
    actions: tuple[dict[str, Any], ...] = (),
    concerns: tuple[str, ...] = ("release-risk",),
) -> str:
    return json.dumps(
        {
            "health": health.model_dump(mode="json"),
            "confidence": confidence,
            "escalate": escalate,
            "explanation": explanation,
            "actions": list(actions),
            "concerns": list(concerns),
        }
    )


# ---------------------------------------------------------------------------
# Completion-fake helpers
# ---------------------------------------------------------------------------


def _success_response(
    content: str | None = None,
    *,
    response_id: str = "resp-1",
    model: str = "gpt-4o",
    usage: dict[str, int] | None = None,
) -> _OpenAIResponse:
    if content is None:
        content = _build_response_content()
    return _OpenAIResponse(
        choices=(_Choice(message=_Message(content=content, refusal=None)),),
        id=response_id,
        model=model,
        system_fingerprint="fp-1",
        usage=usage or {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    )


def _no_content_response() -> _OpenAIResponse:
    return _OpenAIResponse(
        choices=(_Choice(message=_Message(content=None, refusal=None)),),
        id="resp-no-content",
        model="gpt-4o",
        system_fingerprint="fp-1",
        usage=None,
    )


def _empty_choices_response() -> _OpenAIResponse:
    return _OpenAIResponse(
        choices=(),
        id="resp-empty",
        model="gpt-4o",
        system_fingerprint="fp-1",
        usage=None,
    )


def _refusal_response() -> _OpenAIResponse:
    return _OpenAIResponse(
        choices=(
            _Choice(
                message=_Message(
                    content="I cannot comply with that request.",
                    refusal="I cannot comply with that request.",
                )
            ),
        ),
        id="resp-refusal",
        model="gpt-4o",
        system_fingerprint="fp-1",
        usage=None,
    )


def _message_refusal_response() -> _OpenAIResponse:
    return _OpenAIResponse(
        choices=(_Choice(message=_Message(content=None, refusal="Content policy violation")),),
        id="resp-message-refusal",
        model="gpt-4o",
        system_fingerprint="fp-1",
        usage=None,
    )


def _malformed_json_response() -> _OpenAIResponse:
    return _success_response(content="this is not valid json")


def _extra_fields_response() -> _OpenAIResponse:
    content = _build_response_content()
    payload = json.loads(content)
    payload["unexpected_field"] = True
    return _success_response(content=json.dumps(payload))


def _schema_invalid_response() -> _OpenAIResponse:
    content = _build_response_content(confidence=999.0)
    return _success_response(content=content)


class _FakeCompletion:
    """Injected async completion callable matching the OpenAI-compat seam."""

    def __init__(self, responses: list[_OpenAIResponse | Exception] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._responses = list(responses or [_success_response()])
        self._call_index = 0

    async def __call__(
        self, *, url: str, headers: Any, body: Any, timeout: float
    ) -> _OpenAIResponse:
        self.calls.append(
            {"url": url, "headers": dict(headers), "body": dict(body), "timeout": timeout}
        )
        await asyncio.sleep(0)
        if self._call_index < len(self._responses):
            result = self._responses[self._call_index]
            self._call_index += 1
        else:
            result = self._responses[-1]
        if isinstance(result, Exception):
            raise result
        return result


# ---------------------------------------------------------------------------
# 1. Request shape
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_policy_builds_correct_request_kwargs() -> None:
    completion = _FakeCompletion()
    policy = OpenAICompatPolicy(
        model="gpt-4o",
        completion=completion,
        api_key="sk-test",
    )
    context = _make_context()

    await policy.decide(context)

    assert len(completion.calls) == 1
    call = completion.calls[0]
    assert call["url"].endswith("/chat/completions")
    assert call["headers"]["Authorization"] == "Bearer sk-test"
    assert call["body"]["model"] == "gpt-4o"
    messages = call["body"]["messages"]
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[0]["content"] == INSTRUCTIONS
    assert call["body"]["response_format"] == {"type": "json_object"}
    assert call["timeout"] == 120.0


@pytest.mark.asyncio
async def test_policy_prompt_contains_no_deception_instruction() -> None:
    completion = _FakeCompletion()
    policy = OpenAICompatPolicy(
        model="gpt-4o",
        completion=completion,
    )
    context = _make_context()

    await policy.decide(context)

    messages = completion.calls[0]["body"]["messages"]
    full_text = f"{messages[0]['content']} {messages[1]['content']}".lower()
    forbidden_words = {"deceive", "deception", "lie", "fabricate", "misrepresent"}
    for word in forbidden_words:
        assert not re.search(rf"\b{word}\b", full_text), f"Prompt contains forbidden word: {word}"


@pytest.mark.asyncio
async def test_completion_call_receives_timeout() -> None:
    completion = _FakeCompletion()
    policy = OpenAICompatPolicy(
        model="gpt-4o",
        completion=completion,
        timeout_seconds=45.0,
    )
    context = _make_context()

    await policy.decide(context)

    assert len(completion.calls) == 1
    assert completion.calls[0]["timeout"] == 45.0


@pytest.mark.asyncio
async def test_completion_call_receives_default_timeout() -> None:
    completion = _FakeCompletion()
    policy = OpenAICompatPolicy(
        model="gpt-4o",
        completion=completion,
    )
    context = _make_context()

    await policy.decide(context)

    assert completion.calls[0]["timeout"] == 120.0


# ---------------------------------------------------------------------------
# 2. No real network access
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_policy_uses_injected_completion_not_network() -> None:
    completion = _FakeCompletion()
    policy = OpenAICompatPolicy(
        model="gpt-4o",
        completion=completion,
    )

    await policy.decide(_make_context())

    assert len(completion.calls) == 1
    assert completion.calls[0]["body"]["model"] == "gpt-4o"


# ---------------------------------------------------------------------------
# 3. Valid JSON produces expected decision
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_policy_valid_json_produces_expected_decision() -> None:
    completion = _FakeCompletion()
    policy = OpenAICompatPolicy(
        model="gpt-4o",
        completion=completion,
    )
    context = _make_context(agent_id="lead")

    decision = await policy.decide(context)

    assert decision.report.agent_id == "lead"
    assert decision.report.health.reliability == 0.2
    assert decision.memory.turns == 1
    assert decision.provider_metadata["response_id"] == "resp-1"
    assert decision.provider_metadata["requested_model"] == "gpt-4o"
    assert decision.provider_metadata["policy"] == "openai_compat"
    assert decision.provider_metadata["structured_output_mode"] == "json_object"
    # Must not contain any litellm-specific keys
    assert "litellm_version" not in decision.provider_metadata
    assert "requested_provider" not in decision.provider_metadata
    assert "aws_region" not in decision.provider_metadata


# ---------------------------------------------------------------------------
# 4. Fail-closed response errors (all retry then OpenAICompatPolicyError)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_policy_extra_fields_fail_closed() -> None:
    bad = _extra_fields_response()
    completion = _FakeCompletion([bad, bad])
    policy = OpenAICompatPolicy(
        model="gpt-4o",
        completion=completion,
        max_attempts=2,
    )
    with pytest.raises(OpenAICompatPolicyError):
        await policy.decide(_make_context())
    assert len(completion.calls) == 2


@pytest.mark.asyncio
async def test_policy_malformed_json_fails_closed() -> None:
    bad = _malformed_json_response()
    completion = _FakeCompletion([bad, bad])
    policy = OpenAICompatPolicy(
        model="gpt-4o",
        completion=completion,
        max_attempts=2,
    )
    with pytest.raises(OpenAICompatPolicyError):
        await policy.decide(_make_context())
    assert len(completion.calls) == 2


@pytest.mark.asyncio
async def test_policy_missing_content_fails_closed() -> None:
    bad = _no_content_response()
    completion = _FakeCompletion([bad, bad])
    policy = OpenAICompatPolicy(
        model="gpt-4o",
        completion=completion,
        max_attempts=2,
    )
    with pytest.raises(OpenAICompatPolicyError):
        await policy.decide(_make_context())
    assert len(completion.calls) == 2


@pytest.mark.asyncio
async def test_policy_refusal_fails_closed() -> None:
    # A refusal is a deliberate policy decision, not a transient error:
    # it must fail closed immediately with the typed error and NOT retry
    # (retrying an identical refused prompt would refuse again).
    bad = _refusal_response()
    completion = _FakeCompletion([bad, bad])
    policy = OpenAICompatPolicy(
        model="gpt-4o",
        completion=completion,
        max_attempts=2,
    )
    with pytest.raises(OpenAICompatPolicyError):
        await policy.decide(_make_context())
    assert len(completion.calls) == 1


@pytest.mark.asyncio
async def test_policy_empty_choices_fails_closed() -> None:
    bad = _empty_choices_response()
    completion = _FakeCompletion([bad, bad])
    policy = OpenAICompatPolicy(
        model="gpt-4o",
        completion=completion,
        max_attempts=2,
    )
    with pytest.raises(OpenAICompatPolicyError):
        await policy.decide(_make_context())
    assert len(completion.calls) == 2


@pytest.mark.asyncio
async def test_policy_schema_invalid_output_fails_closed() -> None:
    bad = _schema_invalid_response()
    completion = _FakeCompletion([bad, bad])
    policy = OpenAICompatPolicy(
        model="gpt-4o",
        completion=completion,
        max_attempts=2,
    )
    with pytest.raises(OpenAICompatPolicyError):
        await policy.decide(_make_context())
    assert len(completion.calls) == 2


def _live_wrong_shape_response() -> _OpenAIResponse:
    # The exact pathological shape the live mimo-v2.5-free model produced when
    # the prompt named no schema: valid JSON, but a totally different field set
    # than AgentOutput. Surfaced live (tmp/PROBE-RAW-CONTENT-RESULT.md); no
    # hand-authored test fixture had ever returned wrong-field-name JSON, which
    # is why the missing-schema-in-prompt bug was invisible to mock tests.
    content = json.dumps(
        {
            "action": "investigate",
            "reason": "The progress is only at 0.4, which is concerning.",
            "report": "Manager, I'm investigating the payments-service.",
        }
    )
    return _success_response(content=content)


@pytest.mark.asyncio
async def test_policy_live_wrong_field_shape_fails_closed() -> None:
    # Regression guard for the live finding: a response that is valid JSON but
    # uses an invented field set (action/reason/report) must fail closed,
    # exactly like any other schema violation. This pins the live-discovered
    # pathological shape so the adapter can never silently accept it.
    bad = _live_wrong_shape_response()
    completion = _FakeCompletion([bad, bad])
    policy = OpenAICompatPolicy(
        model="gpt-4o",
        completion=completion,
        max_attempts=2,
    )
    with pytest.raises(OpenAICompatPolicyError):
        await policy.decide(_make_context())
    assert len(completion.calls) == 2


def test_instructions_embeds_agent_output_schema_contract() -> None:
    # The live bug: INSTRUCTIONS described no shape, so the model invented one
    # and every call failed extra="forbid". The fix is to embed the AgentOutput
    # schema in the prompt. This test pins the invariant — the prompt MUST name
    # every AgentOutput field and its shape — so the schema can never silently
    # disappear from the prompt again. Deterministic; no network, no key.
    for field in ("health", "confidence", "escalate", "explanation", "actions", "concerns"):
        assert field in INSTRUCTIONS, f"INSTRUCTIONS must name field {field!r}"
    # Nested OperationalHealth fields must be named too.
    for field in ("progress", "quality", "schedule", "reliability"):
        assert field in INSTRUCTIONS, f"INSTRUCTIONS must name health field {field!r}"
    # The array-vs-object shapes the model got wrong live must be explicit.
    assert "actions" in INSTRUCTIONS and "array" in INSTRUCTIONS.lower()
    assert "concerns" in INSTRUCTIONS and "array" in INSTRUCTIONS.lower()
    # And a concrete example must anchor the shapes.
    assert "kind" in INSTRUCTIONS


@pytest.mark.asyncio
async def test_prompt_version_is_v2() -> None:
    # The prompt contract changed (schema now embedded); prompt_version is
    # bumped to v2 so locked replay of v1-recorded caches fails closed.
    policy = OpenAICompatPolicy(model="gpt-4o", completion=_FakeCompletion())
    decision = await policy.decide(_make_context())
    assert decision.provider_metadata["prompt_version"] == "v2"


@pytest.mark.asyncio
async def test_policy_multiple_choices_fails_closed() -> None:
    content = _build_response_content()
    multi = _OpenAIResponse(
        choices=(
            _Choice(message=_Message(content=content, refusal=None)),
            _Choice(message=_Message(content=content, refusal=None)),
        ),
        id="resp-multi",
        model="gpt-4o",
        system_fingerprint="fp-1",
        usage=None,
    )
    completion = _FakeCompletion([multi, multi])
    policy = OpenAICompatPolicy(
        model="gpt-4o",
        completion=completion,
        max_attempts=2,
    )
    with pytest.raises(OpenAICompatPolicyError):
        await policy.decide(_make_context())
    assert len(completion.calls) == 2


@pytest.mark.asyncio
async def test_policy_non_string_content_fails_closed() -> None:
    non_str = _OpenAIResponse(
        choices=(_Choice(message=_Message(content=12345, refusal=None)),),  # type: ignore[arg-type]
        id="resp-nonstr",
        model="gpt-4o",
        system_fingerprint="fp-1",
        usage=None,
    )
    completion = _FakeCompletion([non_str, non_str])
    policy = OpenAICompatPolicy(
        model="gpt-4o",
        completion=completion,
        max_attempts=2,
    )
    with pytest.raises(OpenAICompatPolicyError):
        await policy.decide(_make_context())
    assert len(completion.calls) == 2


@pytest.mark.asyncio
async def test_policy_message_refusal_fails_closed() -> None:
    # message.refusal with null content: still a refusal, fails closed
    # immediately with the typed error, no retry.
    bad = _message_refusal_response()
    completion = _FakeCompletion([bad, bad])
    policy = OpenAICompatPolicy(
        model="gpt-4o",
        completion=completion,
        max_attempts=2,
    )
    with pytest.raises(OpenAICompatPolicyError):
        await policy.decide(_make_context())
    assert len(completion.calls) == 1


@pytest.mark.asyncio
async def test_policy_length_finish_null_content_does_not_retry() -> None:
    # A reasoning model that exhausts max_tokens on reasoning returns HTTP 200
    # with finish_reason="length" and content=None. This is deterministic for
    # identical inputs — retrying is futile, so fail closed immediately as the
    # typed error with ONE provider call. (Surfaced live against mimo-v2.5-free.)
    exhausted = _OpenAIResponse(
        choices=(_Choice(message=_Message(content=None, refusal=None), finish_reason="length"),),
        id="resp-length-null",
        model="gpt-4o",
        system_fingerprint="fp-1",
        usage={"prompt_tokens": 10, "completion_tokens": 16, "total_tokens": 26},
    )
    completion = _FakeCompletion([exhausted, exhausted])
    policy = OpenAICompatPolicy(
        model="gpt-4o",
        completion=completion,
        max_attempts=3,
    )
    with pytest.raises(OpenAICompatPolicyError) as exc_info:
        await policy.decide(_make_context())
    assert "finish_reason=length" in str(exc_info.value)
    assert len(completion.calls) == 1


@pytest.mark.asyncio
async def test_policy_length_finish_with_content_succeeds() -> None:
    # finish_reason="length" with content PRESENT (truncated-but-present output)
    # is a successful, usable response — not an error. The adapter must not
    # reject it just because finish_reason is "length".
    content = _build_response_content()
    truncated = _OpenAIResponse(
        choices=(_Choice(message=_Message(content=content, refusal=None), finish_reason="length"),),
        id="resp-length-content",
        model="gpt-4o",
        system_fingerprint="fp-1",
        usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    )
    completion = _FakeCompletion([truncated])
    policy = OpenAICompatPolicy(
        model="gpt-4o",
        completion=completion,
        max_attempts=2,
    )
    decision = await policy.decide(_make_context())
    assert decision.report.agent_id == _make_context().agent.id
    assert len(completion.calls) == 1


# ---------------------------------------------------------------------------
# 5. Retry on transient errors
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_policy_retries_on_transient_provider_error() -> None:
    transient = httpx.TimeoutException("timeout")
    completion = _FakeCompletion([transient, _success_response()])
    policy = OpenAICompatPolicy(
        model="gpt-4o",
        completion=completion,
        max_attempts=2,
    )
    context = _make_context()

    decision = await policy.decide(context)

    assert len(completion.calls) == 2
    assert decision.report.agent_id == "worker"


@pytest.mark.asyncio
async def test_policy_retries_on_invalid_model_output() -> None:
    completion = _FakeCompletion([_malformed_json_response(), _success_response()])
    policy = OpenAICompatPolicy(
        model="gpt-4o",
        completion=completion,
        max_attempts=2,
    )
    context = _make_context()

    decision = await policy.decide(context)

    assert len(completion.calls) == 2
    assert decision.report.agent_id == "worker"


@pytest.mark.asyncio
async def test_policy_exhausts_retries_and_raises() -> None:
    transient = httpx.TimeoutException("timeout")
    completion = _FakeCompletion([transient, transient])
    policy = OpenAICompatPolicy(
        model="gpt-4o",
        completion=completion,
        max_attempts=2,
    )
    context = _make_context()

    with pytest.raises(OpenAICompatPolicyError, match="2 attempts"):
        await policy.decide(context)

    assert len(completion.calls) == 2


@pytest.mark.asyncio
async def test_policy_api_connection_error_retries() -> None:
    conn_err = httpx.ConnectError("conn")
    completion = _FakeCompletion([conn_err, _success_response()])
    policy = OpenAICompatPolicy(
        model="gpt-4o",
        completion=completion,
        max_attempts=2,
    )

    decision = await policy.decide(_make_context())
    assert len(completion.calls) == 2
    assert decision.report.agent_id == "worker"


@pytest.mark.asyncio
async def test_policy_rate_limit_error_retries() -> None:
    rl = httpx.HTTPStatusError(
        message="rate limited",
        request=httpx.Request("POST", "http://x"),
        response=httpx.Response(429, request=httpx.Request("POST", "http://x")),
    )
    completion = _FakeCompletion([rl, _success_response()])
    policy = OpenAICompatPolicy(
        model="gpt-4o",
        completion=completion,
        max_attempts=2,
    )

    await policy.decide(_make_context())
    assert len(completion.calls) == 2


@pytest.mark.asyncio
async def test_policy_internal_server_error_retries() -> None:
    ise = httpx.HTTPStatusError(
        message="internal error",
        request=httpx.Request("POST", "http://x"),
        response=httpx.Response(500, request=httpx.Request("POST", "http://x")),
    )
    completion = _FakeCompletion([ise, _success_response()])
    policy = OpenAICompatPolicy(
        model="gpt-4o",
        completion=completion,
        max_attempts=2,
    )

    await policy.decide(_make_context())
    assert len(completion.calls) == 2


# ---------------------------------------------------------------------------
# 5b. Retry backoff for transient errors (degraded-endpoint resilience)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_policy_retries_with_backoff_between_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A degraded endpoint intermittently 500s. The adapter must sleep between
    # retry attempts (not burst-fire) so two clustered 500s don't trivially
    # exhaust max_attempts. This is the mock-invisible behavior fixed here:
    # mock-based tests always succeed on attempt 1, so a missing backoff never
    # surfaced offline.
    monkeypatch.setenv("DISTORTION_RETRY_BACKOFF_SECONDS", "0.5")
    ise = httpx.HTTPStatusError(
        message="internal error",
        request=httpx.Request("POST", "http://x"),
        response=httpx.Response(500, request=httpx.Request("POST", "http://x")),
    )
    completion = _FakeCompletion([ise, _success_response()])
    policy = OpenAICompatPolicy(
        model="gpt-4o",
        completion=completion,
        max_attempts=3,
    )

    sleeps: list[float] = []

    async def _spy_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr("distortion_engine.policy.openai_compat_policy._async_sleep", _spy_sleep)
    decision = await policy.decide(_make_context())

    assert len(completion.calls) == 2  # failed once, then succeeded
    assert len(sleeps) == 1  # exactly one backoff sleep before the retry
    assert sleeps[0] > 0  # not zero (a real backoff occurred)
    assert sleeps[0] <= 1.0  # bounded for a 0.5 base on attempt 1
    assert decision.report.agent_id == "worker"


@pytest.mark.asyncio
async def test_policy_backoff_does_not_sleep_before_first_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # No sleep before attempt 0; backoff only gates retries.
    monkeypatch.setenv("DISTORTION_RETRY_BACKOFF_SECONDS", "0.5")
    sleeps: list[float] = []

    async def _spy_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr("distortion_engine.policy.openai_compat_policy._async_sleep", _spy_sleep)
    completion = _FakeCompletion([_success_response()])
    policy = OpenAICompatPolicy(model="gpt-4o", completion=completion, max_attempts=3)

    await policy.decide(_make_context())
    assert sleeps == []


@pytest.mark.asyncio
async def test_policy_backoff_is_env_tunable_to_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    # DISTORTION_RETRY_BACKOFF_SECONDS=0 -> no delay, determinism for tests.
    monkeypatch.setenv("DISTORTION_RETRY_BACKOFF_SECONDS", "0")
    ise = httpx.HTTPStatusError(
        message="internal error",
        request=httpx.Request("POST", "http://x"),
        response=httpx.Response(500, request=httpx.Request("POST", "http://x")),
    )
    completion = _FakeCompletion([ise, _success_response()])
    policy = OpenAICompatPolicy(
        model="gpt-4o",
        completion=completion,
        max_attempts=2,
    )
    sleeps: list[float] = []

    async def _spy_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr("distortion_engine.policy.openai_compat_policy._async_sleep", _spy_sleep)
    await policy.decide(_make_context())
    assert len(completion.calls) == 2
    # With base 0, every backoff delay is 0 (and there is one sleep before the retry).
    assert sleeps and all(d == 0 for d in sleeps)


@pytest.mark.asyncio
async def test_policy_survives_clustered_transient_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The motivating live scenario: a flaky endpoint 500s twice in a row, then
    # recovers. With backoff + max_attempts>=3 the run completes; the prior
    # zero-delay max_attempts=2 policy aborted. (Live mimic: mimops free
    # endpoint intermittently returns 500 ~1-in-5.)
    monkeypatch.setenv("DISTORTION_RETRY_BACKOFF_SECONDS", "0.1")
    ise = httpx.HTTPStatusError(
        message="internal error",
        request=httpx.Request("POST", "http://x"),
        response=httpx.Response(500, request=httpx.Request("POST", "http://x")),
    )

    async def _spy_sleep(delay: float) -> None:
        await asyncio.sleep(0)

    monkeypatch.setattr("distortion_engine.policy.openai_compat_policy._async_sleep", _spy_sleep)
    completion = _FakeCompletion([ise, ise, _success_response()])
    policy = OpenAICompatPolicy(
        model="gpt-4o",
        completion=completion,
        max_attempts=3,
    )

    decision = await policy.decide(_make_context())
    assert len(completion.calls) == 3
    assert decision.report.agent_id == "worker"


@pytest.mark.asyncio
async def test_policy_schema_debug_capture_when_env_set(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The env-gated diagnostic seam: on schema failure with
    # DISTORTION_DEBUG_SCHEMA=1, the raw content is written to tmp/ so live
    # schema mismatches are diagnosable (otherwise `from None` discards it,
    # per ADR #8). Off by default (no leak in production).
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DISTORTION_DEBUG_SCHEMA", "1")
    bad = _live_wrong_shape_response()
    completion = _FakeCompletion([bad, bad])
    policy = OpenAICompatPolicy(
        model="gpt-4o",
        completion=completion,
        max_attempts=2,
    )
    with pytest.raises(OpenAICompatPolicyError):
        await policy.decide(_make_context())
    debug_files = list(tmp_path.glob("tmp/schema-debug-*.json"))
    assert len(debug_files) >= 1
    payload = json.loads(debug_files[0].read_text())
    assert "content" in payload and "action" in payload["content"]
    assert "context_hash" in payload


@pytest.mark.asyncio
async def test_policy_schema_debug_off_by_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Without the env var, schema failure must NOT write any debug file
    # (no content leak in production runs).
    monkeypatch.delenv("DISTORTION_DEBUG_SCHEMA", raising=False)
    monkeypatch.chdir(tmp_path)
    bad = _live_wrong_shape_response()
    completion = _FakeCompletion([bad, bad])
    policy = OpenAICompatPolicy(
        model="gpt-4o",
        completion=completion,
        max_attempts=2,
    )
    with pytest.raises(OpenAICompatPolicyError):
        await policy.decide(_make_context())
    assert list(tmp_path.glob("tmp/schema-debug-*.json")) == []


# ---------------------------------------------------------------------------
# 6. Non-retryable errors fail immediately
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_policy_auth_error_does_not_retry() -> None:
    auth = httpx.HTTPStatusError(
        message="auth failed",
        request=httpx.Request("POST", "http://x"),
        response=httpx.Response(401, request=httpx.Request("POST", "http://x")),
    )
    completion = _FakeCompletion([auth])
    policy = OpenAICompatPolicy(
        model="gpt-4o",
        completion=completion,
        max_attempts=3,
    )

    with pytest.raises(OpenAICompatPolicyError) as exc_info:
        await policy.decide(_make_context())

    assert exc_info.value.__cause__ is auth
    assert "401" in str(exc_info.value)
    assert len(completion.calls) == 1


@pytest.mark.asyncio
async def test_policy_not_found_error_does_not_retry() -> None:
    not_found = httpx.HTTPStatusError(
        message="not found",
        request=httpx.Request("POST", "http://x"),
        response=httpx.Response(404, request=httpx.Request("POST", "http://x")),
    )
    completion = _FakeCompletion([not_found])
    policy = OpenAICompatPolicy(
        model="gpt-4o",
        completion=completion,
        max_attempts=3,
    )

    with pytest.raises(OpenAICompatPolicyError) as exc_info:
        await policy.decide(_make_context())

    assert exc_info.value.__cause__ is not_found
    assert "404" in str(exc_info.value)
    assert len(completion.calls) == 1


@pytest.mark.asyncio
async def test_policy_bad_request_does_not_retry() -> None:
    bad_req = httpx.HTTPStatusError(
        message="bad request",
        request=httpx.Request("POST", "http://x"),
        response=httpx.Response(400, request=httpx.Request("POST", "http://x")),
    )
    completion = _FakeCompletion([bad_req])
    policy = OpenAICompatPolicy(
        model="gpt-4o",
        completion=completion,
        max_attempts=3,
    )

    with pytest.raises(OpenAICompatPolicyError) as exc_info:
        await policy.decide(_make_context())

    assert exc_info.value.__cause__ is bad_req
    assert "400" in str(exc_info.value)
    assert len(completion.calls) == 1


@pytest.mark.asyncio
async def test_policy_permission_error_does_not_retry() -> None:
    perm = httpx.HTTPStatusError(
        message="permission denied",
        request=httpx.Request("POST", "http://x"),
        response=httpx.Response(403, request=httpx.Request("POST", "http://x")),
    )
    completion = _FakeCompletion([perm])
    policy = OpenAICompatPolicy(
        model="gpt-4o",
        completion=completion,
        max_attempts=3,
    )

    with pytest.raises(OpenAICompatPolicyError) as exc_info:
        await policy.decide(_make_context())

    assert exc_info.value.__cause__ is perm
    assert "403" in str(exc_info.value)
    assert len(completion.calls) == 1


@pytest.mark.asyncio
async def test_policy_non_retryable_error_does_not_retry_other_statuses() -> None:
    """Every non-retryable HTTP status fails closed as the typed error.

    Guards the contract surfaced by live testing: callers relying on
    ``OpenAICompatPolicyError`` must catch every terminal failure, not just
    retry-exhaustion. A raw ``httpx.HTTPStatusError`` must never escape.
    """
    req = httpx.Request("POST", "http://x")
    for status in (405, 409, 413, 422):
        err = httpx.HTTPStatusError(
            message=f"status {status}",
            request=req,
            response=httpx.Response(status, request=req),
        )
        completion = _FakeCompletion([err])
        policy = OpenAICompatPolicy(model="gpt-4o", completion=completion, max_attempts=3)

        with pytest.raises(OpenAICompatPolicyError) as exc_info:
            await policy.decide(_make_context())

        assert exc_info.value.__cause__ is err
        assert str(status) in str(exc_info.value)
        assert len(completion.calls) == 1


# ---------------------------------------------------------------------------
# 7. Unexpected programming errors propagate immediately
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_policy_unexpected_type_error_does_not_retry() -> None:
    class _UnexpectedTypeError(TypeError):
        pass

    completion = _FakeCompletion([_UnexpectedTypeError("bad")])
    policy = OpenAICompatPolicy(
        model="gpt-4o",
        completion=completion,
        max_attempts=3,
    )
    context = _make_context()

    with pytest.raises(_UnexpectedTypeError):
        await policy.decide(context)

    assert len(completion.calls) == 1


# ---------------------------------------------------------------------------
# 8. File cache reuses decision without provider call
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_file_cache_reuses_decision_without_provider_call(tmp_path: Path) -> None:
    context = _make_context()
    path = tmp_path / "decisions.json"

    first_completion = _FakeCompletion()
    first = OpenAICompatPolicy(
        model="gpt-4o",
        completion=first_completion,
        decision_cache=FileDecisionCache(path),
    )
    expected = await first.decide(context)

    second_completion = _FakeCompletion()
    second = OpenAICompatPolicy(
        model="gpt-4o",
        completion=second_completion,
        decision_cache=FileDecisionCache(path),
    )
    actual = await second.decide(context)

    assert actual == expected
    assert second_completion.calls == []


@pytest.mark.asyncio
async def test_file_cache_deletion_persists(tmp_path: Path) -> None:
    path = tmp_path / "decisions.json"
    cache = FileDecisionCache(path)
    decision = await OpenAICompatPolicy(
        model="gpt-4o",
        completion=_FakeCompletion(),
    ).decide(_make_context())
    cache["context"] = decision

    del cache["context"]

    assert "context" not in FileDecisionCache(path)


# ---------------------------------------------------------------------------
# 9. Locked cache miss raises DecisionCacheMiss
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_locked_policy_raises_decision_cache_miss(tmp_path: Path) -> None:
    context = _make_context()
    policy = OpenAICompatPolicy(
        model="gpt-4o",
        completion=_FakeCompletion(),
        decision_cache=FileDecisionCache(tmp_path / "decisions.json"),
        cache_mode="locked",
    )

    with pytest.raises(DecisionCacheMiss):
        await policy.decide(context)


@pytest.mark.asyncio
async def test_locked_policy_needs_no_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DISTORTION_API_KEY", raising=False)
    context = _make_context()
    policy = OpenAICompatPolicy(
        model="gpt-4o",
        decision_cache=FileDecisionCache(tmp_path / "decisions.json"),
        cache_mode="locked",
    )

    with pytest.raises(DecisionCacheMiss):
        await policy.decide(context)


# ---------------------------------------------------------------------------
# 10. Concurrent identical contexts make one provider call
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_identical_contexts_make_one_provider_call(tmp_path: Path) -> None:
    context = _make_context()
    completion = _FakeCompletion()
    policy = OpenAICompatPolicy(
        model="gpt-4o",
        completion=completion,
        decision_cache=FileDecisionCache(tmp_path / "decisions.json"),
    )

    first, second = await asyncio.gather(
        policy.decide(context),
        policy.decide(context),
    )

    assert first == second
    assert len(completion.calls) == 1


# ---------------------------------------------------------------------------
# 11. Record / locked mode semantics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_record_mode_stores_decision_in_cache(tmp_path: Path) -> None:
    context = _make_context()
    path = tmp_path / "decisions.json"
    cache = FileDecisionCache(path)
    policy = OpenAICompatPolicy(
        model="gpt-4o",
        completion=_FakeCompletion(),
        decision_cache=cache,
        cache_mode="record",
    )

    await policy.decide(context)

    assert len(cache) == 1


@pytest.mark.asyncio
async def test_locked_mode_does_not_invoke_completion(tmp_path: Path) -> None:
    context = _make_context()
    completion = _FakeCompletion()
    policy = OpenAICompatPolicy(
        model="gpt-4o",
        completion=completion,
        decision_cache=FileDecisionCache(tmp_path / "decisions.json"),
        cache_mode="locked",
    )

    with pytest.raises(DecisionCacheMiss):
        await policy.decide(context)

    assert completion.calls == []


# ---------------------------------------------------------------------------
# 12. Fingerprint changes for parameter variations
# ---------------------------------------------------------------------------


def _fingerprint(model: str = "gpt-4o", **kwargs: Any) -> str:
    policy = OpenAICompatPolicy(model=model, completion=_FakeCompletion(), **kwargs)
    return policy.fingerprint


@pytest.mark.asyncio
async def test_fingerprint_changes_for_model() -> None:
    fp1 = _fingerprint(model="gpt-4o")
    fp2 = _fingerprint(model="gpt-4o-mini")
    assert fp1 != fp2


@pytest.mark.asyncio
async def test_fingerprint_changes_for_api_base() -> None:
    fp1 = _fingerprint()
    fp2 = _fingerprint(api_base="https://example.invalid/v1")
    assert fp1 != fp2


@pytest.mark.asyncio
async def test_fingerprint_changes_for_generation_parameters() -> None:
    fp1 = _fingerprint()
    fp2 = _fingerprint(generation_parameters={"temperature": 0.5})
    assert fp1 != fp2


@pytest.mark.asyncio
async def test_fingerprint_changes_for_timeout() -> None:
    fp1 = _fingerprint()
    fp2 = _fingerprint(timeout_seconds=60.0)
    assert fp1 != fp2


@pytest.mark.asyncio
async def test_fingerprint_changes_for_max_attempts() -> None:
    fp1 = _fingerprint()
    fp2 = _fingerprint(max_attempts=3)
    assert fp1 != fp2


@pytest.mark.asyncio
async def test_fingerprint_starts_with_openai_compat_prefix() -> None:
    fp = _fingerprint()
    assert fp.startswith("openai_compat:")


@pytest.mark.asyncio
async def test_request_body_pins_temperature_zero_by_default() -> None:
    # The adapter must request temperature:0 by default so the live model's
    # sampling is deterministic-intent (a single LLM sample is the engine's
    # measurement; unchecked sampling made reported health swing 0.0..1.0 on
    # byte-identical input — mock-invisible). Mock-blind: a fake completion
    # always returns one decision regardless of temperature.
    completion = _FakeCompletion()
    policy = OpenAICompatPolicy(model="gpt-4o", completion=completion)
    await policy.decide(_make_context())
    assert completion.calls[0]["body"]["temperature"] == 0.0


@pytest.mark.asyncio
async def test_request_body_temperature_overridable_via_generation_parameters() -> None:
    completion = _FakeCompletion()
    policy = OpenAICompatPolicy(
        model="gpt-4o",
        completion=completion,
        generation_parameters={"temperature": 0.7},
    )
    await policy.decide(_make_context())
    # explicit generation_parameters wins over the 0.0 default
    assert completion.calls[0]["body"]["temperature"] == 0.7


@pytest.mark.asyncio
async def test_request_body_temperature_overridable_via_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DISTORTION_TEMPERATURE", "0.4")
    completion = _FakeCompletion()
    policy = OpenAICompatPolicy(model="gpt-4o", completion=completion)
    await policy.decide(_make_context())
    assert completion.calls[0]["body"]["temperature"] == 0.4


@pytest.mark.asyncio
async def test_fingerprint_changes_for_temperature() -> None:
    # temperature is a fingerprint input, so a temperature change invalidates
    # recorded caches (a stale sampling contract must not replay).
    fp1 = _fingerprint()
    fp2 = _fingerprint(generation_parameters={"temperature": 0.5})
    assert fp1 != fp2


# ---------------------------------------------------------------------------
# 13. Fingerprint does not change for cache_mode or completion callable
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fingerprint_unchanged_for_cache_mode() -> None:
    fp1 = _fingerprint(cache_mode="record")
    fp2 = _fingerprint(cache_mode="locked")
    assert fp1 == fp2


@pytest.mark.asyncio
async def test_fingerprint_unchanged_for_injected_completion() -> None:
    c1 = _FakeCompletion()
    c2 = _FakeCompletion()
    fp1 = OpenAICompatPolicy(model="gpt-4o", completion=c1).fingerprint
    fp2 = OpenAICompatPolicy(model="gpt-4o", completion=c2).fingerprint
    assert fp1 == fp2


# ---------------------------------------------------------------------------
# 14. Credentials never appear in fingerprint, metadata, or exceptions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_api_keys_never_appear_in_fingerprint() -> None:
    fp = _fingerprint(api_key="sk-secret-key-value-12345")
    assert "sk-secret" not in fp
    assert "12345" not in fp


@pytest.mark.asyncio
async def test_api_keys_never_appear_in_decision(tmp_path: Path) -> None:
    completion = _FakeCompletion()
    policy = OpenAICompatPolicy(
        model="gpt-4o",
        completion=completion,
        api_key="sk-super-secret-99999",
        decision_cache=FileDecisionCache(tmp_path / "decisions.json"),
    )
    decision = await policy.decide(_make_context())

    serialized = decision.model_dump_json()
    assert "sk-super-secret" not in serialized
    assert "99999" not in serialized


@pytest.mark.asyncio
async def test_api_keys_never_appear_in_exception_message() -> None:
    transient = httpx.TimeoutException("timeout")
    completion = _FakeCompletion([transient, transient])
    policy = OpenAICompatPolicy(
        model="gpt-4o",
        completion=completion,
        api_key="sk-leaked-key-abc123",
        max_attempts=2,
    )

    with pytest.raises(OpenAICompatPolicyError) as exc_info:
        await policy.decide(_make_context())

    assert "sk-leaked-key" not in str(exc_info.value)
    exc = exc_info.value.__cause__
    while exc is not None:
        assert "sk-leaked-key" not in str(exc)
        exc = getattr(exc, "__cause__", None)


@pytest.mark.asyncio
async def test_raw_api_base_url_never_appear_in_fingerprint() -> None:
    url = "https://custom-endpoint.internal:9000/v1/chat"
    fp = _fingerprint(api_base=url)
    assert "custom-endpoint.internal" not in fp
    assert "9000" not in fp
    assert "chat" not in fp


@pytest.mark.asyncio
async def test_api_base_identity_still_affects_fingerprint() -> None:
    fp1 = _fingerprint(api_base="https://alpha.example.invalid/v1")
    fp2 = _fingerprint(api_base="https://beta.example.invalid/v1")
    assert fp1 != fp2


# ---------------------------------------------------------------------------
# 15. Provider metadata uses whitelist
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_provider_metadata_uses_whitelist() -> None:
    completion = _FakeCompletion()
    policy = OpenAICompatPolicy(
        model="gpt-4o",
        completion=completion,
    )
    decision = await policy.decide(_make_context())

    metadata = decision.provider_metadata
    expected_keys = {
        "cache_key",
        "reproducibility",
        "policy",
        "policy_fingerprint",
        "requested_model",
        "response_model",
        "response_id",
        "system_fingerprint",
        "api_base_id",
        "structured_output_mode",
        "prompt_version",
        "schema_version",
        "usage",
    }
    assert set(metadata.keys()) == expected_keys
    assert metadata["policy"] == "openai_compat"
    assert metadata["reproducibility"] == "captured-provider-decision"
    assert metadata["structured_output_mode"] == "json_object"


@pytest.mark.asyncio
async def test_provider_metadata_does_not_copy_hidden_fields() -> None:
    from types import SimpleNamespace

    content = _build_response_content()
    extra_response = SimpleNamespace(
        choices=[
            _Choice(message=_Message(content=content, refusal=None)),
        ],
        id="resp-extra",
        model="gpt-4o",
        system_fingerprint="fp-abc123",
        hidden_fields={"should_not_leak": True},
        usage={"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
    )
    completion = _FakeCompletion([extra_response])
    policy = OpenAICompatPolicy(
        model="gpt-4o",
        completion=completion,
    )
    decision = await policy.decide(_make_context())

    metadata = decision.provider_metadata
    assert "hidden_fields" not in metadata
    assert metadata.get("system_fingerprint") == "fp-abc123"


# ---------------------------------------------------------------------------
# 16. Response model and system_fingerprint capture
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_provider_metadata_captures_response_model() -> None:
    from types import SimpleNamespace

    content = _build_response_content()
    resp = SimpleNamespace(
        choices=[_Choice(message=_Message(content=content, refusal=None))],
        id="resp-rm",
        model="claude-3-5-sonnet",
        system_fingerprint="fp-test-123",
        usage={"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
    )
    completion = _FakeCompletion([resp])
    policy = OpenAICompatPolicy(
        model="gpt-4o",
        completion=completion,
    )
    decision = await policy.decide(_make_context())

    assert decision.provider_metadata["response_model"] == "claude-3-5-sonnet"
    assert decision.provider_metadata["system_fingerprint"] == "fp-test-123"


# ---------------------------------------------------------------------------
# 17. Usage normalization
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_provider_metadata_usage_is_normalized_integers() -> None:
    from types import SimpleNamespace

    content = _build_response_content()
    resp = SimpleNamespace(
        choices=[_Choice(message=_Message(content=content, refusal=None))],
        id="resp-usage",
        model="gpt-4o",
        system_fingerprint="fp-1",
        usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    )
    completion = _FakeCompletion([resp])
    policy = OpenAICompatPolicy(
        model="gpt-4o",
        completion=completion,
    )
    decision = await policy.decide(_make_context())

    usage = decision.provider_metadata["usage"]
    assert usage == {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}


# ---------------------------------------------------------------------------
# 18. Policy name
# ---------------------------------------------------------------------------


def test_policy_name_is_openai_compat() -> None:
    policy = OpenAICompatPolicy(
        model="gpt-4o",
        completion=_FakeCompletion(),
    )
    assert policy.name == "openai_compat"


# ---------------------------------------------------------------------------
# 19. Constructor validation: model
# ---------------------------------------------------------------------------


def test_empty_model_is_rejected() -> None:
    with pytest.raises(ValueError):
        OpenAICompatPolicy(model="", completion=_FakeCompletion())


def test_model_with_provider_prefix_is_rejected() -> None:
    with pytest.raises(ValueError, match="provider prefix"):
        OpenAICompatPolicy(model="openai/gpt-4o", completion=_FakeCompletion())


def test_bare_model_is_accepted() -> None:
    policy = OpenAICompatPolicy(model="gpt-4o", completion=_FakeCompletion())
    assert policy.model == "gpt-4o"


def test_model_leading_whitespace_is_rejected() -> None:
    with pytest.raises(ValueError, match="whitespace"):
        OpenAICompatPolicy(model=" gpt-4o", completion=_FakeCompletion())


def test_model_trailing_whitespace_is_rejected() -> None:
    with pytest.raises(ValueError, match="whitespace"):
        OpenAICompatPolicy(model="gpt-4o ", completion=_FakeCompletion())


# ---------------------------------------------------------------------------
# 20. Constructor validation: timeout
# ---------------------------------------------------------------------------


def test_positive_timeout_is_required() -> None:
    with pytest.raises(ValueError):
        OpenAICompatPolicy(
            model="gpt-4o",
            completion=_FakeCompletion(),
            timeout_seconds=-1.0,
        )


def test_zero_timeout_is_rejected() -> None:
    with pytest.raises(ValueError):
        OpenAICompatPolicy(
            model="gpt-4o",
            completion=_FakeCompletion(),
            timeout_seconds=0.0,
        )


def test_nan_timeout_is_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        OpenAICompatPolicy(
            model="gpt-4o",
            completion=_FakeCompletion(),
            timeout_seconds=float("nan"),
        )


def test_inf_timeout_is_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        OpenAICompatPolicy(
            model="gpt-4o",
            completion=_FakeCompletion(),
            timeout_seconds=float("inf"),
        )


def test_negative_inf_timeout_is_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        OpenAICompatPolicy(
            model="gpt-4o",
            completion=_FakeCompletion(),
            timeout_seconds=float("-inf"),
        )


# ---------------------------------------------------------------------------
# 21. Constructor validation: max_attempts, cache_mode
# ---------------------------------------------------------------------------


def test_max_attempts_at_least_one() -> None:
    with pytest.raises(ValueError):
        OpenAICompatPolicy(
            model="gpt-4o",
            completion=_FakeCompletion(),
            max_attempts=0,
        )


def test_invalid_cache_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="cache_mode"):
        OpenAICompatPolicy(
            model="gpt-4o",
            completion=_FakeCompletion(),
            cache_mode="invalid",  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# 22. Constructor validation: api_base
# ---------------------------------------------------------------------------


def test_api_base_with_userinfo_is_rejected() -> None:
    with pytest.raises(ValueError, match="userinfo|user.?info|credentials?.*url"):
        OpenAICompatPolicy(
            model="gpt-4o",
            api_base="https://user:pass@example.invalid/v1",
            completion=_FakeCompletion(),
        )


def test_api_base_with_query_is_rejected() -> None:
    with pytest.raises(ValueError, match="query|parameter"):
        OpenAICompatPolicy(
            model="gpt-4o",
            api_base="https://example.invalid/v1?key=secret",
            completion=_FakeCompletion(),
        )


def test_api_base_with_fragment_is_rejected() -> None:
    with pytest.raises(ValueError, match="fragment"):
        OpenAICompatPolicy(
            model="gpt-4o",
            api_base="https://example.invalid/v1#section",
            completion=_FakeCompletion(),
        )


def test_api_base_missing_scheme_is_rejected() -> None:
    with pytest.raises(ValueError, match="scheme|http|url"):
        OpenAICompatPolicy(
            model="gpt-4o",
            api_base="example.invalid/v1",
            completion=_FakeCompletion(),
        )


def test_api_base_missing_hostname_is_rejected() -> None:
    with pytest.raises(ValueError):
        OpenAICompatPolicy(
            model="gpt-4o",
            api_base="https:///v1",
            completion=_FakeCompletion(),
        )


def test_api_base_exception_never_contains_secret_url() -> None:
    """A URL with embedded secrets must not appear in any validation message."""
    url_with_secret = "https://user:s3cret@hidden.internal:8443/v1?token=abc123"
    with pytest.raises(ValueError) as exc_info:
        OpenAICompatPolicy(
            model="gpt-4o",
            api_base=url_with_secret,
            completion=_FakeCompletion(),
        )
    msg = str(exc_info.value)
    assert "s3cret" not in msg
    assert "hidden.internal" not in msg
    assert "abc123" not in msg
    assert "user:s3cret" not in msg


# ---------------------------------------------------------------------------
# 23. Constructor validation: generation_parameters
# ---------------------------------------------------------------------------


def test_secret_generation_keys_are_rejected() -> None:
    with pytest.raises(ValueError, match="reserved|secret|token|credential"):
        OpenAICompatPolicy(
            model="gpt-4o",
            completion=_FakeCompletion(),
            generation_parameters={"api_key": "sk-leaked"},
        )


def test_generation_parameter_secret_key_rejected() -> None:
    with pytest.raises(ValueError, match="reserved|secret|token|credential"):
        OpenAICompatPolicy(
            model="gpt-4o",
            completion=_FakeCompletion(),
            generation_parameters={"my_secret_key": "value"},
        )


def test_generation_parameter_token_key_rejected() -> None:
    with pytest.raises(ValueError, match="reserved|secret|token|credential"):
        OpenAICompatPolicy(
            model="gpt-4o",
            completion=_FakeCompletion(),
            generation_parameters={"access_token": "value"},
        )


def test_generation_parameter_password_key_rejected() -> None:
    with pytest.raises(ValueError, match="reserved|secret|token|credential"):
        OpenAICompatPolicy(
            model="gpt-4o",
            completion=_FakeCompletion(),
            generation_parameters={"db_password": "value"},
        )


def test_generation_parameter_credential_key_rejected() -> None:
    with pytest.raises(ValueError, match="reserved|secret|token|credential"):
        OpenAICompatPolicy(
            model="gpt-4o",
            completion=_FakeCompletion(),
            generation_parameters={"credential_file": "/path"},
        )


def test_generation_parameter_authorization_key_rejected() -> None:
    with pytest.raises(ValueError, match="reserved|secret|token|credential"):
        OpenAICompatPolicy(
            model="gpt-4o",
            completion=_FakeCompletion(),
            generation_parameters={"authorization_header": "Bearer x"},
        )


def test_generation_parameter_access_token_still_rejected() -> None:
    # The token marker must still catch genuine credential-bearing keys
    # (token as a whole word), even after max_tokens is allowlisted.
    with pytest.raises(ValueError, match="reserved|secret|token|credential"):
        OpenAICompatPolicy(
            model="gpt-4o",
            completion=_FakeCompletion(),
            generation_parameters={"access_token": "value"},
        )


@pytest.mark.parametrize(
    "key",
    [
        "max_tokens",
        "max_completion_tokens",
        "max_output_tokens",
        "MAX_TOKENS",
        "Max_Completion_Tokens",
        "MAX_OUTPUT_TOKENS",
    ],
)
def test_generation_parameter_token_budget_keys_accepted(key: str) -> None:
    # Output-token-budget keys are legitimate OpenAI generation parameters,
    # not credentials. The substring `token` marker must not false-positive on
    # them (regression: previously rejected by _has_secret_marker).
    policy = OpenAICompatPolicy(
        model="gpt-4o",
        completion=_FakeCompletion(),
        generation_parameters={key: 512},
    )
    assert policy.generation_parameters[key] == 512


@pytest.mark.asyncio
async def test_max_tokens_reaches_request_body() -> None:
    completion = _FakeCompletion()
    policy = OpenAICompatPolicy(
        model="gpt-4o",
        completion=completion,
        generation_parameters={"max_tokens": 512},
    )
    await policy.decide(_make_context())
    assert completion.calls[0]["body"]["max_tokens"] == 512


# ---------------------------------------------------------------------------
# 23b. Cache-lock lifecycle: per-key locks must not accumulate unbounded
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cache_lock_removed_after_record(tmp_path: Path) -> None:
    """The per-cache_key asyncio.Lock must be cleaned up once the decision is
    recorded, so a long-running policy does not leak one Lock per unique
    context. Regression: _cache_locks grew without bound."""
    completion = _FakeCompletion()
    policy = OpenAICompatPolicy(
        model="gpt-4o",
        completion=completion,
        decision_cache=FileDecisionCache(tmp_path / "decisions.json"),
        cache_mode="record",
    )

    await policy.decide(_make_context(agent_id="agent-1"))
    # A second distinct context grows the dict transiently...
    await policy.decide(_make_context(agent_id="agent-2"))
    # ...but completed entries are removed, so the dict does not retain
    # every key ever seen.
    assert len(policy._cache_locks) <= 1, f"cache locks leaked: {list(policy._cache_locks)}"


# ---------------------------------------------------------------------------
# 24. Security: provider response content must not leak into exceptions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_refusal_content_never_leaks_into_exception() -> None:
    """Refusal text must not appear in validation exceptions or cause chain."""
    secret_marker = "ULTRA_SECRET_REFUSAL_CONTENT_7f3a"
    resp = _OpenAIResponse(
        choices=(
            _Choice(message=_Message(content=None, refusal=f"I cannot help with {secret_marker}")),
        ),
        id="resp-refusal-leak",
        model="gpt-4o",
        system_fingerprint="fp-1",
        usage=None,
    )
    completion = _FakeCompletion([resp])
    policy = OpenAICompatPolicy(
        model="gpt-4o",
        completion=completion,
        max_attempts=1,
    )

    with pytest.raises(RuntimeError) as exc_info:
        await policy.decide(_make_context())

    exc = exc_info.value
    while exc is not None:
        assert secret_marker not in str(exc), (
            f"Provider response content leaked into exception: {exc}"
        )
        exc = exc.__cause__


@pytest.mark.asyncio
async def test_malformed_content_never_leaks_into_exception() -> None:
    """Malformed model output must not appear in validation exceptions or cause chain."""
    secret_marker = "ULTRA_SECRET_OUTPUT_9e2b"
    resp = _OpenAIResponse(
        choices=(_Choice(message=_Message(content=secret_marker, refusal=None)),),
        id="resp-malformed-leak",
        model="gpt-4o",
        system_fingerprint="fp-1",
        usage=None,
    )
    completion = _FakeCompletion([resp])
    policy = OpenAICompatPolicy(
        model="gpt-4o",
        completion=completion,
        max_attempts=1,
    )

    with pytest.raises(RuntimeError) as exc_info:
        await policy.decide(_make_context())

    exc = exc_info.value
    while exc is not None:
        assert secret_marker not in str(exc), (
            f"Provider response content leaked into exception: {exc}"
        )
        exc = exc.__cause__


@pytest.mark.asyncio
async def test_schema_invalid_content_never_leaks_into_exception() -> None:
    """Schema validation detail must not leak raw model output into exceptions."""
    secret_marker = "ULTRA_SECRET_SCHEMA_4d1c"
    bad_content = json.dumps(
        {
            "health": {"progress": 0.5, "quality": 0.4, "schedule": 0.3, "reliability": 0.2},
            "confidence": 999.0,
            "escalate": False,
            "explanation": secret_marker,
            "actions": [],
            "concerns": [],
        }
    )
    resp = _OpenAIResponse(
        choices=(_Choice(message=_Message(content=bad_content, refusal=None)),),
        id="resp-schema-leak",
        model="gpt-4o",
        system_fingerprint="fp-1",
        usage=None,
    )
    completion = _FakeCompletion([resp])
    policy = OpenAICompatPolicy(
        model="gpt-4o",
        completion=completion,
        max_attempts=1,
    )

    with pytest.raises(RuntimeError) as exc_info:
        await policy.decide(_make_context())

    exc = exc_info.value
    while exc is not None:
        assert secret_marker not in str(exc), (
            f"Provider response content leaked into exception: {exc}"
        )
        exc = exc.__cause__


# ---------------------------------------------------------------------------
# 25. Falsey callable is not replaced by default
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_falsey_callable_is_not_replaced_by_default() -> None:
    """A valid but falsey callable object must be used, not silently replaced."""
    call_count = 0

    class _FalseyCallable:
        """A callable that is falsey but valid."""

        def __bool__(self) -> bool:
            return False

        async def __call__(
            self, *, url: str, headers: Any, body: Any, timeout: float
        ) -> _OpenAIResponse:
            nonlocal call_count
            call_count += 1
            return _success_response()

    falsey = _FalseyCallable()
    policy = OpenAICompatPolicy(
        model="gpt-4o",
        completion=falsey,
    )
    decision = await policy.decide(_make_context())

    assert call_count == 1
    assert decision.report.agent_id == "worker"
