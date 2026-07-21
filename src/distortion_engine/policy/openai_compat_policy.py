"""OpenAI-compatible policy adapter.

A single lightweight HTTP transport for any OpenAI-compatible chat-completions
endpoint (``{api_base}/chat/completions``), configured with a bare model name,
an API base URL, and an API key. Replaces the former multi-provider LiteLLM
adapter. No provider routing, no SigV4, no boto3: this module talks one
contract, straight over httpx.

Replay integrity is unchanged from the prior adapter:

- The cache key is ``sha256(canonical_json(authorized context)``.
- ``record`` mode captures provider decisions; ``locked`` mode refuses to call
  the provider for an unseen context (fail-closed).
- The policy fingerprint locks the adapter contract that produced a recorded
  decision; a stale cache recorded under an older contract cannot match the
  lean ``openai-compat-v1`` fingerprint, so ``locked`` replay fails closed.

The model-facing schema (:class:`AgentOutput`) is byte-identical to the
prior adapter. The system instructions (:data:`INSTRUCTIONS`) now embed the
``AgentOutput`` JSON schema and a concrete example (prompt version v2): the
prior prompt described no shape, so a live model invented its own field set
and every call failed ``extra="forbid"``. The endpoint accepts but does not
enforce ``response_format`` ``json_schema``, so the prompt is the lever that
works. ``prompt_version`` is bumped to v2 and ``prompt_sha256`` changes
accordingly, so ``locked`` replay of caches recorded under the v1 prompt
fails closed by design (a stale prompt contract must not replay).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

import httpx

from distortion_engine.domain.reports import Report
from distortion_engine.events.store import canonical_json
from distortion_engine.policy.models import (
    AgentContext,
    AgentMemory,
    AgentOutput,
    PolicyDecision,
)

# The structured output contract the model must return. Built from the
# AgentOutput Pydantic schema so the prompt can never silently drift from
# the schema the adapter validates against (the live-only bug this fixes:
# with no schema in the prompt, the model invented its own field set —
# {"action","reason","report"} — and every call failed extra="forbid").
# A concrete example anchors the model on array-vs-object field shapes
# (actions/concerns are arrays, health is a nested object), which a bare
# schema description leaves ambiguous. response_format={"type":"json_object"}
# only constrains the response to *valid* JSON, not to a *shape*; the
# opencode/zen endpoint accepts but does not enforce json_schema, so the
# prompt is the only lever that works (verified live against mimo-v2.5-free).
_AGENT_OUTPUT_SCHEMA_JSON = json.dumps(AgentOutput.model_json_schema(), indent=2, sort_keys=True)

_AGENT_OUTPUT_EXAMPLE = json.dumps(
    {
        "health": {
            "progress": 0.4,
            "quality": 0.6,
            "schedule": 0.5,
            "reliability": 0.7,
        },
        "confidence": 0.5,
        "escalate": False,
        "explanation": "Initial assessment of the scope's status.",
        "actions": [{"kind": "investigate", "item_id": None, "amount": 0.0}],
        "concerns": ["schedule risk"],
    },
    indent=2,
)

INSTRUCTIONS = f"""You are a persistent employee in a simulated company.

Assess the status available to your role, choose zero or more permitted operational actions,
and prepare the status report you would send to your manager. Balance the role objectives,
personal characteristics, evaluation conditions, and evidence supplied in the input. Do not
invent observations or actions.

You MUST respond with a single JSON object that validates against this exact schema
(no other fields, no surrounding prose, no markdown code fences):

{_AGENT_OUTPUT_SCHEMA_JSON}

Field rules:
- "health" is a nested object with exactly progress, quality, schedule, reliability
  (each a float in [0, 1]).
- "confidence" is a float in [0, 1].
- "escalate" is a boolean.
- "explanation" is a string of 1 to 800 characters.
- "actions" is an ARRAY of action objects (use [] for no actions); each action has
  "kind" (one of: work, test, remediate, release, delay_release, reduce_scope,
  investigate, add_qa_capacity), optional "item_id" (string or null), and
  "amount" (float >= 0).
- "concerns" is an ARRAY of strings (use [] for no concerns).

Example of a valid response:
{_AGENT_OUTPUT_EXAMPLE}

Return ONLY the JSON object."""

# Default public OpenAI-compatible endpoint when no api_base is configured.
_DEFAULT_API_BASE = "https://api.openai.com/v1"

# HTTP response status codes treated as transient (retryable) transport errors.
_RETRYABLE_HTTP_STATUS: frozenset[int] = frozenset({429, 500, 502, 503, 504})

# Backoff between retryable attempts. A degraded OpenAI-compatible endpoint can
# intermittently return HTTP 500 (or drop the connection) on a fraction of
# otherwise-identical requests — two consecutive 500s with the prior
# zero-delay, max_attempts=2 policy aborted entire runs even though every
# request was individually transient. We sleep before each retry with
# exponential growth capped at a ceiling, plus a small jitter so concurrent
# callers don't thunder-herd the endpoint. Env-tunable so live probing against
# a flaky free endpoint can raise it without a code change. Mock-based tests
# set the base to 0 implicitly (no env) — but the seam is still exercised.
_RETRY_BASE_DELAY_SECONDS = 0.5
_RETRY_MAX_DELAY_SECONDS = 8.0


async def _retry_backoff_delay(attempt: int) -> float:
    """Exponential backoff with jitter for the given (0-indexed) attempt number.

    Reads the base delay from ``DISTORTION_RETRY_BACKOFF_SECONDS`` when set so
    live runs against a flaky endpoint can tune it; tests leave it default.
    Returns 0.0 for attempt 0 (no delay before the first call).
    """
    if attempt <= 0:
        return 0.0
    env_base = os.environ.get("DISTORTION_RETRY_BACKOFF_SECONDS", "").strip()
    base = float(env_base) if env_base else _RETRY_BASE_DELAY_SECONDS
    delay = min(base * (2.0 ** (attempt - 1)), _RETRY_MAX_DELAY_SECONDS)
    # Deterministic-ish jitter: ±25% using a hash of attempt so tests stay
    # reproducible (no Math.random-equivalent nondeterminism in the retry).
    import hashlib

    h = int(hashlib.sha256(f"{attempt}".encode()).hexdigest(), 16) % 1000
    jitter = 1.0 + ((h / 1000.0) - 0.5) / 2.0  # 0.75 .. 1.25
    return delay * jitter


async def _async_sleep(delay: float) -> None:
    """Module-private sleep seam so tests can intercept the retry backoff
    without shadowing the global ``asyncio.sleep`` (which completion fakes may
    themselves call). Production delegates to ``asyncio.sleep``."""
    await asyncio.sleep(delay)


# Reserved keys that must not appear in generation_parameters.
# Covers transport, credential, control, and response-manipulation fields.
_RESERVED_GENERATION_KEYS: frozenset[str] = frozenset(
    {
        # Transport / request-shape keys
        "model",
        "messages",
        "response_format",
        "stream",
        "n",
        "tool_choice",
        "tools",
        # Endpoint / credential keys (handled by the envelope, not the body)
        "base_url",
        "api_base",
        "api_key",
        "url",
        "headers",
        "timeout",
        "max_retries",
        "request_timeout",
        # Caching / callback / metadata keys
        "cache",
        "caching",
        "success_callback",
        "failure_callback",
        "callbacks",
        "logger",
        "metadata",
        "client",
        "model_list",
        # Response / control manipulation keys
        "modify_params",
    }
)

# Patterns that indicate a secret-bearing key name.  Any generation_parameter
# key whose lowercased name matches one of these is rejected.
_SECRET_KEY_MARKERS: tuple[re.Pattern[str], ...] = (
    re.compile(r"secret", re.IGNORECASE),
    re.compile(r"token", re.IGNORECASE),
    re.compile(r"password", re.IGNORECASE),
    re.compile(r"credential", re.IGNORECASE),
    re.compile(r"authorization", re.IGNORECASE),
)

# Legitimate generation_parameter names that contain a secret-marker
# substring (e.g. "token") but are NOT credentials — they are OpenAI
# output-budget parameters. Exempted from _has_secret_marker so callers can
# actually set an output token budget. Anything credential-shaped
# (access_token, api_token, bearer_token, ...) still matches the marker and
# is rejected.
_TOKEN_BUDGET_ALLOWLIST: frozenset[str] = frozenset(
    {"max_tokens", "max_completion_tokens", "max_output_tokens"}
)


def _has_secret_marker(key: str) -> bool:
    """Return True if the key name contains a secret-bearing marker.

    Known non-secret parameter names that happen to contain a marker
    substring (e.g. ``max_tokens`` contains ``token``) are exempted via
    ``_TOKEN_BUDGET_ALLOWLIST`` so the output-budget knobs remain usable.
    """
    if key.casefold() in _TOKEN_BUDGET_ALLOWLIST:
        return False
    return any(p.search(key) for p in _SECRET_KEY_MARKERS)


class DecisionCacheMiss(RuntimeError):
    """Raised when a replay-locked hosted policy encounters an unseen context."""


class OpenAICompatPolicyError(RuntimeError):
    """Raised when the OpenAI-compatible policy fails after exhausting retries."""


@dataclass(frozen=True, slots=True)
class _Message:
    content: str | None
    refusal: str | None


@dataclass(frozen=True, slots=True)
class _Choice:
    message: _Message
    # OpenAI ``finish_reason``: "stop", "length", "tool_calls", "content_filter"…
    # "length" means the model hit the output token budget before finishing —
    # a deterministic, non-transient outcome the adapter must not retry on.
    finish_reason: str | None = None


@dataclass(frozen=True, slots=True)
class _OpenAIResponse:
    """Normalized OpenAI-compatible chat-completions response.

    Both the default httpx path and an injected test callable return this shape,
    so :meth:`OpenAICompatPolicy._record_decision` has a single parsing path.
    """

    choices: tuple[_Choice, ...]
    id: str | None
    model: str | None
    system_fingerprint: str | None
    usage: dict[str, int] | None


def _validate_api_base(api_base: str | None) -> None:
    """Validate API base URL.  Never interpolate the raw URL into messages."""
    if api_base is None:
        return
    parsed = urlparse(api_base)
    # Require http/https scheme with a hostname
    if parsed.scheme not in ("http", "https"):
        raise ValueError("api_base must use http or https scheme")
    if not parsed.hostname:
        raise ValueError("api_base must contain a hostname")
    if parsed.username or parsed.password:
        raise ValueError("api_base must not contain userinfo (embedded credentials)")
    if parsed.query:
        raise ValueError("api_base must not contain query parameters")
    if parsed.fragment:
        raise ValueError("api_base must not contain a fragment")


def _api_base_identity(api_base: str | None) -> str | None:
    """Compute a canonical identity for the API base URL."""
    if api_base is None:
        return None
    parsed = urlparse(api_base)
    scheme = parsed.scheme.lower()
    hostname = (parsed.hostname or "").lower()
    port = parsed.port
    if (scheme == "https" and port == 443) or (scheme == "http" and port == 80):
        port = None
    path = parsed.path.rstrip("/") or ""
    netloc = hostname
    if port is not None:
        netloc = f"{hostname}:{port}"
    normalized = f"{scheme}://{netloc}{path}"
    return hashlib.sha256(normalized.encode()).hexdigest()


def _normalize_usage(raw_usage: Any) -> dict[str, int] | None:
    """Extract only integer token counts from the usage object."""
    if raw_usage is None:
        return None
    if hasattr(raw_usage, "model_dump"):
        data = raw_usage.model_dump(mode="json")
    elif isinstance(raw_usage, dict):
        data = raw_usage
    else:
        return None
    result: dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        val = data.get(key)
        if val is not None:
            result[key] = int(val)
    return result if result else None


def _endpoint_url(api_base: str | None) -> str:
    """Build the chat-completions URL from a (possibly defaulted) api_base."""
    base = api_base if api_base is not None else _DEFAULT_API_BASE
    return f"{base.rstrip('/')}/chat/completions"


def _parse_openai_response(data: Any) -> _OpenAIResponse:
    """Parse a raw OpenAI-compatible JSON payload into a normalized response."""
    if not isinstance(data, dict):
        raise ValueError("response body must be a JSON object")
    raw_choices = data.get("choices")
    if not isinstance(raw_choices, list):
        raise ValueError("response body must contain a 'choices' array")
    choices: list[_Choice] = []
    for raw in raw_choices:
        if not isinstance(raw, dict):
            raise ValueError("each choice must be a JSON object")
        raw_message = raw.get("message")
        if not isinstance(raw_message, dict):
            raise ValueError("each choice must contain a 'message' object")
        choices.append(
            _Choice(
                message=_Message(
                    content=raw_message.get("content"),
                    refusal=raw_message.get("refusal"),
                ),
                finish_reason=raw.get("finish_reason"),
            )
        )
    raw_usage = data.get("usage")
    usage = _normalize_usage(raw_usage) if raw_usage is not None else None
    return _OpenAIResponse(
        choices=tuple(choices),
        id=data.get("id"),
        model=data.get("model"),
        system_fingerprint=data.get("system_fingerprint"),
        usage=usage,
    )


async def _default_httpx_completion(
    *,
    url: str,
    headers: Mapping[str, str],
    body: Mapping[str, Any],
    timeout: float,
) -> _OpenAIResponse:
    """Default transport: a per-call httpx AsyncClient POST to the endpoint."""
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(url, json=dict(body), headers=dict(headers))
    response.raise_for_status()
    return _parse_openai_response(response.json())


class OpenAICompatPolicy:
    """Policy adapter for a single OpenAI-compatible chat-completions endpoint."""

    name = "openai_compat"

    def __init__(
        self,
        *,
        model: str,
        completion: Any | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
        timeout_seconds: float = 120.0,
        max_attempts: int = 2,
        generation_parameters: Mapping[str, Any] | None = None,
        decision_cache: MutableMapping[str, PolicyDecision] | None = None,
        cache_mode: Literal["record", "locked"] = "record",
    ) -> None:
        # Validation — these are constructor errors, immediate ValueError.
        if not model:
            raise ValueError("model must be non-empty")
        if model != model.strip():
            raise ValueError("model must not contain leading or trailing whitespace")
        if "/" in model:
            raise ValueError("model must be a bare name without a provider prefix ('/')")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be a positive finite number")
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least one")
        if cache_mode not in ("record", "locked"):
            raise ValueError("cache_mode must be 'record' or 'locked'")
        _validate_api_base(api_base)

        if generation_parameters:
            bad_keys = set(generation_parameters.keys()) & _RESERVED_GENERATION_KEYS
            if bad_keys:
                raise ValueError(
                    f"generation_parameters contains reserved keys: {sorted(bad_keys)}"
                )
            secret_keys = [k for k in generation_parameters if _has_secret_marker(k)]
            if secret_keys:
                raise ValueError(
                    f"generation_parameters contains secret-bearing keys: {sorted(secret_keys)}"
                )

        # Store values
        self.model = model
        self._completion = completion if completion is not None else _default_httpx_completion
        self._api_base = api_base
        self._api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts
        # Sampling temperature for the request body. Default 0.0 (deterministic
        # measurement intent); overridable via the explicit generation_parameters
        # mapping OR DISTORTION_TEMPERATURE. If generation_parameters sets
        # "temperature", it wins (it is merged after this in the body).
        env_temp = os.environ.get("DISTORTION_TEMPERATURE", "").strip()
        explicit_temp = None
        if generation_parameters and "temperature" in generation_parameters:
            explicit_temp = float(generation_parameters["temperature"])
        self._temperature = (
            explicit_temp if explicit_temp is not None else (float(env_temp) if env_temp else 0.0)
        )
        if not math.isfinite(self._temperature) or self._temperature < 0:
            raise ValueError("temperature must be a non-negative finite number")
        self.generation_parameters = dict(generation_parameters) if generation_parameters else {}
        self.decision_cache = decision_cache if decision_cache is not None else {}
        self.cache_mode = cache_mode
        self._cache_locks: dict[str, asyncio.Lock] = {}

        # Compute fingerprint
        self.fingerprint = self._compute_fingerprint()

    def _compute_fingerprint(self) -> str:
        """Compute the canonical fingerprint for this policy configuration."""
        payload = {
            "policy_adapter": "openai_compat",
            "adapter_contract_version": "openai-compat-v1",
            "requested_model": self.model,
            "api_base_identity": _api_base_identity(self._api_base),
            "prompt_version": "v2",
            "prompt_sha256": hashlib.sha256(INSTRUCTIONS.encode()).hexdigest(),
            "schema_version": "v1",
            "schema_sha256": hashlib.sha256(
                canonical_json(AgentOutput.model_json_schema()).encode()
            ).hexdigest(),
            "structured_output_mode": "json_object",
            "canonical_generation_parameters": canonical_json(self.generation_parameters),
            "temperature": self._temperature,
            "timeout_seconds": self.timeout_seconds,
            "engine_max_attempts": self.max_attempts,
        }
        canonical = canonical_json(payload)
        digest = hashlib.sha256(canonical.encode()).hexdigest()
        return f"openai_compat:{digest}"

    async def decide(self, context: AgentContext) -> PolicyDecision:
        """Make a policy decision for the given context."""
        context_data = context.model_dump(mode="json")
        cache_key = hashlib.sha256(
            canonical_json(
                {"context": context_data, "policy_fingerprint": self.fingerprint}
            ).encode()
        ).hexdigest()

        # Check cache before acquiring lock
        cached = self.decision_cache.get(cache_key)
        if cached is not None:
            return cached

        lock = self._cache_locks.setdefault(cache_key, asyncio.Lock())
        try:
            async with lock:
                cached = self.decision_cache.get(cache_key)
                if cached is not None:
                    return cached

                if self.cache_mode == "locked":
                    raise DecisionCacheMiss(
                        f"decision {cache_key[:12]} is not captured; "
                        "run record mode before locked replay"
                    )

                return await self._record_decision(context, context_data, cache_key)
        finally:
            # The per-key lock only serializes the cache-fill for this key.
            # Once we exit this block the decision is either recorded or the
            # key is a permanent miss (locked mode), so future callers will
            # hit the cache check above and never need this lock again.
            # Remove it so a long-running policy does not leak one Lock per
            # unique context.
            self._cache_locks.pop(cache_key, None)

    async def _record_decision(
        self, context: AgentContext, context_data: dict[str, Any], cache_key: str
    ) -> PolicyDecision:
        """Record a new decision by calling the OpenAI-compatible endpoint."""
        payload = json.dumps(
            {
                "role_context": context_data,
                "permitted_action_kinds": [
                    "work",
                    "test",
                    "remediate",
                    "release",
                    "delay_release",
                    "reduce_scope",
                    "investigate",
                    "add_qa_capacity",
                ],
            },
            sort_keys=True,
        )

        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": INSTRUCTIONS},
                {"role": "user", "content": payload},
            ],
            "response_format": {"type": "json_object"},
        }
        # Pin sampling temperature for deterministic measurement. A live run
        # showed the agent's reported health for byte-identical input swinging
        # 0.0..1.0 across samples (the engine measures "reported health under
        # a treatment" from a single sample; unchecked sampling makes the
        # measurement noise comparable to the treatment effect, undermining the
        # causal contrast). The opencode/zen free endpoint ignores this for
        # full determinism, but requesting temperature:0 reduces variance and
        # is the engine's intent. Explicit generation_parameters (which
        # .update() below) or DISTORTION_TEMPERATURE override it.
        body["temperature"] = self._temperature
        body.update(self.generation_parameters)

        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self._api_key is not None:
            headers["Authorization"] = f"Bearer {self._api_key}"

        request_url = _endpoint_url(self._api_base)
        request_timeout = self.timeout_seconds

        error: Exception | None = None
        for attempt in range(self.max_attempts):
            if attempt > 0:
                # Back off before the retry (exponential + jitter); see
                # _retry_backoff_delay. Transient 5xx / transport errors that
                # cluster (e.g. a degraded free endpoint) would otherwise
                # exhaust max_attempts in a burst and abort the whole run.
                await _async_sleep(await _retry_backoff_delay(attempt))
            try:
                response = await self._completion(
                    url=request_url,
                    headers=headers,
                    body=body,
                    timeout=request_timeout,
                )

                # --- Response validation (retryable output errors) ---
                choices = response.choices
                if not isinstance(choices, list | tuple) or len(choices) != 1:
                    raise ValueError("response must contain exactly one choice")

                choice = choices[0]
                message = choice.message
                if message is None:
                    raise ValueError("response choice has no message")

                # Detect refusal field on the message object. A refusal is a
                # deliberate policy decision by the model, not a transient
                # error — retrying the identical prompt will refuse again, so
                # fail closed immediately as the typed error (no retry). The
                # refusal text itself is provider content and must NOT appear
                # in the exception (it must not leak into logs).
                refusal = getattr(message, "refusal", None)
                if refusal:
                    raise OpenAICompatPolicyError(
                        f"openai_compat policy failed: model refused the request "
                        f"for model {self.model}"
                    )

                content = getattr(message, "content", None)
                if content is None:
                    finish_reason = getattr(choice, "finish_reason", None)
                    if finish_reason == "length":
                        # The model exhausted its output token budget before
                        # producing any content (e.g. a reasoning model that
                        # spent max_tokens on reasoning). This is a clean 200
                        # response that is deterministic for identical inputs —
                        # retrying is futile. Fail closed immediately.
                        raise OpenAICompatPolicyError(
                            f"openai_compat policy failed: model exhausted output "
                            f"budget with no content (finish_reason=length) "
                            f"for model {self.model}"
                        )
                    # Genuinely-anomalous missing content with no length
                    # signal: treat as transient and retry.
                    raise ValueError("response content is None")
                if not isinstance(content, str):
                    raise ValueError("response content must be a string")

                # Strict Pydantic validation — never leak raw content
                try:
                    output = AgentOutput.model_validate_json(content)
                except Exception:
                    if os.environ.get("DISTORTION_DEBUG_SCHEMA", "") == "1":
                        # Opt-in diagnostic seam: the v2 prompt-fix made live
                        # schema failures undiagnosable otherwise (the `from
                        # None` below discards raw content per ADR #8). Gated
                        # by env so it never leaks content in production runs.
                        try:
                            import time

                            dbg = Path(f"tmp/schema-debug-{int(time.time() * 1000)}.json")
                            dbg.parent.mkdir(parents=True, exist_ok=True)
                            dbg.write_text(
                                json.dumps(
                                    {
                                        "model": self.model,
                                        "content": content,
                                        "context_hash": cache_key,
                                    },
                                    indent=2,
                                ),
                                encoding="utf-8",
                            )
                        except Exception:
                            pass
                    raise ValueError("response failed schema validation") from None

                decision = self._to_decision(
                    context,
                    output,
                    cache_key=cache_key,
                    response_id=response.id,
                    response_model=response.model,
                    system_fingerprint=response.system_fingerprint,
                    usage=response.usage,
                )

                self.decision_cache[cache_key] = decision
                return decision

            except httpx.HTTPStatusError as caught:
                # Transient HTTP statuses retry; non-retryable client errors
                # (400/401/403/404 …) fail closed immediately as the typed
                # OpenAICompatPolicyError, chaining the original so callers
                # have one stable exception type for every terminal failure
                # (matching the LiteLLM/OpenAI convention of typed, never
                # raw-transport, exceptions).
                status = caught.response.status_code
                if status in _RETRYABLE_HTTP_STATUS:
                    error = caught
                    continue
                raise OpenAICompatPolicyError(
                    f"openai_compat policy failed: HTTP {status} for model {self.model}"
                ) from caught
            except httpx.TransportError as caught:
                # Connection / timeout / protocol failures are transient.
                error = caught
                continue
            except ValueError as caught:
                # Response-shape / schema errors are retryable.
                error = caught
                continue
            except Exception:
                # Unexpected programming errors propagate immediately.
                raise

        # Exhausted retries — always wrap in OpenAICompatPolicyError
        raise OpenAICompatPolicyError(
            f"openai_compat policy failed after {self.max_attempts} attempts for model {self.model}"
        ) from error

    def _to_decision(
        self,
        context: AgentContext,
        output: AgentOutput,
        *,
        cache_key: str,
        response_id: str | None,
        response_model: str | None,
        system_fingerprint: str | None,
        usage: dict[str, int] | None,
    ) -> PolicyDecision:
        """Convert AgentOutput to PolicyDecision."""
        permitted_items = set(context.scope)
        actions = tuple(
            action
            for action in output.actions
            if action.item_id is None or action.item_id in permitted_items
        )

        report = Report(
            agent_id=context.agent.id,
            department=context.agent.department,
            depth=context.depth,
            tick=context.tick,
            scope=context.scope,
            health=output.health,
            confidence=output.confidence,
            escalate=output.escalate,
            explanation=output.explanation,
        )

        memory = AgentMemory(
            turns=context.memory.turns + 1,
            concerns=tuple(dict.fromkeys((*context.memory.concerns, *output.concerns)))[-12:],
            trust=context.memory.trust,
        )

        return PolicyDecision(
            report=report,
            actions=actions,
            memory=memory,
            provider_metadata={
                "cache_key": cache_key,
                "reproducibility": "captured-provider-decision",
                "policy": "openai_compat",
                "policy_fingerprint": self.fingerprint,
                "requested_model": self.model,
                "response_model": response_model,
                "response_id": response_id,
                "system_fingerprint": system_fingerprint,
                "api_base_id": _api_base_identity(self._api_base),
                "structured_output_mode": "json_object",
                "prompt_version": "v2",
                "schema_version": "v1",
                "usage": usage,
            },
        )
