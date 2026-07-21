"""Shared policy construction for CLI and API callers.

Resolves OpenAI-compatible model, endpoint, credentials, and numeric
configuration from explicit arguments and environment variables with the
documented precedence.  CLI and API jobs call exactly one helper each.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

from groundline.policy.cache import FileDecisionCache
from groundline.policy.fixture import FixturePolicy
from groundline.policy.models import AgentPolicy
from groundline.policy.openai_compat_policy import (
    OpenAICompatPolicy,
    _api_base_identity,
)

# Maximum length for the filesystem-safe model slug in cache filenames.
_SLUG_MAX_LEN = 80


def _cache_partition_key(model: str, api_base: str | None) -> str:
    """Build a deterministic, filesystem-safe cache filename.

    The name encodes:
    - A bounded, human-readable slug derived from the requested model.
    - A truncated SHA-256 of the exact requested model string and the
      normalized API-base identity (or ``null`` when no custom endpoint).

    Record and locked modes with identical effective configuration resolve
    to the same file.  Changing the custom endpoint changes the hash.
    No raw API base or secret value appears in the name.
    """
    # Bounded slug: keep alphanumeric, hyphens, underscores, dots; collapse runs.
    raw_slug = re.sub(r"[^A-Za-z0-9._-]+", "-", model)
    slug = raw_slug[:_SLUG_MAX_LEN].rstrip("-") or "model"

    # Content-addressed component: model + API-base identity.
    api_base_id = _api_base_identity(api_base)
    digest_input = f"{model}\0{api_base_id or 'null'}"
    short_hash = hashlib.sha256(digest_input.encode()).hexdigest()[:12]

    return f"{slug}__{short_hash}.json"


def resolve_openai_compat_policy(
    *,
    model: str | None = None,
    cache_path: Path | None = None,
    cache_mode: str = "record",
) -> OpenAICompatPolicy:
    """Build an :class:`OpenAICompatPolicy` from resolved configuration.

    Precedence for model:

    1. Explicit *model* argument when non-empty.
    2. ``GROUNDLINE_MODEL`` environment variable.
    3. :class:`ValueError` if neither is available.

    Other settings are read from the environment:

    - ``GROUNDLINE_API_BASE`` / ``GROUNDLINE_API_KEY``
    - ``GROUNDLINE_TIMEOUT_SECONDS`` (default 120)
    - ``GROUNDLINE_MAX_ATTEMPTS`` (default 2)
    """
    resolved_model = (model or "").strip() or os.environ.get("GROUNDLINE_MODEL", "").strip()
    if not resolved_model:
        raise ValueError("model is required: supply --model / request body or set GROUNDLINE_MODEL")

    api_base = os.environ.get("GROUNDLINE_API_BASE", "").strip() or None
    api_key = os.environ.get("GROUNDLINE_API_KEY", "").strip() or None

    timeout_str = os.environ.get("GROUNDLINE_TIMEOUT_SECONDS", "").strip()
    timeout = float(timeout_str) if timeout_str else 120.0

    attempts_str = os.environ.get("GROUNDLINE_MAX_ATTEMPTS", "").strip()
    max_attempts = int(attempts_str) if attempts_str else 2

    decision_cache = None
    if cache_path is not None:
        decision_cache = FileDecisionCache(cache_path)

    return OpenAICompatPolicy(
        model=resolved_model,
        api_base=api_base,
        api_key=api_key,
        timeout_seconds=timeout,
        max_attempts=max_attempts,
        decision_cache=decision_cache,
        cache_mode=cache_mode,  # type: ignore[arg-type]
    )


def build_policy(
    name: str,
    *,
    model: str | None = None,
    artifacts: Path | None = None,
) -> AgentPolicy:
    """Top-level dispatcher: fixture, record, or locked."""
    if name == "fixture":
        return FixturePolicy()
    if name in {"record", "locked"}:
        cache_path = None
        if artifacts is not None:
            resolved = (model or "").strip() or os.environ.get("GROUNDLINE_MODEL", "").strip()
            api_base = os.environ.get("GROUNDLINE_API_BASE", "").strip() or None
            partition = _cache_partition_key(resolved or "default", api_base)
            cache_path = artifacts / "policy-cache" / partition
        return resolve_openai_compat_policy(
            model=model,
            cache_path=cache_path,
            cache_mode="locked" if name == "locked" else "record",
        )
    raise ValueError("policy must be 'fixture', 'record', or 'locked'")
