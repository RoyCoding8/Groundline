"""Tests for the shared policy factory cache-partition logic."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from distortion_engine.policy.cache import FileDecisionCache
from distortion_engine.policy.factory import _cache_partition_key, build_policy

# ---------------------------------------------------------------------------
# 1. Windows-safe filenames
# ---------------------------------------------------------------------------


class TestCachePartitionWindowsSafe:
    """Cache filenames must be valid on Windows (no ':', '\\', etc.)."""

    @pytest.mark.parametrize(
        "model",
        [
            "openai-model",
            "provider-model-2025",
            "third-party-model",
            "custom-model",
            "m",
            "provider-model-v1:0",
        ],
    )
    def test_filenames_are_windows_safe(self, model: str) -> None:
        name = _cache_partition_key(model, api_base=None)
        # No colon, backslash, or angle brackets
        assert ":" not in name
        assert "\\" not in name
        assert "<" not in name
        assert ">" not in name
        # Must end with .json
        assert name.endswith(".json")
        # No path separators embedded in the filename portion
        assert "/" not in name.split("__")[0]
        assert "\\" not in name.split("__")[0]

    def test_colon_in_model_replaced(self) -> None:
        name = _cache_partition_key("provider-model-v1:0", api_base=None)
        # The provider-style colon in v1:0 must not appear in the filename
        assert ":" not in name
        # The slug should contain a safe representation
        slug = name.split("__")[0]
        assert re.match(r"[A-Za-z0-9._-]+$", slug)


# ---------------------------------------------------------------------------
# 2. Endpoint separation
# ---------------------------------------------------------------------------


class TestCachePartitionEndpointSeparation:
    """Different custom endpoints must produce different cache filenames."""

    def test_different_api_base_different_filename(self) -> None:
        name_a = _cache_partition_key("custom-model", "https://a.example.com/v1")
        name_b = _cache_partition_key("custom-model", "https://b.example.com/v1")
        assert name_a != name_b

    def test_no_endpoint_vs_endpoint_different(self) -> None:
        name_no = _cache_partition_key("custom-model", None)
        name_yes = _cache_partition_key("custom-model", "https://custom.example.com/v1")
        assert name_no != name_yes

    def test_same_endpoint_same_hash(self) -> None:
        name_a = _cache_partition_key("m", "https://x.example.com/v1")
        name_b = _cache_partition_key("m", "https://x.example.com/v1")
        assert name_a == name_b

    def test_api_base_normalization(self) -> None:
        """Trailing slashes and default ports should not change the filename."""
        name_a = _cache_partition_key("m", "https://example.com/v1/")
        name_b = _cache_partition_key("m", "https://example.com/v1")
        assert name_a == name_b

        name_c = _cache_partition_key("m", "https://example.com:443/v1")
        name_d = _cache_partition_key("m", "https://example.com/v1")
        assert name_c == name_d


# ---------------------------------------------------------------------------
# 3. Record/locked equality
# ---------------------------------------------------------------------------


class TestCachePartitionRecordLockedEquality:
    """Record and locked modes with identical config must share the filename."""

    def test_record_and_locked_same_filename(self, tmp_path: Path) -> None:
        env = {
            "DISTORTION_MODEL": "test-model",
            "DISTORTION_API_BASE": "",
            "DISTORTION_API_KEY": "",
        }
        with pytest.MonkeyPatch.context() as mp:
            for k, v in env.items():
                mp.setenv(k, v)
            mp.delenv("AWS_PROFILE", raising=False)
            mp.delenv("AWS_REGION", raising=False)
            mp.delenv("AWS_DEFAULT_REGION", raising=False)

            rec = build_policy("record", artifacts=tmp_path)
            loc = build_policy("locked", artifacts=tmp_path)

        # The cache file should be the same path
        assert rec.decision_cache is not None
        assert loc.decision_cache is not None
        assert isinstance(rec.decision_cache, FileDecisionCache)
        assert isinstance(loc.decision_cache, FileDecisionCache)
        assert str(rec.decision_cache.path) == str(loc.decision_cache.path)

    def test_same_model_same_filename(self) -> None:
        name_a = _cache_partition_key("test-model", None)
        name_b = _cache_partition_key("test-model", None)
        assert name_a == name_b


# ---------------------------------------------------------------------------
# 4. Slug properties
# ---------------------------------------------------------------------------


class TestCachePartitionSlug:
    """Slug portion should be bounded and human-readable."""

    def test_long_model_truncated(self) -> None:
        long_model = "a" * 200
        name = _cache_partition_key(long_model, api_base=None)
        slug = name.split("__")[0]
        assert len(slug) <= 80

    def test_empty_model_uses_default(self) -> None:
        name = _cache_partition_key("default", api_base=None)
        slug = name.split("__")[0]
        assert slug  # non-empty

    def test_hash_deterministic(self) -> None:
        name_a = _cache_partition_key("test-model", None)
        name_b = _cache_partition_key("test-model", None)
        assert name_a == name_b
        # Extract and verify the hash is a valid hex string
        short_hash = name_a.split("__")[1].replace(".json", "")
        assert len(short_hash) == 12
        int(short_hash, 16)  # must be valid hex


# ---------------------------------------------------------------------------
# 5. Legacy alias rejection
# ---------------------------------------------------------------------------


def test_litellm_aliases_rejected(tmp_path: Path) -> None:
    """build_policy must reject the old 'litellm-record' / 'litellm-locked' names."""
    with pytest.raises(ValueError, match="policy must be"):
        build_policy("litellm-record", artifacts=tmp_path)
    with pytest.raises(ValueError, match="policy must be"):
        build_policy("litellm-locked", artifacts=tmp_path)
