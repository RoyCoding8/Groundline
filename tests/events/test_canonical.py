"""Canonical JSON/hash helpers must be defined in exactly one place.

Regression for the duplication audit finding: ``canonical_json`` and
``canonical_hash`` were byte-for-byte duplicated in ``events/store.py`` and
``events/artifacts.py``. This test pins a single source of truth so the
duplicate cannot silently reappear (which would let the two copies drift).
"""

from __future__ import annotations

from distortion_engine.events import artifacts, store


def test_canonical_json_is_single_source_of_truth() -> None:
    assert artifacts.canonical_json is store.canonical_json, (
        "canonical_json is duplicated; artifacts.canonical_json must re-export "
        "store.canonical_json, not redefine it."
    )


def test_canonical_hash_is_single_source_of_truth() -> None:
    assert artifacts.canonical_hash is store.canonical_hash, (
        "canonical_hash is duplicated; artifacts.canonical_hash must re-export "
        "store.canonical_hash, not redefine it."
    )


def test_canonical_hash_is_stable_and_correct() -> None:
    # The shared helper must still behave correctly after the dedup.
    assert store.canonical_hash({"b": 2, "a": 1}) == store.canonical_hash({"a": 1, "b": 2})
    assert len(store.canonical_hash({"a": 1})) == 64
