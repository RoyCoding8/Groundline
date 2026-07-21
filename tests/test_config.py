"""Tests for the ``.env`` loading helper.

``load_env()`` must (1) fill unset vars from a ``.env`` file, (2) never
overwrite a value already set in the real environment, and (3) be a no-op
when no ``.env`` file is present. All three properties are exercised against
a temporary working directory so the real environment is not polluted.
"""

from __future__ import annotations

import os
from pathlib import Path

from groundline.config import load_env


def test_load_env_reads_dotenv_when_unset(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GROUNDLINE_MODEL", raising=False)
    (tmp_path / ".env").write_text("GROUNDLINE_MODEL=from-file\n", encoding="utf-8")

    load_env()

    assert os.environ["GROUNDLINE_MODEL"] == "from-file"


def test_load_env_real_env_wins_over_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GROUNDLINE_MODEL", "from-shell")
    (tmp_path / ".env").write_text("GROUNDLINE_MODEL=from-file\n", encoding="utf-8")

    load_env()

    assert os.environ["GROUNDLINE_MODEL"] == "from-shell"


def test_load_env_missing_dotenv_is_not_an_error(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GROUNDLINE_MODEL", raising=False)
    assert not (tmp_path / ".env").exists()

    load_env()  # must not raise

    assert "GROUNDLINE_MODEL" not in os.environ


def test_load_env_is_idempotent(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GROUNDLINE_MODEL", "from-shell")
    (tmp_path / ".env").write_text("GROUNDLINE_MODEL=from-file\n", encoding="utf-8")

    load_env()
    load_env()  # second call must not clobber the shell value

    assert os.environ["GROUNDLINE_MODEL"] == "from-shell"
