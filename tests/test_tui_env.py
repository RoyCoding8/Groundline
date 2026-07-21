"""Regression tests for the TUI launcher's environment-status panel.

The env panel is user-facing (rendered by ``tui.py`` via ``run.sh``/``run.bat``)
and must advertise only the OpenAI-compatible adapter's configuration surface.
The former multi-provider adapter read ``AWS_PROFILE``/``AWS_REGION`` for
Bedrock; that contract was dropped (see the openai-compat ADR), so the panel
must no longer list or mask those keys.

``tui.py`` is a repo-root launcher script, not an installed package, so it is
loaded by path here rather than imported as a module.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TUI_PATH = _REPO_ROOT / "tui.py"


def _load_tui() -> object:
    """Load the root ``tui.py`` as a fresh module (by path)."""
    spec = importlib.util.spec_from_file_location("distortion_tui_under_test", _TUI_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_env_status_lists_only_openai_compat_keys(tmp_path: Path) -> None:
    """The panel surfaces exactly the OpenAI-compat key set — no Bedrock/AWS."""
    tui = _load_tui()
    env = tmp_path / ".env"
    env.write_text(
        "DISTORTION_MODEL=gpt-4o\n"
        "DISTORTION_API_BASE=https://api.openai.com/v1\n"
        "DISTORTION_API_KEY=sk-test\n"
        "DISTORTION_TIMEOUT_SECONDS=120\n"
        "DISTORTION_MAX_ATTEMPTS=2\n"
        "AWS_PROFILE=legacy\n"
        "AWS_REGION=us-east-1\n",
        encoding="utf-8",
    )
    tui.ENV_FILE = env  # panel reads from this module-level path

    status = tui._env_status()

    assert set(status.keys()) == {
        "DISTORTION_MODEL",
        "DISTORTION_API_BASE",
        "DISTORTION_API_KEY",
        "DISTORTION_TIMEOUT_SECONDS",
        "DISTORTION_MAX_ATTEMPTS",
    }
    # Bedrock credentials must not leak into the panel.
    assert "AWS_PROFILE" not in status
    assert "AWS_REGION" not in status


def test_env_status_masks_only_the_real_secret(tmp_path: Path) -> None:
    """Only DISTORTION_API_KEY is masked; AWS keys are gone entirely."""
    tui = _load_tui()
    env = tmp_path / ".env"
    env.write_text("DISTORTION_API_KEY=sk-secret\nAWS_PROFILE=never-shown\n", encoding="utf-8")
    tui.ENV_FILE = env

    status = tui._env_status()

    assert status["DISTORTION_API_KEY"] == "[green]set[/]"
    # A masked-but-present AWS key would be a regression; it must be absent.
    assert "AWS_PROFILE" not in status
