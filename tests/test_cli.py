from importlib.metadata import version
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from distortion_engine import __version__
from distortion_engine.cli import _policy, app


def test_cli_exposes_version_and_required_commands() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert __version__ in result.stdout
    for command in ("validate", "run", "replay", "experiment", "analyze", "serve"):
        assert command in result.stdout
    assert version("distortion-engine") == __version__


def test_cli_invalid_policy_lists_every_accepted_alias(tmp_path: Path) -> None:
    with pytest.raises(typer.BadParameter) as error:
        _policy("unsupported", "model", tmp_path)

    assert str(error.value) == ("policy must be 'fixture', 'record', or 'locked'")
