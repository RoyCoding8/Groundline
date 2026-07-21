from pathlib import Path
from typing import Any

import yaml
from dotenv import find_dotenv, load_dotenv

from groundline.experiments.runner import ExperimentRequest
from groundline.simulation.runner import RunRequest


def load_env() -> None:
    """Load ``.env`` from the current working directory if present.

    Searches from the current working directory upward for a ``.env`` file
    (so it is found relative to where the user invokes ``distortion``, not
    relative to the installed package). Uses ``override=False`` so values
    already set in the real environment (exported shell variables) win over
    file values — the file only fills gaps. A missing ``.env`` file is not
    an error. Idempotent: calling more than once is safe and never clobbers
    an existing environment variable.
    """
    load_dotenv(find_dotenv(usecwd=True), override=False)


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("configuration root must be a mapping")
    return data


def load_run_request(path: Path, *, seed: int | None = None) -> RunRequest:
    data = load_yaml(path)
    if "seeds" in data:
        experiment = ExperimentRequest.model_validate(data)
        if seed is None:
            seed = experiment.seeds[0]
        first_treatment = next(iter(experiment.treatments.values()))
        return RunRequest(
            scenario=experiment.scenario,
            organization=experiment.organization,
            treatment=first_treatment,
            seed=seed,
        )
    if seed is not None:
        data["seed"] = seed
    return RunRequest.model_validate(data)


def load_experiment_request(path: Path) -> ExperimentRequest:
    return ExperimentRequest.model_validate(load_yaml(path))
