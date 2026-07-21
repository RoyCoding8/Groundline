from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        if self.target_name != "wheel" or build_data.get("editable_mode"):
            return
        root = Path(self.root)
        frontend = root / "frontend"
        npm = shutil.which("npm") or shutil.which("npm.cmd")
        if npm is None:
            raise RuntimeError("npm is required to build the packaged frontend")
        if not (frontend / "node_modules").is_dir():
            subprocess.run([npm, "ci"], cwd=frontend, check=True)
        subprocess.run([npm, "run", "build"], cwd=frontend, check=True)
        force_include = build_data.setdefault("force_include", {})
        force_include[str(frontend / "dist")] = "groundline/frontend"
