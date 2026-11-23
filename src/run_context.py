"""
Run context utilities for reproducible experiment outputs.

A RunContext centralizes paths such as outputs/runs/<run_id>/... and prevents
scripts from hardcoding output filenames in multiple places.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.io_utils import ensure_dir, write_json


@dataclass
class RunContext:
    run_id: str
    output_root: Path
    config: dict[str, Any]

    @property
    def run_dir(self) -> Path:
        return self.output_root / self.run_id

    @property
    def metrics_dir(self) -> Path:
        return self.run_dir / "metrics"

    @property
    def annotation_dir(self) -> Path:
        return self.run_dir / "annotation_exports"

    def path(self, relative: str) -> Path:
        return self.run_dir / relative

    def prepare(self, overwrite: bool = False) -> None:
        if overwrite and self.run_dir.exists():
            shutil.rmtree(self.run_dir)
        ensure_dir(self.run_dir)
        ensure_dir(self.metrics_dir)
        ensure_dir(self.annotation_dir)
        write_json(self.config, self.run_dir / "config.json")


def context_from_config(config: dict[str, Any], default_output_root: str | Path = "outputs/runs") -> RunContext:
    run_config = config.get("run", {})
    run_id = run_config.get("run_id")
    if not run_id:
        raise ValueError("Config must define run.run_id")

    output_root = Path(run_config.get("output_root", default_output_root))
    return RunContext(run_id=run_id, output_root=output_root, config=config)