"""MMAudio CounterFlow backend wrapper.

Required environment:
    conda activate MMAudio

Purpose:
    Build and run CounterFlow VGGSound-Sparse commands through external/MMAudio.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence


class MMAudioCounterFlowWrapper:
    """Thin subprocess wrapper around the CounterFlow-modified MMAudio script."""

    def __init__(self, project_root: Path | None = None, repo_dir: Path | None = None) -> None:
        self.project_root = (project_root or Path(__file__).resolve().parents[2]).resolve()
        self.repo_dir = (repo_dir or self.project_root / "external" / "MMAudio").resolve()
        self.script_path = self.repo_dir / "eval_vggsound_sparse.py"
        self.script_src = self.project_root / "counterflow" / "mmaudio" / "eval_vggsound_sparse.py"

    def install_backend_script(self, *, overwrite: bool = False) -> None:
        if not self.script_src.exists():
            raise FileNotFoundError(f"CounterFlow MMAudio script source not found: {self.script_src}")
        if self.script_path.exists() and not overwrite:
            return
        shutil.copy2(self.script_src, self.script_path)

    def validate(self) -> None:
        if not self.repo_dir.exists():
            raise FileNotFoundError(
                f"MMAudio repository not found: {self.repo_dir}. "
                "Run scripts/setup_external_repos.sh first."
            )
        if not self.script_path.exists():
            self.install_backend_script()

    def build_command(
        self,
        *,
        output_dir: Path,
        exp_name: str,
        subset: str = "clean",
        clean_csv_path: Path | None = None,
        extra_args: Sequence[str] | None = None,
    ) -> list[str]:
        self.validate()
        command = [
            sys.executable,
            str(self.script_path),
            "--output_dir",
            str(output_dir),
            "--exp_name",
            exp_name,
            "--subset",
            subset,
        ]
        if clean_csv_path is not None:
            command.extend(["--clean_csv_path", str(clean_csv_path)])
        command.extend(extra_args or [])
        return command

    def run(self, command: Sequence[str], *, dry_run: bool = False) -> subprocess.CompletedProcess[str] | None:
        if dry_run:
            print(" ".join(command))
            return None
        return subprocess.run(command, cwd=self.repo_dir, check=True, text=True)
