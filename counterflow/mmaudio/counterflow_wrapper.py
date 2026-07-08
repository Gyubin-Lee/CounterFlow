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


BACKEND_PATH_OPTIONS = {
    "--clean_csv_path",
    "--csv_path",
    "--precomputed_features_dir",
    "--target_prompt_bank",
    "--video_root",
}


class MMAudioCounterFlowWrapper:
    """Thin subprocess wrapper around the CounterFlow-modified MMAudio script."""

    def __init__(self, project_root: Path | None = None, repo_dir: Path | None = None) -> None:
        self.project_root = (project_root or Path(__file__).resolve().parents[2]).resolve()
        self.repo_dir = (repo_dir or self.project_root / "external" / "MMAudio").resolve()
        self.script_src = self.project_root / "counterflow" / "mmaudio" / "eval_vggsound_sparse.py"
        self.script_path = self.script_src
        self.backend_script_path = self.repo_dir / "eval_vggsound_sparse.py"

    def install_backend_script(self, *, overwrite: bool = False) -> None:
        if not self.script_src.exists():
            raise FileNotFoundError(f"CounterFlow MMAudio script source not found: {self.script_src}")
        if self.backend_script_path.exists() and not overwrite:
            return
        shutil.copy2(self.script_src, self.backend_script_path)

    def validate(self) -> None:
        if not self.repo_dir.exists():
            raise FileNotFoundError(
                f"MMAudio repository not found: {self.repo_dir}. "
                "Run scripts/setup_external_repos.sh first."
            )
        if not self.script_src.exists():
            raise FileNotFoundError(f"CounterFlow MMAudio script source not found: {self.script_src}")

    def _project_path(self, path: str | Path) -> Path:
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = self.project_root / candidate
        return candidate.resolve()

    def _normalize_backend_args(self, args: Sequence[str]) -> list[str]:
        normalized: list[str] = []
        pending_path_option = False
        for arg in args:
            if pending_path_option:
                normalized.append(str(self._project_path(arg)))
                pending_path_option = False
                continue

            normalized.append(arg)
            pending_path_option = arg in BACKEND_PATH_OPTIONS
        return normalized

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
            str(self._project_path(output_dir)),
            "--exp_name",
            exp_name,
            "--subset",
            subset,
        ]
        if clean_csv_path is not None:
            command.extend(["--clean_csv_path", str(self._project_path(clean_csv_path))])
        command.extend(self._normalize_backend_args(extra_args or []))
        return command

    def run(self, command: Sequence[str], *, dry_run: bool = False) -> subprocess.CompletedProcess[str] | None:
        if dry_run:
            print(" ".join(command))
            return None
        return subprocess.run(command, cwd=self.repo_dir, check=True, text=True)
