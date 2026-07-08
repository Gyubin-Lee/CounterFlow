"""HunyuanVideo-Foley CounterFlow backend wrapper.

Required environment:
    conda activate Hunyuan

Purpose:
    Build and run CounterFlow VGGSound-Sparse commands through external/HunyuanVideo-Foley.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence


class HunyuanCounterFlowWrapper:
    """Thin subprocess wrapper around the Hunyuan latent-intervention script."""

    def __init__(self, project_root: Path | None = None, repo_dir: Path | None = None) -> None:
        self.project_root = (project_root or Path(__file__).resolve().parents[2]).resolve()
        self.repo_dir = (repo_dir or self.project_root / "external" / "HunyuanVideo-Foley").resolve()
        self.script_path = self.repo_dir / "eval_vggsound_sparse.py"
        self.script_src = self.project_root / "counterflow" / "hunyuan" / "eval_vggsound_sparse.py"
        self.intervention_src = self.project_root / "counterflow" / "hunyuan" / "infer_latent_intervention.py"
        self.intervention_dst = self.repo_dir / "infer_latent_intervention.py"

    def install_backend_script(self, *, overwrite: bool = False) -> None:
        if not self.script_src.exists():
            raise FileNotFoundError(f"CounterFlow Hunyuan script source not found: {self.script_src}")
        if self.script_path.exists() and not overwrite:
            return
        shutil.copy2(self.script_src, self.script_path)

    def install_intervention_script(self, *, overwrite: bool = False) -> None:
        if not self.intervention_src.exists():
            raise FileNotFoundError(f"CounterFlow intervention source not found: {self.intervention_src}")
        if self.intervention_dst.exists() and not overwrite:
            return
        shutil.copy2(self.intervention_src, self.intervention_dst)

    def validate(self) -> None:
        if not self.repo_dir.exists():
            raise FileNotFoundError(
                f"HunyuanVideo-Foley repository not found: {self.repo_dir}. "
                "Run scripts/setup_external_repos.sh first."
            )
        if not self.script_path.exists():
            self.install_backend_script()
        if not self.intervention_dst.exists():
            self.install_intervention_script()

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
            "--model_path",
            str(self.repo_dir),
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
