"""Dispatch ControlFoley TC-V2A experiments on VGGSound-Sparse."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import date
from pathlib import Path


def find_project_root(start: Path) -> Path:
    for path in (start, *start.parents):
        if (path / "counterflow").is_dir() and (path / "scripts").is_dir():
            return path
    raise RuntimeError(f"Could not locate CounterFlow project root from {start}")


PROJECT_ROOT = find_project_root(Path(__file__).resolve())
SCRIPT_PATH = PROJECT_ROOT / "counterflow" / "controlfoley" / "eval_vggsound_sparse.py"


def default_exp_name() -> str:
    return f"{date.today().isoformat()}_controlfoley_vggsparse_tcvta"


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Run ControlFoley TC-V2A on VGGSound-Sparse")
    parser.add_argument("--exp-name", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_known_args()


def main() -> None:
    args, backend_args = parse_args()
    exp_name = args.exp_name or default_exp_name()
    command = [
        sys.executable,
        str(SCRIPT_PATH),
        "--exp-name",
        exp_name,
        *backend_args,
    ]
    if args.dry_run:
        command.append("--dry-run")
        print(" ".join(command))
        return

    env = os.environ.copy()
    env.setdefault("PYTHONNOUSERSITE", "1")
    subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True, text=True)


if __name__ == "__main__":
    main()
