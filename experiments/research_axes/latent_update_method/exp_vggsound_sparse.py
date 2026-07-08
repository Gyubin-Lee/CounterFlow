"""Run CounterFlow experiments on VGGSound-Sparse.

Required environment:
    MMAudio backend:
        conda activate MMAudio
    Hunyuan backend:
        conda activate Hunyuan

Purpose:
    Dispatch a VGGSound-Sparse CounterFlow experiment to the selected backend.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path


def find_project_root(start: Path) -> Path:
    for path in (start, *start.parents):
        if (path / "counterflow").is_dir() and (path / "scripts").is_dir():
            return path
    raise RuntimeError(f"Could not locate CounterFlow project root from {start}")


PROJECT_ROOT = find_project_root(Path(__file__).resolve())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from counterflow.hunyuan import HunyuanCounterFlowWrapper
from counterflow.mmaudio import MMAudioCounterFlowWrapper


def default_exp_name(backend: str) -> str:
    return f"{date.today().isoformat()}_counterflow-{backend}-default"


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Run CounterFlow on VGGSound-Sparse")
    parser.add_argument("--backend", choices=["mmaudio", "hunyuan"], required=True)
    parser.add_argument("--exp-name", default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "results"
            / "research_axes"
            / "latent_update_method"
            / "evaluation"
            / "VGGSound-Sparse"
            / "qualitative"
        ),
        help="Base directory passed to the backend; backend appends --exp-name.",
    )
    parser.add_argument("--subset", choices=["all", "clean"], default="clean")
    parser.add_argument(
        "--clean-csv-path",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "VGGSound-Sparse" / "vggsound_sparse_clean_fixed_offsets.csv",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the backend command without running it.")
    return parser.parse_known_args()


def main() -> None:
    args, backend_args = parse_args()
    exp_name = args.exp_name or default_exp_name(args.backend)
    clean_csv_path = args.clean_csv_path if args.clean_csv_path.exists() else None

    if args.backend == "mmaudio":
        wrapper = MMAudioCounterFlowWrapper(PROJECT_ROOT)
    else:
        wrapper = HunyuanCounterFlowWrapper(PROJECT_ROOT)

    command = wrapper.build_command(
        output_dir=args.output_dir,
        exp_name=exp_name,
        subset=args.subset,
        clean_csv_path=clean_csv_path,
        extra_args=backend_args,
    )
    wrapper.run(command, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
