"""Minimal CounterFlow demo entry point.

Required environment:
    conda activate MMAudio

Purpose:
    Launch a one-video CounterFlow-MMAudio pilot run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from counterflow.mmaudio import MMAudioCounterFlowWrapper


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Run a minimal CounterFlow demo")
    parser.add_argument("--exp-name", default="demo_counterflow")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "evaluation" / "VGGSound-Sparse" / "qualitative",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the backend command without running it.")
    return parser.parse_known_args()


def main() -> None:
    args, backend_args = parse_args()
    wrapper = MMAudioCounterFlowWrapper(PROJECT_ROOT)
    clean_csv_path = PROJECT_ROOT / "datasets" / "VGGSound-Sparse" / "vggsound_sparse_clean_fixed_offsets.csv"
    command = wrapper.build_command(
        output_dir=args.output_dir,
        exp_name=args.exp_name,
        subset="clean",
        clean_csv_path=clean_csv_path if clean_csv_path.exists() else None,
        extra_args=["--pilot", "--pilot_n", "1", *backend_args],
    )
    wrapper.run(command, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
