"""Generate CounterFlow-MMAudio audio for local video files.

Required environment:
    conda activate MMAudio

Purpose:
    Run a small local-video smoke test without requiring the full
    VGGSound-Sparse CSV/feature-cache setup.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import torch
import torchaudio


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MMAUDIO_REPO = PROJECT_ROOT / "external" / "MMAudio"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(MMAUDIO_REPO) not in sys.path:
    sys.path.insert(0, str(MMAUDIO_REPO))

from counterflow.mmaudio.eval_vggsound_sparse import (  # noqa: E402
    combine_video_audio,
    generate_audio_standard,
    prepare_conditions_standard,
)
from mmaudio.eval_utils import all_model_cfg, load_video, setup_eval_logging  # noqa: E402
from mmaudio.model.networks import get_my_mmaudio  # noqa: E402
from mmaudio.model.utils.features_utils import FeaturesUtils  # noqa: E402


def resolve_path(path_text: str, *, base_dir: Path) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    candidate = (base_dir / path).resolve()
    if candidate.exists():
        return candidate
    return (PROJECT_ROOT / path).resolve()


def load_manifest(manifest_path: Path) -> list[dict[str, str]]:
    rows = []
    with open(manifest_path, newline="") as f:
        reader = csv.DictReader(f)
        required = {"video_path", "prompt"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")
        for row in reader:
            rows.append(row)
    if not rows:
        raise ValueError(f"Manifest has no rows: {manifest_path}")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CounterFlow-MMAudio on local videos")
    parser.add_argument("--manifest", type=Path, required=True, help="CSV with video_path,prompt columns")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "demo" / "mmaudio-local-videos",
    )
    parser.add_argument("--variant", default="large_44k_v2")
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--cfg-strength", type=float, default=4.5)
    parser.add_argument("--num-steps", type=int, default=25)
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--full-precision", action="store_true")
    parser.add_argument("--skip-video-composite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_eval_logging()

    if args.variant not in all_model_cfg:
        raise ValueError(f"Unknown MMAudio variant: {args.variant}")
    if not MMAUDIO_REPO.exists():
        raise FileNotFoundError("external/MMAudio is missing. Run scripts/setup_external_repos.sh first.")

    manifest_path = args.manifest.expanduser().resolve()
    rows = load_manifest(manifest_path)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = f"cuda:{args.gpu}"
    else:
        device = "cpu"
    dtype = torch.float32 if args.full_precision or device == "cpu" else torch.bfloat16

    # MMAudio's ModelConfig uses repo-relative paths for checkpoints.
    os.chdir(MMAUDIO_REPO)

    model = all_model_cfg[args.variant]
    model.download_if_needed()
    seq_cfg = model.seq_cfg

    net = get_my_mmaudio(model.model_name).to(device, dtype).eval()
    net.load_weights(torch.load(model.model_path, map_location=device, weights_only=True))

    if not hasattr(net, "sde_wrapper"):
        raise RuntimeError(
            "MMAudio CounterFlow network patch is not applied. "
            "Run: cd external/MMAudio && git apply ../../patches/mmaudio_networks_counterflow.patch"
        )

    feature_utils = FeaturesUtils(
        tod_vae_ckpt=model.vae_path,
        synchformer_ckpt=model.synchformer_ckpt,
        enable_conditions=True,
        mode=model.mode,
        bigvgan_vocoder_ckpt=model.bigvgan_16k_path,
        need_vae_encoder=False,
    )
    feature_utils = feature_utils.to(device, dtype).eval()

    for index, row in enumerate(rows, start=1):
        video_path = resolve_path(row["video_path"], base_dir=manifest_path.parent)
        prompt = row["prompt"]
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        loaded_video = load_video(video_path, args.duration)
        video_info = loaded_video[0] if isinstance(loaded_video, tuple) else loaded_video
        clip_frames = video_info.clip_frames.unsqueeze(0)
        sync_frames = video_info.sync_frames.unsqueeze(0)

        seq_cfg.duration = video_info.duration_sec
        net.update_seq_lengths(seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len)

        conditions, empty_conditions = prepare_conditions_standard(
            net,
            feature_utils,
            clip_frames,
            sync_frames,
            prompt,
            device,
            dtype,
        )
        audio = generate_audio_standard(
            net,
            feature_utils,
            conditions,
            empty_conditions,
            device,
            dtype,
            num_steps=args.num_steps,
            cfg_strength=args.cfg_strength,
            seed=args.seed + index - 1,
            sigma=args.sigma,
            phase1_sde=True,
        )

        sample_dir = output_dir / video_path.stem
        sample_dir.mkdir(parents=True, exist_ok=True)
        wav_path = sample_dir / f"{video_path.stem}_counterflow_mmaudio.wav"
        mp4_path = sample_dir / f"{video_path.stem}_counterflow_mmaudio.mp4"
        metadata_path = sample_dir / "metadata.json"

        torchaudio.save(str(wav_path), audio, seq_cfg.sampling_rate)
        if not args.skip_video_composite:
            combine_video_audio(video_path, audio, seq_cfg.sampling_rate, str(mp4_path))

        metadata = {
            "video_path": str(video_path),
            "prompt": prompt,
            "variant": args.variant,
            "duration": video_info.duration_sec,
            "cfg_strength": args.cfg_strength,
            "num_steps": args.num_steps,
            "sigma": args.sigma,
            "seed": args.seed + index - 1,
            "audio_path": str(wav_path),
            "video_output_path": str(mp4_path) if not args.skip_video_composite else None,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"[{index}/{len(rows)}] wrote {wav_path}")


if __name__ == "__main__":
    main()
