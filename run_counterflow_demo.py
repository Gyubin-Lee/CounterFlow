"""Run the public CounterFlow-MMAudio cat/dog prompt-switch demo.

Required environment:
    conda activate MMAudio

The script uses the tracked demo videos:
    datasets/demo_videos/cat.mp4
    datasets/demo_videos/dog.mp4
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
MMAUDIO_REPO = PROJECT_ROOT / "external" / "MMAudio"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(MMAUDIO_REPO) not in sys.path:
    sys.path.insert(0, str(MMAUDIO_REPO))

@dataclass(frozen=True)
class DemoExample:
    name: str
    video_path: Path
    source_prompt: str
    target_prompt: str


DEMO_EXAMPLES = [
    DemoExample(
        name="cat_to_horse",
        video_path=PROJECT_ROOT / "datasets" / "demo_videos" / "cat.mp4",
        source_prompt="cat meowing",
        target_prompt="horse neighing",
    ),
    DemoExample(
        name="dog_to_bear",
        video_path=PROJECT_ROOT / "datasets" / "demo_videos" / "dog.mp4",
        source_prompt="dog barking",
        target_prompt="bear growling",
    ),
]


def safe_name(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the public CounterFlow-MMAudio demo")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "demo" / "counterflow-cat-dog",
    )
    parser.add_argument("--variant", default="large_44k_v2")
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--num-steps", type=int, default=25)
    parser.add_argument("--transition-step", type=int, default=8)
    parser.add_argument("--cfg-strength", type=float, default=4.5)
    parser.add_argument("--cfg-video", type=float, default=3.0)
    parser.add_argument("--cfg-text", type=float, default=4.5)
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--init-ode", action="store_true", help="Use ODE for the initial source-video phase.")
    parser.add_argument("--transition-ode", action="store_true", help="Use ODE for the target-prompt phase.")
    parser.add_argument("--neg-src", action="store_true")
    parser.add_argument("--neg-src-both", action="store_true")
    parser.add_argument("--full-precision", action="store_true")
    parser.add_argument("--skip-video-composite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not MMAUDIO_REPO.exists():
        raise FileNotFoundError("external/MMAudio is missing. Run scripts/setup_external_repos.sh first.")
    for example in DEMO_EXAMPLES:
        if not example.video_path.exists():
            raise FileNotFoundError(f"Demo video not found: {example.video_path}")

    from counterflow.mmaudio.eval_vggsound_sparse import (
        combine_video_audio,
        generate_audio_prompt_switch,
        prepare_conditions_transition,
    )
    import torch
    import torchaudio
    from mmaudio.eval_utils import all_model_cfg, load_video, setup_eval_logging
    from mmaudio.model.networks import get_my_mmaudio
    from mmaudio.model.utils.features_utils import FeaturesUtils

    setup_eval_logging()

    if args.variant not in all_model_cfg:
        raise ValueError(f"Unknown MMAudio variant: {args.variant}")

    if args.neg_src_both:
        args.neg_src = True

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = f"cuda:{args.gpu}"
    else:
        device = "cpu"
    dtype = torch.float32 if args.full_precision or device == "cpu" else torch.bfloat16

    # MMAudio's model config stores checkpoint paths relative to its repo root.
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

    for index, example in enumerate(DEMO_EXAMPLES, start=1):
        loaded_video = load_video(example.video_path, args.duration)
        video_info = loaded_video[0] if isinstance(loaded_video, tuple) else loaded_video
        clip_frames = video_info.clip_frames.unsqueeze(0)
        sync_frames = video_info.sync_frames.unsqueeze(0)

        seq_cfg.duration = video_info.duration_sec
        net.update_seq_lengths(seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len)

        (
            init_video_conditions,
            init_text_conditions,
            transition_conditions,
            empty_conditions,
            src_text_conditions,
        ) = prepare_conditions_transition(
            net,
            feature_utils,
            clip_frames,
            sync_frames,
            target_prompt=example.target_prompt,
            source_prompt=example.source_prompt,
            neg_src=args.neg_src,
            neg_src_both=args.neg_src_both,
            device=device,
            dtype=dtype,
        )

        audio = generate_audio_prompt_switch(
            net,
            feature_utils,
            init_video_conditions,
            init_text_conditions,
            transition_conditions,
            empty_conditions,
            src_text_conditions,
            device,
            dtype,
            transition_step=args.transition_step,
            num_steps=args.num_steps,
            cfg_strength=args.cfg_strength,
            cfg_video=args.cfg_video,
            cfg_text=args.cfg_text,
            seed=args.seed + index - 1,
            sigma=args.sigma,
            init_ode=args.init_ode,
            transition_ode=args.transition_ode,
            neg_src=args.neg_src,
            neg_src_both=args.neg_src_both,
        )

        sample_dir = output_dir / example.name
        sample_dir.mkdir(parents=True, exist_ok=True)
        target_slug = safe_name(example.target_prompt)
        wav_path = sample_dir / f"{example.name}_{target_slug}_counterflow_mmaudio.wav"
        mp4_path = sample_dir / f"{example.name}_{target_slug}_counterflow_mmaudio.mp4"
        metadata_path = sample_dir / "metadata.json"

        torchaudio.save(str(wav_path), audio, seq_cfg.sampling_rate)
        if not args.skip_video_composite:
            combine_video_audio(example.video_path, audio, seq_cfg.sampling_rate, str(mp4_path))

        metadata = {
            "name": example.name,
            "video_path": str(example.video_path),
            "source_prompt": example.source_prompt,
            "target_prompt": example.target_prompt,
            "generation_type": "counterflow_prompt_switch",
            "variant": args.variant,
            "duration": video_info.duration_sec,
            "cfg_strength": args.cfg_strength,
            "cfg_video": args.cfg_video,
            "cfg_text": args.cfg_text,
            "num_steps": args.num_steps,
            "transition_step": args.transition_step,
            "sigma": args.sigma,
            "seed": args.seed + index - 1,
            "init_ode": args.init_ode,
            "transition_ode": args.transition_ode,
            "neg_src": args.neg_src,
            "neg_src_both": args.neg_src_both,
            "audio_path": str(wav_path),
            "video_output_path": str(mp4_path) if not args.skip_video_composite else None,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"[{index}/{len(DEMO_EXAMPLES)}] wrote {wav_path}")


if __name__ == "__main__":
    main()
