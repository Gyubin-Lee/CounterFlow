"""Run CounterFlow/MMAudio on AVSBench-S4 counterfactual prompt manifests."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.multiprocessing as mp
import torchaudio
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from counterflow.mmaudio import eval_vggsound_sparse as vgg


DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "datasets"
    / "AVSBench-S4"
    / "avsbench_s4_semantic_replace_temporal_preserve_pass_max050.csv"
)
DEFAULT_OUTPUT_BASE = (
    PROJECT_ROOT
    / "results"
    / "research_axes"
    / "target_prompt_generation"
    / "evaluation"
    / "AVSBench-S4"
    / "qualitative"
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TargetPrompt:
    prompt: str
    rank: int
    target_case: str
    source_target_clap_text_score: str = ""
    clap_text_filter_status: str = ""
    perceptual_separation: str = ""
    confidence: str = ""

    @property
    def key(self) -> str:
        return f"t{self.rank:02d}_{safe_slug(self.prompt)}"


@dataclass
class ManifestEntry:
    sample_id: str
    raw_video_id: str
    start_sec: int
    video_path: Path
    source_category: str
    source_prompt: str
    split: str
    targets: list[TargetPrompt] = field(default_factory=list)


def safe_slug(text: str, max_len: int = 80) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip()).strip("_")
    return (slug or "prompt")[:max_len]


def load_manifest(path: Path) -> list[ManifestEntry]:
    grouped: dict[str, ManifestEntry] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"video_id", "video_path", "source_category", "source_prompt", "target_prompt"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest missing columns {sorted(missing)}: {path}")

        for row in reader:
            sample_id = row["video_id"].strip()
            if not sample_id:
                continue
            entry = grouped.get(sample_id)
            if entry is None:
                entry = ManifestEntry(
                    sample_id=sample_id,
                    raw_video_id=(row.get("raw_video_id") or sample_id).strip(),
                    start_sec=int(row.get("start_sec") or 0),
                    video_path=Path(row["video_path"]),
                    source_category=row["source_category"].strip(),
                    source_prompt=row["source_prompt"].strip(),
                    split=(row.get("split") or "test").strip(),
                )
                grouped[sample_id] = entry

            target_prompt = row["target_prompt"].strip()
            if not target_prompt:
                continue
            rank = int(row.get("target_rank") or len(entry.targets))
            entry.targets.append(
                TargetPrompt(
                    prompt=target_prompt,
                    rank=rank,
                    target_case=row.get("target_case", ""),
                    source_target_clap_text_score=row.get("source_target_clap_text_score", ""),
                    clap_text_filter_status=row.get("clap_text_filter_status", ""),
                    perceptual_separation=row.get("perceptual_separation", ""),
                    confidence=row.get("confidence", ""),
                )
            )

    entries = [entry for entry in grouped.values() if entry.targets]
    if not entries:
        raise ValueError(f"No usable entries found in manifest: {path}")
    return entries


def correct_paths(exp_dir: Path, entry: ManifestEntry) -> tuple[Path, Path]:
    correct_dir = exp_dir / entry.sample_id / "correct"
    return correct_dir / f"{entry.sample_id}_correct.wav", correct_dir / f"{entry.sample_id}_correct.mp4"


def wrong_paths(exp_dir: Path, entry: ManifestEntry, target: TargetPrompt) -> tuple[Path, Path]:
    wrong_dir = exp_dir / entry.sample_id / "wrong"
    stem = f"{entry.sample_id}_{target.key}"
    return wrong_dir / f"{stem}.wav", wrong_dir / f"{stem}.mp4"


def existing_wrong_result(exp_dir: Path, entry: ManifestEntry, target: TargetPrompt) -> dict[str, str] | None:
    wav_path, mp4_path = wrong_paths(exp_dir, entry, target)
    if not (wav_path.exists() and mp4_path.exists()):
        return None
    return build_wrong_result(entry, target, wav_path, mp4_path)


def build_wrong_result(
    entry: ManifestEntry,
    target: TargetPrompt,
    wav_path: Path,
    mp4_path: Path,
) -> dict[str, str]:
    return {
        "source_prompt": entry.source_prompt,
        "target_prompt": target.prompt,
        "audio_file": wav_path.name,
        "video_file": mp4_path.name,
        "target_case": target.target_case,
        "target_rank": target.rank,
        "source_category": entry.source_category,
        "source_target_clap_text_score": target.source_target_clap_text_score,
        "clap_text_filter_status": target.clap_text_filter_status,
        "perceptual_separation": target.perceptual_separation,
        "confidence": target.confidence,
    }


def write_metadata(
    exp_dir: Path,
    entry: ManifestEntry,
    args: argparse.Namespace,
    wrong_results: list[dict[str, str]],
) -> None:
    sample_dir = exp_dir / entry.sample_id
    correct_dir = sample_dir / "correct"
    wrong_dir = sample_dir / "wrong"
    correct_dir.mkdir(parents=True, exist_ok=True)
    wrong_dir.mkdir(parents=True, exist_ok=True)

    common = {
        "dataset_name": "AVSBench-S4",
        "backend": "mmaudio",
        "variant": args.variant,
        "sampler": args.sampler,
        "seed": args.seed,
        "duration": args.duration,
        "num_steps": args.num_steps,
        "transition_step": args.transition_step,
        "cfg_strength": args.cfg_strength,
        "cfg_video": args.cfg_video,
        "cfg_text": args.cfg_text,
        "neg_src": args.neg_src,
        "neg_src_both": args.neg_src_both,
        "init_ode": args.init_ode,
        "transition_ode": args.transition_ode,
        "sigma": args.sigma,
    }
    correct_meta = {
        **common,
        "video_id": entry.raw_video_id,
        "sample_id": entry.sample_id,
        "start_sec": entry.start_sec,
        "video_path": str(entry.video_path),
        "correct_category": entry.source_prompt,
        "source_category": entry.source_category,
        "prompt": entry.source_prompt,
        "generation_type": "standard",
    }
    wrong_meta = {
        **common,
        "video_id": entry.raw_video_id,
        "sample_id": entry.sample_id,
        "start_sec": entry.start_sec,
        "video_path": str(entry.video_path),
        "correct_category": entry.source_prompt,
        "source_category": entry.source_category,
        "generation_type": "prompt_switch",
        "results": wrong_results,
    }
    (correct_dir / "metadata.json").write_text(json.dumps(correct_meta, indent=2), encoding="utf-8")
    (wrong_dir / "metadata.json").write_text(json.dumps(wrong_meta, indent=2), encoding="utf-8")


def process_entry(
    entry: ManifestEntry,
    args: argparse.Namespace,
    net,
    feature_utils,
    seq_cfg,
    sampling_rate: int,
    device: str,
    dtype: torch.dtype,
) -> str | None:
    exp_dir = Path(args.output_dir)
    correct_dir = exp_dir / entry.sample_id / "correct"
    wrong_dir = exp_dir / entry.sample_id / "wrong"
    correct_dir.mkdir(parents=True, exist_ok=True)
    wrong_dir.mkdir(parents=True, exist_ok=True)

    correct_wav, correct_mp4 = correct_paths(exp_dir, entry)
    correct_done = correct_wav.exists() and correct_mp4.exists()
    existing_results: dict[int, dict[str, str]] = {}
    pending_targets: list[TargetPrompt] = []
    for target in entry.targets:
        result = existing_wrong_result(exp_dir, entry, target)
        if result is None:
            pending_targets.append(target)
        else:
            existing_results[target.rank] = result

    if correct_done and not pending_targets:
        write_metadata(exp_dir, entry, args, [existing_results[t.rank] for t in entry.targets])
        return entry.sample_id

    if not entry.video_path.exists():
        log.warning("[SKIP] video not found: %s", entry.video_path)
        return None

    loaded_video = vgg.load_video(entry.video_path, args.duration)
    video_info = loaded_video[0] if isinstance(loaded_video, tuple) else loaded_video
    clip_frames = video_info.clip_frames.unsqueeze(0)
    sync_frames = video_info.sync_frames.unsqueeze(0)
    task_duration = video_info.duration_sec

    seq_cfg.duration = task_duration
    net.update_seq_lengths(seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len)

    if not correct_done:
        correct_conditions, correct_empty = vgg.prepare_conditions_standard(
            net,
            feature_utils,
            clip_frames,
            sync_frames,
            entry.source_prompt,
            device,
            dtype,
        )
        correct_audio = vgg.generate_audio_standard(
            net,
            feature_utils,
            correct_conditions,
            correct_empty,
            device,
            dtype,
            num_steps=args.num_steps,
            cfg_strength=args.cfg_strength,
            seed=args.seed,
            sigma=args.sigma,
            phase1_sde=not args.init_ode,
        )
        torchaudio.save(str(correct_wav), correct_audio, sampling_rate)
        vgg.combine_video_audio(entry.video_path, correct_audio, sampling_rate, str(correct_mp4))

    generated_results: dict[int, dict[str, str]] = {}
    if pending_targets:
        need_src_conditions = vgg.needs_source_conditions(args)
        batch_size = min(max(1, int(args.batch_size)), len(pending_targets))
        for offset in range(0, len(pending_targets), batch_size):
            chunk = pending_targets[offset : offset + batch_size]
            prompts = [target.prompt for target in chunk]
            if len(chunk) > 1:
                init_video, init_text, transition, empty, src_text = vgg.prepare_conditions_transition_batched(
                    net,
                    feature_utils,
                    clip_frames,
                    sync_frames,
                    target_prompts=prompts,
                    source_prompt=entry.source_prompt,
                    neg_src=need_src_conditions,
                    neg_src_both=args.neg_src_both,
                    device=device,
                    dtype=dtype,
                )
                if args.sampler == vgg.SAMPLER_PHASE2_VIDEO_DECAY:
                    audios = vgg.generate_audio_phase2_video_decay_batched(
                        net,
                        feature_utils,
                        init_video,
                        init_text,
                        transition,
                        empty,
                        src_text,
                        batch_size=len(chunk),
                        device=device,
                        dtype=dtype,
                        transition_step=args.transition_step,
                        num_steps=args.num_steps,
                        cfg_video=args.cfg_video,
                        cfg_text=args.cfg_text,
                        seed=args.seed,
                        phase2_schedule=args.phase2_schedule,
                        phase2_w_vid_start=args.phase2_w_vid_start,
                        phase2_w_vid_end=args.phase2_w_vid_end,
                        phase2_w_tar=args.phase2_w_tar,
                        phase2_w_src_const=args.phase2_w_src_const,
                        phase2_src_mode=args.phase2_src_mode,
                        phase2_src_vid_multiplier=args.phase2_src_vid_multiplier,
                    )
                elif args.sampler == vgg.SAMPLER_SMOOTH_MEAN_COUPLED:
                    audios = vgg.generate_audio_smooth_mean_coupled_batched(
                        net,
                        feature_utils,
                        init_video,
                        init_text,
                        empty,
                        src_text,
                        batch_size=len(chunk),
                        device=device,
                        dtype=dtype,
                        num_steps=args.num_steps,
                        seed=args.seed,
                        smooth_schedule=args.smooth_schedule,
                        smooth_w_vid_start=args.smooth_w_vid_start,
                        smooth_w_vid_end=args.smooth_w_vid_end,
                        smooth_w_tar_start=args.smooth_w_tar_start,
                        smooth_w_tar_end=args.smooth_w_tar_end,
                        smooth_src_alpha=args.smooth_src_alpha,
                        smooth_src_basis=args.smooth_src_basis,
                    )
                else:
                    audios = vgg.generate_audio_prompt_switch_batched(
                        net,
                        feature_utils,
                        init_video,
                        init_text,
                        transition,
                        empty,
                        src_text,
                        batch_size=len(chunk),
                        device=device,
                        dtype=dtype,
                        transition_step=args.transition_step,
                        num_steps=args.num_steps,
                        cfg_strength=args.cfg_strength,
                        cfg_video=args.cfg_video,
                        cfg_text=args.cfg_text,
                        seed=args.seed,
                        sigma=args.sigma,
                        init_ode=args.init_ode,
                        transition_ode=args.transition_ode,
                        neg_src=args.neg_src,
                        neg_src_both=args.neg_src_both,
                    )
            else:
                target = chunk[0]
                init_video, init_text, transition, empty, src_text = vgg.prepare_conditions_transition(
                    net,
                    feature_utils,
                    clip_frames,
                    sync_frames,
                    target_prompt=target.prompt,
                    source_prompt=entry.source_prompt,
                    neg_src=need_src_conditions,
                    neg_src_both=args.neg_src_both,
                    device=device,
                    dtype=dtype,
                )
                if args.sampler == vgg.SAMPLER_PHASE2_VIDEO_DECAY:
                    audios = [
                        vgg.generate_audio_phase2_video_decay(
                            net,
                            feature_utils,
                            init_video,
                            init_text,
                            transition,
                            empty,
                            src_text,
                            device,
                            dtype,
                            transition_step=args.transition_step,
                            num_steps=args.num_steps,
                            cfg_video=args.cfg_video,
                            cfg_text=args.cfg_text,
                            seed=args.seed,
                            phase2_schedule=args.phase2_schedule,
                            phase2_w_vid_start=args.phase2_w_vid_start,
                            phase2_w_vid_end=args.phase2_w_vid_end,
                            phase2_w_tar=args.phase2_w_tar,
                            phase2_w_src_const=args.phase2_w_src_const,
                            phase2_src_mode=args.phase2_src_mode,
                            phase2_src_vid_multiplier=args.phase2_src_vid_multiplier,
                        )
                    ]
                else:
                    audios = [
                        vgg.generate_audio_prompt_switch(
                            net,
                            feature_utils,
                            init_video,
                            init_text,
                            transition,
                            empty,
                            src_text,
                            device,
                            dtype,
                            transition_step=args.transition_step,
                            num_steps=args.num_steps,
                            cfg_strength=args.cfg_strength,
                            cfg_video=args.cfg_video,
                            cfg_text=args.cfg_text,
                            seed=args.seed,
                            sigma=args.sigma,
                            init_ode=args.init_ode,
                            transition_ode=args.transition_ode,
                            neg_src=args.neg_src,
                            neg_src_both=args.neg_src_both,
                        )
                    ]

            for target, audio in zip(chunk, audios):
                wav_path, mp4_path = wrong_paths(exp_dir, entry, target)
                torchaudio.save(str(wav_path), audio, sampling_rate)
                vgg.combine_video_audio(entry.video_path, audio, sampling_rate, str(mp4_path))
                generated_results[target.rank] = build_wrong_result(entry, target, wav_path, mp4_path)

    wrong_results = [existing_results.get(t.rank) or generated_results[t.rank] for t in entry.targets]
    write_metadata(exp_dir, entry, args, wrong_results)
    return entry.sample_id


def worker_fn(gpu_id: int, entries: list[ManifestEntry], args: argparse.Namespace) -> None:
    device = f"cuda:{gpu_id}"
    dtype = torch.bfloat16
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model_cfg: vgg.ModelConfig = vgg.all_model_cfg[args.variant]
    model_cfg.download_if_needed()
    seq_cfg = model_cfg.seq_cfg
    sampling_rate = seq_cfg.sampling_rate

    net = vgg.get_my_mmaudio(model_cfg.model_name).to(device, dtype).eval()
    net.load_weights(torch.load(model_cfg.model_path, map_location=device, weights_only=True))
    feature_utils = vgg.FeaturesUtils(
        tod_vae_ckpt=model_cfg.vae_path,
        synchformer_ckpt=model_cfg.synchformer_ckpt,
        enable_conditions=True,
        mode=model_cfg.mode,
        bigvgan_vocoder_ckpt=model_cfg.bigvgan_16k_path,
        need_vae_encoder=False,
    ).to(device, dtype).eval()

    failed = []
    for entry in tqdm(entries, desc=f"GPU {gpu_id}", position=gpu_id):
        try:
            if process_entry(entry, args, net, feature_utils, seq_cfg, sampling_rate, device, dtype) is None:
                failed.append(entry.sample_id)
        except Exception:
            log.exception("[GPU %s] Failed %s", gpu_id, entry.sample_id)
            failed.append(entry.sample_id)
    if failed:
        log.warning("[GPU %s] Failed %d entries", gpu_id, len(failed))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_BASE)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sampler", choices=[vgg.SAMPLER_STRICT_2PHASE, vgg.SAMPLER_SMOOTH_MEAN_COUPLED, vgg.SAMPLER_PHASE2_VIDEO_DECAY], default=vgg.SAMPLER_STRICT_2PHASE)
    parser.add_argument("--neg_src", action="store_true")
    parser.add_argument("--neg_src_both", action="store_true")
    parser.add_argument("--init_ode", type=int, choices=[0, 1], default=1)
    parser.add_argument("--transition_ode", type=int, choices=[0, 1], default=1)
    parser.add_argument("--sigma", type=float, default=0.0)
    parser.add_argument("--transition_step", type=int, default=17)
    parser.add_argument("--num_steps", type=int, default=25)
    parser.add_argument("--cfg_strength", type=float, default=4.5)
    parser.add_argument("--cfg_video", type=float, default=3.0)
    parser.add_argument("--cfg_text", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--variant", default="large_44k_v2")
    parser.add_argument("--smooth_schedule", choices=["smoothstep", "cosine", "linear"], default="smoothstep")
    parser.add_argument("--smooth_w_vid_start", type=float, default=3.5)
    parser.add_argument("--smooth_w_vid_end", type=float, default=1.2)
    parser.add_argument("--smooth_w_tar_start", type=float, default=1.8)
    parser.add_argument("--smooth_w_tar_end", type=float, default=5.0)
    parser.add_argument("--smooth_src_alpha", type=float, default=1.0)
    parser.add_argument("--smooth_src_basis", choices=["mean", "w_vid"], default="mean")
    parser.add_argument("--phase2_schedule", choices=["smoothstep", "cosine", "linear"], default="smoothstep")
    parser.add_argument("--phase2_w_vid_start", type=float, default=3.0)
    parser.add_argument("--phase2_w_vid_end", type=float, default=0.5)
    parser.add_argument("--phase2_w_tar", type=float, default=4.5)
    parser.add_argument("--phase2_w_src_const", type=float, default=None)
    parser.add_argument("--phase2_src_mode", choices=["const", "1p5_vid"], default="const")
    parser.add_argument("--phase2_src_vid_multiplier", type=float, default=1.5)
    args = parser.parse_args()
    args.init_ode = bool(args.init_ode)
    args.transition_ode = bool(args.transition_ode)
    if args.neg_src_both:
        args.neg_src = True
    if args.sampler == vgg.SAMPLER_PHASE2_VIDEO_DECAY:
        args.neg_src = True
        args.neg_src_both = True
    args.output_dir = str(args.output_dir / args.exp_name)
    return args


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)-8s] %(message)s")
    args = parse_args()
    entries = load_manifest(args.manifest)
    if args.limit is not None:
        entries = entries[: args.limit]
    total_targets = sum(len(entry.targets) for entry in entries)
    log.info("Loaded %d videos / %d target rows from %s", len(entries), total_targets, args.manifest)
    log.info("Output: %s", args.output_dir)
    if args.dry_run:
        return

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.num_gpus > 1:
        chunks = [[] for _ in range(args.num_gpus)]
        for idx, entry in enumerate(entries):
            chunks[idx % args.num_gpus].append(entry)
        mp.set_start_method("spawn", force=True)
        procs = []
        for gpu_id, chunk in enumerate(chunks):
            if not chunk:
                continue
            proc = mp.Process(target=worker_fn, args=(gpu_id, chunk, args))
            proc.start()
            procs.append(proc)
        for proc in procs:
            proc.join()
    else:
        worker_fn(args.gpu, entries, args)


if __name__ == "__main__":
    main()
