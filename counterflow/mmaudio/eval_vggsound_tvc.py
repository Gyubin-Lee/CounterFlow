"""Run CounterFlow-MMAudio on the ControlFoley VGGSound-TVC benchmark.

The benchmark TSV supplies one source prompt (label_L0) and four conflicting
target prompts per video. This script generates only the counterfactual target
tasks, so the full benchmark is 5,001 videos x 4 targets = 20,004 generations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import torch
import torchaudio
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MMAUDIO_ROOT = PROJECT_ROOT / "external" / "MMAudio"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(MMAUDIO_ROOT) not in sys.path:
    sys.path.insert(0, str(MMAUDIO_ROOT))

from mmaudio.eval_utils import ModelConfig, all_model_cfg, load_video
from mmaudio.model.networks import MMAudio, get_my_mmaudio
from mmaudio.model.utils.features_utils import FeaturesUtils

from counterflow.mmaudio.eval_vggsound_sparse import (
    combine_video_audio,
    generate_audio_prompt_switch_batched,
    prepare_conditions_transition_batched,
)


DEFAULT_TSV_PATH = PROJECT_ROOT / "external" / "controlfoley" / "VGGSound-TVC" / "conflict_dataset_full.tsv"
DEFAULT_VIDEO_ROOT = Path("/media/daftpunk5/dataset/vggsound/video")
DEFAULT_OUTPUT_BASE = (
    PROJECT_ROOT
    / "results"
    / "research_axes"
    / "latent_update_method"
    / "evaluation"
    / "VGGSound-TVC"
    / "qualitative"
)
DEFAULT_TARGET_FIELDS = ("label_L1_subject", "label_L1_action", "label_L2", "label_L3")

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TVCEntry:
    sample_id: str
    source_prompt: str
    targets: dict[str, str]
    video_root: Path

    @property
    def video_id(self) -> str:
        return self.sample_id.rsplit("_", 1)[0]

    @property
    def start_sec(self) -> int | None:
        parts = self.sample_id.rsplit("_", 1)
        if len(parts) != 2 or not parts[1].isdigit():
            return None
        return int(parts[1])

    @property
    def video_path(self) -> Path:
        return self.video_root / f"{self.sample_id}.mp4"


def safe_component(text: str, *, max_len: int = 64) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip()).strip("_")
    if not slug:
        slug = "prompt"
    return slug[:max_len]


def target_level(target_field: str) -> str:
    if target_field.startswith("label_"):
        return target_field[len("label_"):]
    return target_field


def target_stem(target_field: str, target_prompt: str) -> str:
    digest = hashlib.sha1(f"{target_field}|{target_prompt}".encode("utf-8")).hexdigest()[:8]
    return f"{target_field}_{safe_component(target_prompt)}_{digest}"


def target_paths(exp_dir: Path, entry: TVCEntry, target_field: str) -> tuple[Path, Path]:
    wrong_dir = exp_dir / entry.sample_id / "wrong"
    stem = target_stem(target_field, entry.targets[target_field])
    return (
        wrong_dir / f"{entry.sample_id}_{stem}.wav",
        wrong_dir / f"{entry.sample_id}_{stem}.mp4",
    )


def load_tvc_entries(
    tsv_path: Path,
    *,
    target_fields: Iterable[str],
    video_root: Path,
) -> list[TVCEntry]:
    if not tsv_path.exists():
        raise FileNotFoundError(f"VGGSound-TVC TSV not found: {tsv_path}")

    target_fields = tuple(target_fields)
    entries: list[TVCEntry] = []
    with tsv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        required = {"video_id", "label_L0", *target_fields}
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"TSV is missing required columns {missing}: {tsv_path}")

        for row in reader:
            sample_id = (row.get("video_id") or "").strip()
            source_prompt = (row.get("label_L0") or "").strip()
            targets = {
                field: (row.get(field) or "").strip()
                for field in target_fields
            }
            if not sample_id or not source_prompt or any(not value for value in targets.values()):
                log.warning("Skipping incomplete VGGSound-TVC row: %s", row)
                continue
            entries.append(
                TVCEntry(
                    sample_id=sample_id,
                    source_prompt=source_prompt,
                    targets=targets,
                    video_root=video_root,
                )
            )

    if not entries:
        raise ValueError(f"No valid VGGSound-TVC entries found in {tsv_path}")
    return entries


def existing_target_results(exp_dir: Path, entry: TVCEntry) -> dict[str, dict[str, str]]:
    results: dict[str, dict[str, str]] = {}
    for target_field, target_prompt in entry.targets.items():
        wav_path, mp4_path = target_paths(exp_dir, entry, target_field)
        if wav_path.exists() and mp4_path.exists():
            results[target_field] = {
                "source_field": "label_L0",
                "source_prompt": entry.source_prompt,
                "target_field": target_field,
                "target_level": target_level(target_field),
                "conflict_level": target_level(target_field),
                "target_prompt": target_prompt,
                "audio_file": wav_path.name,
                "video_file": mp4_path.name,
            }
    return results


def is_complete(exp_dir: Path, entry: TVCEntry) -> bool:
    existing = existing_target_results(exp_dir, entry)
    return all(field in existing for field in entry.targets)


def write_metadata(exp_dir: Path, entry: TVCEntry, args: argparse.Namespace) -> None:
    wrong_dir = exp_dir / entry.sample_id / "wrong"
    wrong_dir.mkdir(parents=True, exist_ok=True)

    existing = existing_target_results(exp_dir, entry)
    results = [existing[field] for field in entry.targets if field in existing]
    common = {
        "backend": "mmaudio",
        "method": "counterflow",
        "benchmark": "VGGSound-TVC",
        "variant": args.variant,
        "seed": args.seed,
        "duration": args.duration,
        "num_steps": args.num_steps,
        "sigma": args.sigma,
        "init_ode": args.init_ode,
        "transition_ode": args.transition_ode,
        "transition_step": args.transition_step,
        "cfg_strength": args.cfg_strength,
        "cfg_video": args.cfg_video,
        "cfg_text": args.cfg_text,
        "neg_src": args.neg_src,
        "neg_src_both": args.neg_src_both,
        "batch_size": args.batch_size,
        "dtype": args.dtype,
    }
    wrong_meta = {
        **common,
        "sample_id": entry.sample_id,
        "video_id": entry.video_id,
        "start_sec": entry.start_sec,
        "video_path": str(entry.video_path),
        "correct_category": entry.source_prompt,
        "source_field": "label_L0",
        "source_prompt": entry.source_prompt,
        "target_fields": list(entry.targets.keys()),
        "generation_type": "counterflow_vggsound_tvc_prompt_switch",
        "results": results,
    }
    (wrong_dir / "metadata.json").write_text(
        json.dumps(wrong_meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def write_run_files(exp_dir: Path, args: argparse.Namespace, entries: list[TVCEntry]) -> None:
    exp_dir.mkdir(parents=True, exist_ok=True)
    config_path = exp_dir / "config.json"
    if not config_path.exists():
        config = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "backend": "mmaudio",
            "method": "counterflow",
            "benchmark": "VGGSound-TVC",
            "num_entries_total_for_this_invocation": len(entries),
            "num_target_tasks_total_for_this_invocation": sum(len(e.targets) for e in entries),
            "args": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
        }
        config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    command_path = exp_dir / f"command_shard{args.shard_index:02d}.txt"
    command_path.write_text(" ".join(sys.argv) + "\n", encoding="utf-8")

    launch_record = {
        "launched_at": datetime.now().isoformat(timespec="seconds"),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "gpu": args.gpu,
        "dtype": args.dtype,
        "command": sys.argv,
    }
    with (exp_dir / "launch_history.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(launch_record, ensure_ascii=False) + "\n")

    run_md_path = exp_dir / "RUN.md"
    if not run_md_path.exists():
        status = subprocess.run(
            ["git", "status", "--short"],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=False,
        ).stdout.strip()
        run_md = (
            f"# {args.exp_name}\n\n"
            f"- backend: CounterFlow-MMAudio `{args.variant}`\n"
            f"- benchmark: ControlFoley VGGSound-TVC\n"
            f"- source field: `label_L0`\n"
            f"- target fields: `{', '.join(args.target_fields)}`\n"
            f"- target tasks: `{len(entries)} entries in this shard invocation x {len(args.target_fields)}`\n"
            f"- sampling: ODE phase1={args.init_ode}, ODE phase2={args.transition_ode}, "
            f"steps={args.num_steps}, transition_step={args.transition_step}\n"
            f"- guidance: cfg_video={args.cfg_video}, cfg_text={args.cfg_text}, "
            f"cfg_strength={args.cfg_strength}, neg_src_both={args.neg_src_both}\n"
            f"- git_status_short at launch:\n\n"
            f"```text\n{status}\n```\n"
        )
        run_md_path.write_text(run_md, encoding="utf-8")


def select_shard(entries: list[TVCEntry], shard_index: int, num_shards: int) -> list[TVCEntry]:
    if num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    if not 0 <= shard_index < num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < num-shards")
    return entries[shard_index::num_shards]


def dtype_from_arg(dtype: str) -> torch.dtype:
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float16":
        return torch.float16
    if dtype == "float32":
        return torch.float32
    raise ValueError(f"Unknown dtype: {dtype}")


def load_models(args: argparse.Namespace, device: str, dtype: torch.dtype):
    if args.variant not in all_model_cfg:
        raise ValueError(f"Unknown MMAudio variant: {args.variant}")

    model_cfg: ModelConfig = all_model_cfg[args.variant]
    model_cfg.download_if_needed()
    seq_cfg = model_cfg.seq_cfg
    sampling_rate = seq_cfg.sampling_rate

    net: MMAudio = get_my_mmaudio(model_cfg.model_name).to(device, dtype).eval()
    net.load_weights(torch.load(model_cfg.model_path, map_location=device, weights_only=True))

    feature_utils = FeaturesUtils(
        tod_vae_ckpt=model_cfg.vae_path,
        synchformer_ckpt=model_cfg.synchformer_ckpt,
        enable_conditions=True,
        mode=model_cfg.mode,
        bigvgan_vocoder_ckpt=model_cfg.bigvgan_16k_path,
        need_vae_encoder=False,
    ).to(device, dtype).eval()

    return model_cfg, net, feature_utils, seq_cfg, sampling_rate


@torch.inference_mode()
def process_entry(
    entry: TVCEntry,
    args: argparse.Namespace,
    exp_dir: Path,
    net: MMAudio,
    feature_utils: FeaturesUtils,
    seq_cfg,
    sampling_rate: int,
    device: str,
    dtype: torch.dtype,
) -> str:
    if not entry.video_path.exists():
        log.warning("Skipping missing video: %s", entry.video_path)
        return "missing_video"

    existing = existing_target_results(exp_dir, entry)
    pending_fields = [field for field in entry.targets if field not in existing]
    if not pending_fields:
        write_metadata(exp_dir, entry, args)
        return "skipped_complete"

    sample_dir = exp_dir / entry.sample_id
    (sample_dir / "wrong").mkdir(parents=True, exist_ok=True)

    loaded_video = load_video(entry.video_path, args.duration)
    video_info = loaded_video[0] if isinstance(loaded_video, tuple) else loaded_video
    clip_frames = video_info.clip_frames.unsqueeze(0)
    sync_frames = video_info.sync_frames.unsqueeze(0)
    task_duration = video_info.duration_sec

    seq_cfg.duration = task_duration
    net.update_seq_lengths(seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len)

    for start in range(0, len(pending_fields), args.batch_size):
        fields = pending_fields[start:start + args.batch_size]
        prompts = [entry.targets[field] for field in fields]
        n = len(prompts)

        (
            init_video_cond,
            init_text_cond,
            transition_cond,
            empty_cond,
            src_text_cond,
        ) = prepare_conditions_transition_batched(
            net,
            feature_utils,
            clip_frames,
            sync_frames,
            target_prompts=prompts,
            source_prompt=entry.source_prompt,
            neg_src=args.neg_src,
            neg_src_both=args.neg_src_both,
            device=device,
            dtype=dtype,
        )

        audios = generate_audio_prompt_switch_batched(
            net,
            feature_utils,
            init_video_cond,
            init_text_cond,
            transition_cond,
            empty_cond,
            src_text_cond,
            batch_size=n,
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

        for field, audio in zip(fields, audios):
            wav_path, mp4_path = target_paths(exp_dir, entry, field)
            wav_path.parent.mkdir(parents=True, exist_ok=True)
            torchaudio.save(str(wav_path), audio, sampling_rate)
            combine_video_audio(entry.video_path, audio, sampling_rate, str(mp4_path))

        write_metadata(exp_dir, entry, args)

    return "generated"


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CounterFlow-MMAudio on VGGSound-TVC")
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE)
    parser.add_argument("--tsv-path", type=Path, default=DEFAULT_TSV_PATH)
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--target-fields", nargs="+", default=list(DEFAULT_TARGET_FIELDS))

    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")

    parser.add_argument("--variant", default="large_44k_v2")
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--num_steps", type=int, default=25)
    parser.add_argument("--cfg_strength", type=float, default=4.5)
    parser.add_argument("--cfg_video", type=float, default=3.0)
    parser.add_argument("--cfg_text", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sigma", type=float, default=0.0)
    parser.add_argument("--transition_step", type=int, default=17)
    parser.add_argument("--init_ode", type=int, choices=[0, 1], default=1)
    parser.add_argument("--transition_ode", type=int, choices=[0, 1], default=1)
    parser.add_argument("--neg_src", action="store_true", default=False)
    parser.add_argument("--neg_src_both", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = parse_args(argv)
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    args.init_ode = bool(args.init_ode)
    args.transition_ode = bool(args.transition_ode)
    if args.neg_src_both:
        args.neg_src = True

    exp_dir = args.output_base / args.exp_name
    entries = load_tvc_entries(
        args.tsv_path,
        target_fields=args.target_fields,
        video_root=args.video_root,
    )
    if args.start_index:
        entries = entries[args.start_index:]
    if args.limit is not None:
        entries = entries[:args.limit]
    entries = select_shard(entries, args.shard_index, args.num_shards)

    completed = sum(1 for entry in entries if is_complete(exp_dir, entry))
    total_tasks = sum(len(entry.targets) for entry in entries)
    completed_tasks = sum(len(existing_target_results(exp_dir, entry)) for entry in entries)
    log.info(
        "Experiment %s shard %d/%d: %d entries, %d target tasks, %d completed tasks, %d pending tasks",
        args.exp_name,
        args.shard_index,
        args.num_shards,
        len(entries),
        total_tasks,
        completed_tasks,
        total_tasks - completed_tasks,
    )
    log.info("%d/%d entries are fully complete", completed, len(entries))

    if args.dry_run:
        for entry in entries[:10]:
            log.info(
                "DRY %s | source=%s | targets=%s",
                entry.sample_id,
                entry.source_prompt,
                " ; ".join(f"{field}={entry.targets[field]}" for field in entry.targets),
            )
        return

    write_run_files(exp_dir, args, entries)

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = f"cuda:{args.gpu}"
    else:
        device = "cpu"
        log.warning("CUDA is unavailable; running on CPU")
    dtype = dtype_from_arg(args.dtype)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    _, net, feature_utils, seq_cfg, sampling_rate = load_models(args, device, dtype)
    status_counts: dict[str, int] = {}
    start_time = time.time()

    for entry in tqdm(entries, desc=f"shard {args.shard_index}"):
        try:
            status = process_entry(
                entry,
                args,
                exp_dir,
                net,
                feature_utils,
                seq_cfg,
                sampling_rate,
                device,
                dtype,
            )
        except Exception:
            status = "error"
            error_dir = exp_dir / entry.sample_id
            error_dir.mkdir(parents=True, exist_ok=True)
            (error_dir / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
            log.exception("Failed processing %s", entry.sample_id)
        status_counts[status] = status_counts.get(status, 0) + 1

    elapsed = time.time() - start_time
    summary = {
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "entries": len(entries),
        "target_tasks": total_tasks,
        "elapsed_sec": elapsed,
        "status_counts": status_counts,
    }
    (exp_dir / f"shard_{args.shard_index:02d}_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info("Shard complete: %s", summary)


if __name__ == "__main__":
    main()
