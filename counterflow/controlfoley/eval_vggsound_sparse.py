"""Run ControlFoley TC-V2A on VGGSound-Sparse in CounterFlow eval format.

Required environment:
    PYTHONNOUSERSITE=1 conda run -n MMAudio python ...

This script keeps the external ControlFoley repository read-only and writes
generated media under results/. It emits the same directory/metadata structure
as the existing CounterFlow VGGSound-Sparse evaluation scripts:

    <exp>/<sample_id>/correct/<sample_id>_correct.{wav,mp4}
    <exp>/<sample_id>/wrong/<sample_id>_<target>.{wav,mp4}
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import torch
import torchaudio
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTROLFOLEY_ROOT = PROJECT_ROOT / "external" / "controlfoley"
VIDEO_ROOT = Path("/media/daftpunk5/dataset/vggsound/video")
DEFAULT_OUTPUT_BASE = (
    PROJECT_ROOT
    / "results"
    / "research_axes"
    / "evaluation_metrics"
    / "evaluation"
    / "VGGSound-Sparse"
    / "qualitative"
)
DEFAULT_CSV_PATH = PROJECT_ROOT / "datasets" / "VGGSound-Sparse" / "vggsound_sparse.csv"
DEFAULT_CLEAN_CSV_PATH = (
    PROJECT_ROOT / "datasets" / "VGGSound-Sparse" / "vggsound_sparse_clean_fixed_offsets.csv"
)
DEFAULT_FEATURE_CACHE_DIR = (
    PROJECT_ROOT / "datasets" / "VGGSound-Sparse" / "controlfoley_precomputed" / "video_8s"
)

CATEGORIES = [
    "skateboarding",
    "playing tennis",
    "hammering nails",
    "ice cracking",
    "people eating crisps",
    "dog barking",
    "playing badminton",
    "people sneezing",
    "lions roaring",
    "people eating apple",
    "chopping wood",
    "striking bowling",
]

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class VideoEntry:
    video_id: str
    start_sec: int
    category: str
    split: str

    @property
    def sample_id(self) -> str:
        return f"{self.video_id}_{self.start_sec:06d}"

    @property
    def filename(self) -> str:
        return f"{self.sample_id}.mp4"

    @property
    def video_path(self) -> Path:
        return VIDEO_ROOT / self.filename


def setup_controlfoley_imports() -> None:
    if not CONTROLFOLEY_ROOT.exists():
        raise FileNotFoundError(
            f"ControlFoley repo not found: {CONTROLFOLEY_ROOT}. "
            "Clone https://github.com/xiaomi-research/controlfoley under external/controlfoley first."
        )
    for path in (CONTROLFOLEY_ROOT / "lib", CONTROLFOLEY_ROOT):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def safe_category(category: str) -> str:
    return category.replace(" ", "_").replace("/", "_")


def duration_tag(duration: float) -> str:
    if abs(duration - round(duration)) < 1e-9:
        return f"{int(round(duration))}s"
    return f"{duration:g}".replace(".", "p") + "s"


def stable_seed(base_seed: int, sample_id: str, prompt: str) -> int:
    digest = hashlib.sha1(f"{base_seed}|{sample_id}|{prompt}".encode("utf-8")).hexdigest()
    return (base_seed + int(digest[:8], 16)) % (2**31 - 1)


def load_clean_video_keys(clean_csv_path: Path) -> set[tuple[str, int]]:
    if not clean_csv_path.exists():
        raise FileNotFoundError(f"Clean subset CSV not found: {clean_csv_path}")

    keys: set[tuple[str, int]] = set()
    with clean_csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if "video_id" not in (reader.fieldnames or []):
            raise ValueError(f"Clean CSV must contain a video_id column: {clean_csv_path}")
        for row in reader:
            raw_video_id = (row.get("video_id") or "").strip()
            parts = raw_video_id.rsplit("_", 2)
            if len(parts) != 3:
                continue
            yt_id, start_ms, _ = parts
            if start_ms.isdigit():
                keys.add((yt_id, int(start_ms) // 1000))

    if not keys:
        raise ValueError(f"No valid video IDs found in clean CSV: {clean_csv_path}")
    return keys


def load_vggsound_sparse_entries(
    csv_path: Path,
    *,
    subset: str,
    clean_csv_path: Path,
) -> list[VideoEntry]:
    if not csv_path.exists():
        raise FileNotFoundError(f"VGGSound-Sparse CSV not found: {csv_path}")

    entries: list[VideoEntry] = []
    with csv_path.open(newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 4:
                continue
            try:
                start_sec = int(row[1])
            except ValueError:
                continue
            if row[3] != "test":
                continue
            entries.append(
                VideoEntry(
                    video_id=row[0],
                    start_sec=start_sec,
                    category=row[2],
                    split=row[3],
                )
            )

    if subset == "clean":
        clean_keys = load_clean_video_keys(clean_csv_path)
        before = len(entries)
        entries = [entry for entry in entries if (entry.video_id, entry.start_sec) in clean_keys]
        log.info("Applied clean subset filter: kept %d / %d test videos", len(entries), before)

    missing_categories = sorted({entry.category for entry in entries} - set(CATEGORIES))
    if missing_categories:
        raise ValueError(f"Unexpected VGGSound-Sparse categories: {missing_categories}")

    return entries


def get_wrong_categories(correct_category: str) -> list[str]:
    return [category for category in CATEGORIES if category != correct_category]


def correct_paths(exp_dir: Path, entry: VideoEntry) -> tuple[Path, Path]:
    correct_dir = exp_dir / entry.sample_id / "correct"
    return (
        correct_dir / f"{entry.sample_id}_correct.wav",
        correct_dir / f"{entry.sample_id}_correct.mp4",
    )


def wrong_paths(exp_dir: Path, entry: VideoEntry, target_category: str) -> tuple[Path, Path]:
    wrong_dir = exp_dir / entry.sample_id / "wrong"
    safe_target = safe_category(target_category)
    return (
        wrong_dir / f"{entry.sample_id}_{safe_target}.wav",
        wrong_dir / f"{entry.sample_id}_{safe_target}.mp4",
    )


def is_complete(exp_dir: Path, entry: VideoEntry) -> bool:
    correct_wav, correct_mp4 = correct_paths(exp_dir, entry)
    if not (correct_wav.exists() and correct_mp4.exists()):
        return False
    return all(
        wav.exists() and mp4.exists()
        for wav, mp4 in (wrong_paths(exp_dir, entry, category) for category in get_wrong_categories(entry.category))
    )


def existing_wrong_results(exp_dir: Path, entry: VideoEntry) -> dict[str, dict[str, str]]:
    results: dict[str, dict[str, str]] = {}
    for target_category in get_wrong_categories(entry.category):
        wav_path, mp4_path = wrong_paths(exp_dir, entry, target_category)
        if wav_path.exists() and mp4_path.exists():
            results[target_category] = {
                "source_prompt": entry.category,
                "target_prompt": target_category,
                "audio_file": wav_path.name,
                "video_file": mp4_path.name,
            }
    return results


def write_metadata(exp_dir: Path, entry: VideoEntry, args: argparse.Namespace) -> None:
    sample_dir = exp_dir / entry.sample_id
    correct_dir = sample_dir / "correct"
    wrong_dir = sample_dir / "wrong"
    correct_dir.mkdir(parents=True, exist_ok=True)
    wrong_dir.mkdir(parents=True, exist_ok=True)

    common = {
        "backend": "controlfoley",
        "variant": args.variant,
        "task": "TC-V2A",
        "seed": args.seed,
        "seed_mode": args.seed_mode,
        "duration": args.duration,
        "num_steps": args.num_steps,
        "cfg_strength": args.cfg_strength,
        "prompt_batch_size": args.prompt_batch_size,
        "negative_prompt_mode": args.negative_prompt_mode,
        "mask_away_clip": args.mask_away_clip,
        "feature_cache_dir": str(args.feature_cache_dir) if args.feature_cache_dir else None,
    }

    correct_meta = {
        **common,
        "video_id": entry.video_id,
        "start_sec": entry.start_sec,
        "video_path": str(entry.video_path),
        "correct_category": entry.category,
        "prompt": entry.category,
        "generation_type": "video_text_consistent",
    }
    (correct_dir / "metadata.json").write_text(
        json.dumps(correct_meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    wrong_meta_results = []
    existing_results = existing_wrong_results(exp_dir, entry)
    for target_category in get_wrong_categories(entry.category):
        if target_category in existing_results:
            result = dict(existing_results[target_category])
            result["negative_prompt"] = (
                entry.category if args.negative_prompt_mode == "source" else ""
            )
            wrong_meta_results.append(result)

    wrong_meta = {
        **common,
        "video_id": entry.video_id,
        "start_sec": entry.start_sec,
        "video_path": str(entry.video_path),
        "correct_category": entry.category,
        "generation_type": "text_controlled_video_to_audio",
        "results": wrong_meta_results,
    }
    (wrong_dir / "metadata.json").write_text(
        json.dumps(wrong_meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def run_ffmpeg_mux(video_path: Path, wav_path: Path, mp4_path: Path) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-i",
        str(wav_path),
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-shortest",
        str(mp4_path),
    ]
    subprocess.run(command, check=True)


def save_audio_and_video(audio: torch.Tensor, sample_rate: int, video_path: Path, wav_path: Path, mp4_path: Path) -> None:
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    audio = audio.detach().float().cpu()
    torchaudio.save(str(wav_path), audio, sample_rate)
    run_ffmpeg_mux(video_path, wav_path, mp4_path)


def feature_cache_path(cache_dir: Path, entry: VideoEntry, duration: float) -> Path:
    return cache_dir / f"{entry.sample_id}_{duration_tag(duration)}.pt"


def load_controlfoley_modules() -> dict[str, Any]:
    setup_controlfoley_imports()
    from controlfoley import feature_extractor as feature_extractor_module
    from controlfoley.audio_model import create_audio_generation_model
    from controlfoley.feature_extractor import FeaturesUtils
    from controlfoley.inference_utils import all_model_cfg, load_video
    from lib.flow_matching import FlowMatching

    return {
        "feature_extractor_module": feature_extractor_module,
        "create_audio_generation_model": create_audio_generation_model,
        "FeaturesUtils": FeaturesUtils,
        "all_model_cfg": all_model_cfg,
        "load_video": load_video,
        "FlowMatching": FlowMatching,
    }


def disable_unused_timbre_loader(feature_extractor_module: Any) -> None:
    """Avoid loading MusicGen for TC-V2A, where reference-audio timbre is unused."""

    class _NoTimbreMusicGen:
        @staticmethod
        def get_pretrained(*_: Any, **__: Any) -> None:
            return None

    feature_extractor_module.MusicGen = _NoTimbreMusicGen


def load_models(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    modules = load_controlfoley_modules()
    if args.disable_timbre_model:
        disable_unused_timbre_loader(modules["feature_extractor_module"])

    all_model_cfg = modules["all_model_cfg"]
    if args.variant not in all_model_cfg:
        raise ValueError(f"Unknown ControlFoley variant: {args.variant}")
    model_cfg = all_model_cfg[args.variant]

    model_path = CONTROLFOLEY_ROOT / model_cfg.model_path
    if not model_path.exists():
        raise FileNotFoundError(f"ControlFoley model weights not found: {model_path}")

    log.info("Loading ControlFoley network from %s", model_path)
    net = modules["create_audio_generation_model"](model_cfg.model_name).eval()
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    net.load_weights(state_dict)
    del state_dict
    net = net.to(device, dtype).eval()
    net.update_seq_lengths(345, 64, 32, 192)

    feature_utils_kwargs = {
        "tod_vae_ckpt": str(CONTROLFOLEY_ROOT / "model_weights" / "ext_weights" / "v1-44.pth"),
        "synchformer_ckpt": str(
            CONTROLFOLEY_ROOT / "model_weights" / "ext_weights" / "synchformer_state_dict.pth"
        ),
        "cav_mae_ckpt": str(CONTROLFOLEY_ROOT / "model_weights" / "ext_weights" / "cav_mae_st.pth"),
        "clap_ckpt": None,
        "mode": model_cfg.mode,
        "enable_conditions": True,
        "need_vae_encoder": False,
    }
    missing = [
        path
        for key, path in feature_utils_kwargs.items()
        if key.endswith("_ckpt") and path is not None and not Path(path).exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing ControlFoley support weights: {missing}")

    log.info("Loading ControlFoley feature/vocoder utilities")
    feature_utils = modules["FeaturesUtils"](**feature_utils_kwargs).to(device, dtype).eval()
    fm = modules["FlowMatching"](min_sigma=0, inference_mode="euler", num_steps=args.num_steps)

    return {
        "net": net,
        "feature_utils": feature_utils,
        "fm": fm,
        "load_video": modules["load_video"],
        "audio_sample_rate": 44100,
    }


@torch.inference_mode()
def compute_video_features(
    entry: VideoEntry,
    args: argparse.Namespace,
    feature_utils: Any,
    load_video: Any,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    video_info = load_video(entry.video_path, args.duration, load_all_frames=False)
    duration = float(video_info.total_duration)

    clip_frames = video_info.clip_embeddings.unsqueeze(0)
    visual_frames = video_info.visual_features.unsqueeze(0)
    sync_frames = video_info.sync_embeddings.unsqueeze(0)

    if args.mask_away_clip:
        clip_features = None
    else:
        clip_features = feature_utils.encode_video_with_clip(
            clip_frames.to(device, dtype, non_blocking=True), batch_size=args.clip_batch_size
        )

    visual_video = visual_frames.to(device, dtype, non_blocking=True)
    visual_parts = []
    for batch_idx in range(visual_video.size(0)):
        visual_parts.append(feature_utils.encode_video_with_cav_mae(visual_video[batch_idx]))
    visual_features = torch.cat(visual_parts, dim=0)
    visual_features = torch.mean(visual_features, dim=2)

    sync_features = feature_utils.encode_video_with_sync(
        sync_frames.to(device, dtype, non_blocking=True), batch_size=args.sync_batch_size
    )

    payload = {
        "sample_id": entry.sample_id,
        "video_id": entry.video_id,
        "start_sec": entry.start_sec,
        "category": entry.category,
        "duration": duration,
        "clip_features": clip_features.detach().cpu() if clip_features is not None else None,
        "visual_features": visual_features.detach().cpu(),
        "sync_features": sync_features.detach().cpu(),
    }
    return payload


def get_or_create_video_features(
    entry: VideoEntry,
    args: argparse.Namespace,
    feature_utils: Any,
    load_video: Any,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    if args.feature_cache_dir is None:
        return compute_video_features(entry, args, feature_utils, load_video, device, dtype)

    cache_path = feature_cache_path(args.feature_cache_dir, entry, args.duration)
    if cache_path.exists() and not args.force_feature_recompute:
        return torch.load(cache_path, map_location="cpu", weights_only=True)

    payload = compute_video_features(entry, args, feature_utils, load_video, device, dtype)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(cache_path.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, tmp_path)
    tmp_path.replace(cache_path)
    return payload


@torch.inference_mode()
def encode_texts(feature_utils: Any, prompts: list[str]) -> torch.Tensor:
    return feature_utils.encode_text(prompts)


def expand_feature(feature: torch.Tensor | None, net: Any, bs: int, name: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if feature is not None:
        feature = feature.to(device, dtype, non_blocking=True)
        if name == "clip":
            feature = fit_sequence_length(feature, net.clip_seq_len)
        return feature.expand(bs, -1, -1).contiguous()
    if name == "clip":
        return net.get_empty_clip_sequence(bs)
    raise ValueError(f"Feature {name} cannot be empty")


def fit_sequence_length(feature: torch.Tensor, target_length: int) -> torch.Tensor:
    """Trim or repeat-pad temporal features to the model's current sequence length."""
    current_length = feature.shape[1]
    if current_length == target_length:
        return feature
    if current_length > target_length:
        return feature[:, :target_length].contiguous()
    if current_length <= 0:
        raise ValueError(f"Cannot pad empty feature sequence to length {target_length}")

    pad_count = target_length - current_length
    pad = feature[:, -1:, :].expand(-1, pad_count, -1)
    return torch.cat([feature, pad], dim=1).contiguous()


@torch.inference_mode()
def generate_with_cached_video(
    *,
    net: Any,
    feature_utils: Any,
    fm: Any,
    video_features: dict[str, Any],
    prompts: list[str],
    negative_prompts: list[str] | None,
    sample_id: str,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> list[torch.Tensor]:
    bs = len(prompts)
    clip_features = expand_feature(video_features["clip_features"], net, bs, "clip", device, dtype)
    visual_features = video_features["visual_features"].to(device, dtype, non_blocking=True)
    visual_features = fit_sequence_length(visual_features, net.visual_seq_len)
    visual_features = visual_features.expand(bs, -1, -1).contiguous()
    sync_features = video_features["sync_features"].to(device, dtype, non_blocking=True)
    sync_features = fit_sequence_length(sync_features, net.sync_seq_len)
    sync_features = sync_features.expand(bs, -1, -1).contiguous()

    text_features = encode_texts(feature_utils, prompts)
    audio_features = net.get_empty_audio_sequence(bs)
    timbre_features = net.get_empty_timbre_sequence(bs)
    conditions = net.preprocess_conditions(
        clip_features, visual_features, sync_features, text_features, audio_features, timbre_features
    )

    if negative_prompts is not None:
        negative_text_features = encode_texts(feature_utils, negative_prompts)
    else:
        negative_text_features = None
    empty_conditions = net.get_empty_conditions(bs, negative_text_features=negative_text_features)

    if args.seed_mode == "shared":
        rng = torch.Generator(device=device)
        rng.manual_seed(args.seed)
        x0_single = torch.randn(1, net.latent_seq_len, net.latent_dim, device=device, dtype=dtype, generator=rng)
        x0 = x0_single.expand(bs, -1, -1).contiguous()
    else:
        latents = []
        for prompt in prompts:
            rng = torch.Generator(device=device)
            rng.manual_seed(stable_seed(args.seed, sample_id, prompt))
            latents.append(torch.randn(1, net.latent_seq_len, net.latent_dim, device=device, dtype=dtype, generator=rng))
        x0 = torch.cat(latents, dim=0)

    ode = lambda t, x: net.ode_wrapper(t, x, conditions, empty_conditions, args.cfg_strength)
    x1 = fm.to_data(ode, x0)
    x1 = net.unnormalize(x1)
    spec = feature_utils.decode(x1)
    audios = feature_utils.vocode(spec).float().cpu()
    return [audios[idx] for idx in range(bs)]


def build_wrong_negative_prompts(entry: VideoEntry, targets: list[str], mode: str) -> list[str] | None:
    if mode == "none":
        return None
    if mode == "source":
        return [entry.category for _ in targets]
    raise ValueError(f"Unknown negative prompt mode: {mode}")


def process_entry(
    entry: VideoEntry,
    args: argparse.Namespace,
    exp_dir: Path,
    model_state: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> str:
    if not entry.video_path.exists():
        log.warning("Skipping missing video: %s", entry.video_path)
        return "missing_video"

    correct_wav, correct_mp4 = correct_paths(exp_dir, entry)
    correct_done = correct_wav.exists() and correct_mp4.exists()
    existing_wrong = existing_wrong_results(exp_dir, entry)
    pending_wrong = [
        category for category in get_wrong_categories(entry.category) if category not in existing_wrong
    ]

    if correct_done and not pending_wrong:
        write_metadata(exp_dir, entry, args)
        return "skipped_complete"

    sample_dir = exp_dir / entry.sample_id
    (sample_dir / "correct").mkdir(parents=True, exist_ok=True)
    (sample_dir / "wrong").mkdir(parents=True, exist_ok=True)

    video_features = get_or_create_video_features(
        entry,
        args,
        model_state["feature_utils"],
        model_state["load_video"],
        device,
        dtype,
    )

    if not correct_done:
        audios = generate_with_cached_video(
            net=model_state["net"],
            feature_utils=model_state["feature_utils"],
            fm=model_state["fm"],
            video_features=video_features,
            prompts=[entry.category],
            negative_prompts=None,
            sample_id=entry.sample_id,
            args=args,
            device=device,
            dtype=dtype,
        )
        save_audio_and_video(
            audios[0], model_state["audio_sample_rate"], entry.video_path, correct_wav, correct_mp4
        )
        write_metadata(exp_dir, entry, args)

    for start in range(0, len(pending_wrong), args.prompt_batch_size):
        targets = pending_wrong[start : start + args.prompt_batch_size]
        audios = generate_with_cached_video(
            net=model_state["net"],
            feature_utils=model_state["feature_utils"],
            fm=model_state["fm"],
            video_features=video_features,
            prompts=targets,
            negative_prompts=build_wrong_negative_prompts(entry, targets, args.negative_prompt_mode),
            sample_id=entry.sample_id,
            args=args,
            device=device,
            dtype=dtype,
        )
        for target, audio in zip(targets, audios):
            wrong_wav, wrong_mp4 = wrong_paths(exp_dir, entry, target)
            save_audio_and_video(
                audio, model_state["audio_sample_rate"], entry.video_path, wrong_wav, wrong_mp4
            )
        write_metadata(exp_dir, entry, args)

    return "generated"


def write_run_files(exp_dir: Path, args: argparse.Namespace, entries: list[VideoEntry]) -> None:
    exp_dir.mkdir(parents=True, exist_ok=True)
    config_path = exp_dir / "config.json"
    if not config_path.exists():
        config = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "backend": "controlfoley",
            "task": "TC-V2A on VGGSound-Sparse",
            "num_entries_total_for_this_invocation": len(entries),
            "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "categories": CATEGORIES,
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
            f"- backend: ControlFoley TC-V2A\n"
            f"- subset: {args.subset}\n"
            f"- duration: {args.duration}s\n"
            f"- num_steps: {args.num_steps}\n"
            f"- cfg_strength: {args.cfg_strength}\n"
            f"- negative_prompt_mode: {args.negative_prompt_mode}\n"
            f"- feature_cache_dir: {args.feature_cache_dir}\n"
            f"- git_status_short at launch:\n\n"
            f"```text\n{status}\n```\n"
        )
        run_md_path.write_text(run_md, encoding="utf-8")


def select_shard(entries: list[VideoEntry], shard_index: int, num_shards: int) -> list[VideoEntry]:
    if num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    if not 0 <= shard_index < num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < num-shards")
    return entries[shard_index::num_shards]


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ControlFoley TC-V2A on VGGSound-Sparse")
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE)
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--clean-csv-path", type=Path, default=DEFAULT_CLEAN_CSV_PATH)
    parser.add_argument("--subset", choices=["all", "clean"], default="clean")
    parser.add_argument("--variant", default="large_44k")
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--num-steps", type=int, default=25)
    parser.add_argument("--cfg-strength", type=float, default=4.5)
    parser.add_argument("--prompt-batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed-mode", choices=["shared", "per_prompt"], default="shared")
    parser.add_argument("--negative-prompt-mode", choices=["none", "source"], default="none")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float16")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--feature-cache-dir", type=Path, default=DEFAULT_FEATURE_CACHE_DIR)
    parser.add_argument("--force-feature-recompute", action="store_true")
    parser.add_argument("--clip-batch-size", type=int, default=40)
    parser.add_argument("--sync-batch-size", type=int, default=40)
    parser.add_argument("--mask-away-clip", action="store_true")
    parser.add_argument("--disable-timbre-model", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = parse_args(argv)
    if args.prompt_batch_size <= 0:
        raise ValueError("--prompt-batch-size must be positive")

    exp_dir = args.output_base / args.exp_name
    entries = load_vggsound_sparse_entries(
        args.csv_path,
        subset=args.subset,
        clean_csv_path=args.clean_csv_path,
    )
    if args.start_index:
        entries = entries[args.start_index :]
    if args.limit is not None:
        entries = entries[: args.limit]
    entries = select_shard(entries, args.shard_index, args.num_shards)

    completed = sum(1 for entry in entries if is_complete(exp_dir, entry))
    log.info(
        "Experiment %s shard %d/%d: %d entries, %d already complete, %d pending",
        args.exp_name,
        args.shard_index,
        args.num_shards,
        len(entries),
        completed,
        len(entries) - completed,
    )

    if args.dry_run:
        for entry in entries[:10]:
            log.info("DRY %s %s", entry.sample_id, entry.category)
        return

    write_run_files(exp_dir, args, entries)

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
        log.warning("CUDA is unavailable; running on CPU")
    dtype = torch.float16 if args.dtype == "float16" else torch.float32

    model_state = load_models(args, device, dtype)
    status_counts: dict[str, int] = {}
    start_time = time.time()

    for entry in tqdm(entries, desc=f"shard {args.shard_index}"):
        try:
            status = process_entry(entry, args, exp_dir, model_state, device, dtype)
        except Exception:
            status = "error"
            error_dir = exp_dir / entry.sample_id
            error_dir.mkdir(parents=True, exist_ok=True)
            (error_dir / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
            log.exception("Failed processing %s", entry.sample_id)
        status_counts[status] = status_counts.get(status, 0) + 1

    elapsed = time.time() - start_time
    summary_path = exp_dir / f"shard_{args.shard_index:02d}_summary.json"
    summary = {
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "entries": len(entries),
        "elapsed_sec": elapsed,
        "status_counts": status_counts,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Shard complete: %s", summary)


if __name__ == "__main__":
    main()
