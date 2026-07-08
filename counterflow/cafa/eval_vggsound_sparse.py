"""Run CAFA on VGGSound-Sparse in the CounterFlow evaluation layout.

This wrapper keeps the CAFA repository under baselines/CAFA read-only while
adding the prompt-bank and clean-subset behavior used by CounterFlow runs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import random
import re
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import soundfile as sf
import torch
import torch.multiprocessing as mp


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CAFA_ROOT = PROJECT_ROOT / "baselines" / "CAFA"
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
DEFAULT_PRECOMPUTED_DIR = CAFA_ROOT / "VGGSound-Sparse" / "precomputed"

MODEL_CONFIG_PATH = CAFA_ROOT / "ckpts" / "CAFA_avclip_config.json"
MODEL_PATH = CAFA_ROOT / "ckpts" / "CAFA_avclip.safetensors"
MAX_GENERATION_DURATION_SEC = 10.0

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
    split: str = "test"

    @property
    def sample_id(self) -> str:
        return f"{self.video_id}_{self.start_sec:06d}"

    @property
    def video_path(self) -> Path:
        return VIDEO_ROOT / f"{self.sample_id}.mp4"


@dataclass(frozen=True)
class GenerationTask:
    entry: VideoEntry
    prompt: str
    is_correct: bool
    output_dir: Path
    output_stem: str
    latent_path: Path


def setup_cafa_imports() -> None:
    if not CAFA_ROOT.exists():
        raise FileNotFoundError(f"CAFA repository not found: {CAFA_ROOT}")
    for path in (CAFA_ROOT, CAFA_ROOT / "synchformer"):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def safe_prompt_slug(prompt: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(prompt).strip()).strip("_")
    return slug or "prompt"


def prompt_cache_path(precomputed_dir: Path, prompt: str) -> Path:
    return precomputed_dir / "conditioning" / f"t5_{safe_prompt_slug(prompt)}.pt"


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
            entries.append(VideoEntry(video_id=row[0], start_sec=start_sec, category=row[2], split=row[3]))

    if subset == "clean":
        clean_keys = load_clean_video_keys(clean_csv_path)
        before = len(entries)
        entries = [entry for entry in entries if (entry.video_id, entry.start_sec) in clean_keys]
        log.info("Applied clean subset filter: kept %d / %d test videos", len(entries), before)

    missing_categories = sorted({entry.category for entry in entries} - set(CATEGORIES))
    if missing_categories:
        raise ValueError(f"Unexpected VGGSound-Sparse categories: {missing_categories}")

    return entries


def load_target_prompt_map(prompt_bank_path: Path | None, target_case: str) -> dict[str, dict[str, Any]] | None:
    if prompt_bank_path is None:
        return None
    if not prompt_bank_path.exists():
        raise FileNotFoundError(f"Target prompt bank not found: {prompt_bank_path}")

    with prompt_bank_path.open(encoding="utf-8") as f:
        data = json.load(f)
    items = data.get("items") if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise ValueError(f"Target prompt bank must contain an items list: {prompt_bank_path}")

    prompt_map: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        source_category = item.get("source_category") or item.get("source_caption_original")
        if not source_category:
            raise ValueError(f"Prompt-bank item missing source_category: {item}")

        source_prompt = (
            item.get("source_caption")
            or item.get("source_caption_active")
            or item.get("source_prompt")
            or source_category
        )
        case_rows = item.get(target_case)
        if not isinstance(case_rows, list):
            raise ValueError(
                f"Prompt-bank item for {source_category!r} missing list case {target_case!r}"
            )

        target_prompts: list[str] = []
        for row in case_rows:
            if isinstance(row, str):
                target_prompt = row
            elif isinstance(row, dict):
                target_prompt = row.get("target_prompt") or row.get("target_caption")
            else:
                target_prompt = None
            if target_prompt:
                target_prompts.append(str(target_prompt).strip())

        target_prompts = list(dict.fromkeys(prompt for prompt in target_prompts if prompt))
        if not target_prompts:
            raise ValueError(f"No target prompts for {source_category!r} case {target_case!r}")

        prompt_map[str(source_category)] = {
            "source_prompt": str(source_prompt).strip(),
            "target_prompts": target_prompts,
        }

    if not prompt_map:
        raise ValueError(f"No prompt-bank items loaded from: {prompt_bank_path}")
    return prompt_map


def get_prompt_plan(entry: VideoEntry, prompt_map: dict[str, dict[str, Any]] | None) -> tuple[str, list[str]]:
    if not prompt_map:
        return entry.category, [category for category in CATEGORIES if category != entry.category]
    if entry.category not in prompt_map:
        raise KeyError(f"No prompt-bank entry for source category: {entry.category}")
    spec = prompt_map[entry.category]
    return spec["source_prompt"], list(spec["target_prompts"])


def all_prompts_for_entries(entries: list[VideoEntry], prompt_map: dict[str, dict[str, Any]] | None) -> list[str]:
    prompts: list[str] = []
    for entry in entries:
        source_prompt, target_prompts = get_prompt_plan(entry, prompt_map)
        prompts.append(source_prompt)
        prompts.extend(target_prompts)
    return list(dict.fromkeys(prompts))


def ensure_prompt_conditioning(precomputed_dir: Path, prompts: list[str], device: str) -> None:
    cond_dir = precomputed_dir / "conditioning"
    cond_dir.mkdir(parents=True, exist_ok=True)
    missing_prompts = [prompt for prompt in prompts if not prompt_cache_path(precomputed_dir, prompt).exists()]
    number_cond_path = cond_dir / "number_cond.pt"

    if not missing_prompts and number_cond_path.exists():
        log.info("All CAFA conditioning tensors already exist for %d prompts", len(prompts))
        return

    setup_cafa_imports()
    from stable_audio_tools.models.pretrained import get_pretrained_model_local

    log.info("Computing CAFA conditioning for %d missing prompts", len(missing_prompts))
    gen_model, _ = get_pretrained_model_local(str(MODEL_CONFIG_PATH), str(MODEL_PATH))
    conditioner = gen_model.conditioner.to(device).eval()
    del gen_model
    torch.cuda.empty_cache()

    t5_conditioner = conditioner.conditioners["prompt"]
    number_conditioners = {
        key: value for key, value in conditioner.conditioners.items() if key not in ("prompt", "avclip_signal")
    }

    with torch.inference_mode():
        for prompt in missing_prompts:
            save_path = prompt_cache_path(precomputed_dir, prompt)
            if save_path.exists():
                continue
            t5_emb, t5_mask = t5_conditioner([prompt], device)
            torch.save({"embeddings": t5_emb.cpu(), "mask": t5_mask.cpu()}, save_path)
            log.info("Saved CAFA T5 conditioning: %s", save_path.name)

        if not number_cond_path.exists():
            number_cond: dict[str, Any] = {}
            for key, cond_module in number_conditioners.items():
                if key == "seconds_start":
                    val = cond_module([0], device)
                elif key == "seconds_total":
                    val = cond_module([MAX_GENERATION_DURATION_SEC], device)
                else:
                    continue

                if isinstance(val, (list, tuple)):
                    number_cond[key] = [v.cpu() if isinstance(v, torch.Tensor) else v for v in val]
                elif isinstance(val, torch.Tensor):
                    number_cond[key] = val.cpu()
                else:
                    number_cond[key] = val
            torch.save(number_cond, number_cond_path)
            log.info("Saved CAFA number conditioning: %s", number_cond_path.name)

    del conditioner
    torch.cuda.empty_cache()


def load_conditioning_cache(
    precomputed_dir: Path,
    prompts: list[str],
    device: str,
) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], dict[str, Any]]:
    t5_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for prompt in prompts:
        path = prompt_cache_path(precomputed_dir, prompt)
        if not path.exists():
            raise FileNotFoundError(f"Missing CAFA T5 conditioning for {prompt!r}: {path}")
        data = torch.load(path, map_location="cpu", weights_only=True)
        t5_cache[prompt] = (data["embeddings"], data["mask"])

    number_cond_path = precomputed_dir / "conditioning" / "number_cond.pt"
    if not number_cond_path.exists():
        raise FileNotFoundError(f"Missing CAFA number conditioning: {number_cond_path}")
    number_cond = torch.load(number_cond_path, map_location="cpu", weights_only=True)
    return t5_cache, number_cond


def build_conditioning_tensors(
    t5_cache: dict[str, tuple[torch.Tensor, torch.Tensor]],
    number_cond: dict[str, Any],
    avclip_embedding: torch.Tensor,
    prompt: str,
    device: str,
) -> dict[str, Any]:
    t5_emb, t5_mask = t5_cache[prompt]
    cond_tensors: dict[str, Any] = {
        "prompt": (t5_emb.to(device), t5_mask.to(device)),
    }
    for key, val in number_cond.items():
        if isinstance(val, list):
            cond_tensors[key] = [v.to(device) if isinstance(v, torch.Tensor) else v for v in val]
        elif isinstance(val, torch.Tensor):
            cond_tensors[key] = val.to(device)
        else:
            cond_tensors[key] = val
    cond_tensors["avclip_signal"] = [avclip_embedding.to(device)]
    return cond_tensors


def generate_latent_from_precomputed(model: Any, gen_config: dict[str, Any], conditioning_tensors: dict[str, Any], args: argparse.Namespace, device: str) -> tuple[torch.Tensor, int]:
    from stable_audio_tools.inference.sampling import sample_k, sample_rf

    sample_rate = gen_config["sample_rate"]
    sample_size = gen_config.get("sample_size", int(sample_rate * MAX_GENERATION_DURATION_SEC))
    if model.pretransform is not None:
        sample_size = sample_size // model.pretransform.downsampling_ratio

    torch.manual_seed(args.seed)
    noise = torch.randn([1, model.io_channels, sample_size], device=device)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cudnn.benchmark = False

    conditioning_inputs = model.get_conditioning_inputs(conditioning_tensors)
    model_dtype = next(model.model.parameters()).dtype
    noise = noise.type(model_dtype)
    conditioning_inputs = {
        key: value.type(model_dtype) if value is not None and isinstance(value, torch.Tensor) else value
        for key, value in conditioning_inputs.items()
    }

    if model.diffusion_objective == "v":
        sampled = sample_k(
            model.model,
            noise,
            None,
            None,
            args.steps,
            sampler_type=args.sampler_type,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            **conditioning_inputs,
            cfg_scale=args.cfg,
            asym_cfg=args.asym_cfg,
            batch_cfg=True,
            rescale_cfg=True,
            device=device,
        )
    elif model.diffusion_objective == "rectified_flow":
        sampled = sample_rf(
            model.model,
            noise,
            init_data=None,
            steps=args.steps,
            sigma_max=args.sigma_max,
            **conditioning_inputs,
            cfg_scale=args.cfg,
            batch_cfg=True,
            rescale_cfg=True,
            device=device,
        )
    else:
        raise ValueError(f"Unsupported CAFA diffusion objective: {model.diffusion_objective}")

    del noise, conditioning_inputs
    torch.cuda.empty_cache()
    return sampled.cpu(), sample_rate


def decode_latent_to_audio(pretransform: Any, latent: torch.Tensor, device: str) -> torch.Tensor:
    latent = latent.to(device)
    latent = latent.to(next(pretransform.parameters()).dtype)
    audio = pretransform.decode(latent)
    return audio.cpu()


def save_audio_and_video(
    output_tensor: torch.Tensor,
    sample_rate: int,
    target_duration: float,
    save_dir: Path,
    video_path: Path,
    output_stem: str,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    output_tensor = output_tensor.cpu()
    max_val = torch.amax(torch.abs(output_tensor), dim=(1, 2), keepdim=True) + 1e-8
    output_norm = output_tensor / max_val
    output_norm = output_norm.clamp(-1, 1).mul(32767).to(torch.int16)

    max_length_samples = int(sample_rate * target_duration)
    if output_norm.shape[-1] > max_length_samples:
        output_norm = output_norm[..., :max_length_samples]

    wav_path = save_dir / f"{output_stem}.wav"
    mp4_path = save_dir / f"{output_stem}.mp4"
    sf.write(str(wav_path), output_norm[0].numpy().T, sample_rate, subtype="PCM_16")
    run_ffmpeg_mux(video_path, wav_path, mp4_path)


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
        "-map",
        "0:v",
        "-map",
        "1:a",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        str(mp4_path),
    ]
    subprocess.run(command, check=True)


def correct_paths(exp_dir: Path, entry: VideoEntry) -> tuple[Path, Path]:
    correct_dir = exp_dir / entry.sample_id / "correct"
    return correct_dir / f"{entry.sample_id}_correct.wav", correct_dir / f"{entry.sample_id}_correct.mp4"


def wrong_paths(exp_dir: Path, entry: VideoEntry, target_prompt: str) -> tuple[Path, Path]:
    wrong_dir = exp_dir / entry.sample_id / "wrong"
    slug = safe_prompt_slug(target_prompt)
    return wrong_dir / f"{entry.sample_id}_{slug}.wav", wrong_dir / f"{entry.sample_id}_{slug}.mp4"


def is_entry_complete(exp_dir: Path, entry: VideoEntry, prompt_map: dict[str, dict[str, Any]] | None) -> bool:
    correct_wav, correct_mp4 = correct_paths(exp_dir, entry)
    if not (correct_wav.exists() and correct_mp4.exists()):
        return False
    _, target_prompts = get_prompt_plan(entry, prompt_map)
    return all(
        wav.exists() and mp4.exists()
        for wav, mp4 in (wrong_paths(exp_dir, entry, target_prompt) for target_prompt in target_prompts)
    )


def build_tasks(
    entries: list[VideoEntry],
    exp_dir: Path,
    prompt_map: dict[str, dict[str, Any]] | None,
) -> list[GenerationTask]:
    tasks: list[GenerationTask] = []
    for entry in entries:
        source_prompt, target_prompts = get_prompt_plan(entry, prompt_map)

        correct_wav, correct_mp4 = correct_paths(exp_dir, entry)
        correct_dir = correct_wav.parent
        if not (correct_wav.exists() and correct_mp4.exists()):
            stem = f"{entry.sample_id}_correct"
            tasks.append(
                GenerationTask(
                    entry=entry,
                    prompt=source_prompt,
                    is_correct=True,
                    output_dir=correct_dir,
                    output_stem=stem,
                    latent_path=correct_dir / f".{stem}_latent.pt",
                )
            )

        for target_prompt in target_prompts:
            wrong_wav, wrong_mp4 = wrong_paths(exp_dir, entry, target_prompt)
            if wrong_wav.exists() and wrong_mp4.exists():
                continue
            stem = wrong_wav.stem
            tasks.append(
                GenerationTask(
                    entry=entry,
                    prompt=target_prompt,
                    is_correct=False,
                    output_dir=wrong_wav.parent,
                    output_stem=stem,
                    latent_path=wrong_wav.parent / f".{stem}_latent.pt",
                )
            )
    return tasks


def worker_fn(gpu_id: int, tasks: list[GenerationTask], args: argparse.Namespace, prompts: list[str]) -> None:
    setup_cafa_imports()
    from stable_audio_tools.models.pretrained import get_pretrained_model_local

    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"
    precomputed_dir = Path(args.precomputed_dir)
    log.info("[GPU %d] Starting CAFA worker with %d tasks", gpu_id, len(tasks))

    t5_cache, number_cond = load_conditioning_cache(precomputed_dir, prompts, device)

    run_phase_a = args.phase in (None, "A")
    run_phase_b = args.phase in (None, "B")
    sample_rate = None

    if run_phase_a:
        log.info("[GPU %d] Phase A: loading CAFA diffusion model", gpu_id)
        gen_model, gen_config = get_pretrained_model_local(str(MODEL_CONFIG_PATH), str(MODEL_PATH))
        gen_model.eval()
        if args.half:
            gen_model.model = gen_model.model.half().to(device)
        else:
            gen_model.model = gen_model.model.to(device)
        gen_model.conditioner = gen_model.conditioner.cpu()
        if gen_model.pretransform is not None:
            gen_model.pretransform = gen_model.pretransform.cpu()
        torch.cuda.empty_cache()
        sample_rate = gen_config["sample_rate"]

        for idx, task in enumerate(tasks):
            wav_path = task.output_dir / f"{task.output_stem}.wav"
            if wav_path.exists() or task.latent_path.exists():
                continue

            embed_path = precomputed_dir / "avclip_embeddings" / f"{task.entry.sample_id}.npy"
            if not embed_path.exists():
                log.warning("[GPU %d] Missing CAFA AVCLIP embedding: %s", gpu_id, embed_path)
                continue

            try:
                avclip_embedding = torch.from_numpy(np.load(str(embed_path)))
                conditioning_tensors = build_conditioning_tensors(
                    t5_cache, number_cond, avclip_embedding, task.prompt, device
                )
                with torch.inference_mode():
                    latent, sr = generate_latent_from_precomputed(
                        gen_model, gen_config, conditioning_tensors, args, device
                    )
                sample_rate = sr
                task.output_dir.mkdir(parents=True, exist_ok=True)
                torch.save(latent, task.latent_path)
                del latent, conditioning_tensors, avclip_embedding
                torch.cuda.empty_cache()
            except Exception:
                error_path = task.output_dir / f"{task.output_stem}.phase_a.error.log"
                task.output_dir.mkdir(parents=True, exist_ok=True)
                error_path.write_text(traceback.format_exc(), encoding="utf-8")
                log.exception("[GPU %d] Phase A failed for %s", gpu_id, task.output_stem)
                torch.cuda.empty_cache()

            if (idx + 1) % 10 == 0 or idx == len(tasks) - 1:
                log.info("[GPU %d] Phase A progress: %d/%d", gpu_id, idx + 1, len(tasks))

        del gen_model
        torch.cuda.empty_cache()
    else:
        log.info("[GPU %d] Skipping Phase A", gpu_id)

    del t5_cache, number_cond

    if run_phase_b:
        log.info("[GPU %d] Phase B: loading CAFA pretransform", gpu_id)
        gen_model_b, gen_config_b = get_pretrained_model_local(str(MODEL_CONFIG_PATH), str(MODEL_PATH))
        if sample_rate is None:
            sample_rate = gen_config_b["sample_rate"]
        pretransform = gen_model_b.pretransform.to(device).eval()
        del gen_model_b
        torch.cuda.empty_cache()

        for idx, task in enumerate(tasks):
            wav_path = task.output_dir / f"{task.output_stem}.wav"
            mp4_path = task.output_dir / f"{task.output_stem}.mp4"
            if wav_path.exists() and mp4_path.exists():
                continue
            if not task.latent_path.exists():
                continue

            try:
                latent = torch.load(task.latent_path, map_location="cpu", weights_only=True)
                with torch.inference_mode():
                    audio = decode_latent_to_audio(pretransform, latent, device)
                save_audio_and_video(
                    audio,
                    sample_rate,
                    MAX_GENERATION_DURATION_SEC,
                    task.output_dir,
                    task.entry.video_path,
                    task.output_stem,
                )
                if args.remove_latents:
                    task.latent_path.unlink(missing_ok=True)
                del latent, audio
                torch.cuda.empty_cache()
            except Exception:
                error_path = task.output_dir / f"{task.output_stem}.phase_b.error.log"
                error_path.write_text(traceback.format_exc(), encoding="utf-8")
                log.exception("[GPU %d] Phase B failed for %s", gpu_id, task.output_stem)
                torch.cuda.empty_cache()

            if (idx + 1) % 10 == 0 or idx == len(tasks) - 1:
                log.info("[GPU %d] Phase B progress: %d/%d", gpu_id, idx + 1, len(tasks))

        del pretransform
        torch.cuda.empty_cache()
    else:
        log.info("[GPU %d] Skipping Phase B", gpu_id)

    log.info("[GPU %d] Worker complete", gpu_id)


def write_metadata_for_entries(
    exp_dir: Path,
    entries: list[VideoEntry],
    args: argparse.Namespace,
    prompt_map: dict[str, dict[str, Any]] | None,
) -> None:
    common = {
        "backend": "cafa",
        "variant": "CAFA_avclip",
        "task": "text-video-to-audio",
        "seed": args.seed,
        "duration": MAX_GENERATION_DURATION_SEC,
        "steps": args.steps,
        "cfg": args.cfg,
        "sigma_min": args.sigma_min,
        "sigma_max": args.sigma_max,
        "sampler_type": args.sampler_type,
        "asym_cfg": args.asym_cfg,
        "target_prompt_bank": str(args.target_prompt_bank) if args.target_prompt_bank else None,
        "target_case": args.target_case if prompt_map else None,
    }

    for entry in entries:
        source_prompt, target_prompts = get_prompt_plan(entry, prompt_map)
        correct_wav, correct_mp4 = correct_paths(exp_dir, entry)
        correct_dir = correct_wav.parent
        correct_dir.mkdir(parents=True, exist_ok=True)
        correct_meta = {
            **common,
            "video_id": entry.video_id,
            "start_sec": entry.start_sec,
            "video_path": str(entry.video_path),
            "correct_category": entry.category,
            "source_prompt": source_prompt,
            "prompt": source_prompt,
            "generation_type": "video_text_consistent",
        }
        if correct_wav.exists() and correct_mp4.exists():
            (correct_dir / "metadata.json").write_text(
                json.dumps(correct_meta, indent=2, ensure_ascii=False), encoding="utf-8"
            )

        wrong_dir = exp_dir / entry.sample_id / "wrong"
        wrong_dir.mkdir(parents=True, exist_ok=True)
        results = []
        for target_prompt in target_prompts:
            wrong_wav, wrong_mp4 = wrong_paths(exp_dir, entry, target_prompt)
            if wrong_wav.exists() and wrong_mp4.exists():
                results.append(
                    {
                        "source_prompt": source_prompt,
                        "target_prompt": target_prompt,
                        "target_case": args.target_case if prompt_map else None,
                        "audio_file": wrong_wav.name,
                        "video_file": wrong_mp4.name,
                    }
                )

        wrong_meta = {
            **common,
            "video_id": entry.video_id,
            "start_sec": entry.start_sec,
            "video_path": str(entry.video_path),
            "correct_category": entry.category,
            "source_prompt": source_prompt,
            "generation_type": "text_controlled_video_to_audio",
            "results": results,
        }
        if results:
            (wrong_dir / "metadata.json").write_text(
                json.dumps(wrong_meta, indent=2, ensure_ascii=False), encoding="utf-8"
            )


def write_run_files(
    exp_dir: Path,
    args: argparse.Namespace,
    entries: list[VideoEntry],
    prompt_map: dict[str, dict[str, Any]] | None,
    tasks: list[GenerationTask],
) -> None:
    exp_dir.mkdir(parents=True, exist_ok=True)
    config_path = exp_dir / "experiment_config.json"
    config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "backend": "cafa",
        "exp_name": args.exp_name,
        "subset": args.subset,
        "num_entries": len(entries),
        "num_tasks_pending_at_launch": len(tasks),
        "num_prompt_bank_sources": len(prompt_map or {}),
        "target_case": args.target_case if prompt_map else None,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    (exp_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")

    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    ).stdout.strip()
    run_md = (
        f"# {args.exp_name}\n\n"
        f"- backend: CAFA\n"
        f"- subset: {args.subset}\n"
        f"- target_case: {args.target_case if prompt_map else 'all-category'}\n"
        f"- entries: {len(entries)}\n"
        f"- pending_tasks_at_launch: {len(tasks)}\n"
        f"- num_gpus: {args.num_gpus}\n"
        f"- steps: {args.steps}\n"
        f"- cfg: {args.cfg}\n"
        f"- FAD: not computed by this generation script\n\n"
        f"```text\n{status}\n```\n"
    )
    (exp_dir / "RUN.md").write_text(run_md, encoding="utf-8")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CAFA on VGGSound-Sparse")
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE)
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--clean-csv-path", type=Path, default=DEFAULT_CLEAN_CSV_PATH)
    parser.add_argument("--subset", choices=["all", "clean"], default="clean")
    parser.add_argument("--precomputed-dir", type=Path, default=DEFAULT_PRECOMPUTED_DIR)
    parser.add_argument("--video-root", type=Path, default=VIDEO_ROOT)
    parser.add_argument("--target-prompt-bank", type=Path, default=None)
    parser.add_argument("--target-case", default="semantic_replace_temporal_preserve")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sigma-min", type=float, default=0.5)
    parser.add_argument("--sigma-max", type=float, default=500.0)
    parser.add_argument("--sampler-type", default="dpmpp-3m-sde")
    parser.add_argument("--asym-cfg", type=float, default=0.5)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--phase", choices=["A", "B"], default=None)
    parser.add_argument("--pilot", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--conditioning-device", default="cuda:0")
    parser.add_argument("--skip-conditioning-ensure", action="store_true")
    parser.add_argument("--remove-latents", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = parse_args(argv)

    global VIDEO_ROOT
    VIDEO_ROOT = args.video_root

    setup_cafa_imports()
    from utils import seed_everything
    from synchformer.utils.utils import which_ffmpeg

    if which_ffmpeg() is None:
        raise RuntimeError("ffmpeg not found in PATH")
    seed_everything(args.seed)

    prompt_map = load_target_prompt_map(args.target_prompt_bank, args.target_case)
    entries = load_vggsound_sparse_entries(args.csv_path, subset=args.subset, clean_csv_path=args.clean_csv_path)
    if prompt_map:
        before = len(entries)
        entries = [entry for entry in entries if entry.category in prompt_map]
        log.info("Filtered to prompt-bank source categories: kept %d / %d videos", len(entries), before)

    if args.start_index:
        entries = entries[args.start_index :]
    if args.limit is not None:
        entries = entries[: args.limit]
    if args.pilot is not None:
        random.seed(args.seed)
        entries = random.sample(entries, min(args.pilot, len(entries)))
        log.info("Pilot mode: selected %d videos", len(entries))

    if not entries:
        raise ValueError("No entries selected for CAFA run")

    prompts = all_prompts_for_entries(entries, prompt_map)
    if not args.skip_conditioning_ensure:
        ensure_prompt_conditioning(args.precomputed_dir, prompts, args.conditioning_device)

    missing_embeddings = [
        entry.sample_id
        for entry in entries
        if not (args.precomputed_dir / "avclip_embeddings" / f"{entry.sample_id}.npy").exists()
    ]
    if missing_embeddings:
        raise FileNotFoundError(
            f"Missing CAFA AVCLIP embeddings for {len(missing_embeddings)} entries; "
            f"first missing: {missing_embeddings[:5]}"
        )

    exp_dir = args.output_base / args.exp_name
    tasks = build_tasks(entries, exp_dir, prompt_map)
    completed = sum(1 for entry in entries if is_entry_complete(exp_dir, entry, prompt_map))
    log.info(
        "Experiment %s: %d entries, %d already complete, %d pending tasks",
        args.exp_name,
        len(entries),
        completed,
        len(tasks),
    )

    write_run_files(exp_dir, args, entries, prompt_map, tasks)

    if args.dry_run:
        for entry in entries[:10]:
            source_prompt, target_prompts = get_prompt_plan(entry, prompt_map)
            log.info("DRY %s %s -> %s", entry.sample_id, source_prompt, target_prompts)
        return

    if tasks:
        available_gpus = torch.cuda.device_count()
        num_gpus = min(args.num_gpus, available_gpus)
        if num_gpus <= 0:
            raise RuntimeError("CUDA is required for CAFA generation")
        log.info("Using %d GPU(s) out of %d available", num_gpus, available_gpus)

        if num_gpus == 1:
            worker_fn(0, tasks, args, prompts)
        else:
            task_chunks: list[list[GenerationTask]] = [[] for _ in range(num_gpus)]
            for idx, task in enumerate(tasks):
                task_chunks[idx % num_gpus].append(task)

            mp.set_start_method("spawn", force=True)
            processes = []
            for gpu_id, chunk in enumerate(task_chunks):
                if not chunk:
                    continue
                process = mp.Process(target=worker_fn, args=(gpu_id, chunk, args, prompts))
                process.start()
                processes.append(process)

            failures = 0
            for process in processes:
                process.join()
                if process.exitcode != 0:
                    failures += 1
            if failures:
                raise RuntimeError(f"{failures} CAFA worker process(es) failed")

    write_metadata_for_entries(exp_dir, entries, args, prompt_map)

    remaining = [entry.sample_id for entry in entries if not is_entry_complete(exp_dir, entry, prompt_map)]
    summary = {
        "exp_name": args.exp_name,
        "entries": len(entries),
        "prompts": len(prompts),
        "pending_tasks_at_launch": len(tasks),
        "remaining_incomplete_entries": len(remaining),
        "remaining_sample_ids": remaining[:20],
        "completed_at": datetime.now().isoformat(timespec="seconds"),
    }
    (exp_dir / "generation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if remaining:
        raise RuntimeError(f"CAFA generation incomplete for {len(remaining)} entries")

    log.info("Experiment '%s' complete: %s", args.exp_name, exp_dir)


if __name__ == "__main__":
    main()
