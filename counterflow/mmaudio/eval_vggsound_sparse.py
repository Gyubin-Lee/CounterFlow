"""
Required environment:
    conda activate MMAudio

VGGSound-Sparse Evaluation Script
- For each test video (600 total), generate audio with all 12 categories:
  - Correct category: standard MMAudio (Video + true category)
  - Wrong categories (11): Prompt switch (source=correct, target=wrong)
- Resume support: skips already-completed tasks and continues remaining ones
- Multi-GPU support via torch.multiprocessing
- Pilot mode: random 30 videos
"""

import logging
import argparse
import csv
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
import torchaudio
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MMAUDIO_ROOT = PROJECT_ROOT / "external" / "MMAudio"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(MMAUDIO_ROOT) not in sys.path:
    sys.path.insert(0, str(MMAUDIO_ROOT))

from mmaudio.eval_utils import ModelConfig, all_model_cfg, load_video, setup_eval_logging
from mmaudio.model.networks import MMAudio, get_my_mmaudio
from mmaudio.model.utils.features_utils import FeaturesUtils

SAMPLER_STRICT_2PHASE = "strict_2phase"
SAMPLER_SMOOTH_MEAN_COUPLED = "smooth_mean_coupled"
SAMPLER_PHASE2_VIDEO_DECAY = "phase2_video_decay"

# 12 categories in VGGSound-Sparse
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

CSV_PATH = Path('VGGSound-Sparse/vggsound_sparse.csv')
CLEAN_CSV_PATH = Path('VGGSound-Sparse/vggsound_sparse_clean_fixed_offsets.csv')
VIDEO_ROOT = Path('/media/daftpunk5/dataset/vggsound/video')


def load_clean_video_keys(clean_csv_path: Path):
    """Load clean subset as (youtube_id, start_sec) keys from CSV.

    Clean CSV format: dataset_name, video_id ({yt_id}_{start_ms}_{end_ms}), ...
    We parse the video_id column to extract youtube_id and start_sec.
    """
    if not clean_csv_path.exists():
        raise FileNotFoundError(f"Clean subset CSV not found: {clean_csv_path}")

    keys = set()
    with open(clean_csv_path) as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            video_id_col = row[1].strip()
            if not video_id_col or video_id_col.lower() == 'video_id':
                continue
            # Parse: {youtube_id}_{start_ms}_{end_ms}
            parts = video_id_col.rsplit('_', 2)
            if len(parts) == 3:
                youtube_id = parts[0]
                start_sec = int(parts[1]) // 1000
                keys.add((youtube_id, start_sec))

    if not keys:
        raise ValueError(f"No valid video keys found in clean subset CSV: {clean_csv_path}")

    return keys


def load_test_data(subset='all', clean_csv_path=CLEAN_CSV_PATH):
    """Load VGGSound-Sparse test split and optionally filter to clean subset."""
    rows = []
    with open(CSV_PATH) as f:
        reader = csv.reader(f)
        for row in reader:
            video_id, start_sec, caption, split = row[0], int(row[1]), row[2], row[3]
            if split == 'test':
                rows.append({
                    'video_id': video_id,
                    'start_sec': start_sec,
                    'category': caption,
                    'filename': f'{video_id}_{start_sec:06d}.mp4',
                })

    if subset == 'clean':
        clean_keys = load_clean_video_keys(clean_csv_path)
        original_len = len(rows)
        rows = [row for row in rows if (row['video_id'], row['start_sec']) in clean_keys]
        print(
            f"Applied clean subset filter: kept {len(rows)} / {original_len} "
            f"test videos from {clean_csv_path}"
        )

    return rows


def get_video_id_str(video_entry):
    return f"{video_entry['video_id']}_{video_entry['start_sec']:06d}"


def get_wrong_categories(correct_category):
    return [c for c in CATEGORIES if c != correct_category]


def safe_prompt_slug(prompt: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(prompt).strip()).strip("_")
    return slug or "prompt"


def load_target_prompt_map(prompt_bank_path: str | None, target_case: str) -> dict | None:
    if not prompt_bank_path:
        return None

    path = Path(prompt_bank_path)
    if not path.exists():
        raise FileNotFoundError(f"Target prompt bank not found: {path}")

    with open(path) as f:
        data = json.load(f)

    items = data.get("items") if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise ValueError(f"Target prompt bank must contain an items list: {path}")

    prompt_map = {}
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

        target_prompts = []
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

        prompt_map[source_category] = {
            "source_prompt": str(source_prompt).strip(),
            "target_prompts": target_prompts,
        }

    if not prompt_map:
        raise ValueError(f"No prompt-bank items loaded from: {path}")
    return prompt_map


def get_prompt_plan(video_entry, args=None):
    correct_category = video_entry['category']
    target_prompt_map = getattr(args, 'target_prompt_map', None) if args is not None else None
    if not target_prompt_map:
        return correct_category, get_wrong_categories(correct_category)
    if correct_category not in target_prompt_map:
        raise KeyError(f"No prompt-bank entry for source category: {correct_category}")
    prompt_spec = target_prompt_map[correct_category]
    return prompt_spec["source_prompt"], prompt_spec["target_prompts"]


def needs_source_conditions(args) -> bool:
    return (
        args.neg_src
        or args.neg_src_both
        or args.sampler in {SAMPLER_SMOOTH_MEAN_COUPLED, SAMPLER_PHASE2_VIDEO_DECAY}
    )


def schedule_progress(step_index: int, num_steps: int, schedule: str) -> float:
    if num_steps <= 1:
        return 1.0
    progress = step_index / (num_steps - 1)
    progress = max(0.0, min(1.0, progress))
    if schedule == "linear":
        return progress
    if schedule == "cosine":
        return 0.5 - 0.5 * math.cos(math.pi * progress)
    if schedule == "smoothstep":
        return progress * progress * (3.0 - 2.0 * progress)
    raise ValueError(f"Unknown smooth guidance schedule: {schedule}")


def interpolate_schedule(start: float, end: float, progress: float) -> float:
    return start + (end - start) * progress


def smooth_mean_coupled_weights(step_index: int, num_steps: int, args) -> tuple[float, float, float]:
    progress = schedule_progress(step_index, num_steps, args.smooth_schedule)
    w_vid = interpolate_schedule(args.smooth_w_vid_start, args.smooth_w_vid_end, progress)
    w_tar = interpolate_schedule(args.smooth_w_tar_start, args.smooth_w_tar_end, progress)
    if args.smooth_src_basis == "mean":
        src_basis = 0.5 * (w_vid + w_tar)
    elif args.smooth_src_basis == "w_vid":
        src_basis = w_vid
    else:
        raise ValueError(f"Unknown smooth source guidance basis: {args.smooth_src_basis}")
    w_src = args.smooth_src_alpha * src_basis
    return w_vid, w_tar, w_src


def phase2_progress(step_index: int, transition_step: int, num_steps: int, schedule: str) -> float:
    denom = num_steps - transition_step - 1
    if denom <= 0:
        return 1.0
    progress = (step_index - transition_step) / denom
    progress = max(0.0, min(1.0, progress))
    if schedule == "linear":
        return progress
    if schedule == "cosine":
        return 0.5 - 0.5 * math.cos(math.pi * progress)
    if schedule == "smoothstep":
        return progress * progress * (3.0 - 2.0 * progress)
    raise ValueError(f"Unknown phase2 guidance schedule: {schedule}")


def phase2_video_decay_weights(step_index: int, num_steps: int, transition_step: int, args) -> tuple[float, float, float]:
    progress = phase2_progress(step_index, transition_step, num_steps, args.phase2_schedule)
    w_vid = interpolate_schedule(args.phase2_w_vid_start, args.phase2_w_vid_end, progress)
    w_tar = args.phase2_w_tar
    if args.phase2_src_mode == "const":
        w_src = args.phase2_w_src_const if args.phase2_w_src_const is not None else args.phase2_w_tar
    elif args.phase2_src_mode == "1p5_vid":
        w_src = args.phase2_src_vid_multiplier * w_vid
    else:
        raise ValueError(f"Unknown phase2 source mode: {args.phase2_src_mode}")
    return w_vid, w_tar, w_src


def build_weight_schedule(args) -> list[dict]:
    rows = []
    for step_index in range(args.num_steps):
        if args.sampler == SAMPLER_PHASE2_VIDEO_DECAY:
            if step_index < args.transition_step:
                rows.append({
                    "step": step_index,
                    "phase": 1,
                    "w_video": args.cfg_video,
                    "w_target": args.cfg_text,
                    "w_source": args.cfg_text if args.neg_src else 0.0,
                })
            else:
                w_vid, w_tar, w_src = phase2_video_decay_weights(
                    step_index, args.num_steps, args.transition_step, args
                )
                rows.append({
                    "step": step_index,
                    "phase": 2,
                    "w_video": w_vid,
                    "w_target": w_tar,
                    "w_source": w_src,
                })
        elif args.sampler == SAMPLER_SMOOTH_MEAN_COUPLED:
            w_vid, w_tar, w_src = smooth_mean_coupled_weights(step_index, args.num_steps, args)
            rows.append({
                "step": step_index,
                "phase": "smooth",
                "w_video": w_vid,
                "w_target": w_tar,
                "w_source": w_src,
            })
        else:
            phase = 1 if step_index < args.transition_step else 2
            rows.append({
                "step": step_index,
                "phase": phase,
                "w_video": args.cfg_video if phase == 1 else 0.0,
                "w_target": args.cfg_text if phase == 1 else args.cfg_strength,
                "w_source": (
                    args.cfg_text if (phase == 1 and args.neg_src)
                    else args.cfg_strength if (phase == 2 and args.neg_src_both)
                    else 0.0
                ),
            })
    return rows


def is_video_fully_processed(video_entry, output_dir: Path, args=None):
    """Check whether all expected outputs for a video are already present."""
    video_id_str = get_video_id_str(video_entry)
    correct_dir = output_dir / video_id_str / "correct"
    wrong_dir = output_dir / video_id_str / "wrong"

    correct_audio_path = correct_dir / f"{video_id_str}_correct.wav"
    correct_video_path = correct_dir / f"{video_id_str}_correct.mp4"
    if not (correct_audio_path.exists() and correct_video_path.exists()):
        return False

    _, target_prompts = get_prompt_plan(video_entry, args)
    for wrong_cat in target_prompts:
        safe_cat = safe_prompt_slug(wrong_cat)
        wrong_audio_path = wrong_dir / f"{video_id_str}_{safe_cat}.wav"
        wrong_video_path = wrong_dir / f"{video_id_str}_{safe_cat}.mp4"
        if not (wrong_audio_path.exists() and wrong_video_path.exists()):
            return False

    return True


def filter_pending_videos(video_entries, output_dir: Path, args=None):
    """Split entries into completed vs pending based on existing files."""
    pending_entries = []
    completed_count = 0

    for entry in video_entries:
        if is_video_fully_processed(entry, output_dir, args):
            completed_count += 1
        else:
            pending_entries.append(entry)

    return pending_entries, completed_count


def build_experiment_config(args, num_videos):
    return {
        "exp_name": args.exp_name,
        "sampler": args.sampler,
        "neg_src": args.neg_src,
        "neg_src_both": args.neg_src_both,
        "init_ode": args.init_ode,
        "transition_ode": args.transition_ode,
        "sigma": args.sigma,
        "transition_step": args.transition_step,
        "num_steps": args.num_steps,
        "cfg_strength": args.cfg_strength,
        "cfg_video": args.cfg_video,
        "cfg_text": args.cfg_text,
        "seed": args.seed,
        "variant": args.variant,
        "subset": args.subset,
        "clean_csv_path": args.clean_csv_path if args.subset == 'clean' else None,
        "pilot": args.pilot,
        "pilot_n": args.pilot_n if args.pilot else None,
        "num_videos": num_videos,
        "categories": CATEGORIES,
        "batch_size": args.batch_size,
        "precomputed_features_dir": args.precomputed_features_dir,
        "target_prompt_bank": args.target_prompt_bank,
        "target_case": args.target_case,
        "custom_target_prompts": bool(getattr(args, 'target_prompt_map', None)),
        "targets_per_source": {k: len(v["target_prompts"]) for k, v in (getattr(args, 'target_prompt_map', None) or {}).items()},
        "duration": args.duration,
        "smooth_schedule": args.smooth_schedule,
        "smooth_w_vid_start": args.smooth_w_vid_start,
        "smooth_w_vid_end": args.smooth_w_vid_end,
        "smooth_w_tar_start": args.smooth_w_tar_start,
        "smooth_w_tar_end": args.smooth_w_tar_end,
        "smooth_src_alpha": args.smooth_src_alpha,
        "smooth_src_basis": args.smooth_src_basis,
        "phase2_schedule": args.phase2_schedule,
        "phase2_w_vid_start": args.phase2_w_vid_start,
        "phase2_w_vid_end": args.phase2_w_vid_end,
        "phase2_w_tar": args.phase2_w_tar,
        "phase2_w_src_const": args.phase2_w_src_const,
        "phase2_src_mode": args.phase2_src_mode,
        "phase2_src_vid_multiplier": args.phase2_src_vid_multiplier,
    }


def find_config_mismatches(existing_config, current_config):
    """Compare critical config keys and return mismatches."""
    keys_to_compare = [
        "exp_name", "sampler", "neg_src", "neg_src_both",
        "init_ode", "transition_ode", "sigma",
        "transition_step", "num_steps",
        "cfg_strength", "cfg_video", "cfg_text",
        "seed", "variant", "subset",
        "clean_csv_path", "pilot", "pilot_n",
        "batch_size", "precomputed_features_dir",
        "target_prompt_bank", "target_case", "custom_target_prompts",
        "targets_per_source", "duration",
        "smooth_schedule", "smooth_w_vid_start", "smooth_w_vid_end",
        "smooth_w_tar_start", "smooth_w_tar_end", "smooth_src_alpha",
        "smooth_src_basis",
        "phase2_schedule", "phase2_w_vid_start", "phase2_w_vid_end",
        "phase2_w_tar", "phase2_w_src_const", "phase2_src_mode",
        "phase2_src_vid_multiplier",
    ]

    mismatches = {}
    for key in keys_to_compare:
        if key in existing_config and key in current_config:
            if existing_config[key] != current_config[key]:
                mismatches[key] = {
                    "existing": existing_config[key],
                    "current": current_config[key],
                }
    return mismatches


def combine_video_audio(video_path, audio_tensor, sampling_rate, output_path):
    """Combine video and audio into a single file."""
    if hasattr(audio_tensor, 'numpy'):
        audio_np = audio_tensor.numpy()
    else:
        audio_np = audio_tensor

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_audio:
        tmp_audio_path = tmp_audio.name

    audio_for_save = torch.from_numpy(audio_np) if isinstance(audio_np, np.ndarray) else audio_np
    torchaudio.save(tmp_audio_path, audio_for_save, sampling_rate)

    cmd = [
        'ffmpeg', '-y',
        '-i', str(video_path),
        '-i', tmp_audio_path,
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-b:a', '192k',
        '-map', '0:v:0',
        '-map', '1:a:0',
        '-shortest',
        str(output_path)
    ]

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e.stderr}")
        raise
    finally:
        if os.path.exists(tmp_audio_path):
            os.unlink(tmp_audio_path)


@torch.inference_mode()
def prepare_conditions_standard(net, feature_utils, clip_frames, sync_frames, prompt, device, dtype,
                                precomputed_clip=None, precomputed_sync=None,
                                precomputed_text=None):
    """Prepare conditions for standard MMAudio generation (Video + text)."""
    bs = 1
    if precomputed_clip is not None:
        clip_features = precomputed_clip.to(device, dtype, non_blocking=True)
    else:
        clip_video = clip_frames.to(device, dtype, non_blocking=True)
        clip_features = feature_utils.encode_video_with_clip(clip_video, batch_size=bs * 40)

    if precomputed_sync is not None:
        sync_features = precomputed_sync.to(device, dtype, non_blocking=True)
    else:
        sync_video = sync_frames.to(device, dtype, non_blocking=True)
        sync_features = feature_utils.encode_video_with_sync(sync_video, batch_size=bs * 40)

    if precomputed_text is not None:
        text_features = precomputed_text.to(device, dtype, non_blocking=True)
    else:
        text_features = feature_utils.encode_text([prompt])

    preprocessed_conditions = net.preprocess_conditions(clip_features, sync_features, text_features)
    empty_conditions = net.get_empty_conditions(bs)

    return preprocessed_conditions, empty_conditions


@torch.inference_mode()
def prepare_conditions_transition(net, feature_utils, clip_frames, sync_frames,
                                  target_prompt, source_prompt, neg_src, neg_src_both,
                                  device, dtype,
                                  precomputed_clip=None, precomputed_sync=None,
                                  precomputed_target_text=None, precomputed_source_text=None):
    """Prepare decomposed conditions for prompt switch generation."""
    bs = 1
    if precomputed_clip is not None:
        clip_features = precomputed_clip.to(device, dtype, non_blocking=True)
    else:
        clip_video = clip_frames.to(device, dtype, non_blocking=True)
        clip_features = feature_utils.encode_video_with_clip(clip_video, batch_size=bs * 40)

    if precomputed_sync is not None:
        sync_features = precomputed_sync.to(device, dtype, non_blocking=True)
    else:
        sync_video = sync_frames.to(device, dtype, non_blocking=True)
        sync_features = feature_utils.encode_video_with_sync(sync_video, batch_size=bs * 40)

    if precomputed_target_text is not None:
        target_text_features = precomputed_target_text.to(device, dtype, non_blocking=True)
    else:
        target_text_features = feature_utils.encode_text([target_prompt])

    empty_text = net.get_empty_string_sequence(bs)
    empty_clip = net.get_empty_clip_sequence(bs)
    empty_sync = net.get_empty_sync_sequence(bs)

    init_video_conditions = net.preprocess_conditions(clip_features, sync_features, empty_text)
    init_text_conditions = net.preprocess_conditions(empty_clip, empty_sync, target_text_features)
    transition_conditions = net.preprocess_conditions(empty_clip, empty_sync, target_text_features)
    empty_conditions = net.get_empty_conditions(bs)

    src_text_conditions = None
    if neg_src or neg_src_both:
        if precomputed_source_text is not None:
            source_text_features = precomputed_source_text.to(device, dtype, non_blocking=True)
        else:
            source_text_features = feature_utils.encode_text([source_prompt])
        src_text_conditions = net.preprocess_conditions(empty_clip, empty_sync, source_text_features)

    return init_video_conditions, init_text_conditions, transition_conditions, empty_conditions, src_text_conditions


@torch.inference_mode()
def generate_audio_standard(net, feature_utils, preprocessed_conditions, empty_conditions,
                            device, dtype, num_steps=25, cfg_strength=4.5, seed=42, sigma=0.0,
                            phase1_sde=False):
    """Generate audio with standard MMAudio (ODE or SDE based on phase1 setting)."""
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    bs = 1
    x0 = torch.randn(bs, net.latent_seq_len, net.latent_dim,
                      device=device, dtype=dtype, generator=rng)

    x = x0
    steps = torch.linspace(0, 1, num_steps + 1)

    for ti, t in enumerate(steps[:-1]):
        next_t = steps[ti + 1]
        dt = next_t - t

        if phase1_sde and sigma > 0:
            sigma_t = sigma * (1 - t).sqrt()
            flow = net.sde_wrapper(t, x, preprocessed_conditions, empty_conditions, cfg_strength, sigma=sigma_t)
            x = x + dt * flow + (sigma_t * (dt ** 0.5)) * torch.randn_like(x)
        else:
            flow = net.ode_wrapper(t, x, preprocessed_conditions, empty_conditions, cfg_strength)
            x = x + dt * flow

    x1 = net.unnormalize(x)
    spec = feature_utils.decode(x1)
    audio = feature_utils.vocode(spec)

    return audio.float().cpu()[0]


@torch.inference_mode()
def generate_audio_prompt_switch(net, feature_utils,
                                 init_video_conditions, init_text_conditions,
                                 transition_conditions, empty_conditions,
                                 src_text_conditions,
                                 device, dtype,
                                 transition_step=17, num_steps=25,
                                 cfg_strength=4.5, cfg_video=3.0, cfg_text=5.0,
                                 seed=42, sigma=0.0,
                                 init_ode=True, transition_ode=True,
                                 neg_src=False, neg_src_both=False):
    """Generate audio with prompt switch (decomposed CFG in Phase 1, standard/neg_src in Phase 2)."""
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    bs = 1
    x0 = torch.randn(bs, net.latent_seq_len, net.latent_dim,
                      device=device, dtype=dtype, generator=rng)

    x = x0
    steps = torch.linspace(0, 1, num_steps + 1)

    for ti, t in enumerate(steps[:-1]):
        next_t = steps[ti + 1]
        dt = next_t - t

        if ti < transition_step:
            # Phase 1: decomposed CFG
            if neg_src and src_text_conditions is not None:
                vector_field = net.ode_wrapper_decomposed_cfg_neg_src(
                    t, x, init_video_conditions, init_text_conditions,
                    src_text_conditions, empty_conditions, cfg_video, cfg_text
                )
            else:
                vector_field = net.ode_wrapper_decomposed_cfg(
                    t, x, init_video_conditions, init_text_conditions,
                    empty_conditions, cfg_video, cfg_text
                )

            if not init_ode:
                sigma_t = sigma * (1 - t).sqrt()
                mu_sde = vector_field + 0.5 * (sigma ** 2) * (t * vector_field - x) / (1 - t)
                x = x + dt * mu_sde + (sigma_t * (dt ** 0.5)) * torch.randn_like(x)
            else:
                x = x + dt * vector_field
        else:
            # Phase 2: transition
            if neg_src_both and src_text_conditions is not None:
                sigma_t = sigma * (1 - t).sqrt()
                t_batch = t * torch.ones(len(x), device=x.device, dtype=x.dtype)
                vector_field_empty = net.predict_flow(x, t_batch, empty_conditions)
                vector_field_tar = net.predict_flow(x, t_batch, transition_conditions)
                vector_field_src = net.predict_flow(x, t_batch, src_text_conditions)
                vector_field = vector_field_empty + cfg_strength * (vector_field_tar - vector_field_src)

                if not transition_ode:
                    mu_SDE = vector_field + 0.5 * (sigma_t ** 2) * (t * vector_field - x) / (1 - t)
                    x = x + dt * mu_SDE + (sigma_t * (dt ** 0.5)) * torch.randn_like(x)
                else:
                    x = x + dt * vector_field
            else:
                if not transition_ode:
                    sigma_t = sigma * (1 - t).sqrt()
                    flow = net.sde_wrapper(t, x, transition_conditions, empty_conditions, cfg_strength, sigma=sigma_t)
                    x = x + dt * flow + (sigma_t * (dt ** 0.5)) * torch.randn_like(x)
                else:
                    flow = net.ode_wrapper(t, x, transition_conditions, empty_conditions, cfg_strength)
                    x = x + dt * flow

    x1 = net.unnormalize(x)
    spec = feature_utils.decode(x1)
    audio = feature_utils.vocode(spec)

    return audio.float().cpu()[0]


@torch.inference_mode()
def generate_audio_phase2_video_decay(net, feature_utils,
                                      init_video_conditions, init_text_conditions,
                                      transition_conditions, empty_conditions,
                                      src_text_conditions,
                                      device, dtype,
                                      transition_step=17, num_steps=25,
                                      cfg_video=3.0, cfg_text=5.0,
                                      seed=42,
                                      phase2_schedule="smoothstep",
                                      phase2_w_vid_start=3.0,
                                      phase2_w_vid_end=0.5,
                                      phase2_w_tar=4.5,
                                      phase2_w_src_const=None,
                                      phase2_src_mode="const",
                                      phase2_src_vid_multiplier=1.5):
    """Generate audio with strict Phase 1 and residual video guidance in Phase 2."""
    if src_text_conditions is None:
        raise ValueError("phase2_video_decay sampler requires source text conditions")

    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    bs = 1
    x0 = torch.randn(bs, net.latent_seq_len, net.latent_dim,
                     device=device, dtype=dtype, generator=rng)

    x = x0
    steps = torch.linspace(0, 1, num_steps + 1)
    args_like = argparse.Namespace(
        phase2_schedule=phase2_schedule,
        phase2_w_vid_start=phase2_w_vid_start,
        phase2_w_vid_end=phase2_w_vid_end,
        phase2_w_tar=phase2_w_tar,
        phase2_w_src_const=phase2_w_src_const,
        phase2_src_mode=phase2_src_mode,
        phase2_src_vid_multiplier=phase2_src_vid_multiplier,
    )

    for ti, t in enumerate(steps[:-1]):
        next_t = steps[ti + 1]
        dt = next_t - t

        if ti < transition_step:
            vector_field = net.ode_wrapper_decomposed_cfg_neg_src(
                t, x, init_video_conditions, init_text_conditions,
                src_text_conditions, empty_conditions, cfg_video, cfg_text
            )
        else:
            w_vid, w_tar, w_src = phase2_video_decay_weights(
                ti, num_steps, transition_step, args_like
            )
            t_batch = t * torch.ones(len(x), device=x.device, dtype=x.dtype)
            vector_field_empty = net.predict_flow(x, t_batch, empty_conditions)
            vector_field_video = net.predict_flow(x, t_batch, init_video_conditions)
            vector_field_tar = net.predict_flow(x, t_batch, transition_conditions)
            vector_field_src = net.predict_flow(x, t_batch, src_text_conditions)
            vector_field = (
                vector_field_empty
                + w_vid * (vector_field_video - vector_field_empty)
                + w_tar * (vector_field_tar - vector_field_empty)
                - w_src * (vector_field_src - vector_field_empty)
            )
        x = x + dt * vector_field

    x1 = net.unnormalize(x)
    spec = feature_utils.decode(x1)
    audio = feature_utils.vocode(spec)

    return audio.float().cpu()[0]


@torch.inference_mode()
def generate_audio_smooth_mean_coupled(net, feature_utils,
                                       video_conditions, target_conditions,
                                       empty_conditions, src_text_conditions,
                                       device, dtype,
                                       num_steps=25, seed=42,
                                       smooth_schedule="smoothstep",
                                       smooth_w_vid_start=3.5,
                                       smooth_w_vid_end=1.2,
                                       smooth_w_tar_start=1.8,
                                       smooth_w_tar_end=5.0,
                                       smooth_src_alpha=1.0,
                                       smooth_src_basis="mean"):
    """Generate audio with smooth mean-coupled video/target/source guidance."""
    if src_text_conditions is None:
        raise ValueError("smooth_mean_coupled sampler requires source text conditions")

    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    bs = 1
    x0 = torch.randn(bs, net.latent_seq_len, net.latent_dim,
                     device=device, dtype=dtype, generator=rng)

    x = x0
    steps = torch.linspace(0, 1, num_steps + 1)

    args_like = argparse.Namespace(
        smooth_schedule=smooth_schedule,
        smooth_w_vid_start=smooth_w_vid_start,
        smooth_w_vid_end=smooth_w_vid_end,
        smooth_w_tar_start=smooth_w_tar_start,
        smooth_w_tar_end=smooth_w_tar_end,
        smooth_src_alpha=smooth_src_alpha,
        smooth_src_basis=smooth_src_basis,
    )

    for ti, t in enumerate(steps[:-1]):
        next_t = steps[ti + 1]
        dt = next_t - t

        w_vid, w_tar, w_src = smooth_mean_coupled_weights(ti, num_steps, args_like)
        t_batch = t * torch.ones(len(x), device=x.device, dtype=x.dtype)
        vector_field_empty = net.predict_flow(x, t_batch, empty_conditions)
        vector_field_video = net.predict_flow(x, t_batch, video_conditions)
        vector_field_tar = net.predict_flow(x, t_batch, target_conditions)
        vector_field_src = net.predict_flow(x, t_batch, src_text_conditions)
        vector_field = (
            vector_field_empty
            + w_vid * (vector_field_video - vector_field_empty)
            + w_tar * (vector_field_tar - vector_field_empty)
            - w_src * (vector_field_src - vector_field_empty)
        )
        x = x + dt * vector_field

    x1 = net.unnormalize(x)
    spec = feature_utils.decode(x1)
    audio = feature_utils.vocode(spec)

    return audio.float().cpu()[0]


@torch.inference_mode()
def prepare_conditions_transition_batched(net, feature_utils, clip_frames, sync_frames,
                                          target_prompts, source_prompt,
                                          neg_src, neg_src_both,
                                          device, dtype,
                                          precomputed_clip=None, precomputed_sync=None,
                                          precomputed_target_texts=None,
                                          precomputed_source_text=None):
    """Prepare batched decomposed conditions for prompt switch generation.

    target_prompts: list of N target prompts (or None if using precomputed_target_texts).
    precomputed_target_texts: (N, L, D) tensor (stacked) if available.
    All non-text features (clip, sync, source_text) are expanded to batch size N.
    """
    n = len(target_prompts) if target_prompts is not None else precomputed_target_texts.shape[0]

    # --- clip features: (1, T, D) -> (N, T, D) ---
    if precomputed_clip is not None:
        clip_features = precomputed_clip.to(device, dtype, non_blocking=True)
    else:
        clip_video = clip_frames.to(device, dtype, non_blocking=True)
        clip_features = feature_utils.encode_video_with_clip(clip_video, batch_size=40)
    clip_features = clip_features.expand(n, -1, -1).contiguous()

    # --- sync features ---
    if precomputed_sync is not None:
        sync_features = precomputed_sync.to(device, dtype, non_blocking=True)
    else:
        sync_video = sync_frames.to(device, dtype, non_blocking=True)
        sync_features = feature_utils.encode_video_with_sync(sync_video, batch_size=40)
    sync_features = sync_features.expand(n, -1, -1).contiguous()

    # --- target text features: (N, L, D) ---
    if precomputed_target_texts is not None:
        target_text_features = precomputed_target_texts.to(device, dtype, non_blocking=True)
    else:
        target_text_features = feature_utils.encode_text(target_prompts)

    empty_clip = net.get_empty_clip_sequence(n)
    empty_sync = net.get_empty_sync_sequence(n)

    init_video_conditions = net.preprocess_conditions(
        clip_features, sync_features, net.get_empty_string_sequence(n))
    init_text_conditions = net.preprocess_conditions(
        empty_clip, empty_sync, target_text_features)
    transition_conditions = net.preprocess_conditions(
        empty_clip, empty_sync, target_text_features)
    empty_conditions = net.get_empty_conditions(n)

    src_text_conditions = None
    if neg_src or neg_src_both:
        if precomputed_source_text is not None:
            source_text_single = precomputed_source_text.to(device, dtype, non_blocking=True)
        else:
            source_text_single = feature_utils.encode_text([source_prompt])
        source_text_features = source_text_single.expand(n, -1, -1).contiguous()
        src_text_conditions = net.preprocess_conditions(
            empty_clip, empty_sync, source_text_features)

    return (init_video_conditions, init_text_conditions, transition_conditions,
            empty_conditions, src_text_conditions)


@torch.inference_mode()
def generate_audio_prompt_switch_batched(net, feature_utils,
                                         init_video_conditions, init_text_conditions,
                                         transition_conditions, empty_conditions,
                                         src_text_conditions,
                                         batch_size, device, dtype,
                                         transition_step=17, num_steps=25,
                                         cfg_strength=4.5, cfg_video=3.0, cfg_text=5.0,
                                         seed=42, sigma=0.0,
                                         init_ode=True, transition_ode=True,
                                         neg_src=False, neg_src_both=False):
    """Batched version of generate_audio_prompt_switch.

    Runs the diffusion loop once with batch dim = batch_size. Returns a list of
    per-sample audio tensors (each shape matching the single-sample output).

    Note: to preserve the seed semantics of the single-sample path (same x0 for
    each call), x0 is drawn once and replicated across the batch. SDE noise is
    drawn batched, so results differ slightly from the bs=1 loop but are
    deterministic for a given (seed, batch_size).
    """
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    # Replicate the single-sample x0 across batch (matches per-call seed semantics).
    x0_single = torch.randn(1, net.latent_seq_len, net.latent_dim,
                            device=device, dtype=dtype, generator=rng)
    x = x0_single.expand(batch_size, -1, -1).contiguous()

    steps = torch.linspace(0, 1, num_steps + 1)

    for ti, t in enumerate(steps[:-1]):
        next_t = steps[ti + 1]
        dt = next_t - t

        if ti < transition_step:
            # Phase 1: decomposed CFG
            if neg_src and src_text_conditions is not None:
                vector_field = net.ode_wrapper_decomposed_cfg_neg_src(
                    t, x, init_video_conditions, init_text_conditions,
                    src_text_conditions, empty_conditions, cfg_video, cfg_text
                )
            else:
                vector_field = net.ode_wrapper_decomposed_cfg(
                    t, x, init_video_conditions, init_text_conditions,
                    empty_conditions, cfg_video, cfg_text
                )

            if not init_ode:
                sigma_t = sigma * (1 - t).sqrt()
                mu_sde = vector_field + 0.5 * (sigma ** 2) * (t * vector_field - x) / (1 - t)
                x = x + dt * mu_sde + (sigma_t * (dt ** 0.5)) * torch.randn(
                    x.shape, device=device, dtype=dtype, generator=rng)
            else:
                x = x + dt * vector_field
        else:
            # Phase 2
            if neg_src_both and src_text_conditions is not None:
                sigma_t = sigma * (1 - t).sqrt()
                t_batch = t * torch.ones(len(x), device=x.device, dtype=x.dtype)
                vector_field_empty = net.predict_flow(x, t_batch, empty_conditions)
                vector_field_tar = net.predict_flow(x, t_batch, transition_conditions)
                vector_field_src = net.predict_flow(x, t_batch, src_text_conditions)
                vector_field = vector_field_empty + cfg_strength * (vector_field_tar - vector_field_src)

                if not transition_ode:
                    mu_SDE = vector_field + 0.5 * (sigma_t ** 2) * (t * vector_field - x) / (1 - t)
                    x = x + dt * mu_SDE + (sigma_t * (dt ** 0.5)) * torch.randn(
                        x.shape, device=device, dtype=dtype, generator=rng)
                else:
                    x = x + dt * vector_field
            else:
                if not transition_ode:
                    sigma_t = sigma * (1 - t).sqrt()
                    flow = net.sde_wrapper(t, x, transition_conditions, empty_conditions,
                                           cfg_strength, sigma=sigma_t)
                    x = x + dt * flow + (sigma_t * (dt ** 0.5)) * torch.randn(
                        x.shape, device=device, dtype=dtype, generator=rng)
                else:
                    flow = net.ode_wrapper(t, x, transition_conditions, empty_conditions, cfg_strength)
                    x = x + dt * flow

    x1 = net.unnormalize(x)
    spec = feature_utils.decode(x1)
    audio = feature_utils.vocode(spec)  # (bs, C, T) or (bs, T)

    audio_cpu = audio.float().cpu()
    return [audio_cpu[i] for i in range(batch_size)]


@torch.inference_mode()
def generate_audio_phase2_video_decay_batched(net, feature_utils,
                                              init_video_conditions, init_text_conditions,
                                              transition_conditions, empty_conditions,
                                              src_text_conditions,
                                              batch_size, device, dtype,
                                              transition_step=17, num_steps=25,
                                              cfg_video=3.0, cfg_text=5.0,
                                              seed=42,
                                              phase2_schedule="smoothstep",
                                              phase2_w_vid_start=3.0,
                                              phase2_w_vid_end=0.5,
                                              phase2_w_tar=4.5,
                                              phase2_w_src_const=None,
                                              phase2_src_mode="const",
                                              phase2_src_vid_multiplier=1.5):
    """Batched strict Phase 1 + Phase 2 residual video guidance sampler."""
    if src_text_conditions is None:
        raise ValueError("phase2_video_decay sampler requires source text conditions")

    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    x0_single = torch.randn(1, net.latent_seq_len, net.latent_dim,
                            device=device, dtype=dtype, generator=rng)
    x = x0_single.expand(batch_size, -1, -1).contiguous()

    steps = torch.linspace(0, 1, num_steps + 1)
    args_like = argparse.Namespace(
        phase2_schedule=phase2_schedule,
        phase2_w_vid_start=phase2_w_vid_start,
        phase2_w_vid_end=phase2_w_vid_end,
        phase2_w_tar=phase2_w_tar,
        phase2_w_src_const=phase2_w_src_const,
        phase2_src_mode=phase2_src_mode,
        phase2_src_vid_multiplier=phase2_src_vid_multiplier,
    )

    for ti, t in enumerate(steps[:-1]):
        next_t = steps[ti + 1]
        dt = next_t - t

        if ti < transition_step:
            vector_field = net.ode_wrapper_decomposed_cfg_neg_src(
                t, x, init_video_conditions, init_text_conditions,
                src_text_conditions, empty_conditions, cfg_video, cfg_text
            )
        else:
            w_vid, w_tar, w_src = phase2_video_decay_weights(
                ti, num_steps, transition_step, args_like
            )
            t_batch = t * torch.ones(len(x), device=x.device, dtype=x.dtype)
            vector_field_empty = net.predict_flow(x, t_batch, empty_conditions)
            vector_field_video = net.predict_flow(x, t_batch, init_video_conditions)
            vector_field_tar = net.predict_flow(x, t_batch, transition_conditions)
            vector_field_src = net.predict_flow(x, t_batch, src_text_conditions)
            vector_field = (
                vector_field_empty
                + w_vid * (vector_field_video - vector_field_empty)
                + w_tar * (vector_field_tar - vector_field_empty)
                - w_src * (vector_field_src - vector_field_empty)
            )
        x = x + dt * vector_field

    x1 = net.unnormalize(x)
    spec = feature_utils.decode(x1)
    audio = feature_utils.vocode(spec)

    audio_cpu = audio.float().cpu()
    return [audio_cpu[i] for i in range(batch_size)]


@torch.inference_mode()
def generate_audio_smooth_mean_coupled_batched(net, feature_utils,
                                               video_conditions, target_conditions,
                                               empty_conditions, src_text_conditions,
                                               batch_size, device, dtype,
                                               num_steps=25, seed=42,
                                               smooth_schedule="smoothstep",
                                               smooth_w_vid_start=3.5,
                                               smooth_w_vid_end=1.2,
                                               smooth_w_tar_start=1.8,
                                               smooth_w_tar_end=5.0,
                                               smooth_src_alpha=1.0,
                                               smooth_src_basis="mean"):
    """Batched smooth mean-coupled guidance sampler."""
    if src_text_conditions is None:
        raise ValueError("smooth_mean_coupled sampler requires source text conditions")

    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    x0_single = torch.randn(1, net.latent_seq_len, net.latent_dim,
                            device=device, dtype=dtype, generator=rng)
    x = x0_single.expand(batch_size, -1, -1).contiguous()

    steps = torch.linspace(0, 1, num_steps + 1)
    args_like = argparse.Namespace(
        smooth_schedule=smooth_schedule,
        smooth_w_vid_start=smooth_w_vid_start,
        smooth_w_vid_end=smooth_w_vid_end,
        smooth_w_tar_start=smooth_w_tar_start,
        smooth_w_tar_end=smooth_w_tar_end,
        smooth_src_alpha=smooth_src_alpha,
        smooth_src_basis=smooth_src_basis,
    )

    for ti, t in enumerate(steps[:-1]):
        next_t = steps[ti + 1]
        dt = next_t - t

        w_vid, w_tar, w_src = smooth_mean_coupled_weights(ti, num_steps, args_like)
        t_batch = t * torch.ones(len(x), device=x.device, dtype=x.dtype)
        vector_field_empty = net.predict_flow(x, t_batch, empty_conditions)
        vector_field_video = net.predict_flow(x, t_batch, video_conditions)
        vector_field_tar = net.predict_flow(x, t_batch, target_conditions)
        vector_field_src = net.predict_flow(x, t_batch, src_text_conditions)
        vector_field = (
            vector_field_empty
            + w_vid * (vector_field_video - vector_field_empty)
            + w_tar * (vector_field_tar - vector_field_empty)
            - w_src * (vector_field_src - vector_field_empty)
        )
        x = x + dt * vector_field

    x1 = net.unnormalize(x)
    spec = feature_utils.decode(x1)
    audio = feature_utils.vocode(spec)

    audio_cpu = audio.float().cpu()
    return [audio_cpu[i] for i in range(batch_size)]


def process_single_video(video_entry, args, net, feature_utils, seq_cfg, sampling_rate, device, dtype,
                         text_features_cache=None):
    """Process a single video: generate correct + target prompt audios."""
    video_id = video_entry['video_id']
    start_sec = video_entry['start_sec']
    correct_category = video_entry['category']
    filename = video_entry['filename']
    video_path = VIDEO_ROOT / filename

    # Create output directory: output_dir/video_id
    video_id_str = get_video_id_str(video_entry)
    video_out_dir = Path(args.output_dir) / video_id_str
    correct_dir = video_out_dir / "correct"
    wrong_dir = video_out_dir / "wrong"
    correct_dir.mkdir(parents=True, exist_ok=True)
    wrong_dir.mkdir(parents=True, exist_ok=True)

    correct_audio_path = correct_dir / f"{video_id_str}_correct.wav"
    correct_video_path = correct_dir / f"{video_id_str}_correct.mp4"
    correct_done = correct_audio_path.exists() and correct_video_path.exists()

    source_prompt, wrong_categories = get_prompt_plan(video_entry, args)
    existing_wrong_results = {}
    pending_wrong_categories = []
    for wrong_cat in wrong_categories:
        safe_cat = safe_prompt_slug(wrong_cat)
        wrong_audio_path = wrong_dir / f"{video_id_str}_{safe_cat}.wav"
        wrong_video_path = wrong_dir / f"{video_id_str}_{safe_cat}.mp4"
        if wrong_audio_path.exists() and wrong_video_path.exists():
            existing_wrong_results[wrong_cat] = {
                "source_prompt": source_prompt,
                "target_prompt": wrong_cat,
                "target_case": args.target_case if getattr(args, 'target_prompt_map', None) else None,
                "audio_file": wrong_audio_path.name,
                "video_file": wrong_video_path.name,
            }
        else:
            pending_wrong_categories.append(wrong_cat)

    if correct_done and not pending_wrong_categories:
        print(f"[SKIP] {video_id_str} already completed")
        return video_id_str

    if not video_path.exists():
        print(f"[SKIP] {filename} not found")
        return None

    use_precomputed = args.precomputed_features_dir is not None
    precomputed_clip = None
    precomputed_sync = None
    clip_frames = None
    sync_frames = None

    if use_precomputed:
        feat_path = Path(args.precomputed_features_dir) / 'video' / f"{video_id_str}.pt"
        if not feat_path.exists():
            print(f"[SKIP] precomputed feature not found: {feat_path}")
            return None
        feats = torch.load(feat_path, map_location='cpu', weights_only=True)
        precomputed_clip = feats['clip_features']
        precomputed_sync = feats['sync_features']
        task_duration = float(feats['duration_sec'])
    else:
        # Load video. Upstream MMAudio returns VideoInfo; some local research
        # branches returned (VideoInfo, frames). Support both shapes.
        loaded_video = load_video(video_path, args.duration)
        video_info = loaded_video[0] if isinstance(loaded_video, tuple) else loaded_video
        clip_frames = video_info.clip_frames.unsqueeze(0)
        sync_frames = video_info.sync_frames.unsqueeze(0)
        task_duration = video_info.duration_sec

    seq_cfg.duration = task_duration
    net.update_seq_lengths(seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len)

    def _get_text(cat):
        if text_features_cache is None:
            return None
        return text_features_cache.get(cat)

    # --- Correct category: standard MMAudio ---
    if correct_done:
        print(f"[SKIP] correct already exists: {video_id_str}")
    else:
        correct_conditions, correct_empty = prepare_conditions_standard(
            net, feature_utils, clip_frames, sync_frames, source_prompt, device, dtype,
            precomputed_clip=precomputed_clip, precomputed_sync=precomputed_sync,
            precomputed_text=_get_text(source_prompt),
        )

        # For correct, determine if Phase 1 uses SDE
        phase1_sde = not args.init_ode
        correct_audio = generate_audio_standard(
            net, feature_utils, correct_conditions, correct_empty,
            device, dtype, num_steps=args.num_steps, cfg_strength=args.cfg_strength,
            seed=args.seed, sigma=args.sigma, phase1_sde=phase1_sde
        )

        # Save correct audio only
        torchaudio.save(str(correct_audio_path), correct_audio, sampling_rate)

        # Save correct audio + video
        combine_video_audio(video_path, correct_audio, sampling_rate, str(correct_video_path))

    correct_meta = {
        "video_id": video_id,
        "start_sec": start_sec,
        "video_path": str(video_path),
        "correct_category": correct_category,
        "source_prompt": source_prompt,
        "prompt": source_prompt,
        "generation_type": "standard",
        "sampler": args.sampler,
        "seed": args.seed,
        "cfg_strength": args.cfg_strength,
        "cfg_video": args.cfg_video,
        "cfg_text": args.cfg_text,
        "num_steps": args.num_steps,
        "sigma": args.sigma,
        "init_ode": args.init_ode,
        "transition_ode": args.transition_ode,
        "transition_step": args.transition_step,
        "neg_src": args.neg_src,
        "neg_src_both": args.neg_src_both,
        "variant": args.variant,
        "duration": args.duration,
        "smooth_schedule": args.smooth_schedule,
        "smooth_w_vid_start": args.smooth_w_vid_start,
        "smooth_w_vid_end": args.smooth_w_vid_end,
        "smooth_w_tar_start": args.smooth_w_tar_start,
        "smooth_w_tar_end": args.smooth_w_tar_end,
        "smooth_src_alpha": args.smooth_src_alpha,
        "smooth_src_basis": args.smooth_src_basis,
        "phase2_schedule": args.phase2_schedule,
        "phase2_w_vid_start": args.phase2_w_vid_start,
        "phase2_w_vid_end": args.phase2_w_vid_end,
        "phase2_w_tar": args.phase2_w_tar,
        "phase2_w_src_const": args.phase2_w_src_const,
        "phase2_src_mode": args.phase2_src_mode,
        "phase2_src_vid_multiplier": args.phase2_src_vid_multiplier,
        "target_prompt_bank": args.target_prompt_bank,
        "target_case": args.target_case if getattr(args, 'target_prompt_map', None) else None,
    }
    with open(correct_dir / "metadata.json", "w") as f:
        json.dump(correct_meta, f, ensure_ascii=False, indent=2)

    # --- Wrong categories: prompt switch ---
    generated_wrong_results = {}

    if pending_wrong_categories:
        need_src_conditions = needs_source_conditions(args)
        # Determine batch size (<=11 for wrong categories)
        wrong_batch_size = max(1, int(getattr(args, 'batch_size', 1)))
        wrong_batch_size = min(wrong_batch_size, len(pending_wrong_categories))

        # Build list of (cat, audio) pairs via batched or sequential path
        wrong_audios = {}  # wrong_cat -> audio tensor

        if wrong_batch_size > 1:
            for chunk_start in range(0, len(pending_wrong_categories), wrong_batch_size):
                chunk = pending_wrong_categories[chunk_start:chunk_start + wrong_batch_size]
                n = len(chunk)

                # Collect precomputed target text features (if cache available)
                precomputed_target_texts = None
                if text_features_cache is not None:
                    precomputed_target_texts = torch.cat(
                        [text_features_cache[c] for c in chunk], dim=0
                    )

                (init_video_cond_b, init_text_cond_b, transition_cond_b,
                 empty_cond_b, src_text_cond_b) = prepare_conditions_transition_batched(
                    net, feature_utils, clip_frames, sync_frames,
                    target_prompts=chunk,
                    source_prompt=source_prompt,
                    neg_src=need_src_conditions,
                    neg_src_both=args.neg_src_both,
                    device=device, dtype=dtype,
                    precomputed_clip=precomputed_clip, precomputed_sync=precomputed_sync,
                    precomputed_target_texts=precomputed_target_texts,
                    precomputed_source_text=_get_text(source_prompt),
                )

                if args.sampler == SAMPLER_SMOOTH_MEAN_COUPLED:
                    audios = generate_audio_smooth_mean_coupled_batched(
                        net, feature_utils,
                        init_video_cond_b, init_text_cond_b, empty_cond_b, src_text_cond_b,
                        batch_size=n, device=device, dtype=dtype,
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
                elif args.sampler == SAMPLER_PHASE2_VIDEO_DECAY:
                    audios = generate_audio_phase2_video_decay_batched(
                        net, feature_utils,
                        init_video_cond_b, init_text_cond_b, transition_cond_b, empty_cond_b, src_text_cond_b,
                        batch_size=n, device=device, dtype=dtype,
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
                else:
                    audios = generate_audio_prompt_switch_batched(
                        net, feature_utils,
                        init_video_cond_b, init_text_cond_b, transition_cond_b, empty_cond_b, src_text_cond_b,
                        batch_size=n, device=device, dtype=dtype,
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
                for cat, audio in zip(chunk, audios):
                    wrong_audios[cat] = audio
        else:
            for wrong_cat in pending_wrong_categories:
                (init_video_cond, init_text_cond, transition_cond,
                 empty_cond, src_text_cond) = prepare_conditions_transition(
                    net, feature_utils, clip_frames, sync_frames,
                    target_prompt=wrong_cat,
                    source_prompt=source_prompt,
                    neg_src=need_src_conditions,
                    neg_src_both=args.neg_src_both,
                    device=device, dtype=dtype,
                    precomputed_clip=precomputed_clip, precomputed_sync=precomputed_sync,
                    precomputed_target_text=_get_text(wrong_cat),
                    precomputed_source_text=_get_text(source_prompt),
                )

                if args.sampler == SAMPLER_SMOOTH_MEAN_COUPLED:
                    wrong_audios[wrong_cat] = generate_audio_smooth_mean_coupled(
                        net, feature_utils,
                        init_video_cond, init_text_cond, empty_cond, src_text_cond,
                        device, dtype,
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
                elif args.sampler == SAMPLER_PHASE2_VIDEO_DECAY:
                    wrong_audios[wrong_cat] = generate_audio_phase2_video_decay(
                        net, feature_utils,
                        init_video_cond, init_text_cond, transition_cond, empty_cond, src_text_cond,
                        device, dtype,
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
                else:
                    wrong_audios[wrong_cat] = generate_audio_prompt_switch(
                        net, feature_utils,
                        init_video_cond, init_text_cond, transition_cond, empty_cond, src_text_cond,
                        device, dtype,
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

        for wrong_cat in pending_wrong_categories:
            wrong_audio = wrong_audios[wrong_cat]

            # Sanitize category name for filename
            safe_cat = safe_prompt_slug(wrong_cat)

            # Save wrong audio only
            wrong_audio_path = wrong_dir / f"{video_id_str}_{safe_cat}.wav"
            torchaudio.save(str(wrong_audio_path), wrong_audio, sampling_rate)

            # Save wrong audio + video
            wrong_video_path = wrong_dir / f"{video_id_str}_{safe_cat}.mp4"
            combine_video_audio(video_path, wrong_audio, sampling_rate, str(wrong_video_path))

            generated_wrong_results[wrong_cat] = {
                "source_prompt": source_prompt,
                "target_prompt": wrong_cat,
                "target_case": args.target_case if getattr(args, 'target_prompt_map', None) else None,
                "audio_file": str(wrong_audio_path.name),
                "video_file": str(wrong_video_path.name),
            }

    wrong_results = []
    for wrong_cat in wrong_categories:
        if wrong_cat in existing_wrong_results:
            wrong_results.append(existing_wrong_results[wrong_cat])
        else:
            wrong_results.append(generated_wrong_results[wrong_cat])

    wrong_meta = {
        "video_id": video_id,
        "start_sec": start_sec,
        "video_path": str(video_path),
        "correct_category": correct_category,
        "source_prompt": source_prompt,
        "generation_type": "prompt_switch",
        "sampler": args.sampler,
        "neg_src": args.neg_src,
        "neg_src_both": args.neg_src_both,
        "init_ode": args.init_ode,
        "transition_ode": args.transition_ode,
        "transition_step": args.transition_step,
        "sigma": args.sigma,
        "seed": args.seed,
        "cfg_strength": args.cfg_strength,
        "cfg_video": args.cfg_video,
        "cfg_text": args.cfg_text,
        "num_steps": args.num_steps,
        "variant": args.variant,
        "duration": args.duration,
        "smooth_schedule": args.smooth_schedule,
        "smooth_w_vid_start": args.smooth_w_vid_start,
        "smooth_w_vid_end": args.smooth_w_vid_end,
        "smooth_w_tar_start": args.smooth_w_tar_start,
        "smooth_w_tar_end": args.smooth_w_tar_end,
        "smooth_src_alpha": args.smooth_src_alpha,
        "smooth_src_basis": args.smooth_src_basis,
        "phase2_schedule": args.phase2_schedule,
        "phase2_w_vid_start": args.phase2_w_vid_start,
        "phase2_w_vid_end": args.phase2_w_vid_end,
        "phase2_w_tar": args.phase2_w_tar,
        "phase2_w_src_const": args.phase2_w_src_const,
        "phase2_src_mode": args.phase2_src_mode,
        "phase2_src_vid_multiplier": args.phase2_src_vid_multiplier,
        "target_prompt_bank": args.target_prompt_bank,
        "target_case": args.target_case if getattr(args, 'target_prompt_map', None) else None,
        "results": wrong_results,
    }
    with open(wrong_dir / "metadata.json", "w") as f:
        json.dump(wrong_meta, f, ensure_ascii=False, indent=2)

    return video_id_str


def worker_fn(gpu_id, video_entries, args):
    """Worker function for multi-GPU processing."""
    device = f'cuda:{gpu_id}'
    dtype = torch.bfloat16

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Load model
    model_cfg: ModelConfig = all_model_cfg[args.variant]
    model_cfg.download_if_needed()
    seq_cfg = model_cfg.seq_cfg
    sampling_rate = seq_cfg.sampling_rate

    net: MMAudio = get_my_mmaudio(model_cfg.model_name).to(device, dtype).eval()
    net.load_weights(torch.load(model_cfg.model_path, map_location=device, weights_only=True))

    use_precomputed = args.precomputed_features_dir is not None

    # If using precomputed features, we don't need CLIP/Sync/Tokenizer — only VAE/vocoder.
    feature_utils = FeaturesUtils(
        tod_vae_ckpt=model_cfg.vae_path,
        synchformer_ckpt=None if use_precomputed else model_cfg.synchformer_ckpt,
        enable_conditions=not use_precomputed,
        mode=model_cfg.mode,
        bigvgan_vocoder_ckpt=model_cfg.bigvgan_16k_path,
        need_vae_encoder=False
    ).to(device, dtype).eval()

    text_features_cache = None
    if use_precomputed:
        text_path = Path(args.precomputed_features_dir) / 'text_features.pt'
        if not text_path.exists():
            raise FileNotFoundError(f"text_features.pt not found at {text_path}")
        text_features_cache = torch.load(text_path, map_location='cpu', weights_only=True)
        print(f"[GPU {gpu_id}] Loaded {len(text_features_cache)} precomputed text features.")

    print(f"[GPU {gpu_id}] Model loaded. Processing {len(video_entries)} videos. "
          f"precomputed={use_precomputed}")

    failed = []
    for idx, entry in enumerate(tqdm(video_entries, desc=f"GPU {gpu_id}", position=gpu_id)):
        try:
            result = process_single_video(entry, args, net, feature_utils, seq_cfg, sampling_rate,
                                          device, dtype, text_features_cache=text_features_cache)
            if result is None:
                failed.append(entry['filename'])
        except Exception as e:
            print(f"[GPU {gpu_id}] Error processing {entry['filename']}: {e}")
            failed.append(entry['filename'])

    if failed:
        print(f"[GPU {gpu_id}] Failed: {len(failed)} videos")
    else:
        print(f"[GPU {gpu_id}] All {len(video_entries)} videos completed.")


def main():
    parser = argparse.ArgumentParser(description='VGGSound-Sparse Evaluation')

    # GPU options
    parser.add_argument('--gpu', type=int, default=0, help='GPU device index (single GPU mode)')
    parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs for multi-GPU mode')

    # Output
    parser.add_argument('--output_dir', type=str, default='eval_vggsound_sparse_output',
                        help='Base output directory')
    parser.add_argument('--exp_name', type=str, required=True,
                        help='Experiment name (results saved under output_dir/exp_name/)')

    # Generation method
    parser.add_argument('--sampler', type=str, choices=[
                            SAMPLER_STRICT_2PHASE,
                            SAMPLER_SMOOTH_MEAN_COUPLED,
                            SAMPLER_PHASE2_VIDEO_DECAY,
                        ],
                        default=SAMPLER_STRICT_2PHASE,
                        help='CounterFlow sampler')
    parser.add_argument('--neg_src', action='store_true', default=False,
                        help='Use neg_src: text direction = f(tar) - f(src) in Phase 1')
    parser.add_argument('--neg_src_both', action='store_true', default=False,
                        help='Use neg_src in both Phase 1 and Phase 2')

    # Phase SDE/ODE
    parser.add_argument('--init_ode', type=int, choices=[0, 1], default=1,
                        help='Phase 1: 1=ODE, 0=SDE')
    parser.add_argument('--transition_ode', type=int, choices=[0, 1], default=1,
                        help='Phase 2: 1=ODE, 0=SDE')

    # Sigma and transition step
    parser.add_argument('--sigma', type=float, default=0.0, help='Sigma for SDE')
    parser.add_argument('--transition_step', type=int, default=17, help='Transition step (1~num_steps)')

    # Generation settings
    parser.add_argument('--num_steps', type=int, default=25, help='Number of Euler steps')
    parser.add_argument('--cfg_strength', type=float, default=4.5, help='CFG strength (Phase 2)')
    parser.add_argument('--cfg_video', type=float, default=3.0, help='Decomposed CFG video strength (Phase 1)')
    parser.add_argument('--cfg_text', type=float, default=5.0, help='Decomposed CFG text strength (Phase 1)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--duration', type=float, default=8.0, help='Audio duration')

    # Smooth mean-coupled guidance settings
    parser.add_argument('--smooth_schedule', type=str, choices=['smoothstep', 'cosine', 'linear'],
                        default='smoothstep',
                        help='Monotonic interpolation schedule for smooth mean-coupled guidance')
    parser.add_argument('--smooth_w_vid_start', type=float, default=3.5,
                        help='Initial video guidance weight for smooth mean-coupled guidance')
    parser.add_argument('--smooth_w_vid_end', type=float, default=1.2,
                        help='Final video guidance weight for smooth mean-coupled guidance')
    parser.add_argument('--smooth_w_tar_start', type=float, default=1.8,
                        help='Initial target text guidance weight for smooth mean-coupled guidance')
    parser.add_argument('--smooth_w_tar_end', type=float, default=5.0,
                        help='Final target text guidance weight for smooth mean-coupled guidance')
    parser.add_argument('--smooth_src_alpha', type=float, default=1.0,
                        help='Source-negative multiplier applied to --smooth_src_basis')
    parser.add_argument('--smooth_src_basis', type=str, choices=['mean', 'w_vid'], default='mean',
                        help='Basis for source-negative smooth guidance weight')

    # Phase 2 video decay guidance settings
    parser.add_argument('--phase2_schedule', type=str, choices=['smoothstep', 'cosine', 'linear'],
                        default='smoothstep',
                        help='Phase-2-only interpolation schedule for phase2_video_decay')
    parser.add_argument('--phase2_w_vid_start', type=float, default=3.0,
                        help='Video guidance at the first Phase 2 step')
    parser.add_argument('--phase2_w_vid_end', type=float, default=0.5,
                        help='Video guidance at the final Phase 2 step')
    parser.add_argument('--phase2_w_tar', type=float, default=4.5,
                        help='Target text guidance during Phase 2')
    parser.add_argument('--phase2_w_src_const', type=float, default=None,
                        help='Constant source-negative weight for phase2_src_mode=const. '
                             'Defaults to --phase2_w_tar.')
    parser.add_argument('--phase2_src_mode', type=str, choices=['const', '1p5_vid'], default='const',
                        help='Source-negative Phase 2 schedule: const or multiplier * w_vid')
    parser.add_argument('--phase2_src_vid_multiplier', type=float, default=1.5,
                        help='Multiplier for source-negative weight when phase2_src_mode=1p5_vid')

    # Model
    parser.add_argument('--variant', type=str, default='large_44k_v2',
                        help='Model variant: small_16k, small_44k, medium_44k, large_44k, large_44k_v2')

    # Dataset subset
    parser.add_argument('--subset', type=str, choices=['all', 'clean'], default='all',
                        help='Evaluation subset: all=test split, clean=filter with clean subset CSV')
    parser.add_argument('--clean_csv_path', type=str, default=str(CLEAN_CSV_PATH),
                        help='Path to vggsound_sparse_clean_fixed_offset.csv (used with --subset clean)')

    # Target prompt bank
    parser.add_argument('--target_prompt_bank', type=str, default=None,
                        help='Optional prompt_bank_raw.json with per-source target prompts. '
                             'When set, only the selected --target_case prompts are generated.')
    parser.add_argument('--target_case', type=str, default='semantic_replace_temporal_preserve',
                        help='Prompt-bank case to use, e.g. semantic_replace_temporal_preserve '
                             'or semantic_replace_temporal_shift')

    # Batching for wrong-category generation
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for prompt-switch generation of wrong categories. '
                             '1=sequential (matches bs=1 reproducibility). Max useful=11 (all '
                             'wrong cats per video). Note: RNG noise is drawn batched, so '
                             'batch_size>1 is deterministic but not bit-identical to bs=1.')

    # Precomputed features
    parser.add_argument('--precomputed_features_dir', type=str, default=None,
                        help='Directory with precomputed features (from '
                             'precompute_vggsound_sparse_features.py). If set, CLIP/Sync/text '
                             'encoders are skipped.')

    # Pilot mode
    parser.add_argument('--pilot', action='store_true', default=False,
                        help='Pilot mode: randomly sample 30 videos')
    parser.add_argument('--pilot_n', type=int, default=30,
                        help='Number of videos in pilot mode')

    args = parser.parse_args()

    # Convert int to bool for ODE flags
    args.init_ode = bool(args.init_ode)
    args.transition_ode = bool(args.transition_ode)

    # neg_src_both implies neg_src
    if args.neg_src_both:
        args.neg_src = True
    if args.sampler == SAMPLER_SMOOTH_MEAN_COUPLED:
        if args.sigma != 0.0 or not args.init_ode or not args.transition_ode:
            parser.error(
                "smooth_mean_coupled currently supports deterministic ODE only; "
                "use --sigma 0.0 --init_ode 1 --transition_ode 1"
            )
    if args.sampler == SAMPLER_PHASE2_VIDEO_DECAY:
        if args.sigma != 0.0 or not args.init_ode or not args.transition_ode:
            parser.error(
                "phase2_video_decay currently supports deterministic ODE only; "
                "use --sigma 0.0 --init_ode 1 --transition_ode 1"
            )
        args.neg_src_both = True
        args.neg_src = True

    try:
        args.target_prompt_map = load_target_prompt_map(args.target_prompt_bank, args.target_case)
    except (FileNotFoundError, ValueError) as e:
        parser.error(str(e))

    # Resolve output path: output_dir/exp_name
    args.output_dir = str(Path(args.output_dir) / args.exp_name)

    # Load test data
    try:
        test_data = load_test_data(subset=args.subset, clean_csv_path=Path(args.clean_csv_path))
    except (FileNotFoundError, ValueError) as e:
        parser.error(str(e))
    if args.subset == 'clean':
        print(f"Loaded {len(test_data)} clean test videos from VGGSound-Sparse")
    else:
        print(f"Loaded {len(test_data)} test videos from VGGSound-Sparse")

    if args.target_prompt_map:
        test_categories = {entry['category'] for entry in test_data}
        covered_categories = set(args.target_prompt_map)
        selected_categories = sorted(test_categories & covered_categories)
        missing_categories = sorted(test_categories - covered_categories)
        if not selected_categories:
            parser.error("Target prompt bank does not cover any source category in this split.")
        if missing_categories:
            original_count = len(test_data)
            test_data = [entry for entry in test_data if entry['category'] in covered_categories]
            print(
                "Filtered to prompt-bank source categories: "
                f"kept {len(test_data)} / {original_count} videos for "
                f"{len(selected_categories)} sources; skipped {len(missing_categories)} uncovered sources."
            )
        print(f"Loaded custom target prompts: {len(args.target_prompt_map)} sources, case={args.target_case}")

    # Pilot mode: random subset
    if args.pilot:
        random.seed(args.seed)
        test_data = random.sample(test_data, min(args.pilot_n, len(test_data)))
        print(f"Pilot mode: selected {len(test_data)} videos")

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    config = build_experiment_config(args, len(test_data))
    config_path = Path(args.output_dir) / "experiment_config.json"
    if config_path.exists():
        with open(config_path) as f:
            existing_config = json.load(f)
        mismatches = find_config_mismatches(existing_config, config)
        if mismatches:
            parser.error(
                "Existing experiment_config.json does not match current args. "
                f"Use a new --exp_name. Mismatched keys: {json.dumps(mismatches, ensure_ascii=False)}"
            )
        print(f"Resume mode: matched existing config at {config_path}")
    else:
        with open(config_path, "w") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
    weight_schedule_path = Path(args.output_dir) / "weight_schedule.json"
    if not weight_schedule_path.exists():
        with open(weight_schedule_path, "w") as f:
            json.dump(build_weight_schedule(args), f, ensure_ascii=False, indent=2)

    pending_test_data, completed_count = filter_pending_videos(test_data, Path(args.output_dir), args)
    if completed_count > 0:
        print(f"Resume mode: {completed_count}/{len(test_data)} videos already completed.")
    if not pending_test_data:
        print(f"All tasks already completed for exp_name='{args.exp_name}'. Nothing to do.")
        return
    print(f"Resume mode: processing {len(pending_test_data)} remaining videos.")
    test_data = pending_test_data

    if args.num_gpus > 1:
        # Multi-GPU: split data across GPUs
        mp.set_start_method('spawn', force=True)
        chunks = [[] for _ in range(args.num_gpus)]
        for i, entry in enumerate(test_data):
            chunks[i % args.num_gpus].append(entry)

        processes = []
        for gpu_id in range(args.num_gpus):
            if len(chunks[gpu_id]) == 0:
                continue
            p = mp.Process(target=worker_fn, args=(gpu_id, chunks[gpu_id], args))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        # Single GPU
        worker_fn(args.gpu, test_data, args)

    print(f"\nDone! Results saved to {args.output_dir}")


if __name__ == '__main__':
    main()
