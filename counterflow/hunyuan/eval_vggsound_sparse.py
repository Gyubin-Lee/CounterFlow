#!/usr/bin/env python3
"""
Required environment:
    conda activate Hunyuan

VGGSound-Sparse Evaluation Script for HunyuanVideo-Foley

For each test video (600 total), generate audio with all 12 categories:
  - Correct category: standard HunyuanVideo-Foley (Video + true category)
  - Wrong categories (11): Latent intervention prompt switch (source=correct, target=wrong)

Output structure (identical to MMAudio/eval_vggsound_sparse.py):
  <output_dir>/<exp_name>/
    <video_id>_<start_sec>/
      correct/
        metadata.json
        <video_id>_correct.wav
        <video_id>_correct.mp4
      wrong/
        metadata.json
        <video_id>_<category>.wav
        <video_id>_<category>.mp4

Usage:
    python eval_vggsound_sparse.py --exp_name neg_src_sde_sigma2 \
        --model_path /path/to/model --gpu_id 0

    # Pilot mode (30 random videos):
    python eval_vggsound_sparse.py --exp_name test_pilot --pilot \
        --model_path /path/to/model

    # Clean subset only:
    python eval_vggsound_sparse.py --exp_name clean_run --subset clean \
        --model_path /path/to/model
"""

import argparse
import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torchaudio
from loguru import logger
from tqdm import tqdm

from infer_latent_intervention import (
    set_manual_seed,
    setup_device,
    feature_process_intervention,
    denoise_process_intervention,
)
from hunyuanvideo_foley.utils.model_utils import load_model, denoise_process
from hunyuanvideo_foley.utils.config_utils import AttributeDict
from hunyuanvideo_foley.utils.feature_utils import feature_process
from hunyuanvideo_foley.utils.media_utils import merge_audio_video


def default_model_path() -> str:
    """Resolve a public-safe default for the HunyuanVideo-Foley repository."""
    script_dir = Path(__file__).resolve().parent
    if (script_dir / "hunyuanvideo_foley").exists():
        return str(script_dir)

    project_root = script_dir.parents[1]
    return str(project_root / "external" / "HunyuanVideo-Foley")


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

DEFAULT_NEGATIVE_PROMPT = ""
CSV_PATH = Path('VGGSound-Sparse/vggsound_sparse.csv')
CLEAN_CSV_PATH = Path('VGGSound-Sparse/vggsound_sparse_clean_fixed_offsets.csv')
VIDEO_ROOT = Path('/media/daftpunk5/dataset/vggsound/video')


_text_feat_cache = {}


def load_precomputed_features(precomputed_dir, video_id_str, correct_category,
                              target_category, device, neg_prompt=DEFAULT_NEGATIVE_PROMPT):
    """Load pre-computed visual and text features from disk.

    Returns (visual_feats, text_feats, audio_len_in_s) matching the format
    expected by denoise_process / denoise_process_intervention.
    """
    precomputed_dir = Path(precomputed_dir)
    vid_dir = precomputed_dir / video_id_str

    # Visual features
    vf = torch.load(vid_dir / "visual_feats.pt", map_location=device, weights_only=True)
    visual_feats = AttributeDict({
        'siglip2_feat': vf['siglip2_feat'],
        'syncformer_feat': vf['syncformer_feat'],
    })

    # Audio length
    audio_len_in_s = torch.load(vid_dir / "audio_len.pt", weights_only=True)
    if isinstance(audio_len_in_s, torch.Tensor):
        audio_len_in_s = audio_len_in_s.item()

    # Text features (cached in memory after first load)
    cache_key = str(precomputed_dir)
    if cache_key not in _text_feat_cache:
        _text_feat_cache[cache_key] = torch.load(
            precomputed_dir / "text_feats.pt", map_location=device, weights_only=True
        )
    text_feat_dict = _text_feat_cache[cache_key]

    if target_category is None:
        # Standard inference: text_feat + uncond_text_feat
        text_feats = AttributeDict({
            'text_feat': text_feat_dict[correct_category],
            'uncond_text_feat': text_feat_dict[neg_prompt],
        })
    else:
        # Latent intervention: src + tar + uncond
        text_feats = AttributeDict({
            'uncond_text_feat': text_feat_dict[neg_prompt],
            'tar_text_feat': text_feat_dict[target_category],
            'src_text_feat': text_feat_dict[correct_category],
        })

    return visual_feats, text_feats, audio_len_in_s


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
        logger.info(
            f"Applied clean subset filter: kept {len(rows)} / {original_len} "
            f"test videos from {clean_csv_path}"
        )

    return rows


def process_single_video(video_entry, args, model_dict, cfg):
    """Process a single video: generate correct + 11 wrong category audios."""
    video_id = video_entry['video_id']
    start_sec = video_entry['start_sec']
    correct_category = video_entry['category']
    filename = video_entry['filename']
    video_path = VIDEO_ROOT / filename

    if not video_path.exists():
        logger.warning(f"[SKIP] {filename} not found")
        return None

    video_id_str = f"{video_id}_{start_sec:06d}"
    video_out_dir = Path(args.output_dir) / video_id_str
    correct_dir = video_out_dir / "correct"
    wrong_dir = video_out_dir / "wrong"
    correct_dir.mkdir(parents=True, exist_ok=True)
    wrong_dir.mkdir(parents=True, exist_ok=True)

    # Resolve device and neg_prompt for precomputed feature loading
    precomputed_dir = getattr(args, 'precomputed_dir', None)
    neg_prompt = args.neg_prompt if args.neg_prompt is not None else DEFAULT_NEGATIVE_PROMPT

    # --- Correct category: standard HunyuanVideo-Foley ---
    correct_audio_path = correct_dir / f"{video_id_str}_correct.wav"
    correct_video_path = correct_dir / f"{video_id_str}_correct.mp4"

    if correct_audio_path.exists() and correct_video_path.exists():
        logger.info(f"  Correct already exists, skipping")
    else:
        logger.info(f"  Generating correct (standard inference): {correct_category}")
        set_manual_seed(args.seed)
        if precomputed_dir:
            device = next(model_dict.foley_model.parameters()).device
            visual_feats, text_feats, audio_len_in_s = load_precomputed_features(
                precomputed_dir, video_id_str, correct_category,
                target_category=None, device=device, neg_prompt=neg_prompt,
            )
        else:
            visual_feats, text_feats, audio_len_in_s = feature_process(
                str(video_path), correct_category, model_dict, cfg,
                neg_prompt=args.neg_prompt,
            )
        set_manual_seed(args.seed)
        audio, sr = denoise_process(
            visual_feats, text_feats, audio_len_in_s,
            model_dict, cfg,
            guidance_scale=args.w_cfg,
            num_inference_steps=args.num_inference_steps,
        )
        torchaudio.save(str(correct_audio_path), audio[0].cpu(), sr)
        merge_audio_video(str(correct_audio_path), str(video_path), str(correct_video_path))
        logger.info(f"  Correct saved: {correct_audio_path}")

    correct_meta = {
        "video_id": video_id,
        "start_sec": start_sec,
        "video_path": str(video_path),
        "correct_category": correct_category,
        "prompt": correct_category,
        "generation_type": "standard",
        "seed": args.seed,
        "w_cfg": args.w_cfg,
        "w_vid": args.w_vid,
        "w_txt": args.w_txt,
        "num_inference_steps": args.num_inference_steps,
        "method": args.method,
        "sde_sigma": args.sde_sigma,
        "phase1_sde": args.phase1_sde,
        "phase2_sde": args.phase2_sde,
        "transition_step": args.transition_step,
    }
    with open(correct_dir / "metadata.json", "w") as f:
        json.dump(correct_meta, f, ensure_ascii=False, indent=2)

    # --- Wrong categories: latent intervention prompt switch ---
    wrong_categories = [c for c in CATEGORIES if c != correct_category]
    wrong_results = []

    for wrong_cat in wrong_categories:
        safe_cat = wrong_cat.replace(" ", "_")
        wrong_audio_path = wrong_dir / f"{video_id_str}_{safe_cat}.wav"
        wrong_video_path = wrong_dir / f"{video_id_str}_{safe_cat}.mp4"

        if wrong_audio_path.exists() and wrong_video_path.exists():
            logger.debug(f"  Wrong [{wrong_cat}] already exists, skipping")
            wrong_results.append({
                "source_prompt": correct_category,
                "target_prompt": wrong_cat,
                "audio_file": wrong_audio_path.name,
                "video_file": wrong_video_path.name,
            })
            continue

        logger.info(f"  Generating wrong: {correct_category} -> {wrong_cat}")

        set_manual_seed(args.seed)
        if precomputed_dir:
            device = next(model_dict.foley_model.parameters()).device
            visual_feats, text_feats, audio_len_in_s = load_precomputed_features(
                precomputed_dir, video_id_str, correct_category,
                target_category=wrong_cat, device=device, neg_prompt=neg_prompt,
            )
        else:
            visual_feats, text_feats, audio_len_in_s = feature_process_intervention(
                str(video_path), correct_category, wrong_cat,
                model_dict, cfg, neg_prompt=args.neg_prompt,
            )

        set_manual_seed(args.seed)
        audio, sr = denoise_process_intervention(
            visual_feats, text_feats, audio_len_in_s,
            model_dict, cfg,
            method=args.method,
            transition_step=args.transition_step,
            sde_sigma=args.sde_sigma,
            phase1_sde=args.phase1_sde,
            phase2_sde=args.phase2_sde,
            w_vid=args.w_vid,
            w_txt=args.w_txt,
            w_cfg=args.w_cfg,
            num_inference_steps=args.num_inference_steps,
        )

        torchaudio.save(str(wrong_audio_path), audio[0].cpu(), sr)
        merge_audio_video(str(wrong_audio_path), str(video_path), str(wrong_video_path))

        wrong_results.append({
            "source_prompt": correct_category,
            "target_prompt": wrong_cat,
            "audio_file": wrong_audio_path.name,
            "video_file": wrong_video_path.name,
        })

    wrong_meta = {
        "video_id": video_id,
        "start_sec": start_sec,
        "video_path": str(video_path),
        "correct_category": correct_category,
        "generation_type": "latent_intervention",
        "method": args.method,
        "sde_sigma": args.sde_sigma,
        "phase1_sde": args.phase1_sde,
        "phase2_sde": args.phase2_sde,
        "transition_step": args.transition_step,
        "w_vid": args.w_vid,
        "w_txt": args.w_txt,
        "w_cfg": args.w_cfg,
        "num_inference_steps": args.num_inference_steps,
        "seed": args.seed,
        "results": wrong_results,
    }
    with open(wrong_dir / "metadata.json", "w") as f:
        json.dump(wrong_meta, f, ensure_ascii=False, indent=2)

    return video_id_str


def parse_args():
    p = argparse.ArgumentParser(
        description='VGGSound-Sparse Evaluation for HunyuanVideo-Foley',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- model ----
    p.add_argument("--model_path", type=str,
                   default=default_model_path(),
                   help="Path to pretrained model directory")
    p.add_argument("--config_path", type=str, default=None,
                   help="Path to config YAML (auto-selected from model_size if omitted)")
    p.add_argument("--model_size", type=str, choices=["xl", "xxl"], default="xxl")

    # ---- output ----
    p.add_argument('--output_dir', type=str, default='eval_vggsound_sparse_output',
                   help='Base output directory')
    p.add_argument('--exp_name', type=str, required=True,
                   help='Experiment name (results saved under output_dir/exp_name/)')

    # ---- latent intervention ----
    p.add_argument("--method", type=int, choices=[1, 2, 3], default=3,
                   help="Method variant (1/2/3)")
    p.add_argument("--transition_step", type=int, default=8,
                   help="Step index where Phase 1 -> Phase 2 transition occurs")
    p.add_argument("--sde_sigma", type=float, default=2.0,
                   help="SDE diffusion coefficient (0 = ODE)")
    p.add_argument("--phase1_sde", action="store_true", default=True,
                   help="Use SDE sampling in Phase 1")
    p.add_argument("--phase1_ode", dest="phase1_sde", action="store_false",
                   help="Use ODE sampling in Phase 1")
    p.add_argument("--phase2_sde", action="store_true", default=True,
                   help="Use SDE sampling in Phase 2")
    p.add_argument("--phase2_ode", dest="phase2_sde", action="store_false",
                   help="Use ODE sampling in Phase 2")
    p.add_argument("--w_vid", type=float, default=3.0,
                   help="Video guidance weight in Decomposed CFG")
    p.add_argument("--w_txt", type=float, default=4.5,
                   help="Text guidance weight in Decomposed CFG (Phase 1)")
    p.add_argument("--w_cfg", type=float, default=4.5,
                   help="CFG weight in Phase 2")
    p.add_argument("--num_inference_steps", type=int, default=25,
                   help="Total number of denoising steps")
    p.add_argument("--neg_prompt", type=str, default=None,
                   help="Negative prompt for unconditional baseline")

    # ---- precomputed features ----
    p.add_argument("--precomputed_dir", type=str, default=None,
                   help="Path to pre-computed features from precompute_features.py. "
                        "When set, skips feature extraction (SigLIP2/Syncformer/CLAP) "
                        "and loads cached features from disk.")

    # ---- device ----
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--gpu_ids", type=int, nargs="+", default=None,
                   help="Multiple GPU IDs for model distribution")
    p.add_argument("--enable_offload", action="store_true",
                   help="Enable model offloading to reduce peak VRAM")

    # ---- dataset subset ----
    p.add_argument('--subset', type=str, choices=['all', 'clean'], default='all',
                   help='Evaluation subset: all=test split, clean=filter with clean CSV')
    p.add_argument('--clean_csv_path', type=str, default=str(CLEAN_CSV_PATH),
                   help='Path to clean subset CSV')

    # ---- pilot mode ----
    p.add_argument('--pilot', action='store_true', default=False,
                   help='Pilot mode: randomly sample a few videos')
    p.add_argument('--pilot_n', type=int, default=30,
                   help='Number of videos in pilot mode')

    # ---- misc ----
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_level", type=str, default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = p.parse_args()

    config_mapping = {"xl": "configs/hunyuanvideo-foley-xl.yaml",
                      "xxl": "configs/hunyuanvideo-foley-xxl.yaml"}
    if not args.config_path:
        args.config_path = config_mapping[args.model_size]

    return args


def main():
    args = parse_args()
    set_manual_seed(args.seed)

    logger.remove()
    logger.add(lambda msg: print(msg, end=""), level=args.log_level)

    # Resolve output path: output_dir/exp_name
    args.output_dir = str(Path(args.output_dir) / args.exp_name)

    # Load test data
    try:
        test_data = load_test_data(subset=args.subset, clean_csv_path=Path(args.clean_csv_path))
    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        return

    logger.info(f"Loaded {len(test_data)} test videos from VGGSound-Sparse (subset={args.subset})")

    # Pilot mode
    if args.pilot:
        random.seed(args.seed)
        test_data = random.sample(test_data, min(args.pilot_n, len(test_data)))
        logger.info(f"Pilot mode: selected {len(test_data)} videos")

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Save experiment config
    config = {
        "exp_name": args.exp_name,
        "model": "HunyuanVideo-Foley",
        "method": args.method,
        "sde_sigma": args.sde_sigma,
        "phase1_sde": args.phase1_sde,
        "phase2_sde": args.phase2_sde,
        "transition_step": args.transition_step,
        "w_vid": args.w_vid,
        "w_txt": args.w_txt,
        "w_cfg": args.w_cfg,
        "num_inference_steps": args.num_inference_steps,
        "seed": args.seed,
        "subset": args.subset,
        "clean_csv_path": args.clean_csv_path if args.subset == 'clean' else None,
        "pilot": args.pilot,
        "pilot_n": args.pilot_n if args.pilot else None,
        "num_videos": len(test_data),
        "categories": CATEGORIES,
    }
    with open(Path(args.output_dir) / "experiment_config.json", "w") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    # Setup device and load model
    device = setup_device("auto", args.gpu_id)
    if args.gpu_ids:
        logger.info(f"Multi-GPU mode: gpu_ids={args.gpu_ids}")
    logger.info("Loading models...")
    model_dict, cfg = load_model(
        args.model_path, args.config_path, device,
        enable_offload=args.enable_offload,
        model_size=args.model_size,
        gpu_ids=args.gpu_ids,
    )

    # Process all videos
    failed = []
    for idx, entry in enumerate(tqdm(test_data, desc="VGGSound-Sparse")):
        logger.info(f"\n[{idx+1}/{len(test_data)}] {entry['filename']} | {entry['category']}")
        try:
            result = process_single_video(entry, args, model_dict, cfg)
            if result is None:
                failed.append(entry['filename'])
        except Exception as e:
            logger.error(f"Error processing {entry['filename']}: {e}")
            failed.append(entry['filename'])

    if failed:
        logger.warning(f"Failed: {len(failed)} videos")
        with open(Path(args.output_dir) / "failed.json", "w") as f:
            json.dump(failed, f, indent=2)
    else:
        logger.info(f"All {len(test_data)} videos completed successfully.")

    logger.info(f"\nDone! Results saved to {args.output_dir}")


if __name__ == '__main__':
    main()
