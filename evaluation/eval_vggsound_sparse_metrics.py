"""
Required environment:
    conda activate MMAudio

Evaluate VGGSound-Sparse experiment results: CLAP Score, CLAP Classifier Acc,
ImageBind video-audio score, DeSync, ISC, FAD.

Shared evaluation script for multiple models (MMAudio, CAFA, etc.).
Each model's output folder
(e.g., results/research_axes/latent_update_method/evaluation/VGGSound-Sparse/qualitative/<exp_name>)
is passed via --output_dir.

Usage:
    # Evaluate MMAudio results
    python evaluation/eval_vggsound_sparse_metrics.py \
        --output_dir results/research_axes/latent_update_method/evaluation/VGGSound-Sparse/qualitative/neg_src_sde_sigma2 \
        --gpu 0

    # Evaluate CAFA results
    python evaluation/eval_vggsound_sparse_metrics.py \
        --output_dir baselines/CAFA/eval_vggsound_sparse_output/some_exp \
        --gpu 0

    # Skip DeSync (faster, CLAP only):
    python evaluation/eval_vggsound_sparse_metrics.py \
        --output_dir results/research_axes/latent_update_method/evaluation/VGGSound-Sparse/qualitative/neg_src_sde_sigma2 \
        --skip_desync

Metrics:
    - Correct:
        - CLAP score (generated audio <-> correct category text)
        - DeSync (video <-> generated audio)
    - Wrong:
        - CLAP score (generated audio <-> target/wrong text)
        - CLAP classifier acc (CLAP(audio, target) > CLAP(audio, source) -> 1)
        - ImageBind score (input video <-> generated audio). This is a raw
          compatibility score: higher is useful for source/correct audio, while
          lower is generally better for counterfactual wrong audio when target
          evidence is strong.
        - DeSync (video <-> generated audio)
        - ISC (Inception Score via PANNs logits)
        - FAD (Fréchet Audio Distance via PANNs 2048-dim embeddings)
"""

import argparse
import json
import logging
import math
import os
import re
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
import torchaudio
from tqdm import tqdm

log = logging.getLogger(__name__)

# Shared resources are resolved from the project root so evaluation code stays
# separate from external repositories, datasets, and generated outputs.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
AV_BENCHMARK_DIR = PROJECT_ROOT / 'external' / 'av-benchmark'
VGGSOUND_SPARSE_DIR = PROJECT_ROOT / 'datasets' / 'VGGSound-Sparse'
DEFAULT_QUANT_RESULTS_DIR = (
    PROJECT_ROOT
    / 'results'
    / 'research_axes'
    / 'evaluation_metrics'
    / 'evaluation'
    / 'VGGSound-Sparse'
    / 'quantitative'
)
VIDEO_ROOT = Path('/media/daftpunk5/dataset/vggsound/video')



# ==================== Audio extraction ====================

def format_duration_tag(duration: float) -> str:
    """Format duration as a compact filename-friendly tag like 5s or 5p5s."""
    if math.isclose(duration, round(duration), rel_tol=0.0, abs_tol=1e-9):
        return f'{int(round(duration))}s'
    return f'{duration:g}'.replace('.', 'p') + 's'


def resolve_default_metric_cache(base_dir: Path, stem: str, duration: float) -> Path:
    """Use duration-specific caches when present, otherwise fall back to legacy paths."""
    legacy_path = base_dir / f'{stem}.pt'
    if duration <= 0:
        return legacy_path

    duration_path = base_dir / f'{stem}_{format_duration_tag(duration)}.pt'
    if duration_path.exists():
        return duration_path

    return legacy_path

def extract_audio_from_file(audio_path: Path, sr: int = 48000, duration: float = 0.0) -> torch.Tensor:
    """Extract audio from wav or mp4 file and return mono waveform tensor.

    Args:
        audio_path: path to audio file
        sr: target sample rate
        duration: if > 0, truncate to this many seconds from the start.
                  0 means use the full audio.
    """
    try:
        waveform, sample_rate = torchaudio.load(str(audio_path))
    except Exception:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name
        try:
            subprocess.run(
                ['ffmpeg', '-y', '-i', str(audio_path), '-vn', '-acodec', 'pcm_s16le',
                 '-ar', str(sr), '-ac', '1', tmp_path],
                capture_output=True, check=True
            )
            waveform, sample_rate = torchaudio.load(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    waveform = waveform.mean(dim=0)
    waveform = waveform - waveform.mean()

    if sample_rate != sr:
        resampler = torchaudio.transforms.Resample(sample_rate, sr)
        waveform = resampler(waveform)

    if duration > 0:
        max_samples = int(sr * duration)
        waveform = waveform[:max_samples]

    return waveform


# ==================== Discover experiment results ====================

def discover_experiment(exp_dir: Path) -> Dict:
    """
    Discover all generated files in an experiment directory.

    Returns:
        dict with:
        - correct_entries: list of {video_id, video_path, audio_wav, audio_mp4, prompt, category}
        - wrong_entries: list of {video_id, video_path, audio_wav, audio_mp4, source_prompt, target_prompt}
    """
    correct_entries = []
    wrong_entries = []

    video_dirs = sorted([d for d in exp_dir.iterdir() if d.is_dir()])

    for video_dir in video_dirs:
        correct_dir = video_dir / "correct"
        wrong_dir = video_dir / "wrong"

        # --- Correct ---
        correct_meta_path = correct_dir / "metadata.json"
        if correct_meta_path.exists():
            with open(correct_meta_path) as f:
                correct_meta = json.load(f)

            video_path = Path(correct_meta['video_path'])
            category = correct_meta['correct_category']
            prompt = correct_meta.get('prompt', category)

            # Find wav and mp4 files
            correct_wavs = list(correct_dir.glob('*.wav'))
            correct_mp4s = list(correct_dir.glob('*.mp4'))

            if correct_wavs and correct_mp4s:
                correct_entries.append({
                    'video_id': video_dir.name,
                    'video_path': video_path,
                    'audio_wav': correct_wavs[0],
                    'audio_mp4': correct_mp4s[0],
                    'prompt': prompt,
                    'category': category,
                })

        # --- Wrong ---
        wrong_meta_path = wrong_dir / "metadata.json"
        if wrong_meta_path.exists():
            with open(wrong_meta_path) as f:
                wrong_meta = json.load(f)

            video_path = Path(wrong_meta['video_path'])
            correct_category = wrong_meta['correct_category']

            for result in wrong_meta.get('results', []):
                audio_wav = wrong_dir / result['audio_file']
                audio_mp4 = wrong_dir / result['video_file']

                if audio_wav.exists() and audio_mp4.exists():
                    entry = {
                        'video_id': video_dir.name,
                        'video_path': video_path,
                        'audio_wav': audio_wav,
                        'audio_mp4': audio_mp4,
                        'source_prompt': result['source_prompt'],
                        'target_prompt': result['target_prompt'],
                        'correct_category': correct_category,
                    }
                    for optional_key in (
                        'target_field',
                        'target_level',
                        'conflict_level',
                        'target_case',
                        'source_field',
                    ):
                        if optional_key in result:
                            entry[optional_key] = result[optional_key]
                    wrong_entries.append(entry)

    return {
        'correct_entries': correct_entries,
        'wrong_entries': wrong_entries,
    }


def _parse_sparse_csv_video_id(video_id: str) -> Optional[Tuple[str, int]]:
    """
    Parse VGGSound-Sparse CSV video_id format:
        <youtube_id>_<start_ms>_<end_ms>
    and return:
        (<youtube_id>, start_sec)
    """
    stem = Path(str(video_id)).stem
    parts = stem.rsplit('_', 2)
    if len(parts) != 3:
        return None

    yt_id, start_ms, _ = parts
    if not start_ms.isdigit():
        return None

    return yt_id, int(start_ms) // 1000


def _parse_result_dir_video_id(video_id: str) -> Optional[Tuple[str, int]]:
    """
    Parse result folder video_id format:
        <youtube_id>_<start_sec_zero_padded>
    and return:
        (<youtube_id>, start_sec)
    """
    stem = Path(str(video_id)).stem
    parts = stem.rsplit('_', 1)
    if len(parts) != 2:
        return None

    yt_id, start_sec = parts
    if not start_sec.isdigit():
        return None

    return yt_id, int(start_sec)


def load_video_id_filter_from_csv(csv_path: Path, exp_dir: Path) -> Set[str]:
    """
    Load video ids from CSV and map them to actual result folder names.

    CSV usually stores ids as: <youtube_id>_<start_ms>_<end_ms>
    Result folders are usually: <youtube_id>_<start_sec_zero_padded>
    (e.g. -O4W3NA5-uA_150000_160000 -> -O4W3NA5-uA_000150)
    """
    if not csv_path.exists():
        raise FileNotFoundError(f'CSV file not found: {csv_path}')

    df = pd.read_csv(csv_path)
    if 'video_id' not in df.columns:
        raise ValueError(f"CSV must contain 'video_id' column: {csv_path}")

    existing_video_dirs = {d.name for d in exp_dir.iterdir() if d.is_dir()}
    tuple_to_result_id = {}
    for result_id in existing_video_dirs:
        parsed = _parse_result_dir_video_id(result_id)
        if parsed is not None:
            tuple_to_result_id[parsed] = result_id

    matched_result_ids = set()
    unmatched_csv_ids = []

    for raw_id in df['video_id'].dropna().astype(str):
        csv_id = Path(raw_id).stem

        if csv_id in existing_video_dirs:
            matched_result_ids.add(csv_id)
            continue

        parsed_sparse = _parse_sparse_csv_video_id(csv_id)
        if parsed_sparse is not None and parsed_sparse in tuple_to_result_id:
            matched_result_ids.add(tuple_to_result_id[parsed_sparse])
            continue

        parsed_result = _parse_result_dir_video_id(csv_id)
        if parsed_result is not None and parsed_result in tuple_to_result_id:
            matched_result_ids.add(tuple_to_result_id[parsed_result])
            continue

        unmatched_csv_ids.append(csv_id)

    log.info(f'CSV filter: matched {len(matched_result_ids)} / {len(df)} rows to result folders')
    if unmatched_csv_ids:
        preview = ', '.join(unmatched_csv_ids[:10])
        suffix = ' ...' if len(unmatched_csv_ids) > 10 else ''
        log.warning(
            f'CSV filter: {len(unmatched_csv_ids)} IDs could not be matched to result folders: '
            f'{preview}{suffix}'
        )

    return matched_result_ids


# ==================== CLAP Score Computation ====================

@torch.inference_mode()
def compute_clap_scores(
    audio_files: List[Path],
    prompts: Dict[str, str],
    device: str = 'cuda',
    batch_size: int = 16,
    audio_duration: float = 0.0,
) -> Dict[str, float]:
    """
    Compute LAION CLAP cosine similarity between audio files and text prompts.

    Args:
        audio_files: list of audio file paths (wav or mp4)
        prompts: {str(audio_path): prompt_text}

    Returns:
        {str(audio_path): clap_score}
    """
    import laion_clap

    _clap_ckpt_path = AV_BENCHMARK_DIR / 'weights' / 'music_speech_audioset_epoch_15_esc_89.98.pt'

    log.info('Loading LAION CLAP model...')
    clap_model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')
    clap_model.load_ckpt(str(_clap_ckpt_path), verbose=False)
    clap_model = clap_model.to(device).eval()

    # Extract audio embeddings
    log.info(f'Extracting CLAP audio features for {len(audio_files)} files...')
    audio_embeddings = {}
    for i in tqdm(range(0, len(audio_files), batch_size), desc='CLAP audio'):
        batch_files = audio_files[i:i + batch_size]
        wavs = []
        keys = []
        for f in batch_files:
            wav = extract_audio_from_file(f, sr=48000, duration=audio_duration)
            wavs.append(wav)
            keys.append(str(f))
        max_len = max(w.shape[0] for w in wavs)
        wavs_padded = torch.stack([
            torch.nn.functional.pad(w, (0, max_len - w.shape[0])) for w in wavs
        ]).to(device)
        emb = clap_model.get_audio_embedding_from_data(wavs_padded, use_tensor=True).cpu()
        for j, key in enumerate(keys):
            audio_embeddings[key] = emb[j].squeeze()

    # Extract text embeddings
    log.info('Extracting CLAP text features...')
    unique_prompts = list(set(prompts.values()))
    text_embeddings = {}
    for i in range(0, len(unique_prompts), batch_size):
        batch_texts = unique_prompts[i:i + batch_size]
        text_emb = clap_model.get_text_embedding(batch_texts, use_tensor=True).cpu()
        for j, text in enumerate(batch_texts):
            text_embeddings[text] = text_emb[j].squeeze()

    # Compute similarities
    results = {}
    for key, prompt in prompts.items():
        if key in audio_embeddings and prompt in text_embeddings:
            sim = torch.cosine_similarity(
                audio_embeddings[key].unsqueeze(0),
                text_embeddings[prompt].unsqueeze(0),
                dim=-1
            ).item()
            results[key] = sim

    del clap_model
    torch.cuda.empty_cache()

    return results


# ==================== DeSync Computation ====================

def _pad_feature_segments(x: torch.Tensor, target_segments: int = 14) -> torch.Tensor:
    """Pad a short Synchformer feature sequence by repeating its last segment."""
    if x.shape[0] >= target_segments:
        return x[:target_segments]
    if x.shape[0] == 0:
        raise ValueError('Not enough segments (0)')
    pad_shape = (target_segments - x.shape[0],) + tuple(x.shape[1:])
    pad = x[-1:].expand(pad_shape)
    return torch.cat([x, pad], dim=0)


@torch.inference_mode()
def compute_desync_window_score(
    sync_model,
    sync_grid: torch.Tensor,
    video_feat: torch.Tensor,
    audio_feat: torch.Tensor,
    device: str,
) -> Tuple[float, int]:
    """Score DeSync with available clip length.

    Synchformer compares 14-segment windows. For AVSBench-S4 clips shorter
    than the nominal metric duration, use the available aligned segments and
    repeat the last feature segment only to satisfy the 14-segment comparison
    shape. This avoids padding raw audio/video to 8 seconds while still keeping
    short clips evaluable.
    """
    min_seg = min(video_feat.shape[0], audio_feat.shape[0])
    if min_seg <= 0:
        raise ValueError(f'Not enough segments ({min_seg})')

    v = video_feat[:min_seg].to(device)
    a = audio_feat[:min_seg].to(device)

    first_v = _pad_feature_segments(v[:14]).unsqueeze(0)
    first_a = _pad_feature_segments(a[:14]).unsqueeze(0)
    logits = sync_model.compare_v_a(first_v, first_a)
    top_id = torch.argmax(logits, dim=-1).cpu().item()
    first_score = abs(sync_grid[top_id].item())

    if min_seg < 28:
        return first_score, 1

    last_v = _pad_feature_segments(v[-14:]).unsqueeze(0)
    last_a = _pad_feature_segments(a[-14:]).unsqueeze(0)
    logits = sync_model.compare_v_a(last_v, last_a)
    top_id = torch.argmax(logits, dim=-1).cpu().item()
    last_score = abs(sync_grid[top_id].item())
    return float(np.mean([first_score, last_score])), 2


def _load_video_dataset_sample_for_desync(video_path: Path, duration: float):
    """Load a video sample, retrying with decoded short-clip length if needed."""
    sys.path.insert(0, str(AV_BENCHMARK_DIR))
    from av_bench.data.video_dataset import VideoDataset

    dataset = VideoDataset([Path(video_path)], duration_sec=duration)
    try:
        return dataset.sample(0), duration
    except RuntimeError as e:
        match = re.search(r'Sync video too short .* expected (\d+), got (\d+)', str(e))
        if not match:
            raise

        decoded_frames = int(match.group(2))
        if decoded_frames <= 0:
            raise

        fallback_duration = decoded_frames / 25.0
        if fallback_duration >= duration:
            raise

        log.warning(
            'DeSync video is shorter than requested for %s: %.2fs requested, '
            '%.2fs decoded. Retrying with decoded length.',
            video_path,
            duration,
            fallback_duration,
        )
        return VideoDataset([Path(video_path)], duration_sec=fallback_duration).sample(0), fallback_duration


@torch.inference_mode()
def extract_video_features_from_file(
    video_path: Path,
    sync_model,
    device: str,
    duration: float = 8.0,
) -> torch.Tensor:
    """Extract Synchformer video features, allowing clips shorter than duration."""
    from einops import rearrange

    sample, used_duration = _load_video_dataset_sample_for_desync(video_path, duration)
    if sample is None:
        raise RuntimeError(f'Unable to decode video for DeSync: {video_path}')

    sync_video = sample['sync_video'].unsqueeze(0).to(device)
    b, t, c, h, w = sync_video.shape
    if c != 3 or h != 224 or w != 224:
        raise RuntimeError(
            f'Unexpected Synchformer video tensor shape {tuple(sync_video.shape)} for {video_path}'
        )

    segment_size = 16
    step_size = 8
    num_segments = (t - segment_size) // step_size + 1
    if num_segments <= 0:
        raise RuntimeError(
            f'Not enough video frames for DeSync: {video_path} '
            f'({t} frames from {used_duration:.2f}s)'
        )

    segments = [
        sync_video[:, i * step_size:i * step_size + segment_size]
        for i in range(num_segments)
    ]
    segment_batch = torch.stack(segments, dim=1)
    segment_batch = rearrange(segment_batch, 'b s t c h w -> (b s) 1 t c h w')
    video_feat = sync_model.extract_vfeats(segment_batch)
    video_feat = rearrange(video_feat, '(b s) 1 t d -> b s t d', b=b)
    return video_feat.squeeze(0).cpu()


@torch.inference_mode()
def compute_desync(
    mp4_files: List[Path],
    video_id_map: Dict[str, str],
    device: str = 'cuda',
    video_feature_cache_path: Path = None,
    audio_duration: float = 0.0,
) -> Dict[str, float]:
    """
    Compute DeSync score from generated mp4 files.

    Args:
        mp4_files: generated mp4 files (audio+video combined)
        video_id_map: {str(mp4_path): video_id} — files sharing a video_id
                      have the same video track, so video features are reused.
        video_feature_cache_path: path to pre-extracted video features .pt file.
            If provided, video features are loaded from this file instead of
            being extracted from mp4 files on the fly.

    Returns:
        {str(mp4_path): desync_score}
    """
    sys.path.insert(0, str(AV_BENCHMARK_DIR))
    from av_bench.extract import encode_audio_with_sync
    from av_bench.synchformer.synchformer import Synchformer, make_class_grid

    _syncformer_ckpt_path = AV_BENCHMARK_DIR / 'weights' / 'synchformer_state_dict.pth'

    log.info('Loading Synchformer model...')
    sync_model = Synchformer().to(device).eval()
    sd = torch.load(_syncformer_ckpt_path, weights_only=True)
    sync_model.load_state_dict(sd)

    sync_mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        win_length=400,
        hop_length=160,
        n_fft=1024,
        n_mels=128,
    ).to(device)

    sync_grid = make_class_grid(-2, 2, 21)

    # Load pre-extracted video features if available
    precomputed_video_features = {}
    if video_feature_cache_path and video_feature_cache_path.exists():
        log.info(f'Loading pre-extracted video features from {video_feature_cache_path}...')
        precomputed_video_features = torch.load(video_feature_cache_path, weights_only=True)
        log.info(f'Loaded {len(precomputed_video_features)} pre-extracted video features')

    video_feature_cache = dict(precomputed_video_features)  # video_id -> video_feat tensor
    results = {}
    short_clip_count = 0
    full_window_count = 0
    newly_extracted_video_count = 0

    for mp4_file in tqdm(mp4_files, desc='Computing DeSync'):
        key = str(mp4_file)

        if not mp4_file.exists():
            log.warning(f'mp4 not found: {mp4_file}')
            continue

        vid = video_id_map.get(key)

        # Load video features: prefer pre-extracted cache, then in-memory cache, then extract
        if vid not in video_feature_cache:
            try:
                video_feat = extract_video_features_from_file(
                    mp4_file,
                    sync_model=sync_model,
                    device=device,
                    duration=audio_duration if audio_duration > 0 else 8.0,
                )
                video_feature_cache[vid] = video_feat
                newly_extracted_video_count += 1
                if (video_feature_cache_path and newly_extracted_video_count % 50 == 0):
                    video_feature_cache_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(video_feature_cache, video_feature_cache_path)
            except Exception as e:
                log.error(f'Error extracting video features from {mp4_file}: {e}')
                continue
        video_feat = video_feature_cache[vid]

        # Extract audio (different for each generated mp4)
        try:
            wav = extract_audio_from_file(mp4_file, sr=16000, duration=audio_duration)
        except Exception as e:
            log.error(f'Error extracting audio from {mp4_file}: {e}')
            continue

        wav_tensor = wav.unsqueeze(0).to(device)
        audio_feat = encode_audio_with_sync(sync_model, wav_tensor, sync_mel_spectrogram)
        audio_feat = audio_feat.squeeze(0).cpu()

        try:
            score, window_count = compute_desync_window_score(
                sync_model=sync_model,
                sync_grid=sync_grid,
                video_feat=video_feat,
                audio_feat=audio_feat,
                device=device,
            )
        except ValueError as e:
            log.warning(f'{e} for {mp4_file}, skipping')
            continue

        results[key] = score
        if window_count == 1:
            short_clip_count += 1
        else:
            full_window_count += 1

    if video_feature_cache_path and newly_extracted_video_count > 0:
        video_feature_cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(video_feature_cache, video_feature_cache_path)
        log.info(f'Saved {len(video_feature_cache)} DeSync video features to {video_feature_cache_path}')

    from_cache = sum(1 for vid in video_feature_cache if vid in precomputed_video_features)
    from_extract = len(video_feature_cache) - from_cache
    log.info(f'Video features: {from_cache} from pre-extracted cache, '
             f'{from_extract} extracted on the fly, '
             f'{len(video_feature_cache)} unique videos total')
    log.info(
        'DeSync windows used: %d short-clip single-window, %d full two-window evaluations',
        short_clip_count,
        full_window_count,
    )

    del sync_model
    torch.cuda.empty_cache()

    return results


# ==================== ImageBind Video-Audio Computation ====================

def ensure_torchvision_functional_tensor_compat() -> None:
    """Patch old pytorchvideo imports used by ImageBind/av-benchmark loaders."""
    try:
        import torchvision.transforms.functional as F
        sys.modules.setdefault('torchvision.transforms.functional_tensor', F)
    except Exception as e:
        log.debug('Unable to install torchvision functional_tensor compatibility patch: %s', e)


def encode_video_with_imagebind(imagebind_model, x: torch.Tensor) -> torch.Tensor:
    """Encode av-benchmark ImageBind video tensor.

    Args:
        imagebind_model: loaded ImageBind model.
        x: tensor shaped (B, NUM_CROPS, T, C, H, W).
    """
    from einops import rearrange
    from imagebind.models.imagebind_model import ModalityType

    clips = []
    _, _, t, _, _, _ = x.shape
    for i in range(t - 1):
        clips.append(x[:, :, i:i + 2])
    if not clips:
        raise RuntimeError(f'ImageBind video tensor is too short: shape={tuple(x.shape)}')

    clips = torch.cat(clips, dim=1)
    clips = rearrange(clips, 'b n t c h w -> b n c t h w')
    emb = imagebind_model({ModalityType.VISION: clips})
    return emb[ModalityType.VISION]


@torch.inference_mode()
def extract_imagebind_video_feature(
    video_path: Path,
    imagebind_model,
    device: str,
    duration: float,
) -> torch.Tensor:
    """Extract one ImageBind vision embedding from a video file."""
    ensure_torchvision_functional_tensor_compat()
    sys.path.insert(0, str(AV_BENCHMARK_DIR))
    from av_bench.data.video_dataset import VideoDataset

    dataset = VideoDataset([Path(video_path)], duration_sec=duration)
    sample = dataset[0]
    if sample is None:
        raise RuntimeError(f'Unable to decode video for ImageBind: {video_path}')

    ib_video = sample['ib_video'].unsqueeze(0).to(device)
    feature = encode_video_with_imagebind(imagebind_model, ib_video)
    return feature.squeeze(0).detach().cpu()


@torch.inference_mode()
def extract_imagebind_audio_features(
    audio_files: List[Path],
    imagebind_model,
    device: str,
    batch_size: int,
) -> Dict[str, torch.Tensor]:
    """Extract ImageBind audio embeddings keyed by absolute audio path string."""
    ensure_torchvision_functional_tensor_compat()
    sys.path.insert(0, str(AV_BENCHMARK_DIR))
    from av_bench.data.audio_dataset import ImageBindAudioDataset
    from imagebind.models.imagebind_model import ModalityType

    dataset = ImageBindAudioDataset(audio_files)
    results: Dict[str, torch.Tensor] = {}

    for i in tqdm(range(0, len(audio_files), batch_size), desc='ImageBind audio'):
        batch_files = audio_files[i:i + batch_size]
        wavs = []
        keys = []
        for audio_file in batch_files:
            try:
                wavs.append(dataset.load_and_transform_audio_data(audio_file))
                keys.append(str(audio_file))
            except Exception as e:
                log.error('Error loading audio for ImageBind %s: %s', audio_file, e)

        if not wavs:
            continue

        wav = torch.cat(wavs, dim=0).to(device)
        features = imagebind_model({ModalityType.AUDIO: wav})[ModalityType.AUDIO].detach().cpu()
        for j, key in enumerate(keys):
            results[key] = features[j]

    return results


@torch.inference_mode()
def compute_imagebind_scores(
    audio_video_pairs: List[Tuple[Path, Path, str]],
    device: str = 'cuda',
    batch_size: int = 16,
    duration: float = 8.0,
    video_feature_cache_path: Path | None = None,
) -> Dict[str, float]:
    """Compute ImageBind cosine similarity for generated audio vs source video.

    Args:
        audio_video_pairs: list of (audio_wav, video_path, video_id). Video
            embeddings are cached by video_id and reused across target prompts.
    """
    if not audio_video_pairs:
        return {}

    ensure_torchvision_functional_tensor_compat()
    sys.path.insert(0, str(AV_BENCHMARK_DIR))
    from imagebind.models import imagebind_model

    log.info('Loading ImageBind model...')
    imagebind = imagebind_model.imagebind_huge(pretrained=True).to(device).eval()

    video_features: Dict[str, torch.Tensor] = {}
    if video_feature_cache_path and video_feature_cache_path.exists():
        log.info('Loading ImageBind video feature cache from %s', video_feature_cache_path)
        video_features = torch.load(video_feature_cache_path, map_location='cpu', weights_only=True)
        log.info('Loaded %d cached ImageBind video features', len(video_features))

    video_id_to_path: Dict[str, Path] = {}
    for _, video_path, video_id in audio_video_pairs:
        video_id_to_path.setdefault(video_id, video_path)

    newly_extracted = 0
    for video_id, video_path in tqdm(sorted(video_id_to_path.items()), desc='ImageBind video'):
        if video_id in video_features:
            continue
        try:
            video_features[video_id] = extract_imagebind_video_feature(
                video_path=video_path,
                imagebind_model=imagebind,
                device=device,
                duration=duration,
            )
            newly_extracted += 1
        except Exception as e:
            log.error('Error extracting ImageBind video feature from %s: %s', video_path, e)

        if video_feature_cache_path and newly_extracted > 0 and newly_extracted % 50 == 0:
            video_feature_cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(video_features, video_feature_cache_path)

    if video_feature_cache_path and newly_extracted > 0:
        video_feature_cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(video_features, video_feature_cache_path)
        log.info(
            'Saved %d ImageBind video features to %s',
            len(video_features),
            video_feature_cache_path,
        )

    unique_audio_files = list(dict.fromkeys(audio_file for audio_file, _, _ in audio_video_pairs))
    audio_features = extract_imagebind_audio_features(
        unique_audio_files,
        imagebind_model=imagebind,
        device=device,
        batch_size=batch_size,
    )

    results: Dict[str, float] = {}
    for audio_file, _, video_id in audio_video_pairs:
        audio_key = str(audio_file)
        if audio_key not in audio_features or video_id not in video_features:
            continue
        sim = torch.cosine_similarity(
            audio_features[audio_key].unsqueeze(0),
            video_features[video_id].unsqueeze(0),
            dim=-1,
        ).item()
        results[audio_key] = sim

    del imagebind
    torch.cuda.empty_cache()
    return results


# ==================== PANNs Feature Extraction (ISC + FAD) ====================

@torch.inference_mode()
def extract_panns_features(
    audio_files: List[Path],
    device: str = 'cuda',
    batch_size: int = 16,
    audio_duration: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """
    Extract PANNs (Cnn14) features from audio files.

    Returns:
        dict with 'logits' (N, 527) and '2048' (N, 2048) tensors
    """
    sys.path.insert(0, str(AV_BENCHMARK_DIR))
    from av_bench.panns import Cnn14

    log.info('Loading PANNs (Cnn14) model...')
    panns = Cnn14(
        features_list=["logits", "2048"],
        sample_rate=16000,
        window_size=512,
        hop_size=160,
        mel_bins=64,
        fmin=50,
        fmax=8000,
        classes_num=527,
    )
    panns = panns.to(device).eval()

    all_logits = []
    all_embeds = []
    for i in tqdm(range(0, len(audio_files), batch_size), desc='PANNs features'):
        batch_files = audio_files[i:i + batch_size]
        wavs = []
        for f in batch_files:
            wav = extract_audio_from_file(f, sr=16000, duration=audio_duration)
            target_samples = int(16000 * audio_duration) if audio_duration > 0 else 16000 * 8
            if target_samples <= 0:
                raise ValueError(f'Invalid PANNs target length: {target_samples}')
            if wav.shape[0] > target_samples:
                wav = wav[:target_samples]
            elif wav.shape[0] < target_samples:
                wav = torch.nn.functional.pad(wav, (0, target_samples - wav.shape[0]))
            wavs.append(wav)
        wavs_batch = torch.stack(wavs).float().to(device)  # (B, T)
        features = panns(wavs_batch)
        all_logits.append(features['logits'].cpu())
        all_embeds.append(features['2048'].cpu())

    all_logits = torch.cat(all_logits, dim=0)   # (N, 527)
    all_embeds = torch.cat(all_embeds, dim=0)    # (N, 2048)
    log.info(f'Extracted PANNs features for {all_logits.shape[0]} audio files')

    del panns
    torch.cuda.empty_cache()

    return {'logits': all_logits, '2048': all_embeds}


def compute_inception_score(
    panns_logits: torch.Tensor,
    splits: int = 10,
) -> Dict[str, float]:
    """Compute ISC from pre-extracted PANNs logits."""
    sys.path.insert(0, str(AV_BENCHMARK_DIR))
    from av_bench.metrics.isc import compute_isc

    return compute_isc(
        panns_logits,
        feat_layer_name=None,
        rng_seed=2020,
        samples_shuffle=True,
        splits=splits,
    )


def compute_fad(
    panns_embeds: np.ndarray,
    gt_fad_stats_path: Path,
) -> float:
    """
    Compute FAD between generated audio embeddings and pre-computed GT statistics.

    Args:
        panns_embeds: (N, 2048) numpy array of generated audio embeddings
        gt_fad_stats_path: path to pre-computed GT stats .pt file (mu, sigma)

    Returns:
        FAD score (float)
    """
    sys.path.insert(0, str(AV_BENCHMARK_DIR))
    from av_bench.metrics.fad import calculate_embd_statistics, calculate_frechet_distance

    log.info(f'Loading GT FAD statistics from {gt_fad_stats_path}...')
    gt_stats = torch.load(gt_fad_stats_path, weights_only=False)
    mu_gt, sigma_gt = gt_stats['mu'], gt_stats['sigma']
    log.info(f'GT stats: {gt_stats["num_samples"]} samples, embedding_dim={gt_stats["embedding_dim"]}')

    mu_gen, sigma_gen = calculate_embd_statistics(panns_embeds)

    fad_score = calculate_frechet_distance(mu_gen, sigma_gen, mu_gt, sigma_gt)
    return float(fad_score)


# ==================== Main Evaluation Pipeline ====================

def evaluate(exp_dir: Path, eval_output_dir: Path, device: str, batch_size: int,
             skip_desync: bool, skip_imagebind: bool = False,
             video_feature_cache_path: Path = None,
             imagebind_video_feature_cache_path: Path = None,
             gt_fad_stats_path: Path = None, audio_duration: float = 0.0,
             filter_video_ids: Optional[Set[str]] = None,
             skip_fad: bool = False):
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Discover experiment results
    log.info(f'Discovering results in {exp_dir}...')
    data = discover_experiment(exp_dir)
    correct_entries = data['correct_entries']
    wrong_entries = data['wrong_entries']
    log.info(f'Found {len(correct_entries)} correct entries, {len(wrong_entries)} wrong entries')

    if filter_video_ids is not None:
        before_correct, before_wrong = len(correct_entries), len(wrong_entries)
        correct_entries = [e for e in correct_entries if e['video_id'] in filter_video_ids]
        wrong_entries = [e for e in wrong_entries if e['video_id'] in filter_video_ids]
        log.info(
            f'After CSV filter: {len(correct_entries)}/{before_correct} correct, '
            f'{len(wrong_entries)}/{before_wrong} wrong entries'
        )

    if not correct_entries and not wrong_entries:
        log.error('No entries found!')
        return

    # 2. Collect all audio files and build mappings
    #    For CLAP: we need (audio, text) pairs
    #    For DeSync: we use the generated mp4 directly (video+audio from same file)

    all_audio_files = []        # wav files for CLAP
    all_mp4_files = []          # mp4 files for DeSync
    imagebind_pairs = []        # (wav file, source video file, video_id)
    prompt_map = {}             # key -> prompt (for CLAP score)
    video_id_map = {}           # mp4_key -> video_id (for DeSync video feature caching)

    # For wrong entries: also need source prompt for classifier
    source_prompt_map = {}      # key -> source prompt

    # --- Correct ---
    for entry in correct_entries:
        wav_key = str(entry['audio_wav'])
        mp4_key = str(entry['audio_mp4'])

        all_audio_files.append(entry['audio_wav'])
        prompt_map[wav_key] = entry['prompt']

        all_mp4_files.append(entry['audio_mp4'])
        video_id_map[mp4_key] = entry['video_id']
        imagebind_pairs.append((entry['audio_wav'], entry['video_path'], entry['video_id']))

    # --- Wrong ---
    for entry in wrong_entries:
        wav_key = str(entry['audio_wav'])
        mp4_key = str(entry['audio_mp4'])

        all_audio_files.append(entry['audio_wav'])
        prompt_map[wav_key] = entry['target_prompt']          # CLAP vs target text
        source_prompt_map[wav_key] = entry['source_prompt']   # for classifier

        all_mp4_files.append(entry['audio_mp4'])
        video_id_map[mp4_key] = entry['video_id']
        imagebind_pairs.append((entry['audio_wav'], entry['video_path'], entry['video_id']))

    # 3. Compute CLAP scores (audio vs target/correct text)
    log.info('=' * 60)
    log.info('Computing CLAP scores (audio vs assigned prompt)...')
    log.info('=' * 60)
    clap_target_scores = compute_clap_scores(all_audio_files, prompt_map,
                                             device=device, batch_size=batch_size,
                                             audio_duration=audio_duration)

    # 4. Compute CLAP scores (audio vs source text) for wrong entries only
    #    Needed for classifier accuracy
    wrong_wav_files = [e['audio_wav'] for e in wrong_entries]
    log.info('=' * 60)
    log.info('Computing CLAP scores (wrong audio vs source prompt) for classifier...')
    log.info('=' * 60)
    clap_source_scores = compute_clap_scores(wrong_wav_files, source_prompt_map,
                                             device=device, batch_size=batch_size,
                                             audio_duration=audio_duration)

    # 5. Compute DeSync
    desync_results = {}
    if not skip_desync:
        log.info('=' * 60)
        log.info('Computing DeSync...')
        log.info('=' * 60)
        desync_results = compute_desync(all_mp4_files, video_id_map, device=device,
                                        video_feature_cache_path=video_feature_cache_path,
                                        audio_duration=audio_duration)

    # 6. Compute ImageBind video-audio scores
    imagebind_results = {}
    if not skip_imagebind:
        log.info('=' * 60)
        log.info('Computing ImageBind video-audio scores...')
        log.info('=' * 60)
        imagebind_results = compute_imagebind_scores(
            imagebind_pairs,
            device=device,
            batch_size=batch_size,
            duration=audio_duration if audio_duration > 0 else 8.0,
            video_feature_cache_path=imagebind_video_feature_cache_path,
        )

    # 7. Extract PANNs features for wrong entries (used for both ISC and FAD)
    isc_result = {}
    fad_score = None
    if wrong_wav_files:
        log.info('=' * 60)
        log.info('Extracting PANNs features for ISC & FAD (wrong entries)...')
        log.info('=' * 60)
        panns_features = extract_panns_features(wrong_wav_files, device=device,
                                                batch_size=batch_size,
                                                audio_duration=audio_duration)

        # ISC
        isc_result = compute_inception_score(panns_features['logits'])
        log.info(f'ISC: {isc_result["inception_score_mean"]:.4f} +/- {isc_result["inception_score_std"]:.4f}')

        # FAD
        if skip_fad:
            log.info('Skipping FAD by request')
        elif gt_fad_stats_path and gt_fad_stats_path.exists():
            fad_score = compute_fad(panns_features['2048'].numpy(), gt_fad_stats_path)
            log.info(f'FAD: {fad_score:.4f}')
        elif gt_fad_stats_path:
            log.warning(f'GT FAD stats not found: {gt_fad_stats_path}, skipping FAD')

        del panns_features

    # 8. Build results DataFrames

    # --- Correct results ---
    correct_rows = []
    for entry in correct_entries:
        wav_key = str(entry['audio_wav'])
        mp4_key = str(entry['audio_mp4'])
        row = {
            'video_id': entry['video_id'],
            'category': entry['category'],
            'type': 'correct',
            'prompt': entry['prompt'],
            'clap_score': clap_target_scores.get(wav_key, np.nan),
        }
        if not skip_desync:
            row['desync'] = desync_results.get(mp4_key, np.nan)
        if not skip_imagebind:
            row['imagebind_score'] = imagebind_results.get(wav_key, np.nan)
        correct_rows.append(row)

    df_correct = pd.DataFrame(correct_rows)

    # --- Wrong results ---
    wrong_rows = []
    for entry in wrong_entries:
        wav_key = str(entry['audio_wav'])
        mp4_key = str(entry['audio_mp4'])

        clap_tar = clap_target_scores.get(wav_key, np.nan)
        clap_src = clap_source_scores.get(wav_key, np.nan)

        # Classifier: target > source -> 1 (correct classification)
        if not np.isnan(clap_tar) and not np.isnan(clap_src):
            classifier_correct = int(clap_tar > clap_src)
        else:
            classifier_correct = np.nan

        row = {
            'video_id': entry['video_id'],
            'correct_category': entry['correct_category'],
            'target_category': entry['target_prompt'],
            'type': 'wrong',
            'source_prompt': entry['source_prompt'],
            'target_prompt': entry['target_prompt'],
            'clap_score_target': clap_tar,
            'clap_score_source': clap_src,
            'clap_classifier': classifier_correct,
        }
        for optional_key in ('target_field', 'target_level', 'conflict_level', 'target_case', 'source_field'):
            if optional_key in entry:
                row[optional_key] = entry[optional_key]
        if not skip_desync:
            row['desync'] = desync_results.get(mp4_key, np.nan)
        if not skip_imagebind:
            row['imagebind_score'] = imagebind_results.get(wav_key, np.nan)
        wrong_rows.append(row)

    df_wrong = pd.DataFrame(wrong_rows)

    # 7. Save per-sample results
    df_correct.to_csv(eval_output_dir / 'correct_per_sample.csv', index=False)
    df_wrong.to_csv(eval_output_dir / 'wrong_per_sample.csv', index=False)
    log.info(f'Per-sample results saved to {eval_output_dir}')

    # 8. Compute summaries

    summary = {}

    # --- Correct summary ---
    if not df_correct.empty:
        summary['correct'] = {
            'num_samples': len(df_correct),
            'clap_score_mean': float(df_correct['clap_score'].mean()),
            'clap_score_std': float(df_correct['clap_score'].std()),
        }
        if not skip_desync and 'desync' in df_correct.columns:
            summary['correct']['desync_mean'] = float(df_correct['desync'].mean())
            summary['correct']['desync_std'] = float(df_correct['desync'].std())
        if not skip_imagebind and 'imagebind_score' in df_correct.columns:
            summary['correct']['imagebind_score_mean'] = float(df_correct['imagebind_score'].mean())
            summary['correct']['imagebind_score_std'] = float(df_correct['imagebind_score'].std())

        # Per-category correct summary
        correct_by_cat = df_correct.groupby('category').agg(
            clap_score_mean=('clap_score', 'mean'),
            clap_score_std=('clap_score', 'std'),
            count=('clap_score', 'count'),
        )
        if not skip_desync and 'desync' in df_correct.columns:
            desync_by_cat = df_correct.groupby('category')['desync'].agg(['mean', 'std'])
            correct_by_cat['desync_mean'] = desync_by_cat['mean']
            correct_by_cat['desync_std'] = desync_by_cat['std']
        if not skip_imagebind and 'imagebind_score' in df_correct.columns:
            imagebind_by_cat = df_correct.groupby('category')['imagebind_score'].agg(['mean', 'std'])
            correct_by_cat['imagebind_score_mean'] = imagebind_by_cat['mean']
            correct_by_cat['imagebind_score_std'] = imagebind_by_cat['std']
        correct_by_cat.to_csv(eval_output_dir / 'correct_by_category.csv')

    # --- Wrong summary ---
    if not df_wrong.empty:
        summary['wrong'] = {
            'num_samples': len(df_wrong),
            'clap_score_target_mean': float(df_wrong['clap_score_target'].mean()),
            'clap_score_target_std': float(df_wrong['clap_score_target'].std()),
            'clap_score_source_mean': float(df_wrong['clap_score_source'].mean()),
            'clap_score_source_std': float(df_wrong['clap_score_source'].std()),
            'clap_classifier_acc': float(df_wrong['clap_classifier'].mean()),
        }
        if isc_result:
            summary['wrong']['isc_mean'] = isc_result['inception_score_mean']
            summary['wrong']['isc_std'] = isc_result['inception_score_std']
        if fad_score is not None:
            summary['wrong']['fad'] = fad_score
        if not skip_desync and 'desync' in df_wrong.columns:
            summary['wrong']['desync_mean'] = float(df_wrong['desync'].mean())
            summary['wrong']['desync_std'] = float(df_wrong['desync'].std())
        if not skip_imagebind and 'imagebind_score' in df_wrong.columns:
            summary['wrong']['imagebind_score_mean'] = float(df_wrong['imagebind_score'].mean())
            summary['wrong']['imagebind_score_std'] = float(df_wrong['imagebind_score'].std())

        # Per target-category wrong summary
        wrong_by_target = df_wrong.groupby('target_category').agg(
            clap_score_target_mean=('clap_score_target', 'mean'),
            clap_score_target_std=('clap_score_target', 'std'),
            clap_score_source_mean=('clap_score_source', 'mean'),
            clap_classifier_acc=('clap_classifier', 'mean'),
            count=('clap_score_target', 'count'),
        )
        if not skip_desync and 'desync' in df_wrong.columns:
            desync_by_target = df_wrong.groupby('target_category')['desync'].agg(['mean', 'std'])
            wrong_by_target['desync_mean'] = desync_by_target['mean']
            wrong_by_target['desync_std'] = desync_by_target['std']
        if not skip_imagebind and 'imagebind_score' in df_wrong.columns:
            imagebind_by_target = df_wrong.groupby('target_category')['imagebind_score'].agg(['mean', 'std'])
            wrong_by_target['imagebind_score_mean'] = imagebind_by_target['mean']
            wrong_by_target['imagebind_score_std'] = imagebind_by_target['std']
        wrong_by_target.to_csv(eval_output_dir / 'wrong_by_target_category.csv')

        if 'target_field' in df_wrong.columns:
            wrong_by_target_field = df_wrong.groupby('target_field').agg(
                clap_score_target_mean=('clap_score_target', 'mean'),
                clap_score_target_std=('clap_score_target', 'std'),
                clap_score_source_mean=('clap_score_source', 'mean'),
                clap_classifier_acc=('clap_classifier', 'mean'),
                count=('clap_score_target', 'count'),
            )
            if not skip_desync and 'desync' in df_wrong.columns:
                desync_by_field = df_wrong.groupby('target_field')['desync'].agg(['mean', 'std'])
                wrong_by_target_field['desync_mean'] = desync_by_field['mean']
                wrong_by_target_field['desync_std'] = desync_by_field['std']
            if not skip_imagebind and 'imagebind_score' in df_wrong.columns:
                imagebind_by_field = df_wrong.groupby('target_field')['imagebind_score'].agg(['mean', 'std'])
                wrong_by_target_field['imagebind_score_mean'] = imagebind_by_field['mean']
                wrong_by_target_field['imagebind_score_std'] = imagebind_by_field['std']
            wrong_by_target_field.to_csv(eval_output_dir / 'wrong_by_target_field.csv')

        # Per correct-category (source) wrong summary
        wrong_by_source = df_wrong.groupby('correct_category').agg(
            clap_score_target_mean=('clap_score_target', 'mean'),
            clap_classifier_acc=('clap_classifier', 'mean'),
            count=('clap_score_target', 'count'),
        )
        if not skip_desync and 'desync' in df_wrong.columns:
            desync_by_source = df_wrong.groupby('correct_category')['desync'].agg(['mean', 'std'])
            wrong_by_source['desync_mean'] = desync_by_source['mean']
        if not skip_imagebind and 'imagebind_score' in df_wrong.columns:
            imagebind_by_source = df_wrong.groupby('correct_category')['imagebind_score'].agg(['mean', 'std'])
            wrong_by_source['imagebind_score_mean'] = imagebind_by_source['mean']
        wrong_by_source.to_csv(eval_output_dir / 'wrong_by_source_category.csv')

    # Save overall summary
    with open(eval_output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Print results
    log.info('=' * 60)
    log.info('EVALUATION RESULTS')
    log.info('=' * 60)

    if 'correct' in summary:
        log.info('[Correct (Video + true category)]')
        log.info(f'  CLAP score:  {summary["correct"]["clap_score_mean"]:.4f} +/- {summary["correct"]["clap_score_std"]:.4f}')
        if 'desync_mean' in summary['correct']:
            log.info(f'  DeSync:      {summary["correct"]["desync_mean"]:.4f} +/- {summary["correct"]["desync_std"]:.4f}')
        if 'imagebind_score_mean' in summary['correct']:
            log.info(f'  ImageBind:   {summary["correct"]["imagebind_score_mean"]:.4f} +/- {summary["correct"]["imagebind_score_std"]:.4f}')

    if 'wrong' in summary:
        log.info('[Wrong (Prompt Switch: source -> target)]')
        log.info(f'  CLAP target: {summary["wrong"]["clap_score_target_mean"]:.4f} +/- {summary["wrong"]["clap_score_target_std"]:.4f}')
        log.info(f'  CLAP source: {summary["wrong"]["clap_score_source_mean"]:.4f} +/- {summary["wrong"]["clap_score_source_std"]:.4f}')
        log.info(f'  Classifier:  {summary["wrong"]["clap_classifier_acc"]:.4f}')
        if 'isc_mean' in summary['wrong']:
            log.info(f'  ISC:         {summary["wrong"]["isc_mean"]:.4f} +/- {summary["wrong"]["isc_std"]:.4f}')
        if 'fad' in summary['wrong']:
            log.info(f'  FAD:         {summary["wrong"]["fad"]:.4f}')
        if 'desync_mean' in summary['wrong']:
            log.info(f'  DeSync:      {summary["wrong"]["desync_mean"]:.4f} +/- {summary["wrong"]["desync_std"]:.4f}')
        if 'imagebind_score_mean' in summary['wrong']:
            log.info(f'  ImageBind:   {summary["wrong"]["imagebind_score_mean"]:.4f} +/- {summary["wrong"]["imagebind_score_std"]:.4f}')

    log.info('=' * 60)

    return df_correct, df_wrong


def main():
    parser = argparse.ArgumentParser(description='Evaluate VGGSound-Sparse experiment (multi-model)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to experiment output directory '
                             '(e.g., results/research_axes/latent_update_method/evaluation/'
                             'VGGSound-Sparse/qualitative/exp_name)')
    parser.add_argument('--eval_output_dir', type=str, default=None,
                        help='Evaluation output dir '
                             '(default: results/research_axes/evaluation_metrics/evaluation/'
                             'VGGSound-Sparse/quantitative/YYYY-MM-DD_exp_name)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for CLAP')
    parser.add_argument('--skip_desync', action='store_true', help='Skip DeSync computation')
    parser.add_argument('--skip_imagebind', action='store_true',
                        help='Skip ImageBind video-audio score computation')
    parser.add_argument('--skip_fad', action='store_true',
                        help='Skip FAD computation even when GT FAD stats are available')
    parser.add_argument('--video_feature_cache', type=str,
                        default=None,
                        help='Path to pre-extracted video features .pt file for DeSync '
                             '(default: VGGSound-Sparse/test_video_features.pt)')
    parser.add_argument('--imagebind_video_feature_cache', type=str,
                        default=None,
                        help='Path to cached ImageBind video features .pt file '
                             '(default: eval_output_dir/imagebind_video_features.pt)')
    parser.add_argument('--gt_fad_stats', type=str,
                        default=None,
                        help='Path to pre-computed GT FAD statistics .pt file '
                             '(default: VGGSound-Sparse/gt_fad_stats.pt)')
    parser.add_argument('--audio_duration', type=float, default=0.0,
                        help='Use only the first N seconds of audio for metric computation. '
                             '0 means use full audio (default: 0)')
    parser.add_argument('--filter_csv', type=str, default=None,
                        help="Evaluate only videos listed in this CSV's video_id column. "
                             'Automatically maps VGGSound-Sparse IDs '
                             '(<youtube_id>_<start_ms>_<end_ms>) to result folder names '
                             '(<youtube_id>_<start_sec:06d>).')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='[%(levelname)-8s] %(message)s')

    exp_dir = Path(args.output_dir)
    if not exp_dir.exists():
        log.error(f'Experiment directory not found: {exp_dir}')
        return

    if args.eval_output_dir:
        eval_output_dir = Path(args.eval_output_dir)
    else:
        suffix = 'filtered' if args.filter_csv else 'all'
        default_eval_dir_name = f'{date.today().isoformat()}_{exp_dir.name}_{suffix}'
        eval_output_dir = DEFAULT_QUANT_RESULTS_DIR / default_eval_dir_name

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    log.info(f'Using device: {device}')
    log.info(f'Experiment: {exp_dir}')
    log.info(f'Eval output: {eval_output_dir}')

    sys.path.insert(0, str(AV_BENCHMARK_DIR))

    video_feature_cache_path = Path(args.video_feature_cache) if args.video_feature_cache else \
        resolve_default_metric_cache(VGGSOUND_SPARSE_DIR, 'test_video_features', args.audio_duration)
    imagebind_video_feature_cache_path = Path(args.imagebind_video_feature_cache) \
        if args.imagebind_video_feature_cache else eval_output_dir / 'imagebind_video_features.pt'
    gt_fad_stats_path = Path(args.gt_fad_stats) if args.gt_fad_stats else \
        resolve_default_metric_cache(VGGSOUND_SPARSE_DIR, 'gt_fad_stats', args.audio_duration)

    if args.audio_duration > 0:
        log.info(f'Audio duration limit: {args.audio_duration}s (from start)')
    log.info(f'Video feature cache: {video_feature_cache_path}')
    if not args.skip_imagebind:
        log.info(f'ImageBind video feature cache: {imagebind_video_feature_cache_path}')
    log.info(f'GT FAD stats: {gt_fad_stats_path}')

    filter_video_ids = None
    if args.filter_csv:
        csv_path = Path(args.filter_csv)
        log.info(f'Applying CSV video filter from: {csv_path}')
        filter_video_ids = load_video_id_filter_from_csv(csv_path, exp_dir)
        if not filter_video_ids:
            log.warning('CSV filter matched zero result folders; evaluation may find no entries.')

    evaluate(exp_dir, eval_output_dir, device=device, batch_size=args.batch_size,
             skip_desync=args.skip_desync,
             skip_imagebind=args.skip_imagebind,
             video_feature_cache_path=video_feature_cache_path,
             imagebind_video_feature_cache_path=imagebind_video_feature_cache_path,
             gt_fad_stats_path=gt_fad_stats_path,
             audio_duration=args.audio_duration,
             filter_video_ids=filter_video_ids,
             skip_fad=args.skip_fad)

    log.info('Evaluation complete!')


if __name__ == '__main__':
    main()
