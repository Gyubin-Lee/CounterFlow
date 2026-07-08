"""
Required environment:
    conda activate MMAudio

Compute multiFLAM-based metric for prompt switch evaluation.

For each generated "wrong" audio:
  metric = multiFLAM(audio, wrong_prompt) - multiFLAM(audio, correct_prompt)

Where multiFLAM = max over time frames of FLAM probability for a given text prompt.

A positive score means the wrong prompt is more present in the audio than the correct prompt,
indicating successful prompt switching (latent intervention).

This script also computes onset F1 from raw frame-wise FLAM probabilities:
    - GT stream: P_FLAM(correct_prompt, GT/source audio, frame)
  - GEN stream: P_FLAM(target_prompt, generated_audio, frame)
  - onset = 0->1 transition after thresholding
  - matching = one-to-one matching within tolerance window
"""

import argparse
import json
import csv
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Optional, Set, Tuple

import librosa
import numpy as np
import torch

import openflam

SR = 48000
DURATION = 8.0  # seconds, matching MMAudio generation
DEFAULT_FLAM_FRAME_SEC = 0.3125
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_QUANT_RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "research_axes"
    / "evaluation_metrics"
    / "evaluation"
    / "VGGSound-Sparse"
    / "quantitative"
)


def is_eval_dir_name(name: str) -> bool:
    return name.startswith("__eval") or name.startswith("eval_filtered")


def resolve_device(device_arg: str | None, gpu_id: int) -> str:
    """Resolve runtime device from CLI args."""
    if device_arg is not None:
        return device_arg

    if gpu_id < 0:
        return "cpu"

    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if gpu_id >= num_gpus:
            raise ValueError(
                f"Invalid --gpu_id={gpu_id}. Available CUDA devices: 0..{num_gpus - 1}"
            )
        return f"cuda:{gpu_id}"

    print("CUDA is not available. Falling back to CPU.")
    return "cpu"


def load_flam_model(device: str):
    flam = openflam.OpenFLAM(model_name="v1-base", default_ckpt_path="/tmp/openflam")
    flam.to(device)
    return flam


def load_audio(audio_path: str, max_samples: int) -> np.ndarray:
    """Load audio and pad/truncate to fixed length."""
    audio, _ = librosa.load(audio_path, sr=SR)
    if len(audio) >= max_samples:
        audio = audio[:max_samples]
    else:
        audio = np.pad(audio, (0, max_samples - len(audio)))
    return audio


def compute_flam_activation_batch(
    flam_model, audio_paths: list[str], texts: list[str], device: str
) -> np.ndarray:
    """
    Batch FLAM inference for multiple audios x multiple texts.
    Returns frame-wise FLAM probabilities:
      [num_audios, num_texts, num_frames]
    """
    max_samples = int(DURATION * SR)

    # Load all audios and stack into a single batch tensor
    audios = [load_audio(p, max_samples) for p in audio_paths]
    audio_tensor = torch.tensor(np.stack(audios), dtype=torch.float32).to(device)

    with torch.no_grad():
        # act_map shape: [num_audios, num_texts, T]
        act_map = (
            flam_model.get_local_similarity(
                audio_tensor,
                texts,
                method="unbiased",
                cross_product=True,
            )
            .cpu()
            .numpy()
        )

    return act_map


def compute_multiflam_batch(
    flam_model, audio_paths: list[str], texts: list[str], device: str
) -> np.ndarray:
    """Return max-over-time FLAM probabilities: [num_audios, num_texts]."""
    act_map = compute_flam_activation_batch(flam_model, audio_paths, texts, device)
    return act_map.max(axis=-1)


def extract_onset_frames(scores: np.ndarray, threshold: float) -> np.ndarray:
    """
    Convert frame-wise probabilities into onset frame indices via:
      activity_t = 1[s_t >= threshold]
      onset_t = activity_t == 1 and activity_{t-1} == 0
    """
    activity = np.asarray(scores) >= threshold
    prev = np.concatenate(([False], activity[:-1]))
    onset_mask = activity & (~prev)
    return np.flatnonzero(onset_mask).astype(np.int64)


def match_onsets_one_to_one(
    gt_onsets: np.ndarray,
    gen_onsets: np.ndarray,
    tolerance_frames: float,
) -> tuple[int, int, int]:
    """
    One-to-one greedy nearest matching within tolerance.
    Returns TP, FP, FN.
    """
    gt = np.asarray(gt_onsets, dtype=np.int64)
    gen = np.asarray(gen_onsets, dtype=np.int64)

    if gt.size == 0 and gen.size == 0:
        return 0, 0, 0
    if gt.size == 0:
        return 0, int(gen.size), 0
    if gen.size == 0:
        return 0, 0, int(gt.size)

    gt_sorted = np.sort(gt)
    gen_sorted = np.sort(gen)
    used = np.zeros(gen_sorted.shape[0], dtype=bool)
    tp = 0

    for g in gt_sorted:
        candidates = np.where(~used)[0]
        if candidates.size == 0:
            break

        dists = np.abs(gen_sorted[candidates] - g)
        best_local_idx = int(np.argmin(dists))
        best_dist = float(dists[best_local_idx])
        if best_dist <= tolerance_frames:
            best_gen_idx = int(candidates[best_local_idx])
            used[best_gen_idx] = True
            tp += 1

    fp = int(gen_sorted.size - tp)
    fn = int(gt_sorted.size - tp)
    return tp, fp, fn


def compute_precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """
    Compute precision/recall/F1 with stable handling for empty sets.
    - If no predicted onsets, precision is defined as 1.0.
    - If no GT onsets, recall is defined as 1.0.
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return float(precision), float(recall), float(f1)


def _parse_sparse_csv_video_id(video_id: str) -> Optional[Tuple[str, int]]:
    """
    Parse VGGSound-Sparse CSV video_id format:
      <youtube_id>_<start_ms>_<end_ms>
    """
    stem = Path(str(video_id)).stem
    parts = stem.rsplit("_", 2)
    if len(parts) != 3:
        return None
    yt_id, start_ms, _ = parts
    if not start_ms.isdigit():
        return None
    return yt_id, int(start_ms) // 1000


def _parse_result_sample_id(sample_id: str) -> Optional[Tuple[str, int]]:
    """
    Parse result sample dir format:
      <youtube_id>_<start_sec_zero_padded>
    """
    stem = Path(str(sample_id)).stem
    parts = stem.rsplit("_", 1)
    if len(parts) != 2:
        return None
    yt_id, start_sec = parts
    if not start_sec.isdigit():
        return None
    return yt_id, int(start_sec)


def load_sample_id_filter_from_csv(csv_path: Path, exp_dir: Path) -> Set[str]:
    """
    Load CSV video_id list and map to experiment sample directory names.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Filter CSV not found: {csv_path}")

    sample_dirs = {d.name for d in exp_dir.iterdir() if d.is_dir() and not is_eval_dir_name(d.name)}
    tuple_to_sample_id = {}
    for sample_id in sample_dirs:
        parsed = _parse_result_sample_id(sample_id)
        if parsed is not None:
            tuple_to_sample_id[parsed] = sample_id

    matched_sample_ids = set()
    unmatched_csv_ids = []
    total_rows = 0

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        if "video_id" not in (reader.fieldnames or []):
            raise ValueError(f"CSV must contain 'video_id' column: {csv_path}")

        for row in reader:
            total_rows += 1
            raw_id = row.get("video_id")
            if not raw_id:
                continue
            csv_id = Path(raw_id).stem

            if csv_id in sample_dirs:
                matched_sample_ids.add(csv_id)
                continue

            parsed_sparse = _parse_sparse_csv_video_id(csv_id)
            if parsed_sparse is not None and parsed_sparse in tuple_to_sample_id:
                matched_sample_ids.add(tuple_to_sample_id[parsed_sparse])
                continue

            parsed_result = _parse_result_sample_id(csv_id)
            if parsed_result is not None and parsed_result in tuple_to_sample_id:
                matched_sample_ids.add(tuple_to_sample_id[parsed_result])
                continue

            unmatched_csv_ids.append(csv_id)

    print(f"CSV filter matched {len(matched_sample_ids)} sample dirs from {total_rows} CSV rows.")
    if unmatched_csv_ids:
        preview = ", ".join(unmatched_csv_ids[:10])
        suffix = " ..." if len(unmatched_csv_ids) > 10 else ""
        print(f"CSV filter unmatched {len(unmatched_csv_ids)} IDs: {preview}{suffix}")

    return matched_sample_ids


def process_experiment(
    exp_dir: Path,
    device: str,
    output_path: Path,
    filter_sample_ids: Optional[Set[str]] = None,
    onset_threshold: float = 0.5,
    onset_tolerance_sec: float = DEFAULT_FLAM_FRAME_SEC,
    flam_frame_sec: float = DEFAULT_FLAM_FRAME_SEC,
):
    flam_model = load_flam_model(device)

    sample_dirs = sorted([
        d for d in exp_dir.iterdir()
        if d.is_dir() and not is_eval_dir_name(d.name)
    ])
    if filter_sample_ids is not None:
        before = len(sample_dirs)
        sample_dirs = [d for d in sample_dirs if d.name in filter_sample_ids]
        print(f"After CSV filter: {len(sample_dirs)}/{before} sample dirs will be evaluated.")

    results = []

    for idx, sample_dir in enumerate(sample_dirs):
        correct_meta_path = sample_dir / "correct" / "metadata.json"
        wrong_meta_path = sample_dir / "wrong" / "metadata.json"

        if not wrong_meta_path.exists():
            print(f"[{idx+1}/{len(sample_dirs)}] Skipping {sample_dir.name}: missing wrong metadata")
            continue

        if correct_meta_path.exists():
            with open(correct_meta_path) as f:
                correct_meta = json.load(f)
        else:
            correct_meta = {}

        with open(wrong_meta_path) as f:
            wrong_meta = json.load(f)

        correct_prompt = (
            correct_meta.get("prompt")
            or correct_meta.get("correct_category")
            or wrong_meta.get("source_prompt")
            or wrong_meta.get("correct_category")
        )
        if not correct_prompt:
            print(f"[{idx+1}/{len(sample_dirs)}] Skipping {sample_dir.name}: missing source prompt")
            continue

        correct_audio_path = sample_dir / "correct" / f"{sample_dir.name}_correct.wav"
        has_correct_audio = correct_audio_path.exists()

        # Collect wrong results and build batch
        wrong_results = wrong_meta.get("results", [])
        audio_paths = [str(correct_audio_path)] if has_correct_audio else []
        valid_wrong = []
        for wr in wrong_results:
            wrong_audio_path = sample_dir / "wrong" / wr["audio_file"]
            if wrong_audio_path.exists():
                audio_paths.append(str(wrong_audio_path))
                valid_wrong.append(wr)
            else:
                print(f"  Missing wrong audio: {wrong_audio_path}")

        if not valid_wrong:
            print(f"[{idx+1}/{len(sample_dirs)}] Skipping {sample_dir.name}: no wrong audios")
            continue

        # Build unique text list: correct prompt first, then target prompts
        target_prompts = [wr["target_prompt"] for wr in valid_wrong]
        all_texts = []
        text_to_idx = {}
        for t in [correct_prompt] + target_prompts:
            if t not in text_to_idx:
                text_to_idx[t] = len(all_texts)
                all_texts.append(t)

        # Single batched inference:
        # act_map shape: [num_audios, num_texts, num_frames]
        act_map = compute_flam_activation_batch(flam_model, audio_paths, all_texts, device)
        scores = act_map.max(axis=-1)  # [num_audios, num_texts]

        correct_idx = 0 if has_correct_audio else None
        correct_text_idx = text_to_idx[correct_prompt]
        tolerance_frames = onset_tolerance_sec / flam_frame_sec

        for i, wr in enumerate(valid_wrong):
            wrong_audio_idx = i + 1 if has_correct_audio else i
            target_text_idx = text_to_idx[wr["target_prompt"]]

            wrong_flam_target = float(scores[wrong_audio_idx, target_text_idx])
            wrong_flam_correct = float(scores[wrong_audio_idx, correct_text_idx])
            delta = wrong_flam_target - wrong_flam_correct

            if correct_idx is not None:
                correct_flam_correct = float(scores[correct_idx, correct_text_idx])
                correct_flam_target = float(scores[correct_idx, target_text_idx])

                # Onset F1 from raw frame-wise FLAM probabilities
                s_gt = act_map[correct_idx, correct_text_idx]
                s_gen = act_map[wrong_audio_idx, target_text_idx]
                gt_onsets = extract_onset_frames(s_gt, onset_threshold)
                gen_onsets = extract_onset_frames(s_gen, onset_threshold)
                tp, fp, fn = match_onsets_one_to_one(gt_onsets, gen_onsets, tolerance_frames)
                onset_precision, onset_recall, onset_f1 = compute_precision_recall_f1(tp, fp, fn)
            else:
                correct_flam_correct = None
                correct_flam_target = None
                gt_onsets = np.array([], dtype=np.int64)
                gen_onsets = np.array([], dtype=np.int64)
                tp = fp = fn = None
                onset_precision = onset_recall = onset_f1 = None

            results.append({
                "sample_id": sample_dir.name,
                "correct_prompt": correct_prompt,
                "target_prompt": wr["target_prompt"],
                "target_field": wr.get("target_field"),
                "target_level": wr.get("target_level"),
                "conflict_level": wr.get("conflict_level"),
                "wrong_audio_flam_target": round(wrong_flam_target, 4),
                "wrong_audio_flam_correct": round(wrong_flam_correct, 4),
                "delta_flam": round(delta, 4),
                "correct_audio_flam_correct": (
                    round(correct_flam_correct, 4) if correct_flam_correct is not None else None
                ),
                "correct_audio_flam_target": (
                    round(correct_flam_target, 4) if correct_flam_target is not None else None
                ),
                "gt_onset_count": int(len(gt_onsets)) if correct_idx is not None else None,
                "gen_onset_count": int(len(gen_onsets)) if correct_idx is not None else None,
                "onset_tp": int(tp) if tp is not None else None,
                "onset_fp": int(fp) if fp is not None else None,
                "onset_fn": int(fn) if fn is not None else None,
                "onset_precision": round(onset_precision, 4) if onset_precision is not None else None,
                "onset_recall": round(onset_recall, 4) if onset_recall is not None else None,
                "onset_f1": round(onset_f1, 4) if onset_f1 is not None else None,
            })

        print(f"[{idx+1}/{len(sample_dirs)}] Done: {sample_dir.name} ({correct_prompt})")

    # Write CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id", "correct_prompt", "target_prompt",
        "target_field", "target_level", "conflict_level",
        "wrong_audio_flam_target", "wrong_audio_flam_correct", "delta_flam",
        "correct_audio_flam_correct", "correct_audio_flam_target",
        "gt_onset_count", "gen_onset_count",
        "onset_tp", "onset_fp", "onset_fn",
        "onset_precision", "onset_recall", "onset_f1",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    summary_output_path = output_path.with_name(f"{output_path.stem}_summary.json")
    summary = {
        "experiment": exp_dir.name,
        "total_pairs_evaluated": len(results),
        "results_csv": str(output_path),
        "onset_threshold": onset_threshold,
        "onset_tolerance_sec": onset_tolerance_sec,
        "flam_frame_sec": flam_frame_sec,
    }

    # Print summary
    if results:
        wrong_flam_source_vals = [r["wrong_audio_flam_correct"] for r in results]
        wrong_flam_target_vals = [r["wrong_audio_flam_target"] for r in results]
        deltas = [r["delta_flam"] for r in results]
        wrong_flam_source_mean = float(np.mean(wrong_flam_source_vals))
        wrong_flam_target_mean = float(np.mean(wrong_flam_target_vals))
        delta_mean = float(np.mean(deltas))
        delta_std = float(np.std(deltas))
        delta_pos_ratio = float(np.mean(np.array(deltas) > 0))
        onset_f1s = [r["onset_f1"] for r in results if r["onset_f1"] is not None]
        onset_precs = [r["onset_precision"] for r in results if r["onset_precision"] is not None]
        onset_recs = [r["onset_recall"] for r in results if r["onset_recall"] is not None]
        onset_tps = [r["onset_tp"] for r in results if r["onset_tp"] is not None]
        onset_fps = [r["onset_fp"] for r in results if r["onset_fp"] is not None]
        onset_fns = [r["onset_fn"] for r in results if r["onset_fn"] is not None]

        print(f"\n{'='*60}")
        print(f"Experiment: {exp_dir.name}")
        print(f"Total pairs evaluated: {len(results)}")
        print(f"wrong_audio FLAM vs source prompt mean: {wrong_flam_source_mean:.4f}")
        print(f"wrong_audio FLAM vs target prompt mean: {wrong_flam_target_mean:.4f}")
        print(f"delta_flam (wrong_target - wrong_correct) mean: {delta_mean:.4f}")
        print(f"delta_flam std: {delta_std:.4f}")
        print(f"delta_flam > 0 ratio: {delta_pos_ratio:.4f}")
        print(f"Onset threshold: {onset_threshold:.4f}")
        print(f"Onset tolerance: {onset_tolerance_sec:.4f}s ({onset_tolerance_sec / flam_frame_sec:.2f} frames)")
        if onset_f1s:
            print(f"Onset Precision (macro): {float(np.mean(onset_precs)):.4f}")
            print(f"Onset Recall (macro):    {float(np.mean(onset_recs)):.4f}")
            print(f"Onset F1 (macro):        {float(np.mean(onset_f1s)):.4f}")
        else:
            print("Onset metrics: skipped (no generated correct/source audio present)")
        print(f"Results saved to: {output_path}")

        # Per target category summary
        by_target = defaultdict(list)
        for r in results:
            by_target[r["target_prompt"]].append(r["delta_flam"])
        print(f"\nPer target category:")
        per_target_summary = {}
        for cat in sorted(by_target.keys()):
            vals = by_target[cat]
            cat_rows = [r for r in results if r["target_prompt"] == cat]
            cat_mean = float(np.mean(vals))
            cat_std = float(np.std(vals))
            cat_n = len(vals)
            cat_onset_f1_values = [r["onset_f1"] for r in cat_rows if r["onset_f1"] is not None]
            cat_onset_f1_mean = (
                float(np.mean(cat_onset_f1_values)) if cat_onset_f1_values else None
            )
            print(
                f"  {cat:25s}  mean={cat_mean:+.4f}  std={cat_std:.4f}  "
                f"onset_f1={cat_onset_f1_mean:.4f}  n={cat_n}"
                if cat_onset_f1_mean is not None
                else f"  {cat:25s}  mean={cat_mean:+.4f}  std={cat_std:.4f}  n={cat_n}"
            )
            per_target_summary[cat] = {
                "mean": round(cat_mean, 4),
                "std": round(cat_std, 4),
                "n": cat_n,
            }
            if cat_onset_f1_mean is not None:
                per_target_summary[cat]["onset_f1_mean"] = round(cat_onset_f1_mean, 4)

        per_target_field_summary = {}
        if any(r.get("target_field") for r in results):
            by_target_field = defaultdict(list)
            for r in results:
                if r.get("target_field"):
                    by_target_field[r["target_field"]].append(r)
            print(f"\nPer target field:")
            for field in sorted(by_target_field.keys()):
                rows = by_target_field[field]
                vals = [r["delta_flam"] for r in rows]
                field_onset_f1s = [r["onset_f1"] for r in rows if r["onset_f1"] is not None]
                field_summary = {
                    "mean": round(float(np.mean(vals)), 4),
                    "std": round(float(np.std(vals)), 4),
                    "positive_ratio": round(float(np.mean(np.array(vals) > 0)), 4),
                    "n": len(rows),
                }
                if field_onset_f1s:
                    field_summary["onset_f1_mean"] = round(float(np.mean(field_onset_f1s)), 4)
                print(
                    f"  {field:20s}  mean={field_summary['mean']:+.4f}  "
                    f"std={field_summary['std']:.4f}  "
                    f"pos={field_summary['positive_ratio']:.4f}  n={field_summary['n']}"
                )
                per_target_field_summary[field] = field_summary

        summary.update({
            "wrong_audio_flam_source_prompt_mean": round(wrong_flam_source_mean, 4),
            "wrong_audio_flam_target_prompt_mean": round(wrong_flam_target_mean, 4),
            "delta_flam_mean": round(delta_mean, 4),
            "delta_flam_std": round(delta_std, 4),
            "delta_flam_positive_ratio": round(delta_pos_ratio, 4),
            "onset_precision_macro": round(float(np.mean(onset_precs)), 4) if onset_precs else None,
            "onset_recall_macro": round(float(np.mean(onset_recs)), 4) if onset_recs else None,
            "onset_f1_macro": round(float(np.mean(onset_f1s)), 4) if onset_f1s else None,
            "onset_tp_total": int(np.sum(onset_tps)) if onset_tps else None,
            "onset_fp_total": int(np.sum(onset_fps)) if onset_fps else None,
            "onset_fn_total": int(np.sum(onset_fns)) if onset_fns else None,
            "per_target": per_target_summary,
            "per_target_field": per_target_field_summary,
        })
    else:
        print("No results computed.")
        summary.update({
            "wrong_audio_flam_source_prompt_mean": None,
            "wrong_audio_flam_target_prompt_mean": None,
            "delta_flam_mean": None,
            "delta_flam_std": None,
            "delta_flam_positive_ratio": None,
            "onset_precision_macro": None,
            "onset_recall_macro": None,
            "onset_f1_macro": None,
            "onset_tp_total": None,
            "onset_fp_total": None,
            "onset_fn_total": None,
            "per_target": {},
        })

    with open(summary_output_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Summary JSON saved to: {summary_output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compute multiFLAM metric for prompt switch evaluation")
    parser.add_argument(
        "--exp_dir",
        type=str,
        required=True,
        help="Path to experiment directory (e.g., eval_vggsound_sparse_output/260323_SDE_ts_15_sigma_1.5)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output CSV path. Defaults to "
            "results/research_axes/evaluation_metrics/evaluation/"
            "VGGSound-Sparse/quantitative/YYYY-MM-DD_exp-name/flam_scores*.csv."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Explicit device (e.g., cuda:0, cuda:1, cpu). Overrides --gpu_id if set.",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="CUDA GPU index to use (default: 0). Set -1 to force CPU.",
    )
    parser.add_argument(
        "--filter_csv",
        type=str,
        default=None,
        help=(
            "Evaluate only videos listed in this CSV's video_id column. "
            "Maps VGGSound-Sparse IDs (<youtube_id>_<start_ms>_<end_ms>) "
            "to result sample dirs (<youtube_id>_<start_sec:06d>)."
        ),
    )
    parser.add_argument(
        "--onset_threshold",
        type=float,
        default=0.5,
        help="Threshold tau for binarizing raw frame-wise FLAM probabilities (default: 0.5).",
    )
    parser.add_argument(
        "--onset_tolerance_sec",
        type=float,
        default=DEFAULT_FLAM_FRAME_SEC,
        help=(
            "Tolerance delta (seconds) for one-to-one onset matching "
            f"(default: {DEFAULT_FLAM_FRAME_SEC}, i.e., 1 FLAM frame)."
        ),
    )
    parser.add_argument(
        "--flam_frame_sec",
        type=float,
        default=DEFAULT_FLAM_FRAME_SEC,
        help=f"FLAM frame duration in seconds (default: {DEFAULT_FLAM_FRAME_SEC}).",
    )
    args = parser.parse_args()

    if args.onset_tolerance_sec < 0:
        raise ValueError("--onset_tolerance_sec must be >= 0.")
    if args.flam_frame_sec <= 0:
        raise ValueError("--flam_frame_sec must be > 0.")
    if not (0.0 <= args.onset_threshold <= 1.0):
        raise ValueError("--onset_threshold must be in [0, 1].")

    exp_dir = Path(args.exp_dir)
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    if args.output:
        output_path = Path(args.output)
    else:
        default_eval_dir_name = f"{date.today().isoformat()}_{exp_dir.name}"
        output_file_name = "flam_scores_filtered.csv" if args.filter_csv else "flam_scores.csv"
        output_path = DEFAULT_QUANT_RESULTS_DIR / default_eval_dir_name / output_file_name

    resolved_device = resolve_device(args.device, args.gpu_id)
    filter_sample_ids = None
    if args.filter_csv:
        filter_csv_path = Path(args.filter_csv)
        print(f"Applying CSV filter from: {filter_csv_path}")
        filter_sample_ids = load_sample_id_filter_from_csv(filter_csv_path, exp_dir)
        if not filter_sample_ids:
            print("Warning: CSV filter matched zero sample dirs.")

    print(f"Experiment: {exp_dir}")
    print(f"Output: {output_path}")
    print(f"Device: {resolved_device}")
    print(f"Onset threshold: {args.onset_threshold}")
    print(f"Onset tolerance (sec): {args.onset_tolerance_sec}")
    print(f"FLAM frame duration (sec): {args.flam_frame_sec}")
    print()

    process_experiment(
        exp_dir,
        resolved_device,
        output_path,
        filter_sample_ids=filter_sample_ids,
        onset_threshold=args.onset_threshold,
        onset_tolerance_sec=args.onset_tolerance_sec,
        flam_frame_sec=args.flam_frame_sec,
    )


if __name__ == "__main__":
    main()
