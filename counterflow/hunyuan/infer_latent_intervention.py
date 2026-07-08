#!/usr/bin/env python3
"""
Required environment:
    conda activate Hunyuan

Latent Intervention Inference for HunyuanVideo-Foley

Implements a training-free framework for video-guided Foley generation with
arbitrary text control via latent intervention with SDE simulation.

Key features:
  - Two-phase latent intervention (Phase 1: video+text, Phase 2: text-only)
  - SDE/ODE sampling per phase
  - Decomposed CFG with optional negative source prompting
  - Three method variants:
      Method 1: Decomposed CFG (Phase 1) + Normal target text (Phase 2)
      Method 2: Decomposed CFG + Neg Source (Phase 1) + Normal target text (Phase 2)
      Method 3: Decomposed CFG + Neg Source (Phase 1) + Neg Source Prompt (Phase 2)

Reference: Algorithm 5 - Latent Intervention with SDE simulation
"""

import os
import argparse
import random
import numpy as np
import torch
import pandas as pd
import torchaudio
from loguru import logger
from tqdm import tqdm
from diffusers.utils.torch_utils import randn_tensor

from hunyuanvideo_foley.utils.model_utils import (
    load_model, _get_dac_device,
)
from hunyuanvideo_foley.utils.feature_utils import (
    encode_text_feat, encode_video_features,
)
from hunyuanvideo_foley.utils.config_utils import AttributeDict
from hunyuanvideo_foley.utils.media_utils import merge_audio_video
from hunyuanvideo_foley.utils.schedulers import FlowMatchDiscreteScheduler
from hunyuanvideo_foley.constants import DEFAULT_NEGATIVE_PROMPT


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------
def set_manual_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------
def encode_all_text_features(source_prompt, target_prompt, neg_prompt, model_dict, cfg):
    """Encode source, target, and negative prompts into text features."""
    if neg_prompt is None:
        neg_prompt = DEFAULT_NEGATIVE_PROMPT

    prompts = [neg_prompt, target_prompt, source_prompt]
    text_feat_res, _ = encode_text_feat(prompts, model_dict)

    uncond_text_feat = text_feat_res[0:1]   # negative prompt → unconditional baseline
    tar_text_feat    = text_feat_res[1:2]   # target prompt
    src_text_feat    = text_feat_res[2:3]   # source prompt

    # Truncate to model's max text length
    max_len = cfg.model_config.model_kwargs.text_length
    if max_len < uncond_text_feat.shape[1]:
        uncond_text_feat = uncond_text_feat[:, :max_len]
        tar_text_feat    = tar_text_feat[:, :max_len]
        src_text_feat    = src_text_feat[:, :max_len]

    return AttributeDict({
        'uncond_text_feat': uncond_text_feat,
        'tar_text_feat': tar_text_feat,
        'src_text_feat': src_text_feat,
    })


def feature_process_intervention(video_path, source_prompt, target_prompt,
                                  model_dict, cfg, neg_prompt=None):
    """Extract video features and encode all text prompts."""
    visual_feats, audio_len_in_s = encode_video_features(video_path, model_dict)
    text_feats = encode_all_text_features(
        source_prompt, target_prompt, neg_prompt, model_dict, cfg,
    )

    # Release feature-extraction models (SigLIP2, CLAP, Syncformer) if offloading
    if hasattr(model_dict, 'manager') and hasattr(model_dict.manager, 'release_feature_models'):
        model_dict.manager.release_feature_models()

    return visual_feats, text_feats, audio_len_in_s


# ---------------------------------------------------------------------------
# Model forward helper
# ---------------------------------------------------------------------------
@torch.no_grad()
def model_forward(model, latents, t, text_feat, clip_feat, sync_feat,
                  device, target_dtype):
    """Batched forward pass through the foley model."""
    autocast_enabled = target_dtype != torch.float32
    t_expand = t.repeat(latents.shape[0])

    with torch.autocast(device_type=device.type, enabled=autocast_enabled,
                        dtype=target_dtype):
        output = model(
            x=latents,
            t=t_expand,
            cond=text_feat,
            clip_feat=clip_feat,
            sync_feat=sync_feat,
            return_dict=True,
        )["x"]

    return output.to(torch.float32)


# ---------------------------------------------------------------------------
# Velocity computation  – Phase 1 (Decomposed CFG)
# ---------------------------------------------------------------------------
def compute_velocity_phase1(
    latents, t, model,
    uncond_text, tar_text, src_text,
    siglip2_feat, syncformer_feat,
    empty_clip, empty_sync,
    w_vid, w_txt,
    use_negative_src=False,
    device=None, target_dtype=None,
):
    """
    Phase 1 – Decomposed CFG (with video conditioning).

    Without negative source prompting (Method 1):
      V = V_uncond + w_vid*(V_vid - V_uncond) + w_txt*(V_tar - V_uncond)
      NFE = 3  (uncond, vid-only, txt-tar)

    With negative source prompting (Method 2, 3):
      V = V_uncond + w_vid*(V_vid - V_uncond) + w_txt*(V_tar - V_src)
      NFE = 4  (uncond, vid-only, txt-tar, txt-src)
    """
    if use_negative_src:
        n = 4
        lat_in  = torch.cat([latents] * n)
        clip_in = torch.cat([empty_clip, siglip2_feat, empty_clip, empty_clip])
        sync_in = torch.cat([empty_sync, syncformer_feat, empty_sync, empty_sync])
        text_in = torch.cat([uncond_text, uncond_text, tar_text, src_text])

        pred = model_forward(model, lat_in, t, text_in, clip_in, sync_in,
                             device, target_dtype)
        v_uncond, v_vid, v_tar, v_src = pred.chunk(n)

        return v_uncond + w_vid * (v_vid - v_uncond) + w_txt * (v_tar - v_src)
    else:
        n = 3
        lat_in  = torch.cat([latents] * n)
        clip_in = torch.cat([empty_clip, siglip2_feat, empty_clip])
        sync_in = torch.cat([empty_sync, syncformer_feat, empty_sync])
        text_in = torch.cat([uncond_text, uncond_text, tar_text])

        pred = model_forward(model, lat_in, t, text_in, clip_in, sync_in,
                             device, target_dtype)
        v_uncond, v_vid, v_tar = pred.chunk(n)

        return v_uncond + w_vid * (v_vid - v_uncond) + w_txt * (v_tar - v_uncond)


# ---------------------------------------------------------------------------
# Velocity computation  – Phase 2 (text-only, no video)
# ---------------------------------------------------------------------------
def compute_velocity_phase2(
    latents, t, model,
    uncond_text, tar_text, src_text,
    empty_clip, empty_sync,
    w_cfg,
    use_negative_src=False,
    device=None, target_dtype=None,
):
    """
    Phase 2 – Text-only guidance (no video conditioning).

    Normal update (Method 1, 2):
      V = V_uncond + w_cfg*(V_tar - V_uncond)
      NFE = 2  (uncond, txt-tar)

    Negative source prompting (Method 3):
      V = V_uncond + w_cfg*(V_tar - V_src)
      NFE = 3  (uncond, txt-tar, txt-src)
    """
    if use_negative_src:
        n = 3
        lat_in  = torch.cat([latents] * n)
        clip_in = torch.cat([empty_clip] * n)
        sync_in = torch.cat([empty_sync] * n)
        text_in = torch.cat([uncond_text, tar_text, src_text])

        pred = model_forward(model, lat_in, t, text_in, clip_in, sync_in,
                             device, target_dtype)
        v_uncond, v_tar, v_src = pred.chunk(n)

        return v_uncond + w_cfg * (v_tar - v_src)
    else:
        n = 2
        lat_in  = torch.cat([latents] * n)
        clip_in = torch.cat([empty_clip] * n)
        sync_in = torch.cat([empty_sync] * n)
        text_in = torch.cat([uncond_text, tar_text])

        pred = model_forward(model, lat_in, t, text_in, clip_in, sync_in,
                             device, target_dtype)
        v_uncond, v_tar = pred.chunk(n)

        return v_uncond + w_cfg * (v_tar - v_uncond)


# ---------------------------------------------------------------------------
# SDE / ODE integration step
# ---------------------------------------------------------------------------
def sde_step(latents, velocity, sigma_i, sigma_next, sde_sigma, device):
    """
    Single Euler–Maruyama step for SDE (or plain Euler for ODE when sde_sigma=0).

    Flow-matching convention:
      - sigma goes from 1 (noise) → 0 (clean)
      - Mapping to Algorithm 5's notation:  t_algo = 1 − sigma,  (1−t) = sigma

    SDE update (Algorithm 5, lines 6-8):
      V_corrected = V̂ + σ_sde² / (2·sigma) · ((1−sigma)·V̂ − Z)
      Z_next = Z + dt · V_corrected + σ_sde · √|dt| · ε
    where dt = sigma_next − sigma (< 0).
    """
    dt = sigma_next - sigma_i          # negative (sigma decreases)

    if sde_sigma > 0 and sigma_i > 1e-6:
        t_algo      = 1.0 - sigma_i   # algorithm's "t"
        one_minus_t = sigma_i          # algorithm's "(1 − t)"

        sde_sigma = sde_sigma * one_minus_t.sqrt()

        # Drift correction
        correction  = (sde_sigma ** 2) / (2.0 * one_minus_t) * (
            t_algo * velocity - latents
        )
        v_corrected = velocity + correction

        # Euler–Maruyama step
        noise  = torch.randn_like(latents, device=device)
        latents = (
            latents
            + dt * v_corrected
            + sde_sigma * torch.sqrt(-dt) * noise
        )
    else:
        # Pure ODE (Euler) step
        latents = latents + dt * velocity

    return latents


# ---------------------------------------------------------------------------
# Main denoising loop with latent intervention
# ---------------------------------------------------------------------------
@torch.no_grad()
def denoise_process_intervention(
    visual_feats,
    text_feats,
    audio_len_in_s,
    model_dict,
    cfg,
    # ---- intervention hyper-parameters ----
    transition_step: int   = 8,
    sde_sigma:       float = 2.0,
    phase1_sde:      bool  = True,
    phase2_sde:      bool  = True,
    method:          int   = 3,       # 1, 2, or 3
    w_vid:           float = 3.0,
    w_txt:           float = 4.5,
    w_cfg:           float = 4.5,
    num_inference_steps: int = 25,
    batch_size:      int   = 1,
):
    """
    Two-phase denoising with latent intervention.

    Phase 1 (step 0 … transition_step−1):
        Decomposed CFG with video + target text (± negative source prompting).
    Phase 2 (step transition_step … N−1):
        Text-only guidance with target text (± negative source prompting).
    """
    device       = model_dict.device
    model        = model_dict.foley_model
    target_dtype = model.dtype

    # ---- scheduler ----
    scheduler = FlowMatchDiscreteScheduler(
        shift=cfg.diffusion_config.sample_flow_shift,
        reverse=cfg.diffusion_config.flow_reverse,
        solver=cfg.diffusion_config.flow_solver,
        use_flux_shift=cfg.diffusion_config.sample_use_flux_shift,
        flux_base_shift=cfg.diffusion_config.flux_base_shift,
        flux_max_shift=cfg.diffusion_config.flux_max_shift,
    )
    scheduler.set_timesteps(num_inference_steps, device=device)
    sigmas    = scheduler.sigmas      # [N+1]  1 → 0
    timesteps = scheduler.timesteps   # [N]    sigma[:-1] * 1000

    # ---- initial latent Z_0 ~ N(0, 1) ----
    latents = randn_tensor(
        (batch_size,
         cfg.model_config.model_kwargs.audio_vae_latent_dim,
         int(audio_len_in_s * cfg.model_config.model_kwargs.audio_frame_rate)),
        device=device,
        dtype=torch.float32,
    )

    def _expand_or_validate(feat, name):
        """Accept [1, L, D] (broadcast) or [B, L, D] (already batched)."""
        if feat.ndim != 3:
            raise ValueError(f"{name} must be rank-3 [B, L, D], got shape={tuple(feat.shape)}")
        if feat.shape[0] == batch_size:
            return feat.to(device)
        if feat.shape[0] == 1:
            return feat.repeat(batch_size, 1, 1).to(device)
        raise ValueError(
            f"{name} batch dimension mismatch: expected 1 or {batch_size}, got {feat.shape[0]}"
        )

    # ---- prepare features (ensure everything is on foley device) ----
    siglip2_feat = _expand_or_validate(visual_feats.siglip2_feat, "visual_feats.siglip2_feat")
    syncformer_feat = _expand_or_validate(
        visual_feats.syncformer_feat, "visual_feats.syncformer_feat"
    )

    empty_clip = model.get_empty_clip_sequence(
        bs=batch_size, len=siglip2_feat.shape[1],
    ).to(device)
    empty_sync = model.get_empty_sync_sequence(
        bs=batch_size, len=syncformer_feat.shape[1],
    ).to(device)

    uncond_text = _expand_or_validate(text_feats.uncond_text_feat, "text_feats.uncond_text_feat")
    tar_text = _expand_or_validate(text_feats.tar_text_feat, "text_feats.tar_text_feat")
    src_text = _expand_or_validate(text_feats.src_text_feat, "text_feats.src_text_feat")

    # ---- method → per-phase settings ----
    phase1_neg_src = method in (2, 3)
    phase2_neg_src = method == 3

    logger.info(
        f"Method {method} | transition_step={transition_step}/{num_inference_steps} | "
        f"sde_sigma={sde_sigma} | Phase1 SDE={phase1_sde} | Phase2 SDE={phase2_sde}"
    )
    logger.info(f"w_vid={w_vid}  w_txt={w_txt}  w_cfg={w_cfg}")
    logger.info(
        f"Phase1: Decomposed CFG + neg_src={phase1_neg_src} (NFE={4 if phase1_neg_src else 3})"
    )
    logger.info(
        f"Phase2: {'Neg Source Prompt' if phase2_neg_src else 'Normal target text'} "
        f"(NFE={3 if phase2_neg_src else 2})"
    )

    # ---- denoising loop ----
    for i, t in tqdm(enumerate(timesteps), total=len(timesteps), desc="Denoising"):
        sigma_i    = sigmas[i].float()
        sigma_next = sigmas[i + 1].float()
        latents    = latents.float()

        # --- Phase 1: video + text ---
        if i < transition_step:
            velocity = compute_velocity_phase1(
                latents, t, model,
                uncond_text, tar_text, src_text,
                siglip2_feat, syncformer_feat,
                empty_clip, empty_sync,
                w_vid, w_txt,
                use_negative_src=phase1_neg_src,
                device=device, target_dtype=target_dtype,
            )
            cur_sigma = sde_sigma if phase1_sde else 0.0
        # --- Phase 2: text only ---
        else:
            velocity = compute_velocity_phase2(
                latents, t, model,
                uncond_text, tar_text, src_text,
                empty_clip, empty_sync,
                w_cfg,
                use_negative_src=phase2_neg_src,
                device=device, target_dtype=target_dtype,
            )
            cur_sigma = sde_sigma if phase2_sde else 0.0

        # SDE / ODE step
        latents = sde_step(latents, velocity, sigma_i, sigma_next, cur_sigma, device)

    # ---- decode latents → audio ----
    dac_device = _get_dac_device(model_dict)
    audio = model_dict.dac_model.decode(latents.to(dac_device))
    audio = audio.float().cpu()
    audio = audio[:, :int(audio_len_in_s * model_dict.dac_model.sample_rate)]
    sample_rate = model_dict.dac_model.sample_rate

    if hasattr(model_dict, 'manager') and hasattr(model_dict.manager, 'release_inference_models'):
        model_dict.manager.release_inference_models()

    del latents, siglip2_feat, syncformer_feat, empty_clip, empty_sync
    del uncond_text, tar_text, src_text
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return audio, sample_rate


# ---------------------------------------------------------------------------
# High-level inference wrappers
# ---------------------------------------------------------------------------
def infer_intervention(video_path, source_prompt, target_prompt,
                       model_dict, cfg, **kwargs):
    """Run latent-intervention inference on a single video."""
    neg_prompt = kwargs.pop("neg_prompt", None)

    visual_feats, text_feats, audio_len_in_s = feature_process_intervention(
        video_path, source_prompt, target_prompt,
        model_dict, cfg, neg_prompt=neg_prompt,
    )

    audio, sample_rate = denoise_process_intervention(
        visual_feats, text_feats, audio_len_in_s,
        model_dict, cfg, **kwargs,
    )
    return audio[0], sample_rate


def generate_audio_intervention(model_dict, cfg, csv_path, output_dir, **kwargs):
    """Batch inference from a CSV file (columns: video, source_prompt, target_prompt)."""
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    required_cols = {"video", "source_prompt", "target_prompt"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"CSV must contain columns {required_cols}. Found: {set(df.columns)}"
        )

    for idx, row in df.iterrows():
        video_path    = row["video"]
        source_prompt = row["source_prompt"]
        target_prompt = row["target_prompt"]

        logger.info(f"[{idx}] video={video_path}")
        logger.info(f"  source: {source_prompt}")
        logger.info(f"  target: {target_prompt}")

        out_audio = os.path.join(output_dir, f"{idx:04d}.wav")
        out_video = os.path.join(output_dir, f"{idx:04d}.mp4")

        if os.path.exists(out_audio) and os.path.exists(out_video):
            logger.info(f"  Skipping (already exists)")
            continue

        audio, sr = infer_intervention(
            video_path, source_prompt, target_prompt,
            model_dict, cfg, **kwargs,
        )
        torchaudio.save(out_audio, audio, sr)
        merge_audio_video(out_audio, video_path, out_video)

    logger.info(f"All outputs saved to {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="HunyuanVideo-Foley — Latent Intervention Inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- model ----
    p.add_argument("--model_path", type=str, required=True,
                   help="Path to pretrained model directory")
    p.add_argument("--config_path", type=str, default=None,
                   help="Path to config YAML (auto-selected from model_size if omitted)")
    p.add_argument("--model_size", type=str, choices=["xl", "xxl"], default="xxl",
                   help="Model size")

    # ---- input ----
    input_g = p.add_mutually_exclusive_group(required=True)
    input_g.add_argument("--csv_path", type=str,
                         help="CSV with columns: video, source_prompt, target_prompt")
    input_g.add_argument("--single_video", type=str,
                         help="Path to a single video file")
    p.add_argument("--source_prompt", type=str, default=None,
                   help="Source (original) text prompt (required for --single_video)")
    p.add_argument("--target_prompt", type=str, default=None,
                   help="Target (desired) text prompt (required for --single_video)")
    p.add_argument("--neg_prompt", type=str, default=None,
                   help="Negative prompt for unconditional baseline")

    # ---- output ----
    p.add_argument("--output_dir", type=str, required=True)

    # ---- latent intervention ----
    p.add_argument("--method", type=int, choices=[1, 2, 3], default=3,
                   help="Method variant (1/2/3)")
    p.add_argument("--transition_step", type=int, default=8,
                   help="Step index where Phase 1 → Phase 2 transition occurs")
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

    # ---- diffusion ----
    p.add_argument("--num_inference_steps", type=int, default=50,
                   help="Total number of denoising steps")

    # ---- device ----
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--gpu_ids", type=int, nargs="+", default=None,
                   help="Multiple GPU IDs to distribute models across (e.g., --gpu_ids 0 1 2 3). "
                        "Distribution: GPU0=foley, GPU1=dac+clap, GPU2=siglip2, GPU3=syncformer. "
                        "Supports 2-4 GPUs.")
    p.add_argument("--enable_offload", action="store_true",
                   help="Enable model offloading to reduce peak VRAM")

    # ---- misc ----
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_level", type=str, default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = p.parse_args()

    # Validation
    if args.single_video:
        if not args.source_prompt or not args.target_prompt:
            p.error("--source_prompt and --target_prompt are required "
                    "when using --single_video")

    # Auto-select config
    config_mapping = {"xl": "configs/hunyuanvideo-foley-xl.yaml",
                      "xxl": "configs/hunyuanvideo-foley-xxl.yaml"}
    if not args.config_path:
        args.config_path = config_mapping[args.model_size]

    return args


def setup_device(device_str, gpu_id=0):
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device(f"cuda:{gpu_id}")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device_str == "cuda":
        return torch.device(f"cuda:{gpu_id}")
    return torch.device(device_str)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    set_manual_seed(args.seed)

    logger.remove()
    logger.add(lambda msg: print(msg, end=""), level=args.log_level)

    device = setup_device(args.device, args.gpu_id)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load models
    if args.gpu_ids:
        logger.info(f"Multi-GPU mode: gpu_ids={args.gpu_ids}")
        logger.info("  GPU distribution: GPU0=foley | GPU1=dac,clap | GPU2=siglip2 | GPU3=syncformer")
    logger.info("Loading models…")
    model_dict, cfg = load_model(
        args.model_path, args.config_path, device,
        enable_offload=args.enable_offload,
        model_size=args.model_size,
        gpu_ids=args.gpu_ids,
    )

    # Shared intervention kwargs
    intervention_kwargs = dict(
        method=args.method,
        transition_step=args.transition_step,
        sde_sigma=args.sde_sigma,
        phase1_sde=args.phase1_sde,
        phase2_sde=args.phase2_sde,
        w_vid=args.w_vid,
        w_txt=args.w_txt,
        w_cfg=args.w_cfg,
        num_inference_steps=args.num_inference_steps,
        neg_prompt=args.neg_prompt,
    )

    if args.single_video:
        logger.info(f"Single video: {args.single_video}")
        logger.info(f"  source_prompt: {args.source_prompt}")
        logger.info(f"  target_prompt: {args.target_prompt}")

        audio, sr = infer_intervention(
            args.single_video, args.source_prompt, args.target_prompt,
            model_dict, cfg, **intervention_kwargs,
        )

        video_name = os.path.splitext(os.path.basename(args.single_video))[0]
        out_audio = os.path.join(args.output_dir, f"{video_name}_intervention.wav")
        out_video = os.path.join(args.output_dir, f"{video_name}_intervention.mp4")

        torchaudio.save(out_audio, audio, sr)
        logger.info(f"Audio saved: {out_audio}")

        merge_audio_video(out_audio, args.single_video, out_video)
        logger.info(f"Video saved: {out_video}")
    else:
        generate_audio_intervention(
            model_dict, cfg, args.csv_path, args.output_dir,
            **intervention_kwargs,
        )

    logger.info("Done!")


if __name__ == "__main__":
    main()
