# CounterFlow

CounterFlow is a research repository for CounterFlow-specific code, experiment wrappers, evaluation scripts, and project documentation. Third-party repositories are kept out of the project code path and are installed under `external/`.

## Repository Layout

```text
counterflow/      CounterFlow wrappers and project-owned code
external/         Cloned third-party repositories, ignored by Git
baselines/        Baseline repositories such as CAFA and ReWaS, ignored by Git
datasets/         Local datasets and feature caches, ignored by Git
pretrained/       Local model checkpoints, ignored by Git
experiments/      Experiment entry points grouped by research axis
evaluation/       Evaluation code only
results/          Generated experiment and evaluation outputs, ignored by Git
scripts/          Shared utility scripts and research-axis runners
configs/          Lightweight project-owned configs grouped by research axis
patches/          Reproducible patches for external repositories
```

Private research notes live under `docs/` in the private workspace and are not
included in public exports.

## Setup

Create the CounterFlow-level environment dependencies from `requirements.txt` if needed. Backend-specific dependencies are managed by the external repositories and their Conda environments.

```bash
pip install -r requirements.txt
bash scripts/setup_external_repos.sh
```

This repository contains only CounterFlow-level requirements. MMAudio, HunyuanVideo-Foley, and av-benchmark may require their own dependencies. Please refer to each external repository for backend-specific setup.

## External Repositories

External repositories live under:

```text
external/MMAudio/
external/HunyuanVideo-Foley/
external/av-benchmark/
```

Use:

```bash
bash scripts/setup_external_repos.sh
```

The script clones missing repositories without overwriting existing directories. If MMAudio needs the CounterFlow network changes, apply:

```bash
cd external/MMAudio
git apply ../../patches/mmaudio_networks_counterflow.patch
```

Apply the other backend patches only when needed:

```bash
cd external/HunyuanVideo-Foley
git apply ../../patches/hunyuanvideo_foley_counterflow.patch
```

```bash
cd external/av-benchmark
git apply ../../patches/av_benchmark_counterflow.patch
```

CounterFlow backend entry scripts are stored under `counterflow/mmaudio/` and `counterflow/hunyuan/`. The setup script copies them into the external repositories when the corresponding external files are missing or outdated. The Hunyuan latent-intervention script is stored at `counterflow/hunyuan/infer_latent_intervention.py` and is installed into `external/HunyuanVideo-Foley/infer_latent_intervention.py`.

## Conda Environments

Use the `MMAudio` Conda environment for MMAudio-based CounterFlow experiments and MMAudio inference:

```bash
conda activate MMAudio
```

Create the `MMAudio` environment from a clean Conda environment:

```bash
conda create -n MMAudio python=3.10 -y
conda activate MMAudio

# Recommended on shared machines so user-site packages under ~/.local do not
# shadow the Conda environment.
export PYTHONNOUSERSITE=1

python -m pip install --upgrade pip setuptools wheel

# Install the PyTorch wheel that matches your CUDA driver.
# The 2.6.0/cu118 stack works with MMAudio, av-benchmark, and optional OpenFLAM.
python -m pip install \
  torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu118

# ffmpeg is used when composing generated audio back into videos.
conda install -c conda-forge ffmpeg -y

# Clone external repositories if you have not done so yet.
bash scripts/setup_external_repos.sh

# Apply the CounterFlow network changes needed by the MMAudio backbone.
cd external/MMAudio
git apply ../../patches/mmaudio_networks_counterflow.patch
cd ../..

# Install CounterFlow-level helpers and MMAudio dependencies.
python -m pip install -r requirements.txt
python -m pip install -e external/MMAudio
```

Optional evaluation dependencies:

```bash
python -m pip install -e external/av-benchmark

# Needed only for evaluation/eval_flam_metric.py.
python -m pip install openflam
```

If `git apply` reports that the MMAudio patch is already applied, keep going. Avoid installing OpenFLAM before the PyTorch 2.6.0 stack above is in place; otherwise pip may upgrade the torch packages to a mismatched set. If packages installed in `~/.local` still leak into your Conda environment, keep `PYTHONNOUSERSITE=1` set while running inference and evaluation commands.

Use the `Hunyuan` Conda environment for HunyuanVideo-Foley-based CounterFlow experiments and inference:

```bash
conda activate Hunyuan
```

## Pretrained Models

Do not commit pretrained model files. MMAudio checkpoints are downloaded automatically by the MMAudio backend on first run. For CLAP and DeSync evaluation, prepare av-benchmark checkpoints with:

```bash
bash scripts/download_pretrained_models.sh
```

This downloads:

```text
external/av-benchmark/weights/music_speech_audioset_epoch_15_esc_89.98.pt
external/av-benchmark/weights/synchformer_state_dict.pth
```

Local or manually downloaded model files can also be placed under:

```text
pretrained/mmaudio/
pretrained/hunyuan-video-foley/
pretrained/etc/
```

Checkpoint-like files such as `*.ckpt`, `*.pt`, `*.pth`, `*.safetensors`, and `*.bin` are ignored by Git.

## Datasets

Do not commit datasets. Put local datasets and feature caches under:

```text
datasets/VGGSound-Sparse/
```

Small metadata files used for local experiments can live there as well, but the dataset directory is ignored by Git.

## Baselines

Baseline repositories are kept under:

```text
baselines/CAFA/
baselines/ReWaS/
```

These directories are ignored because they may contain downloaded code, checkpoints, generated outputs, and datasets. Add reproducible setup scripts later if baseline setup needs to be automated.

## Experiments

MMAudio backend:

```bash
conda activate MMAudio
python experiments/research_axes/latent_update_method/exp_vggsound_sparse.py --backend mmaudio --exp-name 20260505_counterflow_mmaudio_default
```

Hunyuan backend:

```bash
conda activate Hunyuan
python experiments/research_axes/latent_update_method/exp_vggsound_sparse.py --backend hunyuan --exp-name 20260505_counterflow_hunyuan_default
```

Use `--dry-run` to print the backend command without launching inference.

The default CounterFlow-MMAudio experiment config matches the public demo: `cfg_text=5.0`, `transition_step=17`, Phase 1 ODE, Phase 2 ODE, `sigma=0.0`, and `seed=42`.

## Demos

Run CounterFlow-MMAudio on a small local video manifest:

```bash
conda activate MMAudio
export PYTHONNOUSERSITE=1
CUDA_VISIBLE_DEVICES=0 python demos/demo_mmaudio_local_videos.py \
  --manifest datasets/demo_vggsound_sparse_2.csv \
  --output-dir results/demo/mmaudio-vggsound-sparse-2 \
  --gpu 0
```

The manifest must be a CSV with `video_path,prompt` columns. Paths may be absolute or relative to the repository root. Example:

```csv
video_path,prompt
datasets/demo_videos/example_000001.mp4,people eating crisps
datasets/demo_videos/example_000002.mp4,striking bowling
```

```bash
conda activate MMAudio
python demos/demo_CounterFlow.py --backend mmaudio --dry-run
```

```bash
conda activate Hunyuan
python demos/demo_CounterFlow.py --backend hunyuan --dry-run
```

## Evaluation

Quantitative VGGSound-Sparse evaluation should run in the `MMAudio` environment:

```bash
conda activate MMAudio
export PYTHONNOUSERSITE=1
python evaluation/eval_vggsound_sparse_metrics.py \
  --output_dir results/research_axes/latent_update_method/evaluation/VGGSound-Sparse/qualitative/20260505_counterflow_mmaudio_default \
  --filter_csv datasets/VGGSound-Sparse/vggsound_sparse_clean_fixed_offsets.csv \
  --gpu 0
```

The CLAP and DeSync metrics require the av-benchmark checkpoints downloaded by `scripts/download_pretrained_models.sh`. FAD is computed when `datasets/VGGSound-Sparse/gt_fad_stats.pt` exists or when a path is provided with `--gt_fad_stats`. DeSync uses `datasets/VGGSound-Sparse/test_video_features.pt` when present; otherwise it extracts video features from the generated mp4 files.

FLAM metric evaluation:

```bash
conda activate MMAudio
export PYTHONNOUSERSITE=1
python evaluation/eval_flam_metric.py \
  --exp_dir results/research_axes/latent_update_method/evaluation/VGGSound-Sparse/qualitative/20260505_counterflow_mmaudio_default \
  --filter_csv datasets/VGGSound-Sparse/vggsound_sparse_clean_fixed_offsets.csv \
  --gpu_id 0
```

## Results Convention

Store new quantitative outputs under:

```text
results/research_axes/<axis>/evaluation/<dataset>/quantitative/YYYYMMDD_model_dataset_setting/
```

Store new qualitative samples, figures, and inspection artifacts under:

```text
results/research_axes/<axis>/evaluation/<dataset>/qualitative/YYYYMMDD_model_dataset_setting/
```

Use `YYYYMMDD_<short_experiment_description>` for every new experiment folder.
Keep the part after the date in lower snake case so result folders sort and
match across prompt, media, metric, and log directories.

Use the axis-grouped paths above as the only canonical result locations. Avoid
creating legacy result aliases or duplicate result folders; update old
commands instead. Generated results are ignored by Git. Keep only
`results/README.md` tracked.

## Git-Ignored Files

The `.gitignore` excludes Python caches, local environments, logs, checkpoints, datasets, pretrained models, generated samples, evaluation outputs, large media files, baseline repositories, and cloned external repositories.
