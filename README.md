# CounterFlow

CounterFlow is a research repository for CounterFlow-specific code, experiment wrappers, evaluation scripts, and project documentation. Third-party repositories are kept out of the project code path and are installed under `external/`.

## Repository Layout

```text
counterflow/      CounterFlow wrappers and project-owned code
external/         Cloned third-party repositories, ignored by Git
baselines/        Baseline repositories such as CAFA and ReWaS, ignored by Git
datasets/         Local datasets and feature caches, ignored by Git
pretrained/       Local model checkpoints, ignored by Git
experiments/      Experiment entry points
evaluation/       Evaluation code only
results/          Generated experiment and evaluation outputs, ignored by Git
scripts/          Setup and utility scripts
patches/          Reproducible patches for external repositories
```

## Setup

Create the CounterFlow-level environment dependencies from `requirements.txt` if needed. Backend-specific dependencies are managed by the external repositories and their Conda environments.

```bash
pip install -r requirements.txt
bash scripts/setup_external_repos.sh
```

This repository contains only CounterFlow-level requirements. MMAudio and av-benchmark may require their own dependencies. Please refer to each external repository for backend-specific setup.

## External Repositories

External repositories live under:

```text
external/MMAudio/
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

CounterFlow backend entry scripts are stored under `counterflow/mmaudio/`. The setup script copies them into MMAudio when the corresponding external files are missing or outdated.

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
# cu118 is a conservative default used by upstream MMAudio docs.
python -m pip install \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
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

If `git apply` reports that the MMAudio patch is already applied, keep going. If packages installed in `~/.local` still leak into your Conda environment, keep `PYTHONNOUSERSITE=1` set while running inference and evaluation commands.

## Pretrained Models

Do not commit pretrained model files. Place them under:

```text
pretrained/mmaudio/
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

```bash
conda activate MMAudio
python experiments/exp_vggsound_sparse.py \
  --exp-name 2026-05-05_counterflow-mmaudio-default \
  --csv_path datasets/VGGSound-Sparse/vggsound_sparse.csv \
  --clean_csv_path datasets/VGGSound-Sparse/vggsound_sparse_clean_fixed_offsets.csv \
  --video_root /path/to/vggsound/video
```

Use `--dry-run` to print the backend command without launching inference.

For a quick clean-subset smoke test, use `--pilot --pilot_n 3` with `--subset clean`:

```bash
conda activate MMAudio
CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 python experiments/exp_vggsound_sparse.py \
  --exp-name smoke_mmaudio_clean3 \
  --subset clean \
  --pilot \
  --pilot_n 3 \
  --gpu 0 \
  --csv_path datasets/VGGSound-Sparse/vggsound_sparse.csv \
  --clean_csv_path datasets/VGGSound-Sparse/vggsound_sparse_clean_fixed_offsets.csv \
  --video_root /path/to/vggsound/video
```

## Demos

Run the public CounterFlow prompt-switch demo with the tracked videos:

```bash
conda activate MMAudio
export PYTHONNOUSERSITE=1
CUDA_VISIBLE_DEVICES=0 python run_counterflow_demo.py --gpu 0
```

This demo uses:

```text
datasets/demo_videos/cat.mp4: source prompt "cat meowing" -> target prompt "horse neighing"
datasets/demo_videos/dog.mp4: source prompt "dog barking" -> target prompt "bear growling"
```

Outputs are written under:

```text
results/demo/counterflow-cat-dog/
```

The default demo config is `cfg_text=5.0`, `transition_step=17`, Phase 1 ODE, Phase 2 ODE, `sigma=0.0`, and `seed=42`.

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
python demos/demo_CounterFlow.py --dry-run
```

## Evaluation

Quantitative VGGSound-Sparse evaluation should run in the `MMAudio` environment:

```bash
conda activate MMAudio
python evaluation/eval_vggsound_sparse_metrics.py \
  --output_dir results/evaluation/VGGSound-Sparse/qualitative/2026-05-05_counterflow-mmaudio-default \
  --filter_csv datasets/VGGSound-Sparse/vggsound_sparse_clean_fixed_offsets.csv
```

FLAM metric evaluation:

```bash
conda activate MMAudio
python evaluation/eval_flam_metric.py \
  --exp_dir results/evaluation/VGGSound-Sparse/qualitative/2026-05-05_counterflow-mmaudio-default \
  --filter_csv datasets/VGGSound-Sparse/vggsound_sparse_clean_fixed_offsets.csv
```

Qualitative analysis notebook:

```bash
jupyter notebook evaluation/qualitative_analysis.ipynb
```

## Results Convention

Store quantitative outputs under:

```text
results/evaluation/VGGSound-Sparse/quantitative/YYYY-MM-DD_model-dataset-setting/
```

Store qualitative samples, figures, and inspection artifacts under:

```text
results/evaluation/VGGSound-Sparse/qualitative/YYYY-MM-DD_model-dataset-setting/
```

Generated results are ignored by Git. Keep only `results/README.md` tracked.

## Git-Ignored Files

The `.gitignore` excludes Python caches, local environments, logs, checkpoints, datasets, pretrained models, generated samples, evaluation outputs, large media files, baseline repositories, and cloned external repositories.
