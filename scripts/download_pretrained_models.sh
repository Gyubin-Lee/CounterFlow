#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

AV_BENCH_WEIGHTS="${ROOT_DIR}/external/av-benchmark/weights"

mkdir -p \
  "${AV_BENCH_WEIGHTS}" \
  "${ROOT_DIR}/pretrained/mmaudio" \
  "${ROOT_DIR}/pretrained/hunyuan-video-foley"

download_if_missing() {
  local url="$1"
  local output="$2"

  if [[ -s "${output}" ]]; then
    echo "[skip] ${output}"
    return
  fi

  echo "[download] ${url}"
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 3 -o "${output}" "${url}"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "${output}" "${url}"
  else
    echo "curl or wget is required to download checkpoints." >&2
    exit 1
  fi
}

download_if_missing \
  "https://huggingface.co/lukewys/laion_clap/resolve/main/music_speech_audioset_epoch_15_esc_89.98.pt" \
  "${AV_BENCH_WEIGHTS}/music_speech_audioset_epoch_15_esc_89.98.pt"

download_if_missing \
  "https://github.com/hkchengrex/MMAudio/releases/download/v0.1/synchformer_state_dict.pth" \
  "${AV_BENCH_WEIGHTS}/synchformer_state_dict.pth"

cat <<MSG
Evaluation checkpoints are ready under:
  ${AV_BENCH_WEIGHTS}

MMAudio model checkpoints are downloaded automatically by the MMAudio backend on first run.
Additional local model files can be placed under:
  pretrained/mmaudio/
  pretrained/hunyuan-video-foley/

Checkpoints are ignored by Git.
MSG
