#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXTERNAL_DIR="${ROOT_DIR}/external"

mkdir -p "${EXTERNAL_DIR}"

clone_if_missing() {
  local name="$1"
  local url="$2"
  local target="${EXTERNAL_DIR}/${name}"

  if [[ -d "${target}" ]]; then
    echo "[skip] ${target} already exists"
    return
  fi

  echo "[clone] ${url} -> ${target}"
  git clone "${url}" "${target}"
}

clone_if_missing "MMAudio" "https://github.com/hkchengrex/MMAudio"
clone_if_missing "HunyuanVideo-Foley" "https://github.com/Tencent-Hunyuan/HunyuanVideo-Foley"
clone_if_missing "av-benchmark" "https://github.com/hkchengrex/av-benchmark"

install_if_changed() {
  local src="$1"
  local dst="$2"

  if [[ -f "${src}" ]] && { [[ ! -f "${dst}" ]] || ! cmp -s "${src}" "${dst}"; }; then
    echo "[install] ${dst}"
    cp "${src}" "${dst}"
  fi
}

install_if_changed \
  "${ROOT_DIR}/counterflow/mmaudio/eval_vggsound_sparse.py" \
  "${EXTERNAL_DIR}/MMAudio/eval_vggsound_sparse.py"

install_if_changed \
  "${ROOT_DIR}/counterflow/hunyuan/eval_vggsound_sparse.py" \
  "${EXTERNAL_DIR}/HunyuanVideo-Foley/eval_vggsound_sparse.py"

HUNYUAN_SCRIPT_SRC="${ROOT_DIR}/counterflow/hunyuan/infer_latent_intervention.py"
HUNYUAN_SCRIPT_DST="${EXTERNAL_DIR}/HunyuanVideo-Foley/infer_latent_intervention.py"

install_if_changed "${HUNYUAN_SCRIPT_SRC}" "${HUNYUAN_SCRIPT_DST}"

echo "External repository setup complete."
