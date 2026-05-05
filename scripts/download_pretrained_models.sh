#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mkdir -p "${ROOT_DIR}/pretrained/mmaudio"

cat <<'MSG'
Pretrained model downloads are intentionally not automated yet.

Place model files under:
  pretrained/mmaudio/

Checkpoints are ignored by Git.
MSG
