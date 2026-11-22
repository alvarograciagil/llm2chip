#!/usr/bin/env bash
# The model served with the online demo: Andrej Karpathy's stories260K (llama2.c, MIT),
# downloaded at a fixed commit, checked and converted, so no weights live in the repository.
set -euo pipefail
cd "$(dirname "$0")/.."
root=$PWD
source=https://huggingface.co/karpathy/tinyllamas/resolve/0bd21da7698eaf29a0d7de3992de8a46ef624add/stories260K
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
cd "$work"
curl -fsSL -o stories260K.bin "$source/stories260K.bin"
curl -fsSL -o tok512.bin "$source/tok512.bin"
sha256sum -c - <<'SUMS'
b0a507e7ad0f626624f17112325e66691f9076d622e1d3274d103d00299f2696  stories260K.bin
037cb335abb25d1fa9e8ecae30ed2a3a8ace9302862ebcdc05d51a6bbb10c312  tok512.bin
SUMS
python3 -I "$root/tools/convert/llama2c.py" stories260K.bin tok512.bin "$root/web/public/models/stories260K"
