#!/usr/bin/env bash
# Homebrew's rustup is keg-only, so cargo is not on PATH by default.
set -euo pipefail
export PATH="/opt/homebrew/opt/rustup/bin:$HOME/.cargo/bin:$PATH"
cd "$(dirname "$0")/.."
wasm-pack build crates/wasm --target web --out-dir ../../web/src/wasm --no-typescript --release
