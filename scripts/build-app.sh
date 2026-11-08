#!/usr/bin/env bash
# Builds the macOS application bundle.
set -euo pipefail
export PATH="/opt/homebrew/opt/rustup/bin:$HOME/.cargo/bin:$PATH"
cd "$(dirname "$0")/.."
./scripts/build-wasm.sh
npm --prefix web run build
cargo build --release -p server
cd app/src-tauri
cargo tauri build --bundles app
echo "App: $(pwd)/target/release/bundle/macos/llm2chip.app"
