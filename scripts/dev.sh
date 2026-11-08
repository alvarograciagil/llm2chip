#!/usr/bin/env bash
# Starts the local engine and the web interface together.
set -euo pipefail
export PATH="/opt/homebrew/opt/rustup/bin:$HOME/.cargo/bin:$PATH"
cd "$(dirname "$0")/.."
./scripts/build-wasm.sh
cargo build --release -p server
./target/release/server &
ENGINE=$!
trap 'kill $ENGINE 2>/dev/null || true' EXIT
cd web
npx vite
