#!/usr/bin/env bash
# Starts the local engine and the web interface together.
# With --lan the interface is also reachable from the local network.
set -euo pipefail
if [[ "${1:-}" == "--lan" ]]; then
  export LLM2CHIP_LAN=1
fi
export PATH="/opt/homebrew/opt/rustup/bin:$HOME/.cargo/bin:$PATH"
cd "$(dirname "$0")/.."
./scripts/build-wasm.sh
cargo build --release -p server
./target/release/server &
ENGINE=$!
trap 'kill $ENGINE 2>/dev/null || true' EXIT
cd web
npx vite
