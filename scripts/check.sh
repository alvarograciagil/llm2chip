#!/usr/bin/env bash
# Local CI: Rust tests and lints, then the web build and tests.
set -euo pipefail
export PATH="/opt/homebrew/opt/rustup/bin:$HOME/.cargo/bin:$PATH"
cd "$(dirname "$0")/.."
cargo test --workspace --exclude llm2chip-wasm
cargo clippy --workspace --all-targets -- -D warnings
npm --prefix web run build
npm --prefix web test
