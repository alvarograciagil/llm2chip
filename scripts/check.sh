#!/usr/bin/env bash
# Local CI: Rust tests and lints, then the web build and tests.
#
#   scripts/check.sh             the quick set, a couple of minutes: for every change
#   scripts/check.sh --completa  also the whole chips on gates, about half an hour: before
#                                changes to the hardware go up
set -euo pipefail
export PATH="/opt/homebrew/opt/rustup/bin:$HOME/.cargo/bin:$PATH"
cd "$(dirname "$0")/.."
slow=()
[ "${1:-}" = "--completa" ] && slow=(-- --include-ignored)
cargo test --release --workspace --exclude llm2chip-wasm ${slow[@]+"${slow[@]}"}
cargo clippy --release --workspace --all-targets -- -D warnings
npm --prefix web run build
npm --prefix web test
