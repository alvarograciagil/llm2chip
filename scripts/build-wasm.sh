#!/usr/bin/env bash
# Homebrew's rustup is keg-only, so cargo is not on PATH by default.
set -euo pipefail
export PATH="/opt/homebrew/opt/rustup/bin:$HOME/.cargo/bin:$PATH"
cd "$(dirname "$0")/.."
wasm-pack build crates/wasm --target web --out-dir ../../web/src/wasm --no-typescript --release
# The chip simulator for the online demo, a bundle of its own loaded only to run a model.
wasm-pack build crates/wasm-sim --target web --out-dir ../../web/src/wasm-sim --no-typescript --release
# With --threads, also the simulator on several threads, for pages served cross-origin
# isolated (a Hugging Face Space): WebAssembly threads still need a nightly toolchain, pinned here.
if [[ "${1:-}" == "--threads" ]]; then
  NIGHTLY=nightly-2026-11-20
  rustup toolchain install "$NIGHTLY" --profile minimal --component rust-src --target wasm32-unknown-unknown
  # A build directory of its own: its standard library and flags differ from every other build.
  # The memory is shared with the workers, up to 4 GB, and each thread gets its own TLS.
  link="--shared-memory --import-memory --max-memory=4294967296"
  link+=" --export=__wasm_init_tls --export=__tls_size --export=__tls_align --export=__tls_base"
  CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$PWD/target}/wasm-threads" CARGO_UNSTABLE_BUILD_STD=panic_abort,std \
  RUSTFLAGS="-C target-feature=+atomics,+bulk-memory $(printf -- '-C link-arg=%s ' $link)" \
    rustup run "$NIGHTLY" wasm-pack build crates/wasm-sim --target web --out-dir ../../web/src/wasm-sim-threads \
    --no-typescript --release --features threads
fi
