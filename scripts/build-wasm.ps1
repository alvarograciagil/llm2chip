# Builds the WebAssembly engine on Windows.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
wasm-pack build crates/wasm --target web --out-dir ../../web/src/wasm --no-typescript --release
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
# The chip simulator for the online demo, a bundle of its own loaded only to run a model.
wasm-pack build crates/wasm-sim --target web --out-dir ../../web/src/wasm-sim --no-typescript --release
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
# With -Threads, also the simulator on several threads, for pages served cross-origin
# isolated (a Hugging Face Space): WebAssembly threads still need a nightly toolchain, pinned here.
if ($args -contains "-Threads") {
  $nightly = "nightly-2026-11-20"
  rustup toolchain install $nightly --profile minimal --component rust-src --target wasm32-unknown-unknown
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
  # A build directory of its own: its standard library and flags differ from every other build.
  $target = if ($env:CARGO_TARGET_DIR) { $env:CARGO_TARGET_DIR } else { Join-Path (Get-Location) "target" }
  $previous = $env:CARGO_TARGET_DIR
  $env:CARGO_TARGET_DIR = Join-Path $target "wasm-threads"
  $env:CARGO_UNSTABLE_BUILD_STD = "panic_abort,std"
  # The memory is shared with the workers, up to 4 GB, and each thread gets its own TLS.
  $link = "--shared-memory", "--import-memory", "--max-memory=4294967296", "--export=__wasm_init_tls", "--export=__tls_size", "--export=__tls_align", "--export=__tls_base"
  $env:RUSTFLAGS = "-C target-feature=+atomics,+bulk-memory " + (($link | ForEach-Object { "-C link-arg=$_" }) -join " ")
  rustup run $nightly wasm-pack build crates/wasm-sim --target web --out-dir ../../web/src/wasm-sim-threads --no-typescript --release --features threads
  $code = $LASTEXITCODE
  Remove-Item Env:CARGO_UNSTABLE_BUILD_STD, Env:RUSTFLAGS
  $env:CARGO_TARGET_DIR = $previous
  if ($code -ne 0) { exit $code }
}
