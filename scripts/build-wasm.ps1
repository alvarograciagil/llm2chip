# Builds the WebAssembly engine on Windows.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
wasm-pack build crates/wasm --target web --out-dir ../../web/src/wasm --no-typescript --release
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
# The chip simulator for the online demo, a bundle of its own loaded only to run a model.
wasm-pack build crates/wasm-sim --target web --out-dir ../../web/src/wasm-sim --no-typescript --release
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
