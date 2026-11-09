# Builds the WebAssembly engine on Windows.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
wasm-pack build crates/wasm --target web --out-dir ../../web/src/wasm --no-typescript --release
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
