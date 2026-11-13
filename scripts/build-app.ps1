# Builds the desktop application on Windows.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
& "$PSScriptRoot/build-wasm.ps1"
npm --prefix web run build
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
cargo build --release -p server
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Set-Location app/src-tauri
cargo tauri build
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Output "Packages in: $(Get-Location)/target/release/bundle"
