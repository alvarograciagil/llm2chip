# Starts the local engine and the web interface together on Windows.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
& "$PSScriptRoot/build-wasm.ps1"
cargo build --release -p server
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$engine = Start-Process -FilePath "target/release/server.exe" -PassThru -NoNewWindow
try {
  Set-Location web
  npx vite
} finally {
  Stop-Process -Id $engine.Id -ErrorAction SilentlyContinue
}
