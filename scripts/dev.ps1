# Starts the local engine and the web interface together on Windows.
# With -Lan the interface is also reachable from the local network.
param([switch]$Lan, [string]$Argument = "")
$ErrorActionPreference = "Stop"
if ($Lan -or $Argument -eq "--lan") { $env:LLM2CHIP_LAN = "1" }
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
