# Local CI on Windows: Rust tests and lints, then the web build and tests.
#
#   scripts\check.ps1             the quick set, a couple of minutes: for every change
#   scripts\check.ps1 -Full       also the whole chips in gates and the slow tests: before
#                                 changes to the hardware go up
param([switch]$Full)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
if ($Full) {
    cargo test --release --workspace --exclude llm2chip-wasm -- --include-ignored
} else {
    cargo test --release --workspace --exclude llm2chip-wasm
}
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
cargo clippy --release --workspace --all-targets -- -D warnings
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
npm --prefix web run build
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
npm --prefix web test
exit $LASTEXITCODE
