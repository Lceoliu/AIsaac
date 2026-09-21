param([switch]$Smoke)
$ErrorActionPreference = 'Stop'
$OutputEncoding = [Console]::OutputEncoding = [Text.UTF8Encoding]::new()
$repo = Split-Path $PSScriptRoot
$workspace = Split-Path (Split-Path $repo)
$env:CARGO_HOME = Join-Path $workspace '.tooling/cargo'
$env:RUSTUP_HOME = Join-Path $workspace '.tooling/rustup'
$env:PATH = "$env:CARGO_HOME/bin;$env:RUSTUP_HOME/toolchains/stable-x86_64-pc-windows-gnu/lib/rustlib/x86_64-pc-windows-gnu/bin;$env:PATH"
$env:PYTHONIOENCODING = 'utf-8'
$out = Join-Path $repo 'runs/motion-viewer'
New-Item -ItemType Directory -Force $out | Out-Null
& cargo +stable-x86_64-pc-windows-gnu build --release --manifest-path "$PSScriptRoot/Cargo.toml" --example motion_trace *> "$out/build.log"
if ($LASTEXITCODE -ne 0) { Get-Content "$out/build.log" -Tail 30; exit $LASTEXITCODE }
$viewerArgs = @("$PSScriptRoot/tools/play_motion.py", '--exe', "$PSScriptRoot/target/release/examples/motion_trace.exe")
if ($Smoke) { $viewerArgs += @('--smoke', "$out/preview.png") }
& python @viewerArgs
exit $LASTEXITCODE
