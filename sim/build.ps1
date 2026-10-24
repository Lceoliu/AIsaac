# 用 D:\Projects\fortune\.tooling 下的便携 Rust 工具链构建/测试 isaac_sim（环境设置与 netfix/scripts/build_rust.ps1 相同）。
# 用法：.\build.ps1            # cargo test
#       .\build.ps1 build      # 任意 cargo 子命令与参数原样透传
[CmdletBinding()]
param([Parameter(ValueFromRemainingArguments = $true)][string[]]$CargoArgs)
$ErrorActionPreference = 'Stop'
$Sim = $PSScriptRoot
$FortuneRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $Sim))
$Tooling = Join-Path $FortuneRoot '.tooling'
$env:CARGO_HOME = Join-Path $Tooling 'cargo'
$env:RUSTUP_HOME = Join-Path $Tooling 'rustup'
$Toolchain = Join-Path $env:RUSTUP_HOME 'toolchains\stable-x86_64-pc-windows-gnu'
$RustBin = Join-Path $Toolchain 'lib\rustlib\x86_64-pc-windows-gnu\bin'
$env:PATH = "$(Join-Path $Tooling 'msys2-binutils\mingw64\bin');$(Join-Path $Tooling 'zig-rust-bin');$(Join-Path $Toolchain 'bin');$RustBin;$(Join-Path $RustBin 'self-contained');$(Join-Path $env:CARGO_HOME 'bin');$env:PATH"
if (-not (Test-Path -LiteralPath (Join-Path $env:CARGO_HOME 'bin\cargo.exe'))) { throw "cargo.exe 不存在：先运行 netfix\scripts\build_rust.ps1 -Bootstrap" }
if (-not $CargoArgs -or $CargoArgs.Count -eq 0) { $CargoArgs = @('test') }
Push-Location $Sim
try { & cargo @CargoArgs; if ($LASTEXITCODE -ne 0) { throw "cargo $($CargoArgs -join ' ') 失败，退出码 $LASTEXITCODE" } }
finally { Pop-Location }
