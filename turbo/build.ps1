[CmdletBinding()]
param(
    [switch]$SkipTests,
    [switch]$SkipVerify
)

# Build isaac_turbo.dll (x86), the injector, and the offline tests with the workspace zig toolchain.
# Nothing here touches the game directory; verify_targets.py only reads isaac-ng.exe.

$ErrorActionPreference = 'Stop'
$OutputEncoding = [Console]::OutputEncoding = [Text.UTF8Encoding]::new()

$Project = $PSScriptRoot
$IsaacRoot = Split-Path -Parent (Split-Path -Parent $Project)
$FortuneRoot = Split-Path -Parent $IsaacRoot
$Zig = Join-Path $FortuneRoot '.tools\zig-x86_64-windows-0.16.0\zig.exe'
if (-not (Test-Path -LiteralPath $Zig)) { throw "Zig 0.16.0 not found: $Zig" }
$env:ZIG_GLOBAL_CACHE_DIR = Join-Path $FortuneRoot '.tooling\zig-global-cache'
$env:ZIG_LOCAL_CACHE_DIR = Join-Path $Project 'build\zig-local-cache'
New-Item -ItemType Directory -Force -Path $env:ZIG_GLOBAL_CACHE_DIR, $env:ZIG_LOCAL_CACHE_DIR | Out-Null

$MinHookRoot = Join-Path $IsaacRoot 'third_party\minhook'
if (-not (Test-Path -LiteralPath (Join-Path $MinHookRoot 'include\MinHook.h'))) {
    throw "MinHook not found: $MinHookRoot"
}
$Build = Join-Path $Project 'build'
New-Item -ItemType Directory -Force -Path $Build | Out-Null

& python (Join-Path $Project 'scripts\gen_targets.py')
if ($LASTEXITCODE -ne 0) { throw 'target header generation failed' }

# MinHook objects (x86)
$MinHookObjDir = Join-Path $Build 'minhook'
New-Item -ItemType Directory -Force -Path $MinHookObjDir | Out-Null
$MinHookObjects = @()
foreach ($Relative in @('buffer.c', 'hook.c', 'trampoline.c', 'hde\hde32.c')) {
    $Source = Join-Path (Join-Path $MinHookRoot 'src') $Relative
    $Object = Join-Path $MinHookObjDir ((($Relative -replace '[\\/]', '_') -replace '\.c$', '.o'))
    & $Zig cc -O2 -target x86-windows-gnu `
        -I (Join-Path $MinHookRoot 'include') -I (Join-Path $MinHookRoot 'src') `
        -c $Source -o $Object
    if ($LASTEXITCODE -ne 0) { throw "MinHook build failed: $Relative" }
    $MinHookObjects += $Object
}

# DLL
$Dll = Join-Path $Build 'isaac_turbo.dll'
$DllArgs = @('c++', '-std=c++17', '-O2', '-Wall', '-Wextra', '-Wno-nullability-completeness',
    '-target', 'x86-windows-gnu', '-shared', '-mstackrealign',
    '-I', (Join-Path $Project 'src'), '-I', (Join-Path $MinHookRoot 'include'),
    (Join-Path $Project 'src\turbo.cpp')) + $MinHookObjects + @('-o', $Dll)
& $Zig @DllArgs
if ($LASTEXITCODE -ne 0) { throw 'isaac_turbo.dll build failed' }
Write-Output "BUILD DLL PASS $Dll"

# Injector (x86, unicode entry)
$Injector = Join-Path $Build 'isaac_turbo_inject.exe'
& $Zig c++ -std=c++17 -O2 -Wno-nullability-completeness -target x86-windows-gnu -municode `
    -I (Join-Path $Project 'src') (Join-Path $Project 'src\injector.cpp') -o $Injector
if ($LASTEXITCODE -ne 0) { throw 'injector build failed' }
Write-Output "BUILD INJECTOR PASS $Injector"

if (-not $SkipTests) {
    $ClockTest = Join-Path $Build 'clock_test.exe'
    & $Zig c++ -std=c++17 -O2 -Wall -Wextra -Wno-nullability-completeness -target x86-windows-gnu `
        (Join-Path $Project 'tests\clock_test.cpp') -o $ClockTest
    if ($LASTEXITCODE -ne 0) { throw 'clock_test build failed' }
    & $ClockTest
    if ($LASTEXITCODE -ne 0) { throw 'clock_test failed' }

    $RegistryTest = Join-Path $Build 'registry_test.exe'
    & $Zig c++ -std=c++17 -O2 -Wall -Wextra -Wno-nullability-completeness -target x86-windows-gnu `
        (Join-Path $Project 'tests\registry_test.cpp') -o $RegistryTest
    if ($LASTEXITCODE -ne 0) { throw 'registry_test build failed' }
    & $RegistryTest
    if ($LASTEXITCODE -ne 0) { throw 'registry_test failed' }

    $Probe = Join-Path $Build 'probe_host.exe'
    & $Zig c++ -std=c++17 -O2 -Wall -Wextra -Wno-nullability-completeness -target x86-windows-gnu `
        (Join-Path $Project 'tests\probe_host.cpp') -o $Probe
    if ($LASTEXITCODE -ne 0) { throw 'probe_host build failed' }
    $env:ISAAC_TURBO_LOG_DIR = Join-Path $Build 'probe-logs'
    & $Probe $Dll
    if ($LASTEXITCODE -ne 0) { throw 'probe_host failed' }
    Remove-Item Env:ISAAC_TURBO_LOG_DIR
}

if (-not $SkipVerify) {
    & python (Join-Path $Project 'scripts\verify_targets.py')
    if ($LASTEXITCODE -eq 1) { throw 'J460 target bytes do not match isaac-ng.exe' }
}
Write-Output 'BUILD ALL PASS'
