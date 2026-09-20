[CmdletBinding()]
param(
    [switch]$Demo,
    [int]$Seed = 7,
    [ValidateSet('empty', 'pillars')]
    [string]$Layout = 'pillars'
)
$ErrorActionPreference = 'Stop'
$OutputEncoding = [Console]::OutputEncoding = [Text.UTF8Encoding]::new()
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONDONTWRITEBYTECODE = '1'
$Arguments = @('-m', 'isaac_room', '--seed', "$Seed", '--layout', $Layout)
if ($Demo) { $Arguments += '--demo' }
Push-Location -LiteralPath $PSScriptRoot
try {
    & python @Arguments
    $Code = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $Code
