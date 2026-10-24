[CmdletBinding()]
param([int]$Port=8764, [switch]$QA)
$ErrorActionPreference='Stop'
$OutputEncoding=[Console]::OutputEncoding=[Text.UTF8Encoding]::new()
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONPATH="$PSScriptRoot\bridge\python;$PSScriptRoot\runs\cuda-deps;$PSScriptRoot\runs\training-deps"
$Arguments=@("$PSScriptRoot\bridge\python\record_human.py",'--port',"$Port")
if($QA){$Arguments+='--qa'}
& python @Arguments
exit $LASTEXITCODE
