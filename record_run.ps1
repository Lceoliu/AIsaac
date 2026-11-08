# 整局录制：人类会话入口（Repentance v1.7.8a.J746 离线 GOG 构建，隔离副本 experiments\run-recorder\game-rec-01）。
#
# 用法：
#   .\rl\record_run.ps1 -SessionName 20261106-evening
#   .\rl\record_run.ps1 -SessionName first-try -InitSaveFrom D:\backup\my-galaxy-save
#   .\rl\record_run.ps1 -SessionName old-way -NoDll          # 回退到第二阶段行为（不注入 DLL）
#
# 流程（DESIGN_BRIEF §5 末条、§6 D5/D6/D11、§7 E1/E2）：
#   1. 检查/复用隔离副本，安装最新 rl\bridge\mod\isaac_run_recorder；默认经挂起式注入器
#      <副本>\rec\rr_launch.exe 加载最小 DLL <副本>\rec\rr_mt.dll（先运行一次
#      python rl\bridge\engine\j746_rec\build.py 构建），游戏以 --luadebug 启动：
#      存档重定向到副本内 <副本>\profile\Documents\My Games\Binding of Isaac Repentance (Galaxy)\，
#      mod 每局开局用随机 salt reseed 全局 MT 并逐帧记录 MT 校验值，录下的局可精确回放；
#      游戏进程默认限制到 1 个逻辑 CPU（-Affinity auto；phase8-mouse：多核并行时引擎的环境特效偶发不确定，
#      录制时发生过就无法精确回放；-Affinity 0 不限制）；
#      -NoDll：不注入，存档在真实 %USERPROFILE%\Documents\My Games\Binding of Isaac Repentance (Galaxy)\，回放只能近似；
#   2. 可选：-InitSaveFrom 把一份存档目录拷贝进本会话的存档目录（上面两者之一；只拷贝，不写回来源）；
#   3. 会话前快照存档目录（排除 *.dmp、log.txt）；options.ini 强制 EnableDebugConsole=0（会话后还原），
#      MouseControl 保留你的设置（mod 另记鼠标流）；
#   4. 启动游戏，启动失败（约 20–25%：崩溃、log 洪水、mod 未加载）自动重试并计数；
#   5. 正常玩：mod 每局一个文件，开局时 Python 再快照一次存档；画面左上角录制期间常驻一个小 "REC"；
#   6. 关闭游戏窗口后：会话后快照、每局压成 .zst 并逐字节校验、写 session.json 与 summary.txt
#      （局数、帧数、小时、MB）。原始 jsonl 校验后移到 <OutRoot>\_raw_staging\<会话名>\，确认无误后可自行删除。
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$SessionName,
    [string]$InitSaveFrom = '',
    [string]$OutRoot = '',
    [ValidateSet('stage', 'keep')][string]$Raw = 'stage',
    [int]$Attempts = 6,
    [double]$AutoSeconds = 0,  # >0：脚本自动按键进游戏、随机玩、按住 R 重开、关窗口（只用于测试）
    [switch]$NoDll,            # 不注入 rr_mt.dll（第二阶段行为）
    [ValidateSet('full', 'keyframe')][string]$Obs = 'full',  # 观测模式：full = 每帧完整（默认）；keyframe = 实体只在关键帧写
    [string]$Affinity = 'auto'  # 游戏进程 CPU 亲和：auto = 逻辑 CPU 0 一个核（默认，录下的局才可能精确回放）；0 = 不限制；0x… = 掩码
)
$ErrorActionPreference = 'Stop'
$OutputEncoding = [Console]::OutputEncoding = [Text.UTF8Encoding]::new()
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONPATH = "$PSScriptRoot\bridge\python"
if (-not $OutRoot) { $OutRoot = Join-Path (Split-Path $PSScriptRoot -Parent) 'experiments\run-recorder\sessions' }
$OutRoot = [IO.Path]::GetFullPath($OutRoot)
if ($InitSaveFrom -and -not (Test-Path -LiteralPath $InitSaveFrom -PathType Container)) {
    throw "InitSaveFrom 不是目录：$InitSaveFrom"
}
$Running = Get-Process -Name 'isaac-ng_repentance' -ErrorAction SilentlyContinue |
    Where-Object { $_.Path -like '*game-rec-01*' }
if ($Running) { throw "game-rec-01 的游戏进程已在运行（pid $($Running.Id -join ',')），先关掉再开始新会话" }

$Arguments = @('-m', 'isaac_bridge.run_recording', '--session', $SessionName, '--out-root', $OutRoot,
    '--raw', $Raw, '--attempts', "$Attempts")
if ($InitSaveFrom) { $Arguments += @('--init-save-from', ([IO.Path]::GetFullPath($InitSaveFrom))) }
if ($AutoSeconds -gt 0) { $Arguments += @('--auto-seconds', "$AutoSeconds") }
if ($NoDll) { $Arguments += @('--no-dll') }
if ($Obs -ne 'full') { $Arguments += @('--obs', $Obs) }
if ($Affinity -ne 'auto') { $Arguments += @('--affinity', $Affinity) }
& python @Arguments
exit $LASTEXITCODE
