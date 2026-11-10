# 整局录制：人类会话入口（Repentance v1.7.8a.J746 离线 GOG 构建，隔离副本 experiments\run-recorder\game-rec-01）。
#
# 用法：
#   .\rl\record_run.ps1 -SessionName 20261106-evening
#   .\rl\record_run.ps1 -SessionName first-try -InitSaveFrom D:\backup\my-galaxy-save
#   .\rl\record_run.ps1 -SessionName old-way -NoDll          # 回退到第二阶段行为（不注入 DLL）
#   .\rl\record_run.ps1 -SessionName with-console -AllowConsole   # 允许调试控制台（命令逐条记录，回放同帧执行；README §8.7）
#   .\rl\record_run.ps1 -Recover D:\...\sessions\20261110-evening  # 收尾一个没收尾的会话（本窗口被关/电脑重启之后；先关游戏）
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
#   3. 会话前快照存档目录（排除 *.dmp、log.txt）；options.ini 强制 EnableDebugConsole=0（会话后还原；-AllowConsole 时不强制，
#      DLL 挂引擎的控制台提交函数，每条命令带帧号记进 run 文件），
#      MouseControl 保留你的设置（mod 另记鼠标流）；
#   4. 启动游戏，启动失败（约 20–25%：崩溃、log 洪水、mod 未加载）自动重试并计数；
#   5. 正常玩：mod 每局一个文件，开局时 Python 再快照一次存档；画面左上角录制期间常驻一个小 "REC"；
#   6. 关闭游戏窗口后：会话后快照、每局压成 .zst 并逐字节校验、写 session.json 与 summary.txt
#      （局数、帧数、小时、MB）。原始 jsonl（和 mod 的开局存档拷贝）压缩并逐字节校验通过后默认送进 Windows 回收站
#      （-Raw recycle；回收站不可用/会被永久删除时自动退回 stage 并在 session.json 里说明）；-Raw stage = 挪到
#      <OutRoot>\_raw_staging\<会话名>\；-Raw keep = 原地保留。校验没通过的局永远保留原始文件。
#   7. 本窗口被关、Python 被结束或电脑重启（会话没收尾：session.json 不完整、options.ini 的 EnableDebugConsole 没还原）：
#      关掉游戏后运行 -Recover <会话目录>：归档已录的局（截断局照常标出）、补写 session.json（state = recovered）、还原 options.ini。
#      新会话开始时若发现 OutRoot 下有没收尾的会话会打印一行提示。
[CmdletBinding()]
param(
    [string]$SessionName = '',
    [string]$Recover = '',     # 收尾一个没收尾的会话目录（见上面第 7 条）
    [string]$InitSaveFrom = '',
    [string]$OutRoot = '',
    [ValidateSet('recycle', 'stage', 'keep')][string]$Raw = 'recycle',  # 校验通过后原始文件去向（见上面第 6 条）
    [int]$Attempts = 6,
    [double]$AutoSeconds = 0,  # >0：脚本自动按键进游戏、随机玩、按住 R 重开、关窗口（只用于测试）
    [switch]$NoDll,            # 不注入 rr_mt.dll（第二阶段行为）
    [ValidateSet('full', 'keyframe', 'fxkey')][string]$Obs = 'full',  # 观测模式：full = 每帧完整（默认）；keyframe = 实体只在关键帧写；fxkey = 装饰性特效只在关键帧写（README §4.11）
    [switch]$AllowConsole,     # 允许调试控制台：不强制 EnableDebugConsole=0，控制台命令逐条记录（需要 DLL）
    [string]$Affinity = 'auto'  # 游戏进程 CPU 亲和：auto = 逻辑 CPU 0 一个核（默认，录下的局才可能精确回放）；0 = 不限制；0x… = 掩码
)
$ErrorActionPreference = 'Stop'
$OutputEncoding = [Console]::OutputEncoding = [Text.UTF8Encoding]::new()
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONPATH = "$PSScriptRoot\bridge\python"
if (-not $OutRoot) { $OutRoot = Join-Path (Split-Path $PSScriptRoot -Parent) 'experiments\run-recorder\sessions' }
$OutRoot = [IO.Path]::GetFullPath($OutRoot)
if ($Recover) {
    & python -m isaac_bridge.run_recording --recover ([IO.Path]::GetFullPath($Recover))
    exit $LASTEXITCODE
}
if (-not $SessionName) { throw '需要 -SessionName（或用 -Recover <会话目录> 收尾旧会话）' }
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
if ($AllowConsole) { $Arguments += @('--allow-console') }
& python @Arguments
exit $LASTEXITCODE
