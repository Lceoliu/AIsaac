#!/bin/bash
# Start one isolated AB+ v1.06 Linux instance from ~/isaac-abplus/runtime (a copy of the
# Steam install; the Steam directory itself is never touched).
# usage: run_instance.sh ID [game args...]     e.g. run_instance.sh i0 --set-stage=1 --set-stage-type=0
# Per instance: its own XDG_DATA_HOME (saves, options.ini, log.txt, mods dir), options with
# SteamCloud=0 / EnableMods=0 / VSync=0 / muted / 480x270 window. Extra environment
# (ABP_PRELOAD -> LD_PRELOAD of the game only, ABP_*, ALSOFT_DRIVERS, ...) comes from the caller.
set -e
# The game lower-cases its save path (XDG_DATA_HOME), so instance ids are lower case.
ID=${1,,}; shift
BASE=${ABP_HOME:?ABP_HOME must point to the isolated project directory}
INST=$BASE/instances/$ID
SAVE="$INST/data/binding of isaac afterbirth+"
# A plain file occupies the mods directory path on purpose: the Workshop sync
# (system("/bin/rm|cp ...")) then fails instead of copying the account's subscribed
# mods into every instance. Mods are not needed; control code comes via ABP_LUA.
mkdir -p "$SAVE"
[ -e "$INST/data/binding of isaac afterbirth+ mods" ] || : > "$INST/data/binding of isaac afterbirth+ mods"
if [ ! -f "$SAVE/options.ini" ]; then
  N=${ID//[!0-9]/}; N=${N:-0}
  cat > "$SAVE/options.ini" <<EOF
[Options]
Language=
MusicVolume=0.0000
SFXVolume=0.0000
MapOpacity=0.3000
Fullscreen=0
Filter=0
Exposure=1.0000
Gamma=1.2000
ControllerHotplug=0
PopUps=0
CameraStyle=1
ShowRecentItems=0
HudOffset=0.0000
TryImportSave=0
FoundHUD=0
EnableMods=0
RumbleEnabled=0
ChargeBars=0
MaxScale=1
MaxRenderScale=1
VSync=0
PauseOnFocusLost=0
SteamCloud=0
MouseControl=0
BossHpOnBottom=1
AnnouncerVoiceMode=0
ConsoleFont=0
FadedConsoleDisplay=0
SaveCommandHistory=0
WindowWidth=480
WindowHeight=270
WindowPosX=$(( (N % 8) * 60 ))
WindowPosY=$(( 40 + (N / 8) * 40 ))
EOF
fi
cd "$BASE/runtime"
export DISPLAY=${DISPLAY:-:0} XAUTHORITY=${XAUTHORITY:-/run/user/1000/gdm/Xauthority}
export SteamAppId=250900 LD_LIBRARY_PATH=lib64 XDG_DATA_HOME="$(realpath --relative-to="$BASE/runtime" "$INST/data")"
# ABP_PRELOAD is applied to the game process only, never to this shell or its helpers.
if [ -n "$ABP_PRELOAD" ]; then export LD_PRELOAD="$ABP_PRELOAD"; fi
exec ./isaac.x64 "$@" > "$INST/stdout.log" 2>&1
