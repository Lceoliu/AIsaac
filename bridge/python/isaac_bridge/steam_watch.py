"""Steam client watch for AB+ training.

Every AB+ process start (worker instance recycling, evaluation, error relaunch) needs a running,
logged-in Steam client: the executable's DRM stub otherwise runs `steam.sh steam://run/250900` and
exits. On 2026-09-25 the client had quit; an evaluation got 0 episodes and the next instance
recycles would have failed (rl/bridge/abplus/README.md, "Steam 依赖").

steam_running() looks for the client process (name `steam`, executable .../ubuntu12_32/steam) of
this user. ~/.steam/registry.vdf is not used: its SteamPID did not match the live client. A client
that runs but is signed out (e.g. signed in on another machine) is not visible to this check; it
shows up as instance starts that fail, which the worker counts and the learner alerts on.

SteamWatch turns the state into alerts: a JSON event in the training log, a STEAM_DOWN marker file
in the run directory, a desktop notification on the host display (notify-send), and optionally a
command from $ABP_ALERT_CMD (run by the shell with the message in $ABP_ALERT_MESSAGE, e.g. a curl
to a phone push service). It repeats every remind_s seconds while Steam stays down and reports the
recovery.
"""
import json
import os
import subprocess
import time
from pathlib import Path

CLIENT_SUFFIX = '/ubuntu12_32/steam'


def steam_running(proc=Path('/proc')):
    """True when a Steam client process of this user is alive."""
    uid = os.getuid()
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if (entry / 'comm').read_text().strip() != 'steam' or entry.stat().st_uid != uid:
                continue
            if os.readlink(entry / 'exe').endswith(CLIENT_SUFFIX):
                return True
        except OSError:
            continue
    return False


def notify(message, title='AB+ training'):
    """Desktop notification on the host display, then $ABP_ALERT_CMD if set. Never raises."""
    env = dict(os.environ)
    env.setdefault('DISPLAY', ':0')
    env.setdefault('DBUS_SESSION_BUS_ADDRESS', f'unix:path=/run/user/{os.getuid()}/bus')
    try:
        subprocess.run(['notify-send', '-u', 'critical', title, message], env=env, timeout=10,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    except (OSError, subprocess.SubprocessError):
        pass
    command = os.environ.get('ABP_ALERT_CMD')
    if command:
        try:
            subprocess.run(command, shell=True, env={**env, 'ABP_ALERT_MESSAGE': message}, timeout=30,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        except (OSError, subprocess.SubprocessError):
            pass


class SteamWatch:
    """Polled once per rollout by the learner; poll() returns whether Steam is up."""

    def __init__(self, out_dir, remind_s=600, check=steam_running, alert=notify, clock=time.time):
        self.marker = Path(out_dir) / 'STEAM_DOWN'
        self.remind_s, self.check, self.alert, self.clock = remind_s, check, alert, clock
        self.ok = True
        self.down_since = self.last_alert = None
        self.launch_failures = 0

    def _event(self, name, message, **extra):
        print(json.dumps(dict(event=name, message=message, time=time.strftime('%Y-%m-%d %H:%M:%S'), **extra)),
              flush=True)
        self.alert(message)
        self.last_alert = self.clock()

    def poll(self):
        ok = bool(self.check())
        now = self.clock()
        if not ok and self.ok:
            self.down_since = now
            self.marker.write_text(json.dumps({'since': time.strftime('%Y-%m-%d %H:%M:%S')}), encoding='utf8')
            self._event('steam_down', 'Steam client is not running: AB+ instances cannot start. '
                        'Log in to Steam on the training host.')
        elif not ok and now - self.last_alert >= self.remind_s:
            self._event('steam_down', f'Steam still not running ({(now - self.down_since) / 60:.0f} min). '
                        'Instance recycles are deferred and evaluations skipped.')
        elif ok and not self.ok:
            self.marker.unlink(missing_ok=True)
            self._event('steam_up', f'Steam client is back after {(now - self.down_since) / 60:.0f} min.')
            self.down_since = None
        self.ok = ok
        return ok

    def launch_failed(self, total):
        """Workers report their cumulative count of instance starts that failed."""
        if total > self.launch_failures:
            self._event('instance_start_failed',
                        f'{total - self.launch_failures} AB+ instance start(s) failed (total {total}); the old '
                        'processes keep running. Is Steam logged in on the training host?', total=total)
            self.launch_failures = total
