"""Steam watch and Steam-aware instance recycling (isaac_bridge/steam_watch.py, abplus_worker.Instance).

No engine and no Steam client needed: the process start, the bridge readiness and the Steam check
are replaced by fakes.
"""
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from isaac_bridge import abplus_worker as W
from isaac_bridge.steam_watch import SteamWatch, steam_running


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class SteamWatchTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.state = {'up': True}
        self.alerts = []
        self.clock = Clock()
        self.watch = SteamWatch(self.dir.name, remind_s=600, check=lambda: self.state['up'],
                                alert=self.alerts.append, clock=self.clock)

    def tearDown(self):
        self.dir.cleanup()

    def test_down_reminder_and_recovery(self):
        marker = Path(self.dir.name) / 'STEAM_DOWN'
        self.assertTrue(self.watch.poll())
        self.assertEqual(self.alerts, [])
        self.state['up'] = False
        self.assertFalse(self.watch.poll())
        self.assertEqual(len(self.alerts), 1)
        self.assertTrue(marker.exists())
        self.clock.t += 300
        self.watch.poll()
        self.assertEqual(len(self.alerts), 1)            # no reminder before remind_s
        self.clock.t += 301
        self.watch.poll()
        self.assertEqual(len(self.alerts), 2)
        self.assertIn('10 min', self.alerts[-1])
        self.state['up'] = True
        self.assertTrue(self.watch.poll())
        self.assertEqual(len(self.alerts), 3)
        self.assertIn('back', self.alerts[-1])
        self.assertFalse(marker.exists())
        self.watch.poll()
        self.assertEqual(len(self.alerts), 3)

    def test_start_failures_alert_once_per_increase(self):
        self.watch.launch_failed(0)
        self.watch.launch_failed(2)
        self.watch.launch_failed(2)
        self.watch.launch_failed(3)
        self.assertEqual(len(self.alerts), 2)


@unittest.skipUnless(hasattr(os, 'getuid'), 'POSIX process table')
class SteamRunningTest(unittest.TestCase):
    def test_finds_only_the_steam_client_executable(self):
        with tempfile.TemporaryDirectory() as root:
            proc = Path(root)

            def entry(pid, comm, exe):
                d = proc / str(pid)
                d.mkdir()
                (d / 'comm').write_text(comm + '\n')
                os.symlink(exe, d / 'exe')
            entry(10, 'bash', '/usr/bin/bash')
            entry(11, 'steam', '/usr/games/steam')              # a launcher script, not the client
            self.assertFalse(steam_running(proc))
            entry(12, 'steam', '/home/u/.local/share/Steam/ubuntu12_32/steam')
            self.assertTrue(steam_running(proc))


class FakeProc:
    def __init__(self):
        self.pid = 4242

    def poll(self):
        return None

    def wait(self, timeout=None):
        return 0


class FakeEnv:
    def __init__(self, port, **kwargs):
        self.port = port
        self.closed = False
        self.bridge = mock.Mock()

    def close(self):
        self.closed = True


class RecycleTest(unittest.TestCase):
    def instance(self, launched):
        config = dict(mode='exact', nice=19, binary_obs=True, tasks=None, recycle_episodes=200)

        def launch(name, port, mode, nice):
            launched.append((name, port))
            return FakeProc()
        patches = [mock.patch.object(W, 'launch_abplus', launch), mock.patch.object(W, 'FrameEnv', FakeEnv),
                   mock.patch.object(W, 'stop_abplus', lambda proc, name: launched.append(('stop', name)))]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        return W.Instance('tr0a', 27400, config, 27432)

    def test_steam_down_defers_and_keeps_the_process(self):
        launched = []
        inst = self.instance(launched)
        old_env = inst.env
        with mock.patch.object(W, 'steam_running', lambda: False):
            inst._recycle()
        self.assertEqual((inst.recycle_deferrals, inst.recycles, inst.start_failures), (1, 0, 0))
        self.assertIs(inst.env, old_env)
        self.assertEqual(launched, [('tr0a', 27400)])
        self.assertEqual(inst.episodes, 200 - W.RETRY_EPISODES)

    def test_failed_replacement_keeps_the_old_process(self):
        launched = []
        inst = self.instance(launched)
        old_env = inst.env
        with mock.patch.object(W, 'steam_running', lambda: True), \
                mock.patch.object(W, 'wait_listening', lambda proc, port, timeout: False):
            inst._recycle()
        self.assertEqual((inst.recycles, inst.start_failures), (0, 1))
        self.assertIs(inst.env, old_env)
        self.assertFalse(old_env.closed)
        self.assertEqual((inst.name, inst.port), ('tr0a', 27400))
        self.assertEqual(launched, [('tr0a', 27400), ('tr0ax', 27432), ('stop', 'tr0ax')])

    def test_replacement_starts_before_the_old_process_stops(self):
        launched = []
        inst = self.instance(launched)
        old_env = inst.env
        with mock.patch.object(W, 'steam_running', lambda: True), \
                mock.patch.object(W, 'wait_listening', lambda proc, port, timeout: True):
            inst._recycle()
            self.assertEqual((inst.name, inst.port, inst.recycles), ('tr0ax', 27432, 1))
            self.assertTrue(old_env.closed)
            self.assertEqual(launched, [('tr0a', 27400), ('tr0ax', 27432), ('stop', 'tr0a')])
            inst._recycle()                                # and back to the first identity
            self.assertEqual((inst.name, inst.port, inst.recycles), ('tr0a', 27400, 2))


if __name__ == '__main__':
    unittest.main()
