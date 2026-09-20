import os
import unittest
from unittest.mock import Mock, patch

from isaac_bridge.turbo import launch_suspended


class WorkerLaunchTest(unittest.TestCase):
    def test_capture_isolation_defaults_on_and_can_be_disabled_per_worker(self):
        for kwargs, expected, archive in (({}, '1', '1'),
                                          ({'disable_capture_overlays': False}, '0', '1'),
                                          ({'archive_path_fix': False}, '1', '0')):
            with self.subTest(expected=expected), \
                 patch.dict(os.environ, {'ISAAC_TURBO_DISABLE_CAPTURE_OVERLAYS': 'parent',
                                         'ISAAC_TURBO_ARCHIVE_PATH_FIX': 'parent'}), \
                 patch('isaac_bridge.turbo.os.path.isfile', return_value=True), \
                 patch('isaac_bridge.turbo.stage_dll', return_value='staged.dll'), \
                 patch('isaac_bridge.turbo.TurboControl') as control, \
                 patch('isaac_bridge.turbo.subprocess.run') as run:
                run.return_value = Mock(returncode=0, stdout='PASS launched pid=123 module=0x400000 status=1', stderr='')
                pid, ctl, _ = launch_suspended(27015, **kwargs)
                self.assertEqual(pid, 123)
                self.assertIs(ctl, control.return_value)
                self.assertEqual(run.call_args.kwargs['env']['ISAAC_TURBO_DISABLE_CAPTURE_OVERLAYS'], expected)
                self.assertEqual(run.call_args.kwargs['env']['ISAAC_TURBO_ARCHIVE_PATH_FIX'], archive)
                self.assertEqual(os.environ['ISAAC_TURBO_DISABLE_CAPTURE_OVERLAYS'], 'parent')
                self.assertEqual(os.environ['ISAAC_TURBO_ARCHIVE_PATH_FIX'], 'parent')
                self.assertIn('--launch', run.call_args.args[0])


if __name__ == '__main__':
    unittest.main()
