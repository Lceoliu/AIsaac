import os
import unittest
from unittest.mock import Mock, patch
from isaac_bridge.rendering import configure_rendering


class RenderingTest(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {name: '0' for name in (
            'ISAAC_TURBO_FONT_GUARD', 'ISAAC_TURBO_FILE_RETRY', 'ISAAC_TURBO_PROBE_DUMP')})
        env.start()
        self.addCleanup(env.stop)

    def test_headless_injects_with_clock_disabled(self):
        control = Mock()
        control.is_initialized.return_value = False
        with patch('isaac_bridge.rendering.TurboControl', return_value=control), patch('isaac_bridge.rendering.inject') as inject:
            self.assertIs(configure_rendering(123, 'headless'), control)
        inject.assert_called_once_with(123)
        control.preconfigure.assert_called_once_with(virtual_clock=False, skip_render=True, render_every=0,
                                                    font_guard=False, file_retry=False, probe_dump=False)
        control.set_flags.assert_called_once_with(virtual_clock=False, skip_render=True)
        control.set_probe_flags.assert_called_once_with(font_guard=False, file_retry=False, probe_dump=False)

    def test_visible_explicit_probe_injects_without_virtual_clock(self):
        control = Mock()
        control.is_initialized.return_value = False
        with patch.dict(os.environ, {'ISAAC_TURBO_PROBE_DUMP': '1'}), \
             patch('isaac_bridge.rendering.TurboControl', return_value=control), \
             patch('isaac_bridge.rendering.inject') as inject:
            self.assertIs(configure_rendering(123, 'visible'), control)
        inject.assert_called_once_with(123)
        control.preconfigure.assert_called_once_with(virtual_clock=False, skip_render=False, render_every=0,
                                                    font_guard=False, file_retry=False, probe_dump=True)
        control.set_probe_flags.assert_called_once_with(font_guard=False, file_retry=False, probe_dump=True)

    def test_existing_worker_applies_explicit_mitigation_flags(self):
        control = Mock()
        control.is_initialized.return_value = True
        with patch.dict(os.environ, {'ISAAC_TURBO_FONT_GUARD': '1', 'ISAAC_TURBO_FILE_RETRY': '1'}), \
             patch('isaac_bridge.rendering.TurboControl', return_value=control), \
             patch('isaac_bridge.rendering.inject') as inject:
            configure_rendering(123, 'headless')
        inject.assert_not_called()
        control.preconfigure.assert_not_called()
        control.set_probe_flags.assert_called_once_with(font_guard=True, file_retry=True, probe_dump=False)

    def test_visible_baseline_does_not_inject(self):
        control = Mock()
        control.is_initialized.return_value = False
        with patch('isaac_bridge.rendering.TurboControl', return_value=control), patch('isaac_bridge.rendering.inject') as inject:
            self.assertIsNone(configure_rendering(123, 'visible'))
        inject.assert_not_called()
        control.close.assert_called_once()

    def test_visible_restores_existing_control_without_clock(self):
        control = Mock()
        control.is_initialized.return_value = True
        with patch('isaac_bridge.rendering.TurboControl', return_value=control), patch('isaac_bridge.rendering.inject') as inject:
            self.assertIs(configure_rendering(123, 'visible'), control)
        inject.assert_not_called()
        control.set_flags.assert_called_once_with(virtual_clock=False, skip_render=False)
        control.set_render_every.assert_called_once_with(0)


if __name__ == '__main__': unittest.main()
