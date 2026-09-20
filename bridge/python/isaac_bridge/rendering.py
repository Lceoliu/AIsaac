"""Training render mode: use the existing J460 hooks without virtual time.

Crash-probe switches (rl/docs/NATIVE_CRASH_ANALYSIS.md) are taken from the environment so the
formal entry points do not change behaviour unless asked:
  ISAAC_TURBO_PROBE_DUMP=1  write a heap-bearing minidump at the first shader-push / archive-access failure
  ISAAC_TURBO_FONT_GUARD=1  skip a text draw when KAGE_ColorTextureShader is unusable (mitigation for crash A)
  ISAAC_TURBO_FILE_RETRY=1  retry a transient archive _access/open failure (mitigation for crash B)
"""
import os

from .turbo import TurboControl, inject


def _env_flag(name):
    return os.environ.get(name, "0") == "1"


def configure_rendering(pid, mode):
    """Call after the bridge handshake; the worker is then at a decision boundary.

    Closing the returned mapping does not change the worker's mode. Switching back
    to visible is explicit, so a completed headless run stays out of the renderer.
    With mode "visible" and no control layer present, the probe flags require an
    injection too; it happens only when one of the ISAAC_TURBO_* switches is set.
    """
    if mode not in ("headless", "visible"):
        raise ValueError(f"Unknown render mode: {mode}")
    probe = {"font_guard": _env_flag("ISAAC_TURBO_FONT_GUARD"), "file_retry": _env_flag("ISAAC_TURBO_FILE_RETRY"),
             "probe_dump": _env_flag("ISAAC_TURBO_PROBE_DUMP")}
    control = TurboControl(pid)
    if not control.is_initialized():
        if mode == "visible" and not any(probe.values()):
            control.close()
            return None
        control.preconfigure(virtual_clock=False, skip_render=(mode == "headless"), render_every=0, **probe)
        inject(pid)
    control.wait_active(30.0)
    control.set_render_every(0)
    control.set_flags(virtual_clock=False, skip_render=(mode == "headless"))
    control.set_probe_flags(**probe)
    return control
