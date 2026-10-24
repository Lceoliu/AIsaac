import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaac_macro.roomconfig import default_archive_path  # noqa: E402


@pytest.fixture(scope='session')
def rc():
    """RoomConfig over the AB+ afterbirthp.a; tests that need it skip when the archive is absent."""
    path = default_archive_path()
    if not os.path.exists(path):
        pytest.skip(f'AB+ archive not found: {path} (set ISAAC_ABPLUS_AFTERBIRTHP)')
    from isaac_macro.roomconfig import default_room_config
    return default_room_config()


@pytest.fixture(scope='session')
def rep_rc():
    """RepRoomConfig over the Repentance+ afterbirthp.a (read only); skipped when it is absent."""
    from isaac_macro.rep.roomconfig import default_archive_path as rep_archive_path
    path = rep_archive_path()
    if not os.path.exists(path):
        pytest.skip(f'Rep+ archive not found: {path} (set ISAAC_REPPLUS_AFTERBIRTHP)')
    from isaac_macro.rep.roomconfig import default_room_config as rep_room_config
    return rep_room_config()
