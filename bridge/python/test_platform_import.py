"""Headless Linux imports must not load the Windows injection backend."""
import ctypes,importlib.util,sys,unittest
from pathlib import Path
from unittest.mock import patch


class PlatformImportTest(unittest.TestCase):
    def test_linux_package_does_not_load_windows_turbo(self):
        path=Path(__file__).parent/'isaac_bridge/__init__.py'
        name='isaac_bridge_linux_probe'
        spec=importlib.util.spec_from_file_location(name,path,submodule_search_locations=[str(path.parent)])
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module
        try:
            with patch.object(sys,'platform','linux'),patch.object(ctypes,'WinDLL',create=True,side_effect=AssertionError('Windows DLL loaded on Linux')):
                spec.loader.exec_module(module)
            self.assertTrue(hasattr(module,'IsaacBridgeEnv'))
            self.assertNotIn('TurboControl',module.__all__)
            self.assertNotIn(name+'.turbo',sys.modules)
        finally:
            for key in list(sys.modules):
                if key==name or key.startswith(name+'.'):del sys.modules[key]


if __name__=='__main__':unittest.main()
