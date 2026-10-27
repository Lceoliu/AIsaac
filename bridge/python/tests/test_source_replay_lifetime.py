"""Run the existing production retention/late-record tests with value-only sources."""
import os
from pathlib import Path
import runpy
import sys

if __name__ == '__main__':
    from isaac_bridge.source_pipeline import SourcePipeline
    original = SourcePipeline.__init__

    def replay_init(self, *args, **kwargs):
        kwargs['replay'] = True
        return original(self, *args, **kwargs)

    SourcePipeline.__init__ = replay_init
    if '--diagnose' in sys.argv:
        import torch
        sys.argv.remove('--diagnose')
        equal = torch.equal
        def measured_equal(a, b):
            result = equal(a, b)
            if not result:
                print('STRICT_EQUAL_DIFFERENCE', tuple(a.shape), float((a-b).abs().max()), flush=True)
            return result
        torch.equal = measured_equal
    if '--numerical' in sys.argv:
        # Explicit supplementary gate; the original bitwise gate is unchanged.
        import torch
        sys.argv.remove('--numerical')
        print('NUMERICAL_ONLY: rtol=1e-6 atol=1e-7; not a bitwise pass', flush=True)
        torch.equal = lambda a, b: torch.allclose(a, b, rtol=1e-6, atol=1e-7)
    test = Path(__file__).with_name(sys.argv.pop(1) + '.py')
    runpy.run_path(str(test), run_name='__main__')
