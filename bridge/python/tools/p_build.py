"""(P package) Build the C++ modules (tok_native: abp_tok_core / abp_tok_py; tok_driver: abp_tok_driver, without a GPU)
and print the first compiler errors: a compile check that runs in a CPU container.

   usage: python tools/p_build.py      exit 0: both built"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from isaac_bridge import tok_driver, tok_native   # noqa: E402

if __name__ == '__main__':
    bad = 0
    for name, mod, kw in (('tok_native', tok_native, {}), ('tok_driver', tok_driver, dict(need_cuda=False))):
        path = mod.build(**kw)
        print(name, path if path else 'FAILED')
        if not path:
            bad += 1
            print(mod.REASON[:6000])
    sys.exit(1 if bad else 0)
