"""One decoded-PNG cache shared by several roots (abp_turbo ABP_PNG_CACHE_FILE, 2026-10-06): are its hits the same
texels as a fresh decode?

--instances roots (threads, each its own instance) share one cache file and run in abp_turbo's PNG check mode
(ABP_FAST=11: bit 8 decodes every image the cache has and compares texels and image fields with the cached entry,
counting png_check_mismatches; the game always uses the fresh decode). Each root resets floors, a lean clone of each
walks --legs legs through doors (abplus_probe_trim.leg), and before it closes its counters are read
(ABP_FAST_STATUS). Entries stored by one root are checked by the others' clones. Also reports the cache's size
(pc_used) and the file's resident size.
usage (PYTHONPATH=.): python abplus_probe_png_shared.py --instances 8 --seeds-per 6 --out ../runs/png-shared
"""
import argparse
import json
import os
import re
import threading
import time
from pathlib import Path

import numpy as np

from abplus_probe_trim import Walker, leg
from goexplore_abplus import default_bridge_lua, default_preload, load_spec
from isaac_bridge.abplus import FORK_ENV
from isaac_bridge.abplus_goexplore import GxConfig, Instance
from isaac_bridge.tok_sampler import default_stub_list

STATUS = "return tostring(os.getenv('ABP_FAST_STATUS'))"
KEYS = ('png_loads', 'png_hits', 'png_misses', 'png_stored', 'png_checks', 'png_check_mismatches', 'png_uncached')


def counters(text):
    return {k: int(v) for k, v in re.findall(r'(\w+)=(\d+)', text or '') if k in KEYS + ('pc_used', 'pc_file', 'pc_cap')}


def run(i, args, cfg, spec, out, lock, totals):
    inst = Instance(f'{args.name}{i}', args.port + i, cfg, spec)
    inst.env.bridge.reset_mode = 'floor'
    try:
        for k in range(args.seeds_per):
            seed = args.seed0 + i * 1000 + k
            inst.reset(seed)
            ep = inst.env.bridge.fork(tag='ep', lean=True, alarm=1800)
            w = Walker([ep], 25)
            rng = np.random.default_rng(seed)
            legs = 0
            for _ in range(args.legs):
                if not leg(w, rng, args):
                    break
                legs += 1
            c = counters(ep.lua(STATUS))
            ep.close()
            with lock:
                for key in KEYS:
                    totals[key] = totals.get(key, 0) + c.get(key, 0)
                totals['episodes'] = totals.get('episodes', 0) + 1
                totals['legs'] = totals.get('legs', 0) + legs
                totals['decisions'] = totals.get('decisions', 0) + w.decisions
                totals['pc_used'] = max(totals.get('pc_used', 0), c.get('pc_used', 0))
                totals['pc_file'] = min(totals.get('pc_file', 1), c.get('pc_file', 0))
                out.write(json.dumps(dict(instance=i, seed=seed, legs=legs, decisions=w.decisions, **c)) + '\n')
                out.flush()
        c = counters(inst.env.bridge.lua(STATUS))   # the root's own loads (the floor starts)
        with lock:
            for key in KEYS:
                totals['root_' + key] = totals.get('root_' + key, 0) + c.get(key, 0)
    finally:
        inst.kill()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--instances', type=int, default=8)
    p.add_argument('--seeds-per', type=int, default=6)
    p.add_argument('--seed0', type=int, default=2147610000)
    p.add_argument('--legs', type=int, default=5)
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--after-clear', type=int, default=15)
    p.add_argument('--fpd', type=int, default=4)
    p.add_argument('--fast', default='11', help='ABP_FAST of the instances (11: 3 with the PNG check mode)')
    p.add_argument('--port', type=int, default=40300)
    p.add_argument('--name', default='mempng')
    p.add_argument('--groups-file', default='../abplus/catalog/scaling2_groups.json')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=False)
    os.environ.update(FORK_ENV)
    os.environ['ABP_FAST'] = args.fast
    os.environ['ISAAC_RL_PU_SKIP'] = '1'
    os.environ['ABP_FORK_LITE'] = '1'
    cache = f'/dev/shm/abp-png-v1-{args.name}-probe'
    os.environ['ABP_PNG_CACHE_FILE'] = cache
    preload = default_preload()
    cfg = GxConfig(bridge_lua=default_bridge_lua(), preload=preload, al_stopped=True, nice=0,
                   stub_list=default_stub_list(preload))
    spec = load_spec(argparse.Namespace(groups_file=args.groups_file, group='normal', tasks='', seconds=0.0))
    out = open(out_dir / 'episodes.jsonl', 'w')
    lock, totals = threading.Lock(), {}
    t0 = time.perf_counter()
    threads = [threading.Thread(target=run, args=(i, args, cfg, spec, out, lock, totals)) for i in range(args.instances)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    try:
        st = os.stat(cache)
        totals['file_mib'] = round(st.st_size / 2 ** 20, 1)
        totals['file_resident_mib'] = round(st.st_blocks * 512 / 2 ** 20, 1)
        os.unlink(cache)
    except OSError:
        pass
    totals['seconds'] = round(time.perf_counter() - t0, 1)
    (out_dir / 'summary.json').write_text(json.dumps(totals, indent=1))
    print('SUMMARY', json.dumps(totals), flush=True)


if __name__ == '__main__':
    main()
