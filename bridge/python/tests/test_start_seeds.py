"""The workers' training start seeds (tok_sampler.start_seed, used by tok_sampler and tok_floor workers): a fresh
single run's sequence is the old one exactly; a resumed run continues past every start the earlier one may have used
(train_tok: state_base = updates done x STATES_PER_UPDATE); K shards interleave without overlap, resumed or not."""
from dataclasses import replace

from isaac_bridge import tok_floor, tok_sampler
from isaac_bridge.tok_sampler import TokSamplerConfig, start_seed


def _seeds(cfg, n):
    return {start_seed(cfg, i, k) for i in range(cfg.workers) for k in range(n)}


def test_fresh_single_run_is_the_old_sequence():
    cfg = TokSamplerConfig(workers=16, seed=1000)
    for i in (0, 5, 15):
        for k in (0, 1, 7, 1000):
            assert start_seed(cfg, i, k) == 1000 + i + 16 * k


def test_resume_continues_past_the_used_starts():
    cfg = TokSamplerConfig(workers=16, seed=1000)
    per_update = 64
    done = 30   # updates before the stop; the earlier run used at most done * per_update starts per worker
    before = _seeds(cfg, done * per_update)
    after = _seeds(replace(cfg, state_base=done * per_update), 500)
    assert not before & after
    # a second stop and resume later on
    done2 = 45
    after2 = _seeds(replace(cfg, state_base=done2 * per_update), 500)
    assert not (before | _seeds(replace(cfg, state_base=done * per_update), (done2 - done) * per_update)) & after2


def test_shards_interleave_without_overlap():
    k = 2
    a = TokSamplerConfig(workers=16, seed=1000, seed_stride=k, seed_phase=0)
    b = replace(a, seed_phase=1)
    assert not _seeds(a, 3000) & _seeds(b, 3000)
    # resumed (the same state_base on both ranks, from the same checkpoint's update count)
    a2, b2 = replace(a, state_base=640), replace(b, state_base=640)
    assert not (_seeds(a, 640) | _seeds(b, 640)) & (_seeds(a2, 500) | _seeds(b2, 500))
    assert not _seeds(a2, 500) & _seeds(b2, 500)


def test_both_worker_kinds_use_it():
    import inspect
    assert 'start_seed(cfg, index, state_n)' in inspect.getsource(tok_sampler)
    assert 'start_seed(cfg, index, state_n)' in inspect.getsource(tok_floor)


def test_worker_streams_per_shard_and_resume():
    """The workers' random streams (tok_sampler.rng_key: resets, start builds, forced pickups, characters, teacher /
    CF draws): a fresh single run's exactly as before; each shard's its own; a resumed run's not the earlier run's."""
    import numpy as np
    from isaac_bridge.tok_sampler import rng_key
    a = TokSamplerConfig(workers=16, seed=1000)
    assert rng_key(a, 3) == [1000, 3] and rng_key(a, 3, 29) == [1000, 3, 29]   # (the old [cfg.seed, index, k])
    two = replace(a, seed_stride=2)
    r0, r1 = replace(two, seed_phase=0), replace(two, seed_phase=1)
    assert rng_key(r0, 3, 29) == [1000, 3, 29]                                # (rank 0 fresh: unchanged)
    draw = lambda c: np.random.default_rng(rng_key(c, 3, 29)).random(8)
    assert not np.allclose(draw(r0), draw(r1))
    r0b, r1b = replace(r0, state_base=640), replace(r1, state_base=640)       # (resumed after 10 updates)
    assert not np.allclose(draw(r0), draw(r0b)) and not np.allclose(draw(r1), draw(r1b))
    assert not np.allclose(draw(r0b), draw(r1b))


def test_floor_worker_streams_use_it():
    import inspect
    src = inspect.getsource(tok_floor)
    assert 'default_rng([cfg.seed' not in src and src.count('rng_key(cfg, index') >= 11
