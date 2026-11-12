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
