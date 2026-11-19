"""Vectorized header checking must preserve the frozen scalar entry's public behavior."""
import numpy as np
import pytest

from _v3_check_batch_baseline import check_batch as scalar
from isaac_bridge import schema_v3 as V3
from isaac_bridge import v3_input as VI


def batch(b, seed=29):
    rng = np.random.default_rng(seed)
    counts = rng.integers(0, [V3.CAP_BODY, V3.CAP_ENTITY + 1, 84, 50], size=(b, 4))
    counts[:, 0] += 1
    words = V3.frame_floats(*counts.T)
    h = np.column_stack((np.full(b, V3.MAGIC), words, counts, np.full(b, V3.VERSION_WORD),
                         rng.integers(-10, 10, b))).astype(np.float32)
    if b:
        h[-1, V3.H_PRODUCER] = np.finfo(np.float32).max
    return h, np.full(b, V3.VERSION, np.int32), (words * 4).astype(np.int32), np.ones((b, 2), np.float32)


def check(fn, h, versions, sizes, p0, p_max=V3.CAP_BODY, n_max=64, s_max=32):
    return fn(h, versions, sizes, p_max, n_max, s_max, p0=p0)


def assert_success(data, **kw):
    old, trunc_old = check(scalar, *data, **kw)
    new, trunc_new = check(VI.check_batch, *data, **kw)
    assert new.shape == old.shape and new.dtype == old.dtype == np.int64
    assert new.tobytes() == old.tobytes() and trunc_old == trunc_new


def assert_failure(data, **kw):
    with pytest.raises(Exception) as old:
        check(scalar, *data, **kw)
    with pytest.raises(type(old.value)) as new:
        check(VI.check_batch, *data, **kw)
    assert type(new.value) is type(old.value)
    assert str(new.value) == str(old.value)
    return str(new.value)


@pytest.mark.parametrize('b', [0, 1, 8, 128, 512])
def test_random_valid_and_empty_batches(b):
    h, v, s, p0 = batch(b, seed=b)
    assert_success((h, v, s, p0))
    # Python int(size) truncates a float value, including a non-integral one.
    assert_success((h[::-1], v[::-1].astype(np.float64), s[::-1].astype(np.float64) + 0.75, p0[::-1]),
                   n_max=None, s_max=0)


def bad(data, i, field, value):
    h, v, s, p0 = data
    dst, index = {'schema': (v, i), 'length': (s, i), 'role': (p0, (i, 0)), 'known': (p0, (i, 1))}.get(
        field, (h, (i, field)))
    dst[index] = value


@pytest.mark.parametrize('field,value', [
    ('schema', V3.VERSION - 1), ('length', 4), ('length', V3.FRAME_BYTES + 4), ('length', 10001),
    ('length', np.nan), ('length', np.inf),
    (V3.H_MAGIC, V3.MAGIC + 1), (V3.H_VERSION, V3.VERSION_WORD + 1),
    (V3.H_PRODUCER, 0.5), (V3.H_PRODUCER, np.nan), (V3.H_PRODUCER, np.inf),
    (V3.H_P, 0), (V3.H_P, V3.CAP_BODY + 1), (V3.H_E, -1), (V3.H_E, V3.CAP_ENTITY + 1),
    (V3.H_N, -1), (V3.H_S, -1), (V3.H_N, np.finfo(np.float32).max), (V3.H_SIZE, 1.5),
    (V3.H_SIZE, 10000), ('role', 0.999), ('known', 0), ('known', np.nan),
])
def test_bad_record_messages_match_at_first_middle_and_last(field, value):
    for i in (0, 3, 7):
        h, v, s, p0 = batch(8)
        data = h, v, s.astype(np.float64), p0
        bad(data, i, field, value)
        assert_failure(data)


def test_length_matches_header_and_budget():
    h, v, s, p0 = batch(8)
    s[5] += 4  # valid byte count but not the length in the header
    assert_failure((h, v, s, p0))
    # Valid finite integer counts and exact length formula, impossible byte budget.
    h[5, V3.H_N] = V3.FRAME_BYTES // V3.NF
    h[5, V3.H_SIZE] = V3.frame_floats(*(int(x) for x in h[5, V3.H_P:V3.H_S + 1]))
    assert_failure((h, v, s, p0))


def test_multiple_errors_keep_the_original_precedence():
    data = batch(8)
    bad(data, 1, 'role', 0)
    bad(data, 6, 'known', 0)
    assert 'record 1, 2 such' in assert_failure(data)
    bad(data, 4, V3.H_MAGIC, -1)
    bad(data, 5, 'schema', V3.VERSION - 1)
    # Headers across the entire batch precede the separate P0 pass, just like the scalar entry.
    assert 'magic -1' in assert_failure(data)
    bad(data, 4, 'length', 7)
    assert 'record 4' in assert_failure(data)
    bad(data, 0, 'schema', V3.VERSION + 1)
    assert 'record 0' in assert_failure(data)
    assert 'padded layout holds 0' in assert_failure(batch(8), p_max=0)


def test_scalar_input_conversions_and_bad_shapes():
    h, v, s, p0 = batch(8)
    assert_success((h, v.astype(str), s.astype(object), p0))
    assert_success(([], [], [], None))
    assert_failure((h[:, :2], v, s, p0))
    assert_failure((h, v[:1], s, p0))
    assert_failure((h, v, s[0], p0))
    assert_failure((h, v.astype(np.float64) + 0.25, s, p0))
    assert_failure((h, v, s, p0[:, 0]))


def test_valid_wire_uses_no_scalar_row_checks(monkeypatch):
    calls = []
    check_row = VI._check_batch_row

    def spy(h, v, s, i):
        calls.append(i)
        return check_row(h, v, s, i)

    monkeypatch.setattr(VI, '_check_batch_row', spy)
    for b in (8, 128, 512):
        assert_success(batch(b))
    assert calls == []
    data = batch(8)
    bad(data, 6, V3.H_MAGIC, -1)
    bad(data, 4, 'length', 7)
    assert_failure(data)
    assert calls == [4]
