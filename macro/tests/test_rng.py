import random

from isaac_macro.rng import RNG, SHIFTS, Seeds, seed_to_string, string_to_seed


def test_shift_table():
    assert len(SHIFTS) == 81
    assert SHIFTS[0x23] == (5, 9, 7)          # the engine's usual triple, also RNG::RNG()'s default
    assert SHIFTS[60] == (9, 5, 14)


def xorshift(x, a, b, c):
    x ^= x >> a
    x ^= (x << b) & 0xFFFFFFFF
    return x ^ (x >> c)


def test_rng_basics():
    assert RNG(0xDEADBEEF, 0x23).next() == xorshift(0xDEADBEEF, 5, 9, 7)
    assert RNG(0xDEADBEEF, 0x1B).next() == xorshift(0xDEADBEEF, *SHIFTS[0x1B])
    r = RNG(123456789, 0x23)
    c = r.copy()
    assert [r.random_int(10) for _ in range(20)] == [c.random_int(10) for _ in range(20)]
    assert RNG(5, 0x23).random_int(0) == 0


def test_seeds_layout():
    s = Seeds(0x12345678)
    assert len(s.stage_seeds) == 13
    r = RNG(0x12345678, 0x1B)
    assert s.stage_seeds == [r.next() for _ in range(13)] and s.player_init_seed == r.next()
    assert s.next_seed() == r.next()


def test_seed_strings_round_trip():
    rnd = random.Random(0)
    for _ in range(2000):
        seed = rnd.getrandbits(32) or 1
        text = seed_to_string(seed)
        assert len(text) == 9 and text[4] == ' '
        assert string_to_seed(text) == seed


def test_seed_strings_reject_bad_input():
    text = seed_to_string(2334308359)
    assert string_to_seed(text.lower()) == 0            # the alphabet is upper case, as in the engine
    assert string_to_seed(text.replace(' ', '')) == 0
    other = 'A' if text[-1] != 'A' else 'B'
    assert string_to_seed(text[:-1] + other) == 0       # checksum byte
