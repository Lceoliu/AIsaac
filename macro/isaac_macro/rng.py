"""The game's RNG and run seeds, translated from Afterbirth+ v1.06 (Linux x64, named symbols).

RNG::RNG / SetSeed / Next / Random            0x42A180-0x42A2A0
RNG::s_Shifts (81 xorshift triples, .rodata)   0x9A3E20
Seeds::SetStartSeed / ForgetStageSeed          0x463630 / 0x4636E0
Seeds::Seed2String / String2Seed               0x4633C0 / 0x4634E0

Floats follow the binary: RNG::Random() converts the uint32 state to float32 and multiplies by the
float32 constant at 0x9A41EC (2.3283061589829401e-10), all in single precision.
"""
from __future__ import annotations


MASK32 = 0xFFFFFFFF
from .f32 import F32  # noqa: E402
RANDOM_FLOAT = F32(2.3283061589829401e-10)

SHIFTS = [
    (1, 3, 10), (1, 5, 16), (1, 5, 19), (1, 9, 29), (1, 11, 6), (1, 11, 16), (1, 19, 3),
    (1, 21, 20), (1, 27, 27), (2, 5, 15), (2, 5, 21), (2, 7, 7), (2, 7, 9), (2, 7, 25), (2, 9, 15),
    (2, 15, 17), (2, 15, 25), (2, 21, 9), (3, 1, 14), (3, 3, 26), (3, 3, 28), (3, 3, 29),
    (3, 5, 20), (3, 5, 22), (3, 5, 25), (3, 7, 29), (3, 13, 7), (3, 23, 25), (3, 25, 24),
    (3, 27, 11), (4, 3, 17), (4, 3, 27), (4, 5, 15), (5, 3, 21), (5, 7, 22), (5, 9, 7), (5, 9, 28),
    (5, 9, 31), (5, 13, 6), (5, 15, 17), (5, 17, 13), (5, 21, 12), (5, 27, 8), (5, 27, 21),
    (5, 27, 25), (5, 27, 28), (6, 1, 11), (6, 3, 17), (6, 17, 9), (6, 21, 7), (6, 21, 13),
    (7, 1, 9), (7, 1, 18), (7, 1, 25), (7, 13, 25), (7, 17, 21), (7, 25, 12), (7, 25, 20),
    (8, 7, 23), (8, 9, 23), (9, 5, 14), (9, 5, 25), (9, 11, 19), (9, 21, 16), (10, 9, 21),
    (10, 9, 25), (11, 7, 12), (11, 7, 16), (11, 17, 13), (11, 21, 13), (12, 9, 23), (13, 3, 17),
    (13, 3, 27), (13, 5, 19), (13, 17, 15), (14, 1, 15), (14, 13, 15), (15, 1, 29), (17, 15, 20),
    (17, 15, 23), (17, 15, 26),
]


class RNG:
    __slots__ = ('seed', 'a', 'b', 'c')

    def __init__(self, seed: int = 0xAA17414F, shift_index: int | None = None):
        if shift_index is None:  # RNG::RNG(): default seed with shifts (5, 9, 7)
            self.seed, (self.a, self.b, self.c) = seed & MASK32, (5, 9, 7)
        else:
            self.set_seed(seed, shift_index)

    def set_seed(self, seed: int, shift_index: int) -> None:
        self.seed = seed & MASK32
        self.a, self.b, self.c = SHIFTS[shift_index]

    def copy(self) -> 'RNG':
        r = RNG.__new__(RNG)
        r.seed, r.a, r.b, r.c = self.seed, self.a, self.b, self.c
        return r

    def next(self) -> int:
        x = self.seed
        x ^= x >> self.a
        x ^= (x << self.b) & MASK32
        x ^= x >> self.c
        self.seed = x
        return x

    def random_int(self, n: int) -> int:
        """RNG::Random(unsigned int): advances first; 0 when n == 0."""
        self.next()
        return self.seed % n if n else 0

    def random_float(self) -> float:
        """RNG::Random(): float32(state) * float32 constant, in single precision."""
        self.next()
        return F32(F32(self.seed) * RANDOM_FLOAT)


class Seeds:
    """Seeds::SetStartSeed: 13 stage seeds and the player init seed from an RNG with shift triple 27.

    The RNG keeps running after that (Seeds+8, copied with the object): Game::Start draws the
    Fortunes seed and then the ItemPool::Init seed from it; Game::StartDebug only the ItemPool seed.
    """

    def __init__(self, start_seed: int):
        if not start_seed:
            raise ValueError('start seed 0 means "random" in the engine; pass the actual seed')
        self.start_seed = start_seed & MASK32
        self.rng = RNG(self.start_seed, 0x1B)
        self.stage_seeds = [self.rng.next() for _ in range(13)]
        self.player_init_seed = self.rng.next()

    @classmethod
    def from_string(cls, text: str) -> 'Seeds':
        seed = string_to_seed(text)
        if not seed:
            raise ValueError(f'invalid seed string {text!r}')
        return cls(seed)

    def stage_seed(self, stage: int) -> int:
        return self.stage_seeds[stage]

    def next_seed(self) -> int:
        """Seeds::GetNextSeed."""
        return self.rng.next()

    def forget_stage_seed(self, stage: int) -> None:
        self.stage_seeds[stage] = RNG(self.stage_seeds[stage], 0x23).next()


SEED_ALPHABET = 'ABCDEFGHJKLMNPQRSTWXYZ01234V6789'
SEED_XOR = 0x0FEF7FFD


def _seed_checksum(value: int) -> int:
    c = 0
    while value:
        s = (c + value) & 0xFF
        c = ((s >> 7) + s * 2) & 0xFF
        value >>= 5
    return c


def seed_to_string(seed: int) -> str:
    """Seeds::Seed2String: the 'XXXX XXXX' form shown on the pause screen and typed at game start."""
    c = _seed_checksum(seed & MASK32)
    x = (seed ^ SEED_XOR) & MASK32
    d = [x >> 27, (x >> 22) & 31, (x >> 17) & 31, (x >> 12) & 31, (x >> 7) & 31, (x >> 2) & 31,
         ((x << 3) | (c >> 5)) & 31, c & 31]
    chars = [SEED_ALPHABET[v] for v in d]
    return ''.join(chars[:4]) + ' ' + ''.join(chars[4:])


def string_to_seed(text: str) -> int:
    """Seeds::String2Seed: 0 when the string is malformed or its checksum does not match.
    Like the engine it is case-sensitive (the alphabet is upper case)."""
    if len(text) != 9 or text[4] != ' ':
        return 0
    d = []
    for ch in text[:4] + text[5:]:
        v = SEED_ALPHABET.find(ch)
        if v < 0:
            return 0
        d.append(v)
    value = ((d[0] << 27) | (d[1] << 22) | (d[2] << 17) | (d[3] << 12) | (d[4] << 7) | (d[5] << 2)
             | (d[6] >> 3)) ^ SEED_XOR
    if ((d[6] << 5) | d[7]) & 0xFF != _seed_checksum(value):
        return 0
    return value
