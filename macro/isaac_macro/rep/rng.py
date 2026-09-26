"""J460 run seeds. The xorshift RNG is byte-identical to AB+ (the 81 shift triples at VA 0xB1F4C8 and
the float constant at VA 0xBA9FF0 match), and so are Seed2String / String2Seed (same alphabet, same
xor 0x0FEF7FFD), so both are reused from isaac_macro.rng.

Seeds::SetStartSeed(uint)   RVA 0x5EB880: 14 stage seeds (AB+: 13), then the player init seed
Seeds::GetStageSeed          RVA 0x338280: stage clamped to [0, 13]
Seeds::ForgetStageSeed       RVA 0x5EB980: one xorshift step with triple 39 = (5, 15, 17) (AB+: 35)
"""
from __future__ import annotations

from ..rng import MASK32, RNG, seed_to_string, string_to_seed  # noqa: F401  (re-exported)

STAGE_SEEDS = 14


class Seeds:
    def __init__(self, start_seed: int):
        if not start_seed:
            raise ValueError('start seed 0 means "random" in the engine; pass the actual seed')
        self.start_seed = start_seed & MASK32
        self.rng = RNG(self.start_seed, 0x1B)
        self.stage_seeds = [self.rng.next() for _ in range(STAGE_SEEDS)]
        self.player_init_seed = self.rng.next()

    @classmethod
    def from_string(cls, text: str) -> 'Seeds':
        seed = string_to_seed(text)
        if not seed:
            raise ValueError(f'invalid seed string {text!r}')
        return cls(seed)

    def stage_seed(self, stage: int) -> int:
        return self.stage_seeds[min(max(stage, 0), STAGE_SEEDS - 1)]

    def next_seed(self) -> int:
        return self.rng.next()

    def forget_stage_seed(self, stage: int) -> None:
        stage = min(max(stage, 0), STAGE_SEEDS - 1)
        self.stage_seeds[stage] = RNG(self.stage_seeds[stage], 39).next()
