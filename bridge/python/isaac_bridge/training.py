"""Production entry point for synchronous, original-engine training sessions.

The older env module remains the transport implementation. Use this class for
new sessions: the server sends both hello and an initial observation on accept.
"""
from pathlib import Path
import json

from .env import BridgeError, IsaacBridgeEnv


class IsaacTrainingEnv(IsaacBridgeEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._arena_seed = None
        self._arena_anchor = None

    def connect(self):
        hello = super().connect()
        self._expect_obs()
        return hello

    def _reset_seeded_start(self, seed):
        # --set-stage boots without menus, but keeps the engine's test-run flag.
        # Its ordinary `seed` console reset silently chooses another run seed.
        # Set the Seeds object explicitly, then regenerate Basement via stage.
        obs, info = self.reset(phases=[
            ["restart 0"],
            [f"lua Game():GetSeeds():SetStartSeed({json.dumps(seed)})", "stage 1"],
        ], settle=2)
        actual = self.query_info()
        if actual["start_seed"] != seed:
            raise BridgeError(f"Requested seed {seed!r}, engine reported {actual['start_seed']!r}")
        return obs, {**info, **actual}

    def reset_monstro(self, seed="9AM0 7PRP"):
        """Isaac, no collectibles, one original Monstro in an empty starting room.

        This is a combat arena, not a generated boss-room/reward simulation.
        Player and boss placement is setup only; combat uses normal game inputs.
        """
        initialize = self._arena_seed != seed
        if initialize:
            # Strip items before entering the seeded room: rewind restores this snapshot.
            self.reset(phases=[['restart 0'], ['remove *',
                f'lua Game():GetSeeds():SetStartSeed({json.dumps(seed)})', 'stage 1']], settle=2)
            # Use the native rewind snapshot for the very first episode too. The
            # test-start run has different transient spawn scaling before this rewind.
            obs, info = self.reset(phases=[['rewind']], settle=2)
            self._arena_seed = seed
            self._arena_anchor = (obs['room']['room_idx'], info['room_spawn_seed'])
        else:
            obs, info = self.reset(phases=[['rewind']], settle=2)
        if info['start_seed'] != seed or (obs['room']['room_idx'], info['room_spawn_seed']) != self._arena_anchor:
            raise BridgeError('rewind did not restore the configured room checkpoint')
        # Console-spawned NPCs are not in the engine's entrance snapshot. Reapply
        # only the local arena template; no restart, reseed or floor regeneration.
        scenario = Path(__file__).resolve().parents[2] / "scenarios" / "monstro_empty.lua"
        self.exec(f"luarun {scenario.as_posix()}")
        for _ in range(90):
            obs, _, _, _, info = self.step({}, repeat=1)
            enemies = self.visible_enemies(obs)
            if len(enemies) == 1 and enemies[0]["type"] == 20:
                player = obs["players"][0]
                if player["ptype"] != 0 or player["active"] != 0 or player["hearts"] != 6:
                    raise BridgeError("Monstro arena player initialization failed")
                if obs["room"]["alive"] != 1 or obs["room"]["clear"]:
                    raise BridgeError("Monstro arena enemy/clear state is invalid")
                return obs, {**info, 'reset_kind': 'initialize' if initialize else 'rewind'}
        raise BridgeError("Monstro did not become visible within 90 setup frames")

    def reset_safe(self, seed="9AM0 7PRP"):
        """Return to an enemy-free room before releasing the blocking connection."""
        if self._arena_seed == seed:
            obs, info = self.reset(phases=[['rewind']], settle=2)
        else:
            obs, info = self._reset_seeded_start(seed)
        if obs["room"]["alive"] != 0:
            raise BridgeError("Safe shutdown room contains enemies")
        return obs, info
