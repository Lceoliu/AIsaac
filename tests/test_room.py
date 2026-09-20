from copy import deepcopy
import random
import subprocess
import sys

import numpy as np
import pytest

from isaac_room.env import Bullet, Config, Enemy, HEIGHT, RoomEnv, WIDTH
from isaac_room.geometry import circle_rect, segment_circle, swept_circle_rect
from isaac_room.policy import scripted_action


def same_obs(a, b):
    assert a.keys() == b.keys()
    for key in a:
        np.testing.assert_array_equal(a[key], b[key], err_msg=key)


def quiet_room(**kwargs):
    env = RoomEnv(Config(action_repeat=1, **kwargs))
    env.reset(seed=1)
    env.enemies = [Enemy(1, 600, 30, hp=1000, cooldown=100000)]
    return env


@pytest.mark.parametrize('layout', ['empty', 'pillars'])
def test_seeded_action_replay(layout):
    a, b = RoomEnv(Config(layout=layout)), RoomEnv(Config(layout=layout))
    same_obs(a.reset(seed=42)[0], b.reset(seed=42)[0])
    actions = random.Random(12)
    for _ in range(250):
        action = actions.randrange(9), actions.randrange(5)
        oa, ra, ta, xa, ia = a.step(action)
        ob, rb, tb, xb, ib = b.step(action)
        same_obs(oa, ob)
        assert (ra, ta, xa, ia) == (rb, tb, xb, ib)
        if ta or xa:
            break


def test_reset_erases_episode_and_reproduces_seed():
    env = RoomEnv()
    original, _ = env.reset(seed=9)
    for _ in range(30):
        env.step((3, 1))
    obs, info = env.reset(seed=9)
    same_obs(original, obs)
    assert info == {'outcome': 'running', 'ticks': 0, 'kills': 0, 'damage_taken': 0}
    assert not env.bullets
    assert env.player.cooldown == env.player.invulnerable == 0


def test_frame_repeat_matches_individual_physics_ticks():
    a, b = RoomEnv(Config(action_repeat=1)), RoomEnv(Config(action_repeat=4))
    a.reset(seed=3)
    b.reset(seed=3)
    for move in range(9):
        reward = sum(a.step((move, 1))[1] for _ in range(4))
        obs, rb, *_ = b.step((move, 1))
        assert reward == rb
        same_obs(a.observe(), obs)


@pytest.mark.parametrize('move', range(9))
def test_player_stays_inside_walls(move):
    env = quiet_room(layout='empty')
    for _ in range(300):
        env.step((move, 0))
        p = env.player
        assert p.radius <= p.x <= WIDTH - p.radius
        assert p.radius <= p.y <= HEIGHT - p.radius


def test_rock_blocks_player_but_allows_sliding():
    env = quiet_room()
    env.player.x, env.player.y = 190, 180
    for _ in range(40):
        env.step((3, 0))
    assert env.player.x <= 208.00001
    old_y = env.player.y
    for _ in range(50):
        env.step((2, 0))
        assert not any(circle_rect(env.player.x, env.player.y, 12, b) for b in env.blocks)
    assert env.player.y < old_y


def test_bullet_sweep_hits_rock_before_enemy():
    env = quiet_room()
    env.enemies = [Enemy(1, 320, 190, cooldown=1000)]
    env.bullets = [Bullet(0, 100, 190, 18000, 0)]
    env.step((0, 0))
    assert env.enemies[0].hp == 6
    assert not env.bullets


def test_bullet_hits_only_first_target_without_tunneling():
    env = quiet_room(layout='empty')
    env.enemies = [Enemy(1, 200, 100, cooldown=1000), Enemy(1, 300, 100, cooldown=1000)]
    env.bullets = [Bullet(0, 100, 100, 18000, 0)]
    env.step((0, 0))
    assert [e.hp for e in env.enemies] == [4, 6]
    assert not env.bullets


def test_bullet_outside_wall_is_removed():
    env = quiet_room()
    env.bullets = [Bullet(0, 639, 100, 360, 0)]
    env.step((0, 0))
    assert not env.bullets


def test_damage_invulnerability_and_contact():
    env = quiet_room(layout='empty')
    p = env.player
    env.bullets = [Bullet(1, p.x, p.y, 0, 0) for _ in range(3)]
    env.step((0, 0))
    assert p.hp == 5 and p.invulnerable == 45
    env.enemies = [Enemy(1, p.x, p.y, cooldown=1000)]
    for _ in range(44):
        env.step((0, 0))
    assert p.hp == 5
    env.step((0, 0))
    assert p.hp == 4 and env.damage_taken == 2


def test_shooting_cadence_and_expiry():
    env = quiet_room(layout='empty')
    for _ in range(13):
        env.step((0, 1))
    assert len(env.bullets) == 2
    env.bullets = [Bullet(0, 50, 50, 0, 0, ttl=1)]
    env.step((0, 0))
    assert not env.bullets


def test_chaser_moves_and_shooter_fires_fan():
    env = quiet_room(layout='empty')
    env.enemies = [Enemy(0, 100, 100), Enemy(1, 500, 100, cooldown=1)]
    env.step((0, 0))
    assert env.enemies[0].x > 100 and env.enemies[0].y > 100
    assert len(env.bullets) == 3
    assert len({(b.vx, b.vy) for b in env.bullets}) == 3


def test_room_clear_is_terminal_and_disallows_further_steps():
    env = quiet_room(layout='empty')
    env.enemies = [Enemy(1, 100, 100, hp=2, cooldown=1000)]
    env.bullets = [Bullet(0, 100, 100, 0, 0)]
    _, reward, terminated, truncated, info = env.step((0, 0))
    assert terminated and not truncated and reward > 5
    assert info['outcome'] == 'cleared' and info['kills'] == 1
    with pytest.raises(RuntimeError):
        env.step((0, 0))


def test_death_has_priority_over_simultaneous_clear():
    env = quiet_room(layout='empty')
    p = env.player
    p.hp = 1
    env.enemies = [Enemy(1, 100, 100, hp=2, cooldown=1000)]
    env.bullets = [Bullet(0, 100, 100, 0, 0), Bullet(1, p.x, p.y, 0, 0)]
    _, _, terminated, truncated, info = env.step((0, 0))
    assert terminated and not truncated and info['outcome'] == 'dead'


def test_time_limit_truncates_at_exact_tick():
    env = RoomEnv(Config(action_repeat=4, max_ticks=3))
    env.reset(seed=2)
    _, _, terminated, truncated, info = env.step((0, 0))
    assert not terminated and truncated and info['ticks'] == 3


@pytest.mark.parametrize('action', [(-1, 0), (9, 1), (0, 5), (0,), (1.5, 0)])
def test_invalid_actions_fail(action):
    env = quiet_room()
    with pytest.raises((TypeError, ValueError)):
        env.step(action)


def test_reset_required_and_close_disables_step():
    env = RoomEnv()
    with pytest.raises(RuntimeError):
        env.step((0, 0))
    env.reset(seed=1)
    env.close()
    with pytest.raises(RuntimeError):
        env.step((0, 0))


def test_hidden_state_noninterference():
    env = RoomEnv()
    original, _ = env.reset(seed=6)
    env.enemies[0].hp = 1
    env.enemies[0].cooldown = 500
    env.enemies.append(Enemy(0, 300, 250, visible=False))
    env.enemies.append(Enemy(1, 1000, 1000))
    env.bullets.append(Bullet(1, 200, 100, 0, 0, visible=False))
    env.rng.seed(12345)
    same_obs(original, env.observe())
    assert scripted_action(original) == scripted_action(env.observe())


def test_observation_has_no_reference_to_mutable_state():
    env = RoomEnv()
    obs, _ = env.reset(seed=3)
    original = deepcopy(obs)
    obs['player'][0] = -999
    obs['enemies'][0, 1] = -999
    same_obs(original, env.observe())


def test_headless_import_does_not_load_pygame():
    result = subprocess.run([sys.executable, '-c',
        "import sys; from isaac_room.env import RoomEnv; e=RoomEnv(); e.reset(seed=1); e.step((0,0)); assert 'pygame' not in sys.modules"],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_swept_geometry_corner_and_miss():
    assert segment_circle(0, 0, 20, 0, 10, 0, 2) == pytest.approx(0.4)
    assert segment_circle(0, 0, 20, 0, 10, 10, 2) is None
    assert swept_circle_rect(0, 8.1, 20, 8.1, 2, (10, 10, 5, 5)) is not None
    assert swept_circle_rect(0, 7, 20, 7, 2, (10, 10, 5, 5)) is None


def test_parallel_matches_serial_episodes():
    from isaac_room.benchmark import evaluate
    args = dict(episodes=4, policy='random', seed=10, max_ticks=90)
    serial, parallel = evaluate(**args), evaluate(workers=2, **args)
    assert serial['episodes_detail'] == parallel['episodes_detail']


def test_render_uses_same_visible_projection():
    from isaac_room.view import Renderer, SIZE, pygame
    env = RoomEnv()
    obs, _ = env.reset(seed=7)
    renderer = Renderer()
    first = renderer.draw(obs)
    env.enemies.append(Enemy(0, 200, 200, visible=False))
    env.enemies[0].hp = 1
    second = renderer.draw(env.observe())
    assert first.get_size() == SIZE
    assert pygame.image.tobytes(first, 'RGB') == pygame.image.tobytes(second, 'RGB')
    assert np.asarray(pygame.surfarray.array3d(first)).std() > 10


def test_keyboard_mapping():
    from collections import defaultdict
    from isaac_room.__main__ import keyboard_action
    from isaac_room.view import pygame
    keys = defaultdict(bool)
    keys[pygame.K_w] = keys[pygame.K_d] = keys[pygame.K_LEFT] = True
    assert keyboard_action(keys, pygame) == (2, 4)
    keys[pygame.K_a] = keys[pygame.K_s] = True
    assert keyboard_action(keys, pygame) == (0, 4)


def test_interactive_event_loop_with_dummy_display(monkeypatch):
    from collections import defaultdict
    from isaac_room import __main__ as app
    from isaac_room.view import pygame
    # Exercise actual UI loop without opening a desktop window or sending OS keys.
    monkeypatch.setenv('SDL_VIDEODRIVER', 'dummy')
    monkeypatch.setattr(sys, 'argv', ['isaac_room', '--seed', '7'])
    created = []

    class ObservedEnv(RoomEnv):
        def __init__(self, config):
            super().__init__(config)
            self.reset_seeds = []
            self.actions = []
            created.append(self)

        def reset(self, **kwargs):
            self.reset_seeds.append(kwargs['seed'])
            return super().reset(**kwargs)

        def step(self, action):
            self.actions.append(action)
            return super().step(action)

    class FixedClock:
        def tick(self, fps):
            return 17

    events = iter([[], [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_TAB)],
                   [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE)],
                   [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r)],
                   [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_n)],
                   [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE)]])
    keys = defaultdict(bool, {pygame.K_d: True, pygame.K_UP: True})
    monkeypatch.setattr(app, 'RoomEnv', ObservedEnv)
    monkeypatch.setattr(pygame.event, 'get', lambda: next(events))
    monkeypatch.setattr(pygame.key, 'get_pressed', lambda: keys)
    monkeypatch.setattr(pygame.time, 'Clock', FixedClock)
    app.main()
    assert created[0].actions[0] == (3, 1)
    assert len(created[0].actions) == 2  # remaining iterations paused
    assert created[0].reset_seeds == [7, 7, 8]
    assert not created[0].ready
