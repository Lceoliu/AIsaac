import argparse
import json
from pathlib import Path

from .env import Config, MOVE, RoomEnv
from .policy import scripted_action


def keyboard_action(keys, pygame):
    dx = int(keys[pygame.K_d]) - int(keys[pygame.K_a])
    dy = int(keys[pygame.K_s]) - int(keys[pygame.K_w])
    move = next(i for i, (mx, my) in enumerate(MOVE)
                if (0 if mx == 0 else (1 if mx > 0 else -1),
                    0 if my == 0 else (1 if my > 0 else -1)) == (dx, dy))
    shoot = next((i for i, key in enumerate((pygame.K_UP, pygame.K_RIGHT,
                                           pygame.K_DOWN, pygame.K_LEFT), 1) if keys[key]), 0)
    return move, shoot


def main():
    parser = argparse.ArgumentParser(description='Independent approximate Isaac single-room prototype')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--layout', choices=('empty', 'pillars'), default='pillars')
    parser.add_argument('--demo', action='store_true', help='start with scripted controller')
    parser.add_argument('--screenshot', type=Path, help='offscreen demo capture, no desktop window')
    parser.add_argument('--frames', type=int, default=360, help='physics ticks for offscreen capture')
    args = parser.parse_args()
    if args.frames < 0:
        parser.error('--frames must be nonnegative')
    from .view import Renderer, SIZE, pygame
    env = RoomEnv(Config(layout=args.layout, action_repeat=1))
    obs, info = env.reset(seed=args.seed)
    renderer = Renderer()
    if args.screenshot:
        for _ in range(args.frames):
            obs, _, terminated, truncated, info = env.step(scripted_action(obs))
            if terminated or truncated:
                break
        args.screenshot.parent.mkdir(parents=True, exist_ok=True)
        pygame.image.save(renderer.draw(obs, outcome=info['outcome']), str(args.screenshot))
        print(json.dumps({'screenshot': str(args.screenshot.resolve()), **info}))
        env.close()
        pygame.quit()
        return
    pygame.display.init()
    screen = pygame.display.set_mode(SIZE)
    pygame.display.set_caption('Isaac Room Lab | standalone prototype')
    clock = pygame.time.Clock()
    demo, paused, running = args.demo, False, True
    accumulator = 0.0
    while running:
        elapsed = min(clock.tick(60) / 1000, 0.25)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_TAB:
                    demo = not demo
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key in (pygame.K_r, pygame.K_n):
                    if event.key == pygame.K_n:
                        args.seed += 1
                    obs, info = env.reset(seed=args.seed)
                    accumulator = 0
        if paused or info['outcome'] != 'running':
            accumulator = 0
        else:
            accumulator += elapsed
            manual_action = keyboard_action(pygame.key.get_pressed(), pygame)
            while accumulator >= 1 / 60 and info['outcome'] == 'running':
                action = scripted_action(obs) if demo else manual_action
                obs, _, _, _, info = env.step(action)
                accumulator -= 1 / 60
        screen.blit(renderer.draw(obs, outcome='paused' if paused else info['outcome'],
                                  mode='SCRIPTED DEMO' if demo else 'MANUAL CONTROL'), (0, 0))
        pygame.display.flip()
    env.close()
    pygame.quit()


if __name__ == '__main__':
    main()
