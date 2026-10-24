"""Pygame presentation built exclusively from the policy observation."""
import os
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')

import pygame

SIZE = (1080, 720)
ORIGIN = (42, 150)
SCALE = 1.2
BG = (23, 27, 29)
INK = (229, 225, 213)
MUTED = (142, 152, 151)
TEAL = (103, 177, 179)
RED = (199, 112, 96)


class Renderer:
    def __init__(self):
        pygame.font.init()
        self.small = pygame.font.SysFont('Segoe UI', 16)
        self.body = pygame.font.SysFont('Segoe UI', 19)
        self.title = pygame.font.SysFont('Segoe UI', 32, bold=True)
        self.big = pygame.font.SysFont('Segoe UI', 36, bold=True)

    def text(self, surface, text, pos, color=INK, font=None):
        surface.blit((font or self.body).render(text, True, color), pos)

    @staticmethod
    def point(x, y):
        return round(ORIGIN[0] + x * SCALE), round(ORIGIN[1] + y * SCALE)

    def draw(self, obs, *, outcome='running', mode='SCRIPTED DEMO'):
        surface = pygame.Surface(SIZE)
        surface.fill(BG)
        self.text(surface, 'ISAAC / ROOM LAB', (40, 27), font=self.title)
        self.text(surface, 'SINGLE-ROOM COMBAT     /     APPROXIMATE MODEL', (42, 75), MUTED, self.small)
        pygame.draw.line(surface, (68, 76, 76), (42, 111), (1038, 111))
        room = pygame.Rect(*ORIGIN, round(obs['room'][0] * SCALE), round(obs['room'][1] * SCALE))
        pygame.draw.rect(surface, (79, 76, 69), room.inflate(22, 22))
        pygame.draw.rect(surface, (57, 59, 56), room)
        # Static floor marks: visual texture must not consume the environment RNG.
        for row in range(8):
            for col in range(12):
                px, py = self.point(col * 55 + 13, row * 53 + 17)
                pygame.draw.line(surface, (63, 65, 60), (px, py), (px + 20, py), 1)
        door = pygame.Rect(room.centerx - 26, room.top - 11, 52, 12)
        pygame.draw.rect(surface, TEAL if outcome == 'cleared' else (118, 106, 86), door)
        old_clip = surface.get_clip()
        surface.set_clip(room)
        for left, top, width, height in obs['obstacles']:
            rect = pygame.Rect(*self.point(left, top), round(width * SCALE), round(height * SCALE))
            pygame.draw.rect(surface, (40, 42, 40), rect.move(4, 6), border_radius=4)
            pygame.draw.rect(surface, (117, 112, 98), rect, border_radius=4)
            pygame.draw.line(surface, (147, 139, 121), rect.topleft, rect.topright, 3)
            pygame.draw.line(surface, (77, 78, 69), rect.midtop, rect.center, 2)
        for kind, x, y, vx, vy, radius in obs['enemies']:
            center = self.point(x, y)
            r = round(radius * SCALE)
            pygame.draw.ellipse(surface, (37, 39, 37), (center[0] - r, center[1] + r - 6, 2 * r, 11))
            color = RED if kind == 0 else (180, 150, 106)
            pygame.draw.circle(surface, color, center, r)
            if kind == 1:
                pygame.draw.circle(surface, (72, 56, 46), center, 7)
            else:
                for dx in (-6, 6):
                    pygame.draw.circle(surface, (54, 40, 38), (center[0] + dx, center[1] - 3), 3)
        px, py, vx, vy, radius, hp, invulnerable = obs['player']
        center = self.point(px, py)
        r = round(radius * SCALE)
        pygame.draw.ellipse(surface, (35, 37, 35), (center[0] - r, center[1] + r - 3, r * 2, 9))
        # Ring is a visible boolean, never an exact remaining invulnerability timer.
        if invulnerable:
            pygame.draw.circle(surface, TEAL, center, r + 6, 2)
        pygame.draw.circle(surface, (222, 199, 168), center, r)
        for dx in (-5, 5):
            pygame.draw.circle(surface, (41, 43, 44), (center[0] + dx, center[1] - 2), 3)
            pygame.draw.line(surface, TEAL, (center[0] + dx, center[1] + 2), (center[0] + dx, center[1] + 9), 2)
        for team, x, y, vx, vy, radius in obs['bullets']:
            center = self.point(x, y)
            color = TEAL if team == 0 else RED
            pygame.draw.line(surface, (70, 103, 103) if team == 0 else (108, 75, 65),
                             self.point(x - vx * 0.025, y - vy * 0.025), center, 3)
            pygame.draw.circle(surface, color, center, round(radius * SCALE))
            pygame.draw.circle(surface, (235, 226, 207), (center[0] - 1, center[1] - 2), 2)
        surface.set_clip(old_clip)

        x = 850
        self.text(surface, 'PLAYER', (x, 154), MUTED, self.small)
        for i in range(6):
            pygame.draw.circle(surface, RED if i < hp else (68, 62, 59), (x + 10 + i * 29, 198), 9)
        self.text(surface, f'{int(hp)} / 6 health units', (x, 221), font=self.small)
        self.text(surface, 'VISIBLE HOSTILES', (x, 274), MUTED, self.small)
        self.text(surface, str(len(obs['enemies'])), (x, 301), font=self.big)
        self.text(surface, f"{obs['tick'] / 60:05.1f}s  simulated", (x, 362))
        self.text(surface, mode, (x, 411), TEAL, self.small)
        for label, yy in [('WASD   move', 461), ('Arrows  shoot', 490), ('TAB      demo / manual', 519),
                          ('SPACE  pause', 548), ('R           same seed', 577), ('N           new seed', 606)]:
            self.text(surface, label, (x, yy), MUTED, self.small)
        if outcome != 'running':
            label = {'cleared': 'ROOM CLEARED', 'dead': 'PLAYER DOWN', 'timeout': 'TIME LIMIT', 'paused': 'PAUSED'}[outcome]
            banner = pygame.Rect(room.left, room.centery - 35, room.width, 70)
            pygame.draw.rect(surface, BG, banner)
            rendered = self.big.render(label, True, TEAL if outcome == 'cleared' else INK)
            surface.blit(rendered, rendered.get_rect(center=banner.center))
        self.text(surface, 'Teal: tears    /    Red: pursuers + hostile shots    /    Sand: fan shooters', (42, 657), MUTED, self.small)
        self.text(surface, 'Local prototype. No original game process, assets or trained policy.', (42, 684), MUTED, self.small)
        return surface
