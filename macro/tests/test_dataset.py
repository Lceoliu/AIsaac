import json

from isaac_macro.dataset import floor_arrays, iter_floors
from isaac_macro.levelgen import GRID, TRAVEL


def test_door_grid_is_symmetric(rc):
    for rec, floor in iter_floors(rc, 10, seed=5):
        a = floor_arrays(floor)
        doors, room = a['doors'], a['room']
        for y in range(GRID):
            for x in range(GRID):
                for d, (dx, dy) in enumerate(TRAVEL):
                    if doors[y, x] >> d & 1:
                        nx, ny = x + dx, y + dy
                        assert room[ny, nx] >= 0 and room[ny, nx] != room[y, x]
                        assert doors[ny, nx] >> ((d + 2) % 4) & 1
        json.dumps(rec)
