//! 房间网格、路径图（Room+0x76c）、边界与视线检测。
//!
//! 常量（均读自 J460 `.rdata`）：格宽 `DAT_00baa904 = 40`；x 原点 40（`(x-40)/40+0.5`）；
//! y 原点 `DAT_00baaa00 = 120`；网格最多 `0x1c0 = 448` 格。1×1 房间网格含墙为 15×9，
//! 可行走区左上角 (60,140)、右下角 (580,420)，由 `FUN_007f2390`（RVA 0x003F2390）第 166–172 行
//! 的公式得出：`topLeft = cellCenter(1,1) - 20`，`bottomRight = topLeft + (w-2, h-2)*40`。

use crate::math::Vec2;

pub const TILE: f32 = 40.0;
pub const GRID_X0: f32 = 40.0;
pub const GRID_Y0: f32 = 120.0;
pub const MAX_CELLS: usize = 448;

/// 路径图取值（Room+0x76c，每格一个 int）：
/// - 0：空地；1..=949：近期被 NPC 占据的标记（每 3 帧衰减 100；玩家不打标记）；
/// - 999：可破坏障碍（`FUN_0071dfc0` 等）；1000：实心（岩石、墙，`FUN_00711590`/`FUN_0071ee90`）；
/// - 3000：坑（`FUN_00714610`）；3999 会被衰减代码重置为 900。
pub const PATH_FREE: i32 = 0;
pub const PATH_MARK_MAX: i32 = 900;
pub const PATH_BREAKABLE: i32 = 999;
pub const PATH_SOLID: i32 = 1000;
pub const PATH_PIT: i32 = 3000;

/// `GridCollisionClass`（enums.lua）。
pub mod grid_collision {
    pub const NONE: i32 = 0;
    pub const PIT: i32 = 1;
    pub const OBJECT: i32 = 2;
    pub const SOLID: i32 = 3;
    pub const WALL: i32 = 4;
    pub const WALL_EXCEPT_PLAYER: i32 = 5;
}

/// `GridEntityType` 的子集。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GridKind {
    Empty,
    Wall,
    Rock,
    Pit,
}

#[derive(Clone, Copy, Debug)]
pub struct GridCell {
    pub kind: GridKind,
    /// 有网格实体时 `Room::GetGridCollision` 直接返回该实体的 `+0x3c` 碰撞类。
    pub collision_class: i32,
}

#[derive(Clone, Debug)]
pub struct Room {
    pub width: i32,
    pub height: i32,
    pub room_type: i32,
    pub top_left: Vec2,
    pub bottom_right: Vec2,
    pub cells: Vec<GridCell>,
    /// Room+0x76c。
    pub path: Vec<i32>,
    pub frame: u32,
    pub doors: [bool; 4],
    pub layout: u32,
}

impl Room {
    /// 建一个只有外墙的矩形房间；`width/height` 含墙（1×1 房为 15×9）。
    pub fn new_rectangular(width: i32, height: i32) -> Room {
        assert!(width >= 3 && height >= 3 && (width * height) as usize <= MAX_CELLS);
        let n = (width * height) as usize;
        let mut cells = vec![
            GridCell {
                kind: GridKind::Empty,
                collision_class: grid_collision::NONE,
            };
            n
        ];
        let mut path = vec![PATH_FREE; n];
        for row in 0..height {
            for col in 0..width {
                if row == 0 || col == 0 || row == height - 1 || col == width - 1 {
                    let i = (row * width + col) as usize;
                    // GridEntity_Wall：碰撞类 WALL(4)，路径值 1000（FUN_00711590 第 16 行）
                    cells[i] = GridCell {
                        kind: GridKind::Wall,
                        collision_class: grid_collision::WALL,
                    };
                    path[i] = PATH_SOLID;
                }
            }
        }
        let mut room = Room {
            width,
            height,
            room_type: 1,
            top_left: Vec2::ZERO,
            bottom_right: Vec2::ZERO,
            cells,
            path,
            frame: 0,
            doors: [false; 4],
            layout: 0,
        };
        // FUN_007f2390 第 166–172 行（矩形房）
        let c11 = room.cell_center(width + 1);
        room.top_left = Vec2::new(c11.x - 20.0, c11.y - 20.0);
        room.bottom_right = Vec2::new(
            room.top_left.x + (width - 2) as f32 * TILE,
            room.top_left.y + (height - 2) as f32 * TILE,
        );
        room
    }

    pub fn cell_count(&self) -> i32 {
        self.width * self.height
    }

    /// 放一块岩石：碰撞类 SOLID(3)，路径 1000。
    pub fn add_rock(&mut self, idx: i32) {
        let i = idx as usize;
        self.cells[i] = GridCell {
            kind: GridKind::Rock,
            collision_class: grid_collision::SOLID,
        };
        self.path[i] = PATH_SOLID;
    }

    /// 放一个坑：碰撞类 PIT(1)，路径 3000（FUN_00714610 第 18/24 行）。
    pub fn add_pit(&mut self, idx: i32) {
        let i = idx as usize;
        self.cells[i] = GridCell {
            kind: GridKind::Pit,
            collision_class: grid_collision::PIT,
        };
        self.path[i] = PATH_PIT;
    }

    pub fn index_of(&self, col: i32, row: i32) -> i32 {
        row * self.width + col
    }

    /// `Room::GetGridIndex`（FUN_00812c90，RVA 0x00412C90）。C 的 `(int)` 是向零截断，`as i32` 相同。
    pub fn grid_index(&self, pos: Vec2) -> i32 {
        let col = ((pos.x - GRID_X0) / TILE + 0.5) as i32;
        if col < 0 || col >= self.width {
            return -1;
        }
        let row = ((pos.y - GRID_Y0) / TILE + 0.5) as i32;
        if row < 0 || row >= self.height {
            return -1;
        }
        self.width * row + col
    }

    /// 格中心（FindGridPath 中反复出现的 `(idx % w)*40 + 40, (idx / w)*40 + 120`）。
    pub fn cell_center(&self, idx: i32) -> Vec2 {
        Vec2::new(
            (idx % self.width) as f32 * TILE + GRID_X0,
            (idx / self.width) as f32 * TILE + GRID_Y0,
        )
    }

    pub fn has_grid_entity(&self, idx: i32) -> bool {
        idx >= 0
            && (idx as usize) < self.cells.len()
            && self.cells[idx as usize].kind != GridKind::Empty
    }

    /// `Room::GetGridCollision`（FUN_007f0800，RVA 0x003F0800）。
    pub fn grid_collision(&self, idx: i32) -> i32 {
        if idx < 0 {
            return grid_collision::WALL;
        }
        let i = idx as usize;
        if i >= self.cells.len() {
            return grid_collision::WALL;
        }
        if self.cells[i].kind != GridKind::Empty {
            return self.cells[i].collision_class;
        }
        let v = self.path[i];
        if v < 3000 {
            if v < 1000 {
                grid_collision::NONE
            } else {
                grid_collision::SOLID
            }
        } else if v < 4000 {
            grid_collision::PIT
        } else {
            grid_collision::SOLID
        }
    }

    /// `Room::GetGridCollisionAtPos`（FUN_007f0780）。
    pub fn grid_collision_at(&self, pos: Vec2) -> i32 {
        self.grid_collision(self.grid_index(pos))
    }

    /// `Room::GetClampedPosition`（FUN_00812f50，矩形房分支）：把点夹到 `[topLeft+m, bottomRight-m]`。
    pub fn clamp_position(&self, pos: Vec2, margin: f32) -> Vec2 {
        let x = pos
            .x
            .max(self.top_left.x + margin)
            .min(self.bottom_right.x - margin);
        let y = pos
            .y
            .max(self.top_left.y + margin)
            .min(self.bottom_right.y - margin);
        Vec2::new(x, y)
    }

    /// `FUN_004cdd30`：给格子打占据标记。`path = path - path%1000 + marker`，仅当现有标记更小。
    pub fn set_marker(&mut self, idx: i32, marker: i32) {
        if idx < 0 || idx as usize >= self.path.len() || marker >= 951 {
            return;
        }
        let v = self.path[idx as usize];
        if v % 1000 < marker {
            self.path[idx as usize] = (v - v % 1000) + marker;
        }
    }

    /// Room::Update（FUN_00802980 第 832–860 行）：游戏帧计数能被 3 整除时衰减标记。
    /// `3999 → 900`；否则标记 `m = v % 1000`，`m < 901` 时 `v -= min(m, 100)`。
    pub fn update_path_markers(&mut self, game_frame: u32) {
        if !game_frame.is_multiple_of(3) {
            return;
        }
        for v in self.path.iter_mut() {
            if *v == 3999 {
                *v = 900;
            } else {
                let m = *v % 1000;
                if m < 0x385 {
                    *v -= m.min(100);
                }
            }
        }
    }

    /// `Room::CheckLine`（FUN_007f0df0，RVA 0x003F0DF0）的 mode 0 子集：
    /// 从 `p1` 向 `p2` 每 10 单位走一步，每步取四个角点偏移 (±5, ±5)（`DAT_00baca70`/`DAT_00bacbd0`），
    /// 任一采样格碰撞类不为 NONE 即视线被挡；路径值大于 `grid_path_threshold` 的采样格累计超过 8 个也算被挡。
    /// `ignore_walls`/`ignore_crushable` 原版参数这里固定为 false。
    pub fn line_check(&self, p1: Vec2, p2: Vec2, grid_path_threshold: i32) -> bool {
        let delta = p1 - p2;
        let dist = delta.length();
        if dist <= 0.0 {
            return true;
        }
        let step = Vec2::new((delta.x / dist) * 10.0, (delta.y / dist) * 10.0);
        let start_cell = self.grid_index(p1);
        let offsets = [
            Vec2::new(-5.0, -5.0),
            Vec2::new(-5.0, 5.0),
            Vec2::new(5.0, 5.0),
            Vec2::new(5.0, -5.0),
        ];
        let mut marked = 0i32;
        let mut cur = p1;
        let mut travelled = 0.0f32;
        while travelled < dist {
            cur -= step;
            for off in offsets.iter() {
                let p = cur + *off;
                let col = ((p.x - GRID_X0) / TILE + 0.5) as i32;
                let row = ((p.y - GRID_Y0) / TILE + 0.5) as i32;
                let idx = if col < 0 || col >= self.width || row < 0 || row >= self.height {
                    -1
                } else {
                    row * self.width + col
                };
                let class = if idx < 0 {
                    grid_collision::WALL
                } else if self.has_grid_entity(idx) {
                    self.cells[idx as usize].collision_class
                } else {
                    let v = self.path[idx as usize];
                    if (3000..4000).contains(&v) {
                        grid_collision::PIT
                    } else if v > 999 {
                        grid_collision::SOLID
                    } else {
                        grid_collision::NONE
                    }
                };
                if class == grid_collision::NONE {
                    if idx != start_cell {
                        let v = if idx >= 0 && (idx as usize) < self.path.len() {
                            self.path[idx as usize]
                        } else {
                            0
                        };
                        if v > grid_path_threshold {
                            marked += 1;
                        }
                    }
                } else {
                    return false;
                }
            }
            travelled += 10.0;
        }
        marked <= 8
    }

    /// `NPCAI_Pathfinder` 用到的按网格碰撞类调整路径值（FUN_007e7fb0，RVA 0x003E7FB0）：
    /// NOPITS(6) 把 3000 段折回；非 GROUND(5) 的其他类把 1000 段与 3000 段都折回。
    pub fn path_value_for_class(v: i32, grid_collision_class: i32) -> i32 {
        let v = v as u32;
        match grid_collision_class {
            6 => {
                let w = v.wrapping_sub(3000);
                if w < 1000 {
                    w as i32
                } else {
                    v as i32
                }
            }
            5 => v as i32,
            _ => {
                let mut w = v.wrapping_sub(1000);
                if w >= 1000 {
                    w = v;
                }
                if w.wrapping_sub(3000) < 1000 {
                    w = w.wrapping_sub(3000);
                }
                w as i32
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_by_one_room_bounds_and_indexing() {
        let room = Room::new_rectangular(15, 9);
        assert_eq!(room.top_left, Vec2::new(60.0, 140.0));
        assert_eq!(room.bottom_right, Vec2::new(580.0, 420.0));
        assert_eq!(room.grid_index(Vec2::new(80.0, 160.0)), 16);
        assert_eq!(room.grid_index(Vec2::new(60.0, 140.0)), 16);
        assert_eq!(room.grid_index(Vec2::new(59.9, 139.9)), 0);
        assert_eq!(room.cell_center(16), Vec2::new(80.0, 160.0));
        assert_eq!(room.grid_collision(0), grid_collision::WALL);
        assert_eq!(room.grid_collision(16), grid_collision::NONE);
        assert_eq!(room.grid_collision(-1), grid_collision::WALL);
    }

    #[test]
    fn markers_decay_by_100_every_third_frame() {
        let mut room = Room::new_rectangular(15, 9);
        room.set_marker(16, 900);
        assert_eq!(room.path[16], 900);
        for f in 1..=27 {
            room.update_path_markers(f);
        }
        assert_eq!(room.path[16], 0);
        room.add_rock(17);
        room.set_marker(17, 900);
        assert_eq!(room.path[17], 1900);
        room.update_path_markers(3);
        assert_eq!(room.path[17], 1800);
    }

    #[test]
    fn line_check_blocked_by_rock() {
        let mut room = Room::new_rectangular(15, 9);
        let a = room.cell_center(room.index_of(2, 4));
        let b = room.cell_center(room.index_of(8, 4));
        assert!(room.line_check(a, b, 0));
        room.add_rock(room.index_of(5, 4));
        assert!(!room.line_check(a, b, 0));
    }
}
