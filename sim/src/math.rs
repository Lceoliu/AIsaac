//! 二维向量。游戏使用 f32；除法/开方按原版顺序写，避免重排导致的数值差异。

use std::ops::{Add, AddAssign, Mul, Neg, Sub, SubAssign};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    pub const ZERO: Vec2 = Vec2 { x: 0.0, y: 0.0 };

    pub const fn new(x: f32, y: f32) -> Vec2 {
        Vec2 { x, y }
    }

    pub fn length_sq(self) -> f32 {
        self.x * self.x + self.y * self.y
    }

    /// 原版 `FUN_00435a50`：`sqrt`，负数直接返回原值（这里负数不会出现）。
    pub fn length(self) -> f32 {
        let s = self.length_sq();
        if s > 0.0 {
            s.sqrt()
        } else {
            s
        }
    }

    /// 原版 `FUN_00a0ffc0`（KAGE Vector::Normalize）：长度大于 0 时缩放为单位向量，否则原样返回。
    pub fn normalized(self) -> Vec2 {
        let len = self.length();
        if len > 0.0 {
            let inv = 1.0 / len;
            Vec2::new(self.x * inv, self.y * inv)
        } else {
            self
        }
    }

    pub fn dot(self, o: Vec2) -> f32 {
        self.x * o.x + self.y * o.y
    }

    pub fn is_zero(self) -> bool {
        self.x == 0.0 && self.y == 0.0
    }
}

impl Add for Vec2 {
    type Output = Vec2;
    fn add(self, o: Vec2) -> Vec2 {
        Vec2::new(self.x + o.x, self.y + o.y)
    }
}
impl Sub for Vec2 {
    type Output = Vec2;
    fn sub(self, o: Vec2) -> Vec2 {
        Vec2::new(self.x - o.x, self.y - o.y)
    }
}
impl Mul<f32> for Vec2 {
    type Output = Vec2;
    fn mul(self, k: f32) -> Vec2 {
        Vec2::new(self.x * k, self.y * k)
    }
}
impl Neg for Vec2 {
    type Output = Vec2;
    fn neg(self) -> Vec2 {
        Vec2::new(-self.x, -self.y)
    }
}
impl AddAssign for Vec2 {
    fn add_assign(&mut self, o: Vec2) {
        self.x += o.x;
        self.y += o.y;
    }
}
impl SubAssign for Vec2 {
    fn sub_assign(&mut self, o: Vec2) {
        self.x -= o.x;
        self.y -= o.y;
    }
}
