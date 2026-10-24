//! 确定性随机数。
//!
//! 原版 `FUN_006eef60`（RVA 0x002EEF60）是一个带自定义 tempering 常量（0xff3a58ad）的
//! 梅森旋转变体，实体级 RNG 是 xorshift（移位 5/7/…）。这里先用 xorshift32 保证可复现，
//! **不与原版比特级一致**；需要复现原版随机序列时再替换。所有调用点都保留了原版的取模用法。

#[derive(Clone, Debug)]
pub struct Rng {
    state: u32,
}

impl Rng {
    pub fn new(seed: u32) -> Rng {
        Rng {
            state: if seed == 0 { 0x9E37_79B9 } else { seed },
        }
    }

    /// 32 位随机整数（对应原版 `FUN_006eef60()` 返回的 uint）。
    pub fn next_u32(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.state = x;
        x
    }

    /// 原版把 `FUN_006eef60()` 的结果按 `(double)i + (i<0 ? 2^32 : 0)` 转成 [0, 2^32) 再乘
    /// `DAT_00ba9ff4 = 2^-32` 得到 [0,1) 的 float。
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u32() as f64 * (1.0 / 4294967296.0)) as f32
    }
}
