//! 粗略吞吐：1×1 空房、1 名玩家占位、4 个 Gaper，跑 N 帧并打印帧/秒。
//! 运行：cargo run --release --example bench
use isaac_sim::{Room, Vec2, World};
use std::time::Instant;

fn main() {
    let frames: u32 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(300_000);
    let mut world = World::new(Room::new_rectangular(15, 9), 1);
    world.spawn_player_stub(Vec2::new(320.0, 280.0));
    for (i, p) in [
        (100.0, 180.0),
        (540.0, 180.0),
        (100.0, 380.0),
        (540.0, 380.0),
    ]
    .iter()
    .enumerate()
    {
        world.spawn_gaper((i % 3) as i32, Vec2::new(p.0, p.1));
    }
    let t = Instant::now();
    for _ in 0..frames {
        world.step();
        world.events.clear();
    }
    let dt = t.elapsed().as_secs_f64();
    println!(
        "{} frames, {} entities, {:.3} s, {:.0} frames/s (single core)",
        frames,
        world.entities.len(),
        dt,
        frames as f64 / dt
    );
}
