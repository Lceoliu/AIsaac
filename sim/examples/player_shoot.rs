//! 玩家站在房间左侧向右射击，一个 Gaper 从右侧逼近：打印玩家/眼泪/Gaper 的位置与事件。
//! 运行：cargo run --example player_shoot

use isaac_sim::{Event, PlayerInput, Room, Vec2, World};

fn main() {
    let mut world = World::new(Room::new_rectangular(15, 9), 42);
    let player = world.spawn_player(Vec2::new(150.0, 280.0));
    let gaper = world.spawn_gaper(1, Vec2::new(480.0, 280.0));
    world.set_player_input(player, PlayerInput::new(0.0, 0.0, 1.0, 0.0));
    println!("frame  player.x  gaper.x  gaper.hp  tears  events");
    for frame in 0..140u32 {
        world.step();
        let player_x = world.entity(player).map(|p| p.pos.x).unwrap_or(0.0);
        let gaper_info = world
            .entity(gaper)
            .map(|g| (format!("{:.1}", g.pos.x), format!("{:.1}", g.hp)));
        let tears: Vec<String> = world
            .tears()
            .map(|t| format!("{:.0}@h{:.1}", t.pos.x, t.tear().height))
            .collect();
        let mut evs = Vec::new();
        for ev in &world.events {
            match ev {
                Event::TearFired { tear, .. } => evs.push(format!("fire#{}", tear)),
                Event::TearHit { damage, .. } => evs.push(format!("hit{:.1}", damage)),
                Event::TearRemoved { cause, .. } => evs.push(format!("gone(c{})", cause)),
                Event::NpcDied { .. } => evs.push("gaper died".to_string()),
                Event::PlayerDamaged { hp_after, .. } => evs.push(format!("hurt(hp {})", hp_after)),
                _ => {}
            }
        }
        world.events.clear();
        if frame % 5 == 0 || !evs.is_empty() {
            let (gx, ghp) = gaper_info
                .clone()
                .unwrap_or_else(|| ("-".into(), "-".into()));
            println!(
                "{:5} {:9.1} {:8} {:9} {:6} {}",
                frame,
                player_x,
                gx,
                ghp,
                tears.join(","),
                evs.join(" ")
            );
        }
        if gaper_info.is_none() {
            println!("Gaper removed at frame {}", frame);
            break;
        }
    }
}
