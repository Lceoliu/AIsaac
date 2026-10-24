//! 打印一个 Gaper 在空的 1×1 Basement 房间里追击静止玩家的轨迹（每 10 帧一行）。
//! 运行：cargo run --example gaper_chase

use isaac_sim::{Event, Room, Vec2, World};

fn main() {
    let mut world = World::new(Room::new_rectangular(15, 9), 42);
    let player = world.spawn_player_stub(Vec2::new(500.0, 300.0));
    let gaper = world.spawn_gaper(1, Vec2::new(100.0, 180.0));
    println!("frame  gaper.x  gaper.y  vel.x  vel.y  |vel|  state  anim");
    for frame in 0..200u32 {
        world.step();
        let g = world.entity(gaper).unwrap();
        let npc = g.npc();
        if frame % 10 == 0 {
            println!(
                "{:5} {:8.2} {:8.2} {:6.2} {:6.2} {:6.2} {:6} {}",
                frame,
                g.pos.x,
                g.pos.y,
                g.vel.x,
                g.vel.y,
                g.vel.length(),
                npc.state,
                npc.anim
            );
        }
        for ev in &world.events {
            if let Event::Contact { .. } = ev {
                println!(
                    "contact at frame {} -> player {:?}",
                    frame,
                    world.entity(player).unwrap().pos
                );
            }
        }
        world.events.clear();
    }
}
