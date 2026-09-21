//! Replay keyboard actions on the same simulation used by training.
//! Input: reset x y fire_delay, or move(0..8) shoot(0..4), one per line.
use isaac_sim::{entity::EntityKind, math::Vec2, player::PlayerInput, room::Room, world::World};
use std::io::{self, BufRead, Write};

fn main() {
    let mut world = World::new(Room::new_rectangular(15, 9), 42);
    let mut player = world.spawn_player(Vec2::new(320.0, 280.0));
    let moves = [
        (0., 0.),
        (0., -1.),
        (1., -1.),
        (1., 0.),
        (1., 1.),
        (0., 1.),
        (-1., 1.),
        (-1., 0.),
        (-1., -1.),
    ];
    let shots = [(0., 0.), (0., -1.), (1., 0.), (0., 1.), (-1., 0.)];
    for line in io::stdin().lock().lines() {
        let line = line.unwrap();
        let fields: Vec<_> = line.split_whitespace().collect();
        if fields[0] == "reset" {
            world = World::new(Room::new_rectangular(15, 9), 42);
            player = world.spawn_player(Vec2::new(
                fields[1].parse().unwrap(),
                fields[2].parse().unwrap(),
            ));
            world.entity_mut(player).unwrap().player_mut().fire_delay = fields[3].parse().unwrap();
        } else if fields[0] == "tear" {
            world = World::new(Room::new_rectangular(15, 9), 42);
            player = world.spawn_player(Vec2::new(320.0, 280.0));
            let f: Vec<f32> = fields[1..].iter().map(|s| s.parse().unwrap()).collect();
            let mut rng = isaac_sim::rng::Rng::new(42);
            let mut t = isaac_sim::tear::fire_tear(
                world.entity_mut(player).unwrap(),
                Vec2::new(f[0], f[1]),
                Vec2::new(f[2], f[3]),
                &mut rng,
            );
            t.pos = Vec2::new(f[0], f[1]);
            t.vel = Vec2::new(f[2], f[3]);
            t.tear_mut().height = f[4];
            t.tear_mut().falling_speed = f[5];
            isaac_sim::physics::resolve_grid_collision(&mut t, &world.room, false);
            world.spawn(t);
        } else if fields[0] == "tick" {
            world.step();
        } else {
            let half = fields[0] == "half";
            let logic = fields[0] == "logic";
            let offset = usize::from(half || logic);
            let m = moves[fields[offset].parse::<usize>().unwrap()];
            let s = shots[fields[offset + 1].parse::<usize>().unwrap()];
            world.events.clear();
            world.set_player_input(player, PlayerInput::new(m.0, m.1, s.0, s.1));
            if half {
                world.interpolate_players();
            } else if logic {
                world.step_logic();
            } else {
                world.step();
            }
        }
        let p = world.entity(player).unwrap();
        print!(
            "{{\"player\":{{\"pos\":[{},{}],\"vel\":[{},{}],\"fire_delay\":{}}},\"tears\":[",
            p.pos.x,
            p.pos.y,
            p.vel.x,
            p.vel.y,
            p.player().fire_delay
        );
        let mut sep = "";
        for t in world
            .entities
            .iter()
            .filter(|e| e.kind == EntityKind::Tear && e.exists)
        {
            print!("{}{{\"id\":{},\"pos\":[{},{}],\"vel\":[{},{}],\"height\":{},\"fall\":{},\"dead\":{}}}",sep,t.id,t.pos.x,t.pos.y,t.vel.x,t.vel.y,t.tear().height,t.tear().falling_speed,t.dead);
            sep = ",";
        }
        println!("]}}");
        io::stdout().flush().unwrap();
    }
}
