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
        } else if fields[0] == "monstro" {
            world = World::new(Room::new_rectangular(15, 9), fields[1].parse().unwrap());
            player = world.spawn_player(Vec2::new(320.0, 380.0));
            world.spawn_monstro(Vec2::new(320.0, 220.0));
        } else if fields[0] == "boss" {
            let v: Vec<f32> = fields[1..].iter().map(|s| s.parse().unwrap()).collect();
            world = World::new(Room::new_rectangular(15, 9), 42);
            player = world.spawn_player(Vec2::new(v[8], v[9]));
            let id = world.spawn_monstro(Vec2::new(v[2], v[3]));
            let b = world.entity_mut(id).unwrap();
            b.vel = Vec2::new(v[4], v[5]);
            b.npc_mut().state = v[0] as i32;
            b.npc_mut().anim_frame = v[1] as i32;
            b.npc_mut().target_pos = Vec2::new(v[6], v[7]);
            b.npc_mut().anim = match v[0] as i32 {
                4 => "Walk",
                6 => "JumpUp",
                7 => "JumpDown",
                8 => "Taunt",
                _ => "Appear",
            };
        } else if fields[0] == "projectile" {
            let v: Vec<f32> = fields[1..].iter().map(|s| s.parse().unwrap()).collect();
            world = World::new(Room::new_rectangular(15, 9), 42);
            player = world.spawn_player(Vec2::new(80.0, 160.0));
            if v.len() == 10 {
                world.entity_mut(player).unwrap().pos = Vec2::new(v[8], v[9]);
            }
            let mut p = isaac_sim::projectile::new(
                Vec2::new(v[0], v[1]),
                Vec2::new(v[2], v[3]),
                v[5],
                1.0,
                0,
            );
            p.projectile.as_mut().unwrap().height = v[4];
            p.projectile.as_mut().unwrap().falling_accel = v[6];
            p.dead = v[7] != 0.0;
            world.spawn(p);
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
            print!("{}{{\"id\":{},\"age\":{},\"pos\":[{},{}],\"vel\":[{},{}],\"height\":{},\"fall\":{},\"size\":{},\"scale\":{},\"dead\":{}}}",sep,t.id,t.time_cur,t.pos.x,t.pos.y,t.vel.x,t.vel.y,t.tear().height,t.tear().falling_speed,t.size,t.tear().scale,t.dead);
            sep = ",";
        }
        print!("],\"hp\":{},\"frame\":{},\"bosses\":[", p.hp, world.frame);
        sep = "";
        for b in world.entities.iter().filter(|e| e.etype == 20) {
            let n = b.npc();
            print!("{}{{\"id\":{},\"age\":{},\"pos\":[{},{}],\"vel\":[{},{}],\"anim\":\"{}\",\"frame\":{},\"collision\":{},\"hp\":{},\"airborne\":{},\"body_visible\":{},\"shadow\":[{},{}],\"flip\":{},\"debug_state\":{},\"debug_target\":[{},{}]}}",sep,b.id,b.time_cur,b.pos.x,b.pos.y,b.vel.x,b.vel.y,n.anim,n.anim_frame,b.entity_collision_class,b.hp,isaac_sim::npc::monstro::airborne(n.anim,n.anim_frame),isaac_sim::npc::monstro::body_visible(n.anim,n.anim_frame),b.pos.x,b.pos.y,n.flip_x,n.state,n.target_pos.x,n.target_pos.y);
            sep = ",";
        }
        print!("],\"projectiles\":[");
        sep = "";
        for b in world
            .entities
            .iter()
            .filter(|e| e.kind == EntityKind::Projectile)
        {
            let q = b.projectile.as_ref().unwrap();
            print!("{}{{\"id\":{},\"age\":{},\"pos\":[{},{}],\"vel\":[{},{}],\"height\":{},\"fall\":{},\"scale\":{},\"dead\":{},\"collision\":{}}}",sep,b.id,b.time_cur,b.pos.x,b.pos.y,b.vel.x,b.vel.y,q.height,q.falling_speed,q.scale,b.dead,b.entity_collision_class);
            sep = ",";
        }
        println!("]}}");
        io::stdout().flush().unwrap();
    }
}
