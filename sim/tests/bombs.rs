use isaac_sim::{bomb, EntityKind, Event, PlayerInput, Room, Vec2, World};

#[test]
fn trigger_inventory_fuse_and_single_hurt_event() {
    let mut w = World::new(Room::new_rectangular(15, 9), 42);
    let p = w.spawn_player(Vec2::new(320., 280.));
    let mut input = PlayerInput::default();
    input.bomb = true;
    w.set_player_input(p, input);
    let mut explosions = 0;
    let mut hurts = 0;
    for frame in 0..46 {
        w.step();
        if frame < 44 {
            assert_eq!(w.entity(p).unwrap().hp, 6.);
        }
        for e in &w.events {
            match e {
                Event::BombExploded { .. } => {
                    assert_eq!(frame, 44);
                    explosions += 1;
                }
                Event::PlayerDamaged { damage, .. } => {
                    assert_eq!(*damage, 2.);
                    hurts += 1;
                }
                _ => {}
            }
        }
    }
    assert_eq!(explosions, 1);
    assert_eq!(hurts, 1);
    assert_eq!(w.entity(p).unwrap().hp, 4.);
    assert_eq!(w.entity(p).unwrap().player().bombs, 0);
    assert!(!w.entities.iter().any(|e| e.kind == EntityKind::Bomb));
}
#[test]
fn repress_requires_thirty_frame_cooldown() {
    let mut w = World::new(Room::new_rectangular(15, 9), 42);
    let p = w.spawn_player(Vec2::new(320., 280.));
    w.entity_mut(p).unwrap().player_mut().bombs = 5;
    for tick in 0..40 {
        let mut a = PlayerInput::default();
        a.bomb = tick % 2 == 0;
        w.set_player_input(p, a);
        w.step();
        assert_eq!(
            w.entity(p).unwrap().player().bombs,
            if tick < 30 { 4 } else { 3 }
        );
    }
}
#[test]
fn blast_queued_damage_airborne_immunity_and_rocks() {
    for airborne in [false, true] {
        let mut w = World::new(Room::new_rectangular(15, 9), 42);
        w.spawn_player(Vec2::new(100., 380.));
        let b = w.spawn_monstro(Vec2::new(400., 280.));
        let n = w.entity_mut(b).unwrap();
        n.npc_mut().state = if airborne { 7 } else { 8 };
        n.npc_mut().anim = if airborne { "JumpDown" } else { "Taunt" };
        n.npc_mut().anim_frame = if airborne { 10 } else { 0 };
        w.room.add_rock(67);
        let mut q = bomb::new(Vec2::new(320., 280.), Vec2::ZERO, 1);
        q.bomb.as_mut().unwrap().countdown = 0;
        w.spawn(q);
        w.step();
        assert_eq!(w.entity(b).unwrap().hp, 250.);
        w.step();
        assert_eq!(w.entity(b).unwrap().hp, if airborne { 250. } else { 150. });
        assert_eq!(w.room.cells[67].collision_class, 0);
    }
}
#[test]
fn native_strict_blast_boundary() {
    for (distance, hp) in [(84., 4.), (85., 6.)] {
        let mut w = World::new(Room::new_rectangular(15, 9), 42);
        let p = w.spawn_player(Vec2::new(320. + distance, 280.));
        let mut b = bomb::new(Vec2::new(320., 280.), Vec2::ZERO, p);
        b.bomb.as_mut().unwrap().countdown = 0;
        w.spawn(b);
        w.step();
        assert_eq!(w.entity(p).unwrap().hp, hp);
    }
}

#[test]
fn settled_bomb_reward_uses_actual_damage_and_clear_once() {
    let mut s = isaac_sim::ffi::Slot::new(42);
    s.world = World::new(Room::new_rectangular(15, 9), 42);
    s.player = s.world.spawn_player(Vec2::new(100., 380.));
    let boss = s.world.spawn_monstro(Vec2::new(400., 280.));
    let n = s.world.entity_mut(boss).unwrap();
    n.hp = 60.;
    n.npc_mut().state = 8;
    n.npc_mut().anim = "Taunt";
    let mut b = bomb::new(Vec2::new(320., 280.), Vec2::ZERO, s.player);
    b.bomb.as_mut().unwrap().countdown = 0;
    s.world.spawn(b);
    s.step(&[0, 0, 0]);
    assert!(s.done);
    assert_eq!(s.outcome, "win");
    assert!((s.reward - 1.29).abs() < 1e-6);
}
