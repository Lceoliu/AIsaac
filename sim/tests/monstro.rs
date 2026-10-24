use isaac_sim::{npc::monstro, projectile, rng::Rng};
use isaac_sim::{Event, Room, Vec2, World};

fn world(seed: u32) -> (World, u32, u32) {
    let mut w = World::new(Room::new_rectangular(15, 9), seed);
    let p = w.spawn_player(Vec2::new(480., 340.));
    let b = w.spawn_monstro(Vec2::new(240., 240.));
    (w, p, b)
}
fn set_action(w: &mut World, b: u32, state: i32, frame: i32, anim: &'static str) {
    let n = w.entity_mut(b).unwrap().npc_mut();
    n.state = state;
    n.anim = anim;
    n.anim_frame = frame;
}

#[test]
fn high_jump_locks_grid_once_after_takeoff_and_shadow_follows_actual_position() {
    let (mut w, p, b) = world(42);
    set_action(&mut w, b, 6, 10, "JumpUp");
    w.step();
    assert_eq!(w.entity(b).unwrap().npc().target_pos, Vec2::ZERO);
    assert_eq!(w.entity(b).unwrap().entity_collision_class, 0);
    w.step();
    let target = w.entity(b).unwrap().npc().target_pos;
    assert_eq!(target, Vec2::new(480., 360.));
    w.entity_mut(p).unwrap().pos = Vec2::new(120., 200.);
    for _ in 0..20 {
        w.step();
        assert_eq!(w.entity(b).unwrap().npc().target_pos, target);
    }
    assert!(w.entity(b).unwrap().pos.x > 240.);
    assert!(!monstro::body_visible("JumpDown", 20));
    assert!(monstro::body_visible("JumpDown", 30));
}

#[test]
fn native_hop_and_fan_event_boundaries() {
    let (mut w, _, b) = world(3);
    set_action(&mut w, b, 4, 5, "Walk");
    w.step();
    assert_eq!(w.entity(b).unwrap().entity_collision_class, 4);
    w.step();
    assert_eq!(w.entity(b).unwrap().entity_collision_class, 0);
    set_action(&mut w, b, 4, 22, "Walk");
    w.step();
    assert_eq!(w.entity(b).unwrap().entity_collision_class, 4);
    set_action(&mut w, b, 8, 20, "Taunt");
    w.step();
    assert!(!w
        .events
        .iter()
        .any(|x| matches!(x, Event::BossVolley { .. })));
    w.step();
    assert!(w
        .events
        .iter()
        .any(|x| matches!(x, Event::BossVolley { count: 13, .. })));
    w.step();
    assert!(!w
        .events
        .iter()
        .any(|x| matches!(x, Event::BossVolley { .. })));
}

#[test]
fn projectile_floor_death_is_visible_on_crossing_frame_then_removed() {
    let (mut w, _, _) = world(4);
    let mut q = projectile::new(Vec2::new(200., 280.), Vec2::ZERO, 0., 1., 0);
    q.projectile.as_mut().unwrap().height = -5.;
    q.projectile.as_mut().unwrap().falling_accel = 0.;
    let id = w.spawn(q);
    w.step();
    let q = w.entity(id).unwrap();
    assert!(q.dead);
    assert!((q.projectile.as_ref().unwrap().height + 4.9).abs() < 1e-6);
    w.step();
    assert!(w.entity(id).is_none());
}

#[test]
fn native_last_animation_frame_is_held_before_state_returns_to_idle() {
    let (mut w, _, b) = world(42);
    set_action(&mut w, b, 7, 62, "JumpDown");
    w.step();
    assert_eq!(w.entity(b).unwrap().npc().anim_frame, 63);
    assert!(!w.entity(b).unwrap().npc().anim_finished);
    w.step();
    assert_eq!(w.entity(b).unwrap().npc().state, 7);
    assert!(w.entity(b).unwrap().npc().anim_finished);
    w.step();
    assert_eq!(w.entity(b).unwrap().npc().state, 3);
}

#[test]
fn airborne_monstro_ignores_tears_but_landed_monstro_takes_damage() {
    for (frame, damage) in [(10, false), (24, true)] {
        let (mut w, p, b) = world(5);
        set_action(&mut w, b, 4, frame, "Walk");
        let pos = w.entity(b).unwrap().pos;
        let mut t =
            isaac_sim::tear::fire_tear(w.entity_mut(p).unwrap(), pos, Vec2::ZERO, &mut Rng::new(2));
        t.pos = pos;
        t.vel = Vec2::ZERO;
        w.spawn(t);
        w.step();
        assert_eq!(w.entity(b).unwrap().pending_damage > 0., damage);
    }
}

#[test]
fn overhead_bullets_do_not_damage_player_and_low_bullets_do() {
    for (height, hp) in [(-70., 6.), (-30., 5.)] {
        let (mut w, p, _) = world(4);
        let mut q = projectile::new(w.entity(p).unwrap().pos, Vec2::ZERO, 0., 1., 0);
        q.projectile.as_mut().unwrap().height = height;
        w.spawn(q);
        w.step();
        assert_eq!(w.entity(p).unwrap().hp, hp);
    }
}

#[test]
fn native_boss_fan_velocity_and_ballistic_parameters() {
    let mut rng = Rng::new(21);
    let shots = projectile::boss_volley(
        Vec2::new(200., 200.),
        Vec2::new(400., 200.),
        1000,
        1,
        &mut rng,
    );
    for p in shots {
        assert!((p.vel - Vec2::new(7., 0.)).length() <= 3.50001);
        let q = p.projectile.unwrap();
        assert_eq!(q.height, -23.);
        assert_eq!(q.falling_accel, 0.32);
        assert!((-19. ..=5.).contains(&q.falling_speed));
        assert!((0.9..1.35).contains(&q.scale));
    }
}

fn run(seed: u32) -> String {
    let (mut w, p, b) = world(seed);
    let mut seen = std::collections::BTreeSet::new();
    let mut result = String::new();
    w.entity_mut(p).unwrap().player_mut().damage_cooldown = 30000;
    for _ in 0..2000 {
        w.step();
        let e = w.entity(b).unwrap();
        seen.insert(e.npc().state);
        result.push_str(&format!(
            "{:?}:{:?}:{}:{};",
            e.pos,
            e.vel,
            e.npc().state,
            w.entities.len()
        ));
    }
    assert!(seen.contains(&4) && seen.contains(&6) && seen.contains(&7) && seen.contains(&8));
    result
}
#[test]
fn seeds_replay_full_room_and_change_actions() {
    assert_eq!(run(42), run(42));
    assert_ne!(run(42), run(43));
}

#[test]
fn type20_dispatch_no_longer_stays_as_inert_npc() {
    let (mut w, _, b) = world(42);
    let start = w.entity(b).unwrap().pos;
    for _ in 0..120 {
        w.step();
    }
    assert!((w.entity(b).unwrap().pos - start).length() > 20.);
}
