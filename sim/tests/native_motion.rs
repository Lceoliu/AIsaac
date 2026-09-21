//! Numeric regression points from original J460, not fitted target constants.
use isaac_sim::{PlayerInput, Room, Vec2, World};

#[test]
fn native_first_logic_boundary_contains_two_player_updates() {
    let mut world = World::new(Room::new_rectangular(15, 9), 42);
    let id = world.spawn_player(Vec2::new(320.0, 280.0));
    world.set_player_input(id, PlayerInput::new(1.0, 0.0, 0.0, 0.0));
    world.step();
    let p = world.entity(id).unwrap();
    assert!((p.pos.x - 320.52819824219).abs() < 0.0001);
    assert!((p.vel.x - 0.9932045340538).abs() < 0.000001);
    assert_eq!(
        p.time_cur, 1.0,
        "one timer update, not two full game frames"
    );
}

#[test]
fn native_wall_reversal_still_resolves_existing_overlap() {
    let mut world = World::new(Room::new_rectangular(15, 9), 42);
    let id = world.spawn_player(Vec2::new(320.0, 280.0));
    world.set_player_input(id, PlayerInput::new(0.0, -1.0, 0.0, 0.0));
    for _ in 0..24 {
        world.step();
    }
    world.set_player_input(id, PlayerInput::new(0.0, 1.0, 0.0, 0.0));
    world.step();
    let p = world.entity(id).unwrap();
    assert!((p.pos.y - 150.08262634277).abs() < 0.0001);
    assert!((p.vel.y - 0.6006823182106).abs() < 0.000001);
}

#[test]
fn native_tear_range_is_converted_at_evaluate_items_tail() {
    let stats = isaac_sim::PlayerStats::isaac();
    assert_eq!(stats.tear_flags, 0);
    assert!((stats.tear_falling_speed - (-0.18337345123291)).abs() < 0.000001);
}

#[test]
fn native_transverse_tear_inherits_pre_movement_half_tick_velocity() {
    let mut world = World::new(Room::new_rectangular(15, 9), 42);
    let id = world.spawn_player(Vec2::new(320.0, 280.0));
    world.entity_mut(id).unwrap().player_mut().fire_delay = -1.0;
    world.set_player_input(id, PlayerInput::new(0.0, -1.0, 1.0, 0.0));
    world.step();
    let t = world.tears().next().unwrap();
    assert_eq!(t.vel.x, 10.0);
    assert!((t.vel.y - (-0.63384544849396)).abs() < 0.000001);
    // A naive 1.2 * final player velocity would be -1.191845, not this value.
}

#[test]
fn split_60_hz_viewer_ticks_equal_rl_step() {
    let mut a = World::new(Room::new_rectangular(15, 9), 42);
    let p = a.spawn_player(Vec2::new(320.0, 280.0));
    a.set_player_input(p, PlayerInput::new(-1.0, 1.0, 1.0, 0.0));
    let mut b = a.clone();
    for _ in 0..60 {
        a.step();
        b.interpolate_players();
        b.step_logic();
        assert_eq!(a.entity(p).unwrap().pos, b.entity(p).unwrap().pos);
        assert_eq!(a.entity(p).unwrap().vel, b.entity(p).unwrap().vel);
        assert_eq!(a.entities.len(), b.entities.len());
    }
}
