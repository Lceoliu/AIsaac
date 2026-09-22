//! Ordinary player bomb only. J460 Init 0x002A2050, Update 0x002A24D0.
//! Native fixtures: 45 visible ticks to explosion, 100 enemy damage, 2 half-heart
//! self damage; collision class 0 Monstro is immune. No bomb-item modifiers.
use crate::{Entity, EntityKind, Vec2};

#[derive(Clone, Debug)]
pub struct BombState {
    pub countdown: i32,
    pub touch_grace: i32,
}

pub fn new(pos: Vec2, vel: Vec2, owner: u32) -> Entity {
    let mut b = Entity::base(0, EntityKind::Bomb, 4, 0, 0, pos);
    b.vel = vel;
    b.spawner = Some(owner);
    b.mass = 6.;
    b.mass2 = 6.;
    b.entity_collision_class = 4;
    crate::physics::set_size(&mut b, 16., Vec2::new(1., 1.), 12);
    b.bomb = Some(BombState {
        countdown: 45,
        touch_grace: 4,
    });
    b
}

pub fn update(b: &mut Entity) {
    if b.dead {
        b.exists = false;
        return;
    }
    crate::physics::update(b);
    // After integration: 0.95 damping and additional 0.5 below speed 1.
    b.vel = b.vel * 0.95;
    if b.vel.length_sq() < 1. {
        b.vel = b.vel * 0.5;
    }
    let q = b.bomb.as_mut().unwrap();
    q.touch_grace = (q.touch_grace - 1).max(0);
    if q.countdown == 0 {
        b.dead = true;
    } else {
        q.countdown -= 1;
    }
}
