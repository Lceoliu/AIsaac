//! Normal Monstro blood shots: FireBossProjectiles RVA 0x002CCDD0,
//! projectile update 0x00253070, height/collision helper 0x0025B500.
use crate::{
    entity::{Entity, EntityKind, TYPE_PROJECTILE},
    math::Vec2,
    rng::Rng,
};

#[derive(Clone, Debug)]
pub struct ProjectileState {
    pub height: f32,
    pub falling_speed: f32,
    pub falling_accel: f32,
    pub scale: f32,
}

pub fn new(pos: Vec2, vel: Vec2, fall: f32, scale: f32, parent: u32) -> Entity {
    let mut e = Entity::base(0, EntityKind::Projectile, TYPE_PROJECTILE, 0, 0, pos);
    e.vel = vel;
    e.mass = 8.0;
    e.mass2 = 8.0;
    e.collision_damage = 1.0;
    e.grid_collision_class = 4;
    e.spawner = Some(parent);
    crate::physics::set_size(&mut e, 5.0, Vec2::new(1.0, 1.0), 8);
    e.projectile = Some(ProjectileState {
        height: -23.0,
        falling_speed: fall,
        falling_accel: 0.32,
        scale,
    });
    e
}

pub fn boss_volley(
    pos: Vec2,
    target: Vec2,
    count: usize,
    parent: u32,
    rng: &mut Rng,
) -> Vec<Entity> {
    (0..count)
        .map(|_| {
            // Native samples a disk about a 7-unit aimed velocity, not equally spaced angles.
            // Angle uses native 2*3.14; radius is uniform (not sqrt-uniform area).
            let theta = rng.next_f32() * 3.14 * 2.0;
            let radial = Vec2::new(theta.cos(), theta.sin());
            let vel = if target.is_zero() {
                radial * 7.0
            } else {
                (target - pos).normalized() * 7.0 + radial * (rng.next_f32() * 3.5)
            };
            let fall = 5.0 - rng.next_f32() * 30.0 * 0.8;
            let scale = (rng.next_u32() & 1) as f32 * 0.4 + 0.9 + rng.next_f32() * 0.05;
            new(pos, vel, fall, scale, parent)
        })
        .collect()
}

pub fn update(e: &mut Entity) {
    if e.dead {
        e.exists = false;
        return;
    }
    let p = e.projectile.as_ref().unwrap();
    if p.height > -5.0 || e.collides_with_grid && p.height > -60.0 {
        e.dead = true;
        crate::physics::update(e);
        return;
    }
    crate::physics::update(e);
    let p = e.projectile.as_mut().unwrap();
    p.falling_speed = 0.1 + 0.9 * p.falling_speed + p.falling_accel;
    p.height += p.falling_speed;
    e.entity_collision_class = if p.height < -50.0 { 0 } else { 4 };
    // Native floor probe: -5 -> -4.9 is already dead in this same observation.
    if p.height > -5.0 {
        e.dead = true;
    }
}
