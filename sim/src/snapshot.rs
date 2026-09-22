//! Raw diagnostic snapshot. Python's actor adapter explicitly selects visible data.
use crate::{room::GridKind, EntityKind, World};
use serde_json::{json, Value};
pub fn state(w: &World, player: u32) -> Value {
    let p = w.entity(player).unwrap();
    let mut tears = vec![];
    let mut bosses = vec![];
    let mut projectiles = vec![];
    let mut bombs = vec![];
    for e in &w.entities {
        let mut v = json!({"id":e.id,"age":e.time_cur,"pos":[e.pos.x,e.pos.y],"vel":[e.vel.x,e.vel.y],"size":e.size,"dead":e.dead,"collision":e.entity_collision_class,"gcoll":e.grid_collision_class,"cdmg":e.collision_damage});
        match e.kind {
            EntityKind::Tear => {
                let q = e.tear();
                v["height"] = json!(q.height);
                v["fall"] = json!(q.falling_speed);
                v["scale"] = json!(q.scale);
                tears.push(v);
            }
            EntityKind::Projectile => {
                let q = e.projectile.as_ref().unwrap();
                v["height"] = json!(q.height);
                v["fall"] = json!(q.falling_speed);
                v["scale"] = json!(q.scale);
                projectiles.push(v);
            }
            EntityKind::Bomb => {
                v["anim"] = json!(if e.dead { "Explode" } else { "Pulse" });
                v["frame"] = json!(if e.dead { 0 } else { 14 + e.time_cur as i32 });
                bombs.push(v);
            }
            EntityKind::Npc if e.etype == 20 => {
                let n = e.npc();
                v["anim"] = json!(n.anim);
                v["frame"] = json!(n.anim_frame);
                v["hp"] = json!(e.hp);
                v["flip"] = json!(n.flip_x);
                v["airborne"] = json!(crate::npc::monstro::airborne(n.anim, n.anim_frame));
                v["body_visible"] = json!(crate::npc::monstro::body_visible(n.anim, n.anim_frame));
                v["shadow"] = json!([e.pos.x, e.pos.y]);
                v["debug_state"] = json!(n.state);
                v["debug_target"] = json!([n.target_pos.x, n.target_pos.y]);
                bosses.push(v);
            }
            _ => {}
        }
    }
    let cells: Vec<_> = w
        .room
        .cells
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let p = w.room.cell_center(i as i32);
            let inside = c.kind != GridKind::Wall;
            json!([
                i,
                p.x,
                p.y,
                c.collision_class,
                inside as u8,
                (inside && c.collision_class == 0) as u8,
                (c.kind == GridKind::Pit) as u8,
                0,
                (c.collision_class >= 2) as u8,
                (c.kind == GridKind::Rock) as u8
            ])
        })
        .collect();
    let positions = [[40., 280.], [320., 120.], [600., 280.], [320., 440.]];
    let clear = bosses.is_empty();
    let doors: Vec<_> = w
        .room
        .doors
        .iter()
        .enumerate()
        .filter(|(_, exists)| **exists)
        .map(|(slot, _)| json!({"slot":slot,"pos":positions[slot],"open":clear,"locked":false}))
        .collect();
    json!({"player":{"pos":[p.pos.x,p.pos.y],"vel":[p.vel.x,p.vel.y],"fire_delay":p.player().fire_delay,"bombs":p.player().bombs},"hp":p.hp,"frame":w.frame,"tears":tears,"bosses":bosses,"projectiles":projectiles,"bombs":bombs,"terrain":{"width":15,"height":9,"cells":cells},"doors":doors,"layout":w.room.layout})
}
