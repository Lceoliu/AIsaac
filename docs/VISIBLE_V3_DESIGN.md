# visible-v3: a Repentance-compatible observation and action interface

Design only. Date 2026-11-12. No code was changed for this document.

## Notation and sources

| Short | Path or URL |
|---|---|
| `WT/` | `<项目根>\temp\wt-block-memory-v1\bridge\` (block-memory v1 tree, schema `isaac-visible-prlmens-g-v2`, VERSION 2026110702) |
| `SI` | `WT/python/isaac_bridge/structured_input.py` |
| `TP` | `WT/python/isaac_bridge/tok_policy.py` |
| `LUA` | `WT/abplus/abp_bridge.lua` |
| `VIS` | `WT/engine/abp_visible.c` |
| `TURBO` | `WT/engine/abp_turbo.c` (AB+ native lean producer, native step-line parser, native input) |
| `ACT` | `WT/engine/abp_actor.cpp` |
| `E_AB` | `<项目根>\experiments\vpt-idm-v1\repentance\local-resources\scripts\enums.lua`. It is an AB+-era enum file: it has no ActiveSlot, and NUM_COLLECTIBLES is 553. That it is byte-identical to AB+ 1.06 Linux is UNVERIFIED. |
| `E_REP` | `...\local-resources\scripts_rep\enums.lua`: the Repentance enums. The spec cites the same line numbers (`PROJECT_SPEC.block-memory-v1.md:163`). |
| `REC` | `<项目根>\isaac-run-recorder\mod\isaac_run_recorder\` (J746 recorder: `README.md`, `state.lua`) |
| `REPA` | `<项目根>\analysis\resources\repentance-a\config\`: resource extract. Believed to be Rep+ J460 per memory note `isaac-abplus-vs-repplus-rules.md`; not re-verified here. |
| `DOC:EP` | https://wofsauge.github.io/IsaacDocs/rep/EntityPlayer.html |
| `DOC:EPAB` | https://wofsauge.github.io/IsaacDocs/abp/EntityPlayer.html |
| `BP:R+` | https://isaacblueprints.com/tutorials/general/repentance_plus_changes/ |

UNVERIFIED means that neither the code nor a document confirms the item. Each one needs a sample on the target build.

---

## 1. Goals and the compatibility contract

**Goal.** The AB+ training run produces a checkpoint that later loads and runs on Rep (J746 = v1.7.8a) and on Rep+, with the same model code, the same tensor shapes and the same schema version. Training on Rep or Rep+ then needs only two new pieces:

- a Rep-side producer of the v3 frame (plus the ROW reward/aux producer, which is needed for training only);
- an engine.

Human recordings from `REC` must also be convertible into v3.

### The contract guarantees

| # | Guarantee |
|---|---|
| G1 | One frame layout, identical on every producer: header, fixed canvases, ragged P/L/E/N/S lists, column order, raw units, bit layouts. Its version is `VERSION = 2026111201` and its schema string is `isaac-visible-prlmens-g-v3`. No shape depends on the game version. |
| G2 | Every Rep/Rep+ id range fits its table with head-room (section 2.8). An out-of-range or negative id maps to a dedicated per-table *unknown* row and never crashes. |
| G3 | A field the version lacks is written as its true value in that version. For example AB+ has no broken hearts, so the column is 0. A field that exists in the game but that the producer cannot read is 0 **with a known flag of 0**. These two cases never share an encoding. |
| G4 | The action space covers all 12 player ButtonActions, LEFT..DROP (`E_AB:2627`, `E_REP:4023`). The heads are mapped to ButtonAction **by name**, because Rep+ renumbered ButtonAction (`BP:R+`). |
| G5 | Only visible information reaches the actor. Privileged data stays in the learner-only critic. |
| G6 | AB+ trajectories for the same seed and the same actions (drop = 0) are unchanged, because producers call only read-only getters. AB+ stays exactly reproducible. |

### The contract does not guarantee

| # | Not covered |
|---|---|
| N1 | Useful values for ids never seen on AB+: Rep-only items, characters, enemies, effects, room types and the `game_version` code. Their embedding rows stay at initialisation until Rep training. |
| N2 | Equal rules. Bomb damage is 60 on AB+ and 100 on Rep (memory note `isaac-abplus-vs-repplus-rules.md`). Boss pools, item effects and numeric distributions also differ (coins up to 999, more hearts, and so on). |
| N3 | Input timing. Rep polls input at 60 Hz with intermediate ticks whose `triggered` drives logic (`REC/README.md:142-150`). The AB+ bridge sets `triggered` for the first logic frame of a 4-frame step (`LUA:914-929`, `LUA:3000-3002`). The behaviour of a press edge can differ; this is documented in section 3.4, not solved. |
| N4 | Analog stick, mouse aiming (`MouseControl`, `REC/README.md:150`), co-op with several controllers, menus and REPENTOGON-only getters. |
| N5 | A Rep ROW producer. Rewards, aux targets and the privileged critic read `tok_obs.ROW` (`WT/python/isaac_bridge/tok_obs.py:90-167`), which only the AB+ native code fills. Inference on Rep needs only the v3 frame; training on Rep needs a ROW producer as well. |
| N6 | Lossless conversion of human recordings. Analog values, 60 Hz press timing and simultaneous opposite or diagonal shoot keys are quantised to the 4-frame step and to the 5-way shoot head. Several fields are not recorded (section 2.9). |

---

## 2. visible-v3 field-by-field

### 2.0 Global rules

- **Wire values are raw game units** as float32. The model applies a frozen scale vector (`schema_v3` constants). This is a change from v2, where the producer scaled; see `SI:472-480` and `VIS:47-48`. The recorder converter and the Rep producer then need no scale tables. Positions stay relative to the room's top-left, as in v2 (`SI:477`, `VIS:275-276`).
- **Integers are exact** up to 2^24 in float32. Every bit word is at most 24 bits; flag words use 16-bit chunks.
- **Categorical codes** are integers; code 0 means "none / not known". They go through embeddings:
  - raw entity ids use the shared `identity` table (`SI:571`);
  - item ids use a new namespaced `item` table;
  - small enums use a new namespaced `cat` table (section 2.8).
- **Reserved columns** are written as 0. Code 0 is the global padding row, and bits and numbers are 0, so a reserved column has no effect until it is used. Using one later changes no shape.

### 2.1 Frame and header

| Item | v2 (`SI:12-27,58-69`) | v3 | Shape change |
|---|---|---|---|
| header[0..5] | magic 61107, size, p, e, n, s | same | – |
| header[6] | 1 | schema version word: `VERSION − 2026000000` (batch 5, 2026-11-18; e.g. 111802). Exact in float32 through 2042. Every reader checks it frame by frame. Batches 1–4 wrote 3 (layout generation, retired: any layout or column-meaning change now raises VERSION). | – |
| header[7] | 0 | producer id: 0 AB+, 1 Rep J746, 2 Rep+, 3 recorder-converted. Diagnostics only; never decoded into model inputs. | – |
| room canvas | 16×28 cells × 4 floats = 1792 | 16×28 × **5** floats = 2240 | yes |
| minimap | 169 floats (8 one-bit channels) | 169 floats (new bit layout, 2.4) | decode only |
| FIXED | 8+1792+169 = 1969 | 8+2240+169 = **2417** | yes |
| per body | P 80 + local 81×4 = 404 | P **192** + local 81×5 = **597** | yes |
| per entity / item / slot | 48 / 4 / 13 | **64 / 8 / 24** | yes |
| size check | `FIXED+p*(PF+324)+e*EF+n*NF+s*SF` | `2417+p*597+e*64+n*8+s*24` | – |
| worst case | – | p 4, e 128, n 64, s 32: 14,240 floats = 57 KB, below FRAME_BYTES 131,072 | – |

The local patch stays 9×9 per body, cut from the same 5-word cells.

### 2.2 R: room canvas, 5 words per cell

**Word 0** keeps the 13 v2 channels of `SI:25-26` with one widened. 20 bits:

| Bits (LSB first) | Field | v2 | v3 |
|---|---|---|---|
| 1 | inside | same | same |
| 3 | collision class | stored as /6 | raw 0..7 |
| 1 each | solid, pit, destructible, hazard, door, door_open, door_locked | same | same |
| **6** | door target RoomType | 5 bits, stored as /30 | raw code 0..63 (Rep 0..29 `E_REP:1587`; Rep+ adds ROOM_DEATHMATCH, value UNVERIFIED, `BP:R+`) |
| 1 each | seen (visited), clear, exit (grid type 17/18) | same | same |

**Word 1 (new).** 23 bits, decoded into 5 channels:

| Bits | Field | AB+ source | Rep source |
|---|---|---|---|
| 6 | door variant | `GridEntityDoor` variant (UNVERIFIED getter on AB+; 0 + known bit) | recorder door row `variant` (`REC/README.md:114`) |
| 4 | door state | door `State` (UNVERIFIED on AB+) | recorder door row `state` |
| 1 | red room | 0 (absent in AB+) | RoomDescriptor flag (UNVERIFIED) |
| 6 + 6 | reserved codes a, b | 0 | 0 |

**Words 2–4.** Grid type, grid variant and grid state, raw. v2 stored them divided by 32, 256 and 256 (`SI:468`). The type range is AB+ 0..22 (`E_AB:334`) and Rep 0..27 (`E_REP:459`); it is decoded as /64.

**Decoded room channels.** RC goes from 16 to **21**: 13 + 5 + 3. This changes the room conv weights `SI:565`, the local conv weights `SI:568` and the canvas decoder `SI:278-283`.

Code channels (door type, door variant, grid type) stay scalar-coded, code/64, as in v2. The reason is that per-cell embeddings would roughly double the room activations in the learner; see open question Q6.

### 2.3 Doors

Doors stay inside the room canvas, as decided in memory note `isaac-token-design-decisions-20261024.md` item 1. On Rep the door fields come from `room:GetDoor(slot)`: `TargetRoomType`, `IsOpen`, `IsLocked`, `TargetRoomIndex` → visited/clear. The calls are the same as in `LUA:1822-1840`; `REC` records the same fields in its door rows.

### 2.4 M: minimap, one float per cell, 16 bits

| Bits | Field | v2 (`SI:489-497`) | AB+ source | Rep source |
|---|---|---|---|---|
| 1 | shown | ch 0 | lean map block (`LUA:1439-1442`) | RoomDescriptor DisplayFlags (`REC/README.md:108`) |
| 1 | visited | ch 1 | same | same |
| 1 | clear | ch 2 | same | same |
| 1 | current room | ch 3 | same | same |
| **6** | icon RoomType: the raw type when the icon is shown or the room was visited, else 0 | v2 had only boss/treasure/shop/"other" one-hots (ch 4–7) | lean map `type` | descriptor `type` |
| 6 | reserved | – | 0 | 0 |

**Model.** The 4 binary channels plus `Embedding(64,8)` of the icon code give 12 channels into the minimap conv (v2 had 8, `SI:570`). This removes the v2 loss of every special-room type outside boss/treasure/shop, for example arcade, curse, library, planetarium and ultra secret.

### 2.5 P: one token per body, PF = 192

Wire values are raw. "AB+ src" is the current producer (`LUA:1805-1820,1947-1994`, `TURBO:2074-2098`). "Rep src" is a Rep 1.7.8a getter; `REC:c` means the recorder column name in `REC/state.lua:85-151`.

| Col | Name | AB+ src | Rep src | Notes / v2 col |
|---|---|---|---|---|
| 0–4 | x, y, vx, vy, size | as v2 | Position, Velocity, Size (`REC:x..vy,size`) | v2 0–4 |
| 5 | hearts (red halves) | GetHearts | same (`REC:hearts`) | masked under Curse of the Unknown |
| 6 | max_hearts | GetMaxHearts | same | |
| 7 | soul halves | GetSoulHearts | same | |
| 8 | bone | GetBoneHearts | same | |
| 9 | eternal | GetEternalHearts | same | |
| 10 | golden | GetGoldenHearts | same | |
| 11 | **broken** | 0 (absent) | GetBrokenHearts (`REC:broken`; DOC:EP: not doubled) | new |
| 12 | **rotten** | 0 (absent) | GetRottenHearts (`REC:rotten`) | new |
| 13 | extra_lives | GetExtraLives | same | |
| 14–16 | coins, bombs, keys | as v2 | same | |
| 17 | **giga_bombs** | 0 | GetNumGigaBombs (`REC:giga_bombs`) | new |
| 18 | **golden_bomb** | 0 | HasGoldenBomb (`REC:golden_bomb`) | new |
| 19 | **golden_key** | 0 | HasGoldenKey (`REC:golden_key`) | new |
| 20 | **poop_mana** | 0 | GetPoopMana (`REC:poop_mana`) | new |
| 21 | **soul_charge** | 0 | GetEffectiveSoulCharge (DOC:EP: 0 unless Bethany = what the HUD shows). Not in REC. | new |
| 22 | **blood_charge** | 0 | GetEffectiveBloodCharge (DOC:EP; Tainted Bethany). Not in REC. | new |
| 23–28 | damage, max_fire_delay, shot_speed, range, move_speed, luck | as v2 (`TURBO:2081` range = 260·h/−23.75) | Damage, MaxFireDelay, ShotSpeed, **TearRange**, MoveSpeed, Luck | unit equality of range between AB+ and Rep is UNVERIFIED |
| 29–32 | can_fly, invulnerable (damage cooldown > 0), controls, dead | as v2 | same | |
| 33–35 | head_dir, fire_dir, move_dir | as v2 | same | |
| 36 | sprite frame | as v2 | `REC:aframe` | |
| 37 | weapon_charge (raw counter) | `abp_player_charge` (`LUA:1792-1795`) | no vanilla getter: 0, known flag 0 | gated by Q4 |
| 38 | body_index | i | i | v2 col 35 |
| 39 | linked_body | 0 | index+1 of GetMainTwin / GetOtherTwin / GetSubPlayer partner (DOC:EP) | new |
| 40–47 | reserved numeric | 0 | 0 | 8 spare |
| 48 | health_known | not Curse of the Unknown | same (curse bit 3, `E_REP:1653`) | v2 col 40 |
| 49 | controller_known | ControllerIndex present | same | v2 col 37 |
| 50 | animation_known | as v2 | same | v2 col 39 |
| 51 | weapon_charge_known | 1 when abp_turbo provides it | 0 | new |
| 52 | bone_slots_known | 1 if IsBoneHeart is readable (DOC:EPAB lists it; semantics UNVERIFIED) | 1 (DOC:EP lists IsBoneHeart) | new |
| 53–55 | reserved known flags | 0 | 0 | |
| 56 | **ptype** (cat) | GetPlayerType | same (`REC:ptype`; 0..40 and −1 POSSESSOR, `E_REP:1666`) | v2 col 34 was ptype/32 |
| 57 | **body_role** (cat) | 1 main, 3 sub-player | 1 main, 2 twin (Esau, T.Lazarus partner), 3 sub-player (Forgotten/Soul), 4 coop baby, 5 coop ghost (IsCoopGhost) | new; detection of 4 is UNVERIFIED |
| 58 | controller (cat) | ControllerIndex+2 | same | v2 col 36 |
| 59 | animation (cat) | 14-name list index (`LUA:1952-1959`) | same list | v2 col 38 |
| 60 | **stage** (cat) | level:GetStage() (AB+ 1..12, `E_AB:1072`) | same (1..13, `E_REP:1550`) | v2 col 42 was stage/16 |
| 61 | **stage_type** (cat) | level:GetStageType() (0..3, `E_AB:1095`; getter UNVERIFIED on AB+) | same (0..5, `E_REP:1578`) | **new**: alt floors become distinguishable |
| 62 | **room_type** (cat) | room:GetType() | same | new to P (v2 had it only in the old ROW) |
| 63 | difficulty (cat) | Game().Difficulty+1 (UNVERIFIED AB+ property) | same (`REC` header `meta.difficulty`) | new; shown on the HUD |
| 64 | game_version (cat) | 1 | 2 Rep, 3 Rep+ | new; see Q3 |
| 65 | control_scheme (cat) | 0 | Rep 0; Rep+ `Options.JacobEsauControls`+1 (`BP:R+`) | new: it changes what Drop does |
| 66–71 | reserved categorical | 0 | 0 | 6 spare |
| 72–103 | curse bits 0..31 | GetCurses | same (Rep adds bit 7 Giant, `E_REP:1653`) | v2 43–74 |
| 104–127 | black-heart slot bits 0..23 | bit i of GetBlackHearts() (a bitmask, DOC:EP/DOC:EPAB) | same (`REC:black`) | v2 col 8 was the mask/65535 |
| 128–151 | bone-heart slot bits 0..23 | IsBoneHeart(i) | IsBoneHeart(i) (not in REC) | new; heart order |
| 152–175 | weapon-type bits 0..23 | HasWeaponType(w), w 0..23 (AB+ uses 1..10, `E_AB:931`) | same (1..15, `E_REP:1396`) | v2 col 33 was mask/2048 over 0..10 only |
| 176–180 | entity overflow per class | as v2 | same | v2 75–79, raw count |
| 181–191 | reserved | 0 | 0 | 11 spare |

**Dropped from v2.** The always-zero `unknown_fire_delay`, `unknown_damage_cooldown` and `unknown_charge` (`SI:18,476,610`) are hidden counters and do not come back. The single `capabilities` column (`SI:482`) is replaced by the known flags and `game_version`.

**Bodies.** v3 lists more bodies than v2:

- every `Isaac.GetPlayer(i)` for i < GetNumPlayers(), plus `GetSubPlayer()` and `GetOtherTwin()` of each, de-duplicated by entity Index, with the main body first;
- on AB+ this makes the inactive Forgotten/Soul body visible. Today it is invisible: `LUA:1730` skips entity type 1, and bodies come only from GetNumPlayers (`LUA:1954`).

Rep cases:

| Character | Bodies |
|---|---|
| Jacob & Esau | two players already (PlayerType 19/20) |
| Tainted Forgotten / Soul | 35/40; whether the second is a player or a twin is UNVERIFIED |
| Tainted Lazarus | the partner exists only with Birthright (DOC:EP GetOtherTwin) |
| Co-op babies | appear as players; marking them as role 4 is UNVERIFIED |

Each body keeps its own inventory and slots, owned by body_index. Whether the Forgotten and the Soul share items is UNVERIFIED.

### 2.6 E: one token per entity, EF = 64

| Col | Name | AB+ src (`TURBO:1939-2064`, `LUA:1711-1787`) | Rep src | Notes |
|---|---|---|---|---|
| 0, 1, 2 | type, variant, subtype (raw ids → `identity`, guarded) | as v2 | `REC` entity row `type`, `variant`, `subtype` | subtype 0 under Blind for pedestals (v2 `SI:506`) |
| 3–14 | x, y, vx, vy, size, size_mx, size_my, coll, gcoll, cdmg, sprite frame, age | as v2 | `REC` + Entity fields | raw |
| 15 | kind (cat) | 0 other, 1 tear, 2 projectile, 3 laser, 4 bomb, 5 pickup, 6 npc, **7 knife, 8 familiar, 9 effect** | same | v2 had 0..6 |
| 16 | npc flags (decoded as 8 bits) | IsEnemy 1, IsVulnerableEnemy 2, IsBoss 4, CanShutDoors 8 | same | v2 numeric/64 |
| 17 | anim (short list) | as v2 | same | |
| 18 | flipX | as v2 | same | |
| 19 | **champion** (cat, code+2) | GetChampionColorIdx | same (`REC` npc extra `champion`; 0..25, `E_REP:4432`) | v2 numeric |
| 20 | boss hp fraction | as v2 | same | visible boss bar |
| 21–23 | height, falling speed, scale | as v2 | `REC` tear/projectile extras | |
| 24–30 | laser geometry | as v2 | `REC` laser extras plus angle | |
| 31 | price | as v2 | `REC` pickup `price` | |
| 32 | identity_known | as v2 | same | v2 col 47 |
| 33 | flags_known bits: 1 tear-ns, 2 projectile, 4 entity flags, 8 anim_id | AB+ native: 1\|2 (4 after package C reads the flags word) | 1\|2\|4 | new |
| 34–41 | **tear-flag words** w0..w7: 128 bits, canonical = Rep `TEARFLAG(n)` numbering (`E_REP:3916-4021`), bit k in word k//16 | tears, lasers (`LUA:1747`); knives and bombs: flags word (UNVERIFIED AB+ getters). AB+ word is remapped (table below). | TearFlags.l / .h (BitSet128; `REC` tear extra `tear_flags_l/h`) | v2: 15 bits only (`SI:513`, `VIS:52`) |
| 42–45 | **projectile-flag words**, 64 bits | ProjectileFlags (`TURBO:1993`) | same; not in REC | identical numbering in AB+ (bits 0..33, `E_AB:2932`) and Rep (0..57, `E_REP:4360`) |
| 46–49 | **entity-flag words**, 64 bits, ANDed with the visible-status mask below | GetEntityFlags (Lua; native needs the AB+ field offset) | GetEntityFlags; not in REC | AB+ 0..37 (`E_AB:664`) ⊂ Rep 0..59 (`E_REP:948`), same numbering |
| 50 | anim_id (hashed anm2 file + animation name → `anim` table) | 0, known 0 (no AB+ getter found; UNVERIFIED) | `GetSprite():GetAnimation()` + file (`REC` `anim`, appearance `file`) | spec 9.7 |
| 51–63 | reserved | 0 | 0 | 13 spare |

**Tear flags: AB+ bit → canonical bit.** Derived from `E_AB:2561-2625` versus `E_REP:3920-4021` by name (MIGAN = MULLIGAN, BOMBERANG = BOOMERANG):

- bits 0–29 map to themselves;
- 30→35, 31→30, 32→31, 33→32, 34→36, 35→37, 36→38, 37→39, 38→40, 39→41, 40→42, 41→43, 42→44, 43→33, 44→46, 45→47, 46→48, 47→49, 48→50, 49→51, 50→52, 51→53, 52→54, 53→55, 54→56, 55→57, 56→58, 57→59, 58→60, 59→61, 60→127 (LUDOVICO).

The 15 bits v2 uses (0..22) are all inside the identity range. Note that v2 applies the same 15 positions to ProjectileFlags words (`VIS:338-339`); v3 separates the two namespaces.

**Visible-status mask** for entity flags. It is a constant in package C; changing it changes no shape:

- FREEZE 5, POISON 6, SLOW 7, CHARM 8, CONFUSION 9, MIDAS_FREEZE 10, FEAR 11, BURN 12;
- SHRINK 24, FRIENDLY 29, BLEED_OUT 34;
- Rep only: ICE_FROZEN 49, ICE 50, MAGNETIZED 51, BAITED 52, WEAKNESS 54, BRIMSTONE_MARKED 56, HELD 57, THROWN 58.

All bit numbers come from `E_REP:948-1015`.

**Effect whitelist.** v2 sends only effect variants {1, 22, 23, 26, 30, 45, 46, 51, 53, 55}; the list is duplicated in `LUA:192-193`, `LUA:1730` and `TURBO:1966-1971`. Rep effect variants run to 199 (`E_REP:492-696`), and Rep+ adds more (`BP:R+`).

- **v3 rule:** send every visible effect except a constant *decorative* list. The recorder's `fx_omit` list is a starting candidate (`REC/README.md:84,206`).
- Effects fall into priority class 4 ("other", `SI:528-536`), so the 128 cap and the overflow counts still hold.
- This changes no shape; see Q2.

### 2.7 N (inventory, NF = 8) and S (slots, SF = 24)

**N: inventory**

| Col | Field | Source |
|---|---|---|
| 0 | owner body_index | as v2 (`LUA:1965-1968`) |
| 1 | collectible id (`item` table) | AB+ ≤ 552 (`E_AB:1693-2261`); Rep ≤ 732 (`E_REP:2729-3481`) |
| 2 | count | GetCollectibleNum |
| 3 | id_known | v2 col 3 |
| 4 | display order | – |
| 5 | display_order_known | – |
| 6–7 | reserved | 0 |

**S: slots**

| Col | Field | AB+ src | Rep src |
|---|---|---|---|
| 0 | owner | as v2 | as v2 |
| 1 | family (cat): 1 active, 2 trinket, 3 pill, 4 card, **5 poop spell, 6 crafting-bag item, 7 crafting output**, 8–15 reserved | 1–4 (`LUA:1972-1990`) | 1–5. 6–7 have no vanilla getter: rows not emitted. Bag of Crafting is not in DOC:EP; it is REPENTOGON-only (UNVERIFIED). |
| 2 | slot | active: 0 primary, 1 Schoolbag (`p.SecondaryActiveItem`, `LUA:1973`); pocket index 0..GetMaxPoketItems−1 | active: ActiveSlot 0..3 (`E_REP:4464`; 2 = pocket active, 3 = one-shot pocket). Pocket index 0..3 (DOC:EP GetCard/GetPill slots 0..3). Poop queue position (GetPoopSpell(i)). |
| 3 | id (`item` table, family namespace): collectible; trinket & 0x7fff; card; pill effect+1 if identified; poop spell | as v2 | GetActiveItem(slot), GetTrinket, GetCard, GetPill + pool, GetPoopSpell |
| 4–5 | occupied, selected (what the button uses now) | as v2 | same |
| 6–7 | charge, max_charge | as v2 | GetActiveCharge(slot), ItemConfig MaxCharges |
| 8–10 | id_known, selected_known, charge_known | as v2 | same |
| 11 | pill colour & 0x7ff (`item` table) | as v2 | GetPill (`E_REP:1735`: GOLD 14, GIANT flag 0x800) |
| 12 | battery charge | GetBatteryCharge() | GetBatteryCharge(slot) (DOC:EP) |
| 13 | golden (trinket 0x8000 / PILL_GOLD) | 0 | flag bits (`E_REP:3483-3693`, `:1735`) |
| 14 | giant pill (0x800) | 0 | flag bit |
| 15–16 | display order, known | 0 / 0 | pocket order. GetPocketItem is broken (DOC:EP), so the reading method is UNVERIFIED and known is 0 until proven. |
| 17 | charge_type (cat): normal / timed / special | 0 | ItemConfig ChargeType (UNVERIFIED property) |
| 18–19 | sub-charge, known | 0 / 0 | GetActiveSubCharge "always returns 0" (DOC:EP), so known = 0 |
| 20–23 | reserved | 0 | 0 |

**Emission rule.** Every slot the HUD shows is emitted, including empty ones (occupied = 0). A slot that does not exist is not emitted. Rep needs different calls from the current Lua:

- use `GetMaxPocketItems`; AB+ spells it `GetMaxPoketItems`;
- `SecondaryActiveItem` no longer exists in Rep (DOC:EP), so use the slot-indexed getters.

### 2.8 Embedding tables, namespaces and id guards

| Table | Shape | Index rule | Range check (AB+ / Rep / Rep+) |
|---|---|---|---|
| `identity` (existing, `SI:571`) | 65536×24 | raw id; `<0` or `≥65535` → **65535** (unknown row); 0 = padding | entity type ≤ 9001 (ENTITY_TEXT; 1000 effects; `E_REP:94-457`); trinket subtypes with the golden flag ≤ 32768+189 |
| `item` (new) | 8192×24 | collectible 0..2047; trinket 2048+id; card 2560+id; pill effect 2816+eff+1; pill colour 2944+c; poop spell 3008+t; crafting 3072+id; reserved to 8190; out of namespace → **8191** | collectibles 552 / 732 / 732 (REPA items.xml max 732); trinkets 128 / 189 / 189; cards 54 / 97 / 97 (`E_REP:3752-3859`, REPA pocketitems max 97); pill effects 46 / 49; pill colours 13 / 14 |
| `cat` (new) | 1024×16 | id = 0 if code == 0, else base + min(code, size−1). Namespaces (base, size): ptype (0, 64) [code = ptype+2]; body_role (64, 16); controller (80, 16); animation (96, 32); stage (128, 32); stage_type (160, 16); room_type (176, 64); difficulty (240, 16); game_version (256, 16); control_scheme (272, 16); entity kind (288, 16); champion (304, 64) [code+2]; slot family (368, 16); charge_type (384, 16); free 400..1023 | ptype 17 / 40 / 40 (REPA players.xml ids 0..40) → 64 rows; stage 12 / 13; stage type 3 / 5; room type 23 / 29 / ~30 (UNVERIFIED); champion 0..25 Rep |
| `anim` (new) | 4096×16 | hash(file, anim) mod 4095 + 1; 0 = unknown | hashing collisions are accepted |
| minimap icon | 64×8 | code 0..63 | room types as above |

The guard is a `torch.where` in the model, so the wire keeps raw values for the converter and diagnostics. Raw id columns are **not** also fed as numbers; v2 fed them as `e/64` (`SI:619`). This keeps large subtypes out of the numeric path.

### 2.9 Recorder coverage

`REC` records the following v3 fields (`REC/state.lua:85-151,573-609`; `REC/README.md:106-133`):

- player: hearts including broken and rotten, giga bombs, golden key/bomb, poop mana, actives 0..3 with charge, battery (slot 0 only), cards 0..1, pills 0..1, trinkets 0..1, TearFlags l/h, controller, animation;
- level and room: stage, stage_type, curses, room and door rows with variant/state, room descriptors with display flags;
- entities: type, variant, subtype, champion, tear flags (tears only), pickup price, animation names.

**Not recorded**, so the converter writes 0 with known 0 or leaves them absent:

- soul/blood charge, poop spells, IsBoneHeart order;
- projectile flags, entity flags, laser/knife/bomb tear flags;
- battery of slots 1..3, cards/pills 2..3;
- the weapon charge counter.

---

## 3. Action space

### 3.1 Heads

v2 heads are (9, 5, 2, 2, 2) (`TP:150-154`). The new head order is frozen:

| # | Head | Size | ButtonAction (by name) | AB+ number (`E_AB:2627`) | Rep J746 (`E_REP:4023`) | Rep+ |
|---|---|---|---|---|---|---|
| 0 | move | 9 | none / UP / UP+RIGHT / … (`LUA:211-215`) | LEFT 0, RIGHT 1, UP 2, DOWN 3 | same | renumbered, values UNVERIFIED (`BP:R+`) |
| 1 | shoot | 5 | none / SHOOTUP / RIGHT / DOWN / LEFT (`LUA:216-218`) | 4..7 | same | UNVERIFIED |
| 2 | bomb | 2 | BOMB | 8 | same | UNVERIFIED |
| 3 | item | 2 | ITEM (active slot 0) | 9 | same | UNVERIFIED |
| 4 | pill | 2 | PILLCARD (pocket position 0: card, pill or, on Rep, the pocket active) | 10 | same | UNVERIFIED |
| 5 | **drop** | 2 | DROP | 11 | same | UNVERIFIED |

**What DROP does** (game semantics, UNVERIFIED except where the AB+ bridge has used it):

- AB+ and Rep: Schoolbag swap, cycling pocket items, hold to drop a trinket or pocket item, Forgotten ↔ Soul switch;
- Rep: Jacob & Esau control;
- Rep+: the behaviour depends on `JacobEsauControls`, which is carried as P col 65.

**Not added:**

- **MAP.** The minimap is already in M; the full map shows no extra rooms. UNVERIFIED for Curse of the Lost.
- **Separate giga bomb or pocket-active heads.** The game picks these through BOMB and PILLCARD (UNVERIFIED for giga bombs).
- **Analog aim and movement.** Not covered (N4).

| Size | v2 | v3 |
|---|---|---|
| `pi` = `Linear(192, Σheads)` (`TP:1086`) | 20 | **22** |
| `pending` one-hot (`SI:576,611,651`) | 20 | **22** |
| pending / actions int64 per decision (`nh`) | 5 | **6** |
| CRN uniform columns (`heads`, `ACT:1694-1695,1938`) | 20 | **22** |
| danger head | 9 | 9 (unchanged) |

`pi.bias` for drop = 1 starts at `USE_BIAS = −4` (`TP:155,1095-1096`).

### 3.2 Wire protocol

| Item | v2 | v3 |
|---|---|---|
| action_code (`abplus.py:559`, `LUA:2317-2318`) | m + 9s + 45b + 90i + 180p (360 codes) | **+ 360d (720 codes)** |
| step line (`LUA:2505-2540`, `TURBO:2297-2316`, `abplus.py:854-870`) | `S rep m s b i [p]` | `S rep m s b i p d` (a 7th number; absent = 0) |
| JSON step (`LUA:32`, `fork_sampler.py:158,164`) | `drop` is already parsed (`LUA:921`) | the client sends `drop` |
| native input mask (`LUA:902-907`, `TURBO:745-780`) | DROP is already in `CONTROLLED` (`LUA:206-210`) | unchanged |

### 3.3 Hold versus trigger over the 4-frame repeat

These are the current semantics, kept unchanged:

- an action value of 1 holds the button for every logic frame of the step;
- `triggered` is 1 only on the step's first logic frame, and only if the button was not held in the previous step (`LUA:922-927`, cleared at `LUA:3000-3002`).

Consequences:

- `drop 0, 1, 0` is one 4-frame press;
- k consecutive 1s are one press held 4k frames;
- the shortest repeated press is 1, 0, 1 (8 frames apart);
- a reset clears both held and pending (`LUA:824`, `ACT:1798-1799`).

### 3.4 The 60 Hz edge difference (documented, not solved)

- Rep polls input on intermediate 60 Hz ticks, whose `triggered` drives items, cards and bombs (`REC/README.md:144`).
- The AB+ hook answers every query in the step window from `state.triggered` (`LUA:931-945`).
- On Rep the same hook may therefore expose the edge on an intermediate tick before the logic update, or on two ticks.
- The hook also answers for **every** player entity without checking ControllerIndex (`LUA:933-936`). That is fine for J&E (one controller), but a Rep producer must filter by controller for co-op.
- Both points need a target-build test (spec 9.6) before any Rep training.

---

## 4. Learner-only parts

| Part | Today | Needed for consistency with v3 |
|---|---|---|
| Privileged critic `tok_priv.py` | `N_COLLECTIBLE = 1024` and ids clipped to 1023 (`tok_priv.py:131,218`; `tok_obs.py:75-76`) | **No range change**: Rep's 732 fits, and ITEM_UNKNOWN 1023 does not collide. Mask the trinket golden flag (& 0x7fff) and the pill giant flag (& 0x7ff) before table lookups. Its inputs come from the AB+ ROW (N5). |
| Aux targets `tok_aux.py` | `N_ROOM_TYPE = 32` (`tok_obs.py:251`, `tok_aux.py:299-334`); STATE from the old ROW (`tok_aux.py:39-75`) | Raise `N_ROOM_TYPE` to 64. Rep uses 0..29 and Rep+ probably 30, which leaves 1 spare today. EVENTS (`tok_aux.py:37`) are version-free: no change. No new aux target is needed for compatibility. |
| Policy pchar `N_CHAR = 32` (`tok_obs.py:253`) | old-ROW path only; not read by the block model | unused by v3; leave |
| Teacher v2 / CF records (`tok_teacher2.py:56-61,140-143`) | `pend0..4`, `act0..4`, `V2_META = 29`, `five()` | add `pend5` and `act5` (V2_META 31) and `five` → `six` (also `tok_floor.py:939-942`). Intents edit only the move (`tok_cf.py:489-500`), so drop passes through as the policy's. |
| Injury teacher `TEACH_META = 16` (`tok_sampler.py:173`) | stores only **3** heads of the action under way. This is already incomplete with items. | store all 6 heads (TEACH_META 19), or document that the loss masks them |
| `ACTION_F = 5` (`tok_sampler.py:175`) | – | 6 |
| Losses | PPO log-prob summed over heads (`gumbel_sample`, `TP:195-207`) | generic over HEADS; entropy and KL per head include drop |

---

## 5. Impact per file

The size estimates are rough counts of changed lines.

| File | Change | Size |
|---|---|---|
| `LUA` | `visible_input_block` v3: new P fields, bodies including sub-players and twins, roles, slot families, Rep-safe getter names, game_version, control_scheme; effect rule (one list replaces 3 copies); entity-flag and bomb/knife flag words; door variant/state; minimap icon code; play-code and step-line `drop` | 250–400 |
| `TURBO` | lean player record (+stage_type, more doubles; `OBS_PLAYER_F`, `RE_PLAYER_F` bump); lean entity record (+entity-flags u64, bomb/knife flags, kinds 7–9; `LEAN_ENTITY_BYTES` bump); effect rule; step-line parser with 7 numbers; symbol or offset look-ups for the AB+ getters (risk: offsets) | 200–350 |
| `VIS` | encoder v3 (5-word cells, raw units, flag remap table, new P/E columns, minimap bits) and `abp_visible_unpack` (`VIS:357-384`: 1792 / 169 / 324 / widths) | 250 (largely a rewrite) |
| `SI` | constants, `pack_*`, decode (host / device / compact), `canvas_decoder` bit layouts, `StructuredSpatial` (new tables, guards, scale vector, bit expansion, minimap embedding), pending 22. **Recommended:** move the producer half (`observation_frame`, `pack_frame`, `prioritize_entities`) into a separate module owned by package C. | 300–400 |
| `TP` | HEADS (6), `pi` 22, drop bias, `heads_of`, `action_gate` (`SI:679-682`) | 30 |
| `tok_obs.py` | `N_ROOM_TYPE` 64; ROW unchanged (old path) | 5 |
| `ACT` / `abp_core.h` | hard-coded layout `frame_fixed = 1969` (`ACT:555`), zero-fill `80/324` (`ACT:832`), expected-size formula (`ACT:1636`) → v3 constants; nh 6 / heads 22 are passed in (generic: `abp_core.h:56-143`, `ACT:645-647`) | 20–40 |
| `abp_learn.c` / `learn_native.py` | `room_n = 1792` (`learn_native.py:156`), widths, `ABP_LEARN_VERSION` bump (`abp_learn.c:14`) | 20–40 |
| `block_paged.py:915`, `block_compile.py:217` | hard-coded shapes (1792, 80, 324) | 10 |
| `tok_teacher2.py`, `tok_cf.py`, `tok_floor.py`, `tok_sampler.py` | six-action records, ACTION_F, TEACH_META | 40–60 |
| `abplus.py`, `fork_sampler.py` | `step_line(..., drop)`, `action_code` + drop | 15 |
| `tok_priv.py` | flag masking | 5 |
| `tok_aux.py` | via `N_ROOM_TYPE` | 0–5 |

### Tests to update or add

| Test | Purpose |
|---|---|
| `tests/test_visible_native.py` | C encoder == Python reference, byte for byte, on AB+ payloads **plus synthetic Rep fixtures**: J&E, Tainted Forgotten, ActiveSlot 2/3, broken/rotten hearts, TearFlags bits 64–127, room type 29/30, id −1 / 70000 / 0x8000+id |
| `test_visible_input.py`, `test_visible_review.py` | schema, masking (Unknown/Blind), known-flag semantics, no hidden field leaks |
| `test_b60_native_pack.py`, `test_native_learner_inputs.py` | C learner packing == `decode_frames_compact` |
| `test_b52_cpp_actor.py`, `test_b58_native_consumer.py`, `test_b58*`, `test_block_protocol.py` | C++ actor vs Python consumer with nh 6 / heads 22; CRN columns |
| `test_cf_smoke.py`, `abplus_probe_teacher_v2.py` | six-tuple records replay equal |
| new: guard test | every table: negative, too large and NaN ids → unknown row, no crash |
| new: drop protocol test | step line, JSON and play code give the same held/triggered sequence; a reset clears them |
| deterministic replay (`abplus_probe_frame_exact.py`, Go-Explore digest returns) | unchanged trajectories with drop = 0 |
| `engine/core_tests` | layout constants |

### To re-measure

- the speed gates: single card, split card, two shards with gradient sync;
- per-step Lua and native producer time (`lean_profile`, `LUA:1884-1938`);
- frame bytes and frame-arena memory per rollout;
- learner peak memory;
- the token-count distribution under the effect rule.

---

## 6. Two parallel packages and the frozen interface

### Frozen artefacts

Both packages code against these artefacts. They are generated from one table, proposed as `schema_v3.py` plus `abp_schema_v3.h`.

| ID | Artefact |
|---|---|
| F1 | `SCHEMA = 'isaac-visible-prlmens-g-v3'`, `VERSION = 2026111201`, header layout (2.1); `FIXED = 2417`; `PF = 192`, `LOCAL = 405`, `EF = 64`, `NF = 8`, `SF = 24`; room word 0/1 bit layouts and word order (2.2); minimap bits (2.4); every column index and its raw unit (2.5–2.7); known-flag semantics (G3); flag-word encoding and the AB+→canonical TearFlags table (2.6); the visible entity-flag mask; code conventions (+2 for ptype, controller and champion; 0 = none); producer ids. |
| F2 | Heads `(9, 5, 2, 2, 2, 2)` in order; the ButtonAction name mapping (3.1); `action_code` with drop; step line with 7 fields; `pending` int64 `[n, 6]`; CRN columns 22. |
| F3 | Golden fixtures. Package C delivers ≥ 20 frames: real AB+ ones (several characters including the Forgotten, Schoolbag, two pockets) and synthetic Rep ones (2.5 cases). Each comes with its lean input and the expected bytes. Package A tests decode and model forward on them before any producer is finished. |
| F4 | Record formats: `V2_FIELDS` with `pend0..5` / `act0..5` (V2_META 31), `TEACH_META`, `ACTION_F = 6`. |

### Package C: producer and native side

Files: `LUA`, `TURBO`, `VIS`, the producer module split out of `SI`, `ACT` / `abp_core.h` constants, `abp_learn.c` + `learn_native.py`, `abplus.py` / `fork_sampler.py` protocol.

### Package A: model and learner side

Files: decode part of `SI` + `StructuredSpatial`, `TP` heads, `block_paged.py` / `block_compile.py` shapes, `tok_priv.py`, `tok_aux.py` / `tok_obs.N_ROOM_TYPE`, `tok_teacher2.py` / `tok_cf.py` / `tok_floor.py` / `tok_sampler.py` records, losses.

### Ordered acceptance

1. F1–F4 merged as constants; both sides import them, with no literal 1969 / 80 / 324 / 48 / 20 left (grep gate).
2. **C:** Python reference == C encoder on fixtures and on ≥ 10k live AB+ steps (`native_check` mode 2, `LUA:1847-1861`, 0 mismatches).
3. **A:** host decode == device decode == C learner pack; model forward and backward on F3; guard tests pass.
4. **C:** Drop end-to-end on AB+ (spec 9.6 list): Schoolbag swap, pocket switch, hold-to-drop trinket, Forgotten/Soul switch, reset clears held/pending. Requested versus applied actions are logged.
5. **C + A:** C++ actor == Python consumer with nh 6; CRN uniforms equal.
6. Deterministic replay: unchanged digests for recorded AB+ action sequences (drop = 0).
7. Teacher v2 and CF probes replay equal with six-tuples.
8. Speed and memory re-measured against the current gates (section 5); the owner decides.
9. Block-smoke training run (`abplus_block_smoke.py`): finite losses, no recompile storms, then the pre-training checklist.

---

## 7. Open questions for the owner

| # | Question | Recommendation |
|---|---|---|
| Q1 | Add the namespaced `item` table (8192×24, +197k params) instead of putting slot and inventory ids into the shared `identity` table, where card 5, trinket 5, collectible 5 and entity type 5 share a row (`SI:621,626`)? | **Yes**. It is collision-free, in line with spec 9.2/9.3, and it changes shapes, so decide now. |
| Q2 | Replace the 10-variant effect whitelist with "all visible effects minus a decorative list"? | **Yes** in v3. It changes no shape. Gate it on measured token counts; the 128 cap with priority still holds. |
| Q3 | Give the model `game_version` (P col 64)? | **Yes**: rules differ (N2). For zero-shot on Rep, copy the AB+ row into the Rep row at load time; this changes no shape. |
| Q4 | Feed the visible charge-bar counter (P 37 + known flag 51)? It is raw on AB+, and Rep has no vanilla getter. | **Yes** with the known flag. It is visible above the player and was previously zeroed as a "hidden counter" (`SI:610`). |
| Q5 | Pass all tear-namespace and projectile flag bits, or a reviewed visible subset? | All tear bits, since they follow from visible items. Projectile bits: mask CHANGE_FLAGS_AFTER_TIMEOUT and CHANGE_VELOCITY_AFTER_TIMEOUT (bits 32–33), which describe hidden future behaviour. A mask change is shape-neutral. |
| Q6 | Room-canvas codes (door type, door variant, grid type) as scalars (code/64) or as per-cell embeddings? | **Scalars** in v3: no activation growth. The minimap icon gets an embedding because it is small. |
| Q7 | Reserved capacity: about 20% spare columns per token type (P 25/192, E 13/64, N 2/8, S 4/24, cat namespaces with head-room)? | **Accept**. The cost is a few kilobytes per frame and about 20k params; it avoids a later shape change. |

---

## 8. Parameter, token and speed estimate

TokPolicy with width 192, 4 layers, 4 heads, items, item_tokens, global_token, charge, ent_ext, pchar, block_memory. These were measured on this tree with the `abplus_block_smoke.py:41` constructor: **10,842,975 parameters**, of which `structured_spatial` has 4,298,112, `identity` 1,572,864, `block_memory` 6,366,080 and `pi` 3,860.

| Module | v2 | v3 | Δ |
|---|---|---|---|
| body MLP (input 80 → 176 numeric/bit + 16 cat × 16 = 432) | 52,608 | 120,192 | +67,584 |
| `cat` table 1024×16 | – | 16,384 | +16,384 |
| entity Linear (input 120 → 41 numeric + 256 flag bits + 8 npc bits + 3×24 ids + 2×16 cats + 16 anim = 425) | 23,232 | 81,792 | +58,560 |
| `anim` table 4096×16 | – | 65,536 | +65,536 |
| `item` table 8192×24 (Q1) | – | 196,608 | +196,608 |
| inventory Linear (28 → 31) | 5,568 | 6,144 | +576 |
| slots Linear (37 → 100) | 7,296 | 19,392 | +12,096 |
| room + local first conv (RC 16 → 21) | – | – | +2,880 |
| minimap conv (8 → 12 ch) + icon table 64×8 | – | – | +1,664 |
| pending (20 → 22), pi (20 → 22) | 7,700 | 8,470 | +770 |
| **Total** | **10,842,975** | **≈ 11,265,600** | **+≈ 422,700 (+3.9%)**; without Q1 +226k (+2.1%) |
| aux head (learner-only), N_ROOM_TYPE 32 → 64 | – | – | +8,224 |

### Tokens

Per row: 30 (28 R + M + G) + 2·bodies + entities + inventory + slots (`SI:196`). The structure is unchanged.

- **AB+:** identical counts, except +2 tokens while the Forgotten/Soul pair exists (sub-body P + L), and more effect entities under Q2 (unmeasured).
- **Rep:** slots up to about 16 (4 actives + 4 pockets + 2 trinkets + 6 poop spells). This fits the slot bucket 16 (`SI:84`); larger counts double, as today.

### Speed

| Item | v2 | v3 |
|---|---|---|
| Typical AB+ frame (1 body, ~10 entities, ~10 items, 3 slots) | ≈ 1969 + 404 + 480 + 40 + 39 = 2,932 floats ≈ 11.7 KB | ≈ 2417 + 597 + 640 + 80 + 72 = 3,806 floats ≈ 15.2 KB (+30%) |

- **Frame bytes:** +30% for the frame arena, H2D and rollout storage. At 32,768 decisions per rollout this is about +115 MB. Re-measure.
- **GPU:** input projections grow, for example the entity token from 120×192 to 425×192 multiply-adds. The 4 Spatial layers cost about 12·192² ≈ 442k MACs per token, so Spatial FLOPs change by less than about 5%. The flag-bit expansion fuses into the compiled decoder (`SI:286-296`).
- **Producer:**
  - AB+ adds per step: GetStageType, GetSubPlayer per body, IsBoneHeart over the heart slots, and one more u64 per entity in the native record;
  - the JSON visible block (`LUA:1947-1994`) grows a little;
  - the Rep producer's `GetCollectibleNum` loop over 732 ids per body per step (as `LUA:1965`) is a known cost to optimise there; it is not part of this schema.
- **No throughput claim** is made before the re-measurement listed in section 5.
