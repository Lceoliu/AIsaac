"""visible-v2 constants (schema isaac-visible-prlmens-g-v2): shared by the decoder (structured_input) and the
producer (visible_producer). Moved verbatim from structured_input.py (2026-11-12, before visible-v3)."""

SCHEMA = 'isaac-visible-p r l m e n s-g-v2'.replace(' ', '')
VERSION = 2026110702
FRAME_BYTES = 131072
P_FIELDS = ('x','y','vx','vy','size','hearts','max_hearts','soul','black','bone','eternal',
            'golden','lives','coins','bombs','keys','damage','fire_delay_max','shot_speed',
            'range','speed','luck','can_fly','invulnerable','controls','head_dir','fire_dir',
            'move_dir','aframe','dead','unknown_fire_delay','unknown_damage_cooldown','unknown_charge','weapons','ptype')
STAGE_INDEX,CURSE_START,OVERFLOW_START = 42,43,75
ENTITY_CLASSES=('interactable','monster','familiar','projectile','other')
PF, EF, NF, SF, RC = 80, 48, 4, 13, 16
HEADER = 8
FIXED = HEADER + 16*28*4 + 13*13
RAGGED = {'body':PF, 'local':9*9*4, 'entity':EF, 'inventory':NF, 'slots':SF}
ROOM_BITS = (1,3,1,1,1,1,1,1,1,5,1,1,1)
ROOM_SCALES = (1,6,1,1,1,1,1,1,1,30,1,1,1)
