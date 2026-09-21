"""Keyboard acceptance viewer; all dynamics execute in the calibrated Rust core.

Uses locally extracted original sprite sheets/ANM2, not redistributed art.
Presentation only: body/head animation and room decoration are not engine clones.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import xml.etree.ElementTree as ET

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--exe',type=Path,required=True)
parser.add_argument('--smoke',type=Path,help='render 240 scripted ticks to PNG without a window')
args=parser.parse_args()
if args.smoke: os.environ['SDL_VIDEODRIVER']='dummy'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT']='1'
import pygame as pg

root=Path(__file__).resolve().parents[3]
art=root/'analysis/resources/gibbed/graphics/resources/gfx'
animations=root/'analysis/resources/animations-a/anm2'
pg.init()
screen=pg.display.set_mode((1120,760))
pg.display.set_caption('Isaac | native-aligned motion')
canvas=pg.Surface((560,360))
font=pg.font.Font(None,22)
atlas=pg.image.load(str(art/'characters/costumes/character_001_isaac.png')).convert_alpha()
tear_atlas=pg.image.load(str(art/'tears.png')).convert_alpha()
player_xml=ET.parse(animations/'001.000_player.anm2')
tear_xml=ET.parse(animations/'002.000_tear.anm2')
player_anims={a.attrib['Name']:a for a in player_xml.findall('.//Animation')}
tear_anim=next(a for a in tear_xml.findall('.//Animation') if a.attrib['Name']=='RegularTear6')
backdrop=pg.image.load(str(art/'backdrop/01_basement.png')).convert_alpha()
background=pg.Surface((560,360)); background.fill((44,28,23))
floor=backdrop.subsurface((52,52,182,104))
background.blit(pg.transform.scale(floor,(520,280)),(20,40))
top=pg.transform.scale(backdrop.subsurface((52,10,182,42)),(520,40))
background.blit(top,(20,0)); background.blit(pg.transform.flip(top,False,True),(20,320))
side=pg.transform.scale(backdrop.subsurface((10,52,42,104)),(20,280))
background.blit(side,(0,40)); background.blit(pg.transform.flip(side,True,False),(540,40))

def sprite(animation,layer,tick,sheet,position):
    node=next(n for n in animation.findall('./LayerAnimations/LayerAnimation') if n.attrib['LayerId']==str(layer))
    frames=node.findall('Frame')
    duration=sum(int(f.attrib['Delay']) for f in frames)
    t=int(tick)%duration if animation.attrib['Loop']=='True' else min(int(tick),duration-1)
    for f in frames:
        if t<int(f.attrib['Delay']): break
        t-=int(f.attrib['Delay'])
    a=f.attrib
    image=sheet.subsurface(tuple(int(a[k]) for k in ('XCrop','YCrop','Width','Height')))
    sx=float(a['XScale'])/100; sy=float(a['YScale'])/100
    image=pg.transform.scale(image,(round(image.get_width()*abs(sx)),round(image.get_height()*abs(sy))))
    image=pg.transform.flip(image,sx<0,sy<0)
    x,y=position
    canvas.blit(image,(round(x-40+float(a['XPosition'])-float(a['XPivot'])*abs(sx)),
                       round(y-100+float(a['YPosition'])-float(a['YPivot'])*abs(sy))))

proc=subprocess.Popen([str(args.exe.resolve())],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                      text=True,bufsize=1,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
def command(line):
    proc.stdin.write(line+'\n'); proc.stdin.flush()
    return json.loads(proc.stdout.readline())

state=command('reset 320 280 -1')
tick=0; walk_time=0.; head='Down'; body='Down'; show_boxes=False
deadline=time.perf_counter()
started=deadline
move_codes={(0,0):0,(0,-1):1,(1,-1):2,(1,0):3,(1,1):4,(0,1):5,(-1,1):6,(-1,0):7,(-1,-1):8}
directions={1:'Up',2:'Right',3:'Down',4:'Left'}
running=True
try:
    while running:
        for event in pg.event.get():
            if event.type==pg.QUIT: running=False
            if event.type==pg.KEYDOWN:
                if event.key==pg.K_ESCAPE: running=False
                if event.key==pg.K_F1: show_boxes=not show_boxes
                if event.key==pg.K_r:
                    state=command('reset 320 280 -1'); tick=0
        if not running: break
        keys=pg.key.get_pressed()
        mx=int(keys[pg.K_d])-int(keys[pg.K_a]); my=int(keys[pg.K_s])-int(keys[pg.K_w])
        shoot=next((i for i,k in enumerate([pg.K_UP,pg.K_RIGHT,pg.K_DOWN,pg.K_LEFT],1) if keys[k]),0)
        if args.smoke:
            mx,my,shoot=(1,0,1) if tick<60 else (-1,0,2) if tick<120 else (0,0,2)
        move=move_codes[mx,my]
        state=command(f"{'half' if tick%2==0 else 'logic'} {move} {shoot}")
        p=state['player']; vx,vy=p['vel']; speed=(vx*vx+vy*vy)**0.5
        if speed>0.05:
            body=('Right' if vx>0 else 'Left') if abs(vx)>abs(vy) else ('Down' if vy>0 else 'Up')
            walk_time+=speed/4.415
        else: walk_time=0
        if shoot: head=directions[shoot]
        elif mx or my: head=body
        canvas.blit(background,(0,0))
        px,py=p['pos']
        pg.draw.ellipse(canvas,(40,28,25),(round(px-50),round(py-103),20,7))
        for t in state['tears']:
            if t['dead']: continue
            tx,ty=t['pos']
            pg.draw.ellipse(canvas,(43,30,26),(round(tx-44),round(ty-102),8,4))
            sprite(tear_anim,0,14,tear_atlas,(tx,ty+t['height']))
            if show_boxes: pg.draw.circle(canvas,(117,199,223),(round(tx-40),round(ty-100)),7,1)
        sprite(player_anims['Walk'+body],1,walk_time,atlas,(px,py))
        sprite(player_anims['Head'+head],4,2 if shoot and p['fire_delay']>7 else 0,atlas,(px,py))
        if show_boxes: pg.draw.circle(canvas,(205,182,124),(round(px-40),round(py-100)),10,1)
        screen.fill((28,25,23)); screen.blit(pg.transform.scale(canvas,(1120,720)),(0,0))
        label=f'WASD move | Arrows shoot | R reset | F1 collision circles | Esc exit    speed {speed:.3f}    60 Hz player / 30 Hz tears'
        screen.blit(font.render(label,True,(221,211,188)),(12,732))
        pg.display.flip(); tick+=1
        if args.smoke and tick==240:
            args.smoke.parent.mkdir(parents=True,exist_ok=True)
            pg.image.save(screen,str(args.smoke)); running=False
        # SDL's integer-millisecond 1000/60 limiter can run at 62.5 Hz.
        # Accumulated high-resolution deadlines preserve the native 60 Hz.
        deadline+=1.0/60.0
        remaining=deadline-time.perf_counter()
        if remaining>0: time.sleep(remaining)
    elapsed=time.perf_counter()-started
finally:
    proc.stdin.close(); exit_code=proc.wait(timeout=5)
    pg.quit()
assert exit_code==0,exit_code
print(json.dumps({'ticks':tick,'engine_exit':exit_code,'last_position':state['player']['pos'],
                  'wall_seconds':elapsed,'ticks_per_second':tick/elapsed}))
