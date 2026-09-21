"""Render the actual viewer function without launching its interactive loop."""
import ast
import os
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

os.environ['SDL_VIDEODRIVER']='dummy'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT']='1'
import pygame as pg

VIEWER=Path(os.environ.get('ISAAC_TEST_VIEWER',Path(__file__).with_name('play_motion.py')))
ROOT=Path(__file__).resolve().parents[3]
ART=ROOT/'analysis/resources/gibbed/graphics/resources/gfx'
ANM=ROOT/'analysis/resources/animations-a/anm2/020.000_monstro.anm2'

class MonstroAnimationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pg.init();pg.display.set_mode((1,1))
        cls.canvas=pg.Surface((640,480),pg.SRCALPHA)
        function=next(n for n in ast.parse(VIEWER.read_text(encoding='utf8')).body
                      if isinstance(n,ast.FunctionDef) and n.name=='sprite')
        scope={'pg':pg,'canvas':cls.canvas}
        exec(compile(ast.Module(body=[function],type_ignores=[]),str(VIEWER),'exec'),scope)
        cls.draw=staticmethod(scope['sprite'])
        tree=ET.parse(ANM)
        cls.anims={a.get('Name'):a for a in tree.findall('.//Animation')}
        sheets={s.get('Id'):pg.image.load(str(ART/s.get('Path').lower()))
                for s in tree.findall('.//Spritesheet')}
        cls.layers={int(l.get('Id')):sheets[l.get('SpritesheetId')]
                    for l in tree.findall('./Content/Layers/Layer')}

    @classmethod
    def tearDownClass(cls):pg.quit()

    def test_all_animation_frames_both_facing_directions(self):
        rendered=0
        for name,animation in self.anims.items():
            for tick in range(int(animation.get('FrameNum'))):
                for layer in animation.findall('./LayerAnimations/LayerAnimation'):
                    layer_id=int(layer.get('LayerId'))
                    for flip in (False,True):
                        with self.subTest(animation=name,frame=tick,layer=layer_id,flip=flip):
                            self.draw(animation,layer_id,tick,self.layers[layer_id],(320,300),flip)
                            rendered+=1
        print(f'rendered_animation_layer_frames={rendered}')

    def test_high_jump_blank_body_keeps_shadow_and_returns_on_descent(self):
        for tick in range(29):
            with self.subTest(frame=tick):
                self.canvas.fill((0,0,0,0))
                self.draw(self.anims['JumpDown'],0,tick,self.layers[0],(320,300))
                self.assertEqual(self.canvas.get_bounding_rect().size,(0,0))
                self.draw(self.anims['JumpDown'],1,tick,self.layers[1],(320,300))
                self.assertGreater(self.canvas.get_bounding_rect().width,0)
        self.canvas.fill((0,0,0,0))
        self.draw(self.anims['JumpDown'],0,29,self.layers[0],(320,300))
        self.assertGreater(self.canvas.get_bounding_rect().width,0)

if __name__=='__main__':unittest.main()
