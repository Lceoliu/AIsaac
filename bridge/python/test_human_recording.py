import copy
import json
from pathlib import Path
import tempfile
import unittest

from isaac_bridge.human_recording import archive_episode, make_replay, validate_episode


class HumanRecordingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)/'episode-0001.jsonl'
        obs = {'game_frame':10,'logic_frames':0,'players':[{'dead':False}],
               'terrain':{'cells':[[0]]},'entities':[]}
        after=copy.deepcopy(obs);after.update(game_frame=11,logic_frames=1)
        self.rows=[dict(type='header',schema='isaac-human-v1',source='qa',obs=obs,
                        action_ids=[0],hooks={'value':2},max_logic_frames=3600,logic_hz=30),
                   dict(type='frame',frame=1,previous_game_frame=10,obs=after,
                        queries=[dict(action=0,hook=2,value=1)]),
                   dict(type='end',outcome='manual_stop',frames=1)]

    def write(self):
        self.path.write_text(''.join(json.dumps(r)+'\n' for r in self.rows),encoding='utf8')

    def test_archive_and_replay_preserve_raw(self):
        self.write();before=self.path.read_bytes()
        result=archive_episode(self.path)
        self.assertEqual(result['active_queries'],1)
        self.assertEqual(self.path.read_bytes(),before)
        self.assertEqual(validate_episode(self.path.with_suffix('.jsonl.gz'))['frames'],1)
        out=self.path.with_suffix('.html');make_replay(self.path,out)
        self.assertNotIn('/*DATA*/',out.read_text(encoding='utf8'))

    def test_truncated_capture_rejected(self):
        self.rows.pop();self.write()
        with self.assertRaisesRegex(ValueError,'incomplete'):validate_episode(self.path)

    def test_native_frame_gap_rejected(self):
        self.rows[1]['obs']['game_frame']=12;self.write()
        with self.assertRaisesRegex(ValueError,'discontinuity'):validate_episode(self.path)

    def test_wrong_preaction_pair_rejected(self):
        self.rows[1]['previous_game_frame']=9;self.write()
        with self.assertRaisesRegex(ValueError,'discontinuity'):validate_episode(self.path)

    def test_absent_input_rejected(self):
        self.rows[1]['queries']=[];self.write()
        with self.assertRaisesRegex(ValueError,'no game input'):validate_episode(self.path)

    def test_fake_death_rejected(self):
        self.rows[-1]['outcome']='death';self.write()
        with self.assertRaisesRegex(ValueError,'Death'):validate_episode(self.path)

    def test_unknown_input_rejected(self):
        self.rows[1]['queries'][0]['action']=999;self.write()
        with self.assertRaisesRegex(ValueError,'Unrecognized'):validate_episode(self.path)

    def test_fake_win_rejected(self):
        self.rows[-1]['outcome']='win';self.rows[1]['obs']['entities']=[{'boss':True,'boss_hp':.5}];self.write()
        with self.assertRaisesRegex(ValueError,'Win outcome'):validate_episode(self.path)

    def test_idle_data_preserved_but_flagged(self):
        self.rows[1]['queries'][0]['value']=0;self.write()
        result=validate_episode(self.path)
        self.assertTrue(result['valid']);self.assertTrue(result['warnings'])

    def test_native_dead_flag_accepts_death_animation_with_positive_hp(self):
        self.rows[-1]['outcome']='win'
        self.rows[1]['obs']['entities']=[{'boss':True,'boss_hp':.5}]
        self.rows[1]['terminal_evidence']={'boss_dead':True}
        self.write();self.assertTrue(validate_episode(self.path)['valid'])

    def test_commands_are_not_executed_from_render_callback(self):
        path=Path(__file__).resolve().parents[1]/'mod/isaac_human_recorder/main.lua'
        render=path.read_text(encoding='utf8').split('ModCallbacks.MC_POST_RENDER,guarded(function()',1)[1]
        self.assertNotIn('command(c.command)',render)
        self.assertNotIn('then start_record()',render)


if __name__=='__main__':unittest.main()
