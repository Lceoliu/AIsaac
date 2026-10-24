import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from isaac_bridge.human_dataset import align_action, prepare_dataset, HumanDemonstrations, masked_bc_loss
from isaac_bridge.human_recording import archive_episode
from test_monstro_gym import RecordedBridge


def fixture(length=8):
    obs = copy.deepcopy(RecordedBridge().original)
    obs.update(combat_schema=3,logic_frames=0,game_frame=100,paused=False)
    obs['players'][0].update(active_ready=False,active=0,bombs=1,controls=True,dead=False)
    for cell in obs['terrain']['cells']:
        cell.extend([int(cell[3]>=2),0])
    header = dict(type='header',schema='isaac-human-v1',source='human',qa_motion=False,qa_safe=False,
                  action_ids=list(range(12)),hooks=dict(value=2,pressed=0,triggered=1),
                  logic_hz=30,max_logic_frames=3600,seed='fixture',room_seed=1,obs=obs)
    frames=[]
    for i in range(1,length+1):
        after=copy.deepcopy(obs);after.update(game_frame=100+i,logic_frames=i)
        after['players'][0]['pos'][0]+=i
        queries=[dict(action=a,hook=2,value=int(a==2),game_frame=100+i) for a in range(8)]
        queries.append(dict(action=8,hook=1,value=False,game_frame=100+i))
        frames.append(dict(type='frame',frame=i,previous_game_frame=99+i,obs=after,
                           wall_delta_ms=33,queries=queries,terminal_evidence=dict(boss_dead=i==length)))
    return [header,*frames,dict(type='end',frames=length,outcome='win')]


class HumanDatasetTests(unittest.TestCase):
    def test_masked_loss_uses_sb3_multicategorical_and_ignores_unknown_heads(self):
        import torch
        from sb3_contrib.common.maskable.distributions import MaskableMultiCategoricalDistribution
        logits=torch.randn(2,49,requires_grad=True)
        distribution=MaskableMultiCategoricalDistribution([45,2,2]).proba_distribution(logits)
        class Policy:
            def get_distribution(self,obs,action_masks):return distribution
        batch=dict(obs={},acts=torch.zeros((2,3),dtype=torch.long),
                   label_mask=torch.tensor([[True,False,False],[True,False,False]]),action_masks=None)
        first=masked_bc_loss(Policy(),batch)
        batch['acts']=batch['acts'].clone()
        batch['acts'][:,1:]=1
        second=masked_bc_loss(Policy(),batch)
        torch.testing.assert_close(first,second)
        first.backward()
        self.assertFalse(logits.grad[:,45:].any())
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_native_direction_mapping(self):
        rows=fixture();a,label,known,_,_=align_action(rows[0],rows[0]['obs'],rows[1:3])
        self.assertEqual(a.tolist(),[5,0,0]);self.assertEqual(label.tolist(),[True,True,False])
        self.assertTrue(known.all())

    def test_all_45_directions_match_bridge_contract(self):
        moves=[(),(2,),(2,1),(1,),(1,3),(3,),(3,0),(0,),(0,2)]
        shoots=[(),(6,),(5,),(7,),(4,)]
        for m,move in enumerate(moves):
            for s,shoot in enumerate(shoots):
                rows=fixture(2)
                for row in rows[1:3]:
                    for q in row['queries']:
                        if q['action']<8:q['value']=int(q['action'] in move+shoot)
                a,label,_,_,_=align_action(rows[0],rows[0]['obs'],rows[1:3])
                self.assertEqual(a[0],m*5+s);self.assertTrue(label[0])

    def test_subframe_change_does_not_become_hard_label(self):
        r=fixture();r[1]['queries'].append(dict(action=2,hook=2,value=0))
        _,label,known,_,reasons=align_action(r[0],r[0]['obs'],r[1:3])
        self.assertFalse(label[0]);self.assertFalse(known[0]);self.assertTrue(reasons)

    def test_missing_second_frame_poll_is_not_filled_from_first(self):
        r=fixture();r[2]['queries']=[q for q in r[2]['queries'] if q['action']!=2]
        self.assertFalse(align_action(r[0],r[0]['obs'],r[1:3])[1][0])

    def test_diagonal_shooting_rejected_not_arbitrarily_prioritized(self):
        r=fixture()
        for row in r[1:3]:
            row['queries'][4]['value']=1;row['queries'][6]['value']=1
        self.assertFalse(align_action(r[0],r[0]['obs'],r[1:3])[1][0])

    def test_late_bomb_not_moved_to_earlier_observation(self):
        r=fixture();r[2]['queries'][-1]['value']=True;r[2]['obs']['players'][0]['bombs']=0
        a,label,_,_,reasons=align_action(r[0],r[0]['obs'],r[1:3])
        self.assertFalse(label[1]);self.assertEqual(a[1],0)
        self.assertIn('bomb_late_or_resource_mismatch',reasons)

    def test_first_frame_bomb_resource_agrees(self):
        r=fixture();r[1]['queries'][-1]['value']=True
        for row in r[1:3]:row['obs']['players'][0]['bombs']=0
        a,label,_,_,_=align_action(r[0],r[0]['obs'],r[1:3])
        self.assertEqual(a[1],1);self.assertTrue(label[1])

    def test_bomb_trigger_without_resource_change_rejected(self):
        r=fixture();r[1]['queries'][-1]['value']=True
        self.assertFalse(align_action(r[0],r[0]['obs'],r[1:3])[1][1])

    def test_no_bombs_does_not_swamp_loss(self):
        r=fixture();r[0]['obs']['players'][0]['bombs']=0
        _,label,known,available,_=align_action(r[0],r[0]['obs'],r[1:3])
        self.assertFalse(label[1]);self.assertTrue(known[1]);self.assertFalse(available[46])

    def test_unsupported_input_is_not_silently_discarded(self):
        r=fixture();r[1]['queries'].append(dict(action=10,hook=1,value=True))
        self.assertFalse(align_action(r[0],r[0]['obs'],r[1:3])[1].any())

    def test_pause_rejected(self):
        r=fixture();r[1]['obs']['paused']=True
        self.assertFalse(align_action(r[0],r[0]['obs'],r[1:3])[1].any())

    def test_episode_split_causal_history_roundtrip_and_raw_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'raw';source.mkdir();originals={}
            for ep in (1,2):
                rows=fixture(141)
                # A different NEXT target must not appear in its input history.
                for row in rows[3:5]:
                    for q in row['queries']:
                        if q['action']<8:q['value']=int(q['action']==1)
                path=source/f'episode-{ep:04}.jsonl'
                path.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf8')
                archive_episode(path);originals[path]=path.read_bytes()
            output=root/'prepared';result=prepare_dataset(source,output,'episode-0002.jsonl.gz')
            train=HumanDemonstrations(output,'train');val=HumanDemonstrations(output,'validation')
            self.assertEqual(len(train),70);self.assertEqual(len(val),70)
            self.assertEqual(result['episodes'][0]['odd_tail_frames'],1)
            self.assertNotEqual(train[0]['episode'],val[0]['episode'])
            first,second,last=train[0],train[1],train[-1]
            self.assertEqual(first['obs']['history_mask'].sum(),1)
            self.assertFalse(first['obs']['previous_action'].any())
            np.testing.assert_array_equal(second['obs']['previous_action'][1],[5,0,0,1])
            self.assertEqual(second['acts'][0],15)
            np.testing.assert_array_equal(train[2]['obs']['previous_action'][2],[15,0,0,1])
            self.assertAlmostEqual(float(second['obs']['time'][1]),2/30)
            self.assertEqual(last['obs']['history_mask'].sum(),64)
            self.assertAlmostEqual(float(last['obs']['time'][-1]),last['logic_frame']/30,places=5)
            # Mutating a retrieved history cannot poison the backing episode.
            first['obs']['player'][:]=123
            self.assertFalse(np.all(train[0]['obs']['player']==123))
            for path,data in originals.items():self.assertEqual(path.read_bytes(),data)

    def test_qa_cannot_become_training_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);raw=root/'raw';raw.mkdir()
            for ep in (1,2):
                rows=fixture();rows[0]['source']='qa'
                path=raw/f'episode-{ep:04}.jsonl'
                path.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf8');archive_episode(path)
            with self.assertRaisesRegex(ValueError,'human demonstration'):
                prepare_dataset(raw,root/'out','episode-0002.jsonl.gz')


if __name__=='__main__':unittest.main()
