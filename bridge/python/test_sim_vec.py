import unittest
import numpy as np
from isaac_bridge.sim_vec import SimVecEnv
from isaac_bridge.transformer_obs import VisibleHistory
from isaac_bridge.sim_obs import rust_visible_observation

class SimVectorTest(unittest.TestCase):
 def test_two_workers_same_seed_replay_and_bomb_mask(self):
  a=SimVecEnv(2,seed=42,history=8,capacity=128);b=SimVecEnv(2,seed=42,history=8,capacity=128)
  try:
   oa=a.reset();ob=b.reset()
   for k in oa:np.testing.assert_array_equal(oa[k],ob[k])
   for step in range(40):
    action=np.array([[17,int(step==0),0],[36,0,0]])
    oa,ra,da,ia=a.step(action);ob,rb,db,ib=b.step(action)
    for k in oa:np.testing.assert_array_equal(oa[k],ob[k])
    np.testing.assert_array_equal(ra,rb)
   self.assertFalse(a.action_masks()[0,46]);self.assertTrue(a.action_masks()[1,46])
   self.assertEqual(oa['entities'].shape,(2,8,128,31))
  finally:a.close();b.close()

 def test_episode_end_autoreset_clears_history(self):
  e=SimVecEnv(2,history=8,capacity=128)
  try:
   e.reset();finished=0
   for step in range(1850):
    o,r,d,info=e.step(np.zeros((2,3),np.int32))
    for i in np.flatnonzero(d):
     finished+=1;self.assertEqual(o['history_mask'][i].sum(),1)
     self.assertIn('terminal_observation',info[i]);self.assertIn(info[i]['outcome'],('death','time_limit','win'))
     self.assertGreater(info[i]['terminal_observation']['time'].max(),0)
    if finished>=2:break
   self.assertGreaterEqual(finished,2)
  finally:e.close()

 def test_hidden_fields_and_cosmetics_do_not_enter_actor(self):
  e=SimVecEnv(1,history=4,capacity=128)
  try:
   e.reset();state=e.batch.states()[0]['state'];raw=rust_visible_observation(state)
   a=VisibleHistory(4,128).append(raw)
   for boss in state['bosses']:boss['debug_state']=999;boss['debug_target']=[-9999,9999]
   raw=rust_visible_observation(state);raw['players'][0].update(anim='UNMODELED',aframe=987,flip=True)
   b=VisibleHistory(4,128).append(raw)
   for k in a:np.testing.assert_array_equal(a[k],b[k])
   self.assertTrue(a['terrain'][0,4].any() or state['layout']==1010)
  finally:e.close()

if __name__=='__main__':unittest.main()
