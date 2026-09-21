"""Real Rust process -> visible schema -> existing CNN/Transformer forward pass."""
import copy,json,subprocess,unittest
from pathlib import Path
import torch
from isaac_bridge.sim_obs import rust_visible_observation
from isaac_bridge.transformer_obs import VisibleHistory,ENTITY_FIELDS
from isaac_bridge.transformer_policy import CombatTransformer

class RustObservationTest(unittest.TestCase):
    def test_live_monstro_observations_and_no_hidden_target_leak(self):
        exe=Path(__file__).resolve().parents[2]/'sim/target/release/examples/motion_trace.exe'
        proc=subprocess.Popen([str(exe)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,encoding='utf8')
        def command(s):
            proc.stdin.write(s+'\n');proc.stdin.flush();return json.loads(proc.stdout.readline())
        window=VisibleHistory(history=8,capacity=128)
        states=set();saw_projectiles=False
        try:
            command('monstro 42')
            for tick in range(1000):
                state=command('0 0')
                raw=rust_visible_observation(state)
                poisoned=copy.deepcopy(state)
                for b in poisoned['bosses']:
                    b['debug_target']=[-99999,88888];b['debug_state']=999
                self.assertEqual(raw,rust_visible_observation(poisoned))
                for b in state['bosses']:states.add(b['anim'])
                saw_projectiles|=bool(state['projectiles'])
                encoded=window.append(raw)
                self.assertTrue(window.space.contains(encoded))
                for i,e in enumerate(raw['entities']):
                    if e['type']==20:
                        n=min(tick,7)
                        self.assertEqual(encoded['entities'][n,i,ENTITY_FIELDS.index('shadow_valid')],1)
            self.assertTrue({'Walk','JumpUp','JumpDown','Taunt'}<=states)
            self.assertTrue(saw_projectiles)
            torch.set_num_threads(1)
            net=CombatTransformer(window.space,layers=1)
            result=net({k:torch.as_tensor(v[None]).float() for k,v in encoded.items()})
            self.assertEqual(tuple(result.shape),(1,256));self.assertTrue(torch.isfinite(result).all())
        finally:
            proc.stdin.close();self.assertEqual(proc.wait(timeout=10),0);proc.stdout.close()

if __name__=='__main__':unittest.main()
