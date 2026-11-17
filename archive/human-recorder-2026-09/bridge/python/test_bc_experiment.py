import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from isaac_bridge.bc_experiment import action_metrics,compare_evaluations,write_comparison_index
from isaac_bridge.evaluation import write_replay


class BCExperimentTests(unittest.TestCase):
    def test_change_metric_excludes_unknown_previous_and_masked_targets(self):
        target=np.array([[5,0,0],[10,1,0],[15,0,0],[20,0,0]])
        predicted=np.array([[0,0,0],[10,1,0],[0,1,0],[20,0,0]])
        masks=np.array([[1,1,0],[1,1,0],[1,1,0],[0,0,0]],bool)
        previous=np.array([[0,0,0,0],[5,0,0,1],[15,0,0,1],[0,0,0,1]])
        m=action_metrics(target,predicted,masks,np.ones((4,3)),previous)
        self.assertEqual(m['changed_action_samples'],1)
        self.assertEqual(m['changed_action_accuracy'],1)
        self.assertEqual(m['joint_labels'],3)
        self.assertEqual(m['bomb_true_positives'],1)
        self.assertEqual(m['bomb_predicted_positives'],2)

    def make_arm(self,path,outcomes,start_x=10):
        path.mkdir();results=[]
        for seed,outcome in enumerate(outcomes,2**31+16):
            rows=[dict(state=dict(hp=6,player=dict(pos=[start_x,20])),done=False,outcome='running'),
                  dict(state=dict(hp=5,player=dict(pos=[11,20])),done=True,outcome=outcome)]
            write_replay(path/f'seed-{seed}',dict(seed=seed),rows)
            results.append(dict(seed=seed,outcome=outcome,frames=60,layout=0,r=0,l=30,replay=f'seed-{seed}.jsonl.gz'))
        (path/'summary.json').write_text(json.dumps(dict(results=results)))

    def test_paired_counts_damage_and_replays(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=Path(tmp)/'a';b=Path(tmp)/'b'
            self.make_arm(a,['win','death']);self.make_arm(b,['time_limit','win'])
            r=compare_evaluations(a,b)
            self.assertEqual(r['gained_wins'],1);self.assertEqual(r['lost_wins'],1)
            self.assertEqual(r['paired_win_gain'],0)
            self.assertEqual(r['original']['mean_hurt_events'],1)
            self.assertEqual(r['original']['mean_health_lost'],1)
            self.assertEqual(r['behavior_cloned']['timeouts'],1)
            write_comparison_index(Path(tmp),r)
            html=(Path(tmp)/'index.html').read_text(encoding='utf8')
            self.assertIn('original/seed-2147483664.html',html)
            self.assertIn('behavior-cloned/seed-2147483665.html',html)

    def test_different_initial_room_cannot_pass_as_paired(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=Path(tmp)/'a';b=Path(tmp)/'b'
            self.make_arm(a,['win']);self.make_arm(b,['win'],start_x=99)
            with self.assertRaisesRegex(ValueError,'initial states differ'):
                compare_evaluations(a,b)


if __name__=='__main__':unittest.main()
