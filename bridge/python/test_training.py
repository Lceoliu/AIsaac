import json
import unittest
from pathlib import Path
from unittest.mock import patch, Mock

from isaac_bridge.training import IsaacTrainingEnv
from isaac_bridge.env import BridgeError


class Transport:
    def __init__(self):
        self.data = b""
        self.push({"type": "hello", "version": "0.2.0"})
        self.push({"type": "obs", "event": "step", "seq": 1, "obs": {"game_frame": 77}})

    def push(self, value):
        self.data += (json.dumps(value) + "\n").encode()

    def setsockopt(self, *args): pass
    def settimeout(self, *args): pass
    def close(self): pass

    def recv(self, n):
        data, self.data = self.data[:n], self.data[n:]
        return data

    def sendall(self, data):
        command = json.loads(data)
        if command["cmd"] == "reset":
            self.push({"type": "obs", "event": "reset", "seq": 2, "obs": {"game_frame": 2}})
        elif command["cmd"] == "step":
            self.push({"type": "obs", "event": "step", "seq": 3, "obs": {"game_frame": 6}})


class TrainingProtocolTest(unittest.TestCase):
    def test_room_checkpoint_is_initialized_once_then_rewound(self):
        raw = json.loads((Path(__file__).parent/'fixtures/monstro_initial_obs.json').read_text(encoding='utf-8'))
        info = {'start_seed':'9AM0 7PRP','room_spawn_seed':2979258163}
        env = IsaacTrainingEnv()
        env.reset = Mock(return_value=(raw, info))
        env.exec = Mock()
        env.step = Mock(return_value=(raw,0,False,False,info))
        self.assertEqual(env.reset_monstro()[1]['reset_kind'],'initialize')
        self.assertEqual(env.reset_monstro()[1]['reset_kind'],'rewind')
        calls = [c.kwargs['phases'] for c in env.reset.call_args_list]
        self.assertEqual(len(calls),3)
        self.assertEqual(calls[0][0],['restart 0'])
        self.assertEqual(calls[0][1][0],'remove *')
        self.assertEqual(calls[1:],[[['rewind']],[['rewind']]])
        env.reset.return_value = ({'room': {'alive': 0}}, info)
        env.reset_safe()
        self.assertEqual(env.reset.call_args.kwargs['phases'],[['rewind']])

    def test_seeded_reset_regenerates_stage_and_reads_actual_seed(self):
        env = IsaacTrainingEnv()
        env.reset = Mock(return_value=({"room": {"alive": 0}}, {"event": "reset"}))
        env.query_info = Mock(return_value={"start_seed": "9AM0 7PRP"})
        _, info = env.reset_safe()
        self.assertEqual(info["start_seed"], "9AM0 7PRP")
        self.assertEqual(env.reset.call_args.kwargs["phases"], [
            ["restart 0"], ['lua Game():GetSeeds():SetStartSeed("9AM0 7PRP")', "stage 1"]])

    def test_seed_mismatch_is_not_reported_as_fixed_scene(self):
        env = IsaacTrainingEnv()
        env.reset = Mock(return_value=({}, {}))
        env.query_info = Mock(return_value={"start_seed": "9YLJ WX90"})
        with self.assertRaises(BridgeError): env._reset_seeded_start("9AM0 7PRP")

    def test_initial_observation_does_not_shift_responses(self):
        env = IsaacTrainingEnv()
        with patch("isaac_bridge.env.socket.create_connection", return_value=Transport()):
            env.connect()
        self.assertEqual(env.last_obs["game_frame"], 77)
        obs, info = env.reset()
        self.assertEqual((obs["game_frame"], info["event"], info["seq"]), (2, "reset", 2))
        obs, _, _, _, info = env.step({}, repeat=4)
        self.assertEqual((obs["game_frame"], info["event"], info["seq"]), (6, "step", 3))
        env.close()


if __name__ == "__main__":
    unittest.main()
