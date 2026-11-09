"""B51: the B44 fixed-input learner benchmark with the history scale chosen (history_scale none / unbiased: the
dumped model config predates the field, so the model's config is replaced after loading).

  python b51_scale_bench.py <learner_inputs.pt> none|unbiased [b44_learner_bench options]"""
import os
import sys
from dataclasses import replace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import b44_learner_bench as bench

MODE = sys.argv.pop(2)
_load = bench.load_model


def load_model(ck, dev):
    model, opt = _load(ck, dev)
    model.block_memory.config = replace(model.block_memory.config, history_scale=MODE)
    return model, opt


bench.load_model = load_model
if __name__ == '__main__':
    bench.main()
