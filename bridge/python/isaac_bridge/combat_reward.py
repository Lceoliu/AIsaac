"""Versioned training objective; simulator physics and native ABI stay unchanged."""
import numpy as np

REWARD_PROFILES=('legacy','combat-v1')
DEADLINE_FRAMES=3600


def combat_v1_reward(reward,outcome,elapsed):
    """Replace native +1 win with +3 plus [0,1] speed bonus; timeout -1."""
    reward=np.asarray(reward,dtype=np.float32);outcome=np.asarray(outcome)
    speed=np.maximum(np.float32(0),np.float32(1)-np.asarray(elapsed,dtype=np.float32)/np.float32(DEADLINE_FRAMES))
    return reward+np.where(outcome==2,np.float32(2)+speed,np.float32(0))-(outcome==3).astype(np.float32)
