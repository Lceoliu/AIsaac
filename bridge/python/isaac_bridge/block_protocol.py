"""Small shared control plane; observation/action pipes retain their existing ABI.

CF slots: 0 free, 2 captured, -1 denied by the two-generation bound, 3 released
by worker. A worker only reuses a released slot after actor acknowledges 0.
Injury jobs pin (episode, hurt-at) sources from selection through ring publication.
The single writer is the worker's qlock; epoch is a seqlock for actor retirement.
"""
import numpy as np
from multiprocessing import shared_memory
from .tok_cf import bank_rows


def decode_reference(record):
    return int(record[0]) | (int(record[1]) << 16)


def shapes(cfg):
    from .tok_sampler import TEACH_RING
    n=cfg.workers
    keep=(cfg.cf_branches+2)*max(cfg.cf_distill_steps,cfg.cf_wall_distill)+1
    return ((n,bank_rows(cfg)),(n,),(n,2*cfg.teacher_queue+4,2),(n,),(n,keep),(n,TEACH_RING))


def sizes(cfg):
    return [int(np.prod(shape))*8 for shape in shapes(cfg)]


def attach(names,cfg):
    blocks=[shared_memory.SharedMemory(name=name) for name in names[:len(shapes(cfg))]]
    arrays=[np.ndarray(shape,np.int64,buffer=block.buf) for shape,block in zip(shapes(cfg),blocks)]
    return blocks,arrays


def write_injury_jobs(arrays,worker,queued,active):
    jobs,epoch=arrays[2],arrays[3]
    items=list(queued)+list(active)
    if len(items)>jobs.shape[1]:
        raise RuntimeError('injury job pins exceed bounded queue plus active search')
    epoch[worker]+=1
    jobs[worker]=-1
    for i,item in enumerate(items):
        jobs[worker,i]=(int(item[5]),int(item[3]))
    epoch[worker]+=1
