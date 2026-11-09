"""Block-memory adapter for the existing search lane and V2 ring protocol.

Search completion must call release_point *after* publishing its winning records.
The same two 16-bit reference halves used by the earlier rebuild-memory prototype
travel through lane_t; actual immutable snapshots stay in the actor registry and
are returned by TokSampler.v2_records(with_block=True).
"""
import os
import time

import numpy as np
import torch

from .block_memory import TeacherGenerations
from .tok_policy import crn_uniforms
from .block_training import BlockActor
from .tok_lanes import LaneService
from .block_protocol import decode_reference

# B47: the actor's step graph calls stop blocking the actor thread (each switch measured alone):
# LANES_ASYNC: a lane round's steps are launched and the round answered at a later pass (after the
#   next main launch / poll), not before step_main returns;
# MAIN_ASYNC: the main (or merged) step is launched and answered after the next main launch
#   (pipelining: its GPU time overlaps the next call's host work);
# LANE_BATCH: a round's steps (both generations, main-only) launched before the round's one
#   synchronisation (implied by the two above).
LANES_ASYNC = os.environ.get('ISAAC_B47_LANES_ASYNC', '0') == '1'
MAIN_ASYNC = os.environ.get('ISAAC_B47_MAIN_ASYNC', '0') == '1'
LANE_BATCH = LANES_ASYNC or MAIN_ASYNC or os.environ.get('ISAAC_B47_LANE_BATCH', '0') == '1'
# B47 coalescing (with LANES_ASYNC / MAIN_ASYNC): a lane-only round (tick / serve_wait, no main
# request to merge into) starts only when its oldest request waited LANE_HOLD s or every lane
# active lately is waiting (the quiet rule off); step_main still takes every waiting lane into the
# main call. Without it the async rounds were 31% more and smaller (prof-b47-i1).
LANE_HOLD = float(os.environ.get('ISAAC_B47_LANE_HOLD', '0.004'))


class _Done:
    """A completed step of an actor without deferred steps (BlockActor): its owners' host outputs."""
    done = True

    def __init__(self, owners):
        self.out = {id(o): (o.host_output, o.last_contexts) for o in owners}

    def output(self, owner):
        return self.out[id(owner)]


def _launch(actor, *args, owners=None, **kw):
    """B47: actor.step launched (a StepTicket) when the actor defers steps, else run (a _Done)."""
    if hasattr(actor, 'complete'):
        return actor.step(*args, owners=owners, defer=True, **kw)
    actor.step(*args, owners=owners, **kw)
    return _Done(dict.fromkeys(owners) if owners is not None else [actor])


def _finish(actor, ticket):
    if not ticket.done:
        actor.complete(ticket)


class ContextRegistry:
    def __init__(self):
        self.next_id = 1
        self.values = {}
        self.owners = {}
        self.completed = set()

    def put(self, owner, snapshot):
        ident = self.next_id
        if ident >= 2**32:
            raise OverflowError("search context reference space exhausted; restart the actor")
        self.next_id += 1
        self.values[ident], self.owners[ident] = snapshot, owner
        return ident

    def get(self, ident):
        # Missing/raw-less contexts must not become silent empty-history teachers.
        return self.values[ident]

    def complete(self, owner):
        self.completed.add(owner)

    def retire(self, live, transient=False):
        dead = [ident for ident, owner in self.owners.items()
                if (transient or owner in self.completed) and ident not in live]
        for ident in dead:
            del self.values[ident]
            del self.owners[ident]
        self.completed.intersection_update(self.owners.values())


class BlockLaneService(LaneService):
    RECLAIM_S=0.1   # B46: at most this often the context / injury-source reclamation (service_control)
    def __init__(self, sampler, model, device, wait=.002, quiet=.0005, distill_steps=8,main_actor=None,stat=None):
        super().__init__(sampler, model, device, len(model.heads), gtok=True,
                         wait=wait, quiet=quiet, async_=False,stat=stat)
        self.generations = TeacherGenerations()
        self.points = {}
        self.point_serial = 0
        self.lane_point = {}
        self.lane_steps = {}
        self.actors = {}
        self.distill_steps = int(distill_steps)
        sampler.block_contexts = self.contexts = ContextRegistry()
        sampler.block_lane_service=self
        self.main_actor=main_actor
        self.main_request=None
        self.injury_sources={}
        self.recent={}
        self.hurt_candidates={}
        self.episode={}
        self.cancelled_requests={}
        self.last_answers={}
        self.work=[]                  # (B47: launched lane rounds / main answers, launch order)
        self.lanes_in_flight=set()
        self.lanes_async,self.main_async,self.lane_batch=LANES_ASYNC,MAIN_ASYNC,LANE_BATCH
        self.lane_hold=LANE_HOLD
        self._origin='wait'   # (B47 counters: where the next round starts)
        if main_actor is not None:
            self.depths=tuple(sorted(set(sampler.cfg.teacher_depths+sampler.cfg.teacher_death_depths)))
            self.recent_size=max(self.depths,default=0)+2

    def observe_rows(self,workers,rows):
        """Called before Episodes.arrive, including terminal records (no action)."""
        for worker,row in zip(workers,rows):
            worker=int(worker); episode=int(row['episode']); step=int(row['t'])
            if row['first']:
                self.episode[worker]=episode
                self.recent[worker]=[]
                self.hurt_candidates[worker]=set()
            if row['hurt']>0:
                for depth in self.depths:
                    key=(worker,episode,step-1-depth)
                    if key[2]>=0 and key in self.injury_sources:
                        self.hurt_candidates.setdefault(worker,set()).add(key)

    def record_main(self,workers,rows,contexts):
        for worker,row,context in zip(workers,rows,contexts):
            worker=int(worker)
            key=(worker,int(row['episode']),int(row['t']))
            self.injury_sources[key]=context
            recent=self.recent.setdefault(worker,[])
            recent.append(key)
            del recent[:-self.recent_size]

    def injury_context(self,worker,episode,step):
        return self.injury_sources[(int(worker),int(episode),int(step))]

    def capture_main(self,workers,rows,pending):
        self.service_control()
        status=self.s.block_arrays[0]
        for i,worker in enumerate(workers):
            worker=int(worker)
            self.main_actor.pending[worker]=pending[i:i+1].copy()
            request=int(self.s.cf_snap[worker])
            if request:
                point=request-1
                snap=self.main_actor.snapshot(worker)
                accepted=self.capture_point(point,snap,self.model,self.main_actor.version)
                status.flat[point]=2 if accepted else -1
                self.s.cf_snap[worker]=0

    def service_control(self):
        if self.main_actor is None:
            return
        from .tok_sampler import ST,TEACH_RING
        status,mode,jobs,epoch,kept=self.s.block_arrays[:5]
        self.s.poll_lanes()
        for point in np.flatnonzero(status.reshape(-1)==3):
            worker=int(point)//status.shape[1]
            begin=int(self.s.lane_io[worker,11])
            requested=begin-1 if begin>0 else self.lane_point.get(worker)
            if requested==int(point):
                self.cancelled_requests[worker]=int(self.s.lane_io[worker,10])
            if self.s.cf_snap[worker]==int(point)+1:
                self.s.cf_snap[worker]=0
            if int(point) in self.points:
                self.release_point(int(point))
            status.flat[point]=0
        # B46: the context / injury-source reclamation below scans every worker's rings (~3 ms);
        # it only frees memory, so it runs at most every RECLAIM_S seconds (cancellations above:
        # every call).
        import time as time_
        now=time_.perf_counter()
        if now-getattr(self,'_reclaimed_t',-1e9)<self.RECLAIM_S:
            return
        self._reclaimed_t=now
        # Read kept refs BEFORE rings: worker publishes ring records before
        # clearing these pins, so neither side of the transfer can disappear.
        live_context=set(map(int,kept.reshape(-1).copy()))
        for records in (self.s.lane_t,self.s.v2_t):
            pairs=np.array(records[...,:2],dtype=np.uint32,copy=True).reshape(-1,2)
            live_context.update((pairs[:,0] | (pairs[:,1]<<16)).tolist())
        self.contexts.retire(live_context,transient=True)
        # Take a stable view of worker queue/active pins. If a writer is in its
        # short critical section, defer reclamation, never the main actor.
        before=epoch.copy()
        if np.any(before%2):
            return
        pins=jobs.copy()
        if not np.array_equal(before,epoch):
            return
        live=set(key for keys in self.recent.values() for key in keys)
        live.update(key for keys in self.hurt_candidates.values() for key in keys)
        for worker in range(self.s.cfg.workers):
            for episode,hurt_at in pins[worker]:
                if episode>=0:
                    live.update((worker,int(episode),int(hurt_at)-1-d) for d in self.depths)
            count=int(self.s.stats[worker,ST['teach']])
            serial=np.arange(max(0,count-TEACH_RING),count,dtype=np.int64)
            slots=serial%TEACH_RING
            commit=self.s.block_arrays[5][worker,slots].copy()
            # Read only two metadata columns, never gather the large full ROW.
            episodes=self.s.teach_rows['episode'][worker,slots].copy()
            steps=self.s.teach_rows['t'][worker,slots].copy()
            valid=(commit==serial+1)&(self.s.block_arrays[5][worker,slots]==commit)
            live.update((worker,e,t) for e,t in zip(episodes[valid].tolist(),steps[valid].tolist()))
        for key in self.injury_sources.keys()-live:
            del self.injury_sources[key]

    # ---------------------------------------------------------------- B47: deferred work
    @property
    def in_flight(self):
        return bool(self.work)

    def idle(self, default):
        """The workers' poll wait: none while a main answer is launched, up to 1 ms while only lane
        rounds are (the next pass answers them; no busy loop beside the learner's thread)."""
        if self.work:
            return 0.0 if any(item['kind']=='main' for item in self.work) else min(default,0.001)
        if (self.lanes_async or self.main_async) and self.s.lane_ready:   # (the coalescing window)
            wait=max(self.wait,self.lane_hold)
            return min(default,max(0.0,wait-(time.perf_counter()-self.s.lane_first_t)))
        return super().idle(default)

    def due(self, now=None):
        """A lane-only round's window (B47: with async rounds, coalesced: LANE_HOLD, no quiet rule)."""
        if not (self.lanes_async or self.main_async):
            return super().due(now)
        from .tok_lanes import window_due, ACTIVE_S
        ready=self.s.lane_ready
        if not ready:
            return False
        now=time.perf_counter() if now is None else now
        first_t=getattr(self.s,'lane_first_t',0.0)
        last_t=getattr(self.s,'lane_last_t',0.0)
        wait=max(self.wait,self.lane_hold)
        if window_due(1,first_t,last_t,now,wait,0.0):
            return True
        rd=np.unique(np.asarray(ready,np.int64))
        recent=self.seen>=now-ACTIVE_S
        recent[rd]=True
        return window_due(len(rd),first_t,last_t,now,wait,0.0,int(recent.sum()))

    def _done(self, item):
        return all(t.done or t.g['set']['event'].query() for t in item['tickets'])

    def _complete(self, item):
        self.work.remove(item)
        if item['kind']=='main':
            self._answer_main(item)
        else:
            self._complete_round(item)

    def complete_all(self):
        """Every launched round / main answer completed (launch order)."""
        did=bool(self.work)
        while self.work:
            self._complete(self.work[0])
        return did

    def _settle(self, before):
        """After a step_main / tick launch: the items launched before it completed; the new ones
        too unless their switch keeps them in flight; any whose GPU work is done answered."""
        from .block_paged import ucount
        old=self.work[:before]
        for item in list(self.work):
            new=not any(item is o for o in old)
            keep=new and ((item['kind']=='main' and self.main_async) or (item['kind']=='lanes' and self.lanes_async))
            if not keep:
                self._complete(item)
        for item in list(self.work):   # (the kept ones whose GPU work is done)
            if self._done(item):
                self._complete(item)
            elif item['kind']=='main':
                ucount('main_deferred',1)
            else:
                ucount('lane_rounds_deferred',1)

    def tick(self):
        self.service_control()
        did=self.complete_all()
        self.s.poll_lanes()
        if self.s.lane_ready and self.due():
            before=len(self.work)
            self._origin='tick'
            self.submit()
            self._origin='wait'
            self._settle(before)
            did=True
        return did

    def serve_wait(self,timeout):
        self.service_control()
        if self.complete_all():
            return True
        return super().serve_wait(timeout)

    def quiesce(self):
        self.complete_all()

    def step_main(self, rows, pending, slots, on_main_ready=None):
        """Dispatch ready main/search rows together only at the same weight version.

        Owner-local slots/snapshots and lane CRN keys survive the merged batch.
        Reply to main immediately after its forward, then finish old generations
        and replay requests while main workers run the next game step. B47: the steps are
        launched; the work launched before this call is answered after it (pipelining), the new
        work too unless MAIN_ASYNC / LANES_ASYNC keep it for a later pass."""
        before=len(self.work)
        self.main_request=(rows,pending,list(slots),on_main_ready)
        self.s.poll_lanes()
        if self.s.lane_ready:
            self._origin='main'
            self.submit()
            self._origin='wait'
        if self.main_request is not None:
            self._step_main_only()
        self._settle(before)

    def _answer_main(self, item):
        ticket=item['tickets'][0]
        _finish(self.main_actor,ticket)
        host,contexts=ticket.output(self.main_actor)
        if item['callback'] is not None:
            item['callback'](host,contexts)

    def _reply_main(self, ticket):
        """The main request launched (alone or merged): answered when its step completes."""
        callback=self.main_request[3]
        self.main_request=None
        item=dict(kind='main',tickets=[ticket],callback=callback)
        if self.main_async:
            self.work.append(item)
        else:   # (B46 order: the main answer before anything else is launched)
            self._answer_main(item)

    def _step_main_only(self):
        rows,pending,slots,_=self.main_request
        with torch.autocast(torch.device(self.device).type,dtype=torch.bfloat16,
                            enabled=self.main_actor.bf16):
            ticket=_launch(self.main_actor,rows,pending,slots)
        self._reply_main(ticket)

    def submit(self):
        """B47: every waiting lane request taken and its round launched (answered by _settle /
        complete_all). A lane whose previous round is still in flight (a cancelled request's lane
        carrying the next one) completes everything first."""
        from .block_paged import ucount
        s=self.s
        idx=s.take_lanes()
        if not len(idx):
            return False
        if self.lanes_in_flight.intersection(idx.tolist()):
            self.complete_all()
            ucount('lane_rounds_forced',1)
        t0=time.perf_counter()
        n=len(idx)
        self.stat['held_s']+=max(0.0,t0-getattr(s,'lane_first_t',t0))
        self.seen[idx]=t0
        rows=s.lane_rows[idx]
        pend=s.lane_io[idx,:self.nh].astype(np.int64)
        seq=s.lane_io[idx,10].astype(np.int64) if s.lane_io.shape[1]>10 else np.zeros(n,np.int64)
        ukeys=(seq,idx) if self.ukeys else None
        f=dict(idx=idx,rows=rows,seq=seq,version=self.version(),t0=t0,ready=False)
        self._begin(idx)
        self._eager(f,rows,pend,idx,ukeys)
        self.lanes_in_flight.update(idx.tolist())
        self.work.append(dict(kind='lanes',f=f,tickets=[g[2] for g in f['groups']]))
        self.stat['eager']+=f.get('eager_calls',0)
        self.stat['s']+=time.perf_counter()-t0
        return True

    def capture_point(self, bank_index, snapshot, actor_model, model_version):
        """Before consuming the selection-point observation. False = backpressure."""
        if bank_index in self.points:
            raise RuntimeError("release a search point before reusing its bank slot")
        if int(model_version) != snapshot.policy_version:
            raise ValueError("point snapshot and supplied actor weights have different versions")
        if not self.generations.publish(actor_model, snapshot.policy_version):
            return False
        generation, version, teacher = self.generations.acquire(snapshot.policy_version)
        self.point_serial += 1
        token = self.point_serial
        self.points[int(bank_index)] = (snapshot.fork(), generation, version, token)
        # A recycled generation's policy object is reused, but all branch state is
        # restored explicitly at begin; no preceding candidate's state is inherited.
        if generation not in self.actors and hasattr(self.main_actor,'lane_actor'):
            # A1: teacher generations share the main actor's page store; branch state is a slot row.
            # (B46: generation g's lanes in the main actor's arena: merged calls replay its step graph)
            self.actors[generation] = self.main_actor.lane_actor(teacher, len(self.s.lane_io), generation=generation)
        if generation not in self.actors:
            self.actors[generation] = BlockActor(teacher, len(self.s.lane_io),
                graph_entities=self.main_actor.graph_entities if self.main_actor is not None else 0,
                bf16=self.main_actor.bf16 if self.main_actor is not None else False)
        return True

    def release_point(self, bank_index):
        snapshot, generation, version, token = self.points.pop(int(bank_index))
        self.generations.release(generation)
        self.contexts.complete(token)

    def _begin(self, idx):
        starts = self.s.lane_io[idx, 11].astype(np.int64)
        for lane, begin in zip(idx, starts):
            lane = int(lane)
            seq=int(self.s.lane_io[lane,10])
            if self.cancelled_requests.get(lane)==seq:
                self.s.lane_io[lane,11]=0
                continue
            if lane in self.last_answers and self.last_answers[lane][0]==seq:
                continue
            if begin == 0:
                continue
            if begin < 0:
                raise ValueError("block-memory search requires an explicit point snapshot")
            point = int(begin) - 1
            snapshot, generation, version, token = self.points[point]
            actor = self.actors[generation]
            actor.version = version
            actor.restore(lane, snapshot)
            self.lane_point[lane] = point
            self.lane_steps[lane] = 0
            self.s.lane_io[lane, 11] = 0

    def _eager(self, f, rows, pend, idx, ukeys):
        """The lane round (block memory): per generation one SlotBlockActor.step -- merged with the
        main request when versions match --, which replays a step graph when it can. B47: the steps
        are launched (StepTicket); _complete_round answers the round. Counters: f['eager_calls']
        (steps on the eager path), lane rows merged / graph / eager."""
        from .block_paged import ucount
        f['t_call']=time.perf_counter()
        f['eager_calls']=0
        n=len(idx)
        a=pend.copy()
        lp,value,vg=(np.zeros(n,np.float32) for _ in range(3))
        versions=np.full(n,-1,np.int64)
        cancelled=np.zeros(n,bool)
        refs=np.zeros((n,self.s.lane_t.shape[1]),np.float32)
        groups={}
        replays=np.zeros(n,bool)
        tokens=np.zeros(n,np.int64)
        for j,lane in enumerate(idx):
            lane=int(lane);seq=int(f['seq'][j])
            if self.cancelled_requests.get(lane)==seq:
                cancelled[j]=True
                continue
            if lane in self.last_answers and self.last_answers[lane][0]==seq:
                _,a[j],lp[j],value[j],vg[j],versions[j],refs[j],cancelled[j]=self.last_answers[lane]
                continue
            point=self.lane_point[int(lane)]
            generation=self.points[point][1]
            tokens[j]=self.points[point][3]
            replays[j]=bool(self.s.block_arrays[1][lane]) if self.main_actor is not None else False
            groups.setdefault(generation,[]).append(j)
        group_items=list(groups.items())
        n_rows=sum(len(v) for v in groups.values())
        if self.main_request is not None:
            matching=[generation for generation in groups
                      if self.actors[generation].version==self.main_actor.version]
            if matching:
                group_items.sort(key=lambda item: item[0]!=matching[0])
                ucount('lane_rounds_main_merged',1);ucount('lane_rows_main_merged',n_rows)
            else:
                self._step_main_only()
                ucount('lane_rounds_main_other',1);ucount('lane_rows_main_other',n_rows)
        else:
            ucount('lane_rounds_'+self._origin,1);ucount('lane_rows_'+self._origin,n_rows)
        launched=[]
        for generation,positions in group_items:
            positions=np.asarray(positions,np.int64)
            lanes=idx[positions]
            replay=replays[positions]
            actor=self.actors[generation]
            keys=None if ukeys is None else (ukeys[0][positions],ukeys[1][positions])
            with torch.autocast(torch.device(self.device).type,dtype=torch.bfloat16,enabled=actor.bf16):
                if self.main_request is not None and actor.version==self.main_actor.version:
                    mr,mp,ms,_=self.main_request
                    # Main rows get fresh random uniforms; branch rows retain the
                    # original (request sequence, lane ID) CRN, not batch positions.
                    # (B46: the uniforms on the host: the step graph's input slab is host memory)
                    main_u=torch.rand((len(mr),sum(actor.model.heads)))
                    if keys is not None:
                        lane_u=torch.as_tensor(crn_uniforms(*keys,sum(actor.model.heads)))
                    else:
                        lane_u=torch.full((len(lanes),sum(actor.model.heads)),.5)
                        lane_u[np.flatnonzero(~replay).tolist()]=torch.rand((int((~replay).sum()),sum(actor.model.heads)))
                    ticket=_launch(self.main_actor,np.concatenate((mr,rows[positions])),
                        np.concatenate((mp,pend[positions])),ms+list(lanes),
                        owners=[self.main_actor]*len(mr)+[actor]*len(lanes),
                        uniform=torch.cat((main_u,lane_u)),
                        evaluate=np.concatenate((np.zeros(len(mr),bool),replay)))
                    self._reply_main(ticket)
                    ucount('lane_rows_merged',len(lanes))
                    self.stat['merged_main_rows']=self.stat.get('merged_main_rows',0)+len(mr)
                    self.stat['merged_lane_rows']=self.stat.get('merged_lane_rows',0)+len(lanes)
                    path=getattr(self.main_actor,'last_path','eager')
                else:
                    ticket=_launch(actor,rows[positions],pend[positions],lanes,ukeys=keys,evaluate=replay)
                    path=getattr(actor,'last_path','eager')
                f['eager_calls']+=int(path=='eager')
                ucount('lane_rows_graph' if path=='graph' else 'lane_rows_eager',len(lanes))
            if not self.lane_batch:   # (B46 order: each step synchronised before the next launch)
                _finish(actor,ticket)
            launched.append((generation,positions,ticket,actor,self.generations.versions[generation],replay))
            self.stat['block_teacher_forward_calls']=self.stat.get('block_teacher_forward_calls',0)+1
            self.stat['block_mixed_replay_calls']=self.stat.get('block_mixed_replay_calls',0)+int(replay.any() and not replay.all())
        f.update(a=a,lp=lp,v=value,vg=vg,version=versions,block_refs=refs,cancelled=cancelled,groups=launched,
                 replays=replays,tokens=tokens)
        ucount('lane_call_s',time.perf_counter()-f['t_call'])

    def _complete_round(self, item):
        """B47: a launched lane round's steps completed, the answers into the lane blocks."""
        from .block_paged import ucount
        t0=time.perf_counter()
        f=item['f']
        idx=f['idx']
        a,lp,value,vg,refs=f['a'],f['lp'],f['v'],f['vg'],f['block_refs']
        for generation,positions,ticket,actor,version,replay in f['groups']:
            _finish(actor,ticket)
            host,contexts=ticket.output(actor)
            aa,pp,vv,gg=host[:,:self.nh],host[:,self.nh],host[:,self.nh+1],host[:,self.nh+2]
            a[positions],lp[positions],value[positions],vg[positions]=aa.astype(np.int64),pp,vv,gg
            f['version'][positions]=version
            for k,j in enumerate(positions):
                lane=int(idx[j])
                if not f['replays'][j] and (self.main_actor is not None or self.lane_steps[lane]<self.distill_steps):
                    ident=self.contexts.put(int(f['tokens'][j]),contexts[k])
                    refs[j,0],refs[j,1]=ident&65535,ident>>16
                    refs[j,-1]=1.0
                self.lane_steps[lane]+=1
        f['ready']=True
        self.lanes_in_flight.difference_update(idx.tolist())
        self.fly=f
        self.finish()
        ucount('lane_call_s',time.perf_counter()-t0);ucount('lane_calls',1);ucount('lane_rows',len(idx))

    def finish(self):
        if self.fly is not None:
            f=self.fly
            for j,lane in enumerate(f['idx']):
                self.last_answers[int(lane)]=(int(f['seq'][j]),f['a'][j].copy(),f['lp'][j],f['v'][j],f['vg'][j],
                                              f['version'][j],f['block_refs'][j].copy(),f['cancelled'][j])
            self.s.lane_t[self.fly["idx"]] = self.fly["block_refs"]
            self.s.lane_out[self.fly['idx'],5]=-self.fly['cancelled'].astype(np.int64)
        return super().finish()

