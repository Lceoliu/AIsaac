"""Block-memory adapter for the existing search lane and V2 ring protocol.

Search completion must call release_point *after* publishing its winning records.
The same two 16-bit reference halves used by the earlier rebuild-memory prototype
travel through lane_t; actual immutable snapshots stay in the actor registry and
are returned by TokSampler.v2_records(with_block=True).
"""
import numpy as np
import torch

from .block_memory import TeacherGenerations
from .tok_policy import crn_uniforms
from .block_training import BlockActor
from .tok_lanes import LaneService
from .block_protocol import decode_reference
from .block_nvtx import push as nv_push, pop as nv_pop


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
        nv_push('lane.control')
        try:
            return self._service_control()
        finally:
            nv_pop()

    def _service_control(self):
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

    def tick(self):
        self.service_control()
        return super().tick()

    def step_main(self, rows, pending, slots, on_main_ready=None):
        """Dispatch ready main/search rows together only at the same weight version.

        Owner-local slots/snapshots and lane CRN keys survive the merged batch.
        Reply to main immediately after its forward, then finish old generations
        and replay requests while main workers run the next game step.
        """
        self.main_request=(rows,pending,list(slots),on_main_ready)
        self.s.poll_lanes()
        if self.s.lane_ready:
            self.submit()
            self.finish()
        if self.main_request is not None:
            self._step_main_only()

    def _reply_main(self):
        callback=self.main_request[3]
        self.main_request=None
        if callback is not None:
            nv_push('lane.reply_main')
            callback(self.main_actor.host_output,self.main_actor.last_contexts)
            nv_pop()

    def _step_main_only(self):
        rows,pending,slots,_=self.main_request
        with torch.autocast(torch.device(self.device).type,dtype=torch.bfloat16,
                            enabled=self.main_actor.bf16):
            self.main_actor.step(rows,pending,slots)
        self._reply_main()

    def serve_wait(self,timeout):
        self.service_control()
        return super().serve_wait(timeout)

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
        main request when versions match --, which replays a step graph when it can. B46 counters:
        f['eager_calls'] (steps on the eager path), lane rows merged / graph / eager, seconds."""
        import time as time_
        from .block_paged import ucount
        t_call = time_.perf_counter()
        nv_push('lane.eager')
        f['eager_calls'] = 0
        n = len(idx)
        a = pend.copy()
        lp, value, vg = (np.zeros(n, np.float32) for _ in range(3))
        versions = np.full(n,-1,np.int64)
        cancelled=np.zeros(n,bool)
        refs = np.zeros((n, self.s.lane_t.shape[1]), np.float32)
        groups = {}
        replays=np.zeros(n,bool)
        for j, lane in enumerate(idx):
            lane=int(lane);seq=int(f['seq'][j])
            if self.cancelled_requests.get(lane)==seq:
                cancelled[j]=True
                continue
            if lane in self.last_answers and self.last_answers[lane][0]==seq:
                _,a[j],lp[j],value[j],vg[j],versions[j],refs[j],cancelled[j]=self.last_answers[lane]
                continue
            point = self.lane_point[int(lane)]
            generation = self.points[point][1]
            replays[j]=bool(self.s.block_arrays[1][lane]) if self.main_actor is not None else False
            groups.setdefault(generation, []).append(j)
        group_items=list(groups.items())
        if self.main_request is not None:
            matching=[generation for generation in groups
                      if self.actors[generation].version==self.main_actor.version]
            if matching:
                group_items.sort(key=lambda item: item[0]!=matching[0])
            else:
                self._step_main_only()
        for generation, positions in group_items:
            positions = np.asarray(positions, np.int64)
            lanes = idx[positions]
            replay=replays[positions]
            actor = self.actors[generation]
            keys = None if ukeys is None else (ukeys[0][positions], ukeys[1][positions])
            with torch.autocast(torch.device(self.device).type,dtype=torch.bfloat16,enabled=actor.bf16):
                if self.main_request is not None and actor.version==self.main_actor.version:
                    mr,mp,ms,_=self.main_request
                    dev=torch.device(self.device)
                    # Main rows get fresh random uniforms; branch rows retain the
                    # original (request sequence, lane ID) CRN, not batch positions.
                    # (B46: the uniforms on the host: the step graph's input slab is host memory)
                    main_u=torch.rand((len(mr),sum(actor.model.heads)))
                    if keys is not None:
                        lane_u=torch.as_tensor(crn_uniforms(*keys,sum(actor.model.heads)))
                    else:
                        lane_u=torch.full((len(lanes),sum(actor.model.heads)),.5)
                        lane_u[np.flatnonzero(~replay).tolist()]=torch.rand((int((~replay).sum()),sum(actor.model.heads)))
                    self.main_actor.step(np.concatenate((mr,rows[positions])),
                        np.concatenate((mp,pend[positions])),ms+list(lanes),
                        owners=[self.main_actor]*len(mr)+[actor]*len(lanes),
                        uniform=torch.cat((main_u,lane_u)),
                        evaluate=np.concatenate((np.zeros(len(mr),bool),replay)))
                    t_ans=time_.perf_counter()
                    self._reply_main()
                    ucount('lane_answer_s',time_.perf_counter()-t_ans)
                    ucount('lane_rows_merged',len(lanes))
                    self.stat['merged_main_rows']=self.stat.get('merged_main_rows',0)+len(mr)
                    self.stat['merged_lane_rows']=self.stat.get('merged_lane_rows',0)+len(lanes)
                    path=getattr(self.main_actor,'last_path','eager')
                else:
                    actor.step(rows[positions], pend[positions], lanes, ukeys=keys,evaluate=replay)
                    path=getattr(actor,'last_path','eager')
                f['eager_calls']+=int(path=='eager')
                ucount('lane_rows_graph' if path=='graph' else 'lane_rows_eager',len(lanes))
            self.stat['block_teacher_forward_calls']=self.stat.get('block_teacher_forward_calls',0)+1
            self.stat['block_mixed_replay_calls']=self.stat.get('block_mixed_replay_calls',0)+int(replay.any() and not replay.all())
            host=actor.host_output
            aa,pp,vv,gg=host[:,:self.nh],host[:,self.nh],host[:,self.nh+1],host[:,self.nh+2]
            a[positions], lp[positions], value[positions], vg[positions] = aa.astype(np.int64), pp, vv, gg
            versions[positions] = self.generations.versions[generation]
            for k, j in enumerate(positions):
                lane = int(idx[j])
                point = self.lane_point[lane]
                token = self.points[point][3]
                if not replays[j] and (self.main_actor is not None or self.lane_steps[lane] < self.distill_steps):
                    ident = self.contexts.put(token, actor.last_contexts[k])
                    refs[j, 0], refs[j, 1] = ident & 65535, ident >> 16
                    refs[j, -1] = 1.0
                self.lane_steps[lane] += 1
        f.update(a=a, lp=lp, v=value, vg=vg, version=versions, block_refs=refs,cancelled=cancelled)
        nv_pop()
        ucount('lane_call_s',time_.perf_counter()-t_call);ucount('lane_calls',1);ucount('lane_rows',n)

    def finish(self):
        nv_push('lane.finish')
        try:
            return self._finish()
        finally:
            nv_pop()

    def _finish(self):
        if self.fly is not None:
            f=self.fly
            for j,lane in enumerate(f['idx']):
                self.last_answers[int(lane)]=(int(f['seq'][j]),f['a'][j].copy(),f['lp'][j],f['v'][j],f['vg'][j],
                                              f['version'][j],f['block_refs'][j].copy(),f['cancelled'][j])
            self.s.lane_t[self.fly["idx"]] = self.fly["block_refs"]
            self.s.lane_out[self.fly['idx'],5]=-self.fly['cancelled'].astype(np.int64)
        return super().finish()

