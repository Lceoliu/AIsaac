"""Block-memory adapter for the existing search lane and V2 ring protocol.

Search completion must call release_point *after* publishing its winning records.
The same two 16-bit reference halves used by the earlier rebuild-memory prototype
travel through lane_t; actual immutable snapshots stay in the actor registry and
are returned by TokSampler.v2_records(with_block=True).
"""
import os

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


class NativePoints:
    """(B58c) BlockLaneService.points over the native point table (the driver captures): point -> (snapshot, generation,
    version, token), the snapshot materialized once per point (a branch restores from it)."""

    def __init__(self, store):
        self.store = store
        self.cache = {}

    def __contains__(self, point):
        return self.store.native.pt_get(int(point)) is not None

    def __getitem__(self, point):
        point = int(point)
        hit = self.cache.get(point)
        got = self.store.native.pt_get(point)
        if got is None:
            self.cache.pop(point, None)
            raise KeyError(point)
        snap, generation, version, token = (int(x) for x in got)
        if hit is not None and hit[3] == token:
            return hit
        from .block_paged import native_snapshots
        value = (native_snapshots(self.store, [snap])[0], generation, version, token)
        self.cache[point] = value
        return value

    def __len__(self):
        return len(self.store.native.pt_points())

    def keys(self):
        return self.store.native.pt_points()


def publish_slot(generations, model, version, slot):
    """(B58c) TeacherGenerations.publish into the slot the native point table chose (its refs are native)."""
    import copy
    if generations.slots[slot] is None:
        generations.slots[slot] = copy.deepcopy(model).eval().requires_grad_(False)
    else:
        generations.slots[slot].load_state_dict(model.state_dict())
    generations.versions[slot] = int(version)


def driver_ring(driver, actor, lane):
    """(B54) The C++ driver's lineage ring of a restored lane row (SlotBlockActor `actor`, its slot `lane`): the
    decisions of the row's last block_steps steps from the lane's log (step s at position s % block_steps, -1 where the
    branch has none), so the blocks the lane completes later get the sources _pages would read from the log."""
    bs=actor.model.block_memory.config.block_steps
    row=actor.base+int(lane)
    step=int(actor.state.step_host[row])
    log=actor.logs[int(lane)]
    lo=max(log.offset,step-bs)
    dec=np.full(bs,-1,np.int64); kv=np.full(bs,-1,np.int64)
    if step>lo:
        d,k=log.decisions(lo,step)
        at=np.arange(lo,step)%bs
        dec[at]=d; kv[at]=k
    driver.set_ring(row,torch.from_numpy(dec),torch.from_numpy(kv),int(step-lo))


# D1 (2026-11-10): the CF points' lives (capture to release: seconds, policy versions crossed) and the seconds both
# teacher generations were pinned, per collect (point_stats), as progress.csv columns
D1_FIELDS = ('d1_released', 'd1_life_p50_s', 'd1_life_p90_s', 'd1_life_p50_v', 'd1_life_p90_v', 'd1_refused', 'd1_gen_both_s')


class BlockLaneService(LaneService):
    RECLAIM_S=0.1   # B46: at most this often the context / injury-source reclamation (service_control)
    DEFER_S=2.0     # B54: a branch request whose point is not captured yet waits at most this long, then is cancelled
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
        # B54: the C++ driver steps the lane rounds (attach_driver): jobs in flight, job id -> (generation, positions,
        # version, lanes, rows, pending, keys, replay)
        self.driver=None
        self.drv_jobs={}
        self.job_serial=0
        self.py_mode=False      # (B54: serve_wait with the driver paused steps rounds on the Python path)
        self.deferred_begin={}  # (B54: lane -> when its branch request first waited for its point's capture)
        self.lanes_only=False   # (B54: the driver runs lanes only while the actor waits for the learner)
        self.ring_dirty=set()   # (B54: (generation, lane) rows the Python path stepped: their ring re-set before a job)
        self.lin=None           # (B58c: the page store whose native lineage keeps the main rows' lineage: attach_lineage)

    def attach_lineage(self, store):
        """(B58c) The driver keeps the main rows' lineage: observe_rows / record_main / capture_deferred are its; the
        CF points live in its point table (self.points a view), injury sources in its table (injury_context materializes
        one), the reclamation of injury sources runs there; a capture whose generation needs new weights comes back as
        a kind 6 record (on_publish)."""
        self.lin = store
        self.points = NativePoints(store)

    def attach_native_lanes(self, driver, stream):
        """(B58d) The lane service runs in C++ (on the driver thread, or on its own thread: ISAAC_BLOCK_CPP_LANE_THREAD):
        tick has nothing to do here; the lane records that still come to Python are a lane graph to capture (kind 1)
        and a job the graphs cannot take (kind 5: stepped on its lane actor, on the lane stream)."""
        self.native_lanes = True
        self.lane_stream = stream

    def on_native_lane_record(self, rec):
        """(B58d) A record of the native lane service; the lane service waits for the answer."""
        import time as time_
        kind = int(rec['kind'])
        if kind == 3:
            raise RuntimeError('native lane service: ' + self.driver.lane_error())
        generation = int(rec['point'])
        actor = self.actors[generation]
        if kind == 1:   # a lane graph to capture (on its owner's graph, as the B54 path)
            with torch.inference_mode(), torch.autocast(torch.device(self.device).type, dtype=torch.bfloat16, enabled=actor.bf16):
                while actor.step_graph.capture_key(rec['cap'], rec['widths']) is None:
                    time_.sleep(0.001)
            self.driver.lane_job_resume()
            return None
        if kind == 5:
            return self._native_python_job(rec, generation, actor)
        raise RuntimeError(f'native lane service: a record of kind {kind}')

    def _native_python_job(self, rec, generation, actor):
        """(B58d, kind 5) A lane job the graphs cannot take, stepped here on its lane actor (the eager path) on the lane
        stream: the rows' logs from the native heads first (the window a step reads), the step, the new decisions /
        blocks back to the native lane service with the outputs, the driver's lineage rings re-set."""
        from .block_paged import native_snapshots, EpisodeLog
        from .tok_obs import ROW
        nat = self.lin.native
        lanes = rec['lane_ids'].numpy(); seq = rec['lane_seq'].numpy(); replay = rec['lane_replay'].numpy().astype(bool)
        rows = rec['rows'].numpy().view(ROW); pend = rec['pending_before'].numpy()
        before = {}
        for lane in lanes.tolist():
            row = actor.base+lane
            end = nat.lin_head_end(row)
            if end >= 0:
                snap = nat.lin_snap_row(row, torch.zeros(len(self.model.heads), dtype=torch.long), 0)
                try:
                    log = native_snapshots(self.lin, [snap], retain=True)[0].log
                finally:
                    nat.lin_release(snap)
                actor.logs[lane] = EpisodeLog(log.episode, log.offset, log.dec[:log.n], log.kvs[:log.n], log.blk[:log.nb], log.blk_end[:log.nb])
            before[lane] = (actor.logs[lane], actor.logs[lane].n, actor.logs[lane].nb)
        with torch.cuda.stream(self.lane_stream), torch.autocast(torch.device(self.device).type, dtype=torch.bfloat16, enabled=actor.bf16):
            actor.step(rows, pend, lanes, ukeys=(seq, lanes), evaluate=replay)
            torch.cuda.current_stream().synchronize()
        k = len(lanes)
        dec = np.zeros(k, np.int64); kvs = np.zeros(k, np.int64); bid = np.full(k, -1, np.int64)
        sb = np.zeros(k, np.int64); first = rows['first'].astype(np.uint8); comp = np.zeros(k, np.uint8)
        for q, lane in enumerate(lanes.tolist()):
            log0, n0, nb0 = before[lane]
            log = actor.logs[lane]
            if log is not log0:   # (a first row: the step began a new log)
                n0, nb0 = 0, 0
            dec[q], kvs[q] = log.dec[log.n-1], log.kvs[log.n-1]
            sb[q] = log.end-1
            if log.nb > nb0:
                bid[q] = log.blk[log.nb-1]; comp[q] = 1
            driver_ring(self.driver, actor, lane)
        out = np.ascontiguousarray(actor.host_output[:k], np.float32)
        self.driver.lane_python_done(torch.from_numpy(out), torch.from_numpy(sb), torch.from_numpy(first),
                                     torch.from_numpy(comp), torch.from_numpy(dec), torch.from_numpy(kvs), torch.from_numpy(bid),
                                     int(rec['version']))
        self.stat['py_lane_jobs_native'] = self.stat.get('py_lane_jobs_native', 0)+1
        return None

    def attach_driver(self, driver):
        """(B54) The lane rounds' steps go to the C++ driver: _begin restores here (and sets the restored rows'
        lineage rings), each generation's rows become one driver job (merged with a main batch of the same policy
        version, else a lane-only call through the generation's lane graph), the answers come back as kind 4 records
        (on_lane_record); tick never waits for them."""
        self.driver=driver
        if os.environ.get('ISAAC_BLOCK_LANE_MERGE','1')=='0':
            driver.set_lane_merge(False)

    def _set_ring(self, actor, lane):
        driver_ring(self.driver, actor, lane)

    def observe_rows(self,workers,rows):
        """Called before Episodes.arrive, including terminal records (no action)."""
        if self.lin is not None:   # (B58c: the driver's)
            return
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
        if self.lin is not None:   # (B58c: the driver's)
            return
        for worker,row,context in zip(workers,rows,contexts):
            worker=int(worker)
            key=(worker,int(row['episode']),int(row['t']))
            self.injury_sources[key]=context
            recent=self.recent.setdefault(worker,[])
            recent.append(key)
            del recent[:-self.recent_size]

    def injury_context(self,worker,episode,step):
        if self.lin is not None:   # (B58c) the native source, materialized
            from .block_paged import native_snapshots
            return native_snapshots(self.lin,[self.lin.native.inj_get(int(worker),int(episode),int(step))])[0]
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

    def capture_deferred(self, workers, requests, snapshots):
        """(B52) capture_main's CF point capture from the driver's record: the pre-step snapshots the consumer
        rebuilt, the workers' requests the driver read (and zeroed) before replying."""
        if self.lin is not None:   # (B58c: the driver captured natively; kind 6 records for the rest)
            return
        status = self.s.block_arrays[0]
        for worker, request, snap in zip(workers, requests, snapshots):
            if request:
                point = int(request)-1
                accepted = self.capture_point(point, snap, self.model, self.main_actor.version)
                status.flat[point] = 2 if accepted else -1

    def service_control(self):
        if getattr(self, 'native_lanes', False):   # (B58d: the native lane service's control)
            return None
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
        if self.lin is not None:   # (B58c: the injury sources' reclamation in the native table, the same rule)
            self.lin.native.inj_reclaim()
            return
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
        if self.driver is None:
            return super().serve_wait(timeout)
        # (B54: the actor waits for the learner: the driver runs the lane jobs only (main_hold: the workers' records
        # wait), the rounds go on as in a collect, their records consumed here; quiesce ends it)
        import time as time_
        if not self.lanes_only:
            self.driver.set_main_hold(True)
            self.driver.resume()
            self.lanes_only=True
        end=time_.perf_counter()+timeout
        did=False
        native=getattr(self,'native_lanes',False)
        while True:   # (B58d native lanes: only lane captures / Python jobs come here: a long wait, no busy loop)
            rec=self.driver.next_record(max(0.0,min(end-time_.perf_counter(),0.05)) if native else 0.0005)
            if rec is not None:
                self.driver_record(rec)
                did=True
            if not getattr(self,'native_lanes',False):
                did=bool(self.tick()) or did
            if time_.perf_counter()>=end:
                return did

    def driver_record(self, rec, resume=True):
        """(B54) A record of the driver while it runs lanes only: lane rows (kind 4), a job for the Python path (5), a
        lane graph to capture (1: captured on its owner's graph, the driver resumed unless resume=False)."""
        import time as time_
        kind=int(rec['kind'])
        if int(rec.get('lane_src',0)):   # (B58d: a record of the native lane service)
            return self.on_native_lane_record(rec)
        if kind==4:
            return self.on_lane_record(rec)
        if kind==5:
            return self.on_lane_python(rec)
        if kind==6:   # (B58c: a capture's generation to publish; none come while main rows wait, kept for safety)
            return self.on_publish(rec)
        if kind==1:
            owners=[self.main_actor,*self.actors.values()]
            own=next(a for a in owners if a.step_graph is not None and a.step_graph.owner==int(rec['owner']))
            with torch.inference_mode(), torch.autocast(torch.device(self.device).type,dtype=torch.bfloat16,enabled=own.bf16):
                while own.step_graph.capture_key(rec['cap'],rec['widths']) is None:
                    time_.sleep(0.001)
            if resume:
                self.driver.resume()
            return None
        if kind==3:
            raise RuntimeError('cpp actor driver: '+self.driver.last_error())
        if kind in (7,9):   # (B58e: a collect target reached late / the learner's wake: nothing to do here)
            return None
        raise RuntimeError(f'cpp actor driver: a record of kind {kind} while it runs lanes only')

    def capture_point(self, bank_index, snapshot, actor_model, model_version):
        """Before consuming the selection-point observation. False = backpressure."""
        if bank_index in self.points:
            raise RuntimeError("release a search point before reusing its bank slot")
        if int(model_version) != snapshot.policy_version:
            raise ValueError("point snapshot and supplied actor weights have different versions")
        if not self.generations.publish(actor_model, snapshot.policy_version):
            self.__dict__['d1_refused'] = self.__dict__.get('d1_refused', 0)+1   # (not in self.stat: point_stats' row goes there)
            return False
        generation, version, teacher = self.generations.acquire(snapshot.policy_version)
        import time as time_
        self.__dict__.setdefault('point_life', {})[int(bank_index)] = (time_.perf_counter(), int(version))
        self._gen_both()
        self.point_serial += 1
        token = self.point_serial
        self.points[int(bank_index)] = (snapshot.fork(), generation, version, token)
        # A recycled generation's policy object is reused, but all branch state is
        # restored explicitly at begin; no preceding candidate's state is inherited.
        if generation not in self.actors and hasattr(self.main_actor,'lane_actor'):
            # A1: teacher generations share the main actor's page store; branch state is a slot row.
            # (B46: generation g's lanes in the main actor's arena: merged calls replay its step graph)
            # (B54: its graph captures with the driver paused; train_tok pre-creates both generations at start-up)
            if self.driver is not None:
                self.driver.pause()
            try:
                self.actors[generation] = self.main_actor.lane_actor(teacher, len(self.s.lane_io), generation=generation)
            finally:
                if self.driver is not None:
                    self.driver.resume()
        if generation not in self.actors:
            self.actors[generation] = BlockActor(teacher, len(self.s.lane_io),
                graph_entities=self.main_actor.graph_entities if self.main_actor is not None else 0,
                bf16=self.main_actor.bf16 if self.main_actor is not None else False)
        return True

    def on_publish(self, rec):
        """(B58c) A kind 6 record: a CF capture the driver could not make because its version's teacher generation
        needs the main weights first (TeacherGenerations.publish into the slot the point table chose); published here,
        the point captured, the worker's next record released."""
        nat=self.lin.native
        point,slot,snap,worker,version=(int(rec[k]) for k in ('point','pub_slot','pub_snap','pub_worker','version'))
        if version not in self.generations.versions:   # (another kind 6 record of this version published it already)
            publish_slot(self.generations,self.model,version,slot)
            torch.cuda.current_stream().synchronize()   # (B58d: the weights in place before the lane stream can use the slot)
            nat.pt_published(slot,version)
        if slot not in self.actors and hasattr(self.main_actor,'lane_actor'):
            # (a lane actor made here pauses the driver from inside a record's handling; with the native lane service the
            # driver could be waiting for a lane record behind this one: attach_lanes requires every generation's lane
            # actor made at start-up, so this branch is the B54 Python lane service's only)
            if getattr(self,'native_lanes',False):
                raise RuntimeError('native lanes: generation %d has no lane actor (precapture_lane_generations at start-up)' % slot)
            self.driver.pause()
            try:
                self.actors[slot]=self.main_actor.lane_actor(self.generations.slots[slot],len(self.s.lane_io),generation=slot)
            finally:
                self.driver.resume()
        try:
            got=nat.pt_capture(point,snap,version)
        finally:
            nat.lin_release(snap)
        if got not in (2,-1):
            raise RuntimeError(f'native point table: capture of point {point} after publishing generation {slot}: {got}')
        self.driver.capture_done([worker])

    def release_point(self, bank_index):
        if self.lin is not None:   # (B58c: the native table releases the generation; the contexts' owner completes)
            token=self.lin.native.pt_release(int(bank_index))
            self.points.cache.pop(int(bank_index),None)
            if token>=0:
                self.contexts.complete(token)
            return
        snapshot, generation, version, token = self.points.pop(int(bank_index))
        self.generations.release(generation)
        self.contexts.complete(token)
        import time as time_
        life = self.__dict__.setdefault('point_life', {}).pop(int(bank_index), None)
        if life is not None:   # (D1)
            self.__dict__.setdefault('d1_lives', []).append((time_.perf_counter()-life[0],
                                                             int(getattr(self.main_actor, 'version', life[1]))-life[1]))
        self._gen_both()

    def _gen_both(self, now=None):
        """(D1) Both teacher generations pinned: the time it lasts."""
        import time as time_
        now = time_.perf_counter() if now is None else now
        both = all(r > 0 for r in self.generations.refs)
        since = self.__dict__.get('both_since')
        if since is not None and not both:
            self.__dict__['d1_gen_both_s'] = self.__dict__.get('d1_gen_both_s', 0.0)+now-since
            self.both_since = None
        elif since is None and both:
            self.both_since = now

    def point_stats(self):
        """(D1) This collect's released points' lives (p50 / p90 of seconds and of policy versions crossed), refused
        captures, seconds both generations were pinned; the counters restart."""
        import time as time_
        import numpy as np_
        now = time_.perf_counter()
        if self.__dict__.get('both_since') is not None:   # (an interval still open: counted up to now)
            self.__dict__['d1_gen_both_s'] = self.__dict__.get('d1_gen_both_s', 0.0)+now-self.both_since
            self.both_since = now
        lives = self.__dict__.pop('d1_lives', [])
        s = np_.array([x[0] for x in lives]) if lives else np_.zeros(0)
        v = np_.array([x[1] for x in lives]) if lives else np_.zeros(0)
        q = lambda a, p: float(np_.percentile(a, p)) if len(a) else ''
        out = dict(d1_released=len(lives), d1_life_p50_s=q(s, 50), d1_life_p90_s=q(s, 90), d1_life_p50_v=q(v, 50),
                   d1_life_p90_v=q(v, 90), d1_refused=self.__dict__.pop('d1_refused', 0), d1_gen_both_s=self.__dict__.pop('d1_gen_both_s', 0.0))
        return out

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
            if self.driver is not None:
                self._set_ring(actor, lane)
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
        if self.py_mode:
            for generation, positions in group_items:
                self.ring_dirty.update((generation, int(idx[j])) for j in positions)
            # (B54 counters: the rounds / rows the Python path steps while the actor waits for the learner)
            self.stat['py_lane_rounds'] = self.stat.get('py_lane_rounds', 0)+int(bool(group_items))
            self.stat['py_lane_rows'] = self.stat.get('py_lane_rows', 0)+sum(len(pos) for _, pos in group_items)
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

    # ------------------------------------------------------------------ B54: the rounds on the C++ driver
    def tick(self):
        if getattr(self, 'native_lanes', False):   # (B58d: the native lane service's)
            return False
        if self.driver is None:
            return super().tick()
        self.service_control()
        did=False
        if self.fly is not None:
            if not self.fly['ready']:   # (its jobs' records not all consumed yet: never wait for them)
                return False
            self.finish()
            did=True
        self.s.poll_lanes()
        if self.s.lane_ready and self.due():
            self.submit()
            if self.fly is not None and self.fly['ready']:
                self.finish()
            did=True
        return did

    def quiesce(self):
        if self.driver is None:
            return super().quiesce()
        if self.lanes_only:   # (B54: back from the learner wait: the driver paused, its lane records consumed)
            if getattr(self,'native_lanes',False):   # (the driver thread may wait for a lane record of this queue)
                from .block_driver import pause_serving
                pause_serving(self.driver, lambda rec: self.driver_record(rec, resume=False))
            else:   # (B54's Python lane service: the driver never waits for Python here)
                self.driver.pause()
            while True:
                rec=self.driver.next_record(0.0)
                if rec is None:
                    break
                self.driver_record(rec, resume=False)
            self.driver.set_main_hold(False)
            self.lanes_only=False
        # (the main weights change with the driver paused; the queued jobs keep their generation's own weights and
        # merge with a main batch only at an equal version: nothing to wait for)
        if self.fly is not None and self.fly['ready']:
            self.finish()

    def _submit(self):
        if self.driver is None:
            return super()._submit()
        if self.fly is not None:   # (one round in flight at a time; tick finishes it first)
            return False
        import time as time_
        s = self.s
        idx = s.take_lanes()
        if not len(idx):
            return False
        t0 = time_.perf_counter()
        # (B54: a branch's first request can arrive before the consumer has taken the main record whose CF capture
        # makes its point -- the driver replies before the record is consumed. Such a request waits for the capture
        # (back into lane_ready), at most DEFER_S, then is answered as cancelled (a point that never came).)
        keep, back = [], []
        for lane in idx.tolist():
            begin, seq = int(s.lane_io[lane, 11]), int(s.lane_io[lane, 10])
            if begin > 0 and (begin-1) not in self.points and self.cancelled_requests.get(lane) != seq:
                first = self.deferred_begin.setdefault(lane, t0)
                if t0-first < self.DEFER_S:
                    back.append(lane); continue
                self.cancelled_requests[lane] = seq
                self.stat['lane_orphan_requests'] = self.stat.get('lane_orphan_requests', 0)+1
            self.deferred_begin.pop(lane, None)
            keep.append(lane)
        if back:
            if not s.lane_ready:
                s.lane_first_t = t0
            s.lane_last_t = t0
            s.lane_ready.extend(back)
            self.stat['lane_deferred_requests'] = self.stat.get('lane_deferred_requests', 0)+len(back)
        idx = np.asarray(keep, np.int64)
        if not len(idx):
            return False
        n = len(idx)
        self.stat['held_s'] += max(0.0, t0 - getattr(s, 'lane_first_t', t0))
        self.seen[idx] = t0
        rows = s.lane_rows[idx]
        pend = s.lane_io[idx, :self.nh].astype(np.int64)
        seq = s.lane_io[idx, 10].astype(np.int64) if s.lane_io.shape[1] > 10 else np.zeros(n, np.int64)
        f = dict(idx=idx, rows=rows, seq=seq, t0=t0, ready=False)
        self.fly = f
        self._begin(idx)
        self._submit_jobs(f, rows, pend, idx, (seq, idx))
        self.stat['s'] += time_.perf_counter() - t0
        return True

    def _submit_jobs(self, f, rows, pend, idx, ukeys):
        """(B54) _eager's grouping (cancelled requests, repeated answers, one group per teacher generation) with
        the steps queued on the driver: one job per generation, the policy version of its lane actor."""
        n = len(idx)
        f['a'] = pend.copy()
        f['lp'], f['v'], f['vg'] = (np.zeros(n, np.float32) for _ in range(3))
        f['version'] = np.full(n, -1, np.int64)
        f['cancelled'] = np.zeros(n, bool)
        f['block_refs'] = np.zeros((n, self.s.lane_t.shape[1]), np.float32)
        f['eager_calls'] = 0
        groups = {}
        replays = np.zeros(n, bool)
        for j, lane in enumerate(idx):
            lane = int(lane); seq = int(f['seq'][j])
            if self.cancelled_requests.get(lane) == seq:
                f['cancelled'][j] = True
                continue
            if lane in self.last_answers and self.last_answers[lane][0] == seq:
                (_, f['a'][j], f['lp'][j], f['v'][j], f['vg'][j], f['version'][j], f['block_refs'][j],
                 f['cancelled'][j]) = self.last_answers[lane]
                continue
            generation = self.points[self.lane_point[lane]][1]
            replays[j] = bool(self.s.block_arrays[1][lane]) if self.main_actor is not None else False
            groups.setdefault(generation, []).append(j)
        items = list(groups.items())
        # (the main policy's generation first, as _eager: the driver merges it with its next main batch)
        items.sort(key=lambda item: self.actors[item[0]].version != self.main_actor.version)
        jobs = []
        for generation, positions in items:
            positions = np.asarray(positions, np.int64)
            actor = self.actors[generation]
            lanes = idx[positions]
            for lane in lanes.tolist():   # (rows the Python path stepped meanwhile: the driver's ring from the log)
                if (generation, lane) in self.ring_dirty:
                    self.ring_dirty.discard((generation, lane))
                    driver_ring(self.driver, actor, lane)
            sub = np.ascontiguousarray(rows[positions])
            self.job_serial += 1
            job = self.job_serial
            # (the points' context owners now: a cancellation may release a point while its job is in flight)
            tokens = [self.points[self.lane_point[int(q)]][3] for q in lanes]
            self.drv_jobs[job] = (generation, positions, int(self.generations.versions[generation]), lanes, sub,
                                  pend[positions].copy(), (ukeys[0][positions], ukeys[1][positions]), replays[positions], tokens)
            jobs.append(job)
            owner = actor.step_graph.owner if actor.step_graph is not None else id(actor)   # (CPU actors: no graphs)
            self.driver.submit_lanes(job, int(owner), int(actor.version),
                                     torch.from_numpy(sub.view(np.uint8).reshape(-1)), len(positions),
                                     torch.from_numpy(np.ascontiguousarray(pend[positions], np.int64).reshape(-1)),
                                     torch.from_numpy((actor.base+lanes).astype(np.int64)),
                                     torch.from_numpy(np.ascontiguousarray(ukeys[0][positions], np.int64)),
                                     torch.from_numpy(np.ascontiguousarray(ukeys[1][positions], np.int64)),
                                     torch.from_numpy(replays[positions].astype(np.uint8)))
        f['jobs'] = set(jobs)
        f['ready'] = not jobs

    def _answer_rows(self, f, generation, positions, version, out, contexts, replay, tokens):
        """(B54) _eager's per-row answers of one generation's rows (outputs, version, context refs, lane steps)."""
        nh = self.nh
        f['a'][positions] = out[:, :nh].astype(np.int64)
        f['lp'][positions], f['v'][positions], f['vg'][positions] = out[:, nh], out[:, nh+1], out[:, nh+2]
        f['version'][positions] = version
        for k, j in enumerate(positions):
            lane = int(f['idx'][j])
            if not replay[k]:   # (registered in _finish, right before lane_t: see there)
                f.setdefault('puts', []).append((j, tokens[k], contexts[k]))
            self.lane_steps[lane] += 1
        self.stat['block_teacher_forward_calls'] = self.stat.get('block_teacher_forward_calls', 0)+1
        self.stat['block_mixed_replay_calls'] = self.stat.get('block_mixed_replay_calls', 0)+int(replay.any() and not replay.all())

    def on_lane_record(self, rec):
        """(B54) A kind 4 record: the lane rows' lineage (logs, contexts, as SlotBlockActor._pages), the answers."""
        from .block_paged import PagedSnapshot, EpisodeLog
        job = int(rec['job'])
        generation, positions, version, lanes, rows, pend, keys, replay, tokens = self.drv_jobs.pop(job)
        f = self.fly
        actor = self.actors[generation]
        out = rec['out'].numpy()
        step_b = rec['step_before'].numpy(); first = rec['first'].numpy().astype(bool)
        comp = rec['complete'].numpy().astype(bool); dec = rec['dec'].numpy(); kvs = rec['kvs'].numpy(); bid = rec['bid'].numpy()
        ver = int(rec['version'])
        contexts = []
        for k, lane in enumerate(lanes.tolist()):
            if first[k]:
                actor.logs[lane] = EpisodeLog((os.getpid(), lane, int(rows['episode'][k])))
            contexts.append(PagedSnapshot(actor.store, actor.logs[lane], int(step_b[k]), pend[k:k+1].copy(), ver))
            log = actor.logs[lane]
            log.push(dec[k:k+1], kvs[k:k+1])
            if comp[k]:
                log.push_block(int(bid[k]), int(step_b[k]))
            actor.pending[lane] = out[k:k+1, :self.nh].astype(np.int64)
        actor.host_output = out
        actor.last_contexts = contexts
        if int(rec['merged']):
            self.stat['merged_lane_rows'] = self.stat.get('merged_lane_rows', 0)+len(lanes)
        self._answer_rows(f, generation, positions, version, out, contexts, replay, tokens)
        f['jobs'].discard(job)
        if not f['jobs']:
            f['ready'] = True

    def on_lane_python(self, rec):
        """(B54) A kind 5 record: a job the driver cannot replay (an invalid / too wide frame); the driver is paused.
        Stepped here on the lane actor (its eager path, as _eager would), the rings set again, the job dropped."""
        job = int(rec['job'])
        self.driver.drop_lane_job(job)
        self._python_job(job)

    def _python_job(self, job):
        """(B54) A queued job stepped here on its lane actor (the driver paused), its rings re-set, answered."""
        generation, positions, version, lanes, rows, pend, keys, replay, tokens = self.drv_jobs.pop(job)
        f = self.fly
        actor = self.actors[generation]
        with torch.autocast(torch.device(self.device).type, dtype=torch.bfloat16, enabled=actor.bf16):
            actor.step(rows, pend, lanes, ukeys=keys, evaluate=replay)
        f['eager_calls'] += int(getattr(actor, 'last_path', 'eager') == 'eager')
        for lane in lanes.tolist():
            self._set_ring(actor, lane)
        self._answer_rows(f, generation, positions, version, actor.host_output, actor.last_contexts, replay, tokens)
        f['jobs'].discard(job)
        if not f['jobs']:
            f['ready'] = True

    def finish(self):
        nv_push('lane.finish')
        try:
            return self._finish()
        finally:
            nv_pop()

    def _finish(self):
        if self.fly is not None and self.fly.get('puts'):
            # B54: the driver path's contexts go into the registry here, just before their refs reach lane_t: between
            # a kind 4 record and this answer the control's reclamation (retire, transient) runs and would drop an
            # ident no lane_t / v2_t row names yet (the Python path registers and answers in one call)
            f=self.fly
            for j,token,ctx in f.pop('puts'):
                ident=self.contexts.put(token,ctx)
                f['block_refs'][j,0],f['block_refs'][j,1]=ident & 65535,ident >> 16
                f['block_refs'][j,-1]=1.0
        if self.fly is not None:
            f=self.fly
            for j,lane in enumerate(f['idx']):
                self.last_answers[int(lane)]=(int(f['seq'][j]),f['a'][j].copy(),f['lp'][j],f['v'][j],f['vg'][j],
                                              f['version'][j],f['block_refs'][j].copy(),f['cancelled'][j])
            self.s.lane_t[self.fly["idx"]] = self.fly["block_refs"]
            self.s.lane_out[self.fly['idx'],5]=-self.fly['cancelled'].astype(np.int64)
        return super().finish()

