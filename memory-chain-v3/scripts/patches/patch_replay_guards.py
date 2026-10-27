"""replay-v1 guards (2026-10-27): with replay the options that assume long-lived graphs are forced off, and the raw
rows kept for replay are accounted (source_replay_raw_bytes). usage: python patch_replay_guards.py <checkout>"""
import sys
root = sys.argv[1] + '/bridge/python'


def patch(path, pairs):
    s = open(path, encoding='utf-8').read()
    for a, b in pairs:
        assert s.count(a) == 1, (path, a[:80], s.count(a))
        s = s.replace(a, b)
    open(path, 'w', encoding='utf-8', newline='').write(s)


patch(root + '/isaac_bridge/source_pipeline.py', [
("""        self.replay = bool(replay)
""",
 """        self.replay = bool(replay)
        if self.replay:
            # (guards, 2026-10-27) replay keeps no long-lived graph: layer checkpointing would recompute a second
            # time inside the replay, 8-bit saved activations have nothing to save, and the VJP's engine calls and
            # streams are the replay's own (version-grouped micro batches, one thread): forced off / ignored.
            notes = [n for n, on in (('--source-layer-checkpoint', layer_checkpoint),
                                     ('--source-checkpoint-blocks', checkpoint_blocks is not None),
                                     ('--source-saved-8bit', saved_8bit), ('--source-vjp-merge', vjp_merge),
                                     ('--source-streams', streams)) if on]
            if notes:
                print('[source] replay=1: ' + ', '.join(notes) + ' forced off / ignored', flush=True)
            layer_checkpoint, checkpoint_blocks, saved_8bit, vjp_merge, streams = False, None, 0, 0, 0
"""),
("""                    source_replay=int(self.replay), source_replay_frames=graph if self.replay else 0,""",
 """                    source_replay=int(self.replay), source_replay_frames=graph if self.replay else 0,
                    source_replay_raw_bytes=self.replay_raw_bytes(),"""),
("""    def value_bytes_per_frame(self):""",
 """    def replay_raw_bytes(self):
        \"\"\"(replay) bytes of the raw rows and pending kept for replay: in the source processes (their last memory
        reply) or, with worker threads, in this process.\"\"\"
        if not self.replay: return 0
        if self.processes: return int(sum(m.get('replay_raw_bytes', 0) for m in self.remote_memory))
        total = 0
        for c in self.chunks:
            if c is None or not c['future'].done(): continue
            r = c['future'].result()
            total += sum(r[k].nbytes for k in ('_replay_rows', '_replay_pending') if k in r)
        return int(total)

    def value_bytes_per_frame(self):"""),
])

patch(root + '/isaac_bridge/source_worker.py', [
("""                          replay_frames=sum(c['rows'] for c in local.chunks if c is not None and not c.get('detached')) if local.replay else 0)""",
 """                          replay_frames=sum(c['rows'] for c in local.chunks if c is not None and not c.get('detached')) if local.replay else 0,
                          replay_raw_bytes=local.replay_raw_bytes() if local.replay else 0)"""),
])
print('ok')
