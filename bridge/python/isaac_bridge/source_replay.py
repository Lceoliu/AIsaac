"""Value-cache VJPs: temporary training graphs, not long-lived episode graphs.

The Reader's cotangents are summed before this call. Replay the original Writer
versions on cached source values, then replay the original Encoder versions on
raw observations. This is a gradient-cache decomposition of the same delayed
gradient, not a current-parameter surrogate or truncated BPTT.

The source path has no stochastic layers. Batching changes floating-point
reduction order; CPU/FP32 and BF16 parity are tested separately.
"""
import time

import numpy as np
import torch


def _batches(jobs, size, length):
    """Pack across episode boundaries; only a parameter-version boundary matters."""
    batch, used = [], 0
    for job in jobs:
        lo = 0
        while lo < length(job):
            hi = min(length(job), lo + size - used)
            batch.append((job, lo, hi)); used += hi - lo; lo = hi
            if used == size:
                yield batch
                batch, used = [], 0
    if batch:
        yield batch


def _packed_indices(starts, counts):
    row = np.repeat(np.arange(len(counts)), counts)
    return starts[row] + np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)


def encode_values(pipe, worker, items, version):
    """One no-grad forward per packed batch, independent allocations per owner.

    Only numerical results outlive this call. A late record cannot keep another
    episode's activation graph or large backing allocation alive.
    """
    groups, group, count = [], [], 0
    for item in items:
        size = len(item[2])
        if group and count + size > pipe.micro:
            groups.append(group); group, count = [], 0
        group.append(item); count += size
    if group: groups.append(group)
    results = []
    for group in groups:
        full = pipe._encode(worker, [part for _, parts, _ in group for part in parts], version)
        lo = 0
        for _, _, keys in group:
            hi = lo + len(keys)
            counts = full['counts'][lo:hi].copy()
            take = _packed_indices(full['starts'][lo:hi], counts)
            results.append(dict(y=full['y'].index_select(0, torch.as_tensor(take, device=full['y'].device)),
                                u=full['u'][lo:hi].clone(), starts=np.cumsum(counts)-counts,
                                counts=counts, grads={}, saved_cpu_bytes=0,
                                _replay_rows=full['_replay_rows'][lo:hi].copy(),
                                _replay_pending=full['_replay_pending'][lo:hi].copy()))
            lo = hi
    return results


def backward_replay(pipe, worker, packets, retain_graph, kind):
    from .source_pipeline import source_parameter

    started = time.perf_counter()
    device = pipe.devices[worker]
    selected = [(c, g) for c, g in packets if c['worker'] == worker]
    if not selected:
        return {}
    used = pipe.touched[worker]; used.clear()
    named = lambda replica: replica._source_named
    first = next(iter(pipe.versions.values()))[worker]
    shared = {n: torch.zeros_like(p) for n, p in named(first) if source_parameter(n)}
    source_jobs, writer_jobs = {}, {}
    encoder_batches = writer_batches = replayed_rows = serviced = 0
    service_seconds = 0.

    def serve():
        nonlocal serviced, service_seconds
        if retain_graph and hasattr(pipe, 'service_forward'):
            at = time.perf_counter()
            serviced += pipe.service_forward()
            service_seconds += time.perf_counter() - at

    def source_job(chunk):
        key = id(chunk)
        if key not in source_jobs:
            source_jobs[key] = (chunk, {})
        return source_jobs[key][1]

    with pipe._context(worker), torch.enable_grad():
        for replicas in pipe.versions.values():
            for n, p in named(replicas[worker]):
                p.grad = shared[n] if n in shared else None
        for chunk, gradients in selected:
            if 'mem' in gradients:
                writer_jobs.setdefault(chunk['version'], []).append((chunk, gradients['mem']))
            else:
                target = source_job(chunk)
                for name, gradient in gradients.items():
                    if name in target: target[name].add_(gradient)
                    else: target[name] = gradient.clone()

        # Writer gradients also contribute to old Encoder rows that have no
        # direct Reader use. Keep every contribution and the original version.
        for version, jobs in writer_jobs.items():
            replica = pipe.versions[version][worker]
            block = replica.reader.block
            for batch in _batches(jobs, max(1, pipe.micro // block), lambda job: len(job[1])):
                refs = np.concatenate([job[0]['future'].result()['_replay_refs'][lo*block:hi*block]
                                       for job, lo, hi in batch])
                incoming = torch.cat([job[1][lo:hi] for job, lo, hi in batch])
                grouped = {cid: np.flatnonzero(refs[:, 0] == cid) for cid in np.unique(refs[:, 0])}
                sample = pipe.chunks[int(refs[0, 0])]['future'].result()['u']
                x = sample.new_empty((len(refs), sample.shape[-1]))
                with torch.no_grad():
                    for cid, positions in grouped.items():
                        u = pipe.chunks[int(cid)]['future'].result()['u']
                        dst = torch.as_tensor(positions, device=device)
                        src = torch.as_tensor(refs[positions, 1], device=device)
                        x.index_copy_(0, dst, u.index_select(0, src))
                # FP32 cotangent accumulation also when the source forward is BF16.
                x = x.float().requires_grad_()
                with pipe._cast_context(replica), torch.autocast(device.type, dtype=torch.bfloat16, enabled=pipe.amp):
                    mem = replica.reader.write(x.to(sample.dtype).reshape(-1, block, x.shape[-1]))
                torch.autograd.backward(mem, incoming)
                for cid, positions in grouped.items():
                    chunk = pipe.chunks[int(cid)]
                    # A deliberately detached ancestor is a constant input.
                    # The existing stale-write audit detects illegal late credit;
                    # the negative-control retention test exercises this case.
                    # Retirement marks the chunk before flushing its last VJP.
                    # Until _detach actually removes the inputs that final
                    # Writer contribution must still reach the Encoder.
                    if chunk.get('detached') and '_replay_rows' not in chunk['future'].result(): continue
                    target = source_job(chunk)
                    if 'u' not in target:
                        target['u'] = torch.zeros_like(chunk['future'].result()['u'], dtype=torch.float32)
                    target['u'].index_add_(0, torch.as_tensor(refs[positions, 1], device=device),
                                          x.grad.index_select(0, torch.as_tensor(positions, device=device)))
                writer_batches += 1
                del x, mem, incoming
                serve()

        by_version = {}
        for job in source_jobs.values():
            by_version.setdefault(job[0]['version'], []).append(job)
        for version, jobs in by_version.items():
            for batch in _batches(jobs, pipe.micro, lambda job: len(job[0]['future'].result()['_replay_rows'])):
                parts = [(job[0]['future'].result()['_replay_rows'][lo:hi],
                          job[0]['future'].result()['_replay_pending'][lo:hi]) for job, lo, hi in batch]
                result = pipe._encode(worker, parts, version, recompute=True)
                outputs, cotangents = [], []
                if any('u' in job[1] for job, _, _ in batch):
                    grad = torch.zeros_like(result['u'], dtype=torch.float32)
                    at = 0
                    for job, lo, hi in batch:
                        if 'u' in job[1]: grad[at:at+hi-lo].copy_(job[1]['u'][lo:hi])
                        at += hi-lo
                    outputs.append(result['u']); cotangents.append(grad)
                if any('y' in job[1] for job, _, _ in batch):
                    grad = torch.zeros_like(result['y'], dtype=torch.float32)
                    at = 0
                    for job, lo, hi in batch:
                        if 'y' in job[1]:
                            old = job[0]['future'].result()
                            src = _packed_indices(old['starts'][lo:hi], old['counts'][lo:hi])
                            dst = _packed_indices(result['starts'][at:at+hi-lo], result['counts'][at:at+hi-lo])
                            grad.index_copy_(0, torch.as_tensor(dst, device=device),
                                             job[1]['y'].index_select(0, torch.as_tensor(src, device=device)))
                        at += hi-lo
                    outputs.append(result['y']); cotangents.append(grad)
                torch.autograd.backward(outputs, cotangents)
                encoder_batches += 1; replayed_rows += sum(hi-lo for _, lo, hi in batch)
                del result, outputs, cotangents, grad
                serve()
        if device.type == 'cuda' and not pipe.ipc_async:
            torch.cuda.current_stream(device).synchronize()
        grads = {n: shared[n] for n in pipe.names if n in used}
        for replicas in pipe.versions.values():
            for _, param in named(replicas[worker]): param.grad = None
    pipe.events.append(dict(kind=kind or ('source_backward' if retain_graph else 'retired_backward'),
                            device=str(device), version=pipe.version, replay=True,
                            replayed_rows=replayed_rows, encoder_batches=encoder_batches,
                            writer_batches=writer_batches,
                            cached_graph_outputs=sum(torch.is_tensor(c['future'].result().get(k)) and
                                c['future'].result()[k].grad_fn is not None
                                for c in pipe.chunks+pipe.writes if c is not None and c['future'].done()
                                for k in ('u','y','mem')),
                            gradient_buffer_bytes=sum(g.numel()*g.element_size() for g in shared.values()),
                            serviced_forward_calls=serviced, forward_service_seconds=service_seconds,
                            start=started, end=time.perf_counter()))
    return grads
