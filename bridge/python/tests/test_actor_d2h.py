"""Same D2H values; a compact pinned view must not wait for queued GPU work."""
import json
import statistics
import time
import torch

torch.set_num_threads(2)
src=torch.arange(32*8,device='cuda',dtype=torch.float32).view(32,8)
wide=torch.zeros((32,10),pin_memory=True)
flat=torch.zeros(32*10,pin_memory=True)
stream=torch.cuda.Stream()
stream.wait_stream(torch.cuda.current_stream());stream.synchronize()
result={}
for name,dst in [('strided',wide[:,:8]),('compact',flat[:32*8].view(32,8))]:
    elapsed=[];overlap=[]
    for _ in range(3):
        ready=torch.cuda.Event()
        with torch.cuda.stream(stream):
            torch.cuda._sleep(100_000_000)
            ready.record()
            start=time.perf_counter()
            dst.copy_(src,non_blocking=True)
            elapsed.append(time.perf_counter()-start)
            overlap.append(not ready.query())
        stream.synchronize()
        torch.testing.assert_close(dst,src.cpu(),rtol=0,atol=0)
    result[name]=dict(host_copy_median_seconds=statistics.median(elapsed),
                      returned_before_prior_gpu_work=overlap,contiguous=dst.is_contiguous(),
                      pinned=dst.is_pinned())
assert not any(result['strided']['returned_before_prior_gpu_work']),result
assert all(result['compact']['returned_before_prior_gpu_work']),result
print(json.dumps(dict(result='PASS',measurements=result)))
