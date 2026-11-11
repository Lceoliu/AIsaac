"""bm1: a CUDA smoke test of the venv on this host (one GPU): init, a matmul checked on the CPU, a cuDNN convolution
checked against the CPU, a CUDA graph captured and replayed (values checked). Prints one JSON line; rc 0 = all ok."""
import json
import sys

out = {}
ok = True
try:
    import torch
    out.update(torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version())
    out['init'] = torch.cuda.is_available() and torch.cuda.device_count() >= 1
    out['device'] = torch.cuda.get_device_name(0)
    g = torch.Generator().manual_seed(1)
    a, b = torch.randn(512, 512, generator=g), torch.randn(512, 512, generator=g)
    torch.backends.cuda.matmul.allow_tf32 = False
    out['matmul_maxdiff'] = float((a.cuda() @ b.cuda()).cpu().sub(a @ b).abs().max())
    x, w = torch.randn(4, 8, 32, 32, generator=g), torch.randn(16, 8, 3, 3, generator=g)
    torch.backends.cudnn.allow_tf32 = False
    out['conv_maxdiff'] = float(torch.nn.functional.conv2d(x.cuda(), w.cuda(), padding=1).cpu()
                                .sub(torch.nn.functional.conv2d(x, w, padding=1)).abs().max())
    xs = torch.ones(1 << 16, device='cuda')
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        y = xs*2
    torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr, stream=s):
        y = xs*2+1
    xs.fill_(3.0)
    gr.replay()
    torch.cuda.synchronize()
    out['graph_replay_ok'] = bool((y == 7).all())
    ok = out['init'] and out['matmul_maxdiff'] < 1e-3 and out['conv_maxdiff'] < 1e-3 and out['graph_replay_ok']
except BaseException as e:
    out['error'] = f'{type(e).__name__}: {e}'[:400]
    ok = False
out['ok'] = bool(ok)
print(json.dumps(out), flush=True)
sys.exit(0 if ok else 1)
