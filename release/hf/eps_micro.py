"""Micro-test: which epsilon does nn.RMSNorm(eps=None) really use, per input dtype / autocast / compile?
Input rows have rms 0.02 so that eps in {2^-7, 1.19e-7} changes the output scale by ~0.4% vs ~0 (clearly visible
after averaging; bf16 rounding is unbiased)."""
import torch, torch.nn as nn
torch.manual_seed(0)
dev = 'cuda'
N = 64
norm = nn.RMSNorm(N).to(dev)                    # eps=None, weight=1
def scale_seen(fn, x):
    y = fn(x).double()
    ref0 = x.double() * torch.rsqrt(x.double().pow(2).mean(-1, keepdim=True))            # eps = 0
    r = (y / ref0).mean().item()                                                           # ~1 for tiny eps, <1 for big eps
    # solve eps from r: r = 1/sqrt(1 + eps/ms)
    ms = x.double().pow(2).mean().item()
    eps = ms * (1 / r**2 - 1)
    return r, eps
base = torch.randn(4096, N, device=dev, dtype=torch.float64) * 0.02
print(f"mean-square of input {base.pow(2).mean().item():.3e}; 2^-7 = {2**-7:.4e}; fp32 eps = {torch.finfo(torch.float32).eps:.4e}")
for comp in (False, True):
    f = torch.compile(norm) if comp else norm
    for dt in (torch.float32, torch.bfloat16):
        for ac in (False, True):
            x = base.to(dt)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=ac), torch.no_grad():
                r, eps = scale_seen(f, x)
            print(f"compile={comp!s:5} input={str(dt):15} autocast={ac!s:5} -> out/ref0={r:.5f}  effective eps ~ {eps:.3e}")
