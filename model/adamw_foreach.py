"""Foreach AdamW for MuonAdamW.

The per-parameter loop issues ~7 kernel launches and 2 allocations per tensor
per step. At 91 tensors that is ~640 launches and ~180 allocations, and for a
small model every kernel is microseconds -- the step becomes launch-bound.
torch.optim.AdamW defaults to the foreach path for exactly this reason.

Two changes:

  1. torch._foreach_* ops, so the launch count is ~9 regardless of how many
     tensors are in the group.
  2. no m_hat / v_hat temporaries. Fold bias correction into the step size and
     the denominator instead:

         step = lr / bc1
         denom = sqrt(v)/sqrt(bc2) + eps
         p -= step * m / denom

     which is algebraically identical to p -= lr * m_hat / (sqrt(v_hat) + eps)
     up to where eps is added -- matching what torch.optim.AdamW does.
"""

import math

import torch


def step_adamw_foreach(group, state):
    lr, wd, eps = group["lr"], group["weight_decay"], group["eps"]
    b1, b2 = group["betas"]

    ps, gs, ms, vs = [], [], [], []
    for p in group["params"]:
        if p.grad is None:
            continue
        st = state[p]
        if "step" not in st:
            st["step"] = 0
            st["m"] = torch.zeros_like(p)
            st["v"] = torch.zeros_like(p)
        st["step"] += 1
        ps.append(p)
        gs.append(p.grad)
        ms.append(st["m"])
        vs.append(st["v"])
    if not ps:
        return

    t = state[ps[0]]["step"]                 # all tensors in a group share t
    bc1 = 1 - b1 ** t
    bc2 = 1 - b2 ** t

    if wd:
        torch._foreach_mul_(ps, 1 - lr * wd)          # decoupled decay

    torch._foreach_mul_(ms, b1)
    torch._foreach_add_(ms, gs, alpha=1 - b1)
    torch._foreach_mul_(vs, b2)
    torch._foreach_addcmul_(vs, gs, gs, value=1 - b2)

    denom = torch._foreach_sqrt(vs)
    torch._foreach_div_(denom, math.sqrt(bc2))
    torch._foreach_add_(denom, eps)

    torch._foreach_addcdiv_(ps, ms, denom, value=-lr / bc1)


if __name__ == "__main__":
    import time
    from torch import nn
    from model.adamw_muon import MuonAdamW

    class Foreach(MuonAdamW):
        def _step_adamw(self, group):
            step_adamw_foreach(group, self.state)

    # --- 1. numerically identical to torch.optim.AdamW ---------------------
    print("=== equivalence after 50 steps ===")
    torch.manual_seed(0)
    ref = nn.Sequential(*[nn.Linear(64, 64) for _ in range(8)])
    a = nn.Sequential(*[nn.Linear(64, 64) for _ in range(8)])
    b = nn.Sequential(*[nn.Linear(64, 64) for _ in range(8)])
    for m in (a, b):
        m.load_state_dict(ref.state_dict())

    oa = torch.optim.AdamW(a.parameters(), lr=1e-3, betas=(0.9, 0.95),
                           weight_decay=0.1, eps=1e-8)
    ob = Foreach([{"params": list(b.parameters()), "use_muon": False,
                   "lr": 1e-3, "weight_decay": 0.1}],
                 betas=(0.9, 0.95), eps=1e-8)
    x = torch.randn(32, 64)
    for _ in range(50):
        for m, o in ((a, oa), (b, ob)):
            o.zero_grad()
            (m(x) ** 2).mean().backward()
            o.step()
    d = max((p - q).abs().max().item()
            for p, q in zip(a.parameters(), b.parameters()))
    print(f"  vs torch.optim.AdamW: max divergence {d:.2e}")
    assert d < 1e-6

    # --- 2. and to the loop version ---------------------------------------
    c = nn.Sequential(*[nn.Linear(64, 64) for _ in range(8)])
    c.load_state_dict(ref.state_dict())
    oc = MuonAdamW([{"params": list(c.parameters()), "use_muon": False,
                     "lr": 1e-3, "weight_decay": 0.1}],
                   betas=(0.9, 0.95), eps=1e-8)
    for _ in range(50):
        oc.zero_grad()
        (c(x) ** 2).mean().backward()
        oc.step()
    d2 = max((p - q).abs().max().item()
             for p, q in zip(b.parameters(), c.parameters()))
    print(f"  vs the loop version : max divergence {d2:.2e}")
    assert d2 < 1e-6

    # --- 3. step time vs tensor count -------------------------------------
    print("\n=== optimizer step time (CPU dispatch, proxy for launch count) ===")
    print(f"  {'tensors':>9}{'loop':>10}{'foreach':>10}{'torch':>10}{'speedup':>9}")
    for n in (16, 48, 91, 182):
        mods = nn.Sequential(*[nn.Linear(32, 32) for _ in range(n // 2)])
        params = list(mods.parameters())
        res = {}
        for label, Opt in (("loop", MuonAdamW), ("foreach", Foreach)):
            o = Opt([{"params": params, "use_muon": False, "lr": 1e-3,
                      "weight_decay": 0.1}])
            for p in params:
                p.grad = torch.randn_like(p)
            for _ in range(5):
                o.step()
            t0 = time.perf_counter()
            for _ in range(100):
                o.step()
            res[label] = (time.perf_counter() - t0) / 100 * 1e3
        o = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.1)
        for _ in range(5):
            o.step()
        t0 = time.perf_counter()
        for _ in range(100):
            o.step()
        res["torch"] = (time.perf_counter() - t0) / 100 * 1e3
        print(f"  {len(params):>9}{res['loop']:>9.3f}m{res['foreach']:>9.3f}m"
              f"{res['torch']:>9.3f}m{res['loop']/res['foreach']:>8.1f}x")
    print("\n  ms per step. On GPU the gap is larger: each of the loop's")
    print("  ~7 launches per tensor costs ~5-10us of launch overhead that")
    print("  the foreach path pays once for the whole group.")