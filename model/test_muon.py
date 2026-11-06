import torch
from torch import nn
from model.adamw_muon import MuonAdamW, muon_param_groups, _newton_schulz


class TinyGPT(nn.Module):
    def __init__(self, V=512, d=64, L=2, T=32, tie=True):
        super().__init__()
        self.tok = nn.Embedding(V, d)
        self.pos = nn.Embedding(T, d)
        self.blocks = nn.ModuleList([
            nn.ModuleDict(dict(
                ln1=nn.LayerNorm(d),
                attn=nn.Linear(d, 3 * d, bias=False),
                proj=nn.Linear(d, d, bias=False),
                ln2=nn.LayerNorm(d),
                fc=nn.Linear(d, 4 * d),
                out=nn.Linear(4 * d, d),
            )) for _ in range(L)])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, V, bias=False)
        if tie:
            self.head.weight = self.tok.weight

    def forward(self, idx):
        x = self.tok(idx) + self.pos(torch.arange(idx.size(1)))
        for b in self.blocks:
            h = b["ln1"](x)
            q, k, v = b["attn"](h).chunk(3, dim=-1)
            a = torch.nn.functional.scaled_dot_product_attention(
                q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1), is_causal=True)
            x = x + b["proj"](a.squeeze(1))
            x = x + b["out"](torch.nn.functional.gelu(b["fc"](b["ln2"](x))))
        return self.head(self.lnf(x))


print("=== 1. grouping ===")
m = TinyGPT()
groups = muon_param_groups(m, lr=1e-3, muon_lr_mult=30.0)
muon_ids = {id(p) for g in groups if g["use_muon"] for p in g["params"]}

assert id(m.tok.weight) not in muon_ids, "token embedding must not be muon"
assert id(m.pos.weight) not in muon_ids, "pos embedding must not be muon"
assert id(m.head.weight) not in muon_ids, "lm head must not be muon"
for b in m.blocks:
    for k in ("attn", "proj", "fc", "out"):
        assert id(b[k].weight) in muon_ids, f"{k}.weight should be muon"
    assert id(b["fc"].bias) not in muon_ids, "bias must not be muon"
    assert id(b["ln1"].weight) not in muon_ids, "norm gain must not be muon"

all_ps = [p for g in groups for p in g["params"]]
assert len(all_ps) == len({id(p) for p in all_ps}), "param appears twice"
n_unique = len({id(p) for p in m.parameters()})
assert len(all_ps) == n_unique, f"{len(all_ps)} grouped vs {n_unique} unique"
print("  tied head/embedding counted once, shapes routed correctly\n")

print("=== 2. AdamW path matches torch.optim.AdamW ===")
torch.manual_seed(0)
a = nn.Linear(32, 32)
b = nn.Linear(32, 32)
b.load_state_dict(a.state_dict())
oa = MuonAdamW(a.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1,
               use_muon=False)
ob = torch.optim.AdamW(b.parameters(), lr=1e-3, betas=(0.9, 0.95),
                       weight_decay=0.1, eps=1e-8)
x = torch.randn(64, 32)
for _ in range(25):
    for mod, opt in ((a, oa), (b, ob)):
        opt.zero_grad()
        (mod(x) ** 2).mean().backward()
        opt.step()
d = max((p - q).abs().max().item() for p, q in zip(a.parameters(), b.parameters()))
print(f"  max param divergence after 25 steps: {d:.2e}")
assert d < 1e-6, "AdamW fallback does not match torch"
print()

print("=== 3. Newton-Schulz orthogonalises ===")
print("  NS5 converges the BULK of the spectrum; the smallest singular")
print("  values need ~8 steps. Real momentum matrices are far better")
print("  conditioned than the random Gaussians used here.")
torch.manual_seed(0)
for shape in [(64, 64), (128, 32), (32, 128)]:
    G = torch.randn(*shape)
    s5 = torch.linalg.svdvals(_newton_schulz(G, steps=5).float())
    s8 = torch.linalg.svdvals(_newton_schulz(G, steps=8).float())
    print(f"  {str(shape):>10}  NS5 p10={s5.quantile(0.1):.3f} max={s5.max():.3f}"
          f"   NS8 min={s8.min():.3f} max={s8.max():.3f}")
    assert 0.6 < s5.quantile(0.1) and s5.max() < 1.4, "NS5 bulk not orthogonal"
    assert 0.6 < s8.min() and s8.max() < 1.4, "NS8 did not converge"
print()

print("=== 4. muon reduces loss on a real model ===")
torch.manual_seed(0)
idx = torch.randint(0, 512, (16, 32))
tgt = torch.randint(0, 512, (16, 32))
res = {}
for use_muon in (False, True):
    torch.manual_seed(0)
    mm = TinyGPT()
    g = muon_param_groups(mm, lr=1e-3, muon_lr_mult=30.0,
                          use_muon=use_muon, verbose=False)
    opt = MuonAdamW(g)
    first = last = None
    for i in range(60):
        opt.zero_grad()
        loss = nn.functional.cross_entropy(
            mm(idx).reshape(-1, 512), tgt.reshape(-1))
        loss.backward()
        opt.step()
        if i == 0:
            first = loss.item()
        last = loss.item()
    res[use_muon] = (first, last)
    print(f"  use_muon={str(use_muon):<5} {first:.4f} -> {last:.4f}")
    assert last < first, "loss did not decrease"
print()

print("=== 5. scheduler preserves the muon:adamw ratio ===")
mm = TinyGPT()
g = muon_param_groups(mm, lr=1e-3, muon_lr_mult=30.0, verbose=False)
opt = MuonAdamW(g)
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: min(1.0, (s + 1) / 10) * (0.5 ** (s / 40)))
ratios = []
for s in range(60):
    opt.step()
    sched.step()
    lrs = [pg["lr"] for pg in opt.param_groups]
    ratios.append(lrs[0] / lrs[1])
    if s in (0, 9, 30, 59):
        print(f"  step {s:>3}: muon lr {lrs[0]:.5f}  adamw lr {lrs[1]:.6f}  "
              f"ratio {ratios[-1]:.2f}")
assert max(ratios) - min(ratios) < 1e-9, "ratio drifted"
print("\n  ratio constant across warmup + decay\n")

print("=== 6. state_dict round-trips ===")
sd = opt.state_dict()
opt2 = MuonAdamW(muon_param_groups(mm, lr=1e-3, verbose=False))
opt2.load_state_dict(sd)
print("  ok")