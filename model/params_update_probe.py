"""Update-to-parameter probe.
 
Measures RMS(delta_p) / RMS(p) per tensor across one optimizer step, which is
the cheapest direct answer to "is the learning rate right for the scale these
weights are actually at". Neither the loss curve nor the grad norm says this.
 
  ~1e-3   healthy
  ~1e-2   thrashing -- LR too high for this parameter's scale
  ~1e-5   barely moving -- LR too low, or the parameter is effectively dead
 
Note RMS(dp)/RMS(p) == ||dp||/||p||: both divide by sqrt(numel).
 
Usage:
 
    probe = UpdateToParamProbe(model)
    ...
    loss.backward()
    probe.snapshot()                    # immediately BEFORE opt.step()
    opt.step()
    if step % 100 == 0:
        stats = probe.measure()         # immediately AFTER opt.step()
        api.post(step, stats["flat"])
 
snapshot() costs one clone of the tracked parameters (110M in fp32 == 440MB
transient), so only call it on steps you intend to measure.
 
Role names are derived from the parameter path, not hardcoded: `blocks.3.
mha.qkv.weight` becomes block 3, role `mha.qkv.weight`. Renaming modules or
swapping GELU-MLP for SwiGLU changes the role names automatically rather than
silently dropping tensors from the report.
"""
 
import re
import statistics
from collections import defaultdict
 
import torch
 
_BLOCK_RE = re.compile(r"^(?:blocks|layers|h)\.(\d+)\.(.+)$")
 
 
def _split(name):
    """'blocks.3.mha.qkv.weight' -> (3, 'mha.qkv.weight');
       'embed.weight'            -> (None, 'embed.weight')"""
    m = _BLOCK_RE.match(name)
    return (int(m.group(1)), m.group(2)) if m else (None, name)
 
 
def _agg(vals):
    if not vals:
        return None
    s = sorted(vals)
    return {
        "mean": statistics.fmean(s),
        "median": s[len(s) // 2],
        "min": s[0],
        "max": s[-1],
        "n": len(s),
    }
 
 
class UpdateToParamProbe:
    def __init__(self, model, include=None, exclude=None):
        """include/exclude: substring filters on parameter names."""
        self.model = model
        self.include = include
        self.exclude = exclude
        self._saved = None
 
    def _tracked(self):
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if self.include and self.include not in n:
                continue
            if self.exclude and self.exclude in n:
                continue
            yield n, p
 
    def snapshot(self):
        """Call immediately BEFORE optimizer.step()."""
        self._saved = {n: p.detach().clone() for n, p in self._tracked()}
 
    @torch.no_grad()
    def measure(self, clear=True):
        """Call immediately AFTER optimizer.step().
 
        Returns:
          {
            "summary":  {median, min, max, p10, p90, spread, n},
            "top":      {name: ratio}                 non-block tensors
            "blocks":   [{role: ratio, ...}, ...]     index == block number
            "by_role":  {role: {mean, median, min, max, n}}
            "param_rms":  same shape as "by_role", RMS of the parameter
            "update_rms": same shape as "by_role", RMS of the update
            "flat":     {"update_ratio/...": float, ...}  ready to POST
          }
        Parameters that are exactly zero (freshly zero-initialised biases) are
        skipped: the ratio is undefined, not zero.
        """
        assert self._saved is not None, "call snapshot() before opt.step()"
 
        ratio, prms, urms = {}, {}, {}
        for n, p in self._tracked():
            old = self._saved.get(n)
            if old is None:
                continue
            new = p.detach().float()
            old = old.float()
            k = new.numel() ** 0.5
            pr = (old.norm() / k).item()
            ur = ((new - old).norm() / k).item()
            prms[n], urms[n] = pr, ur
            if pr > 0.0:
                ratio[n] = ur / pr
        if clear:
            self._saved = None
 
        top, blocks, by_role = {}, defaultdict(dict), defaultdict(list)
        pr_role, ur_role = defaultdict(list), defaultdict(list)
        for n, r in ratio.items():
            b, role = _split(n)
            if b is None:
                top[role] = r
            else:
                blocks[b][role] = r
            by_role[role].append(r)
            pr_role[role].append(prms[n])
            ur_role[role].append(urms[n])
 
        vals = sorted(ratio.values())
        n = len(vals)
        summary = {
            "median": vals[n // 2], "min": vals[0], "max": vals[-1],
            "p10": vals[int(0.1 * n)], "p90": vals[int(0.9 * n)],
            "spread": vals[-1] / vals[0] if vals[0] > 0 else float("inf"),
            "n": n,
        } if n else {}
 
        block_list = [blocks[i] for i in sorted(blocks)] if blocks else []
 
        out = {
            "summary": summary,
            "top": top,
            "blocks": block_list,
            "by_role": {k: _agg(v) for k, v in by_role.items()},
            "param_rms": {k: _agg(v) for k, v in pr_role.items()},
            "update_rms": {k: _agg(v) for k, v in ur_role.items()},
        }
        out["flat"] = {k: float(v) for k, v in self.flatten(out).items()}
        return out
 
    @staticmethod
    def flatten(stats):
        """One scalar per series. Three metric families so they can go on
        separate graphs: update_ratio, param_rms, update_rms."""
        f = {}
        for k, v in stats["summary"].items():
            f[f"update_ratio/summary/{k}"] = float(v)
        for k, v in stats["top"].items():
            f[f"update_ratio/top/{k}"] = v
        for i, blk in enumerate(stats["blocks"]):
            for role, v in blk.items():
                f[f"update_ratio/block{i:02d}/{role}"] = v
            if blk:
                f[f"update_ratio/block_mean/{i:02d}"] = statistics.fmean(
                    blk.values())
        for fam in ("by_role", "param_rms", "update_rms"):
            prefix = {"by_role": "update_ratio/role"}.get(fam, fam)
            for role, a in stats[fam].items():
                if a:
                    f[f"{prefix}/{role}/mean"] = a["mean"]
                    if fam == "by_role":
                        f[f"{prefix}/{role}/min"] = a["min"]
                        f[f"{prefix}/{role}/max"] = a["max"]
        return f
 
 
def print_report(stats, top=4, file=None):
    s = stats["summary"]
    print(f"  n={s['n']}  median {s['median']:.2e}  "
          f"p10 {s['p10']:.2e}  p90 {s['p90']:.2e}  spread {s['spread']:.0f}x",
          file=file)
    rows = sorted(stats["by_role"].items(), key=lambda kv: kv[1]["mean"])
    print(f"  {'role':<24}{'mean':>10}{'min':>10}{'max':>10}{'n':>4}",
          file=file)
    for role, a in rows:
        flag = ("  thrashing" if a["max"] > 1e-2 else
                "  barely training" if a["max"] < 1e-5 else "")
        print(f"  {role:<24}{a['mean']:>10.2e}{a['min']:>10.2e}"
              f"{a['max']:>10.2e}{a['n']:>4}{flag}", file=file)
    if stats["blocks"]:
        means = [statistics.fmean(b.values()) for b in stats["blocks"] if b]
        print(f"  per-block mean, first->last: "
              + "  ".join(f"{m:.1e}" for m in means), file=file)