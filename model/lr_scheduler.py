"""Learning-rate schedules, PyTorch LRScheduler-compatible.

Multiplies each param group's OWN base_lr by a shared factor. That is what
keeps the Muon:AdamW ratio fixed -- Muon runs ~30x AdamW's rate, and a
scheduler that set an absolute lr would destroy that.

Schedules:

  cosine    warmup -> cosine decay to min_ratio.
            Hoffmann et al. 2022 (Chinchilla) report that the cosine cycle
            length must match the actual number of training steps; a run
            stopped early sits at a high LR mid-decay and looks worse than it
            is. So total_steps is a hyperparameter, not a guess.

  wsd       warmup -> constant -> short decay. Hu et al. 2024 (MiniCPM),
            Hägele et al. 2024. Does not require committing to a duration up
            front: hold the stable phase as long as you like and decay when
            you decide to stop. Also gives L(D) at several budgets from one
            run by branching the decay.

  constant  warmup -> flat. For LR-range tests.

Warmup is separate from the schedule because it addresses a different problem:
Adam's second-moment estimate is built from very few samples early on and its
step is correspondingly high-variance (Liu et al. 2020, RAdam). Pre-norm makes
warmup optional for stability (Xiong et al. 2020) but does nothing about this.
"""

import math

from torch.optim.lr_scheduler import LRScheduler


def resolve_warmup(warmup, total_steps):
    """Warmup length in steps. `warmup` in (0, 1) is a fraction of
    `total_steps` (0.05 = first 5%); 0 disables; >= 1 is a step count."""
    if warmup is None or warmup == 0:
        return 0
    if 0 < float(warmup) < 1:
        if not total_steps:
            return 0
        return max(1, int(round(float(warmup) * total_steps)))
    return int(warmup)


def schedule_factor(step, total_steps, warmup_steps=0, schedule="cosine",
                    min_ratio=0.1, decay_frac=0.2, decay_shape="linear",
                    warmup_from_zero=False):
    """Multiplier on base_lr at `step` (0-indexed). Pure function -- plot it,
    test it, no optimizer needed."""
    if warmup_steps and step < warmup_steps:
        if warmup_from_zero:
            return step / warmup_steps
        return (step + 1) / warmup_steps

    if schedule == "constant":
        return 1.0

    done = max(0, step - warmup_steps)
    span = max(1, total_steps - warmup_steps)

    if schedule == "cosine":
        t = min(1.0, done / max(1, span - 1))
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * t))

    if schedule == "wsd":
        decay_steps = max(1, int(decay_frac * span))
        stable = span - decay_steps
        if done < stable:
            return 1.0
        t = min(1.0, (done - stable) / max(1, decay_steps - 1))
        if decay_shape == "cosine":
            f = 0.5 * (1 + math.cos(math.pi * t))
        elif decay_shape == "1-sqrt":
            f = 1.0 - math.sqrt(t)
        else:                                   # linear
            f = 1.0 - t
        return min_ratio + (1 - min_ratio) * f

    raise ValueError(f"unknown schedule {schedule!r}")


class WarmupSchedule(LRScheduler):
    def __init__(self, optimizer, total_steps, warmup_steps=0,
                 schedule="cosine", min_ratio=0.1, decay_frac=0.2,
                 decay_shape="linear", last_epoch=-1, warmup_from_zero=False):
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.schedule = schedule
        self.min_ratio = min_ratio
        self.decay_frac = decay_frac
        self.decay_shape = decay_shape
        self.warmup_from_zero = warmup_from_zero
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        f = schedule_factor(self.last_epoch, self.total_steps,
                            self.warmup_steps, self.schedule,
                            self.min_ratio, self.decay_frac, self.decay_shape,
                            self.warmup_from_zero)
        return [base * f for base in self.base_lrs]

    def factor(self):
        return schedule_factor(self.last_epoch, self.total_steps,
                               self.warmup_steps, self.schedule,
                               self.min_ratio, self.decay_frac,
                               self.decay_shape, self.warmup_from_zero)


def plot_ascii(total=1000, rows=14, cols=68, **kw):
    """Sanity-check the shape without matplotlib."""
    vals = [schedule_factor(s, total, **kw) for s in range(total)]
    hi = max(vals)
    grid = [[" "] * cols for _ in range(rows)]
    for c in range(cols):
        v = vals[int(c * (total - 1) / (cols - 1))]
        r = rows - 1 - int(round((v / hi) * (rows - 1)))
        grid[r][c] = "*"
    print(f"  {hi:.3f} " + "-" * cols)
    for row in grid:
        print("        |" + "".join(row))
    print(f"  0.000 " + "-" * cols)


if __name__ == "__main__":
    import torch
    from torch import nn

    print("=== shapes ===")
    for name, kw in [("cosine", dict(schedule="cosine", warmup_steps=100)),
                     ("wsd", dict(schedule="wsd", warmup_steps=100,
                                  decay_frac=0.2)),
                     ("wsd 1-sqrt", dict(schedule="wsd", warmup_steps=100,
                                         decay_frac=0.2,
                                         decay_shape="1-sqrt"))]:
        print(f"\n{name}:")
        plot_ascii(1000, **kw)

    print("\n=== invariants ===")
    for sch in ("cosine", "wsd", "constant"):
        f0 = schedule_factor(0, 1000, 100, sch)
        fw = schedule_factor(99, 1000, 100, sch)
        fe = schedule_factor(999, 1000, 100, sch)
        assert abs(f0 - 0.01) < 1e-9, f"{sch} warmup start"
        assert abs(fw - 1.0) < 1e-9, f"{sch} warmup end"
        print(f"  {sch:<9} start {f0:.4f}  end-of-warmup {fw:.4f}  final {fe:.4f}")
    assert abs(schedule_factor(999, 1000, 100, "cosine") - 0.1) < 1e-3
    assert abs(schedule_factor(999, 1000, 100, "wsd") - 0.1) < 1e-3
    assert schedule_factor(500, 1000, 100, "wsd") == 1.0, "stable phase"
    print("  cosine and wsd both land on min_ratio; wsd is flat mid-run")

    print("\n=== group ratio preserved ===")
    m = nn.Linear(8, 8)
    opt = torch.optim.SGD([
        {"params": [m.weight], "lr": 0.03},
        {"params": [m.bias], "lr": 0.001},
    ])
    sched = WarmupSchedule(opt, total_steps=200, warmup_steps=20,
                           schedule="cosine")
    ratios = []
    for s in range(200):
        opt.step()
        sched.step()
        lrs = [g["lr"] for g in opt.param_groups]
        ratios.append(lrs[0] / lrs[1])
        if s in (0, 19, 100, 199):
            print(f"  step {s:>3}: muon {lrs[0]:.6f}  adamw {lrs[1]:.7f}  "
                  f"ratio {ratios[-1]:.2f}")
    assert max(ratios) - min(ratios) < 1e-9, "ratio drifted"
    print("  ratio constant to 1e-9")

    print("\n=== state_dict round-trip ===")
    sd = sched.state_dict()
    s2 = WarmupSchedule(opt, total_steps=200, warmup_steps=20)
    s2.load_state_dict(sd)
    assert s2.last_epoch == sched.last_epoch
    assert abs(s2.factor() - sched.factor()) < 1e-12
    print(f"  resumed at step {s2.last_epoch}, factor {s2.factor():.6f}")