"""Muon + AdamW in one optimizer, with the standard torch.optim API.

Muon orthogonalises the momentum matrix before applying it, which only makes
sense for weights where BOTH dimensions are feature dimensions -- the hidden
2D matrices. It must NOT touch:

  - embeddings and the LM head: one dimension is vocab, not features
  - biases, LayerNorm/RMSNorm gains: 1D
  - anything tied between the two

Shape alone cannot distinguish an embedding (V x d) from a hidden matrix
(d x d) -- both are 2D. So grouping walks MODULES, not just parameters.

Scheduling: put both kinds in one optimizer as separate param groups with
different `lr`. LambdaLR (and the cosine schedulers) multiply each group's
own initial_lr by the same factor, so the Muon:AdamW ratio is preserved for
free. Do not bake decay into the optimizer.
"""

import math

import torch
from torch import nn
from model.adamw_foreach import step_adamw_foreach

# --------------------------------------------------------------- grouping --
def muon_param_groups(model, lr=1e-3, muon_lr=None, muon_lr_mult=30.0,
                      weight_decay=0.1, use_muon=True, exclude_names=(),
                      verbose=True):
    """Returns param_groups ready for MuonAdamW(...).

    muon_lr defaults to lr * muon_lr_mult. Muon typically wants a much larger
    step than AdamW (often 20-50x) because the orthogonalised update has unit
    spectral norm regardless of gradient scale.
    """
    if muon_lr is None:
        muon_lr = lr * muon_lr_mult

    # params that must stay on AdamW regardless of shape
    forced = set()
    emb_shapes = set()
    for m in model.modules():
        if isinstance(m, nn.Embedding):
            for p in m.parameters(recurse=False):
                forced.add(id(p))
                emb_shapes.add(tuple(p.shape))
    # the LM head: a Linear whose weight matches an embedding's shape,
    # whether or not it is tied
    for m in model.modules():
        if isinstance(m, nn.Linear):
            if tuple(m.weight.shape) in emb_shapes:
                forced.add(id(m.weight))

    named = dict(model.named_parameters())
    for n in exclude_names:
        if n in named:
            forced.add(id(named[n]))

    muon_p, decay_p, nodecay_p = [], [], []
    seen = set()
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue                      # dedup: tied weights appear twice
        seen.add(id(p))
        if p.ndim < 2:
            nodecay_p.append(p)           # biases, norm gains
        elif id(p) in forced or not use_muon:
            decay_p.append(p)
        else:
            muon_p.append(p)

    groups = []
    if muon_p:
        groups.append(dict(params=muon_p, use_muon=True, lr=muon_lr,
                           weight_decay=weight_decay, momentum=0.95,
                           nesterov=True, ns_steps=5))
    if decay_p:
        groups.append(dict(params=decay_p, use_muon=False, lr=lr,
                           weight_decay=weight_decay))
    if nodecay_p:
        groups.append(dict(params=nodecay_p, use_muon=False, lr=lr,
                           weight_decay=0.0))

    if verbose:
        def cnt(ps):
            return sum(p.numel() for p in ps)
        print(f"muon    {len(muon_p):>3} tensors  {cnt(muon_p):>12,} params  "
              f"lr={muon_lr:g}")
        print(f"adamw   {len(decay_p):>3} tensors  {cnt(decay_p):>12,} params  "
              f"lr={lr:g}  wd={weight_decay:g}")
        print(f"nodecay {len(nodecay_p):>3} tensors  {cnt(nodecay_p):>12,} "
              f"params  lr={lr:g}  wd=0")
    return groups


# ------------------------------------------------------- orthogonalisation --
@torch.no_grad()
def _newton_schulz(G, steps=5, eps=1e-7):
    """Quintic Newton-Schulz iteration for the orthogonal factor of G.
    Runs in bf16: the iteration is contractive, so low precision is fine."""
    assert G.ndim >= 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    transposed = G.size(-2) > G.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X.to(G.dtype)


# ---------------------------------------------------------------- optimizer --
class MuonAdamW(torch.optim.Optimizer):
    """One optimizer, two rules, selected per param group by `use_muon`.

    Standard torch.optim API: state_dict, load_state_dict, zero_grad, and
    LR schedulers all work unmodified.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8,
                 weight_decay=0.0, momentum=0.95, nesterov=True, ns_steps=5,
                 use_muon=False):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps, use_muon=use_muon)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                self._step_muon(group)
            else:
                self._step_adamw(group)
        return loss

    def _step_muon(self, group):
        lr, wd = group["lr"], group["weight_decay"]
        mom, nesterov = group["momentum"], group["nesterov"]
        for p in group["params"]:
            if p.grad is None:
                continue
            g = p.grad
            st = self.state[p]
            if "buf" not in st:
                st["buf"] = torch.zeros_like(g)
            buf = st["buf"]
            buf.mul_(mom).add_(g)
            upd = g.add(buf, alpha=mom) if nesterov else buf

            flat = upd.reshape(upd.size(0), -1) if upd.ndim > 2 else upd
            o = _newton_schulz(flat, group["ns_steps"]).reshape(upd.shape)

            # scale by aspect ratio: taller matrices need a larger step
            scale = max(1.0, p.size(-2) / p.size(-1)) ** 0.5
            if wd:
                p.mul_(1 - lr * wd)
            p.add_(o, alpha=-lr * scale)

    def _step_adamw(self, group):
        step_adamw_foreach(group, self.state)