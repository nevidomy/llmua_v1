"""Feed-forward hidden width: compute it, do not pass it.

The MLP parameter count depends on how many matrices the variant uses:

    GELU    W_in (d x h) + W_out (h x d)                  = 2dh
    SwiGLU  W_gate (d x h) + W_up (d x h) + W_down (h x d) = 3dh

So "expansion 4" means h = 4d for GELU but h = 8d/3 for SwiGLU -- Shazeer 2020
reduces the width by 2/3 precisely so the two are parameter-matched, and LLaMA
does the same. Passing 4d with SwiGLU silently makes every layer 12d^2 instead
of 8d^2, a 50% error in the MLP and 33% in the whole layer.

The useful invariant is the EFFECTIVE EXPANSION:

    eff = (3 if swiglu else 2) * h / (2d)

which is 4.0 for both a 4d GELU MLP and an 8d/3 SwiGLU. That is what
validate_ff_width checks, and it catches the 4d-with-SwiGLU mistake exactly.
"""

FF_MULTIPLE = 64
 
 
def ff_width_for(embeddings, use_swiglu, expansion=4.0, multiple=FF_MULTIPLE):
    """Nearest `multiple` to the parameter-matched target width.
 
    This is the width for every run on the scaling ray.
    """
    if embeddings <= 0 or expansion <= 0:
        raise ValueError(f"embeddings={embeddings} expansion={expansion}")
    target = expansion * embeddings * (2.0 / 3.0 if use_swiglu else 1.0)
    return max(multiple, int(round(target / multiple)) * multiple)
 
 
def gelu_width_matching(h_swiglu):
    """GELU width with MLP parameters identical to a SwiGLU of width h_swiglu.
 
    2*d*h_gelu == 3*d*h_swiglu  =>  h_gelu = 3*h_swiglu/2, exactly.
 
    Deliberately NOT rounded to a multiple of 64: exactness is the point, and
    the control arm is a throwaway ablation run, not part of the ray.
    """
    if h_swiglu <= 0 or h_swiglu % 2:
        raise ValueError(f"h_swiglu={h_swiglu} must be positive and even")
    return 3 * h_swiglu // 2
 
 
def effective_expansion(embeddings, ff_width, use_swiglu):
    """MLP params expressed as a GELU-equivalent expansion factor."""
    return (3 if use_swiglu else 2) * ff_width / (2.0 * embeddings)
 
 
def mlp_params(embeddings, ff_width, use_swiglu):
    """MLP parameters per layer."""
    return (3 if use_swiglu else 2) * embeddings * ff_width
 
 
def validate_ff_width(embeddings, ff_width, use_swiglu, expansion=4.0,
                      tol=0.15):
    """Raise with an actionable message if ff_width is implausible.
 
    tol is deliberately loose: it exists to catch the 4d-with-SwiGLU mistake
    (which is 50% off) and typos, not to enforce exact rounding. Widths from
    ff_width_for and gelu_width_matching both pass.
    """
    if ff_width <= 0:
        raise ValueError(f"ff_width={ff_width} must be positive")
 
    if ff_width % 8:
        raise ValueError(
            f"ff_width={ff_width} is not a multiple of 8; bf16 tensor cores "
            f"will not engage. Nearest: {round(ff_width / 8) * 8}")
 
    if ff_width < embeddings:
        raise ValueError(
            f"ff_width={ff_width} < embeddings={embeddings}: the MLP would "
            f"be a bottleneck, not an expansion. Did you mean "
            f"{ff_width_for(embeddings, use_swiglu, expansion)}?")
 
    eff = effective_expansion(embeddings, ff_width, use_swiglu)
    if abs(eff - expansion) / expansion > tol:
        want = ff_width_for(embeddings, use_swiglu, expansion)
        hint = ""
        if use_swiglu and abs(eff - 1.5 * expansion) / expansion < 0.05:
            hint = ("  This is exactly the 4d-with-SwiGLU mistake: three "
                    "matrices, not two.")
        raise ValueError(
            f"ff_width={ff_width} with use_swiglu={use_swiglu} at d="
            f"{embeddings} gives effective expansion {eff:.2f}, not "
            f"{expansion:.2f} ({mlp_params(embeddings, ff_width, use_swiglu):,}"
            f" MLP params = {(3 if use_swiglu else 2)*ff_width/embeddings:.2f}"
            f" d^2 per layer, want {2*expansion:.2f}). Use "
            f"ff_width={want}.{hint}")
    return ff_width
 
 
def resolve_ff_width(embeddings, ff_width, use_swiglu, expansion=4.0):
    """None -> compute; a value -> validate. Use this in __init__.
 
    The GELU width is derived from the SwiGLU width at the same d, so the two
    variants are parameter-matched by construction rather than by both
    happening to round the same way.
    """
    if ff_width is not None:
        return validate_ff_width(embeddings, ff_width, use_swiglu, expansion)
    h_swiglu = ff_width_for(embeddings, True, expansion)
    return h_swiglu if use_swiglu else gelu_width_matching(h_swiglu)
 
 
if __name__ == "__main__":
    SHAPES = (("P1", 128), ("P2", 192), ("P3", 256),
              ("P4", 384), ("Target", 768))
    DOC_H = {"P1": 320, "P2": 512, "P3": 704, "P4": 1024, "Target": 2048}
 
    print("=== ray widths (multiple of 64) ===")
    print(f"  {'shape':>7}{'d':>6}{'8d/3':>9}{'h_sw':>7}{'dev':>8}"
          f"{'eff':>7}{'doc':>6}{'':>4}")
    for name, d in SHAPES:
        h = ff_width_for(d, True)
        t = 4 * d * 2 / 3
        ok = "ok" if h == DOC_H[name] else "MISMATCH"
        print(f"  {name:>7}{d:>6}{t:>9.1f}{h:>7}{(h-t)/t*100:>+7.1f}%"
              f"{effective_expansion(d, h, True):>7.2f}{DOC_H[name]:>6}  {ok}")
 
    print("\n=== SwiGLU ablation: control derived, not rounded ===")
    print(f"  {'shape':>7}{'h_sw':>7}{'h_gelu':>8}{'sw par':>11}"
          f"{'gelu par':>11}{'ratio':>8}")
    for name, d in SHAPES:
        hs = ff_width_for(d, True)
        hg = gelu_width_matching(hs)
        ps = mlp_params(d, hs, True)
        pg = mlp_params(d, hg, False)
        assert ps == pg, (name, ps, pg)
        print(f"  {name:>7}{hs:>7}{hg:>8}{ps:>11,}{pg:>11,}{ps/pg:>8.3f}")
 
    print("\n=== independent rounding, for contrast (do NOT use) ===")
    print(f"  {'shape':>7}{'h_sw':>7}{'h_gelu':>8}{'ratio':>8}")
    for name, d in SHAPES:
        hs = ff_width_for(d, True)
        hg = ff_width_for(d, False)
        r = mlp_params(d, hs, True) / mlp_params(d, hg, False)
        flag = "" if abs(r - 1) < 1e-9 else "  <-- unmatched"
        print(f"  {name:>7}{hs:>7}{hg:>8}{r:>8.3f}{flag}")
 
    print("\n=== validation ===")
    cases = [
        (128, ff_width_for(128, True), True, "ray swiglu"),
        (128, gelu_width_matching(ff_width_for(128, True)), False,
         "derived gelu control"),
        (128, 512, False, "4d GELU"),
        (128, 512, True, "4d with SwiGLU  <-- the mistake"),
        (128, 341, True, "not a multiple of 8"),
        (128, 64, True, "narrower than d"),
        (768, 2048, True, "target shape"),
    ]
    for d, h, sw, label in cases:
        try:
            validate_ff_width(d, h, sw)
            print(f"  OK    {label}")
        except ValueError as e:
            print(f"  RAISE {label}\n          {str(e)[:130]}")
 
    print("\n=== resolve_ff_width ===")
    for d, v, sw in ((768, None, True), (768, None, False), (768, 2048, True)):
        print(f"  {str(v):>5} swiglu={str(sw):<6} at d={d} -> "
              f"{resolve_ff_width(d, v, sw)}")
