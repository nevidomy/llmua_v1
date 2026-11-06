"""Parameter-count table: model factories x enable-flag combinations.

    python -m analysis.models_sizes

Rows: all 16 on/off combinations of the four ablation flags.
Cell: total params / params without embeddings, in millions.

Note the SwiGLU/GELU rows come out (nearly) equal -- ff_width.py derives
the GELU width from the SwiGLU width so the two are parameter-matched by
construction; only bias/norm bookkeeping moves the count.
"""

import itertools

import torch

from model.transformer import (model_P1, model_P2, model_P3, model_P4,
                               model_target)

FACTORIES = {"P1": model_P1, "P2": model_P2, "P3": model_P3,
             "P4": model_P4, "target": model_target}
FLAGS = ("enable_init", "enable_swiglu", "enable_rms_norm",
         "enable_qk_norm")
SHORT = {"enable_init": "init", "enable_swiglu": "swiglu",
         "enable_rms_norm": "rms", "enable_qk_norm": "qk"}


def counts(factory, combo):
    """(total, without embeddings) in raw params. meta device: shapes only,
    no memory, no real init."""
    with torch.device("meta"):
        m = factory(**dict(zip(FLAGS, combo)))
    return m.num_params(), m.num_params_no_embedings()


def label(combo):
    on = [SHORT[f] for f, v in zip(FLAGS, combo) if v]
    return "+".join(on) if on else "(all off)"


def main():
    combos = list(itertools.product([False, True], repeat=4))
    cells = {(name, c): counts(f, c)
             for name, f in FACTORIES.items() for c in combos}

    lw = max(len(label(c)) for c in combos) + 2
    cw = 15
    print("cell: total / no-embed, millions of params\n")
    print(" " * lw + "".join(f"{n:>{cw}}" for n in FACTORIES))
    for c in combos:
        row = f"{label(c):<{lw}}"
        for n in FACTORIES:
            t, e = cells[n, c]
            row += f"{t/1e6:6.2f}/{e/1e6:<6.2f}  "
        print(row)


if __name__ == "__main__":
    main()
