"""SFT example-aligned batcher + masked-loss unit tests.

Run: python -m sft.test_batcher
"""
import os
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

from model.lr_scheduler import resolve_warmup, schedule_factor
from sft.batcher import (IGNORE_INDEX, SFTEpochBatcher, SFTSplit,
                         levenshtein_copied, loss_targets, lowdiff_copied_mask,
                         pack_rows, pack_xy, response_p0_mask, supervised_mask)
from sft.tokenize import write_split

BOS, USR, ASST, END = 0, 201, 202, 203
T = 32


def ex(q, a):
    return [BOS, USR, *q, ASST, *a, END]


def write_bins(tmpdir, train, val):
    write_split(tmpdir, "train", train, BOS)
    write_split(tmpdir, "val", val, BOS)
    import json
    json.dump({"vocab_size": 32768, "bos_id": BOS, "usr_id": USR,
               "assistant_id": ASST, "end_id": END,
               "chars_per_token_val": 1.0},
              open(os.path.join(tmpdir, "meta.json"), "w"))


def check(cond, label):
    assert cond, label
    print(f"  ok: {label}")


def test_loss_mask():
    print("=== loss mask ===")
    ids = np.array(ex([10, 11, 12], [20, 21]), dtype=np.int64)
    # BOS USR 10 11 12 ASST 20 21 END
    sup = supervised_mask(ids, ASST, BOS)
    want = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)
    check(np.array_equal(sup, want), "supervise answer tokens + END only")
    x, y = pack_xy(ids, T, ASST, BOS, pad_id=BOS)
    # y[t] = ids[t+1] iff ids[t+1] is supervised
    for t in range(T):
        nxt = ids[t + 1] if t + 1 < len(ids) else None
        if nxt is not None and sup[t + 1]:
            check(y[t] == nxt, f"y[{t}] is answer/END token {nxt}")
        else:
            check(y[t] == IGNORE_INDEX, f"y[{t}] ignored")
    check((x[:len(ids)] == ids).all(), "x holds the full example")
    check((x[len(ids):] == BOS).all(), "right-pad with BOS")
    p0 = response_p0_mask(x, y, ASST)
    asst = int(np.flatnonzero(ids == ASST)[0])
    check(int(p0.sum()) == 1, "one p0 position per example")
    check(bool(p0[asst]), "p0 is the ASSISTANT slot (predicts first answer)")
    check(y[asst] == ids[asst + 1], "that target is the first answer token")

    a = ex([1, 2], [3, 4])
    b = ex([5, 6], [7, 8])
    packed = np.concatenate([a, b])
    xd, yd = pack_xy(packed, T, ASST, BOS, pad_id=BOS)
    p0d = response_p0_mask(xd, yd, ASST)
    check(int(p0d.sum()) == 2, "dense row: one p0 per packed example")


def test_dense_pack():
    print("=== dense pack (never truncate) ===")
    # lengths 8, 8, 20  with T=32 -> first two fit (16), third is own row
    a = ex([1, 2], [3, 4])             # 8
    b = ex([5, 6], [7, 8])             # 8
    c = ex(list(range(10, 22)), [9])   # 2+12+1+1+1 wait: BOS USR +12 + ASST +1 + END = 17
    lengths = [len(a), len(b), len(c)]
    check(all(L <= T for L in lengths), "fixtures fit T")
    rows = pack_rows(lengths, [0, 1, 2], T, dense=True)
    check(rows == [[0, 1], [2]], "dense groups whole examples without split")
    for row in rows:
        tot = sum(lengths[i] for i in row)
        check(tot <= T, f"row {row} length {tot} <= T")
    over = pack_rows([8, 8, 8], [0, 1, 2], 16, dense=True)
    check(over == [[0, 1], [2]], "fill then spill to next row")
    try:
        pack_rows([T + 1], [0], T, dense=False)
        check(False, "over-long must assert")
    except AssertionError:
        check(True, "over-long example asserts (never truncate)")


def test_epoch_shuffle():
    print("=== epoch shuffle ===")
    with tempfile.TemporaryDirectory() as d:
        train = [ex([50, 51 + i], [60, 61 + i]) for i in range(7)]
        val = [ex([70, 71], [80, 81])]
        write_bins(d, train, val)
        split = SFTSplit(d, "train")
        b = SFTEpochBatcher(split, T=T, B=2, accum=1, dense=False,
                            seed=47, assistant_id=ASST, bos_id=BOS,
                            shuffle=True, epochs=2)
        check(b.total_steps == 8,  # ceil(7/2)=4 per epoch × 2
              f"steps {b.total_steps} = 4/epoch × 2")
        seen = []
        for _ in range(b.steps_per_epoch[0]):
            x, y, done = b.next_micro()
            check(x.shape == (2, T) and y.shape == (2, T), "micro shape")
            seen.append(done)
        check(seen[-1] == 1, "first epoch completion flagged")
        # two epochs, different permutations (seed+epoch)
        o0 = np.random.RandomState(47).permutation(7)
        o1 = np.random.RandomState(48).permutation(7)
        check(not np.array_equal(o0, o1), "epoch seeds differ")


def test_schedule():
    print("=== warmup ratio / from-zero / decay-to-0 ===")
    check(resolve_warmup(0.05, 1000) == 50, "0.05 of 1000 = 50 steps")
    check(resolve_warmup(200, 1000) == 200, ">=1 is a step count")
    check(resolve_warmup(0, 1000) == 0, "0 disables warmup")
    f0 = schedule_factor(0, 100, warmup_steps=10, warmup_from_zero=True,
                         min_ratio=0.0)
    check(abs(f0) < 1e-12, f"warmup from zero at step 0, got {f0}")
    fpre = schedule_factor(0, 1000, warmup_steps=100, warmup_from_zero=False)
    check(abs(fpre - 0.01) < 1e-9, "pretrain warmup start unchanged")
    fend = schedule_factor(99, 100, warmup_steps=0, schedule="cosine",
                           min_ratio=0.0)
    check(abs(fend) < 1e-6, f"cosine min_ratio=0 ends at 0, got {fend}")
    fwsd = schedule_factor(99, 100, warmup_steps=0, schedule="wsd",
                           decay_frac=0.2, min_ratio=0.0)
    check(abs(fwsd) < 1e-6, f"wsd min_ratio=0 ends at 0, got {fwsd}")


def test_lowdiff_mask():
    print("=== lowdiff masking ===")
    d, match = levenshtein_copied([1, 2, 3, 4], [1, 2, 3, 4])
    check(d == 0 and match.tolist() == [True] * 4, "identical: dist 0, all match")
    d, match = levenshtein_copied([1, 2, 3, 4, 5], [3, 4])
    check(d == 3 and match.tolist() == [True, True],
          "extractive span is all matches (3 deletes)")
    d, match = levenshtein_copied([1, 2, 3], [9, 8, 7])
    check(d == 3 and match.tolist() == [False, False, False],
          "unrelated: no matches")
    d, match = levenshtein_copied([10, 11, 12, 13, 14], [10, 11, 99, 13, 14])
    check(d == 1 and match.tolist() == [True, True, False, True, True],
          "one substitution is not a copy")

    # 4/5 of the answer matches the prompt → at 0.8, mask those 4
    q, a = [10, 11, 12, 13, 14], [10, 11, 99, 13, 14]
    ids = np.array(ex(q, a), dtype=np.int64)
    copied = lowdiff_copied_mask(ids, ASST, BOS, USR, END, threshold=0.8)
    asst = int(np.flatnonzero(ids == ASST)[0])
    # answer tokens sit at asst+1 .. asst+5, END at asst+6
    want = np.zeros(len(ids), dtype=bool)
    want[asst + 1] = True
    want[asst + 2] = True
    want[asst + 4] = True
    want[asst + 5] = True
    check(np.array_equal(copied, want), "80% similar: mask the 4 copies, keep 99")
    check(not copied[asst + 6], "END is never copied")
    x, y = pack_xy(ids, T, ASST, BOS, pad_id=BOS, usr_id=USR, end_id=END,
                   lowdiff_masking=True, lowdiff_threshold=0.8)
    check(y[asst] == IGNORE_INDEX, "copied first token dropped from loss")
    check(y[asst + 2] == 99, "substituted token still supervised")
    check(y[asst + 5] == END, "END still supervised")
    p0 = response_p0_mask(x, y, ASST)
    check(int(p0.sum()) == 0, "p0 also dropped when first token is a copy")

    # 3/5 = 0.6 < 0.8 → do not mask anything
    a60 = [10, 11, 99, 98, 14]
    ids60 = np.array(ex(q, a60), dtype=np.int64)
    copied60 = lowdiff_copied_mask(ids60, ASST, BOS, USR, END, threshold=0.8)
    check(not copied60.any(), "below threshold: no tokens masked")
    x60, y60 = pack_xy(ids60, T, ASST, BOS, pad_id=BOS,
                       lowdiff_masking=True, lowdiff_threshold=0.8)
    asst60 = int(np.flatnonzero(ids60 == ASST)[0])
    check(y60[asst60] == 10, "below threshold: first answer still a target")

    # extractive: answer is a span of the prompt
    qe, ae = [1, 2, 3, 4, 5, 6], [3, 4, 5]
    ide = np.array(ex(qe, ae), dtype=np.int64)
    ce = lowdiff_copied_mask(ide, ASST, BOS, USR, END, threshold=0.8)
    ae0 = int(np.flatnonzero(ide == ASST)[0]) + 1
    check(ce[ae0:ae0 + 3].all() and not ce[ae0 + 3],
          "extractive answer fully masked, END kept")

    # unrelated answers keep full supervision
    idu = np.array(ex([1, 2, 3], [9, 8, 7]), dtype=np.int64)
    xu, yu = pack_xy(idu, T, ASST, BOS, pad_id=BOS, lowdiff_masking=True)
    xu0, yu0 = pack_xy(idu, T, ASST, BOS, pad_id=BOS, lowdiff_masking=False)
    check(np.array_equal(yu, yu0), "dissimilar pair: y matches unmasked")

    # dense: only the similar example is masked
    similar = ex([10, 11, 12, 13, 14], [10, 11, 12, 13, 14])
    other = ex([1, 2], [9, 8])
    packed = np.concatenate([similar, other])
    xd, yd = pack_xy(packed, T, ASST, BOS, pad_id=BOS, lowdiff_masking=True)
    a0 = int(np.flatnonzero(np.array(similar) == ASST)[0])
    a1 = len(similar) + int(np.flatnonzero(np.array(other) == ASST)[0])
    check(yd[a0] == IGNORE_INDEX, "dense: similar example copies masked")
    check(yd[a0 + 5] == END, "dense: similar END kept")
    check(yd[a1] == 9, "dense: dissimilar example unmasked")

    off = loss_targets(ids, ASST, BOS, lowdiff_masking=False)
    on = loss_targets(ids, ASST, BOS, lowdiff_masking=True)
    check(int(off.sum()) == int(on.sum()) + 4,
          "loss_targets drops exactly the copied tokens")


def test_grad_mask():
    print("=== CE ignore_index zeros prompt grads ===")
    ids = np.array(ex([10, 11], [20, 21]), dtype=np.int64)
    x, y = pack_xy(ids, 16, ASST, BOS)
    V = 256
    logits = torch.randn(1, 16, V, requires_grad=True)
    yt = torch.from_numpy(y).unsqueeze(0)
    loss = F.cross_entropy(logits.reshape(-1, V), yt.reshape(-1),
                           ignore_index=IGNORE_INDEX)
    loss.backward()
    g = logits.grad[0]
    for t in range(16):
        if y[t] == IGNORE_INDEX:
            check(float(g[t].abs().sum()) == 0.0, f"prompt/pad pos {t} grad 0")
        else:
            check(float(g[t].abs().sum()) > 0.0, f"answer pos {t} has grad")


def main():
    test_loss_mask()
    test_dense_pack()
    test_epoch_shuffle()
    test_schedule()
    test_lowdiff_mask()
    test_grad_mask()
    print("BATCHER PASS")


if __name__ == "__main__":
    main()
