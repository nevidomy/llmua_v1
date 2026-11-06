"""Example-aligned SFT batching over packed {split}.bin + offsets/lengths.

Pretrain draws random token windows and can start mid-question. SFT never
does that: each example is sliced with offsets/lengths, optionally packed
dense (whole prompts only), then padded to context width.

Loss mask: next-token CE is computed only on {answer}<END> -- every token
after ASSISTANT, including END. Prompt tokens (and padding) use ignore_index.
"""
import math
import os

import numpy as np

IGNORE_INDEX = -100
LOWDIFF_THRESHOLD = 0.8


def resolve_data_dir(data_dir):
    """Existing path, or a name under ``{root}/sft_data/bins/{name}``."""
    if not data_dir:
        raise ValueError("SFT data_dir is required")
    from data.index import get_root
    data_dir = os.path.expanduser(os.path.expandvars(str(data_dir)))
    if os.path.isdir(data_dir):
        return os.path.abspath(data_dir)
    under = os.path.join(get_root(), "sft_data", "bins", data_dir)
    if os.path.isdir(under):
        return os.path.abspath(under)
    return os.path.abspath(data_dir)


class SFTSplit:
    """Memory-map one split: concatenated uint16 ids + example index."""

    def __init__(self, data_dir, split):
        data_dir = resolve_data_dir(data_dir)
        bin_path = os.path.join(data_dir, f"{split}.bin")
        off_path = os.path.join(data_dir, f"{split}.offsets.npy")
        len_path = os.path.join(data_dir, f"{split}.lengths.npy")
        assert os.path.isfile(bin_path), f"missing {bin_path}"
        assert os.path.isfile(off_path), f"missing {off_path}"
        assert os.path.isfile(len_path), f"missing {len_path}"
        self.data_dir = data_dir
        self.split = split
        self.bin_path = bin_path
        self._data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.offsets = np.load(off_path)
        self.lengths = np.load(len_path)
        assert len(self.offsets) == len(self.lengths), (
            f"{split}: offsets ({len(self.offsets)}) vs lengths "
            f"({len(self.lengths)})")
        self.n = int(len(self.lengths))

    def example(self, i):
        o, n = int(self.offsets[i]), int(self.lengths[i])
        return np.asarray(self._data[o:o + n], dtype=np.int64)


def _segments(ids, bos_id):
    """BOS-delimited [lo, hi) ranges. Dense rows have one span per example."""
    ids = np.asarray(ids, dtype=np.int64)
    L = int(ids.size)
    starts = np.flatnonzero(ids == bos_id)
    if starts.size == 0 or int(starts[0]) != 0:
        starts = np.concatenate((np.array([0], dtype=np.int64), starts))
    ends = np.append(starts[1:], L)
    return zip(starts.tolist(), ends.tolist())


def supervised_mask(ids, assistant_id, bos_id):
    """True at tokens that are {answer} or END (targets of the masked loss).

    Walks BOS-delimited segments so dense packing (several prompts in one
    row, each starting with BOS) is the same rule as a single example.
    """
    ids = np.asarray(ids, dtype=np.int64)
    sup = np.zeros(int(ids.size), dtype=bool)
    for lo, hi in _segments(ids, bos_id):
        asst = np.flatnonzero(ids[lo:hi] == assistant_id)
        assert asst.size == 1, (
            f"segment [{lo}:{hi}] has {asst.size} ASSISTANT tokens, want 1")
        start = lo + int(asst[0]) + 1
        assert start < hi, "example has no answer tokens (ASSISTANT at end)"
        sup[start:hi] = True
    return sup


def levenshtein_copied(prompt, answer):
    """Edit-align `answer` to `prompt` (Wagner–Fischer).

    Returns (distance, match) where match[j] is True iff answer[j] is an
    equal token aligned to some prompt token on an optimal path.
    """
    a = [int(x) for x in prompt]
    b = [int(x) for x in answer]
    n, m = len(a), len(b)
    if m == 0:
        return n, np.zeros(0, dtype=bool)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        ai = a[i - 1]
        row, prev = dp[i], dp[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            row[j] = min(prev[j] + 1, row[j - 1] + 1, prev[j - 1] + cost)
    match = np.zeros(m, dtype=bool)
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if a[i - 1] == b[j - 1] else 1
            if dp[i][j] == dp[i - 1][j - 1] + cost:
                if cost == 0:
                    match[j - 1] = True
                i -= 1
                j -= 1
                continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1
    return dp[n][m], match


def lowdiff_copied_mask(ids, assistant_id, bos_id, usr_id=201, end_id=203,
                        threshold=LOWDIFF_THRESHOLD):
    """True at answer tokens copied from the prompt, when similar enough.

    Per BOS segment: Levenshtein-align prompt tokens (USR..ASSISTANT,
    exclusive) to answer tokens (ASSISTANT..END, exclusive of END). If
    matches / len(answer) >= threshold, those matched tokens are copies.
    END is never marked. Below the threshold nothing is marked — a few
    accidental shared tokens should still receive loss.
    """
    ids = np.asarray(ids, dtype=np.int64)
    copied = np.zeros(int(ids.size), dtype=bool)
    thr = float(threshold)
    specials = {int(bos_id), int(usr_id), int(assistant_id), int(end_id)}
    for lo, hi in _segments(ids, bos_id):
        asst = np.flatnonzero(ids[lo:hi] == assistant_id)
        assert asst.size == 1, (
            f"segment [{lo}:{hi}] has {asst.size} ASSISTANT tokens, want 1")
        asst = lo + int(asst[0])
        usr = np.flatnonzero(ids[lo:asst] == usr_id)
        q0 = (lo + int(usr[-1]) + 1) if usr.size else lo
        prompt = [int(t) for t in ids[q0:asst] if int(t) not in specials]
        ans_pos = []
        for p in range(asst + 1, hi):
            tid = int(ids[p])
            if tid == end_id or tid == bos_id:
                break
            if tid in specials:
                continue
            ans_pos.append(p)
        if not ans_pos:
            continue
        answer = [int(ids[p]) for p in ans_pos]
        _dist, match = levenshtein_copied(prompt, answer)
        if match.size and float(match.sum()) / float(match.size) >= thr:
            for p, is_copy in zip(ans_pos, match.tolist()):
                if is_copy:
                    copied[p] = True
    return copied


def loss_targets(ids, assistant_id, bos_id, usr_id=201, end_id=203,
                 lowdiff_masking=False, lowdiff_threshold=LOWDIFF_THRESHOLD):
    """Supervised positions after optional lowdiff copy-masking."""
    sup = supervised_mask(ids, assistant_id, bos_id)
    if lowdiff_masking:
        sup = sup & ~lowdiff_copied_mask(
            ids, assistant_id, bos_id, usr_id=usr_id, end_id=end_id,
            threshold=lowdiff_threshold)
    return sup


def response_p0_mask(x, y, assistant_id, ignore_index=IGNORE_INDEX):
    """True where the target is the first answer token of a response.

    pack_xy puts that token at the ASSISTANT position (y[t] = ids[t+1]
    when x[t] is ASSISTANT). Dense rows have one such position per
    packed example. Works for numpy or torch.
    """
    return (x == assistant_id) & (y != ignore_index)


def pack_xy(ids, T, assistant_id, bos_id, pad_id=0,
            ignore_index=IGNORE_INDEX, usr_id=201, end_id=203,
            lowdiff_masking=False, lowdiff_threshold=LOWDIFF_THRESHOLD):
    """Pad one packed row (one or more full examples) to length T.

    x[t] predicts y[t] = ids[t+1] when that next token is in {answer}<END>;
    every other position is ignore_index. The last real token has no next
    token, so it is never a target. lowdiff_masking also drops answer
    tokens that the edit alignment marks as copies of the prompt.
    """
    ids = np.asarray(ids, dtype=np.int64)
    L = int(ids.size)
    assert 2 <= L <= T, (
        f"packed length {L} does not fit context_width {T} "
        f"(never truncate; drop over-long examples at tokenize time)")
    sup = loss_targets(ids, assistant_id, bos_id, usr_id=usr_id,
                       end_id=end_id, lowdiff_masking=lowdiff_masking,
                       lowdiff_threshold=lowdiff_threshold)
    x = np.full(T, pad_id, dtype=np.int64)
    y = np.full(T, ignore_index, dtype=np.int64)
    x[:L] = ids
    # y[t] supervises ids[t+1]
    tgt = ids[1:]
    take = sup[1:]
    y[:L - 1][take] = tgt[take]
    return x, y


def pack_rows(lengths, order, T, dense):
    """Group example indices into packed rows. Never splits an example.

    dense=False: one example per row.
    dense=True: concatenate as many full examples as fit in T.
    """
    lengths = np.asarray(lengths)
    rows = []
    if not dense:
        for i in order:
            L = int(lengths[i])
            assert L <= T, f"example {int(i)} length {L} > context {T}"
            rows.append([int(i)])
        return rows
    buf, used = [], 0
    for i in order:
        L = int(lengths[i])
        assert L <= T, f"example {int(i)} length {L} > context {T}"
        if buf and used + L > T:
            rows.append(buf)
            buf, used = [int(i)], L
        else:
            buf.append(int(i))
            used += L
    if buf:
        rows.append(buf)
    return rows


def plan_steps(lengths, T, batch, accum, epochs, dense, seed, shuffle=True):
    """Optimizer steps for `epochs` of the split (permutation known a priori).

    Dense packing can yield a different row count per epoch, so each epoch
    is simulated. Returns (total_steps, steps_per_epoch list).
    """
    n = int(len(lengths))
    assert n > 0, "empty SFT split"
    per_step = batch * accum
    assert per_step > 0
    spe = []
    for e in range(epochs):
        if shuffle:
            order = np.random.RandomState(seed + e).permutation(n)
        else:
            order = np.arange(n)
        n_rows = len(pack_rows(lengths, order, T, dense))
        spe.append(max(1, int(math.ceil(n_rows / per_step))))
    return int(sum(spe)), spe


class SFTEpochBatcher:
    """Yields (x, y) micro-batches of shape (B, T).

    Each epoch consumes every example once (plus ignore-padded leftover
    slots so the last optimizer step of the epoch has a full batch).
    """

    def __init__(self, split: SFTSplit, T, B, accum, dense, seed,
                 assistant_id, bos_id, shuffle=True, pad_id=0,
                 ignore_index=IGNORE_INDEX, epochs=1, usr_id=201,
                 end_id=203, lowdiff_masking=False,
                 lowdiff_threshold=LOWDIFF_THRESHOLD):
        self.split = split
        self.T = int(T)
        self.B = int(B)
        self.accum = int(accum)
        self.dense = bool(dense)
        self.seed = int(seed)
        self.assistant_id = int(assistant_id)
        self.bos_id = int(bos_id)
        self.usr_id = int(usr_id)
        self.end_id = int(end_id)
        self.lowdiff_masking = bool(lowdiff_masking)
        self.lowdiff_threshold = float(lowdiff_threshold)
        self.shuffle = bool(shuffle)
        self.pad_id = int(pad_id)
        self.ignore_index = int(ignore_index)
        self.n_epochs = int(epochs)
        self.total_steps, self.steps_per_epoch = plan_steps(
            split.lengths, T, B, accum, self.n_epochs, dense, seed,
            shuffle=shuffle)
        self.epoch = 0          # 0-based, currently filling
        self._rows = []
        self._cursor = 0
        self._slots_left = 0    # remaining micro-slots this epoch (incl. pad)
        self._begin_epoch(0)

    def _order(self, epoch):
        n = self.split.n
        if self.shuffle:
            return np.random.RandomState(self.seed + epoch).permutation(n)
        return np.arange(n)

    def _begin_epoch(self, epoch):
        self.epoch = epoch
        self._rows = pack_rows(self.split.lengths, self._order(epoch),
                               self.T, self.dense)
        self._cursor = 0
        spe = self.steps_per_epoch[epoch]
        self._slots_left = spe * self.B * self.accum

    def _empty_row(self):
        x = np.full(self.T, self.pad_id, dtype=np.int64)
        y = np.full(self.T, self.ignore_index, dtype=np.int64)
        return x, y

    def _one_row(self, ex_ix):
        ids = np.concatenate([self.split.example(i) for i in ex_ix])
        return pack_xy(ids, self.T, self.assistant_id, self.bos_id,
                       pad_id=self.pad_id, ignore_index=self.ignore_index,
                       usr_id=self.usr_id, end_id=self.end_id,
                       lowdiff_masking=self.lowdiff_masking,
                       lowdiff_threshold=self.lowdiff_threshold)

    def next_micro(self):
        """One micro-batch. Pads with ignore-rows after the epoch's examples."""
        xs, ys = [], []
        for _ in range(self.B):
            if self._cursor < len(self._rows):
                x, y = self._one_row(self._rows[self._cursor])
                self._cursor += 1
            else:
                x, y = self._empty_row()
            xs.append(x)
            ys.append(y)
            self._slots_left -= 1
        finished_epoch = None
        if self._slots_left <= 0:
            finished_epoch = self.epoch + 1          # 1-based completed
            nxt = self.epoch + 1
            if nxt < self.n_epochs:
                self._begin_epoch(nxt)
        return np.stack(xs), np.stack(ys), finished_epoch

    def state_dict(self):
        return {"epoch": self.epoch, "cursor": self._cursor,
                "slots_left": self._slots_left}

    def load_state_dict(self, sd):
        self._begin_epoch(int(sd["epoch"]))
        self._cursor = int(sd["cursor"])
        self._slots_left = int(sd["slots_left"])
