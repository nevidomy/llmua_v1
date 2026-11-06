"""Sharded uint16 token bins: one logical stream, many files.

This module is the read side:

  - ShardedBins maps a global token index to (shard, local offset)
  - EpochSampler walks every non-overlapping window of length T exactly
    once per epoch, then reshuffles and re-draws the tiling offset so
    document-split points move

A window that straddles a shard boundary is copied; at 500M tokens per
shard and T ~ 2k that is about one window in 250k.
"""

import json
import os

import numpy as np


def _load_index(bins_dir, split):
    """Return (files, lengths, meta, sharded).

    Train prefers index.json (train_0000.bin, ...). Anything else is a
    single `{split}.bin`, which is also how val stays.
    """
    meta_path = os.path.join(bins_dir, "meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    dtype = np.dtype(meta.get("dtype", "uint16"))

    index_path = os.path.join(bins_dir, "index.json")
    if split == "train" and os.path.exists(index_path):
        idx = json.load(open(index_path))
        shards = idx["shards"]
        files = [os.path.join(bins_dir, s["file"]) for s in shards]
        lengths = [int(s["n_tokens"]) for s in shards]
        return files, lengths, meta, True

    path = os.path.join(bins_dir, f"{split}.bin")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    n = os.path.getsize(path) // dtype.itemsize
    key = f"n_tokens_{split}"
    if key in meta and int(meta[key]) != n:
        # trust the file; meta can lag a recopy
        pass
    return [path], [n], meta, False


class ShardedBins:
    """np.memmap each shard once. `gather(start, n)` is a global slice."""

    def __init__(self, files, lengths, meta=None, sharded=False, dtype=np.uint16):
        if not files:
            raise ValueError("no shard files")
        if len(files) != len(lengths):
            raise ValueError("files/lengths mismatch")
        self.files = list(files)
        self.lengths = np.asarray(lengths, dtype=np.int64)
        self.offsets = np.empty(len(files) + 1, dtype=np.int64)
        self.offsets[0] = 0
        np.cumsum(self.lengths, out=self.offsets[1:])
        self.n_tokens = int(self.offsets[-1])
        self.meta = meta or {}
        self.sharded = sharded
        self.dtype = np.dtype(dtype)
        self.shards = [np.memmap(p, dtype=self.dtype, mode="r") for p in files]
        for mm, n, p in zip(self.shards, self.lengths, files):
            if len(mm) != n:
                raise ValueError(f"{p}: index says {n:,} tokens, file has {len(mm):,}")

    @classmethod
    def open(cls, bins_dir, split="train"):
        files, lengths, meta, sharded = _load_index(bins_dir, split)
        dtype = np.dtype(meta.get("dtype", "uint16"))
        return cls(files, lengths, meta=meta, sharded=sharded, dtype=dtype)

    def as_memmap(self):
        if len(self.shards) != 1:
            raise TypeError("as_memmap() needs a single-file split "
                            f"(got {len(self.shards)} shards)")
        return self.shards[0]

    def _shard_at(self, start):
        i = int(np.searchsorted(self.offsets, start, side="right") - 1)
        if i < 0 or i >= len(self.shards) or start >= self.n_tokens:
            raise IndexError(f"start {start} outside [0, {self.n_tokens})")
        return i, int(start - self.offsets[i])

    def gather(self, start, length):
        """`length` tokens beginning at global `start`. Copies if needed."""
        start = int(start)
        length = int(length)
        if length < 0 or start < 0 or start + length > self.n_tokens:
            raise IndexError(
                f"slice [{start}:{start + length}] outside [0, {self.n_tokens})")
        i, loc = self._shard_at(start)
        if loc + length <= self.lengths[i]:
            return np.asarray(self.shards[i][loc:loc + length])
        out = np.empty(length, dtype=self.dtype)
        o = 0
        while o < length:
            n = min(length - o, int(self.lengths[i] - loc))
            out[o:o + n] = self.shards[i][loc:loc + n]
            o += n
            i += 1
            loc = 0
        return out

    def windows(self, starts, T):
        """x, y int64 arrays of shape (B, T) from global window starts."""
        B = len(starts)
        xs = np.empty((B, T), dtype=np.int64)
        ys = np.empty((B, T), dtype=np.int64)
        for i, s in enumerate(starts):
            buf = self.gather(int(s), T + 1)
            xs[i] = buf[:T]
            ys[i] = buf[1:]
        return xs, ys


def open_split(bins_dir, split="train"):
    return ShardedBins.open(bins_dir, split)


class EpochSampler:
    """Exact coverage, no repeats within an epoch, new tiling each epoch.

        offset = rng.integers(0, T)
        window k covers [offset + k*T, offset + (k+1)*T)

    n_win = (n_tokens - 1 - offset) // T  so y[T-1] is always in range.
    Checkpoint is (epoch, cursor); the permutation is regenerated.
    """

    def __init__(self, n_tokens, T, seed=0):
        if T < 1:
            raise ValueError(f"T must be >= 1, got {T}")
        if n_tokens < T + 1:
            raise ValueError(f"need at least T+1={T + 1} tokens, got {n_tokens}")
        self.n_tokens = int(n_tokens)
        self.T = int(T)
        self.seed = int(seed)
        self.epoch = 0
        self.cursor = 0
        self.offset = 0
        self.n_win = 0
        self.perm = None
        self._build_epoch()

    def _epoch_rng(self):
        return np.random.default_rng([self.seed, self.epoch])

    def _build_epoch(self):
        rng = self._epoch_rng()
        self.offset = int(rng.integers(0, self.T))
        self.n_win = (self.n_tokens - 1 - self.offset) // self.T
        if self.n_win < 1:
            self.offset = 0
            self.n_win = (self.n_tokens - 1) // self.T
        self.perm = rng.permutation(self.n_win)
        self.cursor = 0

    def next_starts(self, B):
        out = np.empty(B, dtype=np.int64)
        filled = 0
        while filled < B:
            if self.cursor >= self.n_win:
                self.epoch += 1
                self._build_epoch()
            take = min(B - filled, self.n_win - self.cursor)
            ks = self.perm[self.cursor:self.cursor + take]
            out[filled:filled + take] = (
                self.offset + ks.astype(np.int64) * self.T)
            self.cursor += take
            filled += take
        return out

    def state_dict(self):
        return {
            "n_tokens": self.n_tokens, "T": self.T, "seed": self.seed,
            "epoch": self.epoch, "cursor": self.cursor,
        }

    def load_state_dict(self, d):
        if d["n_tokens"] != self.n_tokens or d["T"] != self.T:
            raise ValueError(
                f"sampler mismatch: ckpt n_tokens={d['n_tokens']} T={d['T']} "
                f"vs bins n_tokens={self.n_tokens} T={self.T}")
        if d.get("seed", self.seed) != self.seed:
            # allow, but replay from the checkpoint's seed
            self.seed = int(d["seed"])
        self.epoch = int(d["epoch"])
        self._build_epoch()
        cur = int(d["cursor"])
        if not 0 <= cur <= self.n_win:
            raise ValueError(f"cursor {cur} outside [0, {self.n_win}]")
        self.cursor = cur
        return self


def _self_test():
    import tempfile

    # 3 shards, 10 tokens each. Window starting at 8 needs 5 tokens -> spans.
    parts = [
        np.arange(10, dtype=np.uint16),
        np.arange(10, 20, dtype=np.uint16),
        np.arange(20, 30, dtype=np.uint16),
    ]
    with tempfile.TemporaryDirectory() as td:
        files, lengths = [], []
        for i, a in enumerate(parts):
            p = os.path.join(td, f"train_{i:04d}.bin")
            a.tofile(p)
            files.append(p)
            lengths.append(a.size)
        idx = {"dtype": "uint16", "n_tokens": 30,
               "shards": [{"file": os.path.basename(f), "n_tokens": n}
                          for f, n in zip(files, lengths)]}
        json.dump(idx, open(os.path.join(td, "index.json"), "w"))
        json.dump({"dtype": "uint16", "vocab_size": 32,
                   "n_tokens_train": 30},
                  open(os.path.join(td, "meta.json"), "w"))
        bins = ShardedBins.open(td, "train")
        assert bins.sharded and bins.n_tokens == 30
        np.testing.assert_array_equal(bins.gather(0, 10), parts[0])
        np.testing.assert_array_equal(bins.gather(8, 5),
                                      np.array([8, 9, 10, 11, 12], np.uint16))
        xs, ys = bins.windows([8], T=4)
        np.testing.assert_array_equal(xs[0], [8, 9, 10, 11])
        np.testing.assert_array_equal(ys[0], [9, 10, 11, 12])

        # round-trip via open_split on this dir
        s = EpochSampler(n_tokens=30, T=4, seed=0)
        starts = s.next_starts(s.n_win)
        assert len(set(starts.tolist())) == s.n_win
        assert starts.min() >= s.offset
        assert (starts + 4).max() <= 29   # y[T-1] in range
        # resume
        d = s.state_dict()
        s2 = EpochSampler(n_tokens=30, T=4, seed=0)
        s2.load_state_dict(d)
        np.testing.assert_array_equal(s.next_starts(3), s2.next_starts(3))

        # within one epoch, every k appears once
        s3 = EpochSampler(n_tokens=30, T=4, seed=1)
        n = s3.n_win
        got = s3.next_starts(n)
        ks = (got - s3.offset) // s3.T
        assert sorted(ks.tolist()) == list(range(n))

    print("sharded_bins self-test ok")


if __name__ == "__main__":
    _self_test()
