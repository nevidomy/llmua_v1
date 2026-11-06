"""JSONL (already cleaned) -> train.bin / val.bin -> random batches.

The tokenizer is used ONCE, offline. Training reads a flat uint16 array of
token ids via memmap and samples random windows -- that random offset IS the
shuffling. Documents are concatenated end to end, separated by the specials
the template inserts.

Differences from the earlier version, and why:

  - NO cleaning. corpus.jsonl is already cleaned; re-running clean_ua_str is
    a no-op (it is idempotent) that costs a full pass over 9B chars.
  - No Pool, so TOKENIZERS_PARALLELISM can be "true" and encode_batch uses
    all cores. That was the actual bottleneck.
  - Buffered writes: one array per batch per split, not one tofile() per
    document. At 2.9M docs that is ~2,900 syscalls instead of 2.9M.
  - meta.json records chars_per_token per split, computed from counts the
    pass already has -- exact, and no second decode pass over the bins.
  - meta.json records which tokenizer produced the bins, and tk_verify
    refuses a mismatch.
"""

import hashlib
import json
import os
import random
import sys

import numpy as np

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def tk_convert(doc_iter, out_dir, tok, tok_path=None, val_frac=0.005,
               test_frac=0.0, batch=2000, seed=0, val_hashes=None,
               val_only=False):
    """doc_iter yields ALREADY CLEANED strings, e.g. read_jsonl(corpus.jsonl).

    val is split by DOCUMENT, not by position: a positional split would make
    val pure-wikipedia if wiki happened to be written last.

    val_hashes: ordered list of hex digests (data.shard_util.doc_hash_hex).
    When given, the RNG split is not used: a document goes to val iff its
    hash is still unclaimed in the list, everything else goes to train. val
    documents are written at the end in LIST order, and every listed hash
    must be found. This reproduces a val set by content, independent of the
    line order of the corpus file. val_only=True skips train (no train.bin).
    """
    if val_hashes is not None:
        from collections import Counter
        from data.shard_util import doc_hash_hex
        assert not test_frac, "test_frac is not supported with val_hashes"
        want = Counter(val_hashes)
        held = {h: [] for h in want}
    else:
        assert not val_only, "val_only needs val_hashes"
    from tqdm.auto import tqdm
    V = tok.get_vocab_size()
    assert V < 2 ** 16, f"vocab {V} does not fit uint16"

    rng = random.Random(seed)
    os.makedirs(out_dir, exist_ok=True)
    fracs = {"val": val_frac}
    if test_frac:
        fracs["test"] = test_frac
    names = ["train", "val"] + (["test"] if test_frac else [])
    if val_only:
        names = ["val"]

    files = {s: open(os.path.join(out_dir, f"{s}.bin"), "wb") for s in names}
    ntok = {s: 0 for s in names}
    nchar = {s: 0 for s in names}
    ndoc = {s: 0 for s in names}

    def pick():
        r, acc = rng.random(), 0.0
        for name, f in fracs.items():
            acc += f
            if r < acc:
                return name
        return "train"

    def flush_hashed(docs):
        sel, is_val = [], []
        for d in docs:
            h = doc_hash_hex(d)
            v = want.get(h, 0) > 0
            if v:
                want[h] -= 1
            elif val_only:
                continue
            sel.append((h, d))
            is_val.append(v)
        encs = tok.encode_batch([d for _, d in sel])
        train = []
        for (h, d), v, enc in zip(sel, is_val, encs):
            if v:
                held[h].append(enc.ids)
                nchar["val"] += len(d)
                ndoc["val"] += 1
            else:
                train.append(enc.ids)
                nchar["train"] += len(d)
                ndoc["train"] += 1
        if train:
            arr = np.fromiter((i for c in train for i in c), dtype=np.uint16,
                              count=sum(len(c) for c in train))
            arr.tofile(files["train"])
            ntok["train"] += arr.size

    def flush(docs):
        if not docs:
            return
        if val_hashes is not None:
            return flush_hashed(docs)
        parts = {s: [] for s in names}
        for enc, d in zip(tok.encode_batch(docs), docs):  # specials via template
            s = pick()
            parts[s].append(enc.ids)
            nchar[s] += len(d)
            ndoc[s] += 1
        for s, chunks in parts.items():
            if not chunks:
                continue
            arr = np.fromiter((i for c in chunks for i in c),
                              dtype=np.uint16,
                              count=sum(len(c) for c in chunks))
            arr.tofile(files[s])
            ntok[s] += arr.size

    buf = []
    try:
        for doc in tqdm(doc_iter, unit="doc", file=sys.stdout):
            if not doc:
                continue
            buf.append(doc)
            if len(buf) >= batch:
                flush(buf)
                buf = []
        flush(buf)
        if val_hashes is not None:
            missing = sum(want.values())
            assert missing == 0, f"{missing} val_hashes not found in corpus"
            for h in val_hashes:
                ids = held[h].pop(0)
                np.asarray(ids, dtype=np.uint16).tofile(files["val"])
                ntok["val"] += len(ids)
    finally:
        for f in files.values():
            f.close()

    meta = {
        "vocab_size": V,
        "dtype": "uint16",
        "bos_id": tok.token_to_id("<|bos|>"),
        "eos_id": tok.token_to_id("<|eos|>"),
        "tokenizer": os.path.basename(tok_path) if tok_path else None,
        "tokenizer_sha": (hashlib.blake2b(open(tok_path, "rb").read(),
                                          digest_size=8).hexdigest()
                          if tok_path and os.path.exists(tok_path) else None),
    }
    for s in names:
        meta[f"n_docs_{s}"] = ndoc[s]
        meta[f"n_tokens_{s}"] = ntok[s]
        meta[f"n_chars_{s}"] = nchar[s]
        meta[f"chars_per_token_{s}"] = (nchar[s] / ntok[s]) if ntok[s] else None
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{sum(ndoc.values()):,} docs -> {sum(ntok.values())/1e6:.1f}M tokens")
    for s in names:
        print(f"  {s:<6} {ndoc[s]:>10,} docs  {ntok[s]/1e6:>9.2f}M tok  "
              f"cpt {meta[f'chars_per_token_{s}']:.4f}")
    return meta


def tk_verify(out_dir, tok, split="train"):
    """Catch doubled specials, uint16 wraparound, wrong tokenizer.

    Works for a single `{split}.bin` and for sharded train_XXXX.bin
    (index.json). Counts BOS/EOS in chunks so a 20 GB file is not pulled
    into RAM.
    """
    from tokens.sharded_bins import open_split

    meta = json.load(open(os.path.join(out_dir, "meta.json")))
    assert meta["vocab_size"] == tok.get_vocab_size(), (
        f"meta says vocab {meta['vocab_size']}, tokenizer has "
        f"{tok.get_vocab_size()} -- WRONG TOKENIZER FOR THESE BINS")

    bins = open_split(out_dir, split)
    bos, eos, V = meta["bos_id"], meta["eos_id"], meta["vocab_size"]
    nb = ne = mx = 0
    first_eos = None
    chunk = 1 << 22
    pos = 0
    for shard in bins.shards:
        off = 0
        n = len(shard)
        while off < n:
            block = np.asarray(shard[off:off + chunk])
            if block.size:
                nb += int((block == bos).sum())
                ne += int((block == eos).sum())
                mx = max(mx, int(block.max()))
                if first_eos is None:
                    z = np.flatnonzero(block == eos)
                    if z.size:
                        first_eos = pos + off + int(z[0])
            off += chunk
        pos += n

    nd = meta[f"n_docs_{split}"]
    print(f"{bins.n_tokens:,} tokens | BOS {nb:,} | EOS {ne:,} | docs {nd:,}")
    assert first_eos is not None, "no EOS in corpus"
    assert nb == ne, "BOS/EOS mismatch -- doubled or missing specials"
    assert nb == nd, f"{nb} BOS for {nd} docs -- specials doubled or dropped"
    assert mx < V, "id >= vocab_size: uint16 wrapped"

    prefix = bins.gather(0, first_eos + 1)
    txt = tok.decode(prefix.tolist(), skip_special_tokens=True)
    print(f"first doc, {first_eos + 1} tokens:\n  {txt[:200]!r}")
    return bins


def tk_get_batch(out_dir, split, B, T, device="cpu"):
    """Random windows over a single file or sharded train bins."""
    import torch
    from tokens.sharded_bins import open_split
    bins = open_split(out_dir, split)
    ix = np.random.randint(0, bins.n_tokens - T - 1, size=B)
    x, y = bins.windows(ix, T)
    x, y = torch.from_numpy(x), torch.from_numpy(y)
    if device != "cpu":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    return x, y


def bpc(loss_nats, out_dir, split="val"):
    """Loss -> bits/char using the cpt recorded for that split. Cross-entropy
    is NOT comparable across tokenizers without this."""
    meta = json.load(open(os.path.join(out_dir, "meta.json")))
    npc = loss_nats / meta[f"chars_per_token_{split}"]
    return npc, npc / np.log(2)

def num_chars_per_token(out_dir, split="val"):
    meta = json.load(open(os.path.join(out_dir, "meta.json")))
    return meta[f"chars_per_token_{split}"]