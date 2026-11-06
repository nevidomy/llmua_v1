"""Tokenize {question, answer} JSON files into packed uint16 SFT bins.

Each kept example is:

    <BOS> <USR> {question tokens} <ASSISTANT> {answer tokens} <END>

Special token *ids* are parameters -- the strings are not in the pretrain
tokenizer yet. Examples longer than --context-width are dropped, not written.
Each example starts with BOS; offsets.npy / lengths.npy locate pairs exactly.

`-p path[,N]` (default) shuffles that file and keeps N examples that fit
(omit N for all). Then --val-frac of each file's sample goes to val.bin.

`--val-nums` changes `-p` to `path,val_n[,train_n]`: after shuffle and
dropping over-length, the first val_n go to val.bin; the next train_n
(or the rest, if train_n is omitted) go to train.bin. --seed sets the
shuffle RNG.

    python prepare_sft.py tokenize \\
      --val-nums --seed 0 \\
      -p ~/data/llmua/sft_data/text/fidoai.json,100,500 \\
      -p ~/data/llmua/sft_data/text/fidonews.json,200 \\
      --out ~/data/llmua/sft_data/bins/32768 \\
      --bos-id 0 --usr-id 201 --assistant-id 202 --end-id 203 \\
      --context-width 2048 --dry-run
"""
import argparse
import glob
import os
import random

import numpy as np

from data.index import ROOT
from data.shard_util import confirm, utcnow, write_json
from sft.pairs import load_pairs


def encode_pair(tok, question, answer, bos_id, usr_id, assistant_id, end_id):
    """Packed ids: BOS USR {q} ASSISTANT {a} END.

    Returns (ids, n_answer) where n_answer is len(a)+1 (answer tokens and END).
    """
    q = tok.encode(question, add_special_tokens=False).ids
    a = tok.encode(answer, add_special_tokens=False).ids
    ids = [bos_id, usr_id] + list(q) + [assistant_id] + list(a) + [end_id]
    return ids, len(a) + 1


N_SPECIAL = 4  # BOS, USR, ASSISTANT, END


def peel_trailing_ints(spec):
    """Split 'path,1,2' -> ('path', [1, 2]). Stops at the first non-int field."""
    nums = []
    rest = spec.strip()
    while "," in rest:
        left, right = rest.rsplit(",", 1)
        if not right.strip().isdigit():
            break
        nums.append(int(right.strip()))
        rest = left
    nums.reverse()
    return rest.strip(), nums


def parse_source_spec(spec, val_nums=False):
    """Parse one -p argument.

    Default: path or path,N (N fitting examples, then --val-frac).
    --val-nums: path,val_n[,train_n] (prefix of shuffled fit).
    """
    path, nums = peel_trailing_ints(spec)
    if not path:
        raise SystemExit(f"empty path in {spec!r}")
    if val_nums:
        if len(nums) > 2:
            raise SystemExit(
                f"--val-nums expects path,val_n[,train_n], got {spec!r}")
        n_val = nums[0] if nums else 0
        n_train = nums[1] if len(nums) == 2 else None
        if n_val < 0 or (n_train is not None and n_train < 0):
            raise SystemExit(f"counts must be >= 0 in {spec!r}")
        if n_train == 0:
            raise SystemExit(f"train_n must be > 0 in {spec!r}")
        return path, {"n_val": n_val, "n_train": n_train, "n_take": None}
    if len(nums) > 1:
        raise SystemExit(
            f"{spec!r}: path,val_n[,train_n] needs --val-nums "
            f"(without it use path[,N] + --val-frac)")
    n_take = nums[0] if nums else None
    if n_take is not None and n_take <= 0:
        raise SystemExit(f"count must be > 0 in {spec!r}")
    return path, {"n_take": n_take, "n_val": None, "n_train": None}


def expand_sources(specs, val_nums=False):
    """List of (abspath, counts) in -p order. Counts apply per file."""
    out = []
    for spec in specs:
        pat, counts = parse_source_spec(spec, val_nums=val_nums)
        pat = os.path.expanduser(pat)
        files = [os.path.abspath(p) for p in sorted(glob.glob(pat))]
        if not files and os.path.isfile(pat):
            files = [os.path.abspath(pat)]
        if not files:
            raise FileNotFoundError(f"no files match {pat!r}")
        for path in files:
            out.append((path, counts))
    return out


def print_source_inventory(sources, val_nums=False):
    print("pair files")
    total = 0
    for path, counts in sources:
        sz = os.path.getsize(path)
        total += sz
        if val_nums:
            n_val, n_train = counts["n_val"], counts["n_train"]
            rest = "rest train" if n_train is None else f"train {n_train:,}"
            take = f"val {n_val:,}, {rest}"
        else:
            n = counts["n_take"]
            take = f"take {n:,}" if n is not None else "take all"
        print(f"  {sz/1e9:7.2f} GB  {path}  ({take})")
    print(f"  {len(sources)} files, {total/1e9:.2f} GB total")
    return len(sources), total


def _bucket_edges(context_width):
    edges = [0]
    b = 32
    while b < context_width:
        edges.append(b)
        b *= 2
    if edges[-1] != context_width:
        edges.append(context_width)
    edges.append(None)  # +inf
    return edges


def print_length_stats(all_lens, context_width, n_empty, n_tokens_kept):
    all_lens = np.asarray(all_lens, dtype=np.int64)
    n = int(all_lens.size)
    kept = all_lens[all_lens <= context_width]
    dropped = all_lens[all_lens > context_width]
    print(f"\n{n:,} encoded  {len(kept):,} fit  {len(dropped):,} over "
          f"context_width={context_width}  empty skipped {n_empty:,}")
    print(f"  kept tokens {n_tokens_kept:,}")
    if kept.size:
        print(f"  kept length  mean {kept.mean():.0f}  median "
              f"{np.median(kept):.0f}  max {int(kept.max())}")
    print(f"\n{'full seq':>16}{'docs':>12}{'% docs':>9}  {'kept?'}")
    _print_buckets(all_lens, context_width, drop_over=True)


def print_answer_stats(ans_lens, n_tokens_kept, context_width):
    """Answer-side tokens on kept examples: {answer} + END."""
    ans_lens = np.asarray(ans_lens, dtype=np.int64)
    n = int(ans_lens.size)
    if n == 0:
        print("\n  answer (+END)  (no kept examples)")
        return 0
    total = int(ans_lens.sum())
    pct = 100.0 * total / n_tokens_kept if n_tokens_kept else 0.0
    print(f"\n  answer (+END)  {total:,} tokens  ({pct:.1f}% of kept)")
    print(f"  answer length  mean {ans_lens.mean():.0f}  median "
          f"{np.median(ans_lens):.0f}  min {int(ans_lens.min())}  "
          f"max {int(ans_lens.max())}")
    print(f"\n{'ans+END':>16}{'docs':>12}{'% docs':>9}")
    _print_buckets(ans_lens, context_width, drop_over=False)
    return total


def _print_buckets(lens, context_width, drop_over):
    n = int(lens.size)
    edges = _bucket_edges(context_width)
    for lo, hi in zip(edges, edges[1:]):
        if hi is None:
            m = lens > lo
            label = f"({lo}, inf]"
            flag = "dropped" if drop_over and lo >= context_width else ""
        else:
            m = (lens > lo) & (lens <= hi)
            label = f"({lo}, {hi}]"
            flag = "dropped" if drop_over and hi > context_width else (
                "kept" if drop_over else "")
        extra = f"  {flag}" if drop_over else ""
        print(f"{label:>16}{int(m.sum()):>12,}{m.mean()*100 if n else 0:>8.1f}%{extra}")


def answer_cpt(examples):
    """Source chars in {answer} / loss tokens ({answer}+END)."""
    n_tok = sum(x[1] for x in examples)
    if not n_tok or not examples or len(examples[0]) < 3:
        return None
    n_ch = sum(x[2] for x in examples)
    return n_ch / n_tok


def split_kept(kept, val_frac, seed):
    """Shuffle then take round(n * val_frac) for val; the rest is train."""
    if not (0 <= val_frac < 1):
        raise SystemExit("--val-frac must be in [0, 1)")
    n = len(kept)
    n_val = int(round(n * val_frac))
    if n_val == 0:
        return list(kept), []
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    val_ix = set(idx[:n_val])
    train, val = [], []
    for i, ex in enumerate(kept):
        (val if i in val_ix else train).append(ex)
    return train, val


def write_split(out_dir, name, examples, bos_id):
    n_tokens = sum(len(x) for x in examples)
    bin_path = os.path.join(out_dir, f"{name}.bin")
    offsets = np.empty(len(examples), dtype=np.int64)
    lengths = np.empty(len(examples), dtype=np.int32)
    buf = np.empty(n_tokens, dtype=np.uint16)
    pos = 0
    for i, ids_ in enumerate(examples):
        n = len(ids_)
        offsets[i] = pos
        lengths[i] = n
        buf[pos:pos + n] = np.asarray(ids_, dtype=np.uint16)
        pos += n
        assert ids_[0] == bos_id
    tmp = bin_path + ".tmp"
    buf.tofile(tmp)
    os.replace(tmp, bin_path)
    np.save(os.path.join(out_dir, f"{name}.offsets.npy"), offsets)
    np.save(os.path.join(out_dir, f"{name}.lengths.npy"), lengths)
    return bin_path, n_tokens


def run(args):
    val_nums = bool(getattr(args, "val_nums", False))
    sources = expand_sources(args.patterns, val_nums=val_nums)
    print_source_inventory(sources, val_nums=val_nums)
    tok_path = os.path.abspath(os.path.expanduser(args.tokenizer))
    out_dir = os.path.abspath(os.path.expanduser(args.out))
    val_frac = 0.0 if val_nums else args.val_frac
    print(f"\n  tokenizer      {tok_path}")
    print(f"  out            {out_dir}")
    print(f"  context_width  {args.context_width}")
    if val_nums:
        print(f"  split          val-nums  (seed {args.seed})")
    else:
        print(f"  val_frac       {val_frac:g}  (seed {args.seed})")
    print(f"  specials       BOS={args.bos_id} USR={args.usr_id} "
          f"ASSISTANT={args.assistant_id} END={args.end_id}")

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tok_path)
    V = tok.get_vocab_size()
    assert V < 2 ** 16, f"vocab {V} does not fit uint16"
    spec_ids = (args.bos_id, args.usr_id, args.assistant_id, args.end_id)
    if any(i < 0 for i in spec_ids):
        raise SystemExit("special token ids must be >= 0")

    rng = random.Random(args.seed)
    encoded, skipped_empty = [], 0
    file_trains, file_vals, source_recs = [], [], []
    T = args.context_width
    print("\nper file  (shuffle, drop over-length, then take)")
    for path, counts in sources:
        rows = list(load_pairs(path))
        rng.shuffle(rows)
        file_rows, file_empty = [], 0
        for row in rows:
            q = (row.get("question") or "").strip()
            a = (row.get("answer") or "").strip()
            if not q or not a:
                file_empty += 1
                continue
            ids_, n_ans = encode_pair(
                tok, q, a, args.bos_id, args.usr_id,
                args.assistant_id, args.end_id)
            file_rows.append((ids_, n_ans, len(a)))
        skipped_empty += file_empty
        encoded.extend(r[0] for r in file_rows)
        fit = [r for r in file_rows if len(r[0]) <= T]
        n_over = len(file_rows) - len(fit)
        rec = {"src": path, "n_in": len(rows), "n_empty": file_empty,
               "n_encoded": len(file_rows), "n_fit": len(fit),
               "n_over": n_over, **counts}

        if val_nums:
            n_val = counts["n_val"]
            n_train = counts["n_train"]
            if n_val > len(fit):
                print(f"  warn: {os.path.basename(path)}  {len(fit):,} fit "
                      f"< val {n_val:,}")
            va = fit[:n_val]
            rest = fit[n_val:]
            if n_train is None:
                tr = rest
            else:
                if n_train > len(rest):
                    print(f"  warn: {os.path.basename(path)}  {len(rest):,} "
                          f"left after val {len(va):,} < train {n_train:,}")
                tr = rest[:n_train]
            rec["n_sampled"] = len(tr) + len(va)
            rec["n_train"] = len(tr)
            rec["n_val"] = len(va)
            want = (f"val {n_val:,}, "
                    + ("rest train" if n_train is None else f"train {n_train:,}"))
        else:
            n_take = counts["n_take"]
            if n_take is None:
                take = fit
            else:
                take = fit[:n_take]
                if len(fit) < n_take:
                    print(f"  warn: {os.path.basename(path)}  {len(fit):,} fit "
                          f"< requested {n_take:,}")
            tr, va = split_kept(take, val_frac, args.seed + len(source_recs))
            rec["n_sampled"] = len(take)
            rec["n_train"] = len(tr)
            rec["n_val"] = len(va)
            want = "all" if n_take is None else f"{n_take:,}"

        file_trains.append(tr)
        file_vals.append(va)
        source_recs.append(rec)
        print(f"  {os.path.basename(path):<24}  {len(rows):,} in  "
              f"{file_empty:,} empty  {len(fit):,} fit  {n_over:,} over  "
              f"-> train {len(tr):,}  val {len(va):,}  (want {want})")

    train_ex = [x for chunk in file_trains for x in chunk]
    val_ex = [x for chunk in file_vals for x in chunk]
    sampled = train_ex + val_ex
    kept = [x[0] for x in sampled]
    kept_ans = [x[1] for x in sampled]
    n_tokens = sum(len(x) for x in kept)
    n_fit_tok = sum(len(x) for x in encoded if len(x) <= T)
    print_length_stats([len(x) for x in encoded], T, skipped_empty, n_fit_tok)
    print(f"  sampled {len(kept):,} examples  {n_tokens:,} tokens")
    n_answer = print_answer_stats(kept_ans, n_tokens, T)

    train_ids = [x[0] for x in train_ex]
    val_ids = [x[0] for x in val_ex]
    cpt_train = answer_cpt(train_ex)
    cpt_val = answer_cpt(val_ex)
    print(f"\n  train  {len(train_ids):,} examples  "
          f"{sum(len(x) for x in train_ids):,} tokens  "
          f"(answer+END {sum(x[1] for x in train_ex):,})"
          f"{f'  cpt {cpt_train:.4f}' if cpt_train else ''}")
    print(f"  val    {len(val_ids):,} examples  "
          f"{sum(len(x) for x in val_ids):,} tokens  "
          f"(answer+END {sum(x[1] for x in val_ex):,})"
          f"{f'  cpt {cpt_val:.4f}' if cpt_val else ''}"
          f"  ({100 * len(val_ids) / max(len(kept), 1):.1f}% of sampled)")

    summary = {
        "n_examples": len(kept), "n_tokens": n_tokens,
        "n_answer_tokens": n_answer,
        "n_encoded": len(encoded),
        "n_dropped_length": sum(r["n_over"] for r in source_recs),
        "n_skipped_empty": skipped_empty,
        "n_sampled": len(kept),
        "n_train": len(train_ids), "n_val": len(val_ids),
        "chars_per_token_train": cpt_train,
        "chars_per_token_val": cpt_val,
        "val_frac": None if val_nums else val_frac,
        "val_nums": val_nums,
        "seed": args.seed,
        "sources": source_recs,
    }
    if args.dry_run:
        print("\ndry-run: not writing")
        return summary

    confirm(args.yes, prompt="Tokenize these SFT pairs? [y/N] ")
    os.makedirs(out_dir, exist_ok=True)

    splits = [("train", train_ids)]
    if val_ids:
        splits.append(("val", val_ids))
    elif not val_nums and val_frac == 0 and args.name != "train":
        splits = [(args.name, train_ids)]

    created = utcnow()
    meta = {
        "step": "sft_tokenize",
        "created": created,
        "dtype": "uint16",
        "vocab_size": V,
        "tokenizer": os.path.basename(tok_path),
        "n_encoded": len(encoded),
        "n_dropped_length": summary["n_dropped_length"],
        "n_skipped_empty": skipped_empty,
        "n_sampled": len(kept),
        "n_answer_tokens": n_answer,
        "chars_per_token_train": cpt_train,
        "chars_per_token_val": cpt_val,
        "context_width": args.context_width,
        "val_frac": None if val_nums else val_frac,
        "val_nums": val_nums,
        "seed": args.seed,
        "bos_id": args.bos_id,
        "usr_id": args.usr_id,
        "assistant_id": args.assistant_id,
        "end_id": args.end_id,
        "sources": source_recs,
        "template": "<BOS><USR>{question}<ASSISTANT>{answer}<END>",
        "sep": "each example starts with bos_id; use offsets/lengths to slice",
    }
    split_ans = {"train": sum(x[1] for x in train_ex)}
    if val_ids:
        split_ans["val"] = sum(x[1] for x in val_ex)
    elif not val_nums and val_frac == 0 and args.name != "train":
        split_ans = {args.name: sum(x[1] for x in train_ex)}
    for name, examples in splits:
        path, ntok = write_split(out_dir, name, examples, args.bos_id)
        n_ans_split = split_ans.get(name, 0)
        meta[f"n_examples_{name}"] = len(examples)
        meta[f"n_tokens_{name}"] = ntok
        meta[f"n_answer_tokens_{name}"] = n_ans_split
        meta[f"file_{name}"] = os.path.basename(path)
        write_json(os.path.join(out_dir, f"{name}.meta.json"),
                   {**meta, "name": name, "file": os.path.basename(path),
                    "n_examples": len(examples), "n_tokens": ntok,
                    "n_answer_tokens": n_ans_split})
        print(f"\n{len(examples):,} {name}  {ntok:,} tokens "
              f"(answer+END {n_ans_split:,}) -> {path}")
        print(f"  offsets {name}.offsets.npy  lengths {name}.lengths.npy")
    write_json(os.path.join(out_dir, "meta.json"), meta)
    summary.update(meta)
    return summary


def parse_args(argv=None, prog=None):
    p = argparse.ArgumentParser(
        prog=prog, description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-p", "--pattern", dest="patterns", action="append",
                   required=True,
                   help="pair JSON: path[,N] with --val-frac, or "
                        "path,val_n[,train_n] with --val-nums")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--name", default="train",
                   help="file stem when there is no val split (default train)")
    p.add_argument("--tokenizer",
                   default=os.path.join(ROOT, "tokenizers", "ua_bpe_32768.json"))
    p.add_argument("--context-width", type=int, default=2048)
    p.add_argument("--val-frac", type=float, default=0.05,
                   help="fraction of each file's sample randomly assigned to "
                        "val (ignored with --val-nums); 0 writes train only")
    p.add_argument("--val-nums", action="store_true",
                   help="parse -p as path,val_n[,train_n]: first val_n of "
                        "shuffled fit -> val, next train_n or the rest -> train")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed for per-file shuffle (and val-frac split)")
    p.add_argument("--bos-id", type=int, default=0)
    p.add_argument("--usr-id", type=int, default=201)
    p.add_argument("--assistant-id", type=int, default=202)
    p.add_argument("--end-id", type=int, default=203)
    p.add_argument("--yes", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def _self_test():
    import tempfile
    from tokens.tokenizer_definition import tk_train
    from sft.pairs import write_pairs

    sample = [
        "Сонячна система складається з восьми планет та Сонця.",
        "Київ є столицею України.",
        "П'ять об'єктів було знайдено на території.",
    ] * 20
    tok = tk_train(iter(sample), vocab_size=800, progress=False)
    bos, eos = tok.token_to_id("<|bos|>"), tok.token_to_id("<|eos|>")
    usr, asst = 2, 3
    pairs = [
        {"question": "Київ є столицею України. Яке місто?",
         "answer": "Київ є столицею України."},
        {"question": "Сонячна система складається з восьми планет та Сонця. Що?",
         "answer": "Сонячна система складається з восьми планет та Сонця."},
    ] * 20
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "pairs.json")
        write_pairs(src, pairs)
        tok_path = os.path.join(td, "tok.json")
        tok.save(tok_path)
        ns = argparse.Namespace(
            patterns=[src], out=td, name="train", tokenizer=tok_path,
            context_width=2048, bos_id=bos, usr_id=usr, assistant_id=asst,
            end_id=eos, yes=True, dry_run=True, val_frac=0.25, seed=0,
            val_nums=False,
        )
        dry = run(ns)
        packed, n_ans = encode_pair(
            tok, pairs[0]["question"], pairs[0]["answer"], bos, usr, asst, eos)
        n_q = len(tok.encode(pairs[0]["question"], add_special_tokens=False).ids)
        n_a = len(tok.encode(pairs[0]["answer"], add_special_tokens=False).ids)
        assert packed[:2] == [bos, usr]
        assert packed[2 + n_q] == asst and packed[-1] == eos
        assert len(packed) == N_SPECIAL + n_q + n_a
        assert n_ans == n_a + 1
        assert dry["n_examples"] == 40
        assert dry["n_val"] == 10 and dry["n_train"] == 30
        assert dry["n_answer_tokens"] > 0
        assert dry["n_answer_tokens"] < dry["n_tokens"]
        assert dry["chars_per_token_val"] and dry["chars_per_token_val"] > 1
        assert not os.path.exists(os.path.join(td, "train.bin"))
        ns.patterns = [f"{src},8"]
        cap = run(ns)
        assert cap["n_sampled"] == 8
        assert cap["n_train"] + cap["n_val"] == 8
        assert cap["n_val"] == int(round(8 * 0.25))
        ns.val_nums = True
        ns.patterns = [f"{src},10,20"]
        nums = run(ns)
        assert nums["n_val"] == 10 and nums["n_train"] == 20
        ns.patterns = [f"{src},5"]
        rest = run(ns)
        assert rest["n_val"] == 5 and rest["n_train"] == 35
        ns.val_nums = False
        ns.patterns = [src]
        ns.dry_run = False
        meta = run(ns)
        assert meta["n_examples_train"] + meta["n_examples_val"] == 40
        arr = np.fromfile(os.path.join(td, "train.bin"), dtype=np.uint16)
        off = np.load(os.path.join(td, "train.offsets.npy"))
        leng = np.load(os.path.join(td, "train.lengths.npy"))
        assert arr[0] == bos and arr[1] == usr
        assert arr[int(off[0] + leng[0]) - 1] == eos
        val = np.fromfile(os.path.join(td, "val.bin"), dtype=np.uint16)
        assert val[0] == bos
        ns.context_width = 4
        ns.val_frac = 0.0
        meta2 = run(ns)
        assert meta2["n_examples"] == 0
        assert meta2["n_dropped_length"] == 40
    print("sft.tokenize self-test ok")


if __name__ == "__main__":
    import sys
    if "--self-test" in sys.argv:
        _self_test()
    else:
        run(parse_args())
