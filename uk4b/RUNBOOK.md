# uk4b on the paper sentence set — runbook

Goal: score uk4b Small / Medium / Large on exactly the 32,960 sentences your ladder was
scored on, so the comparison has one variable (the model). Run on the 5090 box.

Everything below is copy-paste, in order. Total GPU time ≈ 30–40 min, mostly Large.

---

## 1. Set up a working directory, fetch their code, tokenizer, checkpoints

```bash
export LLMUA_ROOT=${LLMUA_ROOT:-~/data/llmua}   # data root: strings.jsonl lives under it
REPO=/path/to/this/repository                    # the checkout that contains uk4b/
mkdir -p ~/uk4b && cd ~/uk4b
pip install sentencepiece termcolor pandas numpy bitsandbytes   # bitsandbytes: model.py's StableEmbedding; do not stub it

curl -sLO https://raw.githubusercontent.com/proger/uk4b/main/examples/score.py
curl -sLO https://raw.githubusercontent.com/proger/uk4b/main/examples/model.py
curl -sLO https://a.wilab.org.ua/gpt/wiki.model                       # SentencePiece tokenizer
curl -Lo small.pt  https://a.wilab.org.ua/gpt/small13100.pt           # GPT-2 Small  123M
curl -Lo medium.pt https://a.wilab.org.ua/gpt/uk4b_medium/ckpt2.096.pt # GPT-2 Medium 355M
curl -Lo large.pt  https://a.wilab.org.ua/gpt/ckpt10m.pt              # GPT-2 Large  774M (the paper's)
ls -la *.pt wiki.model
```

Large may be several GB if the file carries optimizer state; that's fine, only `model` is read.

The origin host has since gone offline, so the URLs above are the record of what the files were.
The two files the paper uses came from Internet Archive captures of them: `ckpt10m.pt` at
`https://web.archive.org/web/20250716223109/https://a.wilab.org.ua/gpt/ckpt10m.pt` and `wiki.model` at
`https://web.archive.org/web/20250823074634/https://a.wilab.org.ua/gpt/wiki.model`. Small and Medium
were never captured.

If a download came back as an nginx 404 page with the real file behind it, repair it before use.
Each script writes nothing unless the repaired file checks out (CRC for `large.pt`, a load test
for `wiki.model`):

```bash
head -c 16 wiki.model | grep -q '<html' && python $REPO/uk4b/fix_wiki_model.py wiki.model
head -c 16 large.pt   | grep -q '<html' && python $REPO/uk4b/fix_large_pt.py large.pt
```

The patched `score.py` and `model.py` live in `$REPO/uk4b/`. Copy them over the upstream
files fetched above, and use those copies:

```bash
cp $REPO/uk4b/score.py $REPO/uk4b/model.py .
```

## 2. What was changed in `score.py`

The shipped `score.py` already contains these changes, so do not run the sed lines below on it.
They are the record of how the patched copy was made from upstream:

- torch ≥ 2.6 refuses their pickle unless `weights_only=False`.
- batch 1 with a different length per sentence recompiles endlessly unless `dynamic=True`.
- the shipped copy strips the `_orig_mod.` prefix from the checkpoint keys itself, asserts that no
  weights are missing or unexpected (`attn.bias` aside), and applies `torch.compile` after loading.
  A silent `strict=False` load of nothing therefore fails loudly.

```bash
# first-pass edits, applied to upstream text (historical; do not run on the shipped copy)
sed -i 's/torch.load(args.ckpt_path, map_location=device)/torch.load(args.ckpt_path, map_location=device, weights_only=False)/' score.py
sed -i 's/model = torch.compile(model) # requires PyTorch 2.0/model = torch.compile(model, dynamic=True)/' score.py
sed -i "s/^    model.load_state_dict(checkpoint\['model'\], strict=False)\$/    print(model.load_state_dict(checkpoint['model'], strict=False), file=sys.stderr)/" score.py
```

If compile fails on your torch, use the fallback in §7.

## 3. Write the input files from `strings.jsonl`

`uk4b_prep.py` reads `$LLMUA_ROOT/bins/bruk_uk4b_sentenses/strings.jsonl`, checks every sentence (count, hash,
length, no newline or tab), and writes `paper_set.txt` and `paper_set_typo.txt` (apostrophes replaced by U+2019)
to the current directory. The files have no trailing newline, because `score.py` splits on `\n` and an empty
last line becomes a `nan` row. The two 2,000-sentence files for §4 are cut from them the same way:

```bash
python $REPO/uk4b/uk4b_prep.py
wc -l paper_set.txt   # prints 32959 (no trailing newline) — correct
head -n 2000 paper_set.txt      | head -c -1 > apos_a.txt
head -n 2000 paper_set_typo.txt | head -c -1 > apos_b.txt
```

## 4. Apostrophe check on Small (2,000 sentences, ~1 min)

BrUK is natively ASCII-apostrophe, so `paper_set.txt` as-is *is* their published protocol
on the same raw text. This check only tells you whether to add a footnote.

```bash
python score.py small.pt --tsv --spm wiki.model --sentences apos_a.txt > apos_a.tsv 2> apos_a.log
python score.py small.pt --tsv --spm wiki.model --sentences apos_b.txt > apos_b.tsv 2> apos_b.log

grep -m1 "load_state_dict\|All keys\|Incompatible" apos_a.log   # must NOT list missing transformer weights
python - <<'EOF'
import numpy as np, pandas as pd
for f in ['apos_a.tsv', 'apos_b.tsv']:
    df = pd.read_csv(f, sep='\t', quoting=3)
    bits = np.log(df.ppl.values) * df.sentence_len.values / np.log(2)
    print(f, f'{bits.sum() / df.sentence.str.len().sum():.4f} bpc', f'median ppl {df.ppl.median():.1f}')
EOF
```

Sanity: median per-token ppl should be tens, not thousands. Thousands = weights didn't load.
If `apos_b` is better for them by > 0.005, run §5 on `paper_set_typo.txt` **as a second
column** and report it as a footnote; the headline stays on `paper_set.txt`.

## 5. Score the full set with all three checkpoints

Put the checkpoint path first: `--sentences` takes any number of files, so a path after it is read as another
sentence file and the run fails.

```bash
for m in small medium large; do
  python score.py $m.pt --tsv --spm wiki.model --sentences paper_set.txt > $m.tsv 2> $m.log
  grep -c truncated $m.log   # should print 0
done
```

Small ≈ 5 min, Medium ≈ 10, Large ≈ 15–20 at batch 1.

## 6. Join back to your sentences and compute bpc

`uk4b_join.py <name> <tsv>` matches each scored sentence to `strings.jsonl` by sha1, checks that every sentence was
scored and that the character counts agree, prints the pooled bpc, and writes `strings_uk4b_<name>.jsonl` to the
current directory:

```bash
for m in small medium large; do python $REPO/uk4b/uk4b_join.py $m $m.tsv; done
```

Each run prints one line, e.g. `large: 1.4xxx bpc on 32960 sentences, N chars`. These go in the table next to the
ladder's sentence column (Table 5, uk4b rule): P1 1.7189 · P2 1.6096 · P3 1.5372 · P4 1.4478 · target 1.3267.
The paper's uk4b Large figure on the same sentences is 1.4665.

## 7. Fallback if `torch.compile` fails

Comment out the compile line in the copied `score.py`. The weights still load before that line,
and the missing-key assert stays in place, so a failed load still stops the run:

```bash
sed -i 's/^    model = torch.compile(model, dynamic=True)/    # model = torch.compile(model, dynamic=True)/' score.py
```

Then re-run §4. The run must get past the key checks, which allow only `attn.bias` buffers
in `unexpected_keys`.

## 8. Optional: alternative Large checkpoints

`ckpt8.5m.pt` and `ckpt8m.pt` at the same URL prefix are earlier/other Large runs. Scoring
them and reporting their best is the generous version; `ckpt10m.pt` is the one their paper
and README point to.
