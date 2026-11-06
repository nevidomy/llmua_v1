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
pip install sentencepiece termcolor pandas numpy

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

```bash
STRINGS=$LLMUA_ROOT/bins/bruk_uk4b_sentenses/strings.jsonl
cat > uk4b_prep.py <<'EOF'
import json, hashlib, sys
src = sys.argv[1]
rows = [json.loads(l) for l in open(src, encoding='utf-8')]
paper = [r for r in rows if not r['exact_hit']]
assert len(paper) == 32960, len(paper)
for r in paper:
    t = r['text']
    assert t == t.strip() and '\n' not in t and '\t' not in t, r['sha1']
    assert len(t) == r['n_chars'], r['sha1']
    assert hashlib.sha1(t.encode('utf-8')).hexdigest() == r['sha1'], r['sha1']
def dump(path, texts):  # NO trailing newline: score.py splits on '\n'; an empty last line → nan row
    open(path, 'w', encoding='utf-8').write('\n'.join(texts))
texts = [r['text'] for r in paper]
dump('paper_set.txt', texts)
dump('paper_set_typo.txt', [t.replace("'", '\u2019') for t in texts])
dump('apos_a.txt', texts[:2000])
dump('apos_b.txt', [t.replace("'", '\u2019') for t in texts[:2000]])
print('ok', len(texts), 'sentences')
EOF
python uk4b_prep.py $STRINGS
wc -l paper_set.txt   # prints 32959 (no trailing newline) — correct
```

## 4. Apostrophe check on Small (2,000 sentences, ~1 min)

BrUK is natively ASCII-apostrophe, so `paper_set.txt` as-is *is* their published protocol
on the same raw text. This check only tells you whether to add a footnote.

```bash
python score.py --tsv --spm wiki.model --sentences apos_a.txt small.pt > apos_a.tsv 2> apos_a.log
python score.py --tsv --spm wiki.model --sentences apos_b.txt small.pt > apos_b.tsv 2> apos_b.log

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

```bash
for m in small medium large; do
  python score.py --tsv --spm wiki.model --sentences paper_set.txt $m.pt > $m.tsv 2> $m.log
  grep -c truncated $m.log   # should print 0
done
```

Small ≈ 5 min, Medium ≈ 10, Large ≈ 15–20 at batch 1.

## 6. Join back to your sentences and compute bpc

```bash
cat > uk4b_join.py <<'EOF'
import json, sys, numpy as np, pandas as pd
src, model, tsv = sys.argv[1], sys.argv[2], sys.argv[3]
rows = [json.loads(l) for l in open(src, encoding='utf-8')]
paper = [r for r in rows if not r['exact_hit']]
df = pd.read_csv(tsv, sep='\t', quoting=3, dtype={'id': str})
assert len(df) == len(paper) and df.id.is_unique, (len(df), len(paper))
assert np.isfinite(df.ppl.values).all()
bits  = dict(zip(df.id, np.log(df.ppl.values) * df.sentence_len.values / np.log(2)))
chars = dict(zip(df.id, df.sentence.str.len().values))
tot_b = tot_c = 0.0
for r in paper:
    assert r['sha1'] in bits, r['sha1']                 # every sentence scored, none extra
    assert chars[r['sha1']] == r['n_chars'], r['sha1']  # identical string on both sides
    r[f'bits_uk4b_{model}'] = float(bits[r['sha1']])
    tot_b += bits[r['sha1']]; tot_c += r['n_chars']
print(f'{model}: {tot_b / tot_c:.4f} bpc  ({len(paper)} sentences, {int(tot_c)} chars)')
with open(f'strings_uk4b_{model}.jsonl', 'w', encoding='utf-8') as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False) + '\n')
EOF
for m in small medium large; do python uk4b_join.py $STRINGS $m $m.tsv; done
```

Output is three lines, e.g. `small: 1.5xxx bpc`. Those go in the table next to
P1 1.835 · P2 1.713 · P3 1.631 · P4 1.530 · target 1.394.

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
