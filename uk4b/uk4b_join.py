# uk4b_join.py — TSV → bits per sentence by sha1 → bpc on the paper set
import json, os, sys, numpy as np, pandas as pd
model, tsv = sys.argv[1], sys.argv[2]                     # e.g. small small.tsv
rows = [json.loads(l) for l in open(os.path.join(os.environ.get('LLMUA_ROOT', os.path.expanduser('~/data/llmua')), 'bins/bruk_uk4b_sentenses/strings.jsonl'), encoding='utf-8')]
paper = [r for r in rows if not r['exact_hit']]
df = pd.read_csv(tsv, sep='\t', quoting=3, dtype={'id': str})
assert len(df) == len(paper) and df.id.is_unique
bits = dict(zip(df.id, np.log(df.ppl.values) * df.sentence_len.values / np.log(2)))
chars = dict(zip(df.id, df.sentence.str.len().values))
tot_b = tot_c = 0.0
for r in paper:
    assert r['sha1'] in bits, r['sha1']                   # every sentence scored, none extra
    assert chars[r['sha1']] == r['n_chars'], r['sha1']    # identical string on both sides
    r[f'bits_uk4b_{model}'] = float(bits[r['sha1']])
    tot_b += bits[r['sha1']]; tot_c += r['n_chars']
print(f'{model}: {tot_b / tot_c:.4f} bpc on {len(paper)} sentences, {int(tot_c)} chars')
with open(f'strings_uk4b_{model}.jsonl', 'w', encoding='utf-8') as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False) + '\n')