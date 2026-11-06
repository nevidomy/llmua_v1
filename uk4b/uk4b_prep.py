# uk4b_prep.py — strings.jsonl → one sentence per line for score.py
import json, hashlib, os
rows = [json.loads(l) for l in open(os.path.join(os.environ.get('LLMUA_ROOT', os.path.expanduser('~/data/llmua')), 'bins/bruk_uk4b_sentenses/strings.jsonl'), encoding='utf-8')]
paper = [r for r in rows if not r['exact_hit']]
assert len(paper) == 32960
for r in paper:
    t = r['text']
    assert t == t.strip() and '\n' not in t and '\t' not in t, r['sha1']
    assert len(t) == r['n_chars'], r['sha1']
    assert hashlib.sha1(t.encode('utf-8')).hexdigest() == r['sha1'], r['sha1']
# no trailing newline: score.py splits on '\n' and an empty last line → output_length 0 → crash
open('paper_set.txt', 'w', encoding='utf-8').write('\n'.join(r['text'] for r in paper))
open('paper_set_typo.txt', 'w', encoding='utf-8').write('\n'.join(r['text'].replace("'", '’') for r in paper))