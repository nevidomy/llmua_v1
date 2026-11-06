# LLMUA v1: a controlled scaling ladder for Ukrainian language models

Code for the technical report **A Controlled Scaling Ladder for Ukrainian Language Models: Registered
Predictions and a 124M Baseline**, by Vitalii Nevidomyi (2026).

**[Read the paper (PDF)](paper/llmua_paper1.pdf)**

A 124M-parameter decoder-only model, trained from scratch in Ukrainian on 2.52B tokens of cleaned
FineWeb-2 and Ukrainian Wikipedia text. A ladder of four smaller models, 5.3M to 33.8M parameters,
selected the architecture, initialisation and learning rates. It also fitted the scaling curve that
predicted the 124M model's loss at 20, 40 and 80 tokens per parameter. Those predictions were
registered before the 124M run started. Everything ran on one RTX 5090.

## Highlights

**Predictions held.** Measured losses came in 0.3%, 1.1% and 1.6% below the registered predictions,
in bits per character (bpc) on the full validation set ([§4](paper/llmua_paper1.pdf), Table 2):


| horizon             | training tokens | registered | measured   |
| ------------------- | --------------- | ---------- | ---------- |
| 20 tokens/parameter | 2.49B           | 1.0091     | 1.0062     |
| 40 tokens/parameter | 4.97B           | 0.9858     | 0.9747     |
| 80 tokens/parameter | 9.94B           | 0.9704     | **0.9549** |


**Repeated data at 80×.** The 80× run sees each corpus token about 3.94 times. Its gains over the
shorter horizons stay close to a power-law continuation. This is a consistency check, not a
measurement of the cost of repetition (§4.6).

**Beats the previous from-scratch Ukrainian baseline.** The baseline is uk4b. Both models were scored
on the same 32,960 Brown-Ukrainian (BrUK) sentences under uk4b's own scoring rule, and uk4b Large was
run with its own code (§6, Tables 5 and 6):

- The 124M model reaches **1.327 bpc**; uk4b Large (774M parameters, about 13× the training compute)
reaches 1.4665.
- At 20 tokens per parameter the 124M model is already 0.105 bpc lower.
- The 33.8M rung at 80× is 0.019 bpc lower.


| model (80×)      | validation | BrUK documents | BrUK sentences |
| ---------------- | ---------- | -------------- | -------------- |
| P1, 5.3M         | 1.4414     | 1.5395         | 1.7189         |
| P4, 33.8M        | 1.1102     | 1.2234         | 1.4478         |
| **124M**         | **0.9549** | **1.0756**     | **1.3267**     |
| uk4b Large, 774M | –          | –              | 1.4665         |


**The evaluation set changed the conclusions.** Mid-study, evaluation moved from a 32-window subset to
the full 6,097-window validation set (§7):

- Paired first-order effects moved by at most 0.0005 bpc.
- Every fitted curvature changed.
- The move exposed a confound between learning rate and horizon in the ladder's longer branches.

**Recipe effects, measured with paired seeds across sizes (§5):**

- Block-diagonal document masking lowers loss by 0.003 to 0.006 bpc.
- Resetting RoPE positions at document boundaries has no measurable effect once masking is on.
- A residual-scaled initialisation was, unexpectedly, worse in this size range.
- Small batches cost loss under AdamW, and Muon+AdamW recovers that penalty.

**Next rungs.** The paper ends with registered predictions for 500M and 1B runs under the same recipe
(§8).

## Documents


|                                                  |                                                                                  |
| ------------------------------------------------ | -------------------------------------------------------------------------------- |
| [paper/llmua_paper1.pdf](paper/llmua_paper1.pdf) | the paper (final version of 6 November 2026)                                  |
| [experiments/README.md](experiments/README.md)   | every run of the study as a command, and which run group backs which table       |
| [uk4b/RUNBOOK.md](uk4b/RUNBOOK.md)               | how the uk4b baseline was recovered and scored                                   |
| [release/hf/README.md](release/hf/README.md)     | exporting checkpoints to Hugging Face format (`model_type: llmua`)               |
| [paper/figures/](paper/figures/)                 | the scripts and data behind every figure                                         |
| [results/](results/)                             | validation-set hashes, BrUK sentence hashes and flags, the study's 32k tokenizer |


Data and checkpoints are not in this repository. The weights are on Hugging Face:
[nevidomy/llmua-v1-124m](https://huggingface.co/nevidomy/llmua-v1-124m) (the 124M model at 80×) and
[nevidomy/llmua-v1-124m-sft](https://huggingface.co/nevidomy/llmua-v1-124m-sft) (its fine-tuning probe, Appendix M).

## Setup

Python 3.12+ and one CUDA GPU. Doc-masked training needs a GPU, because FlexAttention has no backward
pass on CPU.

```
pip install -r requirements.txt
export LLMUA_SOURCES=~/data         # raw datasets (read)
export LLMUA_ROOT=~/data/llmua      # corpus, bins, tokenizers, runs (written)
./run_tests.sh
```

Raw datasets expected under `$LLMUA_SOURCES`:


| path                                                   | dataset                                              |
| ------------------------------------------------------ | ---------------------------------------------------- |
| `fineweb-2/data/ukr_Cyrl/train/*.parquet`              | FineWeb-2, `ukr_Cyrl`                                |
| `ruvimx/UkrLM-wiki/wikipedia/train-*-of-00012.parquet` | Ukrainian Wikipedia                                  |
| `bruk/data/good/`                                      | Brown-Ukrainian; evaluation only, never training     |
| `fido-ai/`, `fido-ua-news/`                            | UA-SQuAD and UA-News from `ua_datasets` (Appendix M) |
| `ua-gec/data/gec-fluency/`                             | UA-GEC 2.0 (Appendix M)                              |
| `CohereLabs/aya_dataset/data/`                         | Aya Dataset (Appendix M)                             |




## Layout


| path                                             | what                                                              |
| ------------------------------------------------ | ----------------------------------------------------------------- |
| `model/`                                         | transformer, Muon+AdamW, LR schedules, training loop, packed eval |
| `model_runner.py`, `sft_runner.py`               | one pretraining run, one fine-tuning run                          |
| `main_runner.py`, `experiments/`                 | ordered run sets; every run as a plain command                    |
| `evaluate.py`, `evaluation/`                     | full validation, 32-window subset, BrUK protocols                 |
| `prepare_corpus.py`, `data/`, `tokens/`          | corpus, tokenizers, bins, BrUK sets                               |
| `prepare_sft.py`, `sft/`, `sft_training_loop.py` | fine-tuning data, training, chat                                  |
| `analysis/`, `perf/`                             | position loss, junk test, weight RMS, step profiles               |
| `uk4b/`                                          | recovery and scoring of the uk4b baseline                         |
| `paper/`                                         | the paper, figure scripts and their data, numeric checks          |
| `release/hf/`                                    | Hugging Face export and the `llmua_hf` package                    |
| `results/`                                       | validation-set hashes, BrUK sentence hashes and flags, tokenizer  |




## Reproducing the paper



### 1. Corpus, tokenizers, bins

Run in this order. The tokenizers were trained on the first stage only, and each stage refuses to run
out of turn.

```
python prepare_corpus.py sample --stage 1
python prepare_corpus.py tokenizers
python prepare_corpus.py sample --stage 2
python prepare_corpus.py sample --stage 3
cp results/ua_bpe_32768.json $LLMUA_ROOT/tokenizers/
python prepare_corpus.py bins --val-hashes results/val_hashes.json
```

The study's corpus file was written by unordered workers, so its line order cannot be rebuilt. A
rebuild has the same documents in a different order, with two consequences:

- `tokenizers` samples by line, so a rebuilt tokenizer differs slightly from the study's. The `cp`
line restores the study's own 32k tokenizer.
- A plain `bins` would draw a different validation split. `--val-hashes` selects the paper's
validation documents by content instead.

With both, `val.bin` comes out byte-identical to the one the study used, and `train.bin` holds the
same documents in a different order. To rebuild only `val.bin`, add `--val-only --out <dir>`.

### 2. Training

Every run of the final study is one entry in `experiments/runs.json`, with its exact command and final
metrics. See [experiments/README.md](experiments/README.md) for which group backs which table.

```
python experiments/replay.py --list
python experiments/replay.py --group p2_cooldown_branching --dry-run
python experiments/replay.py --group target_ladder
```

Runs execute in order, because a branch needs its trunk. The earlier sweeps behind Appendix C and the
80× probes are named sets:

```
python main_runner.py --list
python main_runner.py p2_lr_grid
```

SIGTERM pauses a run gracefully: it writes `ckpt_paused.pt` and exits with 42. To resume, run the same
command plus `--load_from_file <run_dir>/ckpt_paused.pt --overwrite`.

### 3. Evaluation

```
python evaluate.py instruments --ckpt <run_dir>/ckpt_final.pt   # validation, BrUK documents, BrUK sentences
python evaluate.py subset      --ckpt <run_dir>/ckpt_final.pt   # the early 32-window instrument
python evaluate.py full-val    --ckpt <run_dir>/ckpt_final.pt \
    --dataset bruk_uk4b_sentenses --mode packed_uk4b
```

Evaluation is read-only unless you pass `--save`, which stores the result on the checkpoint. Stored
values came from the GPU under bfloat16; CPU float32 differs in the fourth decimal. Build the BrUK sets
first:

```
python -m data.bruk_sentences --yes     # sentences: bins/bruk_uk4b_sentenses
python -m data.from_bruk --mode files --out $LLMUA_ROOT/bins/bruk --yes
```

`bruk_uk4b_sentenses` is misspelled on purpose, because it is part of the stored result keys. Bits per
character count Unicode code points. `packed_uk4b` is uk4b's rule: a BOS prefix, no end token, and
pooled sentence bits divided by pooled characters.

### 4. Fine-tuning probe (Appendix M)

```
python prepare_sft.py fido --train $LLMUA_SOURCES/fido-ai/train.json \
    --val $LLMUA_SOURCES/fido-ai/val.json --out-dir $LLMUA_ROOT/sft_data/text --yes
python prepare_sft.py news --train $LLMUA_SOURCES/fido-ua-news/train.csv \
    --test $LLMUA_SOURCES/fido-ua-news/test.csv --out-dir $LLMUA_ROOT/sft_data/text --yes
python prepare_sft.py gec --root $LLMUA_SOURCES/ua-gec/data/gec-fluency \
    --out-dir $LLMUA_ROOT/sft_data/text --yes
python prepare_sft.py aya --dir $LLMUA_SOURCES/CohereLabs/aya_dataset/data \
    --out-dir $LLMUA_ROOT/sft_data/text --yes
T=$LLMUA_ROOT/sft_data/text
python prepare_sft.py tokenize --val-nums --seed 47 \
    -p $T/fidoai.json,500,5000 -p $T/fidonews.json,500,6000 \
    -p $T/uagec.json,500,3678 -p $T/aya.json,200,322 \
    --out $LLMUA_ROOT/sft_data/bins/15k \
    --bos-id 0 --usr-id 201 --assistant-id 202 --end-id 203 --context-width 2048
python experiments/replay.py --id 105
python -m sft.chat --ckpt <run_dir>/ckpt_final.pt --temperature 0
```

Run #105 starts from the target's 80× checkpoint, so run `target_ladder` first. Runs #94 to #102 used an
earlier set of 2,000 training examples, built with the same tokenize command plus `--val-frac 0.2 --seed 0`
and `-p $T/fidoai.json,720 -p $T/fidonews.json,720 -p $T/uagec.json,840 -p $T/aya.json,120`, written to
`sft_data/bins/2000`.

### 5. Figures

```
OUT_DIR=/tmp/figures python paper/figures/make_figures.py
OUT_DIR=/tmp/figures python paper/figures/make_appendix_figures.py
```

Without `OUT_DIR` the scripts overwrite `paper/figures/`. The committed figures were made with
matplotlib 3.10.9; newer versions give the same plots with slightly different font metrics.

## Where each result comes from


| paper                                              | code                                                                                                                       |
| -------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| §2, Appendices A and B: corpus, cleaner, tokenizer | `prepare_corpus.py`, `data/cleanup_ua_fast.py`, `tokens/`                                                                  |
| §3, Table 1: parameter counts                      | `python -m analysis.models_sizes`                                                                                          |
| §4, Tables 2 and 3: ladder, target, horizons       | `replay.py` groups `p1`..`p4_cooldown_branching`, `target_ladder`                                                          |
| §5: masking, RoPE reset, batch size                | `main_runner.py p1_doc_mask_arms` .. `p4_doc_mask_arms`; `replay.py` `p1_basesweep*`, `p1_re_adamw_arm*`, `p1_adamw_batch` |
| §6, Tables 5 and 6: BrUK, uk4b                     | `evaluate.py instruments`, `data/bruk_*.py`, `uk4b/`                                                                       |
| Appendix C: configuration choices                  | `experiments/appendix_c.py`                                                                                                |
| Appendix C.8: position-bucketed loss               | `python -m analysis.position_bucketing`                                                                                    |
| Appendix F: junk test                              | `python -m analysis.junk_test`                                                                                             |
| Appendix G: BrUK sentence set                      | `data/bruk_sentences.py`, `results/bruk_sentence_hashes.csv`                                                               |
| Appendix H: uk4b recovery                          | `uk4b/RUNBOOK.md`, `uk4b/fix_*.py`                                                                                         |
| Appendix I: weight RMS vs Muon rate                | `python -m analysis.muon_scaling_check_a` and `_b`                                                                         |
| Appendix J: Muon overhead                          | `perf/`                                                                                                                    |
| Appendix M: fine-tuning probe                      | `prepare_sft.py`, `sft_runner.py`, `replay.py --group sft_probe`                                                           |


Entropy floors are in `model/baseline.py`. The QK-norm logit table comes from
`python -m model.qknorm_logits_bound`.

## Known limits

- `experiments/appendix_c.py` is rebuilt from lab notes and old launch scripts. The flags of the P2 to
P4 sweeps were not recorded, so those sets assume the standard recipe. Those early runs also predate
some code changes, so they will land near the logged numbers, not on them.
- The target ran as a 40× trunk (#92), a 20× branch (#93) and an extension to 80× (#104).
`replay.py --group target_ladder` repeats that sequence.
- `analysis/muon_scaling_check*.py` use the folder names of the original early runs, which differ from
what `experiments/appendix_c.py` writes. Edit the lists at the top of each script to point at your
runs.
- The causal P2 to P4 baselines predate `runs.json`. They are the sets `p2_base_5_seeds`,
`p3_base_5_seeds` and `p4_base_3_seeds`.
- A full corpus rebuild has been exercised at small scale only.
- `paper/verify/` holds the numeric checks from the review rounds. They predate the final P3 parameter
count: they use 16,416,768, not 16,422,144.



## Data and licences

- **Code:** Apache-2.0 ([LICENSE](LICENSE)).
- **uk4b code:** `uk4b/score.py` and `uk4b/model.py` come from [proger/uk4b](https://github.com/proger/uk4b)
under MIT ([uk4b/LICENSE](uk4b/LICENSE)).
- **Training data:** the corpus is built from FineWeb-2 (ODC-By 1.0) and Ukrainian Wikipedia
(CC BY-SA 4.0). Neither is redistributed here.
- **Brown-Ukrainian:** CC BY-NC-SA 4.0. It was used for evaluation only, never for training. This
repository carries only sentence hashes and flags, never its text.



## Citation

```bibtex
@misc{nevidomyi2026llmua,
  title       = {A Controlled Scaling Ladder for Ukrainian Language Models:
                 Registered Predictions and a 124M Baseline},
  author      = {Nevidomyi, Vitalii},
  year        = {2026},
  note        = {Technical report},
  url         = {https://github.com/nevidomy/llmua_v1}
}
```

