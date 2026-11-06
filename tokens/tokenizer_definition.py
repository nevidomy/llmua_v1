"""Byte-level BPE definition and trainer.

Properties enforced by construction:
  - every digit is its own token, with no punctuation or space attached, so
    "7" has the same id in every context
  - the Ukrainian orthographic apostrophe stays inside the word (п'ять)
  - no token spans a newline, or carries more than one leading space
  - byte-level fallback: no UNK is possible on any input
  - BOS/EOS added by a TemplateProcessing baked into the saved JSON, so
    encode() applies them and the rule ships with the artifact

Sampling lives in data.pipeline_ua; held-out splitting, checks, and fertility
comparison live in tokens.training.
"""

import contextlib
import json
import os
import sys
import threading

from tokenizers import Tokenizer, Regex, models, pre_tokenizers, decoders
from tokenizers import trainers, processors

TK_BOS, TK_EOS = "<|bos|>", "<|eos|>"
TK_VOCAB = 32768          # total, specials and the 256-byte alphabet included
TK_MAX_TOKEN_LEN = 32     # byte-level units; Cyrillic is 2/char, so ~16 chars

# Branches, in order:
#   1. optional non-letter prefix + letter run, apostrophes kept inside
#   2. a bare single digit -- NO prefix, so ids are context-free
#   3. punctuation run, absorbing trailing newlines
#   4. newline runs
#   5. any remaining whitespace
TK_PATTERN = Regex(
    r"[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+(?:'[\p{L}\p{M}]+)*"
    r"|\p{N}"
    r"|[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+"
)


def tk_build() -> Tokenizer:
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(TK_PATTERN, behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tok.decoder = decoders.ByteLevel()
    return tok


def tk_finalize(tok: Tokenizer) -> Tokenizer:
    bos, eos = tok.token_to_id(TK_BOS), tok.token_to_id(TK_EOS)
    tok.post_processor = processors.Sequence([
        processors.ByteLevel(trim_offsets=False),
        processors.TemplateProcessing(
            single=f"{TK_BOS} $A {TK_EOS}",
            special_tokens=[(TK_BOS, bos), (TK_EOS, eos)],
        ),
    ])
    return tok


@contextlib.contextmanager
def tk_progress(enabled=True, heartbeat=20.0):
    """indicatif output vanishes on a non-TTY (notebooks included); the json
    format always emits, so capture fd 2 and drive tqdm from it.

    Only 'Compute merges' has real granularity -- 'Tokenize words' and
    'Count pairs' emit two events each, and there are silent windows before
    the first stage and after the last. Hence the heartbeat.
    """
    import time
    try:
        from tqdm.auto import tqdm
    except ImportError:
        tqdm = None
    if not enabled or tqdm is None:
        yield
        return

    read_fd, write_fd = os.pipe()
    saved = os.dup(2)
    os.dup2(write_fd, 2)
    os.close(write_fd)

    lock = threading.Lock()
    state = {"phase": "draining iterator / counting words",
             "since": time.time(), "beat": time.time(), "stop": False}
    t0 = time.time()
    bar = {"obj": None}
    other = []

    def fmt(s):
        m, s = divmod(int(s), 60)
        h, m = divmod(m, 60)
        return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s" if m else f"{s}s"

    def pump():
        with os.fdopen(read_fd, "r", buffering=1, errors="replace") as pipe:
            for line in pipe:
                try:
                    ev = json.loads(line)
                    stage, cur, tot = ev["stage"], ev["current"], ev["total"]
                except (ValueError, KeyError):
                    other.append(line)
                    continue
                with lock:
                    state["beat"] = time.time()
                    if stage != state["phase"]:
                        if bar["obj"] is not None:
                            bar["obj"].close()
                            bar["obj"] = None
                        print(f"  done  {state['phase']} "
                              f"[{fmt(time.time()-state['since'])}]", flush=True)
                        state["phase"] = stage
                        state["since"] = time.time()
                        print(f"> {stage}  (total={tot:,})", flush=True)
                    if 0 < cur < tot:
                        if bar["obj"] is None:
                            bar["obj"] = tqdm(total=tot, desc=stage,
                                              file=sys.stdout, leave=False)
                        bar["obj"].update(cur - bar["obj"].n)

    def beat():
        while True:
            time.sleep(1.0)
            with lock:
                if state["stop"]:
                    return
                if (time.time() - state["beat"] >= heartbeat
                        and bar["obj"] is None):
                    print(f"  ... {state['phase']} "
                          f"[{fmt(time.time()-state['since'])}, "
                          f"total {fmt(time.time()-t0)}]", flush=True)
                    state["beat"] = time.time()

    print(f"> {state['phase']}", flush=True)
    tp = threading.Thread(target=pump, daemon=True)
    tb = threading.Thread(target=beat, daemon=True)
    tp.start()
    tb.start()
    try:
        yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)
        tp.join(timeout=5)
        with lock:
            state["stop"] = True
            if bar["obj"] is not None:
                bar["obj"].close()
            print(f"  done  {state['phase']} "
                  f"[{fmt(time.time()-state['since'])}]", flush=True)
        print("> assembling vocab", flush=True)
        for line in other:
            sys.stderr.write(line)
        print(f"= total {fmt(time.time()-t0)}", flush=True)


def tk_train(corpus_iter, vocab_size=TK_VOCAB, min_frequency=2,
             max_token_length=TK_MAX_TOKEN_LEN, length=None, progress=True):
    """corpus_iter: an iterable of cleaned strings."""
    tok = tk_build()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        max_token_length=max_token_length,
        special_tokens=[TK_BOS, TK_EOS],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=progress,
        progress_format="json" if progress else "none",
    )
    with tk_progress(progress):
        tok.train_from_iterator(corpus_iter, trainer=trainer, length=length)
    return tk_finalize(tok)


if __name__ == "__main__":
    sample = [
        "Сонячна система складається з восьми планет та Сонця.",
        "У 2024 році населення міста становило 1234567 осіб.",
        "П'ять об'єктів було знайдено на території заповідника.",
        "Київ є столицею України.\n\nДругий абзац тексту.",
    ] * 300
    tok = tk_train(iter(sample), vocab_size=1000, progress=False)
    assert tok.get_vocab_size() <= 1000
    e = tok.encode("У 2024 році п'ять.\n\nАбзац.")
    assert e.tokens[0] == TK_BOS and e.tokens[-1] == TK_EOS
    assert tok.decode(e.ids, skip_special_tokens=True) == "У 2024 році п'ять.\n\nАбзац."
    ids = {}
    for ctx in ["{d}", " {d}", "-{d}", ".{d}", "({d})"]:
        for d in "0123456789":
            enc = tok.encode(ctx.format(d=d), add_special_tokens=False)
            pos = [i for i, t in enumerate(enc.tokens) if t == d]
            assert len(pos) == 1
            ids.setdefault(d, enc.ids[pos[0]])
            assert ids[d] == enc.ids[pos[0]]
    print(f"vocab {tok.get_vocab_size()}, digits stable, roundtrip exact")