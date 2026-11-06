"""Generate text from a trained run.

    python sample_generator.py /path/to/runs/llmua_P1/Phase0_..._seed_47
    python sample_generator.py <run_dir> --ckpt ckpt_best.pt --tokens 500 \
        --prompt "Іван" --width 100

Architecture is restored from the checkpoint's model_arch field. For
checkpoints saved before that field existed, re-pass the train-time model
flags, e.g. --model P1 --enable_init --enable_swiglu --enable_rms_norm
--enable_qk_norm
"""

import argparse
import os
import textwrap

import torch

from model.transformer import (Transformer, model_P1, model_P2, model_P3,
                               model_P4, model_target, model_test)

FACTORIES = {"test": model_test, "P1": model_P1, "P2": model_P2,
             "P3": model_P3, "P4": model_P4, "target": model_target}


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", help="run directory holding the checkpoint")
    p.add_argument("--ckpt", default="ckpt_final.pt",
                   help="checkpoint filename inside run_dir")
    p.add_argument("--tokens", type=int, default=1000,
                   help="number of tokens to generate")
    p.add_argument("--prompt", default=None, help="optional prompt string")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="softmax temperature; 0 = greedy")
    p.add_argument("--top_p", type=float, default=None,
                   help="nucleus sampling p in (0, 1]; omit for raw softmax")
    p.set_defaults(kv_cache=True)
    p.add_argument("--kv_cache", dest="kv_cache", action="store_true",
                   help="incremental KV cache during decode (default on)")
    p.add_argument("--no_kv_cache", dest="kv_cache", action="store_false",
                   help="full-window forward each token")
    p.add_argument("--show_tokens", action="store_true",
                   help="print per-token breakdown (prompt vs generated)")
    p.add_argument("--width", type=int, default=120,
                   help="output wrap width; 0 disables wrapping")
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    # legacy checkpoints only (no model_arch stored)
    p.add_argument("--model", choices=list(FACTORIES), default=None)
    p.add_argument("--enable_init", action="store_true")
    p.add_argument("--enable_swiglu", action="store_true")
    p.add_argument("--enable_rms_norm", action="store_true")
    p.add_argument("--enable_qk_norm", action="store_true")
    p.add_argument("--doc_masking", action="store_true")
    p.add_argument("--doc_rope", action="store_true")
    return p.parse_args()


def wrap_text(text, width):
    """Wrap on word boundaries; only split a word if it alone exceeds width."""
    if not width:
        return text
    return "\n".join(
        textwrap.fill(para, width=width,
                      break_long_words=True, break_on_hyphens=False)
        for para in text.splitlines())


def load_model(ckpt_path, device, model=None, enable_init=False,
               enable_swiglu=False, enable_rms_norm=False,
               enable_qk_norm=False, doc_masking=False, doc_rope=False):
    """Reconstruct the model from a checkpoint. Architecture comes from the
    checkpoint's model_arch field; for legacy checkpoints the train-time
    factory/flags must be re-passed explicitly."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if ck.get("model_arch"):
        m = Transformer(**ck["model_arch"])
    else:
        assert model, (
            f"{ckpt_path} predates model_arch; re-pass the train-time "
            f"flags: --model {{{','.join(FACTORIES)}}} --enable_init ...")
        m = FACTORIES[model](
            enable_init=enable_init, enable_swiglu=enable_swiglu,
            enable_rms_norm=enable_rms_norm, enable_qk_norm=enable_qk_norm,
            doc_masking=doc_masking, doc_rope=doc_rope)
    sd = ck["model"]
    if any(k.startswith("_orig_mod.") for k in sd):  # legacy compiled ckpt
        sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    m.load_state_dict(sd)
    return m.to(device).eval(), ck


def header(model, ck, ckpt_path):
    """Model/ckpt summary block printed above the generated text. arch comes
    from the constructed model, so it works for legacy checkpoints too."""
    a = model.arch
    flags = [k for k in ("custom_init", "no_biases", "residual_init",
                         "use_swiglu", "rms_norm", "qk_norm",
                         "doc_masking", "doc_rope") if a.get(k)]
    tokens = ck.get("tokens", 0)
    lines = [
        f"model  {a['name']} -- {a.get('description') or 'no description'}",
        f"arch   {a['embeddings']}d x {a['blocks']} blocks, "
        f"ff {a['ff_width']}, head {a['headsize']}, ctx {a['width']}, "
        f"vocab {a['vocab_size']:,}  [{', '.join(flags) or 'plain'}]",
        f"params {model.num_params():,} total, "
        f"{model.num_params_no_embedings():,} non-embedding",
    ]
    cfg = ck.get("cfg") or {}
    if "lr" in cfg:
        wd = cfg.get("weight_decay", 0.1)
        if cfg.get("use_muon"):
            m = cfg.get("muon_lr")          # None -> lr * muon_lr_mult
            m = m if m is not None else cfg["lr"] * cfg.get("muon_lr_mult", 30.0)
            lines.append(f"optim  muon lr={m:g} + adamw lr={cfg['lr']:g}, "
                         f"wd {wd:g}")
        else:
            lines.append(f"optim  adamw lr={cfg['lr']:g} (muon off), "
                         f"wd {wd:g}")
    lines += [
        f"ckpt   {ckpt_path}: step {ck.get('step')}",
        f"train  {tokens/1e6:.1f}M tokens "
        f"= {tokens/model.num_params():.1f} tokens/param",
    ]
    val = ck.get("final") or (ck.get("history") or [None])[-1]
    if val:
        at = f" @ step {val['step']}" if "step" in val else ""
        lines.append(f"val    {val['val_nats']:.4f} nats/tok = "
                     f"{val['val_bpc']:.4f} bpc{at}")
    return "\n".join(lines)


def load_sample_decoder(model, ck=None):
    """TokenDecoder for sample display; falls back to the shared BPE file."""
    from sft.chat import _resolve_tokenizer, specials_from_cfg
    from sft.probe import load_decoder
    cfg = (ck or {}).get("cfg") or {}
    sp = specials_from_cfg(cfg, model)
    dec = load_decoder(sp.get("data_dir"), sp["bos_id"], sp["usr_id"],
                       sp["assistant_id"], sp["end_id"])
    return _resolve_tokenizer(dec, sp.get("data_dir"), model.vocab_size)


def _decode_ids(decoder, ids):
    ids = [int(t) for t in ids]
    if not ids:
        return ""
    if decoder._hf is not None:
        return decoder._hf.decode(ids, skip_special_tokens=True)
    return "".join(decoder.piece(t) for t in ids)


def generate_sample(model, tokens, prompt=None, progress_cb=None,
                    temperature=1.0, top_p=None, use_kv_cache=True,
                    decoder=None, ck=None, prompt_ids=None):
    """Generate and return text plus a prompt/generated token split.

    `tokens` is the number of *new* ids to sample. When `prompt` / `prompt_ids`
    is set those ids are the prefix (role=prompt), always starting with BOS
    (prepended if the encoded prompt does not already). Everything after is
    generated. Empty prompt is unconditional (generate seeds BOS internally
    and strips it from the returned ids).
    """
    dec = decoder or load_sample_decoder(model, ck)
    if prompt_ids is None:
        prompt_ids = []
        if prompt:
            if dec._hf is None:
                raise RuntimeError("no tokenizer to encode the sample prompt")
            prompt_ids = list(dec._hf.encode(
                prompt, add_special_tokens=False).ids)
    else:
        prompt_ids = [int(t) for t in prompt_ids]
    bos = int(getattr(model, "bos_id", 0) or 0)
    if prompt_ids and prompt_ids[0] != bos:
        prompt_ids = [bos] + prompt_ids
    ids = model.generate(tokens, prompt=prompt_ids, progress_cb=progress_cb,
                         temperature=temperature, top_p=top_p,
                         use_kv_cache=use_kv_cache)
    n_prompt = len(prompt_ids)
    if prompt_ids and list(ids[:n_prompt]) != prompt_ids:
        n_prompt = 0
    pieces = []
    for i, tid in enumerate(ids):
        piece = dec.token(int(tid))
        piece["role"] = "prompt" if i < n_prompt else "generated"
        pieces.append(piece)
    gen_ids = ids[n_prompt:]
    return {
        "text": _decode_ids(dec, ids),
        "prompt": prompt or "",
        "prompt_text": _decode_ids(dec, prompt_ids),
        "generated_text": _decode_ids(dec, gen_ids),
        "tokens": pieces,
        "n_prompt": n_prompt,
        "n_generated": len(gen_ids),
        "use_kv_cache": bool(use_kv_cache),
        "temperature": temperature,
        "top_p": top_p,
    }


def generate_text(model, tokens, prompt=None, progress_cb=None,
                  temperature=1.0, top_p=None, use_kv_cache=True,
                  decoder=None, ck=None):
    """Core generation: returns the decoded text (unwrapped)."""
    return generate_sample(
        model, tokens, prompt=prompt, progress_cb=progress_cb,
        temperature=temperature, top_p=top_p, use_kv_cache=use_kv_cache,
        decoder=decoder, ck=ck)["text"]


def main():
    args = parse_args()
    ckpt_path = os.path.join(args.run_dir, args.ckpt)
    model, ck = load_model(ckpt_path, args.device, model=args.model,
                           enable_init=args.enable_init,
                           enable_swiglu=args.enable_swiglu,
                           enable_rms_norm=args.enable_rms_norm,
                           enable_qk_norm=args.enable_qk_norm,
                           doc_masking=args.doc_masking,
                           doc_rope=args.doc_rope)
    rec = generate_sample(model, args.tokens, prompt=args.prompt,
                          temperature=args.temperature, top_p=args.top_p,
                          use_kv_cache=args.kv_cache, ck=ck)

    print(header(model, ck, ckpt_path))
    if args.prompt:
        print(f"prompt {args.prompt!r}")
    print(f"kv     {'on' if rec['use_kv_cache'] else 'off'}  "
          f"{rec['n_prompt']} prompt + {rec['n_generated']} generated")
    print()
    if rec["n_prompt"]:
        print("PROMPT")
        print(wrap_text(rec["prompt_text"], args.width))
        print()
        print("GENERATED")
    print(wrap_text(rec["generated_text"] or rec["text"], args.width))
    if args.show_tokens:
        print()
        print("TOKENS")
        for tk in rec["tokens"]:
            tag = "P" if tk["role"] == "prompt" else "G"
            print(f"  {tag}  {tk['id']:>6}  {tk['t']}")


if __name__ == "__main__":
    main()
