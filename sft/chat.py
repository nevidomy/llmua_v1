"""Interactive SFT chat: format, generate with KV cache, probe each token.

Each send is an independent example, same template as SFT:

    <BOS> <USR> {text} <ASSISTANT>  {generated …}  <END>

Prior turns are not in the context window. The KV cache is dropped at the
start of every send so nothing from the last reply can leak.
"""
import os

import torch
import torch.nn.functional as F

from sft.probe import TOP_K, load_decoder

DEFAULT_USR, DEFAULT_ASSISTANT, DEFAULT_END = 201, 202, 203


def specials_from_cfg(cfg, model=None):
    cfg = cfg or {}
    bos = int(cfg.get("bos_id", getattr(model, "bos_id", 0) or 0))
    return {
        "bos_id": bos,
        "usr_id": int(cfg.get("usr_id", DEFAULT_USR)),
        "assistant_id": int(cfg.get("assistant_id", DEFAULT_ASSISTANT)),
        "end_id": int(cfg.get("end_id", DEFAULT_END)),
        "data_dir": cfg.get("data_dir"),
    }


def format_user_turn(text, *, bos_id, usr_id, assistant_id, encode):
    """Ids for one isolated user message: BOS USR {text} ASSISTANT."""
    q = list(encode(text))
    return [int(bos_id), int(usr_id)] + q + [int(assistant_id)]


def _resolve_tokenizer(decoder, data_dir, vocab_size):
    if decoder._hf is not None:
        return decoder
    from data.index import ROOT
    cands = []
    if data_dir:
        cands.append(os.path.join(data_dir, f"ua_bpe_{vocab_size}.json"))
    cands.append(os.path.join(ROOT, "tokenizers", f"ua_bpe_{vocab_size}.json"))
    for p in cands:
        if p and os.path.isfile(p):
            from tokenizers import Tokenizer
            decoder._hf = Tokenizer.from_file(p)
            break
    return decoder


def encode_text(decoder, text):
    if decoder._hf is None:
        raise RuntimeError("no tokenizer loaded for chat encode")
    return list(decoder._hf.encode(text, add_special_tokens=False).ids)


def decode_piece(decoder, tid):
    """Plain text for display; specials are omitted."""
    tid = int(tid)
    if tid in decoder.labels:
        return ""
    if decoder._hf is None:
        return ""
    try:
        s = decoder._hf.decode([tid], skip_special_tokens=False)
    except Exception:
        s = None
    return s or ""


def probe_choice(logits, decoder, chosen_id, k=TOP_K):
    """Deep-probe-shaped record for one generated token (no gold label).

    `expected` is the sampled id; `pred` is greedy argmax. hit = they match.
    """
    logits = logits.float().view(-1)
    probs = F.softmax(logits, dim=-1)
    V = probs.numel()
    k = int(min(k, V))
    chosen_id = int(chosen_id)
    argmax_id = int(torch.argmax(probs).item())
    p_chosen = float(probs[chosen_id].item())
    p_pred = float(probs[argmax_id].item())
    rank = int(1 + (probs > probs[chosen_id]).sum().item())
    top_p, top_i = torch.topk(probs, k)
    opts, seen = [], set()
    for r in range(k):
        oid = int(top_i[r].item())
        seen.add(oid)
        piece = decoder.token(oid)
        opts.append({
            "id": oid, "t": piece["t"], "spec": piece["spec"],
            "p": float(top_p[r].item()), "k": r + 1,
            "expected": oid == chosen_id,
        })
    if chosen_id not in seen:
        piece = decoder.token(chosen_id)
        opts.append({
            "id": chosen_id, "t": piece["t"], "spec": piece["spec"],
            "p": p_chosen, "k": rank, "expected": True,
        })
    opts.sort(key=lambda o: (-o["p"], o["k"]))
    return {
        "expected": decoder.token(chosen_id),
        "pred": decoder.token(argmax_id),
        "p_expected": p_chosen,
        "p_pred": p_pred,
        "rank": rank,
        "hit": argmax_id == chosen_id,
        "opts": opts,
    }


class ChatSession:
    def __init__(self, model, decoder, bos_id, usr_id, assistant_id, end_id,
                 temperature=0.0, top_p=None, use_cache=True):
        self.model = model
        self.decoder = decoder
        self.bos_id = int(bos_id)
        self.usr_id = int(usr_id)
        self.assistant_id = int(assistant_id)
        self.end_id = int(end_id)
        self.temperature = float(temperature)
        self.top_p = top_p
        self.use_cache = bool(use_cache)
        self.ids = []
        self.cache = None
        self.messages = []
        self._saved_doc_masking = bool(getattr(model, "doc_masking", False))
        model.doc_masking = False
        model.eval()

    def close(self):
        self.model.doc_masking = self._saved_doc_masking

    def _encode(self, text):
        return encode_text(self.decoder, text)

    def _device(self):
        return self.model.lmh.weight.device

    def _cache_len(self):
        if self.cache is None:
            return 0
        return int(self.cache[0][0].shape[-2])

    def _reset_context(self):
        """Drop the last turn's ids and KV so the next send is isolated."""
        self.ids = []
        self.cache = None

    def _logits_full(self):
        """Last-position logits from a full forward (no KV cache)."""
        x = torch.tensor([self.ids], device=self._device())
        logits = self.model(x)
        return logits[0, -1, :]

    def _forward_new(self, new_ids):
        """Prefill `new_ids`, using the cache when it matches self.ids prefix."""
        if not self.use_cache:
            self.cache = None
            return self._logits_full()
        device = self._device()
        if not new_ids:
            raise RuntimeError("empty prefill")
        prefix_len = len(self.ids) - len(new_ids)
        if self.cache is None or self._cache_len() != prefix_len:
            x = torch.tensor([self.ids], device=device)
            logits, self.cache = self.model(x, use_cache=True)
        else:
            x = torch.tensor([new_ids], device=device)
            logits, self.cache = self.model(x, kv_cache=self.cache, use_cache=True)
        return logits[0, -1, :]

    def _decode_one(self, tid):
        if not self.use_cache:
            return self._logits_full()
        device = self._device()
        x = torch.tensor([[int(tid)]], device=device)
        logits, self.cache = self.model(x, kv_cache=self.cache, use_cache=True)
        return logits[0, -1, :]

    def send(self, text, max_tokens=None, temperature=None, top_p=None,
             use_cache=None, on_token=None, on_user=None, abort=None):
        """Generate a reply for one user message. Prior turns are not used.

        Returns (user_msg, assistant_msg); both are stored on self.messages
        for inspection only.
        """
        text = "" if text is None else str(text)
        if temperature is None:
            temperature = self.temperature
        if top_p is None:
            top_p = self.top_p
        if use_cache is not None:
            self.use_cache = bool(use_cache)
        self._reset_context()
        prefix = format_user_turn(
            text, bos_id=self.bos_id, usr_id=self.usr_id,
            assistant_id=self.assistant_id, encode=self._encode)
        width = int(self.model.context_width)
        if len(prefix) >= width:
            raise ValueError(
                f"message does not fit in context "
                f"({len(prefix)} >= {width})")
        user_tokens = [self.decoder.token(i) for i in prefix]
        user_msg = {"role": "user", "text": text, "tokens": user_tokens}
        self.messages.append(user_msg)
        if on_user is not None:
            on_user(user_msg)
        self.ids = list(prefix)

        logits = self._forward_new(prefix)
        room = width - len(self.ids)
        if max_tokens is None or int(max_tokens) <= 0:
            limit = room
        else:
            limit = min(int(max_tokens), room)
        gen_ids, outputs, pieces = [], [], []
        stop = "limit"
        for _ in range(max(limit, 0)):
            if abort is not None and abort.is_set():
                stop = "abort"
                break
            nid = int(self.model.get_next_token_probs(
                logits.float(), temperature, top_p))
            piece = decode_piece(self.decoder, nid)
            pieces.append(piece)
            gen_ids.append(nid)
            # stream the piece before top-k probe work
            if on_token is not None:
                on_token(piece, nid)
            rec = probe_choice(logits, self.decoder, nid)
            rec["pos"] = len(self.ids) - 1
            rec["prev"] = self.decoder.token(self.ids[-1])
            outputs.append(rec)
            self.ids.append(nid)
            done = (nid == self.end_id or len(self.ids) >= width
                    or (abort is not None and abort.is_set()))
            if nid == self.end_id:
                stop = "end"
            elif len(self.ids) >= width:
                stop = "limit"
            elif abort is not None and abort.is_set():
                stop = "abort"
            if done:
                if self.use_cache and len(self.ids) < width:
                    self._decode_one(nid)
                break
            logits = self._decode_one(nid)

        asst_tokens = [self.decoder.token(i) for i in gen_ids]
        msg = {
            "role": "assistant",
            "text": "".join(pieces),
            "tokens": asst_tokens,
            "outputs": outputs,
            "stop": stop,
            "n": len(gen_ids),
        }
        self.messages.append(msg)
        return user_msg, msg


def main():
    import argparse
    import torch
    from sample_generator import load_model

    p = argparse.ArgumentParser(
        description="Terminal chat with an SFT checkpoint: one line in, "
                    "generated text out. Each line is an independent prompt.")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=None,
                   help="nucleus p in (0, 1]; omit for raw softmax")
    p.add_argument("--no_cache", action="store_true",
                   help="disable the KV cache during decode")
    p.add_argument("--max_tokens", type=int, default=None)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = load_model(args.ckpt, device)
    spec = specials_from_cfg(ck.get("cfg"), model)
    decoder = load_decoder(spec.get("data_dir"), spec["bos_id"], spec["usr_id"],
                           spec["assistant_id"], spec["end_id"])
    _resolve_tokenizer(decoder, spec.get("data_dir"), model.vocab_size)
    if decoder._hf is None:
        raise RuntimeError("could not load tokenizer for chat")
    session = ChatSession(
        model, decoder, spec["bos_id"], spec["usr_id"], spec["assistant_id"],
        spec["end_id"], temperature=args.temperature, top_p=args.top_p,
        use_cache=not args.no_cache)
    try:
        while True:
            try:
                line = input("> ")
            except EOFError:
                break
            if not line.strip():
                continue
            _user, reply = session.send(line, max_tokens=args.max_tokens)
            print(reply["text"])
    finally:
        session.close()


if __name__ == "__main__":
    main()
