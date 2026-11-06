"""Chat format + session (CPU, tiny model).

Run: python -m sft.test_chat
"""
import torch

from model.transformer import Transformer
from sft.chat import (ChatSession, format_user_turn, probe_choice,
                      specials_from_cfg)
from sft.probe import load_decoder

BOS, USR, ASST, END = 0, 201, 202, 203


def check(cond, label):
    assert cond, label
    print(f"  ok: {label}")


class FakeTok:
    def encode(self, text, add_special_tokens=False):
        class R:
            ids = [10 + (ord(c) % 40) for c in (text or "x")]
        return R()

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(97 + (int(i) % 26)) for i in ids)

    def id_to_token(self, i):
        return f"[{i}]"


def encode(text):
    return FakeTok().encode(text).ids


def test_format():
    print("=== format_user_turn ===")
    a = format_user_turn("ab", bos_id=BOS, usr_id=USR,
                         assistant_id=ASST, encode=encode)
    check(a[0] == BOS and a[1] == USR and a[-1] == ASST,
          f"BOS USR … ASSISTANT, got {a}")
    check(len(a) == 2 + 2 + 1, "BOS+USR+2 chars+ASSISTANT")
    b = format_user_turn("z", bos_id=BOS, usr_id=USR,
                         assistant_id=ASST, encode=encode)
    check(b[0] == BOS and b[1] == USR and b[-1] == ASST,
          f"every turn starts BOS, got {b}")


def test_session_isolated_turns():
    print("=== ChatSession isolated turns ===")
    m = tiny().eval()
    s = session(m)
    u1, a = s.send("hi", max_tokens=6)
    check(a["role"] == "assistant" and a["n"] == 6, f"capped at 6, stop={a['stop']}")
    check(a["stop"] == "limit", "no END within cap")
    check(u1["tokens"][0]["id"] == BOS and u1["tokens"][1]["id"] == USR,
          "first message tokens start BOS USR")
    check(s.ids[0] == BOS and s.ids[1] == USR, "context starts BOS USR")
    ids1 = list(s.ids)
    u2, a2 = s.send("again", max_tokens=4)
    again = encode("again")
    want = [BOS, USR, *again, ASST]
    check(u2["tokens"][0]["id"] == BOS, "second message also has BOS")
    check(s.ids[:len(want)] == want,
          f"context is only this turn, got {s.ids[:len(want)]}")
    check(len(s.ids) == len(want) + 4, "prefix + 4 generated, no history")
    check(ids1[:len(want)] != want, "first turn was a different prompt")
    check(s._cache_len() <= len(s.ids), "KV is not longer than this turn")
    check(len(s.messages) == 4, "session keeps 2 user + 2 assistant")
    check(len(a2["outputs"]) == a2["n"], "probe row per generated token")
    s.close()


def test_specials_from_cfg():
    print("=== specials_from_cfg ===")
    s = specials_from_cfg({})
    check(s["usr_id"] == 201 and s["end_id"] == 203, "defaults")
    s = specials_from_cfg({"bos_id": 1, "usr_id": 9, "assistant_id": 8,
                           "end_id": 7, "data_dir": "/tmp/x"})
    check(s["bos_id"] == 1 and s["data_dir"] == "/tmp/x", "cfg wins")


def test_probe_choice_marks_sampled():
    print("=== probe_choice ===")
    dec = load_decoder(None, BOS, USR, ASST, END)
    logits = torch.zeros(32)
    logits[5] = 4.0
    logits[7] = 1.0
    rec = probe_choice(logits, dec, chosen_id=5)
    check(rec["hit"] and rec["pred"]["id"] == 5, "greedy hit")
    rec = probe_choice(logits, dec, chosen_id=7)
    check(not rec["hit"] and rec["expected"]["id"] == 7, "sampled != argmax")
    check(any(o["expected"] and o["id"] == 7 for o in rec["opts"]),
          "sampled present in opts")


def tiny():
    return Transformer(name="chat_tiny", vocab_size=256, width=64,
                       embeddings=32, blocks=2, headsize=16, no_biases=True,
                       custom_init=True, dropout_p=0.0)


def session(m, use_cache=True):
    dec = load_decoder(None, BOS, USR, ASST, END)
    dec._hf = FakeTok()
    return ChatSession(m, dec, BOS, USR, ASST, END, temperature=0.0,
                       use_cache=use_cache)


def test_session_end_token():
    print("=== ChatSession stop on END ===")
    m = tiny().eval()
    m.get_next_token_probs = lambda logits, temperature=None, top_p=None: END
    s = session(m)
    _u, a = s.send("x", max_tokens=20)
    check(a["stop"] == "end" and a["n"] == 1, f"stop=end n=1, got {a['stop']} n={a['n']}")
    check(a["tokens"][0]["id"] == END, "generated END")
    s.close()


def test_cache_aligned_after_end():
    print("=== cache length after END ===")
    m = tiny().eval()
    m.get_next_token_probs = lambda logits, temperature=None, top_p=None: END
    s = session(m)
    s.send("x", max_tokens=8)
    check(s._cache_len() == len(s.ids),
          f"cache {s._cache_len()} == ids {len(s.ids)}")
    s.send("y", max_tokens=8)
    check(s._cache_len() == len(s.ids), "cache still aligned after turn 2")
    s.close()


def test_per_send_sampling():
    print("=== per-send temperature / top_p ===")
    m = tiny().eval()
    seen = []
    m.get_next_token_probs = lambda logits, temperature=None, top_p=None: (
        seen.append((temperature, top_p)) or END)
    s = session(m)
    s.send("x", max_tokens=4, temperature=0.7, top_p=0.9)
    check(seen and seen[0] == (0.7, 0.9), f"first send {seen[:1]}")
    s.send("y", max_tokens=4, temperature=0.0, top_p=None)
    check(seen[-1] == (0.0, None), f"cleared top_p {seen[-1]}")
    s.close()


def test_use_cache_off_matches_on():
    print("=== use_cache off matches on ===")
    m = tiny().eval()
    s1 = session(m)
    _, a1 = s1.send("hi", max_tokens=8)
    ids1 = list(s1.ids)
    s1.close()
    s2 = session(m, use_cache=False)
    _, a2 = s2.send("hi", max_tokens=8)
    check(s2.cache is None, "no KV stored")
    check(s2._cache_len() == 0, "cache length 0")
    check(ids1 == list(s2.ids), f"same ids {ids1} vs {s2.ids}")
    check(a1["n"] == a2["n"], "same generated length")
    s2.close()
    s3 = session(m)
    _, a3 = s3.send("hi", max_tokens=8, use_cache=False)
    check(a3["n"] == a1["n"], "per-send use_cache=False")
    s3.close()


def main():
    test_format()
    test_specials_from_cfg()
    test_probe_choice_marks_sampled()
    test_session_isolated_turns()
    test_session_end_token()
    test_cache_aligned_after_end()
    test_per_send_sampling()
    test_use_cache_off_matches_on()
    print("all chat tests passed")


if __name__ == "__main__":
    main()
