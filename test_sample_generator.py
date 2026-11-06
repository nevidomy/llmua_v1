"""Sample generation: KV cache option + prompt/generated token split.

Run: python -m test_sample_generator
"""
import torch

from model.transformer import Transformer
from sample_generator import generate_sample, generate_text
from sft.probe import TokenDecoder


def check(cond, label):
    assert cond, label
    print(f"  ok: {label}")


def tiny(T=32):
    return Transformer(
        name="sample_tiny", vocab_size=128, width=T, embeddings=32, blocks=2,
        headsize=16, no_biases=True, custom_init=True, dropout_p=0.0)


def decoder():
    return TokenDecoder(None, {0: "<BOS>"})


def test_roles_and_counts():
    print("=== prompt vs generated roles ===")
    torch.manual_seed(0)
    m = tiny().eval()
    prompt = [3, 5, 9]
    rec = generate_sample(m, 6, prompt_ids=prompt, temperature=0.0,
                          use_kv_cache=False, decoder=decoder())
    want = [0, 3, 5, 9]
    check(rec["n_prompt"] == 4, f"n_prompt 4 (BOS+3), got {rec['n_prompt']}")
    check(rec["n_generated"] == 6, f"n_generated 6, got {rec['n_generated']}")
    check(len(rec["tokens"]) == 10, "4 prompt + 6 generated ids")
    check(all(t["role"] == "prompt" for t in rec["tokens"][:4]),
          "prefix role=prompt")
    check(all(t["role"] == "generated" for t in rec["tokens"][4:]),
          "suffix role=generated")
    check([t["id"] for t in rec["tokens"][:4]] == want,
          "BOS prepended before encoded prompt")
    check(rec["tokens"][0]["spec"] == "bos", "first prompt token is BOS")
    check(rec["use_kv_cache"] is False, "flag stored")
    # already starts with BOS: do not double-prepend
    rec2 = generate_sample(m, 2, prompt_ids=[0, 7], temperature=0.0,
                           use_kv_cache=False, decoder=decoder())
    check([t["id"] for t in rec2["tokens"][:2]] == [0, 7],
          "existing leading BOS is not duplicated")


def test_unconditional_all_generated():
    print("=== unconditional: no prompt tokens ===")
    torch.manual_seed(1)
    m = tiny().eval()
    rec = generate_sample(m, 5, temperature=0.0, use_kv_cache=True,
                          decoder=decoder())
    check(rec["n_prompt"] == 0, "no prompt")
    check(rec["n_generated"] == 5, f"5 generated, got {rec['n_generated']}")
    check(all(t["role"] == "generated" for t in rec["tokens"]),
          "every token is generated")


def test_kv_cache_greedy_match():
    print("=== KV cache greedy matches full forward ===")
    torch.manual_seed(2)
    m = tiny(T=48).eval()
    prompt = [0, 3, 5, 9, 11]
    a = generate_sample(m, 12, prompt_ids=prompt, temperature=0.0,
                        use_kv_cache=False, decoder=decoder())
    torch.manual_seed(2)
    b = generate_sample(m, 12, prompt_ids=prompt, temperature=0.0,
                        use_kv_cache=True, decoder=decoder())
    check([t["id"] for t in a["tokens"]] == [t["id"] for t in b["tokens"]],
          "greedy ids match with and without cache")
    check(b["use_kv_cache"] is True, "cache path flagged on")
    text = generate_text(m, 4, temperature=0.0, use_kv_cache=True,
                         decoder=decoder())
    check(isinstance(text, str) and text, "generate_text still returns a string")


def test_bos_piece():
    print("=== specials labelled ===")
    torch.manual_seed(0)
    m = tiny().eval()
    rec = generate_sample(m, 2, prompt_ids=[4], temperature=0.0,
                          use_kv_cache=True, decoder=decoder())
    check(rec["tokens"][0]["t"] == "<BOS>" and rec["tokens"][0]["spec"] == "bos",
          "BOS labelled in the prompt")
    check(rec["tokens"][1]["id"] == 4, "encoded prompt follows BOS")


def main():
    test_roles_and_counts()
    test_unconditional_all_generated()
    test_kv_cache_greedy_match()
    test_bos_piece()
    print("SAMPLE GENERATOR PASS")


if __name__ == "__main__":
    main()
