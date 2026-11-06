"""KV cache must match full-window forward; default path unchanged.

Run: python -m model.test_kv_cache
"""
import torch

from model.transformer import Transformer


def check(cond, label):
    assert cond, label
    print(f"  ok: {label}")


def tiny(T=32, doc_rope=False, doc_masking=False):
    return Transformer(
        name="kv_tiny", vocab_size=128, width=T, embeddings=32, blocks=2,
        headsize=16, no_biases=True, custom_init=True, dropout_p=0.0,
        doc_rope=doc_rope, doc_masking=doc_masking)


def test_forward_without_cache_is_tensor():
    print("=== default forward return ===")
    m = tiny().eval()
    x = torch.randint(0, 128, (2, 8))
    out = m(x)
    check(torch.is_tensor(out) and out.shape == (2, 8, 128),
          f"logits tensor {tuple(out.shape)}")
    y = x.clone()
    logits, loss = m(x, y)
    check(torch.is_tensor(logits) and loss.ndim == 0, "loss path unchanged")


def test_prefill_matches_cached_decode():
    print("=== cached decode == full forward ===")
    torch.manual_seed(0)
    m = tiny().eval()
    ids = torch.randint(0, 128, (1, 12))
    full = m(ids)

    logits, cache = m(ids[:, :5], use_cache=True)
    check(torch.allclose(logits, full[:, :5], atol=1e-5, rtol=1e-4),
          "prefill logits match the prefix of a full forward")
    for t in range(5, 12):
        step, cache = m(ids[:, t:t + 1], kv_cache=cache, use_cache=True)
        check(torch.allclose(step, full[:, t:t + 1], atol=1e-5, rtol=1e-4),
              f"decode pos {t} matches full forward")


def test_incremental_prefill():
    print("=== incremental prefill of several tokens ===")
    torch.manual_seed(1)
    m = tiny().eval()
    ids = torch.randint(0, 128, (1, 10))
    full = m(ids)
    _, cache = m(ids[:, :4], use_cache=True)
    step, cache = m(ids[:, 4:10], kv_cache=cache, use_cache=True)
    check(torch.allclose(step, full[:, 4:10], atol=1e-5, rtol=1e-4),
          "6-token incremental prefill matches")


def test_generate_cache_greedy():
    print("=== generate greedy cache vs no cache ===")
    torch.manual_seed(2)
    m = tiny(T=48).eval()
    prompt = [0, 3, 5, 9, 11]
    a = m.generate(20, prompt=prompt, temperature=0.0, use_kv_cache=False)
    torch.manual_seed(2)
    b = m.generate(20, prompt=prompt, temperature=0.0, use_kv_cache=True)
    check(a == b, f"greedy ids match\n  {a}\n  {b}")


def test_generate_cache_doc_rope():
    print("=== generate greedy + doc_rope ===")
    torch.manual_seed(3)
    m = tiny(T=48, doc_rope=True).eval()
    prompt = [0, 7, 8, 2]
    a = m.generate(16, prompt=prompt, temperature=0.0, use_kv_cache=False)
    b = m.generate(16, prompt=prompt, temperature=0.0, use_kv_cache=True)
    check(a == b, "doc_rope greedy match")


def test_stop_id():
    print("=== generate stop_id ===")
    m = tiny().eval()
    first = m.generate(8, prompt=[0, 1], temperature=0.0, use_kv_cache=True)
    stop = first[len([0, 1])]
    stopped = m.generate(8, prompt=[0, 1], temperature=0.0, use_kv_cache=True,
                         stop_id=stop)
    check(len(stopped) == len([0, 1]) + 1,
          f"stop after one token, got {len(stopped)}")


def main():
    test_forward_without_cache_is_tensor()
    test_prefill_matches_cached_decode()
    test_incremental_prefill()
    test_generate_cache_greedy()
    test_generate_cache_doc_rope()
    test_stop_id()
    print("all kv-cache tests passed")


if __name__ == "__main__":
    main()
