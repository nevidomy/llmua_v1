"""Test the RNG batches for the evaluation set.

eval_batches is the total number of sequences. They are drawn in one
RNG call, then sliced into full-batch (micro_batch * count) forwards.
The windows are the same for a given eval_seed + total regardless of
how that total is sliced, as long as total % (micro * count) == 0.
"""
import numpy as np

def _batch(B, rng=None):
        rng = rng or np.random
        ix = rng.integers(0, 100, size=B) if hasattr(rng, "integers") \
            else rng.randint(0, 100, size=B)
        return ix

def test_make_eval_batches(eval_seed, micro_batch, eval_batches, count=1):
        B = micro_batch * count
        assert eval_batches % B == 0
        g = np.random.default_rng(eval_seed)
        ix = _batch(eval_batches, g)
        return [ix[i:i + B] for i in range(0, eval_batches, B)]

if __name__ == "__main__":
    A = np.concatenate(test_make_eval_batches(47, 16, 32, 1)).tolist()
    B = np.concatenate(test_make_eval_batches(47, 8, 32, 1)).tolist()
    C = np.concatenate(test_make_eval_batches(47, 16, 32, 2)).tolist()
    assert A == B == C, "The batches should be the same"
    try:
        test_make_eval_batches(47, 16, 32, 3)   # 32 % 48
        raise SystemExit("expected assert for 32 % (16*3)")
    except AssertionError:
        pass
    try:
        test_make_eval_batches(47, 16, 30, 1)
        raise SystemExit("expected assert for 30 % 16")
    except AssertionError:
        pass
    print("PASSED")
