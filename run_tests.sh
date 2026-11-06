#!/bin/bash
# Every test, as a module from the repo root. Running sft/test_*.py by path
# breaks: sft/tokenize.py would shadow the stdlib tokenize. CPU only; the GPU
# section of sft.test_sft_train skips itself without CUDA.
set -u
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
fail=0
for t in model.test_kv_cache model.test_muon model.test_packed_val \
         model.transformer_test evaluation.test_custom_full_val \
         sft.test_batcher sft.test_chat sft.test_probe sft.test_sft_train \
         test_rng_batches test_sample_generator; do
    if python -m "$t" > /dev/null 2>&1; then
        echo "ok    $t"
    else
        echo "FAIL  $t"
        fail=1
    fi
done
exit $fail
