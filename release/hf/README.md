# Hugging Face export

Tools to turn a training checkpoint (`ckpt_final.pt`) into a Hugging Face model folder. The model keeps
its own architecture, `model_type: "llmua"` (`LlmuaForCausalLM`). The weights are the trained tensors
under their training names: `embed`, `blocks.i.mha.qkv`, `ff.lin1` and so on. The fused QKV and gate/up
weights are not reshuffled, and the output layer is tied to the embedding.

The released weights, with model cards, are
[nevidomy/llmua-v1-124m](https://huggingface.co/nevidomy/llmua-v1-124m) and
[nevidomy/llmua-v1-124m-sft](https://huggingface.co/nevidomy/llmua-v1-124m-sft).

## Files

| file | what |
|---|---|
| `llmua_hf/` | `LlmuaConfig` and `LlmuaForCausalLM` for transformers (KV cache, padding masks, `generate`); installable with `pip install ./release/hf/llmua_hf` |
| `convert_to_hf.py` | checkpoint → HF folder: safetensors, config, tokenizer, generation config, and for fine-tuned checkpoints a chat template |
| `verify_hf.py` | checks against the native model: logits, full-validation bpc, greedy generation, tokenizer |
| `verify_llmua_extra.py` | batched generation with left padding, save/load round trip, loading through `register()` |
| `export_qwen3_compat.py` | optional export in the stock `Qwen3ForCausalLM` layout, for tools that only run known architectures (vLLM, llama.cpp). It re-splits the tensors, so it is not the release format |
| `eps_probe.py`, `eps_micro.py` | how the RMSNorm epsilon was determined (below) |

## Use

    python release/hf/convert_to_hf.py <run_dir>/ckpt_final.pt out/llmua-124m

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("out/llmua-124m")
model = AutoModelForCausalLM.from_pretrained("out/llmua-124m", trust_remote_code=True)
ids = tok("Київ — столиця", return_tensors="pt").input_ids     # the tokenizer prepends BOS
print(tok.decode(model.generate(ids, max_new_tokens=40)[0], skip_special_tokens=True))
```

After `pip install ./release/hf/llmua_hf`, call `llmua_hf.register()` and drop `trust_remote_code`.

## What was checked

These checks ran on the 124M target (80×), the fine-tuned model and a P1 rung:

- **Logits:** identical to the native model on the same attention path.
- **Full-validation bpc:** identical to the native model in fp32.
- **Greedy generation:** the fine-tuned model matches `sft/chat.py` token for token.
- **Batched generation:** left-padded batches equal per-prompt generation.
- **Round trip:** a save/load round trip is bit-identical.

## Two details

- **RMSNorm epsilon.** The model's norms are built with `nn.RMSNorm(eps=None)`. In the torch version used for
  training (2.11), every norm used the fp32 epsilon, 1.19e-7. That includes the QK-norms, which receive bf16
  inputs. The converter writes this value explicitly, so loading does not depend on the torch version. An
  epsilon of 2^-7 would cost about 2.5 bpc.
- **Fine-tuning specials.** The fine-tuning tokens `<|usr|>`, `<|assistant|>` and `<|end|>` reuse ids 201–203.
  In the base vocabulary those ids are the byte tokens for VT, FF and CR. The exported fine-tuned tokenizer
  renames them, so raw VT, FF and CR characters in text are dropped. Normalise `\r\n` to `\n` before
  tokenising.
