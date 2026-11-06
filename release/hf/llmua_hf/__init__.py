"""LLMUA for Hugging Face transformers. `import llmua_hf; llmua_hf.register()` makes
`AutoModelForCausalLM.from_pretrained(path)` work without trust_remote_code."""
from .configuration_llmua import LlmuaConfig
from .modeling_llmua import LlmuaForCausalLM, LlmuaModel, LlmuaPreTrainedModel

__all__ = ["LlmuaConfig", "LlmuaModel", "LlmuaForCausalLM", "LlmuaPreTrainedModel", "register"]


def register(exist_ok: bool = True):
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
    AutoConfig.register("llmua", LlmuaConfig, exist_ok=exist_ok)
    AutoModel.register(LlmuaConfig, LlmuaModel, exist_ok=exist_ok)
    AutoModelForCausalLM.register(LlmuaConfig, LlmuaForCausalLM, exist_ok=exist_ok)
