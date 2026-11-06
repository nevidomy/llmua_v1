import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model.ff_width import resolve_ff_width

try:
    from torch.nn.attention.flex_attention import (
        flex_attention, create_block_mask)
    _flex = torch.compile(flex_attention)
    _make_block_mask = torch.compile(create_block_mask)   # once per (B, T)
    _HAS_FLEX = True
except ImportError:                                       # torch < 2.5
    _flex = _make_block_mask = None
    _HAS_FLEX = False


def doc_block_mask(token_ids, bos_id):
    """Causal + same-document BlockMask. Segment id is BOS-to-BOS cumsum.

    Built outside the compiled model: mask_mod walks the (T, T) index grid.
    """
    if not _HAS_FLEX:
        raise RuntimeError("doc_masking requires torch.nn.attention.flex_attention")
    B, T = token_ids.shape
    seg = (token_ids == bos_id).cumsum(-1)
    def mask_mod(b, h, q_idx, kv_idx):
        return (seg[b, q_idx] == seg[b, kv_idx]) & (q_idx >= kv_idx)
    return _make_block_mask(mask_mod, B=B, H=None, Q_LEN=T, KV_LEN=T,
                            device=token_ids.device)


def _doc_positions(token_ids, bos_id):
    """Per-token offset from the last BOS in the window (0 at BOS).
    Tokens before the first in-window BOS are measured from position 0."""
    T = token_ids.shape[-1]
    ar = torch.arange(T, device=token_ids.device)
    marks = torch.where(token_ids == bos_id, ar, torch.full_like(ar, -1))
    last = torch.cummax(marks, dim=-1).values.clamp_(min=0)
    return ar - last

class MultiheadAttention(nn.Module):
    def __init__(self, width, embeddings, headsize, no_biases, qk_norm, dropout_p=0.0):
        super().__init__()
        assert embeddings % headsize == 0, "Embeddings should be multiple of headsize"
        self.headsize = headsize
        self.headcount = embeddings // headsize
        self.dropout_p = dropout_p
        self.qkv = nn.Linear(embeddings, 3*embeddings, bias=False)
        self.proj = nn.Linear(embeddings, embeddings, bias=(not no_biases))
       
        self.q_norm = nn.RMSNorm(headsize) if qk_norm else None
        self.k_norm = nn.RMSNorm(headsize) if qk_norm else None

        # RoPE  
        ang = torch.arange(0, width).view(-1,1) * torch.tensor([10000.0**(-2*i/headsize) for i in range(headsize//2)])
        self.register_buffer("rope_cos", torch.cos(ang), persistent=False)
        self.register_buffer("rope_sin", torch.sin(ang), persistent=False)

    def rope(self, x, pos=None):                 # x: [B, H, T, W]
        if pos is None:
            N = x.shape[-2]
            cos, sin = self.rope_cos[:N], self.rope_sin[:N]        # [T, W//2]
        else:
            cos = self.rope_cos[pos].unsqueeze(1)                  # [B, 1, T, W//2]
            sin = self.rope_sin[pos].unsqueeze(1)
        xx, yy = x.chunk(2, -1)
        return torch.cat([xx*cos - yy*sin, xx*sin + yy*cos], -1).type_as(x)

    def forward(self, x, return_logits=False, attention_mask=None, block_mask=None,
                rope_pos=None, kv_cache=None, use_cache=False):
        B,T,E = x.shape
        H,W = self.headcount, self.headsize
        q,k,v = self.qkv(x).view(B,T,3,H,W).permute(2,0,3,1,4) # B,H,T,W

        # per-HEAD normalisation, over headsize -- what ViT-22B and Chameleon do
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q, k = self.rope(q, rope_pos), self.rope(k, rope_pos)

        if kv_cache is not None:
            pk, pv = kv_cache
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)

        if return_logits: # diagnostic only
            return (q @ k.transpose(-2, -1)) * W ** -0.5


        #a = (q @ k.transpose(-2,-1)) * W**-0.5 # B,H,T,T
        #a = torch.masked_fill(a, ~self.causal_mask[:T,:T], float("-inf"))
        #a = F.softmax(a, dim=-1)
        #a = self.drop(a)
        #a = (a @ v)
        drop = self.dropout_p if self.training else 0.0
        if block_mask is not None:
            assert not (self.training and self.dropout_p > 0), \
                "flex_attention has no dropout; set dropout_p=0"
            a = _flex(q, k, v, block_mask=block_mask)
        elif attention_mask is None:
            q_len, k_len = q.size(-2), k.size(-2)
            if q_len == k_len:
                # identical to the pre-cache path (training, full prefill)
                a = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                                   dropout_p=drop)
            else:
                # decode / incremental prefill: queries are the newest
                # tokens, causal aligned to the right edge of K
                start = k_len - q_len
                q_idx = torch.arange(q_len, device=q.device)[:, None]
                k_idx = torch.arange(k_len, device=q.device)[None, :]
                allow = k_idx <= (start + q_idx)
                a = F.scaled_dot_product_attention(q, k, v, attn_mask=allow,
                                                   dropout_p=drop)
        else:
            # SDPA bool: True = attend. Opposite of nn.MultiheadAttention.
            a = F.scaled_dot_product_attention(q, k, v, dropout_p=drop,
                                               attn_mask=attention_mask)
        a = a.transpose(-3,-2).reshape(B,T,E)
        out = self.proj(a)
        if use_cache:
            return out, (k, v)
        return out

class FeedForward(nn.Module):
    def __init__(self, embeddings, ff_width, no_biases, use_swiglu):
        super().__init__()
        self.use_swiglu = use_swiglu
        if use_swiglu:
            # gate and up fused into one matmul
            self.lin1 = nn.Linear(embeddings, 2 * ff_width, bias=(not no_biases))
        else:
            self.lin1 = nn.Linear(embeddings, ff_width, bias=(not no_biases))
            self.act = nn.GELU()
        self.lin2 = nn.Linear(ff_width, embeddings, bias=(not no_biases))
 
    def forward(self, x):
        if self.use_swiglu:
            gate, up = self.lin1(x).chunk(2, dim=-1)
            x = F.silu(gate) * up
        else:
            x = self.act(self.lin1(x))
        return self.lin2(x)

class AttentionBlock(nn.Module):
    def __init__(self, width, embeddings, ff_width, headsize, no_biases, use_swiglu, rms_norm, qk_norm, dropout_p=0.0):
        super().__init__()
        self.mha = MultiheadAttention(
            width=width, 
            embeddings=embeddings, 
            headsize=headsize, 
            no_biases=no_biases,
            qk_norm=qk_norm,
            dropout_p=dropout_p)
        self.ff = FeedForward(
            embeddings=embeddings,
            ff_width=ff_width,
            no_biases=no_biases,
            use_swiglu=use_swiglu)
        self.layn1 = nn.RMSNorm(embeddings) if rms_norm else nn.LayerNorm(embeddings, bias=(not no_biases))
        self.layn2 = nn.RMSNorm(embeddings) if rms_norm else nn.LayerNorm(embeddings, bias=(not no_biases))
        self.drop = nn.Dropout(dropout_p)

    def forward(self, x, attention_mask=None, block_mask=None, rope_pos=None,
                kv_cache=None, use_cache=False):
        n1 = self.layn1(x)
        if use_cache:
            a, new_c = self.mha(n1, attention_mask=attention_mask,
                                block_mask=block_mask, rope_pos=rope_pos,
                                kv_cache=kv_cache, use_cache=True)
        else:
            a = self.mha(n1, attention_mask=attention_mask,
                         block_mask=block_mask, rope_pos=rope_pos)
            new_c = None
        x = x + self.drop(a)
        x = x + self.drop(self.ff(self.layn2(x)))
        if use_cache:
            return x, new_c
        return x

class Transformer(nn.Module):
    def __init__(self, 
            name, vocab_size, width, 
            embeddings,
            blocks, 
            headsize, 
            ff_width=None,
            ff_expansion=4.0,
            dropout_p=0.0,
            no_biases=False,
            custom_init=False,
            residual_init=False,
            use_swiglu=False,
            rms_norm=False,
            qk_norm=False,
            description="",
            bos_id=0,
            doc_masking=False,
            doc_rope=False,
            _meta_probe=False):
        super().__init__()
        if _meta_probe:
            # cheap stand-in for param counting / metadata: no allocation,
            # no init. Modules and params are absent -- architecture fields
            # are all that exist. Used for metadata lookups.
            self.name, self.description = name, description
            self.vocab_size, self.context_width = vocab_size, width
            self.arch = dict(name=name, vocab_size=vocab_size, width=width,
                             embeddings=embeddings, blocks=blocks,
                             headsize=headsize, ff_width=ff_width,
                             ff_expansion=ff_expansion, dropout_p=dropout_p,
                             no_biases=no_biases, custom_init=custom_init,
                             residual_init=residual_init, use_swiglu=use_swiglu,
                             rms_norm=rms_norm, qk_norm=qk_norm,
                             bos_id=bos_id, doc_masking=doc_masking,
                             doc_rope=doc_rope, description=description)
            self._probe_params = self._count_params(embeddings, blocks)
            return
        self.bos_id = bos_id
        self.doc_masking = doc_masking
        self.doc_rope = doc_rope
        if doc_masking and not _HAS_FLEX:
            raise RuntimeError(
                "doc_masking requires torch.nn.attention.flex_attention")
        self.name = name
        self.description = description
        self.vocab_size = vocab_size
        self.context_width = width
        self.embed = nn.Embedding(vocab_size, embeddings)
        self.drop = nn.Dropout(dropout_p)
        self.ff_width = resolve_ff_width(embeddings, ff_width, use_swiglu, ff_expansion)
        # ModuleList, not a Python list: otherwise blocks are not registered
        # (state_dict / parameters / modules skip them). Not Sequential,
        # because Sequential.forward cannot take attention_mask.
        self.blocks = nn.ModuleList([
            AttentionBlock(
                    width=width,
                    embeddings=embeddings,
                    ff_width=self.ff_width,
                    headsize=headsize,
                    no_biases=no_biases,
                    use_swiglu=use_swiglu,
                    rms_norm=rms_norm,
                    qk_norm=qk_norm,
                    dropout_p=dropout_p)
                for _ in range(blocks)])
        self.layn = nn.RMSNorm(embeddings) if rms_norm else nn.LayerNorm(embeddings, bias=(not no_biases))
        self.lmh = nn.Linear(embeddings, vocab_size, bias=False)
        self.embed.weight = self.lmh.weight

        # full construction record: Transformer(**model.arch) is an exact
        # clone; saved into checkpoints so loaders never guess architecture
        self.arch = dict(name=name, vocab_size=vocab_size, width=width,
                         embeddings=embeddings, blocks=blocks,
                         headsize=headsize, ff_width=self.ff_width,
                         ff_expansion=ff_expansion, dropout_p=dropout_p,
                         no_biases=no_biases, custom_init=custom_init,
                         residual_init=residual_init,
                         use_swiglu=use_swiglu, rms_norm=rms_norm,
                         qk_norm=qk_norm, bos_id=bos_id,
                         doc_masking=doc_masking, doc_rope=doc_rope,
                         description=description)

        # params initialization
        if custom_init:
            for mod in self.modules():
                if isinstance(mod, (nn.Linear, nn.Embedding)):
                    fan_in = mod.weight.shape[-1]
                    nn.init.normal_(mod.weight, 0.0, 0.55 / math.sqrt(fan_in))
                    if getattr(mod, "bias", None) is not None:
                        nn.init.zeros_(mod.bias)

        if residual_init:
            for p_name, p in self.named_parameters(): 
                if p_name.endswith("proj.weight") or p_name.endswith("lin2.weight"):
                    p.data.mul_(1.0 / math.sqrt(2 * blocks))



    def forward(self, x, y=None, reduce=True, attention_mask=None,
                block_mask=None, doc_rope=None, use_cache=False, kv_cache=None):
        B, T = x.shape
        assert T <= self.context_width, "Input length must be less than or equal to context width"
        if (self.doc_masking and block_mask is None and attention_mask is None
                and not use_cache):
            raise RuntimeError(
                "doc_masking=True but no block_mask supplied -- "
                "build it with doc_block_mask(x, bos_id) in the caller")
        if doc_rope is None:
            doc_rope = self.doc_rope
        past = 0
        if use_cache and kv_cache is not None:
            past = kv_cache[0][0].shape[-2]
            if past + T > self.context_width:
                raise RuntimeError(
                    f"KV cache length {past}+{T} exceeds context "
                    f"width {self.context_width}")
        if doc_rope:
            if past == 0:
                rope_pos = _doc_positions(x, self.bos_id)
            else:
                # absolute positions continuing the cached prefix; chat
                # never inserts a new BOS so this matches _doc_positions
                # on the full sequence
                rope_pos = torch.arange(past, past + T, device=x.device)
                rope_pos = rope_pos.unsqueeze(0).expand(B, T)
        elif past == 0:
            rope_pos = None
        else:
            rope_pos = torch.arange(past, past + T, device=x.device)
            rope_pos = rope_pos.unsqueeze(0).expand(B, T)
        h = self.drop(self.embed(x))
        # attention_mask: optional bool, True = allowed to attend (SDPA
        # convention). When None, is_causal=True — same as before.
        # block_mask is built by the caller (trainer) so create_block_mask
        # stays outside the compiled graph.
        new_cache = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            if use_cache:
                layer_c = None if kv_cache is None else kv_cache[i]
                h, layer_c = block(h, attention_mask=attention_mask,
                                   block_mask=block_mask, rope_pos=rope_pos,
                                   kv_cache=layer_c, use_cache=True)
                new_cache.append(layer_c)
            else:
                h = block(h, attention_mask=attention_mask,
                          block_mask=block_mask, rope_pos=rope_pos)
        logits = self.lmh(self.layn(h))
        if y is None:
            return (logits, new_cache) if use_cache else logits
        loss = F.cross_entropy(logits.view(-1, self.vocab_size), y.view(-1), reduction="none")
        loss = (loss.view(B, T).mean() if reduce else loss.view(B, T))
        if use_cache:
            return logits, loss, new_cache
        return logits, loss

    @torch.no_grad()
    def generate(self, limit, prompt=[], progress_cb=None, temperature=1.0,
                 top_p=None, use_kv_cache=False, stop_id=None):
        was_training = self.training
        self.eval()
        device = self.lmh.weight.device
        if prompt:
            out = list(prompt)
        else:
            out = [0]
        cache, cached_len = None, 0
        for i in range(limit):
            window = out[-self.context_width:]
            if not use_kv_cache:
                x = torch.tensor(window, device=device).view(1, -1)
                bm = doc_block_mask(x, self.bos_id) if self.doc_masking else None
                logits = self(x, block_mask=bm)[0, -1, :].float()
            elif cache is None or cached_len != len(window) - 1:
                x = torch.tensor(window, device=device).view(1, -1)
                logits, cache = self(x, use_cache=True)
                logits = logits[0, -1, :].float()
                cached_len = len(window)
            else:
                x = torch.tensor([[window[-1]]], device=device)
                logits, cache = self(x, kv_cache=cache, use_cache=True)
                logits = logits[0, -1, :].float()
                cached_len = len(window)
            nid = int(self.get_next_token_probs(logits, temperature, top_p))
            out.append(nid)
            if progress_cb:
                progress_cb(i + 1, limit)
            if stop_id is not None and nid == stop_id:
                break
        self.train(was_training)
        return out if prompt else out[1:]

    def get_next_token_probs(self, logits, temperature=None, top_p=None):
        if temperature is not None:
            # support zero temperature, with greedy sampling
            if temperature < 1e-8:
                return torch.argmax(logits, dim=-1).item()
            logits = logits / temperature

        probabilities = F.softmax(logits, dim=-1)
        # use top-p sampling
        if top_p is not None:
            sorted_probabilities, sorted_indices = torch.sort(probabilities, descending=True)
            cumulative_probs = torch.cumsum(sorted_probabilities, dim=-1)
            sorted_probabilities[(cumulative_probs - sorted_probabilities) > top_p] = 0
            sorted_probabilities = sorted_probabilities / sorted_probabilities.sum()
            return sorted_indices[torch.multinomial(sorted_probabilities, 1)].item()
        # unmodified probability
        return torch.multinomial(probabilities, 1).item()

    def num_params(self):
        if hasattr(self, "_probe_params"):
            return self._probe_params[0]
        return sum(p.numel() for p in self.parameters())

    def num_params_no_embedings(self):
        if hasattr(self, "_probe_params"):
            return self._probe_params[1]
        return sum(p.numel() for p in self.parameters()) - self.embed.weight.numel()

    def _count_params(self, embeddings, blocks):
        """Analytic (total, non-embedding) param count, _meta_probe only.

        Mirrored against num_params() in tests; keep in sync with __init__.
        """
        a = self.arch
        V, E = a["vocab_size"], embeddings
        ffw = resolve_ff_width(E, a["ff_width"], a["use_swiglu"],
                               a["ff_expansion"])
        bias = lambda n: 0 if a["no_biases"] else n
        per_block = (
            3 * E * E                            # qkv (bias=False always)
            + E * E + bias(E)                    # proj
            + (2 * E * ffw if a["use_swiglu"]    # lin1 (gate+up) or plain
               else E * ffw) + bias(ffw)
            + ffw * E + bias(E)                  # lin2
            + 2 * E                              # layn1+layn2 weights
            + (0 if a["rms_norm"] else 2 * bias(E))  # LayerNorm biases
            + (2 * a["headsize"] if a["qk_norm"] else 0))  # per-head norms
        embed = V * E                            # tied: embed == lmh.weight
        total = embed + blocks * per_block + E + (0 if a["rms_norm"]
                                                  else bias(E))
        return total, total - embed

DEFAULT_CONTEXT_WIDTH = 2048
DEFAULT_HEADSIZE = 64
DEFAULT_VOCAB_SIZE = 32768

def model_test(context_width=DEFAULT_CONTEXT_WIDTH, vocab_size=DEFAULT_VOCAB_SIZE, headsize=DEFAULT_HEADSIZE,
                enable_init=True, enable_swiglu=False, enable_rms_norm=False, enable_qk_norm=False,
                residual_init=False, no_biases=False, custom_init=False,
                doc_masking=False, doc_rope=False, _meta_probe=False):
    assert vocab_size in (8192, 16384, 32768), "Unsupported vocab size"
    return Transformer(
        name="nano_llmua",
        description="Nano LLM UA Model. Testing model for warmup runs",
        vocab_size=vocab_size,
        width=context_width,
        embeddings=128,
        blocks=8,
        headsize=headsize,

        # testable params
        no_biases=enable_init or no_biases,
        custom_init=enable_init or custom_init,
        residual_init=enable_init or residual_init,
        use_swiglu=enable_swiglu,
        rms_norm=enable_rms_norm,
        qk_norm=enable_qk_norm,
        doc_masking=doc_masking, doc_rope=doc_rope, _meta_probe=_meta_probe)

def model_P1(context_width=DEFAULT_CONTEXT_WIDTH, vocab_size=DEFAULT_VOCAB_SIZE, headsize=DEFAULT_HEADSIZE,
            enable_init=True, enable_swiglu=False, enable_rms_norm=False, enable_qk_norm=False,
            residual_init=False, no_biases=False, custom_init=False,
            doc_masking=False, doc_rope=False, _meta_probe=False):
    assert vocab_size in (8192, 16384, 32768), "Unsupported vocab size"
    return Transformer(
        name="llmua_P1",
        description="LLMUA Model - Small.p1",
        vocab_size=vocab_size,
        width=context_width,
        embeddings=128,
        blocks=6,
        headsize=headsize,

        # testable params
        no_biases=enable_init or no_biases,
        custom_init=enable_init or custom_init,
        residual_init=enable_init or residual_init,
        use_swiglu=enable_swiglu,
        rms_norm=enable_rms_norm,
        qk_norm=enable_qk_norm,
        doc_masking=doc_masking, doc_rope=doc_rope, _meta_probe=_meta_probe)

def model_P2(context_width=DEFAULT_CONTEXT_WIDTH, vocab_size=DEFAULT_VOCAB_SIZE, headsize=DEFAULT_HEADSIZE,
            enable_init=True, enable_swiglu=False, enable_rms_norm=False, enable_qk_norm=False,
            residual_init=False, no_biases=False, custom_init=False,
            doc_masking=False, doc_rope=False, _meta_probe=False):
    assert vocab_size in (8192, 16384, 32768), "Unsupported vocab size"
    return Transformer(
        name="llmua_P2",
        description="LLMUA Model - Small.p2",
        vocab_size=vocab_size,
        width=context_width,
        embeddings=192,
        blocks=8,
        headsize=headsize,

        # testable params
        no_biases=enable_init or no_biases,
        custom_init=enable_init or custom_init,
        residual_init=enable_init or residual_init,
        use_swiglu=enable_swiglu,
        rms_norm=enable_rms_norm,
        qk_norm=enable_qk_norm,
        doc_masking=doc_masking, doc_rope=doc_rope, _meta_probe=_meta_probe)

def model_P3(context_width=DEFAULT_CONTEXT_WIDTH, vocab_size=DEFAULT_VOCAB_SIZE, headsize=DEFAULT_HEADSIZE,
            enable_init=True, enable_swiglu=False, enable_rms_norm=False, enable_qk_norm=False,
            residual_init=False, no_biases=False, custom_init=False,
            doc_masking=False, doc_rope=False, _meta_probe=False):
    assert vocab_size in (8192, 16384, 32768), "Unsupported vocab size"
    return Transformer(
        name="llmua_P3",
        description="LLMUA Model - Small.p3",
        vocab_size=vocab_size,
        width=context_width,
        embeddings=256,
        blocks=10,
        headsize=headsize,

        # testable params
        no_biases=enable_init or no_biases,
        custom_init=enable_init or custom_init,
        residual_init=enable_init or residual_init,
        use_swiglu=enable_swiglu,
        rms_norm=enable_rms_norm,
        qk_norm=enable_qk_norm,
        doc_masking=doc_masking, doc_rope=doc_rope, _meta_probe=_meta_probe)

def model_P4(context_width=DEFAULT_CONTEXT_WIDTH, vocab_size=DEFAULT_VOCAB_SIZE, headsize=DEFAULT_HEADSIZE,
            enable_init=True, enable_swiglu=False, enable_rms_norm=False, enable_qk_norm=False,
            residual_init=False, no_biases=False, custom_init=False,
            doc_masking=False, doc_rope=False, _meta_probe=False):
    assert vocab_size in (8192, 16384, 32768), "Unsupported vocab size"
    return Transformer(
        name="llmua_P4",
        description="LLMUA Model - Small.p4",
        vocab_size=vocab_size,
        width=context_width,
        embeddings=384,
        blocks=12,
        headsize=headsize,

        # testable params
        no_biases=enable_init or no_biases,
        custom_init=enable_init or custom_init,
        residual_init=enable_init or residual_init,
        use_swiglu=enable_swiglu,
        rms_norm=enable_rms_norm,
        qk_norm=enable_qk_norm,
        doc_masking=doc_masking, doc_rope=doc_rope, _meta_probe=_meta_probe)

def model_target(context_width=DEFAULT_CONTEXT_WIDTH, vocab_size=DEFAULT_VOCAB_SIZE, headsize=DEFAULT_HEADSIZE, 
                enable_init=True, enable_swiglu=False, enable_rms_norm=False, enable_qk_norm=False, residual_init=False, no_biases=False, custom_init=False,
                doc_masking=False, doc_rope=False, _meta_probe=False):
    assert vocab_size in (8192, 16384, 32768), "Unsupported vocab size"
    return Transformer(
        name="llmua",
        description="LLMUA Model Final model.",
        vocab_size=vocab_size,
        width=context_width,
        embeddings=768,
        blocks=14,
        headsize=headsize,

        # testable params
        no_biases=enable_init or no_biases,
        custom_init=enable_init or custom_init,
        residual_init=enable_init or residual_init,
        use_swiglu=enable_swiglu,
        rms_norm=enable_rms_norm,
        qk_norm=enable_qk_norm,
        doc_masking=doc_masking, doc_rope=doc_rope, _meta_probe=_meta_probe)