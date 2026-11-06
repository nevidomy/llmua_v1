"""One bf16-autocast forward of the native Transformer, as in training/eval; for every nn.RMSNorm record
input dtype, row mean-square stats, and which eps (fp32 eps 1.19e-7 vs bf16 eps 2^-7) its output matches.
usage: python eps_probe.py <ckpt> <val.bin>"""
import sys, os, collections, numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from model.transformer import Transformer
from model.packed_val import packed_doc_attention_mask
ck_path, val_bin = sys.argv[1], sys.argv[2]
ck = torch.load(ck_path, map_location='cpu', weights_only=False)
m = Transformer(**ck['model_arch']); m.load_state_dict(ck['model']); m = m.cuda().eval()
m.doc_masking = False                                   # SDPA + explicit doc mask (same semantics as the BlockMask)
T = m.context_width
data = np.memmap(val_bin, dtype=np.uint16, mode='r')
x = torch.from_numpy(np.stack([data[i*T:(i+1)*T] for i in range(4)]).astype(np.int64)).cuda()
EPS_F32, EPS_BF16 = torch.finfo(torch.float32).eps, torch.finfo(torch.bfloat16).eps
rec = collections.OrderedDict()
def hook(name):
    def f(mod, inp, out):
        xi = inp[0].detach().double(); w = mod.weight.detach().double()
        ms = xi.pow(2).mean(-1, keepdim=True)
        ra = xi * torch.rsqrt(ms + EPS_F32) * w; rb = xi * torch.rsqrt(ms + EPS_BF16) * w
        o = out.detach().double(); den = o.abs().mean()
        rec[name] = dict(in_dt=str(inp[0].dtype).replace('torch.', ''), out_dt=str(out.dtype).replace('torch.', ''),
                         ms_p1=torch.quantile(ms.flatten().float()[::97], 0.01).item(), ms_med=ms.median().item(),
                         err_f32=((o - ra).abs().mean() / den).item(), err_bf16=((o - rb).abs().mean() / den).item(),
                         sens=((ra - rb).abs().mean() / den).item())
    return f
for n, mod in m.named_modules():
    if isinstance(mod, nn.RMSNorm):
        assert mod.eps is None, (n, mod.eps)             # eps=None on every norm
        mod.register_forward_hook(hook(n))
with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
    m(x, attention_mask=packed_doc_attention_mask(x, m.bos_id))
print(f"{ck_path.split('/')[-2][:50]}  eps_f32={EPS_F32:.3e} eps_bf16={EPS_BF16:.3e}")
print(f"{'norm':26} {'in':9} {'out':9} {'ms p1':>9} {'ms med':>9} {'err@f32':>9} {'err@bf16':>9} {'sens(f32 vs bf16)':>18}")
groups = collections.defaultdict(list)
for n, r in rec.items():
    kind = n.split('.')[-1] if 'blocks' in n else n
    groups[kind].append((n, r))
    if n.startswith('blocks.0.') or n.startswith('blocks.7.') or 'blocks' not in n:
        print(f"{n:26} {r['in_dt']:9} {r['out_dt']:9} {r['ms_p1']:9.2e} {r['ms_med']:9.2e} {r['err_f32']:9.2e} {r['err_bf16']:9.2e} {r['sens']:18.2e}")
print('\nsummary over all layers:')
for kind, lst in groups.items():
    dts = sorted({r['in_dt'] for _, r in lst})
    print(f"  {kind:8} n={len(lst):2} input dtype(s)={dts}  matches eps_f32 in {sum(r['err_f32'] < r['err_bf16'] for _, r in lst)}/{len(lst)} "
          f"| max sensitivity {max(r['sens'] for _, r in lst):.2e} | min ms_p1 {min(r['ms_p1'] for _, r in lst):.2e}")
