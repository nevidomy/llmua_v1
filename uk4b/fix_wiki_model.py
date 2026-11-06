#!/usr/bin/env python3
"""Repair wiki.model (SentencePiece ModelProto) whose first N bytes were overwritten by an nginx 404 page.

The lost region holds the first few vocabulary entries (<unk>, <s>, </s>, control symbols …). Everything
needed to rebuild them — the symbol lists, ids, vocab_size — is in trainer_spec at the end of the file.
Writes nothing unless the rebuilt head is exactly N bytes and the repaired model loads with the expected size.
Run:  python fix_wiki_model.py wiki.model
"""
import struct, sys

path = sys.argv[1] if len(sys.argv) > 1 else 'wiki.model'
raw = open(path, 'rb').read()
i = raw.find(b'</html>')
assert i != -1, 'no HTML prefix found — nothing to fix'
N = i + len(b'</html>')
if raw[N:N+2] == b'\r\n':          # nginx page ends with exactly one CRLF; the next byte may itself be 0x0d
    N += 2
elif raw[N:N+1] == b'\n':
    N += 1
print(f'HTML prefix: {N} bytes; file {len(raw)} bytes')

# ---- minimal protobuf reader -------------------------------------------------------------------
def rvarint(b, p):
    x = s = 0
    while True:
        c = b[p]; p += 1
        x |= (c & 0x7F) << s; s += 7
        if not c & 0x80:
            return x, p

def fields(b):
    """yield (field_no, wire_type, value) over a message body"""
    p = 0
    while p < len(b):
        tag, p = rvarint(b, p)
        fno, wt = tag >> 3, tag & 7
        if wt == 0:
            v, p = rvarint(b, p)
        elif wt == 1:
            v, p = b[p:p+8], p + 8
        elif wt == 2:
            ln, p = rvarint(b, p); v, p = b[p:p+ln], p + ln
        elif wt == 5:
            v, p = b[p:p+4], p + 4
        else:
            raise ValueError(f'wire type {wt} at {p}')
        yield fno, wt, v

# ---- locate the first complete entry after the damage ------------------------------------------
# survivor begins inside the piece string of one entry:  0d '<|endofmeta|>' 15 <score> 18 <type>
assert raw[N] == 0x0d and raw[N+1:N+14] == b'<|endofmeta|>', raw[N:N+16]
partial_inner = raw[N:N+1+13+5+2]                        # len byte + string + score field + type field
assert partial_inner[14] == 0x15 and partial_inner[19] == 0x18, partial_inner
first_complete = N + len(partial_inner)
# reconstruct the 3 lost bytes of that entry: outer tag+len, inner piece tag
inner_full = b'\x0a' + partial_inner                     # tag(1) + len + string + score + type
endofmeta_entry = b'\x0a' + bytes([len(inner_full)]) + inner_full
lost_head_bytes = N - 3                                  # bytes before the <|endofmeta|> entry
print(f'<|endofmeta|> entry rebuilt ({len(endofmeta_entry)} B); {lost_head_bytes} B of earlier entries to rebuild')

# ---- parse the surviving top-level fields ------------------------------------------------------
tail = raw[first_complete:]
pieces_after = []
trainer = None
for fno, wt, v in fields(tail):
    if fno == 1:
        d = dict((f, val) for f, _, val in fields(v))
        pieces_after.append(d[1].decode('utf-8', 'replace'))
    elif fno == 2:
        trainer = v
assert trainer is not None, 'trainer_spec not found in the surviving tail'
print(f'surviving complete entries: {len(pieces_after)}   first: {pieces_after[:6]}   last: {pieces_after[-2:]}')

ts = {}
for fno, wt, v in fields(trainer):
    ts.setdefault(fno, []).append(v)
def s1(f, default=None): return ts[f][0].decode() if f in ts else default
def i1(f, default=None):
    v = ts[f][0] if f in ts else None
    if v is None: return default
    return v - (1 << 64) if v >= (1 << 63) else v          # int32 negatives are sign-extended varints
vocab_size = i1(4)
unk_id, bos_id, eos_id, pad_id = i1(40, 0), i1(41, 1), i1(42, 2), i1(43, -1)
unk_p, bos_p, eos_p, pad_p = s1(45, '<unk>'), s1(46, '<s>'), s1(47, '</s>'), s1(48, '<pad>')
control = [x.decode() for x in ts.get(30, [])]
userdef = [x.decode() for x in ts.get(31, [])]
print(f'trainer_spec: vocab_size {vocab_size}  ids unk/bos/eos/pad {unk_id}/{bos_id}/{eos_id}/{pad_id}')
print(f'  control_symbols ({len(control)}): {control}')
print(f'  user_defined_symbols ({len(userdef)}): {userdef[:10]}{" …" if len(userdef) > 10 else ""}')

# ---- reconstruct the leading entries -----------------------------------------------------------
UNKNOWN, CONTROL, USER = 2, 3, 4
slots = {}
if unk_id >= 0: slots[unk_id] = (unk_p, UNKNOWN)
if bos_id >= 0: slots[bos_id] = (bos_p, CONTROL)
if eos_id >= 0: slots[eos_id] = (eos_p, CONTROL)
if pad_id >= 0: slots[pad_id] = (pad_p, CONTROL)
nxt = 0
for sym, typ in [(c, CONTROL) for c in control] + [(u, USER) for u in userdef]:
    while nxt in slots: nxt += 1                       # control/user symbols take the lowest free ids
    slots[nxt] = (sym, typ)
ordered = [slots[k] for k in sorted(slots) if k < len(control) + len(userdef) + 4]
print(f'  id order: {[(k, slots[k][0]) for k in sorted(slots)][:8]} …')

n_survive = len(pieces_after) + 1                        # + the rebuilt <|endofmeta|>
n_lost = vocab_size - n_survive
print(f'entries lost before <|endofmeta|>: {n_lost}')
assert 0 < n_lost <= len(ordered), (n_lost, len(ordered))
assert ordered[n_lost][0] == '<|endofmeta|>', f'entry {n_lost} in trainer order is {ordered[n_lost]!r}, expected <|endofmeta|>'

def varint(x):
    out = bytearray()
    while True:
        c = x & 0x7F; x >>= 7
        out.append(c | (0x80 if x else 0))
        if not x: return bytes(out)

def entry(piece, typ):
    s = piece.encode()
    inner = b'\x0a' + varint(len(s)) + s + b'\x15' + struct.pack('<f', 0.0) + b'\x18' + varint(typ)
    return b'\x0a' + varint(len(inner)) + inner

head = b''.join(entry(p, t) for p, t in ordered[:n_lost])
print(f'rebuilt {n_lost} entries = {len(head)} B  (need {lost_head_bytes})   {[p for p, _ in ordered[:n_lost]]}')
assert len(head) == lost_head_bytes, 'byte count does not match — file untouched'

fixed = head + endofmeta_entry + tail
assert len(fixed) == len(raw), (len(fixed), len(raw))
open(path + '.fixed', 'wb').write(fixed)

# ---- verify by loading -------------------------------------------------------------------------
import sentencepiece as spm
sp = spm.SentencePieceProcessor(model_file=path + '.fixed')
print('loaded. piece_size', sp.get_piece_size(), ' id 50256 =', repr(sp.id_to_piece(50256)),
      ' unk/bos/eos/pad ids =', sp.unk_id(), sp.bos_id(), sp.eos_id(), sp.pad_id())
t = 'Сонячна система складається з восьми планет та Сонця.'
ids = sp.encode(t)
print(len(ids), 'tokens;', sp.decode(ids) == t, 'roundtrip;', sp.encode(t, out_type=str)[:8])
assert sp.get_piece_size() == vocab_size and sp.id_to_piece(50256) == eos_p and sp.decode(ids) == t
assert sp.id_to_piece(0) == pad_p and sp.id_to_piece(6) == '<|endofmeta|>'
import os; os.replace(path + '.fixed', path)
print(f'{path} repaired in place')
