#!/usr/bin/env python3
"""Repair large.pt whose first N bytes were overwritten by an nginx 404 page (wget -c resumed from byte N).

Writes nothing unless (a) the surviving bytes continue the reconstructed pickle exactly and
(b) CRC32 of reconstructed-prefix + surviving data equals the CRC in the zip's central directory
    (and in the data descriptor that follows the entry).
Run:  python fix_large_pt.py large.pt
"""
import struct, sys, zipfile, zlib

path = sys.argv[1] if len(sys.argv) > 1 else 'large.pt'
f = open(path, 'r+b')
head = f.read(4096)
i = head.find(b'</html>')
assert i != -1, 'no HTML prefix found — nothing to fix'
N = i + len(b'</html>')
while head[N:N+1] in (b'\r', b'\n'):
    N += 1
print(f'HTML prefix: {N} bytes')

z = zipfile.ZipFile(path)                                   # central directory is at the end, intact
infos = sorted(z.infolist(), key=lambda e: e.header_offset)
pkl, nxt = infos[0], infos[1]
name = pkl.filename.encode()
print(f'first entry {pkl.filename!r}  size {pkl.file_size}  crc {pkl.CRC:08x}  next header at {nxt.header_offset}')
assert pkl.header_offset == 0 and pkl.filename.endswith('data.pkl') and pkl.compress_type == 0

# torch pads so entry data starts on a 64-byte boundary: 30-byte header + name + 'FB' padding extra
extra_len = 64 - 30 - len(name)
assert 4 <= extra_len < 64, extra_len
data_start = 64

# data descriptor (PK\x07\x08 crc csize usize) sits right after the data; it's a second CRC oracle
f.seek(nxt.header_offset - 16)
sig, dcrc, dcs, dus = struct.unpack('<4sIII', f.read(16))
assert sig == b'PK\x07\x08', f'no data descriptor before next header: {sig!r}'
assert dcrc == pkl.CRC and dus == pkl.file_size, (hex(dcrc), dus)
assert nxt.header_offset - 16 == data_start + pkl.file_size, 'data.pkl is not laid out as expected — stop'
print(f'layout ok: header {data_start} B, data [{data_start}, {data_start + pkl.file_size}), descriptor, next entry')
assert N > data_start, 'HTML shorter than the local header?'

lost_pkl = N - data_start
f.seek(N)
surviving = f.read(data_start + pkl.file_size - N)          # true bytes [N, data_end)
print(f'lost pickle bytes: {lost_pkl}; survivor starts {surviving[:8]!r}')

def preamble(first_key: str) -> bytes:                       # protocol-2 pickle of {first_key: OrderedDict(...)}
    k = first_key.encode()
    return (b'\x80\x02' + b'}' + b'q\x00' + b'(' + b'X' + struct.pack('<I', len(k)) + k + b'q\x01'
            + b'ccollections\nOrderedDict\n' + b'q\x02' + b')' + b'R' + b'q\x03' + b'(')

key = b'_orig_mod.transformer.wte.weight'
key_op = b'X' + struct.pack('<I', len(key)) + key
found = None
for first_key in ['model', 'model_state_dict', 'state_dict', 'model_state', 'net']:
    pre = preamble(first_key) + key_op
    if len(pre) <= lost_pkl:
        continue
    tail = pre[lost_pkl:]                                    # what the survivor must begin with
    if surviving[:len(tail)] != tail:
        print(f'  {first_key!r}: survivor would need to start with {tail!r} — no')
        continue
    crc = zlib.crc32(pre[:lost_pkl] + surviving) & 0xFFFFFFFF
    ok = crc == pkl.CRC
    print(f'  {first_key!r}: crc {crc:08x} {"MATCH" if ok else "mismatch"}')
    if ok:
        found = pre[:lost_pkl]
        break
if found is None:
    f.close()
    sys.exit('no reconstruction matched the CRC — file untouched')

Y, M, D, h, m, s = pkl.date_time
dostime = (h << 11) | (m << 5) | (s // 2)
dosdate = ((Y - 1980) << 9) | (M << 5) | D
extra = b'FB' + struct.pack('<H', extra_len - 4) + b'Z' * (extra_len - 4)   # PyTorch's padding field
has_desc = bool(pkl.flag_bits & 0x08)
lcrc, lcs, lus = (0, 0, 0) if has_desc else (pkl.CRC, pkl.compress_size, pkl.file_size)
local = struct.pack('<4sHHHHHIIIHH', b'PK\x03\x04', pkl.extract_version, pkl.flag_bits, 0,
                    dostime, dosdate, lcrc, lcs, lus, len(name), len(extra)) + name + extra
assert len(local) == data_start, len(local)
f.seek(0)
f.write(local + found)
f.close()
print('prefix rewritten. verifying every entry CRC (walks the whole file, a few minutes) …')
bad = zipfile.ZipFile(path).testzip()
print('bad entry:', bad)
if bad is None:
    print("large.pt is byte-exact. next: python -c \"import torch; ck=torch.load('large.pt', map_location='cpu', weights_only=False); print(list(ck.keys()))\"")