#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""直接读 ARW 自己的 TIFF/EXIF/MakerNote（零依赖手写解析），看机内 WB 到底写了什么。

同时列出 rawpy 的全部公开属性，确认它有没有把 EXIF/MakerNote 暴露出来。
"""
import struct
import sys

ARW = sys.argv[1] if len(sys.argv) > 1 else r'C:\Users\Administrator\Desktop\新建文件夹 (4)\DSC03492.ARW'
data = open(ARW, 'rb').read()
print(f'文件大小 {len(data)/1e6:.1f} MB   头 8 字节: {data[:8]!r}')

bo = '<' if data[:2] == b'II' else '>'
magic, ifd0_off = struct.unpack_from(bo + 'HI', data, 2)
print(f'字节序={"小端" if bo=="<" else "大端"}  magic={magic}  IFD0 offset={ifd0_off}')

TYPE_SIZE = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8}
TYPE_FMT = {1: 'B', 3: 'H', 4: 'I', 6: 'b', 8: 'h', 9: 'i'}


def read_ifd(off):
    if off <= 0 or off + 2 > len(data):
        return [], 0
    n = struct.unpack_from(bo + 'H', data, off)[0]
    out = []
    for i in range(n):
        e = off + 2 + i * 12
        if e + 12 > len(data):
            break
        tag, typ, cnt = struct.unpack_from(bo + 'HHI', data, e)
        sz = TYPE_SIZE.get(typ, 1) * cnt
        if sz <= 4:
            raw = data[e + 8:e + 8 + sz]
        else:
            voff = struct.unpack_from(bo + 'I', data, e + 8)[0]
            raw = data[voff:voff + sz] if voff + sz <= len(data) else b''
        out.append((tag, typ, cnt, raw))
    nxt = struct.unpack_from(bo + 'I', data, off + 2 + n * 12)[0] if off + 2 + n * 12 + 4 <= len(data) else 0
    return out, nxt


def as_text(raw):
    return raw.split(b'\x00')[0].decode('utf-8', 'replace')


def as_nums(tag, typ, cnt, raw):
    if typ in TYPE_FMT:
        try:
            return list(struct.unpack(bo + TYPE_FMT[typ] * cnt, raw[:TYPE_SIZE[typ] * cnt]))
        except Exception:
            return None
    if typ in (5, 10):                     # RATIONAL / SRATIONAL
        f = 'II' if typ == 5 else 'ii'
        vals = []
        for k in range(min(cnt, 6)):
            num, den = struct.unpack_from(bo + f, raw, k * 8)
            vals.append(round(num / den, 4) if den else None)
        return vals
    return None


NAMES = {0x010F: 'Make', 0x0110: 'Model', 0x0131: 'Software', 0x0132: 'DateTime',
         0x8769: 'ExifIFD', 0x927C: 'MakerNote', 0x8825: 'GPSIFD', 0xA403: 'WhiteBalance',
         0x9208: 'LightSource', 0x9207: 'MeteringMode', 0x8827: 'ISOSpeedRatings',
         0x829A: 'ExposureTime', 0x829D: 'FNumber', 0x920A: 'SubjectDistance'}

print('\n=== IFD0 ===')
ifd0, nxt = read_ifd(ifd0_off)
exif_off = mn_raw = None
for tag, typ, cnt, raw in ifd0:
    nm = NAMES.get(tag, f'0x{tag:04X}')
    if tag in (0x010F, 0x0110, 0x0131, 0x0132):
        print(f'  {nm:16} = {as_text(raw)!r}')
    elif tag == 0x8769:
        exif_off = struct.unpack_from(bo + 'I', raw)[0]
        print(f'  ExifIFD 偏移     = {exif_off}')
    elif tag == 0x927C:
        mn_raw = raw
        print(f'  MakerNote        = {len(raw)} 字节')
    elif tag in (0x9208, 0xA403):
        print(f'  {nm:16} = {as_nums(tag, typ, cnt, raw)}')
print('  IFD0 标签总数:', len(ifd0))

print('\n=== Exif IFD ===')
if exif_off:
    ex, _ = read_ifd(exif_off)
    for tag, typ, cnt, raw in ex:
        nm = NAMES.get(tag, f'0x{tag:04X}')
        if tag in NAMES:
            v = as_text(raw) if typ == 2 else as_nums(tag, typ, cnt, raw)
            print(f'  {nm:16} = {v!r}')
        if tag == 0x927C and mn_raw is None:
            mn_raw = raw
    print('  Exif IFD 标签总数:', len(ex))

print('\n=== MakerNote（Sony 常把 WB 设置写在这里） ===')
if not mn_raw:
    print('  没有 MakerNote 数据')
else:
    hdr = mn_raw[:12]
    print('  头 12 字节:', hdr)
    off = 12 if hdr.startswith(b'SONY') else 0
    try:
        mn_ifd, _ = read_ifd(exif_off and 0 or 0) if False else (None, None)
    except Exception:
        pass
    # MakerNote 里是一个独立 IFD，但偏移基址是文件绝对地址（Sony 就是绝对偏移）
    abs_off = None
    if hdr.startswith(b'SONY'):
        abs_off = (exif_off if exif_off else 0)
    # 找到 MakerNote 在文件中的绝对位置：在 Exif IFD 里它的值是偏移
    pos = data.find(mn_raw[:16])
    print('  MakerNote 在文件中的位置:', pos)
    if pos >= 0:
        mn_ifd_off = pos + (12 if hdr.startswith(b'SONY') else 0)
        ents, _ = read_ifd(mn_ifd_off)
        print('  MakerNote 标签数:', len(ents))
        for tag, typ, cnt, raw in ents:
            vals = as_nums(tag, typ, cnt, raw)
            if typ == 2:
                vals = as_text(raw)
            s = str(vals)
            likely = any(isinstance(v, (int, float)) and 2000 <= v <= 12000 for v in
                         (vals if isinstance(vals, list) else []))
            mark = '   ← 可能是色温 K' if likely else ''
            print(f'    0x{tag:04X} typ={typ} cnt={cnt} = {s[:70]}{mark}')

print('\n=== rawpy 全部公开属性 ===')
try:
    import rawpy
    with rawpy.imread(ARW) as raw:
        names = sorted(n for n in dir(raw) if not n.startswith('_'))
        print('  ', ', '.join(names))
        for n in names:
            if any(k in n.lower() for k in ('exif', 'meta', 'note', 'other', 'temp', 'wb', 'white')):
                try:
                    print(f'    {n} = {getattr(raw, n)!r}'[:120])
                except Exception:
                    pass
except Exception as e:
    print('  rawpy 失败:', e)
