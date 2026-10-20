#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""基线渲染：把标准 LUT 工作流完整跑一遍并落盘，作为后续改动的对照物。

对标的正是 colorlab.py 的 _transform() 在「套 LUT」分支上的执行顺序：
    RAW 解码 → _grade（源空间调色）→ Pipe.go（转到 LUT 输入空间）→ _al（四面体套 LUT）
（colorlab.py:984-1002；RAW 时 tp=0，见 :991）

输出到 tests/baseline/：
    <tag>.tif   16bit TIFF（看精度）
    <tag>.png   8bit PNG（肉眼对照）
    <tag>.json  耗时分解 + 逐通道统计 + float32 结果 sha1（看是否被改动）

用法：
    .venv\\Scripts\\python.exe tests\\baseline_render.py --tag before-step1
    .venv\\Scripts\\python.exe tests\\baseline_render.py --tag before-step1 --full   # 33MP 全尺寸
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import colorlab as cl            # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--tag', default='baseline')
ap.add_argument('--arw', default=r'C:\Users\Administrator\Desktop\新建文件夹 (4)\DSC03492.ARW')
ap.add_argument('--lut', default=None, help='默认 lut\\V-Gamut-V-Log\\Fujifilm\\Fuji_Velvia.cube')
ap.add_argument('--full', action='store_true', help='默认半尺寸；加此参数跑 33MP 全尺寸')
# 与界面默认对齐的调色参数
ap.add_argument('--ev', type=float, default=0.3)
ap.add_argument('--ct', type=float, default=0.10)
ap.add_argument('--sat', type=float, default=0.15)
args = ap.parse_args()

OUT = ROOT / 'tests' / 'baseline'
OUT.mkdir(parents=True, exist_ok=True)
arw = Path(args.arw)
if not arw.exists():
    sys.exit(f'输入 ARW 不存在：{arw}')
lut = Path(args.lut) if args.lut else ROOT / 'lut' / 'V-Gamut-V-Log' / 'Fujifilm' / 'Fuji_Velvia.cube'
if not lut.exists():
    sys.exit(f'LUT 不存在：{lut}')

li, lc = (None, None)
lut_root = ROOT / 'lut'
try:
    # 与 colorlab._sl() 一致的取法：色域-曲线 由「相对 lut/ 的第一层」决定，不是直接父目录
    rel = lut.relative_to(lut_root)
    if len(rel.parts) > 1:
        li, lc = cl._folder_input(rel.parts[0])
except ValueError:
    li, lc = cl._folder_input(lut.parent.name)
if li is None:
    sys.exit(f'无法从第一层文件夹名推断 LUT 输入空间：{lut}（相对 lut/ 的第一层应为「色域-曲线」）')

inf = cl._cp(str(lut))
sg, sc = cl._RAW_WORK_DEF, 'sRGB'
t = {}
T = time.perf_counter
print(f'输入 : {arw.name}   half={not args.full}')
print(f'LUT  : {lut.relative_to(ROOT)}   type={inf["t"]} size={inf["n"]}   输入空间={li}/{lc}')
print(f'调色 : ev={args.ev} ct={args.ct} sat={args.sat}  (RAW→tp=0)')

_shot = cl.read_shot_cct(arw) or 6500.0
print(f'  机内 WB 等效色温   : {_shot:.0f} K   （滑块默认落在这里；未手动改则增益=1，即"按机内"）')
t0 = T(); a = cl._load_float(str(arw), half=not args.full, work=cl._RAW_WORK_DEF, wb=(_shot, 0.0, _shot)); t['decode'] = T() - t0
mp = a.shape[0] * a.shape[1] / 1e6
print(f'  解码 {a.shape} ({mp:.1f} MP) : {t["decode"]:.2f}s')

# 走与 App._transform 完全相同的那条链（模块级 cl.transform，含 numba 分支），
# 避免"基准测的不是 App 跑的那条路"
s = {'sg': sg, 'sc': sc, 'tg': 'sRGB', 'tc': 'sRGB', 'raw': True, 'rw': cl._RAW_WORK_DEF,
     'lut': True, 'cd': inf, 'li': li, 'lc': lc, 'ootf': 'auto',
     'ev': args.ev, 'ct': args.ct, 'sat': args.sat, 'tp': 6500.0,
     'dither': False}
seg = {}
# JIT 预热：numba 首次调用要编译内核（秒级）。必须与计时分开，否则编译时间会被算进第一个环节。
_w = np.random.RandomState(0).rand(64, 64, 3).astype(np.float32)
t0 = T(); _ = cl.transform(_w, s); t['warmup'] = T() - t0
print(f'  JIT 预热（首次编译/载入缓存）: {t["warmup"]:.2f}s')
t0 = T(); a = cl.transform(a, s, timing=seg); t['process'] = T() - t0
t['grade'] = seg.get('grade', 0.0); t['pipe'] = seg.get('pipe', 0.0); t['lut'] = seg.get('lut', 0.0)
cst_neg = seg.get('cst_neg', -1)
print(f'  路径                       : {"numba" if cl._NB else "numpy"}   缩限={cl._GC_MODE}')
print(f'  调色                       : {t["grade"]:.2f}s')
print(f'  CST（含缩限）→ {li}/{lc}  : {t["pipe"]:.2f}s   CST 负分量通道数 = {cst_neg}')
print(f'  套 LUT                     : {t["lut"]:.2f}s')
a = np.asarray(a, dtype=np.float32)
preclip = {'below0': int((a < 0).sum()), 'above1': int((a > 1).sum()), 'nan': int(np.isnan(a).sum())}
a = np.clip(a, 0, 1)
t['total'] = t['decode'] + t['grade'] + t['pipe'] + t['lut']

sha = hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()
import tifffile                                              # noqa: E402
from PIL import Image                                        # noqa: E402
tif = OUT / f'{args.tag}.tif'
png = OUT / f'{args.tag}.png'
t0 = T(); tifffile.imwrite(tif, (a * 65535 + 0.5).astype(np.uint16), photometric='rgb', compression='deflate')
t['write_tiff'] = T() - t0
t0 = T(); Image.fromarray((a * 255 + 0.5).astype(np.uint8)).save(png); t['write_png'] = T() - t0
stats = {
    'shape': list(a.shape), 'mp': round(mp, 2), 'half': not args.full,
    'mean': [round(float(a[..., c].mean()), 6) for c in range(3)],
    'min': [round(float(a[..., c].min()), 6) for c in range(3)],
    'max': [round(float(a[..., c].max()), 6) for c in range(3)],
    'nan': int(np.isnan(a).sum()), 'below0': int((a < 0).sum()), 'above1': int((a > 1).sum()),
}
meta = {'tag': args.tag, 'arw': str(arw), 'lut': str(lut.relative_to(ROOT)), 'lut_input_space': f'{li}/{lc}',
        'grade': {'ev': args.ev, 'ct': args.ct, 'sat': args.sat},
        'numa': cl._NB, 'sha1_float32': sha,
        'version': getattr(cl, 'APP_VERSION', '?'),
        'gc_mode': getattr(cl, '_GC_MODE', 'n/a'), 'cst_neg_channels': cst_neg, 'preclip': preclip,
        'seconds': {k: round(v, 3) for k, v in t.items()}, 'stats': stats,
        'tiff': tif.name, 'png': png.name, 'tiff_bytes': tif.stat().st_size}
(OUT / f'{args.tag}.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
print(f'  写出 tiff/png              : {t["write_tiff"]:.2f}s / {t["write_png"]:.2f}s')
print(f'\n结果 sha1(float32) = {sha}')
print(f'统计 = {json.dumps(stats, ensure_ascii=False)}')
print(f'处理合计 = {t["total"]:.2f}s   （numba={cl._NB}）')
print(f'已写入 {OUT.relative_to(ROOT)}/{{{args.tag}.tif, {args.tag}.png, {args.tag}.json}}')
