#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""性能基线门禁 —— 走 App._export_one 的**真实**代码路径。

3.3 的教训：基准脚本直接调 _grade/Pipe.go/_al，测的根本不是 App 跑的那条路。
所以这里用一个最小 stub 冒充 App 实例，直接调 `colorlab.App._export_one`，
保证"测的就是 App 导出的那条路"。每个阶段另外单独计时，用于定位。

用法：
    .venv\\Scripts\\python.exe tests\\perf_baseline.py                # 默认 3 张 PNG
    .venv\\Scripts\\python.exe tests\\perf_baseline.py --files 5 --fmt tiff
    .venv\\Scripts\\python.exe tests\\perf_baseline.py --tag p1-after
"""
import argparse, json, statistics, sys, time
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--tag', default='baseline')
ap.add_argument('--root', default=None, help='要测量的 colorlab 所在目录（默认本脚本的上一级；用它可以 A/B 对比 3.3 与 4.0）')
ap.add_argument('--dir', default=r'C:\Users\Administrator\Desktop\新建文件夹 (4)')
ap.add_argument('--files', type=int, default=3)
ap.add_argument('--fmt', default='png', choices=['png', 'tiff', 'jpg'])
ap.add_argument('--ev', type=float, default=0.3)
ap.add_argument('--ct', type=float, default=0.10)
ap.add_argument('--sat', type=float, default=0.15)
ap.add_argument('--lut', default=None)
ap.add_argument('--outdir', default=None, help='结果落盘目录（默认 <root>/tests/perf；A/B 时指定同一个可避免污染对照树）')
args = ap.parse_args()

ROOT = Path(__file__).resolve().parent.parent
if args.root:
    ROOT = Path(args.root).resolve()
sys.path.insert(0, str(ROOT))
import colorlab as cl                                     # noqa: E402

OUTDIR = Path(args.outdir).resolve() if args.outdir else ROOT / 'tests' / 'perf'
OUT = OUTDIR / 'render'
OUT.mkdir(parents=True, exist_ok=True)

EXT = {'png': '.png', 'tiff': '.tiff', 'jpg': '.jpg'}[args.fmt]
lut = Path(args.lut) if args.lut else ROOT / 'lut' / 'V-Gamut-V-Log' / 'Fujifilm' / 'Fuji_Velvia.cube'
rel = lut.relative_to(ROOT / 'lut')
li, lc = cl._folder_input(rel.parts[0])
inf = cl._cp(str(lut))

arws = sorted(Path(args.dir).glob('*.ARW'))[:args.files]
if not arws:
    sys.exit(f'找不到 ARW: {args.dir}')


class _Stub:
    """冒充 App 实例，喂给 App._export_one 用（它只用到这几样）。"""
    def __init__(self, od, raw_work):
        self._od = str(od); self._raw_work = raw_work
        self._shot = 6500.0
        self._png_level = cl._png_level() if hasattr(cl, '_png_level') else 6
    def _wb_args(self):
        return (float(self._shot), 0.0, float(self._shot))
    def _transform_tiled(self, a, s):
        return cl._tiled(a, lambda x: cl.transform(x, s, self._raw_work))


stub = _Stub(OUT, cl._RAW_WORK_DEF)

# JIT 预热，别把编译时间算进第一个环节
_w = np.random.RandomState(0).rand(64, 64, 3).astype(np.float32)
_snap0 = {'sg': cl._RAW_WORK_DEF, 'sc': 'sRGB', 'raw': True, 'rw': cl._RAW_WORK_DEF,
          'tg': 'sRGB', 'tc': 'sRGB', 'lut': True, 'ootf': 'auto', 'li': li, 'lc': lc,
          'cd': inf, 'cd_path': str(lut), 'ev': args.ev, 'ct': args.ct, 'sat': args.sat,
          'tp': 6500.0, 'dither': False, 'ext': EXT, 'fmt': 'PNG 8-bit'}
cl.transform(_w, _snap0)

print(f'Color-Lab {cl.APP_VERSION}  后端={"numba" if cl._NB else "numpy"}  GC={cl._GC_MODE}')
print(f'输入 {len(arws)} 张  LUT={rel}  格式={args.fmt.upper()}')
print(f'{"文件":<16}{"解码":>8}{"处理":>8}{"写盘":>8}{"合计":>8}')
print('-' * 50)

T = time.perf_counter
rows = []
for f in arws:
    shot = cl.read_shot_cct(f) or 6500.0
    stub._shot = shot
    snap = dict(_snap0); snap['tp'] = shot

    # 分段计时（与 _export_one 同序同步）
    t0 = T(); a = cl._load_float(str(f), work=stub._raw_work, wb=stub._wb_args()); t_dec = T() - t0
    t0 = T(); a = np.clip(cl._tiled(a, lambda x: cl.transform(x, snap)), 0, 1); t_proc = T() - t0
    del a

    # 真实端到端：直接调 App._export_one
    t0 = T(); ok = cl.App._export_one(stub, f, snap); t_tot = T() - t0
    if not ok:
        sys.exit(f'导出失败: {f.name}（见 export_error.log）')
    t_wr = t_tot - t_dec - t_proc

    rows.append(dict(name=f.name, decode=t_dec, process=t_proc, write=t_wr, total=t_tot))
    print(f'{f.name:<16}{t_dec:7.2f}s{t_proc:7.2f}s{t_wr:7.2f}s{t_tot:7.2f}s')

def med(k): return statistics.median(r[k] for r in rows)
tot = med('total')
print('-' * 50)
print(f'{"中位数":<16}{med("decode"):7.2f}s{med("process"):7.2f}s{med("write"):7.2f}s{tot:7.2f}s')
print(f'\n占比: 解码 {med("decode")/tot*100:.0f}%  处理 {med("process")/tot*100:.0f}%  写盘 {med("write")/tot*100:.0f}%')
print(f'70 张预计 {tot*70/60:.1f} 分钟')
print(f'→ 色彩运算若降到 0，端到端最多省 {med("process")/tot*100:.1f}%')

meta = dict(tag=args.tag, version=cl.APP_VERSION, backend='numba' if cl._NB else 'numpy',
            fmt=args.fmt, gc=cl._GC_MODE, files=[r['name'] for r in rows],
            median={k: round(med(k), 4) for k in ('decode', 'process', 'write', 'total')},
            per_file=[{k: round(v, 4) for k, v in r.items() if k != 'name'} | {'name': r['name']} for r in rows],
            lut=str(rel), grade=dict(ev=args.ev, ct=args.ct, sat=args.sat))
(OUTDIR / f'{args.tag}.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
try:                     # A/B 时 OUTDIR 可能在被测树之外（例如测 3.3 却把结果写到 4.0）
    _shown = OUTDIR.relative_to(ROOT)
except ValueError:
    _shown = OUTDIR
print(f'\n已写 {_shown}/{args.tag}.json')
