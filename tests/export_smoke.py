#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""导出冒烟：真正跑一次 App._export_one（PNG 8-bit），验证导出链路端到端可用。

用法（需要桌面会话）：
    .venv\\Scripts\\python.exe tests\\export_smoke.py [输出目录] [--keep]
默认输出到 tests/export_out/（不动用户的 output 目录）。
跑完会**自动删掉两个测试产物**（33MP 的 PNG+TIFF 合计约 200MB），
否则每跑一次就往项目里堆 200MB。要留档就加 --keep。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import colorlab as cl                                     # noqa: E402

ARW = Path(r'C:\Users\Administrator\Desktop\新建文件夹 (4)\DSC03492.ARW')
_args = [a for a in sys.argv[1:] if not a.startswith('--')]
KEEP = '--keep' in sys.argv
OUT = Path(_args[0]) if _args else ROOT / 'tests' / 'export_out'
OUT.mkdir(parents=True, exist_ok=True)

app = cl.App()
app.update_idletasks()
app.update()

app._preview_file = str(ARW)
app._od = str(OUT)
print('输出目录 :', OUT)
print('可写预检 :', app._out_writable(str(OUT)))

snap = app._snapshot()
snap['fmt'] = 'PNG 8-bit'
snap['ext'] = cl._OUT_FMTS[snap['fmt']]
print('格式     :', snap['fmt'], '→', snap['ext'], '| LUT:', snap['lut'], '| dither:', snap['dither'])

ok = app._export_one(ARW, snap)
print('导出返回 :', ok)
p = OUT / (ARW.stem + snap['ext'])
if p.exists():
    print(f'产物     : {p}  {p.stat().st_size/1e6:.1f} MB')
else:
    print('产物缺失 :', p)

# 再试一次 TIFF 16-bit（不同编码分支 + ICC）
snap2 = dict(snap); snap2['fmt'] = 'TIFF 16-bit'; snap2['ext'] = cl._OUT_FMTS[snap2['fmt']]
ok2 = app._export_one(ARW, snap2)
p2 = OUT / (ARW.stem + snap2['ext'])
print('TIFF 返回:', ok2, '| 产物:', (f'{p2.stat().st_size/1e6:.1f} MB' if p2.exists() else '缺失'))

app.destroy()

# 自清理：这两个产物合计约 200MB，留在项目里会越堆越多
_made = [p for p in (p, p2) if p.exists()]
if _made and not KEEP:
    _mb = sum(q.stat().st_size for q in _made) / 1e6
    for q in _made:
        try:
            q.unlink()
        except Exception:
            pass
    try:                                   # 目录空了就一起收掉
        if not any(OUT.iterdir()):
            OUT.rmdir()
    except Exception:
        pass
    print(f'已清理测试产物 {len(_made)} 个 / {_mb:.0f} MB（--keep 可保留）')
print('导出冒烟结束')
