#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""UI 视觉检查（无头）：窗口是否在屏幕内、波形两种模式能否正常出图。

用法：.venv\\Scripts\\python.exe tests\\ui_visual_check.py
需要桌面会话（会真的建一次窗口，随即销毁）。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np                # noqa: E402
import colorlab as cl             # noqa: E402

app = cl.App()
app.update_idletasks()
app.update()

app._fit_window()
sw, sh = app.winfo_screenwidth(), app.winfo_screenheight()
print(f'屏幕: {sw}x{sh}')
print(f'窗口: {app.geometry()}  state={app.state()}')
geo_ok = True
try:
    _g = app.geometry()                       # 形如 WxH+X+Y
    _wh, _xy = _g.split('+')[0], _g.split('+')[1:]
    _w, _h = (int(v) for v in _wh.split('x'))
    _x, _y = (int(v) for v in _xy)
    geo_ok = (_w <= sw) and (_h <= sh) and (_x >= 0) and (_y >= 0) and (_x + _w <= sw) and (_y + _h <= sh)
    print(f'  解析: {_w}x{_h}+{_x}+{_y}  完全在屏幕内: {geo_ok}')
except Exception as e:
    print('  geometry 解析失败:', e)

src = app._pv_lin_small if app._pv_lin_small is not None else app._pv_lin
if src is None:
    print('没有预览图，跳过波形检查')
else:
    u8 = cl._lin_to_u8(src)[::4, ::4]
    for mode in (False, True):
        app._scope_rgb = mode
        app._draw_waveform(u8)
        ph = app._scope_photo
        cw, chh = app._scope_cv.winfo_width(), app._scope_cv.winfo_height()
        ok = (ph.width() == cw - 34 - 1) and (ph.height() == chh)
        print(f'  mode={"RGB" if mode else "Y "} 画布 {cw}x{chh} 波形图 {ph.width()}x{ph.height()} '
              f'尺寸匹配={ok}')

    if src is not None:
        app._render_scope()
        app.update_idletasks()
        sc, cv = app._scope, app._scope_cv
        # 注意：winfo_rootx/rooty 与 winfo_width 是同一套（物理）像素，而
        # winfo_screenwidth/geometry() 是 Tk 逻辑像素；在 175% 缩放下两者差 1.75 倍，
        # 混着比会得出"跑到屏幕外"的错误结论。所以这里一律用"相对窗口"判断。
        pr = sc.winfo_rootx() - app.winfo_rootx() + sc.winfo_width()
        pb = sc.winfo_rooty() - app.winfo_rooty() + sc.winfo_height()
        print(f'  窗口 {app.winfo_width()}x{app.winfo_height()}（物理） / {app.geometry()}（Tk 逻辑）')
        print(f'  面板 相对窗口 右边缘 {pr} / 下边缘 {pb}')
        print(f'  面板在窗口内: {pr <= app.winfo_width() and pb <= app.winfo_height()}')

app.destroy()
print('UI 视觉检查结束')