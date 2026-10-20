#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""UI 冒烟测试：验证「界面」与「管线」在源侧这件事上说的是同一句话。

做的事：
  1. 真正构建一次 App（会跑 _build_left，包含新增的 src_hint）
  2. 切到「非 RAW」：源色域/曲线应可编辑，RAW工作空间应置灰
  3. 切到「RAW」：源色域/曲线应置灰并显示解码层的真实值（rw + Linear），RAW工作空间可调
用法（需要桌面会话）：
    .venv\\Scripts\\python.exe tests\\ui_smoke.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import colorlab as cl            # noqa: E402


def state(w):
    try:
        return str(w.cget('state'))
    except Exception as e:                                   # noqa: BLE001
        return f'?({e})'


def main():
    app = cl.App()
    app.update_idletasks()
    app.update()

    app._preview_file = None                                  # 非 RAW
    app._update_pipe()
    print(f'非 RAW : sd={state(app.sd)}  cv={state(app.cv)}  rw={state(app.rw)}')
    assert state(app.sd) == 'normal' and state(app.cv) == 'normal', '非 RAW 时源侧应可编辑'
    assert state(app.rw) == 'disabled', '非 RAW 时 RAW 工作空间应置灰'

    app._preview_file = str(ROOT / 'x.ARW')                    # RAW（只按扩展名判定）
    app._raw_work = 'Rec.2020'
    app._update_pipe()
    print(f'RAW    : sd={state(app.sd)}({app.sd.get()})  cv={state(app.cv)}({app.cv.get()})  rw={state(app.rw)}')
    assert state(app.sd) == 'disabled' and state(app.cv) == 'disabled', 'RAW 时源侧必须锁死'
    assert state(app.rw) == 'normal', 'RAW 时工作空间应可调'
    assert app.sd.get() == 'Rec.2020' and app.cv.get() == 'Linear', 'RAW 源侧必须显示解码层的真实值'
    print('提示文字:', str(app.src_hint.cget('text'))[:44], '…')

    app.destroy()
    print('UI 冒烟通过：源侧控件状态 / 显示值与管线一致')


if __name__ == '__main__':
    main()
