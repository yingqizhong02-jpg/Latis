#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""启动路径回归测试 —— 用 `__main__` 身份跑 colorlab.py，捕获 Tk 回调里的异常。

**为什么必须有这个测试**
项目里其它测试都是 `import colorlab`，而 App 实际是 `python colorlab.py` 启动的。
这两者**不等价**：以脚本运行时本模块叫 `__main__`，colorlab_nb 里的
`import colorlab as _cl` 会把 colorlab.py **再执行一遍**，循环导入因此有不同的时序。
曾经就因此炸过（模块级取半成品 _nb 的属性 → `_cl` 从未绑定 → 非 RAW 预览 NameError），
而当时所有 import 方式的测试全绿，一个都没抓到。

**注入点为什么选 colorlab_nb**
colorlab 会有两份（__main__ 与被 colorlab_nb 反向导入的 colorlab），但 **colorlab_nb 只有一份**。
所以打补丁必须打在 colorlab_nb 上，否则补丁落在另一份 colorlab 上、测试假通过
（这个坑第一版就踩了）。

用法：
    .venv\\Scripts\\python.exe tests\\launch_check.py
    .venv\\Scripts\\python.exe tests\\launch_check.py --inject-error   # 自检：确认本测试真能抓到错
"""
import argparse, sys, time, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ap = argparse.ArgumentParser()
ap.add_argument("--inject-error", action="store_true", help="故意注入回调异常，验证本测试能抓到")
ap.add_argument("--seconds", type=float, default=9.0)
args = ap.parse_args()

import customtkinter as ctk                                     # noqa: E402
import colorlab_nb as _nb                                       # noqa: E402

ERRORS = []
RENDERED = [0]

_orig_report = ctk.CTk.report_callback_exception


def _report(self, exc, val, tb):
    """Tk 回调里的异常默认只打到控制台 —— 这里全部收集。"""
    ERRORS.append("".join(traceback.format_exception(exc, val, tb)))


ctk.CTk.report_callback_exception = _report

# 计数 + 可选注入：都打在 colorlab_nb 上（唯一共享的那一份）
_orig_grade = _nb.grade_fast
_state = {"n": 0}


def _grade_probe(*a, **k):
    _state["n"] += 1
    RENDERED[0] += 1
    if args.inject_error and _state["n"] == 1:
        raise RuntimeError("注入的测试异常（launch_check 应当抓到它）")
    return _orig_grade(*a, **k)


_nb.grade_fast = _grade_probe

_orig_mainloop = ctk.CTk.mainloop


def _drain(self):
    """跑一段真实事件循环：让启动时的 after 回调（_restore / _fit_window / 预览刷新）都走一遍。"""
    t0 = time.time()
    while time.time() - t0 < args.seconds:
        try:
            self.update()
        except Exception:
            ERRORS.append(traceback.format_exc())
        time.sleep(0.02)
    try:
        self.destroy()
    except Exception:
        pass


ctk.CTk.mainloop = _drain

# 预览错误会被 App 吞掉（画到画布上 + 写 preview_error.log），不会冒到 Tk 回调层。
# 所以真正的探测器是**日志有没有增长** —— 这正是用户那次「预览错误」的落点。
LOG = ROOT / "preview_error.log"
_before = LOG.stat().st_size if LOG.exists() else 0

# ---- 测试前置：确保配置里有一张能开的图 ----
# 启动已经改成「窗口先显示、文件延后加载」，而且配置里的文件列表可能为空；
# 不塞图的话本测试必然报"一帧都没渲染"，那是测试前提没满足，不是代码坏了。
_CFG = ROOT / ".colorlab_config"
_cfg_bak = _CFG.read_bytes() if _CFG.exists() else None
_tmp_img = ROOT / "_launch_check_probe.jpg"
try:
    from PIL import Image as _Im
    _Im.new("RGB", (96, 96), (128, 96, 64)).save(_tmp_img)
    import json as _json
    _d = {}
    if _cfg_bak:
        try:
            _d = _json.loads(_cfg_bak.decode("utf-8"))
        except Exception:
            _d = {}
    _d["files"] = [str(_tmp_img)]
    _d.pop("img", None)
    _CFG.write_text(_json.dumps(_d, ensure_ascii=False), encoding="utf-8")
    print(f"已注入测试图：{_tmp_img.name}")
except Exception as _e:
    print(f"注入测试图失败（不影响后续判定）：{_e}")

print(f"以 __main__ 身份执行 colorlab.py（{args.seconds}s 事件循环，注入={args.inject_error}）…")
try:
    import runpy
    runpy.run_path(str(ROOT / "colorlab.py"), run_name="__main__")
except SystemExit:
    pass
except Exception:
    ERRORS.append(traceback.format_exc())

_after = LOG.stat().st_size if LOG.exists() else 0
NEW_LOG = ""
if _after > _before:
    with open(LOG, "r", encoding="utf-8", errors="replace") as f:
        f.seek(_before)
        NEW_LOG = f.read()

print("=" * 72)
print(f"渲染帧次数 = {RENDERED[0]}   Tk回调异常 = {len(ERRORS)}   新增预览错误日志 = {_after - _before} 字节")
# 直接探根因：colorlab_nb 的 `import colorlab as _cl` 若失败会把 _HAVE_PARENT 置 False，
# 之后 _cl 未绑定 → 任何 temp≠0 的调色都会 NameError。这里是这个 bug 的最短判据。
print(f"colorlab_nb._HAVE_PARENT = {_nb._HAVE_PARENT}   （必须 True）")

if not _nb._HAVE_PARENT:
    print("失败：colorlab_nb 反向导入 colorlab 失败（循环导入/二次导入）→ _cl 未绑定")
    print("      典型触发：以 __main__ 身份运行时 colorlab.py 被再执行一遍，")
    print("      第二份执行碰到半成品 _nb。见 colorlab.py 顶部的 sys.modules.setdefault 修复。")
    sys.exit(1)

# ---- 还原配置与临时文件 ----
try:
    if _cfg_bak is not None:
        _CFG.write_bytes(_cfg_bak)
    elif _CFG.exists():
        _CFG.unlink()
    if _tmp_img.exists():
        _tmp_img.unlink()
except Exception:
    pass

if RENDERED[0] == 0:
    print("失败：一帧都没渲染 —— 启动后没加载任何图片，本测试是空转的")
    sys.exit(1)

if ERRORS or NEW_LOG:
    if NEW_LOG:
        print("失败：预览层报错（被 App 吞进画布与日志，import 方式的测试看不到）")
        for ln in NEW_LOG.strip().splitlines()[:14]:
            print("   ", ln)
    if ERRORS:
        print(f"失败：Tk 回调层 {len(ERRORS)} 个异常")
        for e in ERRORS[:3]:
            print("-" * 72); print(e.rstrip())
    print("=" * 72)
    sys.exit(1)

print("通过：启动路径无异常（Tk 回调层 + 预览层都干净），且确实渲染过")
sys.exit(0)
