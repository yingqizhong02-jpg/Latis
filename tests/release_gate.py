#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""发布闸门：一键跑完所有「能不能发」的检查。

区别于单元测试：这里检查的是**用户真实会碰到的东西** ——
  · 启动是否够快（窗口多久出来）
  · 真实鼠标能不能拖动侧栏、会不会卡死
  · LUT 弹窗能不能打开/切换/关闭
  · 导出是否真的产出文件
  · 管线数值是否仍然正确（125 项色彩科学断言）

用法：
    .venv\\Scripts\\python.exe tests\\release_gate.py
"""
import ctypes
import ctypes.wintypes as wt
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
sys.path.insert(0, str(ROOT))

RESULTS = []


def check(name, fn):
    t = time.perf_counter()
    try:
        ok, detail = fn()
    except Exception:
        ok, detail = False, traceback.format_exc(limit=3).splitlines()[-1]
    dt = (time.perf_counter() - t) * 1000
    RESULTS.append((name, ok, detail, dt))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:<34} {detail:<44} {dt:7.0f} ms")
    return ok


# ---------------- 真实鼠标输入 ----------------
u32 = ctypes.windll.user32
LD, LU, MV, AB = 0x0002, 0x0004, 0x0001, 0x8000


class _MI(ctypes.Structure):
    _fields_ = [("dx", wt.LONG), ("dy", wt.LONG), ("md", wt.DWORD),
                ("fl", wt.DWORD), ("tm", wt.DWORD), ("ex", ctypes.POINTER(wt.ULONG))]


class _INP(ctypes.Structure):
    _fields_ = [("ty", wt.DWORD), ("mi", _MI)]


def _send(flags, x=0, y=0):
    sw, sh = u32.GetSystemMetrics(0), u32.GetSystemMetrics(1)
    ax = int(x * 65535 / max(1, sw - 1)) if flags & AB else 0
    ay = int(y * 65535 / max(1, sh - 1)) if flags & AB else 0
    u32.SendInput(1, ctypes.byref(_INP(0, _MI(ax, ay, 0, flags, 0, None))),
                  ctypes.sizeof(_INP))


def move_to(x, y):
    _send(MV | AB, x, y)


def down():
    _send(LD)


def up():
    _send(LU)


def cursor():
    p = wt.POINT(); u32.GetCursorPos(ctypes.byref(p)); return p.x, p.y


def main():
    import importlib
    import colorlab as cl
    importlib.reload(cl)

    print("═" * 78)
    print("  发布闸门")
    print("═" * 78)
    print("\n【1】单元 / 色彩科学")
    # numeric_equiv.py 是手动工具（要 dump/compare 子命令），不进闸门
    for t in ("tests/color_science_check.py",):
        p = ROOT / t
        if not p.exists():
            print(f"  [SKIP] {t} 不存在")
            continue

        def _run(p=p):
            r = subprocess.run([PY, str(p)], capture_output=True, text=True,
                               encoding="utf-8", errors="replace", timeout=900)
            out = (r.stdout or "") + (r.stderr or "")
            m = [l for l in out.splitlines() if "汇总" in l or "通过" in l or "FAIL" in l]
            return r.returncode == 0, (m[-1][:44] if m else "无输出")
        check(p.name, _run)

    print("\n【2】启动路径")
    t0 = time.perf_counter()
    app = cl.App()
    ctor = (time.perf_counter() - t0) * 1000

    def _ctor():
        return ctor < 4000, f"构造 {ctor:.0f} ms"
    check("App 构造耗时 < 4s", _ctor)

    for _ in range(40):
        app.update(); time.sleep(0.03)
    app.attributes("-topmost", True)
    app.lift()
    try:
        app.focus_force()
    except Exception:
        pass
    for _ in range(10):
        app.update(); time.sleep(0.04)
    app.update_idletasks()

    def _maximized():
        return app.state() == "zoomed", f"state={app.state()}"
    check("默认最大化", _maximized)

    def _keys():
        miss = [k for k in ("<space>", "<z>", "<f>") if not app.bind(k)]
        return not miss, ("全部已绑" if not miss else f"缺 {miss}")
    check("快捷键 space/z/f", _keys)

    print("\n【3】真实鼠标拖动侧栏")
    old = cursor()
    try:
        for h, tag in ((app._h_left, "左手柄"), (app._h_right, "右手柄")):
            hx = h.winfo_rootx() + h.winfo_width() // 2
            hy = h.winfo_rooty() + h.winfo_height() // 2
            w0 = (app._lf_ctrl.master.winfo_width() if tag == "左手柄"
                  else app.sf.master.winfo_width())
            move_to(hx, hy); app.update(); time.sleep(0.05)
            down()
            for _ in range(6):
                app.update(); time.sleep(0.02)
            pressed = getattr(app, "_drag_on", None) is True
            for i in range(30):
                move_to(hx + 4 * (i + 1), hy)
                app.update(); time.sleep(0.004)
            for _ in range(6):
                app.update(); time.sleep(0.02)
            w1 = (app._lf_ctrl.master.winfo_width() if tag == "左手柄"
                  else app.sf.master.winfo_width())
            move_to(hx + 300, hy - 150)
            app.update(); time.sleep(0.04)
            up()
            for _ in range(20):
                app.update(); time.sleep(0.02)
            released = getattr(app, "_drag_on", None) is False
            moved = abs(w1 - w0) > 10
            check(f"{tag} 按下/移动/松开",
                  lambda p=pressed, r=released, m=moved, w0=w0, w1=w1:
                  (p and r and m, f"press={p} release={r} {w0}→{w1}"))

        # 拖动后 UI 仍响应
        def _alive():
            z0 = app._pv_zlbl.cget("text")
            app._zoom_btn(1); app.update(); time.sleep(0.03)
            z1 = app._pv_zlbl.cget("text")
            app._zoom_fit(); app.update()
            return z0 != z1, f"{z0}→{z1}"
        check("拖动后 UI 仍响应", _alive)
    finally:
        try:
            up()
        except Exception:
            pass
        move_to(*old)

    print("\n【4】LUT 交互")
    def _popup():
        app._lut_popup()
        for _ in range(8):
            app.update(); time.sleep(0.03)
        opened = getattr(app, "_lut_win", None) is not None
        app._lut_popup()          # 再点即关
        for _ in range(6):
            app.update(); time.sleep(0.03)
        closed = getattr(app, "_lut_win", None) is None
        return opened and closed, f"开={opened} 关={closed}"
    check("弹窗开关", _popup)

    def _switch():
        if len(app._luts) < 2:
            return False, "LUT 不足 2 个"
        before = app.ld.get()
        target = [l["disp"] for l in app._luts if l["disp"] != before][0]
        app.ld.set(target); app._pk()
        for _ in range(6):
            app.update(); time.sleep(0.02)
        return app.ld.get() == target, f"{before}→{target}"
    check("切换 LUT", _switch)

    def _outspace():
        snap = app._snapshot()
        lo, loc = snap.get("lo"), snap.get("loc")
        return bool(lo and loc), f"lo={lo} loc={loc}（未声明→Rec.709）"
    check("LUT 输出空间可解析", _outspace)

    def _target():
        snap = app._snapshot()
        return bool(snap.get("tg") and snap.get("tc")), \
            f"tg={snap.get('tg')} tc={snap.get('tc')}"
    check("文件目标空间独立", _target)

    app.destroy()

    print("\n【5】导出")
    def _export():
        r = subprocess.run([PY, str(ROOT / "tests" / "export_smoke.py")],
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=900)
        out = (r.stdout or "") + (r.stderr or "")
        m = [l for l in out.splitlines() if "导出返回" in l]
        return r.returncode == 0 and "True" in (m[-1] if m else ""), \
            (m[-1][:44] if m else "无输出")
    check("导出冒烟", _export)

    print("\n" + "═" * 78)
    npass = sum(1 for _, ok, _, _ in RESULTS if ok)
    nfail = len(RESULTS) - npass
    print(f"  结果：{npass} 通过 / {nfail} 失败   共 {len(RESULTS)} 项")
    if nfail:
        print("\n  失败项：")
        for name, ok, detail, _ in RESULTS:
            if not ok:
                print(f"    ✗ {name}: {detail}")
    print("═" * 78)
    return 0 if nfail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
