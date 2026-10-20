#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""逐级指纹：把「文件 → 屏幕」拆成 5 级，每级都给**实际值 vs 独立期望值**。

## 它解决什么问题

A/B 对比图只能证明"两者不同"，证明不了**谁对**，更不能**定位**。
这个工具把每一级摊开：哪一级偏了、偏多少，直接指出来。

    ①  文件码值         ← 公开参考值（ColorChecker sRGB，BabelColor）
    ②  解码 → 线性      ← 期望：encode(decode(x)) == x 且白点归一
    ③  CST → 目标空间   ← 期望：中性轴保持、白点保持
    ④  显示端 → 显示器码值 ← 期望：与「直接转到显示器空间」逐位一致
    ⑤  显示器码值 → 绝对 XYZ/Lab ← **终点。与①的绝对颜色比 ΔE76。**

第⑤级就是 ground truth：**文件的绝对颜色是确定的**，屏幕该发出那个颜色。
这一级过了，说明整条链自洽；没过，前面四级会告诉你是哪一级坏的。

## 什么**不**在它的能力范围内

它验证的是「文件 → 显示器码值」这一段。**显示器码值 → 面板实际发出的光**
需要色度计，不在本工具范围内 —— 那一段只能信任显示器档案。

用法：
    python tools/chain_fingerprint.py                     # 自动读显示器档案
    python tools/chain_fingerprint.py --display "Display P3"
    python tools/chain_fingerprint.py --target Rec.709    # 让管线输出 Rec.709
    python tools/chain_fingerprint.py --all               # 跑多种组合
"""
import sys
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import colorlab as cl      # noqa: E402

# ---- ColorChecker Classic 的 sRGB 参考值（BabelColor 公开数据）----
CHECKER = [
    ("深肤", (115, 82, 68)), ("浅肤", (194, 150, 130)),
    ("天蓝", (98, 122, 157)), ("叶绿", (87, 108, 67)),
    ("花蓝", (133, 128, 177)), ("蓝绿", (103, 189, 170)),
    ("橙", (214, 126, 44)), ("紫蓝", (80, 91, 166)),
    ("中红", (193, 90, 99)), ("紫", (94, 60, 108)),
    ("黄绿", (157, 188, 64)), ("橙黄", (224, 163, 46)),
    ("蓝", (56, 61, 150)), ("绿", (70, 148, 73)),
    ("红", (175, 54, 60)), ("黄", (231, 199, 31)),
    ("品红", (187, 86, 149)), ("青", (8, 133, 161)),
    ("白", (243, 243, 242)), ("中性8", (200, 200, 200)),
    ("中性6.5", (160, 160, 160)), ("中性5", (122, 122, 121)),
    ("中性3.5", (85, 85, 85)), ("黑", (52, 52, 52)),
]
PRIMARIES = [("R", (255, 0, 0)), ("G", (0, 255, 0)), ("B", (0, 0, 255)),
             ("C", (0, 255, 255)), ("M", (255, 0, 255)), ("Y", (255, 255, 0))]
NEUTRALS = {"白", "中性8", "中性6.5", "中性5", "中性3.5", "黑"}


# ---------------------------------------------------------------- 独立实现
# 这一节**故意不复用** colorlab 的 Pipe / _mat / _cat —— 独立算一遍才有鉴别力。
def _xy_to_xyz(x, y):
    return np.array([x / y, 1.0, (1.0 - x - y) / y])


def _rgb_to_xyz(prims, white_xy):
    """由原色坐标 + 白点构造 RGB→XYZ(该白点) 矩阵（标准归一化）。"""
    cols = [_xy_to_xyz(x, y) for x, y in prims]
    M = np.stack(cols, axis=1)
    w = _xy_to_xyz(*white_xy)
    s = np.linalg.solve(M, w)
    return M * s


def _bradford(src_xy, dst_xy):
    """Bradford 色适应矩阵（独立实现）。"""
    B = np.array([[0.8951, 0.2664, -0.1614],
                  [-0.7502, 1.7135, 0.0367],
                  [0.0389, -0.0685, 1.0296]])
    ws = B @ _xy_to_xyz(*src_xy)
    wd = B @ _xy_to_xyz(*dst_xy)
    return np.linalg.inv(B) @ np.diag(wd / ws) @ B


def _xyz_to_lab(xyz, white_xyz):
    t = xyz / white_xyz
    e, k = 216 / 24389, 24389 / 27
    f = np.where(t > e, np.cbrt(t), (k * t + 16) / 116)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


D65 = (0.3127, 0.3290)
D50 = (0.34567, 0.35850)
SRGB_PRIM = [(0.6400, 0.3300), (0.3000, 0.6000), (0.1500, 0.0600)]
M_SRGB = _rgb_to_xyz(SRGB_PRIM, D65)                       # 独立矩阵
LUT_SRGB_DEC = getattr(cl.CURVE["sRGB"][1], "__call__", None)


def _srgb_decode(v):
    v = np.asarray(v, np.float64)
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def _srgb_encode(x):
    x = np.asarray(x, np.float64)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.clip(x, 0, None) ** (1 / 2.4) - 0.055)


# ---------------------------------------------------------------- 指纹
def _space_mat(gamut):
    """取 colorlab 对某色域的 RGB→XYZ（原生白点）矩阵 —— 被测对象，不是期望值。"""
    return np.asarray(cl._mat(gamut)[0], np.float64)


def _space_wp(gamut):
    g = cl._GAMUT.get(gamut, {})
    wp = g.get("wp", "D65")
    xy = cl._WP_XYZ.get(wp, D65)
    return (float(xy[0]), float(xy[1]))


def fingerprint_one(rgb8, tg, tc, dg, dc):
    """返回该色块的 5 级数值 + 每级偏差。"""
    r = {}

    # ① 文件码值
    r["file"] = np.asarray(rgb8, np.float64) / 255.0

    # ② 解码 → 线性（实际：colorlab 的曲线；期望：独立 sRGB 解码）
    dec = cl.CURVE["sRGB"][1]
    got_lin = np.asarray(dec(np.asarray(rgb8, np.float32) / 255.0), np.float64)
    want_lin = _srgb_decode(r["file"])
    r["lin"] = got_lin
    r["lin_err"] = float(np.abs(got_lin - want_lin).max())

    # ③ CST → 目标空间（实际：Pipe.go；期望：独立矩阵 + 曲线编码）
    a = r["file"].astype(np.float32).reshape(1, 1, 3)
    # 与 transform 同源：目标编码用 display_decode_curve(tc)（不是 tc 本身）。
    # 早先这里直接传 tc，绕过了真实管线 —— 修了 colorlab 工具却测不出来。
    got_t = np.asarray(cl.Pipe.go(a, "sRGB", "sRGB", tg,
                                  cl.display_decode_curve(tc), ootf="off", gc=2),
                       np.float64).reshape(3)
    # 期望：XYZ(D65) → 目标原生白点 → 目标编码
    xyz = M_SRGB @ want_lin
    Ms = _space_mat(tg)
    w_dst = _space_wp(tg)
    if w_dst != D65:
        xyz = _bradford(D65, w_dst) @ xyz
    want_lin_t = np.linalg.solve(Ms, xyz)
    enc = cl.CURVE[cl.display_decode_curve(tc)][0]
    want_t = np.asarray(enc(np.clip(want_lin_t, 0, None).astype(np.float32)), np.float64)
    r["target"] = got_t
    r["target_err"] = float(np.abs(got_t - want_t).max())

    # ④ 显示端 → 显示器码值（实际：display_leg_nb；期望：直接转到显示器空间）
    snap = {"lut": False, "cd": None, "tg": tg, "tc": tc}
    got_d = np.asarray(cl.display_leg_nb(got_t.astype(np.float32).reshape(1, 1, 3),
                                         snap), np.float64).reshape(3)
    want_d = np.asarray(cl.Pipe.go(a, "sRGB", "sRGB", dg, dc, ootf="off", gc=2),
                        np.float64).reshape(3)
    r["disp"] = got_d
    r["disp_err"] = float(np.abs(got_d - want_d).max())

    # ⑤ 显示器码值 → 绝对颜色，与①比 ΔE76（**终点判据**）
    ddec = cl.CURVE[dc][1]
    d_lin = np.asarray(ddec(np.clip(got_d, 0, 1).astype(np.float32)), np.float64)
    Md = _space_mat(dg)
    xyz_d = Md @ d_lin
    w_d = _space_wp(dg)
    if w_d != D65:
        xyz_d = _bradford(w_d, D65) @ xyz_d
    lab_a = _xyz_to_lab(M_SRGB @ want_lin, _xy_to_xyz(*D65))
    lab_b = _xyz_to_lab(xyz_d, _xy_to_xyz(*D65))
    r["dE"] = float(np.linalg.norm(lab_a - lab_b))
    r["lab_in"] = lab_a
    r["lab_out"] = lab_b
    return r


def run(tg, tc, dg, dc, verbose):
    # display_leg_nb / gpu_display_leg 内部读**全局** display_space()，
    # 所以必须真的把它改掉，否则 --display 只是改了期望值、被测对象纹丝不动
    # —— 第一版就栽在这里，报出 38 的假 ΔE。
    _saved = cl._DISPLAY_SPACE
    cl._DISPLAY_SPACE = (dg, dc, f"指纹工具指定 {dg}/{dc}")
    try:
        return _run(tg, tc, dg, dc, verbose)
    finally:
        cl._DISPLAY_SPACE = _saved


def _run(tg, tc, dg, dc, verbose):
    print(f"  目标空间（文件）= {tg}/{tc}     显示器空间 = {dg}/{dc}")
    print()
    print(f"  {'色块':<10}{'②解码':>9}{'③CST':>9}{'④显示端':>10}{'⑤ ΔE76':>10}   判读")
    print("  " + "-" * 68)
    worst = {"lin": (0, ""), "target": (0, ""), "disp": (0, ""), "dE": (0, "")}
    rows = []
    for nm, rgb in PRIMARIES + CHECKER:
        r = fingerprint_one(rgb, tg, tc, dg, dc)
        rows.append((nm, rgb, r))
        for k in worst:
            if r[{"lin": "lin_err", "target": "target_err",
                  "disp": "disp_err", "dE": "dE"}[k]] > worst[k][0]:
                worst[k] = (r[{"lin": "lin_err", "target": "target_err",
                               "disp": "disp_err", "dE": "dE"}[k]], nm)

    # 只打印偏得最多的几个 + 全部中性
    rows.sort(key=lambda t: -t[2]["dE"])
    shown = rows[:6] + [x for x in rows if x[1][0] in NEUTRALS and x not in rows[:6]]
    for nm, rgb, r in shown:
        flag = "✅" if r["dE"] < 1.0 else ("⚠" if r["dE"] < 3.0 else "❌")
        print(f"  {nm:<10}{r['lin_err']:>9.2e}{r['target_err']:>9.2e}"
              f"{r['disp_err']:>10.2e}{r['dE']:>10.2f}   {flag}")
    print("  …")
    print()
    print(f"  最大偏差：② 解码 {worst['lin'][0]:.2e}（{worst['lin'][1]}）  "
          f"③ CST {worst['target'][0]:.2e}（{worst['target'][1]}）")
    print(f"            ④ 显示端 {worst['disp'][0]:.2e}（{worst['disp'][1]}）  "
          f"⑤ ΔE76 {worst['dE'][0]:.2f}（{worst['dE'][1]}）")

    # 定位：第一个超阈的级
    stage = None
    if worst["lin"][0] > 1e-5:
        stage = f"② 解码（曲线不一致，最大 {worst['lin'][0]:.2e}）"
    elif worst["target"][0] > 2e-3:
        stage = f"③ CST（矩阵/白点/缩限不一致，最大 {worst['target'][0]:.2e}）"
    elif worst["disp"][0] > 2e-3:
        stage = f"④ 显示端（与直接转换不一致，最大 {worst['disp'][0]:.2e}）"
    elif worst["dE"][0] > 3.0:
        stage = f"⑤ 绝对颜色（ΔE76 {worst['dE'][0]:.2f}，链路自洽但结果偏）"
    print()
    if stage:
        print(f"  ❌ **第一个出问题的环节：{stage}**")
        return False
    print(f"  ✅ 五级全部通过 —— 文件到显示器码值这一段自洽且正确")
    print(f"     ⚠ 但「显示器码值 → 面板实际发光」不在本工具范围内（需色度计）")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="sRGB", help="管线输出（文件）空间色域名")
    ap.add_argument("--display", default=None, help="显示器空间；缺省=自动读系统档案")
    ap.add_argument("--all", action="store_true", help="跑多种组合")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()

    if a.display:
        dg, dc = a.display, cl._CURVE_MATCH.get(a.display, "sRGB")
        note = "手动指定"
    else:
        dg, dc, note = cl.display_space()
    print(f"  显示器空间：{dg}/{dc}   （{note}）")
    print()

    tg = a.target
    tc = cl._CURVE_MATCH.get(tg, "sRGB")
    if a.all:
        combos = [(tg, tc, dg, dc),
                  ("Rec.709", "Rec.709", dg, dc),
                  ("Display P3", "sRGB", dg, dc),
                  ("sRGB", "sRGB", "Display P3", "sRGB"),
                  ("sRGB", "sRGB", "Adobe RGB", "Gamma 2.2")]
        ok = True
        for t1, t2, d1, d2 in combos:
            print("  " + "=" * 72)
            ok = run(t1, t2, d1, d2, a.verbose) and ok
            print()
    else:
        ok = run(tg, tc, dg, dc, a.verbose)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
