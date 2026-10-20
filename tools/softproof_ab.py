#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""软打样 A/B 对比图：让"显示器空间到底有没有影响"变成**看得见**的事。

为什么要它：用户说"我没感觉到什么色彩的差异"。当时显示器被认成 sRGB，
显示链是恒等变换 —— **确实什么都没发生**。但下次换了宽色域，光靠"感觉"
同样判断不了对错。这个工具把差异**画出来**。

## 两块内容，各管一件事

1. **六色原色块**（R G B C M Y）—— 看**幅度**。
   饱和原色正落在色域边界上，裸值直接送屏 vs 正确转换，差得最狠。
2. **24 色图**（X-Rite ColorChecker Classic）—— 看**方向**，客观锚点。
   你知道真实 ColorChecker 长什么样；B 那边应该更像它。
3. 控制台还给一张**数值表**（每块的 ΔE76）。**主观看 + 客观量，两条腿。**

## A 和 B 分别是什么

    A = 裸值直接送屏（显示器被当成 sRGB 时的样子）
    B = 按目标显示器空间正确转换（走 App 同一条显示链）

- 显示器**真的是 sRGB** → A == B（ΔE≈0）。说明没有软打样可做，也没做错。
- 显示器是**宽色域且处于原生模式** → A 过饱和、B 正确。**A≠B 就说明有区别**。
- 显示器是宽色域但**开了 sRGB 模拟** → A == B，同样无需转换。

## ⚠ 它的能力边界（比功能本身更重要）

这图**只回答一个问题**：「显示器空间到底有没有影响？」—— 它**不能判对错**，
更不能告诉你**哪个环节**出了问题。

    看到 A ≠ B  →  只能说明"两者不同"          ✗ 不是"B 对"
    看到 A == B →  只能说明"没有转换发生"        ✗ 不是"链路正确"

原因很硬：A 和 B **都是我们渲染的**，没有第三方基准。屏幕上哪个"对"，
只有色度计能说。**这一点 PS 的软打样也一样** —— 它同样只是"读显示器档案然后信它"。

**要判对错、要定位环节，用 `tools/chain_fingerprint.py`。**
它把「文件→屏幕」拆成 5 级，每级给实际值 vs 独立期望值，
并直接报出**第一个出问题的环节**；第⑤级用 ColorChecker 的公开参考值做绝对判据。

用法：
    python tools/softproof_ab.py                 # 自动读系统档案
    python tools/softproof_ab.py "Display P3"    # 指定目标显示器空间
    python tools/softproof_ab.py "Adobe RGB" --out 对比.png
"""
import sys
import os
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import colorlab as cl                            # noqa: E402
from PIL import Image, ImageDraw, ImageFont      # noqa: E402

# ---- X-Rite ColorChecker Classic 的 sRGB 参考值（BabelColor 公开数据，8bit）----
# 标准排列：1-6 第一行，7-12 第二行，13-18 第三行，19-24 第四行
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


# ---------------------------------------------------------------- 色彩计算
def _srgb_to_xyz(rgb):
    """sRGB 8bit → XYZ(D65)。用我们自己的解码曲线与矩阵，保证与 App 同源。"""
    dec = cl.CURVE["sRGB"][1]
    M = np.asarray(cl._mat("sRGB")[0], np.float64)
    lin = np.asarray(dec(np.asarray(rgb, np.float32) / 255.0), np.float64)
    return M @ lin


def _xyz_to_lab(xyz, wp=(0.3127, 0.3290)):
    w = np.array([wp[0] / wp[1], 1.0, (1 - wp[0] - wp[1]) / wp[1]])
    t = xyz / w
    e, k = 216 / 24389, 24389 / 27
    f = np.where(t > e, np.cbrt(t), (k * t + 16) / 116)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def delta_e(a_rgb, b_rgb):
    return float(np.linalg.norm(_xyz_to_lab(_srgb_to_xyz(a_rgb))
                                - _xyz_to_lab(_srgb_to_xyz(b_rgb))))


def convert(colors, dg, dc):
    """把一批 sRGB 8bit 值转换到目标显示器空间 —— 走 App 同一条 Pipe。"""
    a = (np.asarray(colors, np.float32).reshape(1, -1, 3)) / 255.0
    out = np.asarray(cl.Pipe.go(a.astype(np.float32), "sRGB", "sRGB", dg, dc,
                                ootf="off"), np.float64)
    return np.clip(out[0] * 255.0, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- 画图
CELL, GAP, PAD, HEAD = 92, 8, 16, 34
COLS, ROWS_C = 6, 4
FONTS = [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf",
         r"C:\Windows\Fonts\arial.ttf"]


def _font(size):
    for p in FONTS:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def _draw_panel(img, x0, y0, title, get_rgb, f_t, f_s):
    d = ImageDraw.Draw(img)
    d.text((x0, y0), title, font=f_t, fill=(255, 255, 255))
    y = y0 + HEAD
    for i, (nm, _) in enumerate(PRIMARIES):
        px = x0 + i * (CELL + GAP)
        d.rectangle([px, y, px + CELL, y + CELL], fill=get_rgb(i, True))
        d.text((px + 4, y + CELL + 2), nm, font=f_s, fill=(200, 200, 200))
    y += CELL + 28
    for i, (nm, _) in enumerate(CHECKER):
        r, c = divmod(i, COLS)
        px = x0 + c * (CELL + GAP)
        py = y + r * (CELL + 24)
        d.rectangle([px, py, px + CELL, py + CELL], fill=get_rgb(i, False))
        d.text((px + 3, py + CELL + 2), nm, font=f_s, fill=(180, 180, 180))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("space", nargs="?", default=None,
                    help="目标显示器空间（色域名）；缺省=自动读系统档案")
    ap.add_argument("--out", default=None, help="输出文件名（落在 output/）")
    args = ap.parse_args()

    if args.space:
        dg = args.space
        dc = cl._CURVE_MATCH.get(dg, "sRGB")
        src = "手动指定"
    else:
        dg, dc, src = cl.display_space()

    print(f"  目标显示器空间：{dg} / {dc}    （{src}）")
    if (dg, dc) == ("sRGB", "sRGB"):
        print()
        print("  ⚠ 目标就是 sRGB —— A 和 B 会**完全一样**，这张图看不出差别。")
        print("    这不是 bug：显示器被认作 sRGB 时，显示链本来就不该改任何像素。")
        print("    想看出来，指定一个宽色域：")
        print('        python tools/softproof_ab.py "Display P3"')

    # ---- 数值表（先给量，再看图）----
    rows = []
    for nm, rgb in PRIMARIES + CHECKER:
        b = tuple(int(v) for v in convert([rgb], dg, dc)[0])
        rows.append((nm, rgb, b, delta_e(rgb, b)))
    rows.sort(key=lambda r: -r[3])
    print()
    print(f"  {'色块':<10}{'A 裸值':<20}{'B 转换后':<20}{'ΔE76':>7}")
    print("  " + "-" * 60)
    for nm, a, b, e in rows[:12]:
        print(f"  {nm:<10}{str(a):<20}{str(b):<20}{e:>7.2f}")
    es = [r[3] for r in rows]
    print(f"  …（共 {len(rows)} 块）最大 ΔE76 = {max(es):.2f}（{rows[0][0]}）  "
          f"平均 = {np.mean(es):.2f}  非零块数 = {sum(1 for e in es if e > 0.5)}")

    # ---- 出图 ----
    pw = COLS * (CELL + GAP)
    W = PAD * 2 + pw * 2 + 40
    H = PAD * 2 + HEAD + CELL + 28 + ROWS_C * (CELL + 24) + 10
    img = Image.new("RGB", (W, H), (18, 18, 18))
    f_t, f_s = _font(19), _font(12)

    def a_rgb(i, is_prim):
        return PRIMARIES[i][1] if is_prim else CHECKER[i][1]

    _b_prim = convert([c for _, c in PRIMARIES], dg, dc)
    _b_chk = convert([c for _, c in CHECKER], dg, dc)

    def b_rgb(i, is_prim):
        v = _b_prim[i] if is_prim else _b_chk[i]
        return tuple(int(x) for x in v)

    _draw_panel(img, PAD, PAD, "A｜裸值直接送屏（现在看到的样子）", a_rgb, f_t, f_s)
    _draw_panel(img, PAD + pw + 40, PAD,
                f"B｜转换到 {dg}/{dc}（正确的 sRGB 观感）", b_rgb, f_t, f_s)

    od = Path(cl.__file__).parent / "output"
    od.mkdir(exist_ok=True)
    name = args.out or f"软打样AB_{dg.replace('.', '')}_{dc.replace('.', '')}.png"
    fp = od / name
    img.save(fp)
    print(f"\n  已写出：{fp}")
    return str(fp)


if __name__ == "__main__":
    main()
