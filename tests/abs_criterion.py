#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""绝对判据：声明为 S 的码值 v，显示器上**应该**是多少亮度 vs 我们**实际**给出多少。

为什么必须有这个（见 `色彩管线-输出与显示.md` §三、§四）：

  一致性（CL 显示 == 第三方显示）**推不出**正确性。当两处错得一模一样时
  （例如 LUT 输出是 Rec.709，但 CL 和文件 ICC 都按 sRGB 处理），
  "互相比较"的差值是 **0** —— **相对参照永远发现不了这类错误**。
  只有绝对判据能穿透它。

判据来源：内容被设计成在**它自己的显示端标准**下观看（那个标准决定显示亮度）。
  · sRGB、纯 gamma（2.2/2.6/1.8）→ 编码与显示互逆 → 显示意图 = **自身曲线**
  · Rec.709 / Rec.2020（相机侧 OETF）→ 显示端是 **BT.1886（γ2.4）**
  · 场景 Log → **没有唯一显示意图**（渲染含创作判断）→ 只能报"不可判断"

⚠️ 「应为」那一列本身依赖**采用的约定**（见下）。不同约定给出不同的数。
   这份工具的价值是：把约定显式化，并把偏差**量化**，而不是让它藏在"看着差不多"里。
"""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import colorlab as cl      # noqa: E402

SELF = ("self", None)
INTENT = {
    "sRGB": SELF, "Gamma 2.2": SELF, "Gamma 2.6": SELF, "Gamma 1.8": SELF,
    "Rec.709": ("bt1886", 2.4), "Rec.2020": ("bt1886", 2.4),
    "S-Log3": ("none", None), "V-Log": ("none", None), "ARRI LogC3": ("none", None),
    "RED Log3G10": ("none", None), "ACEScct": ("none", None), "Linear": ("none", None),
}

VS = np.array([0.02, 0.05, 0.10, 0.18, 0.30, 0.50, 0.70, 0.90])
_dg, _dc = cl.display_space()[:2]
_, _disp_decode = cl.CURVE[_dc]          # 本机显示器的 EOTF：显示码值 → 显示亮度


def _measure(curve, gamut):
    """把码值走一遍**真实显示链**，返回显示器上的亮度。

    注意：管线输出是**码值**；显示链把它转成显示器码值；最后显示器自己的 EOTF
    才给出亮度。早先这里把已解码的线性值又过了一次显示器 EOTF（重复解码），
    结果自洽空间也报出偏差。
    """
    a = np.stack([VS, VS, VS], 1)[None, :, :].astype(np.float32)
    # 4.0 简化后：**文件空间 = 目标（tg/tc）**，lo/loc 只是 LUT 输出的声明。
    # 要测"文件里是 curve 空间"，就得把它设成**目标**。
    snap = {"lut": True, "cd": {"t": "3d"}, "tg": gamut, "tc": curve,
            "lo": gamut, "loc": curve}
    out = np.asarray(cl.display_leg_nb(a, snap), np.float64)[0, :, 0]
    return np.asarray(_disp_decode(out.astype(np.float32)), np.float64)


def run():
    print("  绝对判据：码值 → 显示器上应有的亮度（相对白点） vs 实测")
    print(f"  本机显示器 = {_dg}/{_dc}")
    print()
    print(f"  {'曲线':<14}{'显示意图':<14}{'v=0.50 应有':>13}{'实测':>10}{'偏差':>10}   判读")
    print("  " + "-" * 78)
    bad = []
    for curve, (kind, gamma) in INTENT.items():
        if curve not in cl.CURVE:
            continue
        if kind == "self":
            _, dec = cl.CURVE[curve]
            want = np.asarray(dec(VS.astype(np.float32)), np.float64)
            got = _measure(curve, _dg)
            d = float(np.abs(got - want).max())
            w5, g5 = float(want[5]), float(got[5])
            ok = d < 2e-3
            if not ok:
                bad.append(curve)
            print(f"  {curve:<14}{'自身（自洽）':<14}{w5:>13.4f}{g5:>10.4f}{g5-w5:>10.4f}"
                  f"   {'✅ 0 偏差' if ok else '❌'}")
        elif kind == "bt1886":
            want = VS ** gamma
            got = _measure(curve, _dg)
            w5, g5 = float(want[5]), float(got[5])
            dev = abs(g5 / max(w5, 1e-9) - 1.0)
            if dev < 0.02:
                print(f"  {curve:<14}{'BT.1886 γ2.4':<14}{w5:>13.4f}{g5:>10.4f}{g5-w5:>10.4f}"
                      f"   ✅ 已对齐（偏差 {dev*100:.1f}%）")
            else:
                bad.append(curve)
                print(f"  {curve:<14}{'BT.1886 γ2.4':<14}{w5:>13.4f}{g5:>10.4f}{g5-w5:>10.4f}"
                      f"   ❌ 偏亮 {100*(g5/max(w5,1e-9)-1):.0f}%")
        else:
            print(f"  {curve:<14}{'无（需渲染）':<14}{'—':>13}{'—':>10}{'—':>10}   ⛔ 显示端不可判断")

    print()
    print("  ── Rec.709 逐码值（相机侧 OETF 被当成显示曲线用的偏差分布）──")
    print(f"  {'码值 v':>8}{'应为(BT.1886)':>15}{'我们给出':>11}{'倍数':>9}{'偏差':>10}")
    _, rdec = cl.CURVE["Rec.709"]
    for v in (0.05, 0.10, 0.18, 0.30, 0.49, 0.70, 0.90):
        w = v ** 2.4
        g = float(rdec(np.float32(v)))
        print(f"  {v:>8.2f}{w:>15.4f}{g:>11.4f}{g/w:>9.2f}{100*(g/w-1):>9.0f}%")
    print()
    print("  签名：**暗部/中间调偏亮最多，高光基本对** —— 典型的「γ 用错」特征。")
    print("  18% 灰附近（BT.1886 下码值 ≈0.49）：应给 0.180，实际给 0.250，偏亮 39%。")
    print()
    print("  ⚠️ 「应为」这一列依赖采用的约定：这里取「内容按自己的显示端标准观看」")
    print("     （Rec.709 → BT.1886）。换一套约定数值会不同 —— 但那正是必须被")
    print("     **显式选定**的东西，而不是默认藏在实现里。")
    return bad


if __name__ == "__main__":
    run()
