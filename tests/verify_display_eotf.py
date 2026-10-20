#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""全链路验证 C 方案：显示端 + 导出 ICC 是否同时正确（有效性）且一致（同一性）。

三条要同时成立：
  ① 有效性：CL 显示链——Rec.709 内容应落到 **BT.1886(γ2.4)**，不再是 OETF 逆
  ② 有效性：导出 ICC 的 TRC 也应是 BT.1886
  ③ 同一性：CL 显示 == 第三方按 ICC 规范解码 —— 两处必须给出**同一个**亮度

① 和 ② 只做一半就是「用有效性换同一性」，③ 就是检这个。
"""
import sys, io, struct
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import colorlab as cl      # noqa: E402

VS = np.array([0.02, 0.05, 0.10, 0.18, 0.30, 0.50, 0.70, 0.90], np.float32)
_, sd_dec = cl.CURVE[cl.display_space()[1]]


def cl_display_lum(curve):
    """CL 真实显示链：码值 → 显示器亮度。"""
    a = np.stack([VS, VS, VS], 1)[None, :, :].astype(np.float32)
    snap = {"lut": True, "cd": {"t": "3d"}, "tg": "sRGB", "tc": curve,
            "lo": "sRGB", "loc": curve}
    out = np.asarray(cl.display_leg_nb(a, snap), np.float64)[0, :, 0]
    return np.asarray(sd_dec(out.astype(np.float32)), np.float64)


def icc_trc_curve(gamut, curve):
    """从**导出的 ICC 字节**里取出 TRC，还原成 码值→线性光 的函数。"""
    b = cl._icc_bytes(gamut, curve)
    assert b, "ICC 生成失败"
    n = struct.unpack(">I", b[128:132])[0]
    tags = {}
    for i in range(n):
        sig, ofs, sz = struct.unpack(">4sII", b[132 + i*12:144 + i*12])
        tags[sig.decode("latin1")] = (ofs, sz)
    ofs, sz = tags["rTRC"]
    sig = b[ofs:ofs+4]
    if sig == b"para":
        fn, = struct.unpack(">H", b[ofs+8:ofs+10])
        npar = {0: 1, 1: 3, 2: 4, 3: 5, 4: 7}[fn]
        vals = [x/65536.0 for x in struct.unpack(">" + "i"*npar, b[ofs+12:ofs+12+4*npar])]
        # 只处理我们关心的两类：type 0（纯幂）与 type 3（分段）
        if fn == 0:
            g = vals[0]
            return lambda v: np.power(np.asarray(v, np.float64), g), f"para(0) γ={g}"
        if fn == 3:
            g, a, bb, c, d = vals
            return (lambda v: np.where(np.asarray(v) < d,
                                       np.asarray(v)/c,
                                       ((np.asarray(v) + bb)/a) ** (1.0/g))), \
                   f"para(3) [{g:.4f},{a:.4f},{bb:.4f},{c:.4f},{d:.5f}]"
    raise AssertionError(f"未预期的 TRC 类型 {sig!r}")


print("══ 1. 映射表 ══")
for c in ("sRGB", "Gamma 2.2", "Gamma 2.6", "Rec.709", "Rec.2020", "S-Log3"):
    print(f"   {c:<10} → 显示 EOTF = {cl.display_decode_curve(c)}")

print()
print("══ 2. 有效性：CL 显示链 vs BT.1886(γ2.4) ══")
print(f"  {'码值 v':>8}{'CL 显示':>11}{'BT.1886':>11}{'偏差':>10}   判读")
print("  " + "-" * 56)
want = np.asarray(VS, np.float64) ** 2.4
got = cl_display_lum("Rec.709")
ok_all = True
for i, v in enumerate(VS):
    d = got[i] - want[i]
    ok = abs(d) < 2e-3
    ok_all = ok_all and ok
    print(f"  {v:>8.2f}{got[i]:>11.5f}{want[i]:>11.5f}{d:>10.5f}   {'✅' if ok else '❌'}")
print(f"  → CL 显示链{'✅ 已等于 BT.1886' if ok_all else '❌ 仍不等于 BT.1886'}")

print()
print("══ 3. 有效性 + 同一性：导出 ICC 的 TRC ══")
fn, desc = icc_trc_curve("sRGB", "Rec.709")
print(f"   rTRC = {desc}")
icc_lum = fn(VS)
print()
print(f"  {'码值 v':>8}{'CL 显示':>11}{'ICC 解码':>11}{'差':>10}{'vs BT.1886':>12}   判读")
print("  " + "-" * 62)
same = True
for i, v in enumerate(VS):
    dd = abs(icc_lum[i] - got[i])
    dv = abs(icc_lum[i] - want[i])
    same = same and dd < 2e-3
    print(f"  {v:>8.2f}{got[i]:>11.5f}{icc_lum[i]:>11.5f}{dd:>10.6f}{dv:>12.6f}"
          f"   {'✅' if dd < 2e-3 and dv < 2e-3 else '❌'}")
print(f"  → 同一性{'✅ CL 显示 == 第三方按 ICC 解码' if same else '❌ 两者不一致（掉到 B 了）'}")

print()
print("══ 4. 自洽空间必须不受影响（不能误伤）══")
for c in ("sRGB", "Gamma 2.2", "Gamma 2.6", "Gamma 1.8"):
    if c not in cl.CURVE:
        continue
    _, dec = cl.CURVE[c]
    w = np.asarray(dec(VS.astype(np.float32)), np.float64)
    g = cl_display_lum(c)
    print(f"   {c:<12} 与自身曲线最大差 {np.abs(g-w).max():.2e}   "
          f"{'✅' if np.abs(g-w).max() < 2e-3 else '❌'}")

print()
print("══ 5. 描述名必须保持不变（否则 icc_to_space 认不出自家文件）══")
import tempfile
tp = Path(tempfile.mkdtemp()) / "roundtrip.jpg"
from PIL import Image
im = Image.new("RGB", (8, 8), (128, 128, 128))
im.save(tp, "JPEG", icc_profile=cl._icc_bytes("sRGB", "Rec.709"))
g2, c2, note = cl.icc_to_space(str(tp))
print(f"   写 Rec.709 内容档案 → 读回 {g2}/{c2}   {note}")
print(f"   → {'✅ 往返一致' if g2 == 'sRGB' else '❌ 认不出了'}")
