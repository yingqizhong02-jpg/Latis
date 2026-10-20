#!/usr/bin/env python3
# colorlab 2.1 —— 色彩空间转换 + LUT + 基础调色，带实时预览
import sys, os, re, threading, json
from collections import OrderedDict
from pathlib import Path
import numpy as np
import customtkinter as ctk
from PIL import Image, ImageOps, ImageTk, ImageCms, ImageFilter, ImageDraw
try:
    import tifffile
    _TIFF = tifffile
except Exception:
    _TIFF = None
try:
    import pillow_heif
    pillow_heif.register_heif_opener()   # 让 Pillow 打开 .heif/.heic
    _HEIF = True
except Exception:
    _HEIF = False
try:
    import rawpy
    _RAW = rawpy
except Exception:
    _RAW = None
from tkinter import filedialog, messagebox, Canvas, ttk

Image.MAX_IMAGE_PIXELS = None  # 本地可信大图：关闭解压炸弹像素上限

APP_VERSION = "4.0"   # 改动请同步写进 CHANGELOG.md（回归测试第 12 组会校验两者一致）

# ---- 消除「以脚本方式运行时被二次导入」这个坑 ----
# colorlab_nb 里有 `import colorlab as _cl`，而 colorlab 又 `import colorlab_nb`：循环导入。
# 直接 `python colorlab.py` 时本模块叫 __main__，于是 colorlab_nb 那句 import 会把
# colorlab.py **再执行一遍**（第二份模块对象）。第二份执行到模块级代码时 _nb 还是半成品，
# 任何 `_nb.xxx` 都会炸 → 被 colorlab_nb 的 try/except 吞掉 → `_cl` 从未绑定 →
# 之后凡是 temp≠0 的调色就 NameError（**非 RAW 预览必然踩中**，RAW 因为 tp 归零而侥幸逃过）。
# 把 "colorlab" 这个名字指向 __main__ 本体即可根除，顺带避免 CURVE/_GC_MODE 出现两份。
if __name__ == "__main__":
    sys.modules.setdefault("colorlab", sys.modules["__main__"])

# ---- numba 加速内核（默认启用，numba 缺失或 COLORLAB_NUMPY=1 时回退 numpy）----
try:
    import colorlab_nb as _nb
    _NB = os.environ.get("COLORLAB_NUMPY", "0") != "1"
except Exception:
    _nb = None
    _NB = False

# ---- CUDA 后端（4.0 新增；纯可选，导入失败/无设备/编译失败一律静默回退 CPU）----
# COLORLAB_BACKEND = cupy | numba | numpy   （默认 auto：能上 GPU 就上）
_BACKEND_PREF = os.environ.get("COLORLAB_BACKEND", "auto").lower()
try:
    import colorlab_gpu as _gpu
except Exception:
    _gpu = None
_GPU_BE = None          # 惰性创建：kernel 编译有开销，且失败要能被捕获
_GPU_TRIED = False
_GPU_ERR = ""
_GPU_PREC = os.environ.get("COLORLAB_GPU_PREC", "float64")


def gpu_backend():
    """返回已就绪的 CUDA 后端；不可用返回 None（绝不抛异常）。"""
    global _GPU_BE, _GPU_TRIED, _GPU_ERR
    if _BACKEND_PREF in ("numba", "numpy", "off") or _gpu is None:
        return None
    if _GPU_TRIED:
        return _GPU_BE
    _GPU_TRIED = True
    try:
        if not _gpu.available():
            _GPU_ERR = _gpu.last_error(); return None
        _GPU_BE = _gpu.Backend(_GPU_PREC)
    except Exception as e:
        _GPU_ERR = f"{type(e).__name__}: {e}"; _GPU_BE = None
    return _GPU_BE


def backend_name():
    return "cupy" if gpu_backend() is not None else ("numba" if _NB else "numpy")


_GPU_FAILS = 0
def _note_gpu_error(e):
    """运行期 GPU 出错：记一次并回退 CPU。连续 3 次就直接放弃 GPU，
    免得每一帧都抛异常、每帧都白试一遍（工具要能一直用下去，不能变砖）。"""
    global _GPU_FAILS, _GPU_ERR, _GPU_BE
    _GPU_FAILS += 1
    _GPU_ERR = f"{type(e).__name__}: {e}"
    if _GPU_FAILS >= 3:
        _GPU_BE = None


def gpu_status():
    """给界面/日志用：(后端名, 精度, 说明)。"""
    be = gpu_backend()
    if be is None:
        return ("numba" if _NB else "numpy"), "-", (_GPU_ERR or "未启用")
    if _GPU_FAILS:
        return "cupy", _GPU_PREC, f"降级中（已出错 {_GPU_FAILS} 次: {_GPU_ERR}）"
    return "cupy", _GPU_PREC, ""

# ==================== 色彩科学内核（沿用 2.0，仅改 float32） ====================
CURVE = {}
def _reg(n, e, d):
    if callable(e) or callable(d):
        CURVE[n] = (e, d)
R = _reg
R("sRGB", lambda x: np.where(x <= .0031308, x*12.92, 1.055*x**(1/2.4) - .055),
          lambda x: np.where(x <= .04045, x/12.92, ((x + .055)/1.055)**2.4))
R("Rec.709", lambda x: np.where(x < .018, x*4.5, 1.099*x**.45 - .099),
             lambda x: np.where(x < .081, x/4.5, ((x + .099)/1.099)**(1/.45)))
_SL3th = 171.2102946929/1023
R("S-Log3", lambda x: np.where(x >= .01125, (420 + np.log10((x + .01)/(.18 + .01))*261.5)/1023,
                                (x*(171.2102946929 - 95)/.01125 + 95)/1023),
            lambda x: np.where(x >= _SL3th, (10**((x*1023 - 420)/261.5))*(.18 + .01) - .01,
                               (x*1023 - 95)*.01125/(171.2102946929 - 95)))
_LC3 = (.010591, 5.555556, .052272, .247190, .385537, 5.367655, .092809)
_LC3c, _LC3a, _LC3b, _LC3c2, _LC3d, _LC3e, _LC3f = _LC3
# ARRI 官方定义：线性段 y = e*x + f；对数段 y = c*log10(a*x + b) + d
# （旧实现把 e/f 与 c/d 两组系数用反了：x=1.0 得 4.112 而非官方 0.5706，断点处跳变 -5.42）
_LC3et = _LC3e*_LC3c + _LC3f
R("ARRI LogC3", lambda x: np.where(x <= _LC3c, _LC3e*x + _LC3f,
                                   _LC3c2*np.log10(_LC3a*x + _LC3b) + _LC3d),
                lambda x: np.where(x <= _LC3et, (x - _LC3f)/_LC3e,
                                   (10**((x - _LC3d)/_LC3c2) - _LC3b)/_LC3a))
R("V-Log", lambda x: np.where(x < 0.01, 5.6*x + 0.125,
                              0.241514*np.log10(x + 0.00873) + 0.598206),
           lambda x: np.where(x < 0.181, (x - 0.125)/5.6,
                              10**((x - 0.598206)/0.241514) - 0.00873))
_RGa, _RGb, _RGc = .224282, .155954, .386231
_RGl = np.log(10)
R("RED Log3G10", lambda x: np.where(x >= _RGb, _RGc + _RGa*np.log10(x/_RGb),
                                    _RGc + _RGa*(x - _RGb)/(_RGb*_RGl)),
                 lambda x: np.where(x >= _RGc, _RGb*10**((x - _RGc)/_RGa),
                                    _RGb + (x - _RGc)*_RGb*_RGl/_RGa))
# ACEScct（ACES 官方 Log 曲线）。此前 _CURVE_MATCH 把 ACEScct 色域映射到 "ACEScct" 曲线，
# 但该曲线从未注册 → 一旦把源/目标设为 ACEScct 就 KeyError 崩溃。
_CCT_X, _CCT_Y = 0.0078125, 0.155251141552511
_CCT_A, _CCT_B = 10.5402377416545, 0.0729055341958355
R("ACEScct", lambda x: np.where(x <= _CCT_X, _CCT_A*x + _CCT_B,
                                (np.log2(np.maximum(x, 1e-12)) + 9.72)/17.52),
              lambda x: np.where(x <= _CCT_Y, (x - _CCT_B)/_CCT_A,
                                 2.0**(x*17.52 - 9.72)))
for g in (1.8, 2.2, 2.4, 2.6):
    R(f"Gamma {g}", lambda x, g_=g: np.maximum(x, 0)**(1/g_),
                    lambda x, g_=g: np.maximum(x, 0)**g_)
R("Linear", lambda x: x, lambda x: x)
_prio = ["S-Log3", "V-Log", "ARRI LogC3", "RED Log3G10", "ACEScct", "sRGB", "Rec.709", "Gamma 2.2", "Gamma 2.4", "Linear"]
CURVE_NAMES = [n for n in _prio if n in CURVE] + [n for n in CURVE if n not in _prio]

_WP_XYZ = {"D65": (0.3127, 0.3290), "D60": (0.32168, 0.33767), "D50": (0.34567, 0.35850),
           "DCI": (0.3140, 0.3510)}      # DCI 白点（SMPTE RP 431-2）——注意不是 D50
def _wpv(n):
    x, y = _WP_XYZ[n]
    return np.array([x/y, 1.0, (1-x-y)/y])
_GAMUT_DEFS = [("sRGB", (0.6400, 0.3300), (0.3000, 0.6000), (0.1500, 0.0600), "D65"),
               ("S-Gamut3.Cine", (0.7660, 0.2750), (0.2250, 0.8000), (0.0890, -0.0870), "D65"),
               ("V-Gamut", (0.7300, 0.2800), (0.1650, 0.8400), (0.1000, -0.0300), "D65"),
               ("ARRI WG", (0.6840, 0.3130), (0.2210, 0.6970), (0.0860, -0.1010), "D65"),
               ("Display P3", (0.6800, 0.3200), (0.2650, 0.6900), (0.1500, 0.0600), "D65"),
               ("DCI-P3", (0.6800, 0.3200), (0.2650, 0.6900), (0.1500, 0.0600), "DCI"),
               ("Rec.709", (0.6400, 0.3300), (0.3000, 0.6000), (0.1500, 0.0600), "D65"),
               ("Rec.2020", (0.7080, 0.2920), (0.1700, 0.7970), (0.1310, 0.0460), "D65"),
               ("Adobe RGB", (0.6400, 0.3300), (0.2100, 0.7100), (0.1500, 0.0600), "D65"),
               ("ProPhoto RGB", (0.7347, 0.2653), (0.1596, 0.8404), (0.0366, 0.0001), "D50"),
               ("REDWideGamut", (0.7800, 0.3040), (0.3250, 0.8700), (0.0970, -0.0720), "D65"),
               ("S-Gamut3", (0.7300, 0.2800), (0.1400, 0.8550), (0.1000, -0.0500), "D65"),
               ("ACEScg", (0.7130, 0.2930), (0.1650, 0.8300), (0.1280, 0.0440), "D60"),
               ("ACEScct", (0.7130, 0.2930), (0.1650, 0.8300), (0.1280, 0.0440), "D60"),
               ("ACES2065-1", (0.7347, 0.2653), (0.0000, 1.0000), (0.0001, -0.0770), "D60")]
GAMUT_NAMES = [d for d, _, _, _, _ in _GAMUT_DEFS]
_GAMUT = {d: {"r": r, "g": g, "b": b, "wp": wp} for d, r, g, b, wp in _GAMUT_DEFS}
# 各目标色域的默认编码曲线。注意：这里必须**符合该色域的标准定义**，
# 否则导出的像素本身就不是那个空间的东西（ICC 再怎么标也救不回来）。
# 4.0 修正：Adobe RGB 标准是**纯 gamma 2.2**、ProPhoto/ROMM 是**纯 gamma 1.8**，
# 此前两者都被错误地映射成 sRGB 分段曲线 —— 与 Adobe/微软的权威档案互转时
# 分别差 6 LSB 和 19 LSB，改后归零。
_CURVE_MATCH = dict(zip(GAMUT_NAMES, ["sRGB", "S-Log3", "V-Log", "ARRI LogC3", "sRGB", "Gamma 2.6",
                                      "Rec.709", "Rec.709", "Gamma 2.2", "Gamma 1.8", "RED Log3G10", "S-Log3",
                                      "Linear", "ACEScct", "Linear"]))
# RGB→XYZ 矩阵统一由 _mat() 从各色域原色坐标+白点自动计算（保证白点一致，修复 DCI-P3 D50/D65 冲突）
_m2x = {n: None for n in GAMUT_NAMES}
_ws = {n: _wpv(_GAMUT[n]["wp"]) for n in GAMUT_NAMES if n in _GAMUT}
_MCACHE = {}
def _mat(g):
    if g in _MCACHE: return _MCACHE[g]
    M = _m2x.get(g)
    if M is None:
        w = _GAMUT[g]; R, G, B, wp = w["r"], w["g"], w["b"], w["wp"]
        Xr, Yr, Zr = R[0]/R[1], 1, (1-R[0]-R[1])/R[1]
        Xg, Yg, Zg = G[0]/G[1], 1, (1-G[0]-G[1])/G[1]
        Xb, Yb, Zb = B[0]/B[1], 1, (1-B[0]-B[1])/B[1]
        M = np.array([[Xr, Xg, Xb], [Yr, Yg, Yb], [Zr, Zg, Zb]])
        S = np.linalg.solve(M, _wpv(wp))
        M = M * S   # 每列乘 S（缩放各原色 XYZ 使白点匹配）
    M = np.asarray(M, dtype=np.float32)
    _MCACHE[g] = (M, np.linalg.inv(M).astype(np.float32),
                  _ws.get(g, _wpv(_GAMUT[g]["wp"])).astype(np.float32))
    return _MCACHE[g]
_BM = np.array([[0.8951, 0.2664, -0.1614], [-0.7502, 1.7135, 0.0367], [0.0389, -0.0685, 1.0296]], dtype=np.float32)
_BMi = np.linalg.inv(_BM).astype(np.float32)
def _cat(s, t):
    return _BMi @ np.diag((_BM @ t) / (_BM @ s)) @ _BM
def _roll(r, k=0.9):
    """DaVinci 式色域缩限：亮度 _shoulder 滚降 + 色度超界沿色相压缩回 [0,1]，柔和 roll-off，保持色相。

    修复：不越界像素原样保留（此前对全图应用 Ys 滚降会改变界内亮度）。"""
    mx = np.max(r, axis=-1); mn = np.min(r, axis=-1)
    bad = (mx > 1.0) | (mn < 0.0)
    if not bad.any():
        return r
    Y = 0.2126*r[..., 0] + 0.7152*r[..., 1] + 0.0722*r[..., 2]
    Ys = _shoulder(Y, k)                       # 亮度软滚降
    s_mx = np.where(mx > 1.0, (1.0 - Ys)/np.maximum(mx - Y, 1e-6), 1.0)
    s_mn = np.where(mn < 0.0, Ys/np.maximum(Y - mn, 1e-6), 1.0)
    s = np.clip(np.minimum(s_mx, s_mn), 0.0, 1.0)
    t = 1.0 - s
    s = 1.0 - t*t*(3 - 2*t)                    # smoothstep roll-off
    out = Ys[..., None] + (r - Y[..., None]) * s[..., None]
    out = np.where(bad[..., None], out, r)     # 不越界像素原样保留
    return np.clip(out, 0, 1)
def _dither(a, tile=1024):
    """8-bit 输出抖动：加 ±0.5 LSB 随机噪声量化，隐藏色阶断层（10-bit 源尤其有效）。

    分块生成噪声：旧实现一次性开 size=a.shape 的随机数组，33MP 时 float64 中间量约 786MB、
    float32 结果约 393MB，是导出环节的内存尖峰。"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    out = np.empty(a.shape, dtype=np.uint8)
    rng = np.random.default_rng()
    for i in range(0, a.shape[0], tile):
        blk = np.clip(a[i:i + tile], 0, 1)*255 + 0.5
        blk += rng.uniform(-0.5, 0.5, size=blk.shape).astype(np.float32)
        out[i:i + tile] = np.clip(blk, 0, 255).astype(np.uint8)
    return out

_LOG_CURVES = {"S-Log3", "V-Log", "ARRI LogC3", "RED Log3G10", "ACEScct"}

# ---- 显示 EOTF：内容被设计成在**什么 EOTF** 下观看 ----
# 「编码曲线」与「显示 EOTF」是两件事，只在 sRGB 下重合（sRGB 规范里编码就是 EOTF 的逆）。
#
#   · 自洽（编码与显示互逆）：sRGB、纯 gamma（2.2/2.4/2.6/1.8）→ 显示 EOTF = 自身
#   · **视频 OETF**（相机侧「光→信号」）：BT.709 / BT.2020 → 显示端是 **BT.1886**
#     BT.1886: L = a(V+b)^γ, γ=2.4；Lb=0 时 b=0 → 归一化后就是 **V^2.4**
#     即我们已有的 `Gamma 2.4` 曲线，无需新增。
#   · 场景 Log：没有唯一显示意图（需要渲染，含创作判断）→ 不在表里，保持自身
#
# 少了这一层，Rec.709 内容会被按系统伽马 ≈1.0 显示，而它的规范意图是 ≈1.08~1.2 ——
# 实测暗部偏亮 2~53 倍（v=0.18 偏亮 2.9x）。签名：暗部最差、高光基本对。
_DISPLAY_EOTF = {"Rec.709": "Gamma 2.4", "Rec.2020": "Gamma 2.4"}


def display_cannot_judge(curve):
    """该内容在显示端**能不能被正确渲染**？

    场景参考（Log）内容做不到：场景→显示的渲染**没有唯一解** ——
    色域超出怎么压、白点怎么适应、观看环境补偿多少，全是创作判断。
    所以显示端只能如实展示，**必须标注"不可据此判断成片"**，并说明
    「要看渲染结果，渲染必须进输出管线（进管线就进文件，才保持同一性）」。

    自洽曲线与视频 OETF 内容**能**正确显示（有 _DISPLAY_EOTF 兜住），返回 False。
    """
    c = display_decode_curve(curve)
    return c in _LOG_CURVES or c == "Linear"


def display_decode_curve(curve):
    """把「内容的编码曲线」换成「它该用哪条 EOTF 解码」。

    显示端与导出 ICC 的 TRC **都必须**走这里 —— 两处不一致就会破坏同一性
    （CL 显示与第三方读同一文件结果不同）。
    """
    return _DISPLAY_EOTF.get(curve, curve)
# 显示参考曲线（**按曲线名，不按色域名**）。参考类型由曲线唯一决定：
#   Log 曲线   = 场景参考（scene-referred，相机光编码）
#   显示曲线   = 显示参考（display-referred，可直接上屏）
# 色域名推不出参考类型 —— Rec.2020+Gamma2.4 与 Rec.2020+V-Log 是同一个色域的两种参考。
# 显示参考曲线。**Gamma 2.4 不能漏**：它就是 BT.1886（Lb=0 时 L=V^2.4），
# 是 Rec.709 内容的显示端。漏了它 _ref_type 会返回 unknown，
# 于是 ootf="auto" 误判成"源不是显示参考"而多加一次正向 OOTF（实测差 7%）。
_DISPLAY_CURVES = {"sRGB", "Rec.709", "Gamma 2.2", "Gamma 2.4", "Gamma 2.6",
                   "Gamma 1.8", "Gamma 1.0"}


def _ref_type(curve):
    """曲线的参考类型：'scene' / 'display' / 'unknown'。"""
    if curve in _LOG_CURVES:
        return "scene"
    if curve in _DISPLAY_CURVES:
        return "display"
    return "unknown"
# 显示参考（display-referred）空间 + 各空间 OOTF 指数（L_d = L_s^γ，环境亮度补偿）
_DISPLAY_REF = {"sRGB", "Display P3", "DCI-P3", "Rec.709", "Rec.2020", "Adobe RGB"}
# OOTF 指数（L_d = L_s^γ，**系统伽马**）。
#
# ⚠ 别和「显示伽马」搞混 —— 差一倍，改错方向比不改更糟：
#     · 显示伽马：码值 → 屏幕光。sRGB 曲线 ≈2.2；Rec.709 内容走 BT.1886 γ2.4
#       （见 _DISPLAY_EOTF）。它是**曲线**，不是这个表。
#     · 系统伽马（本表）：场景光 → 屏幕光。SDR 链 OETF γ0.45 × EOTF γ2.4 ≈ 1.08；
#       BT.2100（HDR）规范值 ≈1.2。**量级是 1 附近，不是 2.2。**
#
# 曾加过"观看环境"三档缩放（影院暗房/参考半暗/桌面明亮）。**已删除**：
# 对桌面修图工具，"影院暗房"这类档位用不上，且真实环境补偿还要看绝对亮度、
# 周围视野占比，各家模型并不一致 —— 多一个控件加一套没有依据的模型，是过度设计。
_OOTF_GAMMA = {"sRGB": 1.1, "Display P3": 1.1, "Rec.709": 1.2, "Rec.2020": 1.2,
               "DCI-P3": 1.2, "Adobe RGB": 1.1}

def _ootf(x, gam):                        # 场景光 → 显示光（正向 OOTF）
    return np.power(np.maximum(x, 0.0), gam)

def _inv_ootf(y, gam):                    # 显示光 → 场景光（反向 OOTF）
    return np.power(np.maximum(y, 0.0), 1.0/gam)

# ==================== 色域缩限：ACES Reference Gamut Compression (RGC) ====================
# 依据：ACES 官方规范 https://docs.acescentral.com/rgc/specification/
#       参考实现 ampas/aces-vwg-gamut-mapping-2020/model/gamut_compress.ctl
#       （即 ACES 1.3 的 LMT.Academy.GamutCompress）
#
# 为什么换掉旧的 _roll：
#   1) RGC 是公开标准：逐通道 threshold（zone of trust 内完全不动）+ 逐通道 limit
#      + 有理压缩曲线，limit 处正好压到色域边界；_roll 是自研启发式。
#   2) 旧 Pipe.go 写成 `if tc not in _LOG_CURVES: _roll(...)`，
#      于是 Log 目标（＝RAW→Log→LUT 这条工作流）整段跳过缩限，最后被 _al 的 np.clip 硬裁
#      —— 正是"橙色点集被压成一个点"的那个问题。
# 规范原文提醒：保护区之外的"高饱和但仍在色域内"的颜色也会被轻微移动，这是 RGC 的既定行为。
_GC_MODE = os.environ.get("COLORLAB_GC", "rgc").strip().lower()   # rgc（默认）| roll（旧行为）
_RGC_LIM = (1.147, 1.264, 1.312)      # 逐通道 limit：R=cyan, G=magenta, B=yellow
_RGC_THR = (0.815, 0.803, 0.880)      # 逐通道 threshold（zone of trust 边界）
_RGC_PWR = 1.2

def _rgc_scale(l, t, p=_RGC_PWR):
    """规范 Equation 4 里的 s = (l - t) / (((1 - t)/(l - t))^(-p) - 1)^(1/p)"""
    return (l - t) / (((1.0 - t) / (l - t)) ** (-p) - 1.0) ** (1.0 / p)

_RGC_S = tuple(_rgc_scale(l, t) for l, t in zip(_RGC_LIM, _RGC_THR))

def gamut_compress(rgb):
    """ACES RGC 正向压缩（规范 Step 2–5）。输入/输出为同一线性 RGB 空间，逐像素、逐通道。"""
    a = np.asarray(rgb, dtype=np.float32)
    A = a.max(axis=-1)
    Aa = np.abs(A)[..., None]
    z = Aa == 0.0
    safe = np.where(z, np.float32(1.0), Aa)
    dn = np.where(z, np.float32(0.0), (A[..., None] - a) / safe)      # Step 3
    dc = np.empty_like(dn)
    for c in range(3):
        t, s = _RGC_THR[c], _RGC_S[c]
        d = dn[..., c]
        out = d.copy()
        m = d >= t                                                    # Step 4：保护区内不动
        if np.any(m):
            nd = (d[m] - t) / s
            out[m] = t + s * nd / (1.0 + nd ** _RGC_PWR) ** (1.0 / _RGC_PWR)
        dc[..., c] = out
    return A[..., None] - dc * safe                                   # Step 5

def gamut_decompress(rgb):
    """ACES RGC 逆运算（规范 Equation 4b），用于验证可逆性。"""
    a = np.asarray(rgb, dtype=np.float32)
    A = a.max(axis=-1)
    Aa = np.abs(A)[..., None]
    z = Aa == 0.0
    safe = np.where(z, np.float32(1.0), Aa)
    dn = np.where(z, np.float32(0.0), (A[..., None] - a) / safe)
    dc = np.empty_like(dn)
    for c in range(3):
        t, s = _RGC_THR[c], _RGC_S[c]
        d = dn[..., c]
        out = d.copy()
        m = (d >= t) & (d <= t + s)                                   # 超过 limit 的原样返回
        if np.any(m):
            nd = (d[m] - t) / s                                       # 注意：这里是压缩后的归一化值
            dp = nd ** _RGC_PWR
            ok = dp < 1.0                                             # 与参考实现一致，避免负数开根
            idx = np.flatnonzero(m)[ok]
            if idx.size:
                dpk = dp[ok]
                out.ravel()[idx] = t + s * (dpk / (1.0 - dpk)) ** (1.0 / _RGC_PWR)
        dc[..., c] = out
    return A[..., None] - dc * safe

class Pipe:
    @staticmethod
    def go(r, sg, sc, tg, tc, k=0.9, ootf="auto", gc=None):
        r = np.asarray(r, dtype=np.float32)
        if sg == tg and sc == tc:
            return r   # 完全一致的转换直接返回，省掉整个管线
        _, d = CURVE[sc]; l = d(r)          # 反向 OETF → scene 线性
        # auto：参考类型看**曲线**，不看色域（见 _ref_type 的说明）
        if ootf == "reverse" or (ootf == "auto" and _ref_type(sc) == "display"
                                 and _ref_type(tc) != "display"):
            l = _inv_ootf(l, _OOTF_GAMMA.get(sg, 1.2))  # display→scene 反向 OOTF（剥离补偿）
        M, _, W = _mat(sg); x = l @ M.T
        _, _, W2 = _mat(tg)
        if not np.allclose(W, W2, atol=1e-6):
            x = x @ _cat(W, W2).T
        _, Mi, _ = _mat(tg); l2 = x @ Mi.T
        # 色域缩限（scene 线性下做，避免 OOTF 后饱和度突变）
        # 旧实现只在"非 Log 目标"下做自研 _roll，Log 目标（正是 LUT 工作流）整段跳过 → 硬裁。
        # 现默认对所有目标都用 ACES RGC；COLORLAB_GC=roll 可退回旧行为做对比。
        # gc=2 表示不缩限：给显示器转换用（纯色度，避免二次映射）。
        _gcm = (2 if gc == 2 else (1 if _GC_MODE == "roll" else 0)) if gc is not None else \
               (1 if _GC_MODE == "roll" else 0)
        if _gcm == 2:
            pass
        elif _gcm == 1:
            if tc not in _LOG_CURVES:
                l2 = _roll(l2, k)
        else:
            l2 = gamut_compress(l2)
        if ootf == "forward" or (ootf == "auto" and _ref_type(tc) == "display"
                                 and _ref_type(sc) != "display"):
            l2 = _ootf(l2, _OOTF_GAMMA.get(tg, 1.2))   # scene→display 显式正向 OOTF（加补偿）
        e, _ = CURVE[tc]
        return e(l2)                        # 正向 EOTF（编码输出信号）

# ==================== LUT 解析（修复：1D LUT 死代码 + 支持 1D） ====================
_CP_CACHE = OrderedDict()   # (path, mtime_ns, size) -> 解析结果；小 LRU，避免堆内存
_CP_CACHE_MAX = 4
_CP_HDR = re.compile(r"(?im)^[ \t]*(?:TITLE|LUT_3D_SIZE|LUT_1D_SIZE|DOMAIN_MIN|DOMAIN_MAX|#)[^\n]*$")


def _cp_parse(p):
    """解析 .cube。**数据块一次性交给 numpy**（C 速度），头用正则定位。

    旧实现对数据块逐行 `s.split()` + `[float(x) for x in ...]` + `list.append`，
    实测 275K 行要 358ms、35K 行要 59ms，而且跑在 UI 线程上 —— 选一次卡一次。
    """
    d3 = s1 = 0
    ti = ""
    mn, mx = [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]
    with open(p, encoding="utf-8", errors="replace") as f:
        raw = f.read()
    pos = 0
    for m in _CP_HDR.finditer(raw):
        pos = m.end()
        s = m.group(0).strip()
        u = s.upper()
        if u.startswith("TITLE"):
            mt = re.match(r'TITLE\s+"(.+)"', s)
            if mt:
                ti = mt.group(1)
        elif u.startswith("LUT_3D_SIZE"):
            d3 = int(s.split()[-1])
        elif u.startswith("LUT_1D_SIZE"):
            s1 = int(s.split()[-1])
        elif u.startswith("DOMAIN_MIN"):
            q = s.split(); mn = [float(q[1]), float(q[2]), float(q[3])]
        elif u.startswith("DOMAIN_MAX"):
            q = s.split(); mx = [float(q[1]), float(q[2]), float(q[3])]
    body = raw[pos:]
    arr = np.fromstring(body, dtype=np.float32, sep=" ") if body.strip() else np.zeros(0, np.float32)
    if d3:
        need = d3 * d3 * d3 * 3
        if arr.size != need:
            # 有的 LUT 每个数单独一行、或行尾带注释；退回宽松路径，但只走一次
            tk = body.split()
            arr = np.array(tk, dtype=np.float32) if tk else np.zeros(0, np.float32)
        if arr.size != need:
            raise Exception(f"LUT 3D 数据量不符: 期望 {need} 个数值 实际 {arr.size}")
        d = arr.reshape(d3, d3, d3, 3).transpose(2, 1, 0, 3)
        return {"t": "3d", "n": d3, "d": d, "ti": ti, "mn": mn, "mx": mx}
    if s1:
        a = arr
        if a.ndim == 1:
            a = a[:, None]
        if a.shape[1] == 3:
            if len(a) == s1: d = a.T
            else: raise Exception("1D LUT RGB 数据量不符")
        else:
            v = a[:, 0]
            if len(v) == s1: d = np.tile(v, (3, 1))
            elif len(v) == 3 * s1: d = v.reshape(3, s1)
            else: raise Exception("1D LUT 单值数据量不符")
        return {"t": "1d", "n": s1, "d": d, "ti": ti, "mn": mn, "mx": mx}
    raise Exception("LUT 缺少 LUT_3D_SIZE / LUT_1D_SIZE")


def _cp(p):
    """带缓存的 .cube 读取。重复选同一个 LUT 时瞬时返回。"""
    try:
        st = os.stat(p)
        key = (str(p), int(st.st_mtime_ns), int(st.st_size))
    except Exception:
        return _cp_parse(p)
    hit = _CP_CACHE.get(key)
    if hit is not None:
        _CP_CACHE.move_to_end(key)
        return hit
    d = _cp_parse(p)
    _CP_CACHE[key] = d
    while len(_CP_CACHE) > _CP_CACHE_MAX:      # 只留最近几个，别把几十 MB 数组堆在内存里
        _CP_CACHE.popitem(last=False)
    return d


# ==================== LUT 应用（自研 _al / lut_fast，唯一路径） ====================
# 曾经的 ffmpeg lut3d 备选路径已整体移除。原因：本机根本没装 ffmpeg，UI 却提供"套LUT引擎"
# 选项，选中后静默退回自研 —— 既不科学也不诚实；而且那条路是未分块全图处理（内存更差），
# 与自研路径的数值也不一致（预览与导出会出两种结果）。
def _al(img, inf):
    img = np.asarray(img, dtype=np.float32)
    # 输入已是 0~1（工具编码域），_al3d/_al1d 按 0~1 网格索引，勿按 DOMAIN 缩放
    # （DOMAIN 0-255 时若除 255 会把输入压到黑区，导致查表错误）
    img = np.clip(img, 0.0, 1.0)
    if inf["t"] == "3d":
        out = _al3d(img, inf["d"], inf["n"])
    else:
        out = _al1d(img, inf["d"])
    if inf["d"].max() > 1.01:          # LUT 数据为整数域（0~255/0~1023）时缩回 0~1
        out = out / inf["d"].max()
    return np.clip(out, 0.0, 1.0)

def _al3d(img, L, N):
    s = float(N - 1)
    c = np.clip(img, 0, 1)*s
    i0 = np.floor(c).astype(np.int32)
    i1 = np.minimum(i0 + 1, N - 1)
    f = c - i0
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    r0, g0, b0 = i0[..., 0], i0[..., 1], i0[..., 2]
    r1, g1, b1 = i1[..., 0], i1[..., 1], i1[..., 2]
    c000 = L[r0, g0, b0]; c100 = L[r1, g0, b0]; c010 = L[r0, g1, b0]; c110 = L[r1, g1, b0]
    c001 = L[r0, g0, b1]; c101 = L[r1, g0, b1]; c011 = L[r0, g1, b1]; c111 = L[r1, g1, b1]
    o = np.empty_like(img)
    for m, w0, v0, w1, v1, w2, v2, w3, v3 in [
        ((r >= g) & (g >= b), 1 - r, c000, r - g, c100, g - b, c110, b, c111),
        ((r >= b) & (b > g), 1 - r, c000, r - b, c100, b - g, c101, g, c111),
        ((g > r) & (r >= b), 1 - g, c000, g - r, c010, r - b, c110, b, c111),
        ((g >= b) & (b > r), 1 - g, c000, g - b, c010, b - r, c011, r, c111),
        ((b > r) & (r >= g), 1 - b, c000, b - r, c001, r - g, c101, g, c111),
        ((b > g) & (g > r), 1 - b, c000, b - g, c001, g - r, c011, r, c111)]:
        if not m.any(): continue
        o[m] = v0[m]*w0[m, None] + v1[m]*w1[m, None] + v2[m]*w2[m, None] + v3[m]*w3[m, None]
    return o

def _al1d(img, d):
    N = d.shape[1]; sl = float(N - 1)
    out = np.empty_like(img)
    for c in range(3):
        L = d[c]
        idx = np.clip(img[..., c], 0, 1)*sl
        i0 = np.floor(idx).astype(np.int32)
        i1 = np.minimum(i0 + 1, N - 1)
        f = idx - i0
        out[..., c] = L[i0] + (L[i1] - L[i0])*f
    return out

# ==================== 基础调色（在线性域做，避免编码域乘法的失真） ====================
def _shoulder(x, k, knee=None):
    """高光软肩：knee 以上平滑压向 1.0，knee 处斜率=1 无抬升。k 为滚降强度（上限放宽到 3）"""
    x = np.asarray(x, dtype=np.float32)
    if knee is None:
        knee = 0.9 - 0.12*max(0.0, float(k))   # k=1→0.78, k=3→0.54，强度越大膝越低
    knee = max(0.35, min(0.95, knee))
    over = np.maximum(x - knee, 0.0)
    c = 1.0 / (1.0 - knee)                          # 保证 knee 处斜率=1，平滑无抬升
    comp = (over / (over + 1.0/c)) * (1.0 - knee)
    y = np.where(x > knee, knee + comp, x)
    return np.minimum(y, 1.0)

def _grade(a, work, ev=0.0, contrast=0.0, sat=0.0, temp=0.0):
    """显示域调色：解码到线性 → 曝光/对比/饱和/色温 → 编码回去。

    注：早期的「亮度 + 滚降」（内部 shoulder 软肩）已整条删除。"""
    if not (ev or contrast or sat or temp):
        return a
    a = a.astype(np.float32, copy=False)
    _, d = CURVE[work]; e, _ = CURVE[work]
    lin = d(a)   # 解码到工作空间线性域
    if ev:
        lin = lin * np.float32(2.0**ev)      # 曝光：纯线性
    if contrast:
        lin = (lin - np.float32(0.18)) * np.float32(1.0 + contrast) + np.float32(0.18)
    if sat:
        Y = 0.2126*lin[..., 0:1] + 0.7152*lin[..., 1:2] + 0.0722*lin[..., 2:3]
        lin = Y + (lin - Y) * np.float32(1.0 + sat)
    if temp:
        kg = _kelvin_gain(float(temp))
        lin[..., 0] *= np.float32(kg[0])
        lin[..., 1] *= np.float32(kg[1])
        lin[..., 2] *= np.float32(kg[2])
    lin = np.maximum(lin, 0.0)   # 负光无物理意义，编码前夹到 0，避免 NaN
    return e(lin)

# ==================== 分块执行（内存可控，兼容大图 / 32 位 Python） ====================
def _tiled(a, fn, tile=1536):
    h = a.shape[0]
    if h <= tile: return fn(a)
    return np.concatenate([fn(a[i:i+tile]) for i in range(0, h, tile)], axis=0)

# ==================== 色彩管理：显示器空间（4.0 新增） ====================
# 此前软件**完全没有色彩管理**：管线算出的「目标空间」像素被直接丢给 Tk 上屏，
# 于是只有"目标恰好等于显示器空间"时才碰巧正确。实测：
#     sRGB 目标   与正确显示转换差 0.0%（1 LSB）—— 因为目标和显示器同为 sRGB
#     DCI-P3 目标 96.5% 像素错、最大 97 LSB、平均偏亮 16 个码值
#     Rec.2020   96.3% 像素错、平均偏暗 13 个码值
# 这正是"选宽色域就发淡/发闷"的根因。
# 正确结构：源 → 调色 → CST(目标) → [LUT] → **目标→显示器转换** → 上屏
# 最后这一级**只服务监看，绝不参与导出**（导出的是目标空间像素 + 对应 ICC profile）。
def _monitor_profile_path():
    """读 Windows 当前显示器的默认 ICC 配置文件路径（失败返回 None）。"""
    try:
        import ctypes
        from ctypes import wintypes
        hdc = ctypes.windll.user32.GetDC(0)
        buf = ctypes.create_unicode_buffer(600); size = wintypes.DWORD(600)
        ok = ctypes.windll.gdi32.GetICMProfileW(hdc, ctypes.byref(size), buf)
        ctypes.windll.user32.ReleaseDC(0, hdc)
        p = buf.value
        return p if ok and p and os.path.exists(p) else None
    except Exception:
        return None


def _icc_primaries(path):
    """从 ICC 取原色/白点的色度。只认矩阵类档案；取不到返回 None。"""
    try:
        import struct
        b = open(path, "rb").read()
        if b[36:40] != b"acsp":
            return None
        n = struct.unpack(">I", b[128:132])[0]
        tab = {}
        for i in range(n):
            sig, ofs, sz = struct.unpack(">4sII", b[132 + i * 12:144 + i * 12])
            tab[sig] = ofs
        def xy(sig):
            ofs = tab.get(sig)
            if ofs is None:
                return None
            X, Y, Z = (v / 65536.0 for v in struct.unpack(">3i", b[ofs + 8:ofs + 20]))
            s = X + Y + Z
            return (X / s, Y / s) if s else None
        r, g, bl, w = xy(b"rXYZ"), xy(b"gXYZ"), xy(b"bXYZ"), xy(b"wtpt")
        if None in (r, g, bl, w):
            return None
        return r, g, bl, w
    except Exception:
        return None


# 显示器空间的判定结果：(gamut, curve, 说明)。惰性求值并缓存。
_DISPLAY_SPACE = None
_DISP_GAMUT = "__DISPLAY__"       # 自定义显示器色域的键（不进 GAMUT_NAMES）


def _icc_trc_gamma(path):
    """档案的 TRC 若是**纯 gamma**就返回它，否则 None。

    只处理两种最常见、且我们能精确表达的写法：
      · curv，count=1 → u8Fixed8 的 gamma
      · para type 0   → (g,)
    其它（分段曲线、采样表）返回 None，调用方按 sRGB 并**如实说明**。
    """
    try:
        b = Path(path).read_bytes()
        nt = int.from_bytes(b[128:132], "big")
        tags = {}
        for i in range(nt):
            sig = b[132 + i*12:136 + i*12].decode("latin1")
            ofs = int.from_bytes(b[136 + i*12:140 + i*12], "big")
            tags[sig] = ofs
        ofs = tags.get("rTRC")
        if ofs is None:
            return None
        sig = b[ofs:ofs+4]
        if sig == b"curv":
            cnt = int.from_bytes(b[ofs+8:ofs+12], "big")
            if cnt == 1:
                return int.from_bytes(b[ofs+12:ofs+14], "big") / 256.0
            return None
        if sig == b"para":
            fn = int.from_bytes(b[ofs+8:ofs+10], "big")
            if fn == 0:
                return int.from_bytes(b[ofs+12:ofs+16], "big", signed=True) / 65536.0
        return None
    except Exception:
        return None


def _register_display_gamut(prim, white_xy):
    """把显示器档案的实测原色/白点注册成自定义色域，返回色域名。

    `prim` 是 ICC 里**已适配到 PCS 白点 D50** 的色料 (r,g,b) xy，
    必须先**反适配**回显示器原生白点，否则 `_mat()` 会二次适配、白点就错了。
    """
    global _DISP_GAMUT, _WP_XYZ
    try:
        # 注意 _wpv() 收的是**名字**不是坐标元组 —— 早先直接传 (x,y) 会 KeyError，
        # 被 except 吞掉后静默返回 None，表现为"原色无法注册"。
        _wx, _wy = float(white_xy[0]), float(white_xy[1])
        _wxyz = np.array([_wx/_wy, 1.0, (1.0-_wx-_wy)/_wy], dtype=np.float64)
        A = np.asarray(_cat(_wpv("D50"), _wxyz), dtype=np.float64)
        nat = []
        for k in range(3):
            v = A @ np.array([prim[k][0], prim[k][1], 1 - prim[k][0] - prim[k][1]])
            t = v.sum()
            nat.append((float(v[0]/t), float(v[1]/t)))
        _WP_XYZ[_DISP_GAMUT] = (float(white_xy[0]), float(white_xy[1]))
        _GAMUT[_DISP_GAMUT] = {"r": nat[0], "g": nat[1], "b": nat[2], "wp": _DISP_GAMUT}
        _MCACHE.pop(_DISP_GAMUT, None)
        return _DISP_GAMUT
    except Exception:
        return None


def display_space():
    """返回 (gamut, curve, 说明)。决定「文件空间 → 屏幕」怎么转。

    4.0：**读显示器档案并真正使用它**（在已知色域表里精确匹配原色）。
    此前只认 sRGB，其它档案一律退回 sRGB —— 在宽色域显示器上会把内容
    压进 sRGB，**超出 sRGB 的颜色在显示端就被裁掉**。

    ⚠ 前提是系统里关联的档案**确实描述了显示器当前的状态**：
      · 装了显示器真实档案（Display P3 / Adobe RGB …），或
      · 开启了 Windows 11 的「自动管理应用颜色 / Advanced Color」
    若关联档案就是 sRGB（Windows 默认），本函数只能解析出 sRGB ——
    **这不是软件的限制，是系统告诉我们的就是 sRGB。**"""
    global _DISPLAY_SPACE
    if _DISPLAY_SPACE is not None:
        return _DISPLAY_SPACE
    p = _monitor_profile_path()
    if not p:
        _DISPLAY_SPACE = ("sRGB", "sRGB", "未读到显示器档案，按 sRGB 监看")
        return _DISPLAY_SPACE
    pr = _icc_primaries(p)
    name = os.path.basename(p)
    if pr is None:
        _DISPLAY_SPACE = ("sRGB", "sRGB", f"显示器档案非矩阵类（{name}），按 sRGB 监看")
        return _DISPLAY_SPACE
    r, g, b, w = pr
    # ICC 的 rXYZ/gXYZ/bXYZ 是**已适配到 PCS 白点 D50** 的色料，不能直接与原生原色比：
    # 实测 sRGB Color Space Profile.icm 的绿原色是 (0.3212,0.5978)，而 sRGB 原生绿是 (0.3000,0.6000)；
    # 把 sRGB 原色 Bradford 适配到 D50 后恰好是 (0.3212,0.5979)。所以基准必须同样先适配。
    #
    # 4.0：**对所有已知色域做匹配，而不是"不是 sRGB 就用 sRGB"**。
    # 原实现在宽色域显示器上会把内容一律压进 sRGB → 超出 sRGB 的颜色
    # **在显示端就被裁掉**，用户看不到（但文件里还在）。
    got = [r, g, b]
    tol = 0.004
    best, best_err = None, 1e9

    def _ref_for(gamut):
        A = np.asarray(_cat(_wpv(_GAMUT[gamut]["wp"]), _wpv("D50")), dtype=np.float64)
        M = np.asarray(_mat(gamut)[0], dtype=np.float64)
        out = []
        for k in range(3):
            v = A @ M[:, k]; t = v.sum()
            out.append((v[0]/t, v[1]/t))
        return out

    for _gname in GAMUT_NAMES:
        if _gname not in _GAMUT:
            continue
        try:
            _ref = _ref_for(_gname)
        except Exception:
            continue
        _e = max(abs(got[i][0]-_ref[i][0]) + abs(got[i][1]-_ref[i][1]) for i in range(3))
        if _e < best_err:
            best, best_err = _gname, _e
    if best is not None and best_err < tol:
        _cv = _CURVE_MATCH.get(best, "sRGB")
        # 白点也要对得上（显示档案的 wtpt 存的是**实际白点**）
        _w = _WP_XYZ.get(_GAMUT[best].get("wp", "D65"), (0.3127, 0.3290))
        if abs(w[0]-_w[0]) < 0.006 and abs(w[1]-_w[1]) < 0.006:
            _DISPLAY_SPACE = (best, _cv, f"显示器 = {best}（{name}）")
        else:
            _DISPLAY_SPACE = ("sRGB", "sRGB",
                              f"⚠ 显示器档案 {name} 的原色像 {best} 但白点不同"
                              f"（{w[0]:.4f},{w[1]:.4f} vs {_w[0]:.4f},{_w[1]:.4f}），按 sRGB 监看")
    elif _register_display_gamut(pr, (w[0], w[1])):
        # 不在已知表里 —— **别退回 sRGB**。真实显示器的实测原色与标准原色本就不同，
        # 精确匹配必然失败；退回去就等于把宽色域显示器压成 sRGB，
        # 用户"看不到 sRGB 以外的颜色"。直接用档案的实测原色。
        _gg = _icc_trc_gamma(p)
        if _gg and abs(_gg - 2.2) < 0.05:
            _cv2 = "Gamma 2.2"
        elif _gg and abs(_gg - 1.8) < 0.05:
            _cv2 = "Gamma 1.8"
        elif _gg and abs(_gg - 2.4) < 0.05:
            _cv2 = "Gamma 2.4"
        elif _gg:
            _cv2 = "Gamma 2.2"
        else:
            _cv2 = "sRGB"
        _width = max(abs(got[k][0]-[(0.6485, 0.3309), (0.3212, 0.5978),
                                    (0.1559, 0.0660)][k][0]) for k in range(3))
        _DISPLAY_SPACE = (_DISP_GAMUT, _cv2,
                          f"显示器 = 档案实测原色（{name}）"
                          + (f"，TRC γ{_gg:.2f}" if _gg else "，TRC 非纯 gamma → 按 sRGB 曲线")
                          + f"；与 sRGB 原色最大偏移 {_width:.3f}")
    else:
        _DISPLAY_SPACE = ("sRGB", "sRGB",
                          f"⚠ 显示器档案 {name} 的原色无法注册（最接近 {best}，"
                          f"偏差 {best_err:.4f}），按 sRGB 监看")
    return _DISPLAY_SPACE


def display_leg_out_space(snap):
    """管线输出（=导出像素）所处的空间。监看转换与导出 ICC **都靠它**。

    · 不套 LUT：就是用户选的 target（tg/tc）。
    · 套 LUT：**代码里无从得知 LUT 的输出空间**（transform 只把数据送进它的输入空间，
      之后 tg/tc 就不参与运算了），所以由用户在界面上声明（lo / loc）。

      曾经硬编码按 sRGB 处理。对多数风格化 LUT 尚可（Rec.709 与 sRGB 同色料、
      只差曲线），但**胶片模拟类常常输出 Rec.709 曲线甚至 Log** —— 按 sRGB 监看
      会明显错，Log 尤其（发灰、发平）。而且导出的 ICC 会跟着说谎。

    4.0 简化（用户指出）：**「目标」永远就是文件最终所处的空间**。
    套 LUT 时管线末端会补一级 CST（lo/loc → tg/tc），所以出口恒为 (tg, tc)。
    `lo/loc` 不再声称"文件是什么"，只声明"LUT 吐出来的是什么" —— 它是
    那一级 CST 的**起点**。

    想要 LUT 输出的原值？把「目标」设成 LUT 输出的空间即可 ——
    那一级 CST 自动变成恒等。
    """
    return (snap.get("tg", "sRGB"), snap.get("tc", "sRGB"))


def pipeline_out_space(snap):
    """LUT 吐出来的数据所处的空间 = 末端那一级 CST 的**起点**（= 用户声明的 lo/loc）。

    不套 LUT 时没有这一级，文件直接就是 (tg, tc)。
    """
    return (snap.get("lo") or "sRGB", snap.get("loc") or "sRGB")


def out_cst_gain(snap):
    """开二次 CST 时，源色域比目标色域宽多少（用于裁切警告）。

    返回 (需要警告?, 说明)。色域相同 → 无损（如 Rec.709 → sRGB，同 BT.709 色料）。
    """
    if not (snap.get("lut") and snap.get("cd") is not None):
        return False, ""            # 不套 LUT 就没有这一级
    fg, fc = pipeline_out_space(snap)
    tg, tc = display_leg_out_space(snap)
    if fg == tg and fc == tc:
        return False, ""
    if fg not in _GAMUT or tg not in _GAMUT:
        return False, ""
    try:
        # 源色域的三个原色在目标色域下的坐标：任一分量 >1 就说明目标装不下
        M = np.linalg.solve(np.asarray(_mat(tg)[0], np.float64),
                            np.asarray(_mat(fg)[0], np.float64))
        worst = float(np.max(M))
    except Exception:
        return False, ""
    if worst <= 1.0 + 1e-6:
        return False, f"{fg} 完全落在 {tg} 内 —— 无损"
    return True, (f"⚠ {fg} 的原色超出 {tg}（{worst:.3g}×）—— 超色域的颜色会被**裁切**，"
                  f"不可逆。若这是你要的交付空间，请确认；否则改用同色料的空间")


def display_leg_nb(a, snap):
    """目标空间 → 显示器空间（numpy/numba 路径）。纯色度 + 裁剪，不做色域映射、不加 OOTF。

    **解码用内容的「显示 EOTF」，不是它的编码曲线**（见 display_decode_curve）：
    内容码值 --E_c--> 显示亮度 --逆(E_d)--> 显示器码值。Rec.709 的 E_c 是 BT.1886(γ2.4)。
    """
    sg, sc = display_leg_out_space(snap)
    sc = display_decode_curve(sc)
    dg, dc = display_space()[:2]
    if sg == dg and sc == dc:
        return a
    if sc not in CURVE or dc not in CURVE or sg not in _GAMUT or dg not in _GAMUT:
        return a
    if _NB:
        return _nb.pipe_fast(a, sg, sc, dg, dc, ootf="off", gc=2)
    return Pipe.go(a, sg, sc, dg, dc, ootf="off", gc=2)



# ---- 色域警告（4.0）----
# 判据：目标空间 → 显示器空间之后、**夹取之前**，任一通道越出 [0,1]，
# 就说明这个颜色在你的监看上被裁掉了 —— 你看到的不是真色。
# 阈值取「半个 8bit 码值」而不是「只要越界」：夹取量小于半码时输出本来也不会变，
# 标出来只会让浮点噪声把大片像素误染。
_GW_TH = 0.5 / 255.0
_GW_RGB = np.array([255, 0, 255], dtype=np.uint8)      # 惯例的品红


def gamut_mask(a):
    """从未夹取的显示器空间像素求「超出显示器色域」掩码，返回 (H,W) bool。"""
    return ((a < -_GW_TH) | (a > 1.0 + _GW_TH)).any(axis=2)


# ==================== 线性域预览工具（4.0） ====================
# 预览每帧的处理像素预算。按 33MP≈474ms 的管线速率折算，2.0MP ≈ 29ms。
# 调小 → 拖滑块更跟手但预览更软；调大 → 更清晰但每帧更慢。
# 可用环境变量 COLORLAB_PV_BUDGET 覆盖（单位：像素个数，如 1200000）。
def _pv_budget():
    try:
        return max(1.0e5, float(os.environ.get("COLORLAB_PV_BUDGET", 2.0e6)))
    except Exception:
        return 2.0e6
_PV_PX_BUDGET = _pv_budget()

def _down_lin(a, size):
    """线性域面积平均降采样（只缩不放）。size=(w,h)。

    **必须在线性域做。** 对 gamma 编码值取平均会系统性压暗（Jensen 不等式）：
    实测把 sRGB 域平均的结果解码回线性，相对误差中位 13%，且 100% 的像素偏低。
    此前预览的 `_make_small` 与 `_process_region` 都是在 8bit sRGB 上做 LANCZOS，
    属于典型的 gamma 域重采样错误。
    """
    ow, oh = int(size[0]), int(size[1])
    h, w = a.shape[0], a.shape[1]
    if ow >= w or oh >= h:
        return a
    if _NB:
        return _nb._box_down(np.ascontiguousarray(a, dtype=np.float32), oh, ow)
    out = np.empty((oh, ow, 3), dtype=np.float32)     # 无 numba 时的兜底（慢但域正确）
    for k in range(3):
        im = Image.fromarray(np.ascontiguousarray(a[..., k]), mode="F")
        out[..., k] = np.asarray(im.resize((ow, oh), Image.LANCZOS), dtype=np.float32)
    return out


def _lin_to_u8(a):
    """线性 float32 → 8bit sRGB。**仅用于屏幕显示**，不参与管线运算。

    只对**线性**数据成立。非 RAW 输入（JPEG/TIFF/PNG/HEIF）的预览源是**文件原生编码**
    （JPEG 就是 sRGB），对它再编码一次就是双重 gamma —— 实测偏亮 168%。
    那种情况下用 _src_to_u8()，它按源的实际编码分派。"""
    x = np.clip(a, 0.0, 1.0)
    if _NB:
        enc = _nb._encode_srgb(np.ascontiguousarray(x, dtype=np.float32))
    else:
        e, _ = CURVE["sRGB"]; enc = e(x)
    return (np.asarray(enc, dtype=np.float32) * 255 + 0.5).astype(np.uint8)


def _src_to_u8(a, curve):
    """把预览源数据转成 8bit 屏幕显示，按源的真实编码分派。

    curve == "Linear"（RAW 解码层）→ 需要编码到 sRGB 才能正确显示；
    其它（非 RAW 的文件原生编码）→ 原样量化，与 3.3 的显示行为一致。"""
    if curve == "Linear":
        return _lin_to_u8(a)
    return (np.clip(a, 0.0, 1.0) * 255 + 0.5).astype(np.uint8)


# ==================== 文件扫描 / 配置 ====================
def _folder_input(folder):
    """严格匹配第一层「色域-曲线」专属文件夹 → (色域, 曲线)；未命中返回 (None, None)"""
    for g in GAMUT_NAMES:
        c = _CURVE_MATCH.get(g, "")
        if folder == f"{g}-{c}":
            return g, c
    return None, None

def _ensure_lut_folders():
    """预先设立所有「色域-曲线」专属文件夹，用户加 LUT 无需自建"""
    d = Path(__file__).parent / "lut"
    d.mkdir(exist_ok=True)
    for g in GAMUT_NAMES:
        c = _CURVE_MATCH.get(g, "")
        if c:
            (d / f"{g}-{c}").mkdir(exist_ok=True)

def _sl():
    _ensure_lut_folders()
    d = Path(__file__).parent / "lut"
    if not d.is_dir(): return []
    r = []
    for f in sorted(d.rglob("*.cube")):
        n = str(f.relative_to(d))[:-5].replace(os.sep, "/")   # 含子目录的相对名，避免重名
        li = None
        try:
            rel = f.relative_to(d)
            if len(rel.parts) > 1:        # 期望输入唯一由第一层「色域-曲线」文件夹决定，不猜文件名
                g, _ = _folder_input(rel.parts[0])
                if g: li = g
        except Exception:
            pass
        r.append({"path": str(f), "name": n, "li": li})
    # 显示名：默认本体名（橙色高亮）；重名时用含路径的相对名保证唯一
    stems = {}
    for l in r:
        stems.setdefault(Path(l["path"]).stem, []).append(l)
    for base, items in stems.items():
        for l in items:
            l["disp"] = base if len(items) == 1 else l["name"]
    return r

def _orient(a, o):
    if o == 2: return a[:, ::-1]                                # FLIP_LEFT_RIGHT
    if o == 3: return a[::-1, ::-1]                             # ROTATE_180
    if o == 4: return a[::-1]                                   # FLIP_TOP_BOTTOM
    if o == 5: return np.transpose(a, (1, 0, 2))                # TRANSPOSE
    if o == 6: return np.rot90(a, 3, (1, 0))                    # ROTATE_270
    if o == 7: return np.transpose(a[:, ::-1], (1, 0, 2))       # TRANSVERSE
    if o == 8: return np.rot90(a, 1, (1, 0))                    # ROTATE_90
    return a

def _is_raw(rp):
    return Path(rp).suffix.lower() in _RAW_EXTS

# ---- 高光重建 ----
# 传感器各通道各自饱和：一个通道先到顶、其余还在涨时，这块高光的颜色比例就崩了
# （绿先卡住 → 品红高光；红先卡住 → 青）。重建就是用**未裁切的通道**去估算被裁的那个。
# 注意 rawpy 的默认是 Clip（最粗糙），必须显式指定，否则就是默默用最差档。
_HL_MODES = ("重建（推荐）", "混合", "裁切（最差）", "不裁（会超界）")
_HL_RAW = {"重建（推荐）": "ReconstructDefault", "混合": "Blend",
           "裁切（最差）": "Clip", "不裁（会超界）": "Ignore"}
# 默认用重建：高光回到中性（实测高光区饱和度 0.082 → 0.031），且最大值仍能到 1.0。
# Blend 的最大值只到 0.74，被压住了，更差。
#
# **重要权衡（实测）**：LibRaw 的 -H 不只重建高光，它还抬高了内部 maximum，
# 于是 scale_mul 变小 → **整幅图被拉暗约 2.23 倍**。证据：按亮度分档，从最暗
# (0~0.01) 到次亮 (0.5~0.85) 的 Recon/Clip 比值恒为 **0.448**，与高光无关；
# 只有最亮档 (0.85+) 是 0.631 —— 那才是重建真正起作用的地方。
# 这是 dcraw/LibRaw 的实现方式，不是本工具引入的。想要旧观感就切回「裁切（最差）」。
_HL_DEF = "重建（推荐）"
_HL_MODE = _HL_DEF                # 解码参数，全图一致：预览与导出共用同一个值


def _hl_enum():
    """当前高光重建模式对应的 rawpy 枚举；拿不到返回 None（则不传该参数）。"""
    try:
        return getattr(_RAW.HighlightMode, _HL_RAW.get(_HL_MODE, "Blend"))
    except Exception:
        return None


_RAW_WORK_DEF = "ACES2065-1"   # RAW 内部宽工作空间默认（AP0，D60，比 ProPhoto 更宽）
_RAW_WORK_OPT = {"ACES2065-1": _RAW.ColorSpace.ACES if _RAW else None,
                 "Rec.2020": _RAW.ColorSpace.Rec2020 if _RAW else None,
                 "ProPhoto RGB": _RAW.ColorSpace.ProPhoto if _RAW else None}
_KG_CACHE = {}
_KG_REF = None
# ==================== 相机双光源标定矩阵（可选增强） ====================
# 为什么只拿这个：DNG 的 ColorMatrix1/2 是**相机自身的标定数据**（XYZ→相机，
# 与 rawpy 的 rgb_xyz_matrix 同约定），按色温插值是物理上正确的做法。
# ACR 也这么做，不是因为它有品味，而是因为**单矩阵在远离基准光源时有固有色相误差**。
#
# 明确**不要**的东西：HueSatMap / LookTable / Adobe Standard 曲线 —— 那是 Adobe 的
# 审美（2 万多个不透明数字），按机型逐台调，没有可辩护的物理依据。
# 把它们塞进来等于把 CL 变成 ACR 前端，正好违背这个工具存在的意义。
_DCP_ON = os.environ.get("COLORLAB_DCP", "1").strip() not in ("0", "off", "no", "")
_DCP_CACHE = {}          # (make, model) -> profile dict | None
_MODEL_CACHE = {}        # path -> (make, model)

# EXIF/DNG CalibrationIlluminant → 色温(K)
_ILLUM_K = {1: 5500.0, 2: 4150.0, 3: 2856.0, 4: 5500.0, 17: 2856.0, 18: 4874.0,
            19: 6774.0, 20: 5503.0, 21: 7504.0, 22: 5003.0, 23: 3200.0, 24: 2856.0}


def _shot_model(path):
    """RAW 的 (Make, Model)。tifffile 读 IFD0，结果缓存。"""
    p = str(path)
    if p in _MODEL_CACHE:
        return _MODEL_CACHE[p]
    mk = md = ""
    try:
        import tifffile
        with tifffile.TiffFile(p) as tf:
            t = tf.pages[0].tags
            if "Make" in t: mk = str(t["Make"].value).strip()
            if "Model" in t: md = str(t["Model"].value).strip()
    except Exception:
        pass
    _MODEL_CACHE[p] = (mk, md)
    return mk, md


def _dcp_read(path, make, model):
    """读一个 .dcp，返回 {'M1','M2','T1','T2'}；不合机型或没有双矩阵则 None。"""
    try:
        import tifffile
        with tifffile.TiffFile(path) as tf:
            t = tf.pages[0].tags
            ucm = t.get(0xC614)
            if ucm is not None:
                want = ("%s %s" % (make, model)).strip().lower()
                if str(ucm.value).strip().lower() != want:
                    return None

            def _mat(tag):
                tk = t.get(tag)
                if tk is None:
                    return None
                a = np.asarray(tk.value, np.float64).ravel()
                if a.size == 18:                       # 9 个有理数 [num,den]
                    a = a.reshape(9, 2)
                    a = a[:, 0] / np.where(a[:, 1] == 0, 1.0, a[:, 1])
                if a.size != 9:
                    return None
                return a.reshape(3, 3)

            m1, m2 = _mat(0xC621), _mat(0xC622)
            if m1 is None or m2 is None:
                return None
            i1, i2 = t.get(0xC65A), t.get(0xC65B)
            k1 = _ILLUM_K.get(int(i1.value)) if i1 is not None else None
            k2 = _ILLUM_K.get(int(i2.value)) if i2 is not None else None
            if not k1 or not k2:
                return None
            return {"M1": m1, "M2": m2, "T1": float(k1), "T2": float(k2), "src": os.path.basename(path)}
    except Exception:
        return None


def _dcp_find(make, model):
    """按机型找 Adobe Standard 的 .dcp。找不到返回 None（不是错误，退回单矩阵）。"""
    key = (make, model)
    if key in _DCP_CACHE:
        return _DCP_CACHE[key]
    prof = None
    if _DCP_ON and make and model:
        roots = [r"C:\ProgramData\Adobe\CameraRaw\CameraProfiles",
                 os.path.join(os.environ.get("APPDATA", ""), "Adobe", "CameraRaw", "CameraProfiles")]
        for root in roots:
            if not root or not os.path.isdir(root):
                continue
            for f in sorted(__import__("glob").glob(
                    os.path.join(root, "**", "*%s*Adobe Standard*.dcp" % model), recursive=True)):
                prof = _dcp_read(f, make, model)
                if prof:
                    break
            if prof:
                break
    _DCP_CACHE[key] = prof
    return prof


def _dcp_matrix(prof, K):
    """按色温在两组标定矩阵之间插值。用 **mired（1/T）线性** —— 摄影色温本就是 mired 线性。
    超出标定区间就夹住（区间外用最近那套 + 对角白平衡，是通行做法）。"""
    if not prof:
        return None
    T1, T2 = prof["T1"], prof["T2"]
    d = 1.0/T1 - 1.0/T2
    if abs(d) < 1e-12 or not K or K <= 0:
        return prof["M1"]
    f = (1.0/T1 - 1.0/float(K)) / d
    f = min(1.0, max(0.0, f))
    return prof["M1"] * (1.0 - f) + prof["M2"] * f


def _illum_rgb(K, cam_xyz=None, tint=0.0):
    """色温 K（含色调 tint）所对应的**光源**在该色彩空间下的 RGB（G 归一为 1）。

    cam_xyz 给出时用**相机原始通道**求逆 —— LibRaw 的 user_wb 就作用在这个域上，
    这也是能对齐 ACR 的关键（此前用 ACES2065-1 反解，色料不同，观感就是"简单色偏移"）。
    """
    x, y = _kc_xy(float(K))
    if tint:
        # ACR 的色调是沿**垂直于普朗克轨迹**的色度方向偏移，而不是简单抬降绿通道
        # （后者会连带改变亮度）。切线用对 K 的数值导数求，法线即色調方向。
        dT = max(1.0, abs(float(K))*0.002)
        x2, y2 = _kc_xy(float(K) + dT)
        tx, ty = x2 - x, y2 - y
        n = float(np.hypot(tx, ty))
        if n > 1e-12:
            # 法线取 (ty, -tx)：+tint 让**光源**偏绿 → 校正后画面偏品红（与 ACR 一致）
            nx, ny = ty/n, -tx/n
            k = (float(tint)/150.0)*0.045          # ±150 → ±0.045 色度偏移
            x, y = x + nx*k, y + ny*k
    y = max(y, 1e-6)
    xyz = np.array([x/y, 1.0, (1.0 - x - y)/y], dtype=np.float64)
    if cam_xyz is not None:
        try:
            M = np.asarray(cam_xyz, dtype=np.float64)[:3, :3]
            if M.shape == (3, 3) and abs(np.linalg.det(M)) > 1e-12:
                # **方向**：rawpy 的 rgb_xyz_matrix 名字有误导，实测它是 **XYZ→相机**。
                # 证据（交叉验证到每一位）：
                #   inv(M) @ (1/daylight_wb) → 色度 (0.312727, 0.329023)，D65 偏差 0.0000
                #   M @ XYZ_D65 归一化 = [0.36253, 1, 0.80731] == 实测 D65 响应，逐位相同
                # 所以 XYZ→相机 就是 **M @**，不是 np.linalg.solve(M, xyz)。
                # 此前写成 solve(M, xyz) 等于把方向当反了：色温"越暖"的方向碰巧还对
                # （分子分母过了同一个错变换），但**通道之间的比例是错的 → 色相就错了**。
                rgb = M @ xyz
            else:
                rgb = None
        except Exception:
            rgb = None
    else:
        rgb = None
    if rgb is None:
        _, Mi, _ = _mat("ACES2065-1")              # 兜底（拿不到相机矩阵时）
        rgb = np.asarray(Mi, dtype=np.float64) @ xyz
    g = rgb[1] if abs(rgb[1]) > 1e-12 else 1.0
    return rgb / g


def _wb_gain(K, tint=0.0, cam_xyz=None):
    """(色温, 色调) → 三通道白平衡**校正增益**（G=1）。

    方向约定与 ACR/Lightroom 一致：
      · **K 越高画面越暖**（校正增益 = 1/光源颜色，光源越蓝、补偿越暖）
      · **色调为正 → 画面偏品红**（光源偏绿，补偿后偏品红）
    """
    L = np.maximum(_illum_rgb(K, cam_xyz, tint), 1e-6)
    g = 1.0 / L
    g = g / (g[1] if abs(g[1]) > 1e-12 else 1.0)
    # 再归一化到 **K=6500 为单位增益**（保持旧接口语义：_kelvin_gain(6500) == (1,1,1)）。
    # 调用方取的是 _wb_gain(t)/_wb_gain(ref) 的**比值**，这个常数会约掉，不影响结果。
    Lr = np.maximum(_illum_rgb(6500.0, cam_xyz, 0.0), 1e-6)
    gr = 1.0 / Lr
    gr = gr / (gr[1] if abs(gr[1]) > 1e-12 else 1.0)
    return g / gr


def _kelvin_gain(K, cam_xyz=None):
    """仅色温的校正增益（保留旧名字，供外部/测试调用）。K 越高越暖。"""
    return tuple(float(v) for v in _wb_gain(K, 0.0, cam_xyz))


def _kc_xy(T):
    if T <= 4000:
        x = -0.2661239e9/T**3 - 0.2343580e6/T**2 + 0.8776956e3/T + 0.179910
    else:
        x = -3.0258469e9/T**3 + 2.1070379e6/T**2 + 0.2226347e3/T + 0.240390
    if T <= 2222:
        y = -1.1063814*x**3 - 1.34811020*x**2 + 2.18555832*x - 0.20219683
    elif T <= 4000:
        y = -0.9549476*x**3 - 1.37418593*x**2 + 2.09137015*x - 0.16748867
    else:
        y =  3.0817580*x**3 - 5.87338670*x**2 + 3.75112997*x - 0.37001483
    return x, y

# ---- RAW 黑边裁剪（消除 sensor 右侧/底部未曝光黑区） ----
def _flip_bbox(flip, T, L, Hh, Ww, raw_h, raw_w):
    """把 raw 坐标系有效区(行T..T+Hh,列L..L+Ww)映射到 flip 后输出坐标。
    返回 (top, left, height, width)。基于 dcraw flip 变换。"""
    corners = [(T, L), (T, L + Ww), (T + Hh, L), (T + Hh, L + Ww)]
    mapped = []
    for (y, x) in corners:
        if flip & 4:
            y, x = x, y
        if flip & 2:
            x = (raw_w - 1 - x) if not (flip & 4) else (raw_h - 1 - x)
        if flip & 1:
            y = (raw_h - 1 - y) if not (flip & 4) else (raw_w - 1 - y)
        mapped.append((y, x))
    ys = [p[0] for p in mapped]; xs = [p[1] for p in mapped]
    return min(ys), min(xs), max(ys) - min(ys), max(xs) - min(xs)

def _crop_raw_border(a, raw, half):
    """按 raw crop 参数 + flip 裁剪解码结果，去掉右侧/底部黑边。a: float32 数组(H,W,3)。"""
    try:
        s = raw.sizes
        if s.crop_width <= 0 or s.crop_height <= 0:
            return a
        top, left, hh, ww = _flip_bbox(s.flip, s.crop_top_margin, s.crop_left_margin,
                                       s.crop_height, s.crop_width, s.raw_height, s.raw_width)
        if half:                      # half 解码尺寸减半
            top, left = top // 2, left // 2
            hh, ww = max(1, hh // 2), max(1, ww // 2)
        h, w = a.shape[0], a.shape[1]
        top = max(0, min(top, h - 1)); left = max(0, min(left, w - 1))
        hh = max(1, min(hh, h - top)); ww = max(1, min(ww, w - left))
        return a[top:top + hh, left:left + ww]
    except Exception:
        return a

# ---- 机内白平衡 → 等效色温（只用于"滑块默认落在机内"这个锚点） ----
# WB 是主观操作，这里不追求色度学严谨：只需从 EXIF/LibRaw 给出的机内 WB 数值
# 算出一个等效色温，作为滑块的默认位置与"相对机内"的增益基准。
# 做法：1/白平衡乘数 = 相机对光源的响应 → rgb_xyz_matrix 逆 → 光源色度 → McCamy。
# 自检：daylight_whitebalance 应还原出 ≈6500K（D65，x=0.3127 y=0.3290）。
_SHOT_CCT_CACHE = {}


def _cct_from_wb(cam_mul, rgb_xyz):
    """机内白平衡乘数 → 等效色温(K)；算不出来返回 None（UI 退回 6500 并标注"机内未知"）。"""
    try:
        m = np.array(cam_mul[:3], dtype=np.float64)
        m = m / (m[1] if m[1] else 1.0)
        M = np.array(rgb_xyz, dtype=np.float64)[:3, :3]
        if M.shape != (3, 3) or not np.isfinite(M).all() or abs(np.linalg.det(M)) < 1e-12:
            return None
        XYZ = np.linalg.inv(M) @ (1.0 / m)
        s = float(XYZ.sum())
        if not np.isfinite(s) or s <= 0:
            return None
        x, y = float(XYZ[0]) / s, float(XYZ[1]) / s
        if not (0.0 < x < 1.0 and 0.05 < y < 1.0):
            return None
        n = (x - 0.3320) / (0.1858 - y)
        cct = 449 * n ** 3 + 3525 * n ** 2 + 6823.3 * n + 5520.33
        return float(min(max(cct, 1500.0), 20000.0))
    except Exception:
        return None


def shot_cct(path):
    """某张 RAW 的机内等效色温（解码时缓存）；未知返回 None。"""
    return _SHOT_CCT_CACHE.get(str(path)) if path else None


def read_shot_cct(path):
    """只读 RAW 头拿机内等效色温（不解码像素，约百毫秒），结果入缓存。"""
    p = str(path)
    if p in _SHOT_CCT_CACHE:
        return _SHOT_CCT_CACHE[p]
    try:
        with _RAW.imread(p) as raw:
            c = _cct_from_wb(raw.camera_whitebalance, raw.rgb_xyz_matrix)
        _SHOT_CCT_CACHE[p] = c
        return c
    except Exception:
        return None


def _load_float(rp, half=False, work=None, wb=None, encode=True):
    """按原生位深读入 float32 [0,1] 数组。RAW 经 libraw 解拜尔+白平衡→线性宽色域→sRGB 编码。

    work 指定宽工作空间名（默认 _RAW_WORK_DEF）；wb=(temp,tint) 为解码层白平衡偏移
    （色温±1=暖/冷，色调±1=绿/品红，0=机内白平衡），LR 式作用于 Bayer 域增益。
    输出编码到 sRGB 曲线使预览/调色在显示域；_transform 反解回线性宽色域调色，最后才 CST 到目标。

    encode（4.0 新增，仅影响 RAW 分支）：
      · False（主路径）→ 返回**线性**宽色域。LibRaw 本来就是 gamma=(1,1) 解码，数据天生线性，
        直接交给管线即可，全程不做 sRGB 编解码往返。
      · True（兼容旧调用方）→ 再编码到 sRGB 曲线，供 8bit 显示代理使用。"""
    _sx = Path(rp).suffix.lower()
    if _RAW is not None and _sx in _RAW_EXTS:
        cs = _RAW_WORK_OPT.get(work) or _RAW_WORK_OPT[_RAW_WORK_DEF]
        with _RAW.imread(rp) as raw:
            cam = raw.camera_whitebalance.copy()
            g = cam[1] if cam[1] else 1.0
            uwb = np.array([cam[0]/g, 1.0, cam[2]/g, 0.0], dtype=np.float64)   # 机内 WB 数值
            _shot = _cct_from_wb(cam, raw.rgb_xyz_matrix)
            _SHOT_CCT_CACHE[str(rp)] = _shot          # 供 UI 把滑块默认落在"机内"上
            if wb:
                t, tt = wb[0], wb[1]
                # 增益基准：调用方给的（当前图片机内等效色温）→ 刚算出的 → 6500
                ref = wb[2] if len(wb) > 2 and wb[2] else (_shot or 6500.0)
                # 增益必须算在**相机原始通道**上（user_wb 就在这个域），
                # 相机自带的 rgb_xyz_matrix 给出了 相机RGB→XYZ，求逆即得 XYZ→相机RGB
                try:
                    # 注意：这个矩阵是 **XYZ→相机**（名字 rgb_xyz_matrix 有误导，
                    # 见 _illum_rgb 里的交叉验证）。_wb_gain 内部已按正确方向使用。
                    _cam = np.asarray(raw.rgb_xyz_matrix, dtype=np.float64)[:3, :3]
                    if _cam.shape != (3, 3) or abs(np.linalg.det(_cam)) < 1e-12:
                        _cam = None
                except Exception:
                    _cam = None
                # 可选增强：本机若装了 Adobe 的相机 profile，就用它的**双光源标定矩阵**
                # 按色温插值（只取客观标定数据，不碰主观风格）。读不到就退回上面那套单矩阵。
                _prof = None
                try:
                    _mk, _md = _shot_model(rp)
                    _prof = _dcp_find(_mk, _md)
                except Exception:
                    _prof = None

                def _mat_for(K):
                    # 注意不能写 ... or _cam —— numpy 数组的真值判断是歧义的
                    _M = _dcp_matrix(_prof, K) if _prof else None
                    return _M if _M is not None else _cam

                _wb_off = False
                if t or tt:
                    _kt = float(t) if t else float(ref)
                    _gn = _wb_gain(_kt, float(tt), _mat_for(_kt))
                    _gr = _wb_gain(float(ref), 0.0, _mat_for(float(ref)))
                    uwb[:3] *= (_gn / _gr)
                # 注意：色调不再是单独的绿通道乘数 —— 它和色温一起决定同一个光源点，
                # 再对 R/B 一起作用，因此不会像以前那样顺带改变亮度。
            _hlkw = {}
            _hle = _hl_enum()
            if _hle is not None:
                _hlkw["highlight_mode"] = _hle      # 不传就会落到 rawpy 默认的 Clip
            rgb = raw.postprocess(use_camera_wb=False, user_wb=uwb.tolist(), output_color=cs,
                                  gamma=(1, 1), no_auto_bright=True,
                                  output_bps=16, half_size=half, **_hlkw)
            rgb = _crop_raw_border(rgb.astype(np.float32) / 65535.0, raw, half)   # 去掉黑边
            # 这里**故意不做**亮度归一化。
            # 曾加过一个"亮度锁定"（拿机内那次的图像平均亮度当锚，把带偏移的解码拉回同一亮度），
            # 目的是模仿 ACR 的"纯色度滑块"。拆掉的理由：
            #   1) 随设定白平衡而来的亮度变化是**物理自然**的 —— 把光源假设从 L1 改成 L2，
            #      画面就是 (L_actual/L_assumed)·反射率，亮度本就该跟着动。
            #   2) 那个锚是**内容相关**的（全图平均亮度），不同照片会得到不同归一化，不科学。
            #   3) 本工具的目标不是"像 ACR"。
            # 增益形状被 LibRaw 的 G 归一钉死、没有自由度，能做的只有事后乘标量 —— 那正是被去掉的东西。
        a = rgb
        # encode=False：直接给线性宽色域。这是 4.0 主路径——管线全程留在**线性域**，
        # 不做任何 sRGB 编解码往返（此前这个往返让 4 次 pow 白跑，实测 33MP 约 1.0s）。
        if not encode:
            return a
        # encode=True：编码成 sRGB 曲线（8bit 显示代理 / 旧调用方）。
        # 必须走 numba 内核。numpy 的 CURVE["sRGB"][0] 是对 9800 万像素逐元素算 pow，
        # 实测 33MP **1.49s**（其内存带宽下界只有 146ms，慢 10.6×）；
        # _nb._encode_srgb 同样 33MP 只要 **0.231s（6.5×）**，两者最大差 1.8e-07
        # （float32 舍入级，远低于 8bit 的 1/255=3.9e-3 与 16bit 的 1.5e-5）。
        if _NB:
            return _nb._encode_srgb(a)
        e, _ = CURVE["sRGB"]
        return e(a)
        # 注意：不能加 no_auto_scale=True —— 它会连白平衡增益一起跳过（实测白平衡失效）
    elif _TIFF is not None and _sx in (".tif", ".tiff"):
        with _TIFF.TiffFile(rp) as tf:
            a = tf.asarray()
            o = tf.pages[0].tags.get("Orientation")
            o = o.value if o is not None else 1
        if a.dtype == np.uint16: a = a.astype(np.float32) / 65535.0
        elif a.dtype == np.uint8: a = a.astype(np.float32) / 255.0
        else: a = a.astype(np.float32)
        if o != 1: a = _orient(a, o)
    elif _HEIF and _sx in (".heif", ".heic"):
        hf = pillow_heif.open_heif(rp, convert_hdr_to_8bit=False)
        mode, (w, h), stride = hf.mode, hf.size, hf.stride
        raw16 = np.frombuffer(hf.data, dtype=np.uint16)
        nch = 4 if ("RGBA" in mode or "BGRA" in mode) else (3 if ("RGB" in mode or "BGR" in mode) else 1)
        row = stride // 2
        a = raw16.reshape(h, row)[:, :w * nch].reshape(h, w, nch).astype(np.float32)
        if nch >= 3: a = a[..., :3]
        a = a / 65535.0      # libheif 已把 10/12-bit 左移到 16-bit 域
    else:
        with Image.open(rp) as im:
            im = ImageOps.exif_transpose(im)
            m = im.mode
            if m in ("I;16", "I;16L", "I;16B"):
                a = np.asarray(im, dtype=np.float32) / 65535.0
            elif m == "I":
                a = np.asarray(im, dtype=np.float32)
                mx = a.max()
                if mx > 65535: a = a / mx          # 32-bit int：归一到最大值
                elif mx > 0: a = a / 65535.0       # 16-bit int 以 I 模式存储
            elif m == "F":
                a = np.asarray(im, dtype=np.float32)
            elif m == "L":
                a = np.asarray(im, dtype=np.float32) / 255.0
            elif m == "LA":
                a = np.asarray(im, dtype=np.float32)[..., :1] / 255.0
                a = np.repeat(a, 3, axis=2)
            elif m == "RGB":
                a = np.asarray(im, dtype=np.float32) / 255.0
            elif m == "RGBA":
                a = np.asarray(im, dtype=np.float32)[..., :3] / 255.0
            else:
                a = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=2)
    elif a.ndim == 3 and a.shape[2] != 3:
        a = a[..., :3] if a.shape[2] > 3 else np.repeat(a, 3, axis=2)
    return np.clip(a, 0.0, 1.0)

def _trunc(s, n):
    return s if len(s) <= n else s[:n-1] + "…"

_INPUT_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".heif", ".heic",
               ".raf", ".cr2", ".cr3", ".arw", ".nef", ".dng", ".orf", ".rw2", ".pef", ".srw", ".x3f"}
_RAW_EXTS = {".raf", ".cr2", ".cr3", ".arw", ".nef", ".dng", ".orf", ".rw2", ".pef", ".srw", ".x3f"}
_OUT_FMTS = {"PNG 8-bit": ".png", "JPEG 8-bit": ".jpg", "TIFF 16-bit": ".tiff",
             "HEIF 444 无损": ".heic", "HEIF 422": ".heic", "HEIF 420": ".heic"}
# EXIF 基本 tag（不含 Orientation，程序已处理旋转）
_EXIF_BASIC = {271, 272, 33434, 33437, 34850, 34855, 37377, 37378, 37380, 37381,
               36867, 36868, 42036}
# 归属：只有 Make/Model 属于 IFD0，其余（ExposureTime 等）都属于 **Exif 子 IFD(34665)**。
# 旧实现把它们全写进 IFD0 —— 读图软件会去 34665 里找，找不到，等于没写。
_EXIF_IFD0 = {271, 272}
_EXIF_SUB = _EXIF_BASIC - _EXIF_IFD0
# TIFF extratags 用的 TIFF 数据类型码：1=BYTE 2=ASCII 3=SHORT 4=LONG 5=RATIONAL 10=SRATIONAL
_EXIF_DTYPE = {271: 2, 272: 2, 33434: 5, 33437: 5, 34850: 3, 34855: 3,
               37377: 10, 37378: 5, 37380: 10, 37381: 5, 36867: 2, 36868: 2, 42036: 2}
_EXIF_CACHE = {}


def _exif_merge(dst, ex):
    """把 Pillow Exif 对象里的基本 tag 收进 dst（顶层 + 34665 子 IFD 都看）。"""
    try:
        for c in _EXIF_BASIC:
            if c in ex:
                dst[c] = ex[c]
    except Exception:
        pass
    try:
        sub = ex.get_ifd(34665)
        if sub:
            for c in _EXIF_BASIC:
                if c in sub and c not in dst:
                    dst[c] = sub[c]
    except Exception:
        pass


def _exif_pairs(rp):
    """读出 {tag码: 值}。返回 dict（可能为空）。

    4.0 修复：**ARW/CR2/NEF/DNG 都是 TIFF 容器**，文件里根本没有 JPEG 那种
    b"Exif\x00\x00" 标记。旧代码 `head.find(b"Exif\x00\x00")` 返回 -1 就放弃了，
    于是四个导出格式**一个 EXIF 都没有**。改为按 TIFF 结构读 IFD0 + 34665 子 IFD。
    """
    if rp in _EXIF_CACHE:
        return _EXIF_CACHE[rp]
    out = {}
    try:
        if _is_raw(rp):
            try:
                import tifffile
                n2c = {v: k for k, v in tifffile.TIFF.TAGS.items() if isinstance(v, str)}
                with tifffile.TiffFile(rp) as tf:
                    pg = tf.pages[0]
                    for c, t in pg.tags.items():
                        if c in _EXIF_BASIC:
                            out[c] = t.value
                    et = pg.tags.get(34665)
                    if et is not None and isinstance(et.value, dict):
                        for nm, val in et.value.items():
                            c = n2c.get(nm)
                            if c in _EXIF_BASIC:
                                out[c] = val
            except Exception:
                pass
            if not out:
                # 兜底：少数 RAW（如部分 RAF）确实内嵌 JPEG 式 Exif 块
                with open(rp, "rb") as f:
                    head = f.read(4 * 1024 * 1024)
                idx = head.find(b"Exif\x00\x00")
                if idx >= 0:
                    ex = Image.Exif()
                    ex.load(head[idx:])
                    _exif_merge(out, ex)
        else:
            with Image.open(rp) as im:
                _exif_merge(out, im.getexif())
    except Exception:
        out = {}
    _EXIF_CACHE[rp] = out
    return out


def _extract_exif(rp):
    """返回**结构合规**的 EXIF bytes：IFD0 放 Make/Model，其余进 Exif 子 IFD(34665)。"""
    pairs = _exif_pairs(rp)
    if not pairs:
        return None
    try:
        ne = Image.Exif()
        for c in _EXIF_IFD0:
            v = pairs.get(c)
            if v is not None:
                ne[c] = v
        sub = ne.get_ifd(34665)
        for c in _EXIF_SUB:
            v = pairs.get(c)
            if v is None:
                continue
            try:
                sub[c] = v
            except Exception:
                pass
        return ne.tobytes()
    except Exception:
        return None


def _exif_tiff_extratags(rp):
    """TIFF 用的 extratags。

    tifffile 没有写「IFD 型标签」的能力（源码里连 34665 都不存在），
    所以 TIFF 只能把这些标签写进 IFD0。这不完全符合 EXIF 规范的分层，
    但 exiftool 等工具会扫描 IFD0，实际可读 —— 总比一个都没有好。
    """
    out = []
    for c, v in _exif_pairs(rp).items():
        dt = _EXIF_DTYPE.get(c)
        if dt is None:
            continue
        try:
            if dt == 2:                       # ASCII：必须 NUL 结尾，count 是字节数
                bb = v if isinstance(v, bytes) else str(v).encode("ascii", "replace")
                bb = bb.rstrip(b"\x00") + b"\x00"
                out.append((c, 2, len(bb), bb, True))
            elif dt in (5, 10):               # RATIONAL / SRATIONAL
                if isinstance(v, (tuple, list)) and len(v) == 2:
                    out.append((c, dt, 1, (int(v[0]), int(v[1])), True))
            else:                             # SHORT
                out.append((c, dt, 1, int(v), True))
        except Exception:
            continue
    return out


_ICC_CACHE = {}

# ==================== ICC 档案生成（4.0 修复） ====================
# 为什么要自己生成，而不是像之前那样从系统借现成的：
#   系统里那些是**显示器**档案，描述的是某一块屏幕，TRC 也是那块屏幕的。
#   实测借来的档案与我们的编码曲线对不上（下游按档案解码后的相对误差）：
#     Rec.709 / Rec.2020            → 档案纯 gamma 2.398，我们用 BT.709 OETF 编码
#                                     0.02 处 −84.5%、0.18 处 −34.9%   ← 硬伤
#     Display P3 / Adobe RGB / ProPhoto → 档案纯 2.199，我们用 sRGB 分段曲线
#                                     0.02 处 −21.0%
#     DCI-P3                        → 2.6016，碰巧吻合
#   自己按 _GAMUT_DEFS + _CURVE_MATCH 现场生成，才能保证**像素与标签同源**。
#
# 结构照 ICC v2 的通行约定（与 Microsoft sRGB / AdobeRGB1998 / ProPhoto 三个档案实测一致）：
#   · device class = 'mntr'，PCS illuminant = D50
#   · rXYZ/gXYZ/bXYZ 是**已 Bradford 适配到 D50** 的色料（三者相加 = D50）
#   · wtpt 写**原生白点**，不写 chad 标签
import struct as _struct
import datetime as _dt


# ICC 规范里 PCS 照明体 D50 是**定值**，不是从 xy 反算的。
# 从 xy=(0.34567,0.35850) 反算得 Z=0.82519，与规范的 0.8249 差 3e-4，
# 会让"色料和"对不上"声明的 PCS 照明体"（AdobeRGB1998 / ProPhoto 都是精确相等）。
_ICC_D50 = np.array([0.9642, 1.0, 0.8249])


def _s15f16(v):
    return _struct.pack(">i", int(round(float(v) * 65536.0)))


def _xyz_tag(v):
    return b"XYZ " + b"\x00" * 4 + _s15f16(v[0]) + _s15f16(v[1]) + _s15f16(v[2])


def _text_tag(t):
    return b"text" + b"\x00" * 4 + t.encode("ascii", "replace") + b"\x00"


def _desc_tag(t):
    """textDescriptionType：ASCII + 空 unicode/scriptcode 段（v2 标准布局）。"""
    a = t.encode("ascii", "replace") + b"\x00"
    return (b"desc" + b"\x00" * 4 + _struct.pack(">I", len(a)) + a
            + _struct.pack(">I", 0) + _struct.pack(">I", 0)      # unicode 语言码 / 条数
            + _struct.pack(">H", 0) + b"\x00" + b"\x00" * 67)    # scriptcode 码 / 条数 / 文本


def _curv_tag(vals=None, gamma=None):
    if vals is None and gamma is None:
        return b"curv" + b"\x00" * 4 + _struct.pack(">I", 0)                       # 恒等
    if gamma is not None:
        return (b"curv" + b"\x00" * 4 + _struct.pack(">I", 1)
                + _struct.pack(">H", int(round(float(gamma) * 256.0))))            # u8Fixed8
    n = len(vals)
    body = b"".join(_struct.pack(">H", int(round(max(0.0, min(1.0, float(v))) * 65535)))
                    for v in vals)
    return b"curv" + b"\x00" * 4 + _struct.pack(">I", n) + body


def _para_tag(fn, params):
    """parametricCurveType。type 3 = Y=(aX+b)^g (X>=d) 否则 cX。"""
    return (b"para" + b"\x00" * 4 + _struct.pack(">H", fn) + b"\x00" * 2
            + b"".join(_s15f16(p) for p in params))


# ICC 的 TRC 存的是**设备值 → 线性**，也就是我们 CURVE 里的 decode。
# 这里统一用**采样表**（curv 多点），照 Microsoft 的 sRGB 档案做法（它就是 curv 1024 点）：
#   · 分段曲线（sRGB / BT.709 有线性趾部）本来就没法用单一 gamma 表达
#   · 纯 gamma 若用 curv(1)，gamma 只能存成 u8Fixed8 —— 2.6 会被量化成 2.6015625，
#     暗部因此有约 0.6% 的相对误差。采样表把这一项消掉。
_GAMMA_RE = re.compile(r"^Gamma\s+([0-9.]+)$")

# ICC 的 `desc` 是别的软件显示这个档案时用的名字（XnView MP 就按它显示）。
# 行业惯例是**标准色彩空间名**：实测权威档案为
# sRGB IEC61966-2.1 / Adobe RGB (1998) / ProPhoto RGB。
_ICC_DESC = {
    "sRGB": "sRGB IEC61966-2.1",
    "Rec.709": "ITU-R BT.709",
    "Rec.2020": "ITU-R BT.2020",
    "DCI-P3": "DCI-P3",
    "Display P3": "Display P3",
    "Adobe RGB": "Adobe RGB (1998)",
    "ProPhoto RGB": "ProPhoto RGB",
    "S-Gamut3.Cine": "Sony S-Gamut3.Cine/S-Log3",
    "S-Gamut3": "Sony S-Gamut3/S-Log3",
    "V-Gamut": "Panasonic V-Gamut/V-Log",
    "ARRI WG": "ARRI Wide Gamut 3/LogC3",
    "REDWideGamut": "REDWideGamutRGB/Log3G10",
    "ACEScg": "ACEScg",
    "ACEScct": "ACEScct",
    "ACES2065-1": "ACES2065-1",
}

# 解析式 TRC（parametricCurveType）。type 3 = Y=(aX+b)^g (X>=d) 否则 cX。
# sRGB 分段曲线与 BT.709 OETF 的逆正是这个形状，可以精确表达，
# 不必像以前那样用采样表去逼近。与 Photoshop 的 Display P3 档案写法一致。
_PARA_SRGB = (3, [2.4, 1.0 / 1.055, 0.055 / 1.055, 1.0 / 12.92, 0.04045])
_PARA_R709 = (3, [1.0 / 0.45, 1.0 / 1.099, 0.099 / 1.099, 1.0 / 4.5, 0.081])


def _trc_tag(name, n=1024):
    if name == "Linear":
        return _curv_tag()                      # curv 点数 0 = 恒等
    if name == "sRGB":
        return _para_tag(*_PARA_SRGB)
    if name == "Rec.709":
        return _para_tag(*_PARA_R709)
    _m = _GAMMA_RE.match(name or "")
    if _m:
        # 纯 gamma 用 para 类型 0：gamma 以 s15Fixed16 存储，比 curv(1) 的
        # u8Fixed8 精确得多（curv(1) 会把 2.6 量化成 2.6015625）。
        return _para_tag(0, [float(_m.group(1))])
    fn = CURVE.get(name)
    if fn is None:
        return None
    _, dec = fn
    xs = np.linspace(0.0, 1.0, n).astype(np.float32)
    try:
        ys = np.asarray(dec(xs), dtype=np.float64)
    except Exception:
        return None
    if ys.shape != xs.shape or not np.all(np.isfinite(ys)):
        return None
    # Log 曲线（S-Log3 / V-Log / LogC3 / Log3G10 / ACEScct）没有解析式，只能采样。
    # 它们的 decode 会超过 1.0（V-Log 在设备值 1.0 处约 1.24），而 ICC 值域是 [0,1]。
    # **不能夹取**：夹取会让曲线后半段整段变平（V-Gamut 曾有 1075/2048 点），
    # 那等于宣称"设备值 0.5 以上全映射到线性 1.0"，作为 ICC 曲线不合法。
    # 按端点归一化是唯一既合法又保持单调的表达。
    _mx = float(np.max(ys))
    if _mx > 1.0:
        ys = ys / _mx
    return _curv_tag(vals=np.clip(ys, 0.0, 1.0))


def _sf32_tag(mat):
    """s15Fixed16ArrayType —— chad（色适应矩阵）用的类型。"""
    vals = [mat[i][j] for i in range(3) for j in range(3)]
    return b"sf32" + b"\x00" * 4 + b"".join(_s15f16(v) for v in vals)


def _icc_id(b):
    """按 ICC 规范算 profileID：把 flags(44-47)、rendering intent(64-67)、
    profile ID(84-99) 三处清零后取 MD5。Photoshop 的档案有这个值，我们原来全 0。"""
    import hashlib
    m = bytearray(b)
    m[44:48] = b"\x00" * 4
    m[64:68] = b"\x00" * 4
    m[84:100] = b"\x00" * 16
    return hashlib.md5(bytes(m)).digest()


def _icc_build(gamut, curve=None):
    """按设计表现场生成 ICC **v4** 矩阵/TRC 档案。

    骨架照 Photoshop 的写法（用户实测 PS 生成的能正确显示、我们的不行）：
      · version 4.0.0（此前是 2.4.0）
      · 带 `chad` 色适应矩阵 —— v4 里色料适配到 D50 时这是**必需**的，
        此前我们写的是 v2 约定（隐含适配、无 chad）
      · TRC 用 parametricCurveType（此前是采样表）
      · 不带 `bkpt`（PS 没有）
      · profileID 写真实 MD5（此前全 0）
      · cmm 写 0（ICC 规范：非 CMM 创建应写 0，不是 ASCII 'none'）
    """
    _cv = curve or _CURVE_MATCH.get(gamut)
    # TRC 描述「码值 → 线性光」，对显示参考内容那就是它的**显示 EOTF**，不是编码曲线。
    # 必须与显示端同源（display_decode_curve），否则 CL 与第三方读同一文件结果不同。
    # 注：档案**描述名**保持不变（仍是 "ITU-R BT.709" 等），否则 icc_to_space 认不出自家文件。
    tag = _trc_tag(display_decode_curve(_cv))
    if tag is None:
        return None
    g = _GAMUT[gamut]
    M = np.asarray(_mat(gamut)[0], dtype=np.float64)           # 原生 RGB→XYZ
    Wn = np.asarray(_wpv(g["wp"]), dtype=np.float64)           # 原生白点
    # 色料 = Bradford 适配到 PCS 白点 D50（用规范的 D50 定值，不是从 xy 反算的）
    Cat = np.asarray(_cat(Wn, _ICC_D50), dtype=np.float64)
    col = Cat @ M
    _base = _ICC_DESC.get(gamut, gamut)
    _def = _CURVE_MATCH.get(gamut)
    desc = _base if (_cv == _def or not _cv) else f"{_base} ({_cv})"
    tags = [
        (b"desc", _desc_tag(desc)),
        (b"cprt", _text_tag("Color-Lab 4.0")),
        (b"wtpt", _xyz_tag(Wn)),                               # 原生白点
        (b"chad", _sf32_tag(Cat)),                             # v4 必需
        (b"rXYZ", _xyz_tag(col[:, 0])),
        (b"gXYZ", _xyz_tag(col[:, 1])),
        (b"bXYZ", _xyz_tag(col[:, 2])),
        (b"rTRC", tag), (b"gTRC", tag), (b"bTRC", tag),
    ]
    tags.sort(key=lambda t: t[0])                              # 规范要求按签名升序
    n = len(tags)
    off = 128 + 4 + n * 12
    table, body = b"", b""
    _seen = {}
    for sig, data in tags:
        # 相同数据只存一份：rTRC/gTRC/bTRC 内容一样，共用同一偏移。
        # Microsoft 的 sRGB 与 Photoshop 的档案都这么做。
        if data in _seen:
            table += sig + _struct.pack(">II", _seen[data], len(data))
            continue
        pad = (-len(data)) % 4
        _seen[data] = off
        table += sig + _struct.pack(">II", off, len(data))
        body += data + b"\x00" * pad
        off += len(data) + pad
    h = bytearray(128)
    _struct.pack_into(">I", h, 0, 128 + 4 + n * 12 + len(body))
    h[4:8] = b"\x00" * 4                                      # CMM 签名：未设置写 0
    _struct.pack_into(">I", h, 8, 0x04000000)                  # v4.0.0（对齐 PS）
    h[12:16] = b"mntr"
    h[16:20] = b"RGB "
    h[20:24] = b"XYZ "
    now = _dt.datetime.now()
    _struct.pack_into(">6H", h, 24, now.year, now.month, now.day,
                      now.hour, now.minute, now.second)
    h[36:40] = b"acsp"
    h[40:44] = b"MSFT"
    _struct.pack_into(">I", h, 64, 0)                          # rendering intent = perceptual
    for k, v in enumerate((0.9642, 1.0, 0.8249)):              # PCS 照明体 = D50
        _struct.pack_into(">i", h, 68 + k * 4, int(round(v * 65536)))
    out = bytes(h) + _struct.pack(">I", n) + table + body
    return out[:84] + _icc_id(out) + out[100:]                 # 填 profileID(MD5)


def _icc_bytes(gamut, curve=None):
    """返回目标色域的 ICC profile 字节（用于导出嵌入）。

    `curve` 是**实际用来编码的曲线**（界面上的 tc）。必须传进来 ——
    4.0 修复：此前这里只收色域、忽略曲线，用户把曲线改成 Gamma 2.4 时，
    导出用 2.4 编码却嵌入按默认曲线描述的档案，像素与标签又对不上。
    """
    _key = (gamut, curve)
    if _key in _ICC_CACHE:
        return _ICC_CACHE[_key]
    b = None
    # 4.0：优先**自己按设计表生成** —— 系统里借来的档案描述的是某块显示器，
    # TRC 与我们的编码曲线不一致（Rec.709 暗部实测差 −84.5%）。
    try:
        b = _icc_build(gamut, curve)
    except Exception as _e:
        # 绝不静默：这里一旦失败就会退回"从系统借档案"（TRC 与我们的编码对不上），
        # 而借来的档案体积各不相同，看上去还挺"正常" —— 曾经因此把一个
        # 全线 NameError 掩盖了很久。所以必须出声。
        print(f"[ICC] 生成 {gamut}/{curve} 失败，将退回系统档案：{type(_e).__name__}: {_e}")
        b = None
    try:
        if b is not None:
            pass
        elif gamut == "sRGB":
            b = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
        else:
            # 从系统 ICC 目录找匹配的 profile。
            # 此前只认死文件名（DisplayP3.icc / Rec2020.icc / Rec709.icc），
            # 而本机装的是 ASUS_Display_DCIP3.icm 这类名字 → 一个都匹配不上 →
            # DCI-P3 / Display P3 / Rec.2020 / Rec.709 导出时**不嵌任何 profile**，
            # 别的软件只能按 sRGB 解释宽色域数据 → 画面发淡。所以改成：
            # 先试标准文件名，再按关键词模糊匹配。
            _exact = {"Adobe RGB": ["AdobeRGB1998.icc", "AdobeRGB1998.icm"],
                      "Display P3": ["DisplayP3.icc", "Display P3.icc"],
                      "DCI-P3": ["DCIP3.icc", "DCI-P3.icc", "DCIP3.icm"],
                      "Rec.2020": ["Rec2020.icc", "Rec2020.icm"],
                      "Rec.709": ["Rec709.icc", "Rec709.icm"],
                      "ProPhoto RGB": ["ProPhoto.icm", "ProPhoto.icc"]}
            # 关键词匹配用的规范化片段（忽略大小写与 - _ 空格）
            _keys = {"Adobe RGB": ["adobergb1998", "adobergb"],
                     "Display P3": ["displayp3"],
                     "DCI-P3": ["dcip3"],
                     "Rec.2020": ["rec2020", "bt2020"],
                     "Rec.709": ["rec709", "bt709"],
                     "ProPhoto RGB": ["prophoto"]}
            _dirs = [Path(os.environ.get("WINDIR", "C:/Windows")) / "System32/spool/drivers/color",
                     Path("C:/Windows/System32/spool/drivers/color")]
            names = _exact.get(gamut, [])
            for root in _dirs:
                for nm in names:
                    p = root / nm
                    if p.exists():
                        b = p.read_bytes(); break
                if b:
                    break
            if b is None:
                frags = _keys.get(gamut)
                if frags and _dirs[0].is_dir():
                    cands = []
                    for p in _dirs[0].iterdir():
                        if p.suffix.lower() not in (".icc", ".icm"):
                            continue
                        norm = re.sub(r"[^a-z0-9]", "", p.name.lower())
                        if not any(f in norm for f in frags):
                            continue
                        try:
                            raw = p.read_bytes()
                        except Exception:
                            continue
                        # 必须是纯色彩空间档案；OEM 前缀与长名字排后面
                        is_space = _icc_is_space_profile(raw)
                        oem = not norm.startswith(("dcip3", "displayp3", "rec2020", "rec709",
                                                   "adobergb", "prophoto", "srgb"))
                        cands.append(((0 if is_space else 1), (1 if oem else 0), len(p.name), raw))
                    if cands:
                        # 只要存在干净档案，就绝不用校准档案
                        cands.sort(key=lambda t: (t[0], t[1], t[2]))
                        if cands[0][0] == 0:
                            b = cands[0][3]
    except Exception:
        b = None
    _ICC_CACHE[_key] = b
    return b

_CFG = Path(__file__).parent / ".colorlab_config"
def _cl():
    try: return json.load(open(_CFG))
    except Exception: return {}

# ---- PNG 压缩级别（4.0 新增）----
# PNG 写盘是 33MP 导出的第二大项（实测 3.52s / 占 37%），而它是 zlib 编码问题，不是算力问题。
# 真实 33MP 图的实测取舍（Pillow）：
#     级别 9 → 20.10s / 32.7MB      级别 6（Pillow 默认，3.3 实际用的）→ 3.15s / 34.9MB
#     级别 3 →  1.65s / 36.3MB  ← 拐点：快 1.9×，体积只涨 11%
#     级别 1 →  1.19s / 55.2MB      级别 0 → 0.84s / 98.3MB
# 故默认取 3。可用环境变量 COLORLAB_PNG_LEVEL 或配置项 pnglvl 覆盖（0-9）。
_PNG_LEVEL_DEF = 3
def _png_level(c=None):
    v = os.environ.get("COLORLAB_PNG_LEVEL")
    if v is None:
        v = (c or _cl()).get("pnglvl", _PNG_LEVEL_DEF)
    try:
        return max(0, min(9, int(v)))
    except Exception:
        return _PNG_LEVEL_DEF

# ==================== 主窗口 ====================
# 真实世界里同一色彩空间的 ICC 描述名不止一种。这里是**已知别名**（数据，不是猜）。
# 认不出的仍然返回 None，交给用户 —— 猜错比认不出更糟。
_ICC_ALIAS = {
    "srgb built-in": "sRGB",
    "srgb v4 icc preference profiles": "sRGB",
    "iec 61966-2.1 default rgb colour space - srgb": "sRGB",
    "iec61966-2.1": "sRGB",
    "adobe rgb (1998)": "Adobe RGB",
    "adobe rgb": "Adobe RGB",
    "display p3": "Display P3",
    "dci-p3": "DCI-P3",
    "prophoto rgb": "ProPhoto RGB",
    "rec. 2020": "Rec.2020",
    "rec.2020": "Rec.2020",
    "itu-r bt.709": "Rec.709",
    "rec. 709": "Rec.709",
    "aces2065-1": "ACES2065-1",
}


def icc_to_space(path):
    """读文件嵌入的 ICC，映射成我们的 (色域, 曲线, 说明)。

    **认不出就返回 None，绝不猜** —— 猜等于又换了个地方偷偷替换用户的声明。
    调用方拿到 None 时应如实提示（"ICC 无法识别，请手动指定源空间"），
    并保留用户手动选的源。

    没有嵌入 ICC 时：按业界通行约定当 sRGB 处理，但**明确说明**是"未标记"，
    而不是假装读到了。
    """
    try:
        with Image.open(path) as im:
            p = im.info.get("icc_profile")
    except Exception as e:
        return None, None, f"打开失败: {e}"
    if not p:
        return "sRGB", _CURVE_MATCH.get("sRGB", "sRGB"), "未嵌入 ICC —— 按通行约定当 sRGB"
    try:
        import io as _io
        prof = ImageCms.ImageCmsProfile(_io.BytesIO(p))
        raw_desc = ImageCms.getProfileDescription(prof)
    except Exception as e:
        return None, None, f"ICC 解析失败: {e}"
    desc = (raw_desc or "").replace("\x00", "").strip()
    dl = desc.strip().lower()

    def _hit(g, how):
        return g, _CURVE_MATCH.get(g, "sRGB"), f"ICC: {desc}" + ("" if not how else f" → {g}")

    # ① 精确对上我们自己用的标准描述名
    for g, d in _ICC_DESC.items():
        if d.strip().lower() == dl:
            return _hit(g, "")
    # ② 已知别名（真实世界里同一空间的其它写法）
    if dl in _ICC_ALIAS:
        return _hit(_ICC_ALIAS[dl], "别名")
    # ③ 宽松包含：档案名带厂商前缀/后缀时
    for g, d in _ICC_DESC.items():
        if d.strip().lower() in dl:
            return _hit(g, "包含匹配")
    for a, g in _ICC_ALIAS.items():
        if a in dl and g in GAMUT_NAMES:
            return _hit(g, "别名包含")
    # ④ 认不出 —— **不猜**，交给用户
    return None, None, f"ICC 无法识别: {desc!r}（请手动指定源空间）"


def resolve_source(s, raw_work=_RAW_WORK_DEF):
    """统一解析「源侧」——管线与 UI 必须给出同一个答案。

    · RAW：源侧由解码层决定（工作空间色域 + **Linear** 曲线，见 _load_float 的说明），用户不可选；
      `rw`（工作空间 / 解码器输出色域）是唯一合法的可调项。
      4.0 起这里是 **Linear** 而非 sRGB：解码层交付的本来就是线性数据，
      宣称 sRGB 会强迫管线做一次"编码→立刻解码"的往返（实测 33MP 约 1.0s 白跑）。
    · 非 RAW（TIFF/JPEG 等）：源色域 / 曲线由用户选择，`rw` 不参与。

    所有取源侧的地方都必须走这里，避免再出现"UI 显示一个源、管线用另一个源"。
    """
    if s.get("raw"):
        return s.get("rw", raw_work), "Linear"
    return s["sg"], s["sc"]


def transform(a, s, raw_work=_RAW_WORK_DEF, timing=None):
    """单张图的完整处理链：调色 → CST（含色域缩限）→ 可选套 LUT。

    抽成模块级函数，供 App._transform、基准脚本、离线工具共用——避免"基准测的不是 App 跑的那条路"
    （此前 bench 直接调 _grade/Pipe.go/_al，于是 numba 分支的收益根本没被测到）。
    s 为界面快照字典，字段见 App._snapshot()。timing 传 dict 时回填分段耗时与 CST 负分量计数。
    """
    def _t0():
        if timing is None:
            return 0.0
        import time as _time
        return _time.perf_counter()

    def _t1(k, t0):
        if timing is not None:
            import time as _time
            timing[k] = _time.perf_counter() - t0

    sg, sc = resolve_source(s, raw_work)
    raw = s.get("raw")
    tg, tc = s["tg"], s["tc"]
    # 调色 > CST > LUT（DaVinci 工作流）：先在源空间（sc 线性域）调色保精度，再转换/套 LUT
    # RAW 的色温/色调已烘焙在解码层（_load_float user_wb），此处归零避免二次应用
    tp = s["tp"] if not raw else 0.0
    lut_on = s["lut"] and s["cd"] is not None
    if _NB:
        # numba 内核。此前条件是 `_NB and not (lut and cd)`——只要选了 LUT 就整段退回 numpy，
        # 于是最耗时的套 LUT 那一步永远用不上加速版。现在 LUT 分支也走 numba。
        t0 = _t0(); a = _nb.grade_fast(a, sc, s["ev"], s["ct"], s["sat"], tp); _t1('grade', t0)
        t0 = _t0()
        if lut_on:
            li, lc = s["li"], s["lc"]
            a = _nb.pipe_fast(a, sg, sc, li, lc, ootf=s.get("ootf", "auto"))
        else:
            a = _nb.pipe_fast(a, sg, sc, tg, tc, ootf=s.get("ootf", "auto"))
        _t1('pipe', t0)
        if timing is not None:
            timing['cst_neg'] = int((np.asarray(a) < 0).sum())
        if lut_on:
            t0 = _t0(); a = _nb.lut_fast(a, s["cd"]); _t1('lut', t0)   # LUT 输出即最终画面
        if lut_on:
            # 末端 CST：LUT 的输出空间 → **目标**（tg/tc）。两侧都用**显示 EOTF**
            #（与显示端同源）→ 与显示层互为镜像。
            # ootf="auto"：显示参考→显示参考 时为 off；**Log→显示参考 时为 forward**
            #（这才是场景参考 LUT 输出能被正确渲染、并被写进文件的地方）。
            _fg, _fc = pipeline_out_space(s)
            t0 = _t0()
            # gc=2（只裁切）：与显示端一致。末端 CST 是**文件格式转换**，
            # 不该做 RGC 那种创意性的色域缩限 —— 那会按像素、看其它通道改颜色，
            # 镜像关系就断了（实测不同目标空间观感差 14%）。
            a = _nb.pipe_fast(a, _fg, display_decode_curve(_fc), tg,
                              display_decode_curve(tc),
                              ootf=s.get("ootf2", "off"), gc=2)
            _t1('out', t0)
        return a
    t0 = _t0(); a = _grade(a, sc, s["ev"], s["ct"], s["sat"], tp); _t1('grade', t0)
    t0 = _t0()
    if lut_on:
        li, lc = s["li"], s["lc"]
        if li != sg or lc != sc:
            a = Pipe.go(a, sg, sc, li, lc, ootf=s.get("ootf", "auto"))
    else:
        a = Pipe.go(a, sg, sc, tg, tc, ootf=s.get("ootf", "auto"))
    _t1('pipe', t0)
    if timing is not None:
        timing['cst_neg'] = int((np.asarray(a) < 0).sum())
    if lut_on:
        t0 = _t0(); a = _al(a, s["cd"]); _t1('lut', t0)               # LUT 输出即最终画面
    if lut_on:
        _fg, _fc = pipeline_out_space(s)
        t0 = _t0()
        a = Pipe.go(a, _fg, display_decode_curve(_fc), tg,
                    display_decode_curve(tc), ootf=s.get("ootf2", "off"), gc=2)
        _t1('out', t0)
    return a


# ---- GPU 版同一条链（4.0）----
# 与 transform() 逐条对应：grade（解码→调色→再编码）→ CST（含缩限）→ 可选套 LUT。
# 参数解析共用 _nb.pipe_params，避免"CPU/GPU 两套解析逻辑"漂移。
_RGC_TUPLE = None


def _rgc_params():
    """RGC 常量。**必须惰性求值**：模块级直接取 `_nb._RGC_THR_R` 会在循环导入时炸——
    colorlab 被 colorlab_nb 反向往回导入的那一刻，_nb 还是半成品。"""
    global _RGC_TUPLE
    if _RGC_TUPLE is None and _nb is not None:
        _RGC_TUPLE = (_nb._RGC_THR_R, _nb._RGC_THR_G, _nb._RGC_THR_B,
                      _nb._RGC_S_R, _nb._RGC_S_G, _nb._RGC_S_B)
    return _RGC_TUPLE


def gpu_transform(g, s, be, gL=None, lut_n=0, lut_scale=1.0, raw_work=_RAW_WORK_DEF):
    """GPU 版 transform：输入/输出都是显存数组。任何异常都由调用方回退 CPU。"""
    sg, sc = resolve_source(s, raw_work)
    raw = s.get("raw")
    # RAW 的色温/色调已烘焙在解码层，此处归零避免二次应用（与 transform 一致）
    tp = s["tp"] if not raw else 0.0
    lut_on = bool(s["lut"] and gL is not None)
    sc_id = _nb.curve_id(sc)
    kg = _kelvin_gain(float(tp)) if tp else (1.0, 1.0, 1.0)
    # 曲线名未知时必须报错（与 CPU 路径一致，不静默回落 Linear）
    if sc not in CURVE:
        raise KeyError(f'未知曲线: {sc}')
    g = be.grade(g, s["ev"], s["ct"], s["sat"], kg, pre_id=sc_id, post_id=sc_id)
    if lut_on:
        P = _nb.pipe_params(sg, sc, s["li"], s["lc"], ootf=s.get("ootf", "auto"))
    else:
        P = _nb.pipe_params(sg, sc, s["tg"], s["tc"], ootf=s.get("ootf", "auto"))
    if not P["same"]:
        g = be.pipe(g, P["M"], P["Mi"], P["Cat"], P["sc_id"], P["tc_id"],
                    P["sg_disp"], P["tg_disp"], P["ootf_mode"],
                    P["gam_sg"], P["gam_tg"], P["k"], P["use_roll"], P["gc_mode"],
                    _rgc_params())
    if lut_on:
        g = be.lut3d(g, gL, lut_n, lut_scale)
    if lut_on:
        # 与 CPU 两条路径同一套解析（pipe_params），避免三套逻辑漂移
        _fg, _fc = pipeline_out_space(s)
        P2 = _nb.pipe_params(_fg, display_decode_curve(_fc),
                             s["tg"], display_decode_curve(s["tc"]),
                             ootf=s.get("ootf2", "off"), gc=2)
        if not P2["same"]:
            g = be.pipe(g, P2["M"], P2["Mi"], P2["Cat"], P2["sc_id"], P2["tc_id"],
                        P2["sg_disp"], P2["tg_disp"], P2["ootf_mode"],
                        P2["gam_sg"], P2["gam_tg"], P2["k"], P2["use_roll"],
                        P2["gc_mode"], _rgc_params())
    return g


def gpu_display_leg(g, snap, be):
    """目标空间 → 显示器空间（GPU 路径）。与 display_leg_nb 同一套参数解析。"""
    sg, sc = display_leg_out_space(snap)
    sc = display_decode_curve(sc)
    dg, dc = display_space()[:2]
    if sg == dg and sc == dc:
        return g
    if sc not in CURVE or dc not in CURVE or sg not in _GAMUT or dg not in _GAMUT:
        return g
    P = _nb.pipe_params(sg, sc, dg, dc, ootf="off", gc=2)
    if P["same"]:
        return g
    return be.pipe(g, P["M"], P["Mi"], P["Cat"], P["sc_id"], P["tc_id"],
                   P["sg_disp"], P["tg_disp"], P["ootf_mode"],
                   P["gam_sg"], P["gam_tg"], P["k"], P["use_roll"], P["gc_mode"],
                   _rgc_params())


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title(f"colorlab v{APP_VERSION}"); self.geometry("1180x760"); self.minsize(980, 640)
        self._files = []; self._running = False; self._rows = []
        self._preview_file = None; self._pv_after = None
        self._pv_show_orig = False
        # 4.0：预览源是 **float32 线性**（不再是 8bit sRGB PIL）。
        # 管线输入精度与导出完全一致；8bit 只在最后上屏时由 _lin_to_u8 生成。
        self._pv_lin = None; self._pv_lin_small = None
        # 预览源的**实际编码**：RAW 是 Linear（解码层原生），非 RAW 是文件原生编码
        # （JPEG 即 sRGB）。显示时必须据此分派，否则会对 sRGB 数据二次编码。
        self._pv_src_curve = "Linear"
        # 显存驻留（4.0）：预览源与 LUT 上传一次，拖滑块时不再重复上传。
        # 用「代次」而不是 id() 判失效——id() 会被复用，是经典陷阱。
        self._pv_gen = 0
        self._g_gen = -1
        self._g_lin = None; self._g_lin_small = None
        self._g_lut = None; self._g_lut_key = None; self._g_lut_n = 0; self._g_lut_sc = 1.0
        self._pv_prod = None; self._pv_full = None; self._pv_has = False
        self._pv_fit = True; self._pv_zoom_val = 1.0; self._pv_s = 1.0
        self._pv_drag_data = None
        self._scope_drag_data = None; self._scope_photo = None; self._scope_rgb = False
        self._scope_after = None; self._sel_row = None; self._row_frames = []
        self._pv_cache = {}; self._preload_after = None; self._preload_gen = 0
        self._hist = []; self._hist_idx = -1
        self._thumb_cache = {}; self._thumb_cv = []; self._thumb_photos = []
        self._thumb_gen = 0; self._thumb_busy = False; self._thumb_pending = []
        c = _cl()
        self._dpi = c.get("dpi", 300)
        self._png_level = _png_level(c)     # PNG 压缩级别（4.0），默认 3
        self._od = c.get("out", "") or str(Path(__file__).parent / "output")
        self._id = c.get("img", ""); self._ld = c.get("lut", "")
        self._sg = c.get("sg", "S-Gamut3.Cine"); self._sc = c.get("sc", "S-Log3")
        # 非 RAW 的「源曲线」是**用户声明**，必须与 RAW 时代码强制写进下拉框的显示值分开存。
        # 否则会出现这个链条：打开 RAW → _update_pipe 把 cv 设成 Linear（RAW 的正确显示）
        # → _save_cfg 又把它当成用户选择存下来 → 下次打开 JPEG 时源曲线仍是 Linear
        # → 管线把 sRGB 数据当线性处理 → 预览整体跑偏。
        # （3.3 时代强制的是 sRGB，恰好也是 JPEG 的合理默认值，所以这个 bug 一直没暴露。）
        self._sc_user = self._sc
        # **色域**也要有同样的"用户声明"存储：上面这段注释描述的正是同类 bug
        # （RAW 分支把控件值机械写进配置，下次打开 JPEG 就残留），
        # 但当时只修了曲线，色域一直没修 —— 于是 sRGB JPEG 被当成 ACES2065-1 解释。
        self._icc_note = ""
        self._sg_user = c.get("sg_user", self._sg)
        if self._sg_user not in GAMUT_NAMES:
            self._sg_user = "sRGB"
        _lastf = (c.get("files") or [""])[-1]
        if self._sc_user == "Linear" and _lastf and not _is_raw(_lastf):
            self._sc_user = "sRGB"; self._sc = "sRGB"     # 一次性修复旧代码机械写入的值
        self._td = c.get("tg", "sRGB"); self._tc = c.get("tc", "sRGB")
        self._li = c.get("li", "sRGB")
        self._raw_work = c.get("rw", _RAW_WORK_DEF)
        global _HL_MODE
        self._hl = c.get("hl", _HL_DEF)
        if self._hl not in _HL_MODES:
            self._hl = _HL_DEF
        _HL_MODE = self._hl
        self._g_ev = c.get("ev", 0.0); self._g_ct = c.get("ct", 0.0)
        self._g_sat = c.get("sat", 0.0)
        self._g_tp = float(c.get("tp", 6500.0)); self._g_tp = 2000.0 if self._g_tp < 2000.0 else min(10000.0, self._g_tp)
        # 色温：默认落在"机内 WB 的等效色温"上，用户在区间内主观调整
        self._cg = c.get("cm", True)          # 色彩管理默认开（关掉即旧的无管理行为）
        self._cfg_cm = bool(c.get("cm", True))
        self._cfg_gw = bool(c.get("gw", False))   # 色域警告默认关（诊断叠加，不该默认糊在图上）
        self._gw_ratio = None                     # 最近一帧「超显示器色域」占比
        self._wb_ref = 6500.0        # 当前图片的机内等效色温；未知时退回 6500
        self._wb_manual = False      # 用户是否手动调过（调过之后切图不再自动覆盖滑块）
        self._g_tt = float(c.get("tt", 0.0)); self._g_tt = max(-150.0, min(150.0, self._g_tt))
        self._lut_fav = list(c.get("lut_fav", []) or [])      # 收藏（置顶）
        self._lo = c.get("lo", "sRGB") if c.get("lo") in GAMUT_NAMES else "sRGB"
        self._loc = c.get("loc", "sRGB") if c.get("loc") in CURVE_NAMES else "sRGB"
        self._luts = _sl(); self._decorate_luts()
        self._cd = None; self._cd_path = ""
        self._build()
        self.after(150, self._fit_window)

    # ---------- 控件构建 ----------
    def _fit_window(self):
        """把窗口拉回屏幕内。

        窗口一旦比屏幕宽、或右边缘跑到屏幕外，右侧内容（波形示波器右半边、导出栏的
        「浏览」按钮）就被切掉——这是"可见性"问题里最容易被忽略的一个。只在确实越界时
        才调整，最大化/图标化状态一律不动。
        """
        try:
            if self.state() != "normal":
                return
            self.update_idletasks()
            sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
            w, h, x, y = self.winfo_width(), self.winfo_height(), self.winfo_x(), self.winfo_y()
            nw, nh = min(w, sw - 40), min(h, sh - 90)
            nx = min(max(x, 8), max(8, sw - nw - 8))
            ny = min(max(y, 8), max(8, sh - nh - 48))
            if (nw, nh, nx, ny) != (w, h, x, y):
                self.geometry(f"{nw}x{nh}+{nx}+{ny}")
        except Exception:
            pass

    def _sec(self, parent, title):
        f = ctk.CTkFrame(parent, corner_radius=8)
        f.pack(fill="x", pady=(0, 4))
        ctk.CTkLabel(f, text=title, font=("Microsoft YaHei", 10, "bold"), text_color="#A0A0A0").pack(anchor="w", padx=8, pady=(3, 1))
        return f

    def _row(self, parent):
        r = ctk.CTkFrame(parent, fg_color="transparent")
        r.pack(fill="x", pady=0, padx=4)
        return r

    def _build(self):
        ctk.CTkLabel(self, text=f"曲线 {len(CURVE)} | 色域 {len(GAMUT_NAMES)} | LUT {len(self._luts)} | 空格=原图/效果",
                     font=("Microsoft YaHei", 9), text_color="#888").pack(pady=(4, 0))
        self._build_top_export()     # 一级导出栏：常驻顶部、不随左栏滚动
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=8, pady=(0, 4))
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(0, weight=1)
        left = ctk.CTkFrame(body, width=340)
        left.grid(row=0, column=0, sticky="nsw", padx=(0, 8))
        left.grid_propagate(False)          # 固定左栏宽度，避免内容撑宽
        left.grid_rowconfigure(0, weight=1)
        left.grid_columnconfigure(0, weight=1)
        self._lf_ctrl = ctk.CTkScrollableFrame(left, width=320)
        self._lf_ctrl.grid(row=0, column=0, sticky="nsew")
        right = ctk.CTkFrame(body, fg_color="transparent")
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(1, weight=1)
        self._build_left(self._lf_ctrl)
        self._build_right(right)
        self._build_files(left)
        self._bind_resize()
        self._restore()     # 先恢复上次的文件列表（避免 _pk 的 _save_cfg 用空列表覆盖记忆）
        self._pk()          # 启动即加载默认 LUT，否则勾选 LUT 会无效
        self._on_disc()

    def _build_top_export(self):
        """一级导出栏：常驻窗口顶部（不随左栏滚动），导出当前图一步到位。"""
        bar = ctk.CTkFrame(self, fg_color="transparent")
        bar.pack(fill="x", padx=8, pady=(2, 0))
        # 导出当前图：大按钮 + 格式 + 输出路径，永不见底
        row = ctk.CTkFrame(bar, fg_color="transparent")
        row.pack(fill="x")
        self.gc = ctk.CTkButton(row, text="导出当前图", command=self._export_current, height=36,
                                font=("Microsoft YaHei", 14, "bold"),
                                fg_color="#2a7a4b", hover_color="#1f6e3f")
        self.gc.pack(side="left", padx=(0, 8))
        ctk.CTkLabel(row, text="格式", font=("Microsoft YaHei", 11)).pack(side="left", padx=(0, 4))
        self.fd = ctk.CTkOptionMenu(row, values=list(_OUT_FMTS.keys()), width=130)
        self.fd.pack(side="left", padx=(0, 8)); self.fd.set("PNG 8-bit")
        self.ol = ctk.CTkLabel(row, text=self._short_path(self._od), width=260,
                               fg_color=("gray90", "gray25"), corner_radius=4, anchor="w", height=30)
        self.ol.pack(side="left", fill="x", expand=True, padx=(0, 4))
        ctk.CTkButton(row, text="浏览", width=56, command=self._po).pack(side="left")
        self.st = ctk.CTkLabel(bar, text="就绪", font=("Microsoft YaHei", 10), anchor="w")
        self.st.pack(fill="x", padx=2, pady=(1, 0))

    def _build_left(self, L):
        # ---- 切换模式：色彩转换 / 基础调色（同一时间只显示一块，给文件区腾空间）----
        bar = ctk.CTkFrame(L, fg_color="transparent")
        bar.pack(fill="x", padx=6, pady=(0, 4))
        self._mode_sw = ctk.CTkSegmentedButton(bar, values=["色彩转换", "基础调色"],
                                               command=self._switch_mode, font=("Microsoft YaHei", 11))
        self._mode_sw.pack(fill="x")
        self._mode_sw.set("色彩转换")
        self._mode = "色彩转换"
        self._stack = ctk.CTkFrame(L, fg_color="transparent")
        self._stack.pack(fill="x")
        self._cst_panel = ctk.CTkFrame(self._stack, fg_color="transparent")
        self._grade_panel = ctk.CTkFrame(self._stack, fg_color="transparent")
        self._cst_panel.pack(fill="x")

        # ---- CST ----
        s = self._sec(self._cst_panel, "色彩空间转换")
        r = self._row(s)
        ctk.CTkLabel(r, text="RAW工作空间", width=72, font=("Microsoft YaHei", 10)).pack(side="left")
        self.rw = ctk.CTkOptionMenu(r, values=list(_RAW_WORK_OPT), width=110,
                                    command=self._rw_changed)
        self.rw.pack(side="left", padx=2); self.rw.set(self._raw_work)
        r = self._row(s)
        ctk.CTkLabel(r, text="高光重建", width=72, font=("Microsoft YaHei", 10)).pack(side="left")
        self.hl = ctk.CTkOptionMenu(r, values=list(_HL_MODES), width=130,
                                    command=self._hl_changed)
        self.hl.pack(side="left", padx=2); self.hl.set(self._hl)
        self.hl_hint = ctk.CTkLabel(s, text="", font=("Microsoft YaHei", 9), text_color="#777")
        self.hl_hint.pack(anchor="w", padx=8)
        self.after(60, self._hl_hint_update)
        ctk.CTkLabel(r, text="源", width=26).pack(side="left")
        self.sd = ctk.CTkOptionMenu(r, values=GAMUT_NAMES, width=125, command=self._sd_changed)
        self.sd.pack(side="left", padx=2); self.sd.set(self._sg)
        self.cv = ctk.CTkOptionMenu(r, values=CURVE_NAMES, width=85, command=self._cv_changed)
        self.cv.pack(side="left", padx=2); self.cv.set(self._sc)
        self.src_hint = ctk.CTkLabel(s, text="", font=("Microsoft YaHei", 9), text_color="#777")
        self.src_hint.pack(anchor="w", padx=8)
        r = self._row(s)
        ctk.CTkLabel(r, text="目标", width=26).pack(side="left")
        self.td = ctk.CTkOptionMenu(r, values=GAMUT_NAMES, width=125, command=self._td_changed)
        self.td.pack(side="left", padx=2); self.td.set(self._td)
        self.cv2 = ctk.CTkOptionMenu(r, values=CURVE_NAMES, width=85, command=lambda _: self._on_disc())
        self.cv2.pack(side="left", padx=2); self.cv2.set(self._tc)
        # 说真话：启用 LUT 时像素不再做目标转换（LUT 输出即最终画面），目标只写进文件的 ICC 描述
        ctk.CTkLabel(s, text="目标：非 LUT 时输出到此空间；启用 LUT 时像素不再转目标（LUT 输出即最终画面），"
                             "此处仅写进文件的 ICC 描述",
                     font=("Microsoft YaHei", 9), text_color="#777", wraplength=286,
                     justify="left").pack(anchor="w", padx=8)
        ctk.CTkLabel(s, text="OOTF", font=("Microsoft YaHei", 10), text_color="#A0A0A0").pack(anchor="w", padx=8, pady=(3,0))
        self._ootf_sw = ctk.CTkSegmentedButton(s, values=["自动", "反向", "正向", "关"],
                                               command=lambda _: self._on_disc(), font=("Microsoft YaHei", 10))
        self._ootf_sw.pack(fill="x", padx=8)
        self._ootf_sw.set("自动")
        self._ootf_lbl = ctk.CTkLabel(s, text="", font=("Microsoft YaHei", 9), text_color="#8b8")
        self._ootf_lbl.pack(anchor="w", padx=8)

        # ---- LUT ----
        s = self._sec(self._cst_panel, "LUT")
        self.lv = ctk.StringVar(value="off")
        ctk.CTkCheckBox(s, text="启用 LUT", variable=self.lv, onvalue="on", offvalue="off",
                        command=self._tg).pack(anchor="w", padx=8, pady=(0, 1))
        self.lf = ctk.CTkFrame(s, fg_color="transparent")
        self.lf.pack(fill="x", padx=6)
        r = self._row(self.lf)
        self.ld = ctk.CTkOptionMenu(r, values=[l["disp"] for l in self._luts] or ["(无)"], width=150,
                                    text_color="#FF8A16", dropdown_text_color="#FF8A16",
                                    command=lambda _: self._pk())
        self.ld.pack(side="left", padx=2)
        if self._luts: self.ld.set(self._luts[0]["disp"])
        ctk.CTkButton(r, text="导入 .cube", width=72, command=self._im).pack(side="left", padx=2)
        r = self._row(self.lf)
        self._lut_fav_btn = ctk.CTkButton(r, text="\u2606 收藏", width=72, height=24,
                                          font=("Microsoft YaHei", 10), fg_color="#4a3a1a",
                                          hover_color="#5f4a22", command=self._lut_toggle_fav)
        self._lut_fav_btn.pack(side="left", padx=2)
        ctk.CTkButton(r, text="在文件夹中显示", width=104, height=24, font=("Microsoft YaHei", 10),
                      fg_color="#3a3a3a", hover_color="#4a4a4a",
                      command=self._lut_reveal).pack(side="left", padx=2)
        ctk.CTkButton(r, text="删除此 LUT", width=88, height=24, font=("Microsoft YaHei", 10),
                      fg_color="#7a2b2b", hover_color="#9a3b3b",
                      command=self._lut_delete).pack(side="left", padx=2)
        self._info = ctk.CTkLabel(self.lf, text="", font=("Consolas", 10), anchor="w", text_color="#8b8")
        self._info.pack(fill="x", padx=6)
        self._lut_path = ctk.CTkLabel(self.lf, text="", font=("Consolas", 9), anchor="w", text_color="#666")
        self._lut_path.pack(fill="x", padx=6)
        r = self._row(self.lf)
        ctk.CTkLabel(r, text="LUT 期望输入", width=78, font=("Microsoft YaHei", 10)).pack(side="left")
        self.li = ctk.CTkOptionMenu(r, values=GAMUT_NAMES, width=125, command=self._li_changed)
        self.li.pack(side="left", padx=2); self.li.set(self._li)
        self.lc_hint = ctk.CTkLabel(r, text="", font=("Microsoft YaHei", 9), text_color="#777")
        self.lc_hint.pack(side="left")
        r = self._row(self.lf)
        ctk.CTkLabel(r, text="LUT 输出", width=78, font=("Microsoft YaHei", 10)).pack(side="left")
        self.lo = ctk.CTkOptionMenu(r, values=GAMUT_NAMES, width=125, command=self._lo_changed)
        self.lo.pack(side="left", padx=2); self.lo.set(self._lo)
        self.loc = ctk.CTkOptionMenu(r, values=CURVE_NAMES, width=85, command=self._lo_changed)
        self.loc.pack(side="left", padx=2); self.loc.set(self._loc)
        ctk.CTkLabel(self.lf, text="LUT 输出即最终画面。输出空间必须如实声明 —— "
                                   "监看转换与导出 ICC 都按它来；胶片 LUT 常输出 Rec.709 或 Log，"
                                   "按 sRGB 看会偏（Log 尤其发灰发平）",
                     font=("Microsoft YaHei", 9), text_color="#777",
                     wraplength=300, justify="left").pack(anchor="w", padx=8)

        # ---- 末端 CST 提示（不再是独立开关）----
        # LUT 输出空间 → **目标**（tg/tc）这一级 CST 永远存在。
        # 「目标」= 文件最终所处的空间；想要 LUT 的原值就把目标设成上面这个空间。
        self.ocst_hint = ctk.CTkLabel(self.lf, text="", font=("Microsoft YaHei", 9),
                                      text_color="#777", wraplength=300, justify="left")
        self.ocst_hint.pack(anchor="w", padx=8)
        self._ocst_hint_update()

        # ---- 调色（基础调色面板）----
        s = self._sec(self._grade_panel, "基础调色")
        self.sl_ev, self.en_ev = self._slider(s, "曝光EV", -3.0, 3.0, self._g_ev)
        self.sl_ct, self.en_ct = self._slider(s, "对比", -0.6, 0.6, self._g_ct)
        self.sl_sat, self.en_sat = self._slider(s, "饱和", -1.0, 1.0, self._g_sat)
        self.sl_tp, self.en_tp = self._slider(s, "色温K", 2000.0, 10000.0, self._g_tp, fmt="{:.0f}")
        self.sl_tp.configure(command=lambda v: (self.en_tp.delete(0, "end"), self.en_tp.insert(0, "{:.0f}".format(v)), self._wb_changed()))
        self.sl_tt, self.en_tt = self._slider(s, "色调", -150.0, 150.0, self._g_tt, fmt="{:.0f}")
        self.sl_tt.configure(command=lambda v: (self.en_tt.delete(0, "end"), self.en_tt.insert(0, "{:.0f}".format(v)), self._wb_changed()))
        # 色温基准提示 + 「按机内」复位（WB 是主观操作：默认落在机内，用户可在区间内自由调）
        self._wb_hint = ctk.CTkLabel(s, text="", font=("Microsoft YaHei", 9), text_color="#777",
                                     wraplength=286, justify="left")
        self._wb_hint.pack(anchor="w", padx=8)
        ctk.CTkButton(s, text="按机内", width=64, height=22, font=("Microsoft YaHei", 10),
                      command=self._wb_reset).pack(anchor="w", padx=8, pady=(0, 2))

        # ---- 输出（批量；单张导出在顶部一级栏）----
        s = self._sec(L, "批量输出")
        self._dither_var = ctk.StringVar(value="on")
        ctk.CTkCheckBox(s, text="输出抖动（消除色阶断层）", variable=self._dither_var,
                        onvalue="on", offvalue="off", font=("Microsoft YaHei", 10)).pack(anchor="w", padx=8)
        r = self._row(s)
        self.gb = ctk.CTkButton(r, text="批量转换", command=self._go, height=30, state="disabled",
                                font=("Microsoft YaHei", 13, "bold"))
        self.gb.pack(fill="x", padx=6)
        self.pb = ctk.CTkProgressBar(s, mode="determinate")
        self.pb.pack(fill="x", padx=8, pady=(0, 2)); self.pb.set(0)
        self.pl = ctk.CTkLabel(s, text="", font=("Consolas", 9), anchor="w", text_color="#A0A0A0")
        self.pl.pack(fill="x", padx=10, pady=(0, 6))

        # ---- 生产画幅 / DPI ----
        s = self._sec(L, "生产画幅")
        self._prod_lbl = ctk.CTkLabel(s, text="未加载图片", font=("Consolas", 10), anchor="w")
        self._prod_lbl.pack(fill="x", padx=10)
        r = self._row(s)
        ctk.CTkLabel(r, text="DPI", width=34).pack(side="left")
        self._dpi_ent = ctk.CTkEntry(r, width=70)
        self._dpi_ent.insert(0, str(self._dpi))
        self._dpi_ent.pack(side="left", padx=4)
        self._print_lbl = ctk.CTkLabel(r, text="", font=("Microsoft YaHei", 10), text_color="#888")
        self._print_lbl.pack(side="left", padx=6)
        self._dpi_ent.bind("<KeyRelease>", lambda e: self._on_dpi())

    def _switch_mode(self, m):
        """色彩转换 / 基础调色 面板切换"""
        self._mode = m
        if m == "色彩转换":
            self._grade_panel.pack_forget(); self._cst_panel.pack(fill="x")
        else:
            self._cst_panel.pack_forget(); self._grade_panel.pack(fill="x")

    def _slider(self, parent, name, lo, hi, val, fmt="{:+.2f}", steps=240):
        """滑块 + 数值输入框联动。返回 (slider, entry)。双击名称回归默认值"""
        r = self._row(parent)
        nl = ctk.CTkLabel(r, text=name, width=44, cursor="hand2")
        nl.pack(side="left")
        nl.bind("<Double-Button-1>", lambda e: self._slider_reset(s, ent, val, fmt))
        s = ctk.CTkSlider(r, from_=lo, to=hi, number_of_steps=steps)
        s.set(val); s.pack(side="left", fill="x", expand=True, padx=6)
        ent = ctk.CTkEntry(r, width=56, font=("Consolas", 10))
        ent.insert(0, fmt.format(val))
        ent.pack(side="left", padx=(0, 2))
        def _on_slider(v):
            ent.delete(0, "end"); ent.insert(0, fmt.format(v))
            self._dirty()
        s.configure(command=_on_slider)
        def _commit(_e=None):
            try:
                v = max(lo, min(hi, float(ent.get())))
                ent.delete(0, "end"); ent.insert(0, fmt.format(v))
                s.set(v); self._dirty()
            except Exception:
                pass
        ent.bind("<Return>", _commit)
        ent.bind("<FocusOut>", _commit)
        def _live(_e):
            try:
                v = float(ent.get())
                if lo <= v <= hi:
                    s.set(v); self._dirty()
            except Exception:
                pass
        ent.bind("<KeyRelease>", _live)
        return s, ent

    def _slider_reset(self, s, ent, val, fmt):
        """双击滑块名称：回归默认值（原位）"""
        s.set(val)
        ent.delete(0, "end"); ent.insert(0, fmt.format(val))
        self._dirty()

    def _build_right(self, R):
        R.grid_columnconfigure(0, weight=1)
        R.grid_rowconfigure(1, weight=1)
        bar = ctk.CTkFrame(R, fg_color="transparent")
        bar.grid(row=0, column=0, sticky="ew", padx=6)
        # 场景参考（Log/Linear）内容在显示端**无法被正确渲染** —— 渲染没有唯一解
        # （色域超出、白点意图、观看环境都是创作判断）。做不到就必须说做不到。
        self._pv_warn = ctk.CTkLabel(bar, text="", font=("Microsoft YaHei", 9, "bold"),
                                     text_color="#ffb020")
        self._pv_warn.pack(side="left", padx=(10, 0))
        self._pv_orig_ck = ctk.CTkCheckBox(bar, text="原图", command=self._pv_toggle)
        self._pv_orig_ck.pack(side="left")
        # 色彩管理（4.0）：把「目标空间」转成「显示器空间」再上屏。
        # 关掉它 = 旧行为（目标空间像素直显）—— 只有目标恰好是 sRGB 时才正确，
        # 留这个开关是为了能直观对比"有没有色彩管理"的差别。
        self._cm_on = bool(self._cfg_cm)
        self._cm_ck = ctk.CTkCheckBox(bar, text="色彩管理", command=self._cm_toggle)
        self._cm_ck.pack(side="left")
        (self._cm_ck.select if self._cm_on else self._cm_ck.deselect)()
        # 色域警告：把「超出显示器色域、被裁掉」的像素标成品红。
        # 它依赖色彩管理（要先算出显示器空间才知道哪些越界），所以 CM 关时置灰。
        self._gw_on = bool(self._cfg_gw)
        self._gw_ck = ctk.CTkCheckBox(bar, text="色域警告", command=self._gw_toggle)
        self._gw_ck.pack(side="left")
        (self._gw_ck.select if self._gw_on else self._gw_ck.deselect)()
        self._sync_gw_state()
        ctk.CTkLabel(bar, text="缩放", font=("Microsoft YaHei", 10)).pack(side="left", padx=(16, 2))
        self._pv_zminus = ctk.CTkButton(bar, text="−", width=26, height=24, font=("Microsoft YaHei", 12, "bold"),
                                        command=lambda: self._zoom_btn(-1))
        self._pv_zminus.pack(side="left", padx=1)
        self._pv_zlbl = ctk.CTkLabel(bar, text="适应", width=46, font=("Consolas", 11))
        self._pv_zlbl.pack(side="left", padx=2)
        self._pv_zplus = ctk.CTkButton(bar, text="+", width=26, height=24, font=("Microsoft YaHei", 12, "bold"),
                                       command=lambda: self._zoom_btn(1))
        self._pv_zplus.pack(side="left", padx=1)
        self._pv_zfit = ctk.CTkButton(bar, text="适应", width=42, height=24, font=("Microsoft YaHei", 10),
                                      command=self._zoom_fit)
        self._pv_zfit.pack(side="left", padx=4)
        self._pv_dims = ctk.CTkLabel(bar, text="", font=("Consolas", 10), text_color="#888")
        self._pv_dims.pack(side="left", padx=12)
        self._pv_pxl = ctk.CTkLabel(bar, text="", font=("Consolas", 10), text_color="#8b8")
        self._pv_pxl.pack(side="right", padx=8)
        pv = ctk.CTkFrame(R, fg_color="#2d2d2d", corner_radius=8)
        pv.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
        pv.grid_rowconfigure(0, weight=1); pv.grid_columnconfigure(0, weight=1)
        self._pv_canvas = Canvas(pv, bg="#191919", highlightthickness=0, bd=0)
        self._pv_canvas.grid(row=0, column=0, sticky="nsew")
        sb_x = ttk.Scrollbar(pv, orient="horizontal", command=self._pv_canvas.xview)
        sb_y = ttk.Scrollbar(pv, orient="vertical", command=self._pv_canvas.yview)
        sb_x.grid(row=1, column=0, sticky="ew")
        sb_y.grid(row=0, column=1, sticky="ns")
        self._pv_canvas.configure(xscrollcommand=sb_x.set, yscrollcommand=sb_y.set)
        self._pv_canvas.bind("<MouseWheel>", self._wheel)
        self._pv_canvas.bind("<Motion>", self._pv_motion)
        self._pv_canvas.bind("<Double-Button-1>", self._dc)
        self._pv_canvas.bind("<Button-1>", self._pv_drag_start)
        self._pv_canvas.bind("<B1-Motion>", self._pv_drag)
        self._pv_canvas.bind("<ButtonRelease-1>", self._pv_drag_end)
        # 波形示波器：叠在预览区右上角。
        # 缩放用**滚轮 + 拖右下角**（双击复位），不再用 +/- 按钮 ——
        # 那样每点一次只跳固定步长，既慢又没法微调。
        self._scope = ctk.CTkFrame(pv, fg_color="#242424", corner_radius=6,
                                   border_width=1, border_color="#4a4a4a")
        self._scope.place(relx=1.0, x=-28, y=10, anchor="ne")
        tb = ctk.CTkFrame(self._scope, fg_color="transparent")
        tb.pack(fill="x", padx=6, pady=(4, 0))
        ctk.CTkLabel(tb, text="波形", font=("Microsoft YaHei", 9),
                     text_color="#bbb").pack(side="left", padx=(0, 6))
        self._scope_mode = ctk.CTkButton(tb, text="Y", width=86, height=20,
                                         font=("Microsoft YaHei", 9),
                                         fg_color="#3a3a3a", hover_color="#4d4d4d",
                                         command=self._scope_toggle_mode)
        self._scope_mode.pack(side="left")
        ctk.CTkLabel(tb, text="滚轮缩放 · 拖右下角 · 双击复位", font=("Microsoft YaHei", 7),
                     text_color="#6f6f6f").pack(side="right", padx=(4, 0))
        self._scope_cv = Canvas(self._scope, width=560, height=300, bg="#0f0f0f",
                                highlightthickness=0, bd=0, cursor="crosshair")
        self._scope_cv.pack(padx=6, pady=(3, 6))
        for w in (self._scope, self._scope_cv):
            w.bind("<Button-1>", self._scope_drag_start)
            w.bind("<B1-Motion>", self._scope_drag)
            w.bind("<ButtonRelease-1>", self._scope_drag_end)
        for w in (self._scope, self._scope_cv, tb, self._scope_mode):
            w.bind("<MouseWheel>", self._scope_wheel)
        self._scope_cv.bind("<Double-Button-1>", lambda _e: self._scope_reset_size())

    def _build_files(self, left):
        f = ctk.CTkFrame(left, corner_radius=8)
        f.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        bar = ctk.CTkFrame(f, fg_color="transparent")
        bar.pack(fill="x", padx=8, pady=(6, 2))
        ctk.CTkLabel(bar, text="文件", font=("Microsoft YaHei", 11, "bold")).pack(side="left")
        self.cn = ctk.CTkLabel(bar, text="0 个", font=("Microsoft YaHei", 10)); self.cn.pack(side="right")
        ctk.CTkButton(bar, text="+文件", width=58, height=24, command=self._af).pack(side="right", padx=2)
        ctk.CTkButton(bar, text="+文件夹", width=66, height=24, command=self._ad).pack(side="right", padx=2)
        ctk.CTkButton(bar, text="清空", width=46, height=24, command=self._cl).pack(side="right", padx=2)
        ctk.CTkLabel(bar, text="点文件名=预览", font=("Microsoft YaHei", 9), text_color="#777").pack(side="left", padx=8)
        self.sf = ctk.CTkScrollableFrame(f, height=400)
        self.sf.pack(fill="x", padx=8, pady=(0, 6))
        self.sf.grid_columnconfigure(0, weight=1)
        self.sf.grid_columnconfigure(1, weight=1)

    # ---------- 状态 / 快照 ----------
    def _snapshot(self):
        li = self.li.get()
        _raw = bool(self._preview_file and _is_raw(self._preview_file))
        # 源侧统一解析：RAW 由解码层决定（工作空间 + sRGB 曲线），非 RAW 用用户选择。
        # 这样任何消费 snapshot 的地方（预览 / 导出 / 将来的新入口）都不可能再拿到"假的源空间"。
        _sg, _sc = resolve_source({"raw": _raw, "rw": self._raw_work,
                                   "sg": self.sd.get(), "sc": self._sc_user}, self._raw_work)
        return {"sg": _sg, "sc": _sc,
                "raw": _raw,
                "rw": self._raw_work,
                "tg": self.td.get(), "tc": self.cv2.get(),
                "lut": self.lv.get() == "on",
                # 直接把「自动」解析成明确方向再往下传 —— 管线层不再需要 _ref_type 猜
                "ootf": self._ootf_direction(),
                "ootf2": self._ootf_direction2(),
                "li": li, "lc": _CURVE_MATCH.get(li, self.cv.get()),
                "cd": self._cd, "cd_path": self._cd_path,
                "lo": self.lo.get(), "loc": self.loc.get(),
                "ev": float(self.sl_ev.get()), "ct": float(self.sl_ct.get()),
                "sat": float(self.sl_sat.get()), "tp": float(self.sl_tp.get()),
                "dither": self._dither_var.get() == "on"}

    def _transform(self, a, s):
        # 实际处理链在模块级 transform()，App / 基准脚本 / 离线工具共用同一条路
        return transform(a, s, self._raw_work)

    def _transform_tiled(self, a, s):
        return _tiled(a, lambda x: self._transform(x, s))

    def _ootf_direction2(self):
        """末端那一级 CST（LUT 输出空间 → 目标）的方向。

        与主 CST 同一套判定：**参考类型由曲线决定，不由色域**，且两侧都先过
        `display_decode_curve`（真正拿来解码的是显示 EOTF）。
          · 显示参考 → 显示参考  → off
          · 非显示参考 → 显示参考 → forward（场景参考 LUT 输出在这里被渲染）
          · 显示参考 → 非显示参考 → reverse
        """
        if self.lv.get() != "on" or self._cd is None:
            return "off"
        sc = display_decode_curve(self.loc.get())
        tc = display_decode_curve(self.cv2.get())
        if _ref_type(sc) == "display" and _ref_type(tc) != "display":
            return "reverse"
        if _ref_type(tc) == "display" and _ref_type(sc) != "display":
            return "forward"
        return "off"

    def _update_pipe(self):
        tg, tc = self.td.get(), self.cv2.get()
        _raw = bool(self._preview_file and _is_raw(self._preview_file))
        sg, sc = resolve_source({"raw": _raw, "rw": self._raw_work,
                                 "sg": self.sd.get(), "sc": self.cv.get()}, self._raw_work)
        # UI 必须与管线说同一件事：
        #   RAW   → 源色域/曲线锁死为解码层的实际值（rw + sRGB），可调的是「RAW工作空间」
        #   非 RAW → 源色域/曲线由用户选，「RAW工作空间」不参与（置灰）
        try:
            if _raw:
                if self.sd.get() != sg:
                    self.sd.configure(command=None); self.sd.set(sg); self.sd.configure(command=self._sd_changed)
                if self.cv.get() != sc:
                    self.cv.configure(command=None); self.cv.set(sc)
                    self.cv.configure(command=self._cv_changed)
                self.sd.configure(state="disabled"); self.cv.configure(state="disabled")
                self.rw.configure(state="normal")
                self._pv_warn_update()
                self.src_hint.configure(text="RAW：源侧由解码层决定（工作空间色域 + Linear 曲线），不可选；"
                                             "可调的是上面的 RAW 工作空间")
            else:
                # 恢复**用户声明**的源空间：上一次若是 RAW，两个下拉框里
                # 留的是解码层的值（工作空间色域 + Linear）。**色域和曲线都要恢复** ——
                # 此前只恢复了曲线，于是 sRGB JPEG 被当成 ACES2065-1 解释。
                if self.sd.get() != self._sg_user:
                    self.sd.configure(command=None); self.sd.set(self._sg_user)
                    self.sd.configure(command=self._sd_changed)
                if self.cv.get() != self._sc_user:
                    self.cv.configure(command=None); self.cv.set(self._sc_user)
                    self.cv.configure(command=self._cv_changed)
                self.sd.configure(state="normal"); self.cv.configure(state="normal")
                self.rw.configure(state="disabled")
                self._pv_warn_update()
                self.src_hint.configure(
                    text="非 RAW：源色域 / 曲线由你指定；RAW 工作空间不参与（已置灰）"
                         + (f"　｜　{self._icc_note}" if getattr(self, "_icc_note", None) else ""))
        except Exception:
            pass          # UI 尚未建好时不影响管线
        # 色温：默认锚点 = 机内 WB 的等效色温（未手动调过就随图片走）
        _c = shot_cct(self._preview_file) if _raw else None
        self._wb_ref = float(_c) if _c else 6500.0
        try:
            if _raw and (not self._wb_manual) and abs(float(self.sl_tp.get()) - self._wb_ref) > 1.0:
                self.sl_tp.set(round(self._wb_ref))
                self.en_tp.delete(0, "end"); self.en_tp.insert(0, "{:.0f}".format(self._wb_ref))
            self._wb_hint.configure(
                text=(f"色温基准：机内 ≈{self._wb_ref:.0f}K（读自 EXIF；拖动即主观调整，可点「按机内」复位）"
                      if (_raw and _c) else
                      ("色温基准：机内未知（EXIF 无 WB 信息）→ 退回 6500K" if _raw else
                       "非 RAW：色温为显示域调整，基准 6500K")))
        except Exception:
            pass
        if self.lv.get() == "on" and self._cd:
            li = self.li.get()
            txt = f"{sg}/{sc} → 匹配LUT输入[{li}/{_CURVE_MATCH.get(li, '')}] → LUT[{self.ld.get()}] → 输出"
        else:
            txt = f"{sg}/{sc} → {tg}/{tc}"
        if any(abs(self.sl_ev.get()) > .001 or abs(self.sl_ct.get()) > .001 or
               abs(self.sl_sat.get()) > .001 or abs(float(self.sl_tp.get()) - self._wb_ref) > 50.0
               for _ in [0]):
            txt += " + 调色"
        self.pl.configure(text=txt)

    def _dirty(self):
        self._update_pipe()
        if self._pv_after:
            try: self.after_cancel(self._pv_after)
            except Exception: pass
        self._pv_after = self.after(35, self._refresh_preview)

    def _wb_args(self):
        """统一给解码层的白平衡参数：(目标K, 色调, 机内基准K)。"""
        return (float(self.sl_tp.get()), float(self.sl_tt.get()), float(self._wb_ref))

    def _wb_reset(self):
        """回到机内 WB：滑块落回机内等效色温、色调归零。"""
        self.sl_tp.set(round(self._wb_ref))
        self.en_tp.delete(0, "end"); self.en_tp.insert(0, "{:.0f}".format(self._wb_ref))
        self.sl_tt.set(0.0)
        self.en_tt.delete(0, "end"); self.en_tt.insert(0, "0")
        self._wb_changed()
        self._wb_manual = False      # 复位后仍算"按机内"

    def _wb_changed(self):
        """色温/色调滑块：RAW 需重解码（WB 烘焙在解码层）。
        防抖 + 后台线程重解码，拖动不阻塞 UI；非 RAW 走显示域调色。"""
        self._wb_manual = True       # 用户主观调整过
        self._save_cfg()
        if self._preview_file and _is_raw(self._preview_file):
            self._pv_cache.clear()                 # 旧 WB 缓存失效
            self._preload_gen += 1                 # 使过期预载作废
            if getattr(self, "_wb_after", None):
                try: self.after_cancel(self._wb_after)
                except Exception: pass
            self._wb_after = self.after(250, self._wb_redecode)
        else:
            self._dirty()

    def _wb_redecode(self):
        """后台线程重解码当前 RAW（含新 WB），完成后主线程刷新预览"""
        self._wb_after = None
        rp = self._preview_file
        if not rp: return
        gen = self._preload_gen
        def work():
            try:
                a = _load_float(rp, half=True, work=self._raw_work,
                                wb=self._wb_args(), encode=False)      # 线性
                lin = np.ascontiguousarray(a, dtype=np.float32)
                small = self._make_small_lin(lin)
            except Exception as e:
                print(f"[wb] {e}"); return
            if gen != self._preload_gen or rp != self._preview_file:
                return   # 列表/文件已变，丢弃过期结果
            self.after(0, lambda: self._pv_apply_wb(lin, small))
        threading.Thread(target=work, daemon=True).start()

    def _pv_apply_wb(self, lin, small):
        """主线程应用重解码结果（WB 变化不重建 _pv_prod，仅换源图）"""
        self._pv_lin = lin
        self._pv_lin_small = small
        self._pv_src_curve = "Linear" if (self._preview_file and _is_raw(self._preview_file)) else self._sc_user
        self._pv_gen += 1              # 显存缓存随之失效，下帧重传
        self._dirty()

    def _cv_changed(self, _=None):
        """用户手改「源曲线」。这是**用户声明**，单独记住：
        RAW 预览时下拉框会被锁死并显示 Linear，那不是用户的选择，不能覆盖这里。"""
        self._sc_user = self.cv.get()
        self._on_disc()

    def _sd_changed(self, _):
        """选色域自动匹配对应曲线。这是**用户声明**，单独记住 ——
        RAW 预览时下拉框会被锁死并显示 RAW 工作空间，那不是用户的选择。"""
        self._sg_user = self.sd.get()
        c = _CURVE_MATCH.get(self.sd.get())
        if c and c in CURVE_NAMES:
            self.cv.set(c)
            self._sc_user = c
        self._save_cfg()
        self._on_disc()

    def _td_changed(self, _):
        try:
            self.after(0, self._ocst_hint_update)
        except Exception:
            pass
        c = _CURVE_MATCH.get(self.td.get())
        if c and c in CURVE_NAMES:
            self.cv2.set(c)
        self._on_disc()

    def _hl_changed(self, _):
        """切换高光重建：解码参数变了，必须清缓存重解（与 RAW 工作空间同理）"""
        global _HL_MODE
        _HL_MODE = self._hl = self.hl.get()
        self._pv_cache.clear(); self._thumb_cache.clear()
        self._preload_gen += 1
        self._hl_hint_update()
        self._save_cfg()
        self._on_disc()

    def _hl_hint_update(self):
        t = {"重建（推荐）": "高光回中性；但会让整幅变暗约 2.2 倍（LibRaw 的 -H 机制）",
             "混合": "用未裁通道混合；同样变暗，且最大值只到 0.74",
             "裁切（最差）": "不处理 —— 高光发品红/青，但保持旧观感与亮度",
             "不裁（会超界）": "不裁切，数值可能 >1；本管线会在显示转换前夹回 [0,1]"}
        try:
            self.hl_hint.configure(text=t.get(self._hl, ""))
        except Exception:
            pass

    def _rw_changed(self, _):
        """切换 RAW 宽工作空间：解码输出与 CST 源侧同步变更，需清缓存重解"""
        self._raw_work = self.rw.get()
        self._pv_cache.clear(); self._thumb_cache.clear()
        self._preload_gen += 1
        self._save_cfg()
        self._on_disc()

    def _ootf_direction(self):
        """当前 OOTF 决策方向。手动选则由用户决定；「自动」按**曲线的参考类型**判断。

        关键：**不是色域，是曲线**。参考类型由曲线唯一决定 ——
        Rec.2020+Gamma2.4 是显示参考，Rec.2020+V-Log 是场景参考，同名色域两种参考，
        色域名推不出参考类型（旧实现正是用色域名在猜，那是错的键）。

        **目标侧必须取 CST 的实际目标**，不是 LUT 的输出空间：

            数据流：RAW线性 --CST--> [LUT输入空间] --LUT--> [LUT输出空间] --显示转换--> 监看

        · OOTF 只作用在 **CST 这一段**上 —— 套 LUT 时 CST 的目标是 LUT 的**输入**曲线（lc），
          不套 LUT 时才是目标曲线（tc）。
        · LUT 的输出空间驱动的是**下游显示转换**（display_leg_out_space），是另一个阶段。

        这两者曾被混为一谈，后果很具体：一个官方 S-Log3.cine 还原 LUT
        （要求 S-Log3 输入、产出 Rec.709），被判成 forward，于是在 **S-Log3 编码之前**
        先乘了显示伽马 —— LUT 收到的是被显示补偿过的数据，等于喂错了。
        """
        sel = self._ootf_sw.get()
        if sel == "反向": return "reverse"
        if sel == "正向": return "forward"
        if sel == "关": return "off"
        try:
            _raw = bool(self._preview_file and _is_raw(self._preview_file))
            _sg, sc = resolve_source({"raw": _raw, "rw": self._raw_work,
                                      "sg": self.sd.get(), "sc": self._sc_user}, self._raw_work)
            if self.lv.get() == "on" and self._cd is not None:
                # 套 LUT：CST 的目标就是 LUT 的**输入**空间 → 取它的曲线
                tc = _CURVE_MATCH.get(self.li.get(), self.cv.get())
            else:
                tc = self.cv2.get()
        except Exception:
            return "off"
        # 判据是「**是不是显示参考**」，不是「是不是场景参考」：
        # unknown（如 RAW 的 Linear —— 本就是线性场景光）应归入**非显示**侧。
        s_dis = _ref_type(sc) == "display"
        t_dis = _ref_type(tc) == "display"
        if s_dis and not t_dis: return "reverse"
        if not s_dis and t_dis: return "forward"
        return "off"

    def _update_ootf_lbl(self):
        """显示 OOTF 决策**及其依据** —— 判定现在由用户声明的曲线驱动，
        所以必须让人看得见是哪两条曲线决定了它，否则又是一个黑箱。"""
        sel = self._ootf_sw.get()
        d = self._ootf_direction()
        tail = "（自动）" if sel == "自动" else ""
        if sel == "自动":
            try:
                _raw = bool(self._preview_file and _is_raw(self._preview_file))
                _, sc = resolve_source({"raw": _raw, "rw": self._raw_work,
                                        "sg": self.sd.get(), "sc": self._sc_user}, self._raw_work)
                if self.lv.get() == "on" and self._cd is not None:
                    tc = _CURVE_MATCH.get(self.li.get(), self.cv.get())
                    tail = f"（自动：{sc}[{_ref_type(sc)}] → {tc}[{_ref_type(tc)}] · CST目标=LUT输入）"
                else:
                    tc = self.cv2.get()
                    tail = f"（自动：{sc}[{_ref_type(sc)}] → {tc}[{_ref_type(tc)}]）"
            except Exception:
                tail = "（自动）"
        self._ootf_lbl.configure(text=f"方向: {d}  {tail}")

    def _on_disc(self):
        self._lc_hint()
        self._update_ootf_lbl()
        self._dirty()
        self._save_cfg()

    def _lo_changed(self, _=None):
        try:
            self.after(0, self._ocst_hint_update)
        except Exception:
            pass
        """LUT 输出空间变了：只影响监看与导出 ICC，管线像素不变 —— 但要重绘预览与波形"""
        self._save_cfg()
        self._on_disc()
        if getattr(self, "_pv_has", False):
            try:
                self.after(60, self._render_scope)
            except Exception:
                pass

    def _ocst_hint_update(self):
        """说明末端那一级 CST 做了什么、会不会裁切。"""
        try:
            snap = self._snapshot()
            if not (snap.get("lut") and snap.get("cd") is not None):
                self.ocst_hint.configure(
                    text="不套 LUT：没有这一级，文件直接就是「目标」空间",
                    text_color="#777")
                return
            fg, fc = pipeline_out_space(snap)
            tg, tc = display_leg_out_space(snap)
            if fg == tg and fc == tc:
                self.ocst_hint.configure(
                    text=f"LUT 输出空间 = 目标（{tg}/{tc}）→ 末端 CST 为恒等，"
                         f"文件里就是 LUT 的原值",
                    text_color="#777")
                return
            warn, msg = out_cst_gain(snap)
            if warn:
                self.ocst_hint.configure(text=msg, text_color="#ffb020")
            else:
                self.ocst_hint.configure(
                    text=f"末端 CST：{fg}/{fc} → 目标 {tg}/{tc}"
                         f"（两侧都按显示 EOTF，与显示层互为镜像）。"
                         f"{msg or '色料相同 → 纯曲线重编码，无损'}。"
                         f"想要 LUT 原值？把「目标」设成 {fg}/{fc} 即可",
                    text_color="#8b8")
        except Exception:
            pass

    def _li_changed(self, _=None):
        """手动选期望输入：更新预览，并询问是否把当前 LUT 移进对应文件夹"""
        self._on_disc()
        self._park_lut()
        self._offer_move_lut()

    def _park_lut(self):
        """确保「期望输入」对应的文件夹存在"""
        li = self.li.get(); c = _CURVE_MATCH.get(li, "")
        if not c: return
        dst_dir = Path(__file__).parent / "lut" / f"{li}-{c}"
        if not dst_dir.exists():
            dst_dir.mkdir(parents=True)

    def _offer_move_lut(self):
        """把当前选中的 LUT **真的移进**「期望输入」对应的文件夹。

        这一步是关键：文件夹名是以后自动识别输入空间的**唯一依据**。
        原来这里只建空文件夹、把归位甩给文件管理器，等于这个信息随时会丢
        （lut 根目录下已经躺了 10 个 li=None 的 LUT 就是这个后果）。
        """
        import shutil
        l = self._cur_lut()
        if not l: return
        src = Path(l["path"])
        li = self.li.get(); c = _CURVE_MATCH.get(li, "")
        if not c: return
        dst_dir = Path(__file__).parent / "lut" / f"{li}-{c}"
        try:
            if src.parent.resolve() == dst_dir.resolve():
                return                      # 已在正确位置
        except Exception:
            pass
        if not messagebox.askyesno("归位 LUT",
                f"把这个 LUT 移入「{li}-{c}」文件夹？\n\n{l['name']}\n\n"
                "文件夹名就是以后自动识别它期望输入空间的依据。"):
            return
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / src.name
        if dst.exists() and dst.resolve() != src.resolve():
            if not messagebox.askyesno("归位 LUT", f"目标已存在同名文件：\n{dst.name}\n\n覆盖？"):
                return
        try:
            shutil.move(str(src), str(dst))
        except Exception as e:
            messagebox.showerror("移动失败", str(e)); return
        self._tidy_lut_dir(src.parent)
        self._refresh_lut_list(select=src.stem)

    def _lc_hint(self):
        li = self.li.get(); lc = _CURVE_MATCH.get(li, "")
        self.lc_hint.configure(text=lc)

    def _save_cfg(self):
        try:
            json.dump({"out": self._od, "img": self._id, "lut": self._ld,
                       "sg": self.sd.get(), "sg_user": getattr(self, "_sg_user", self.sd.get()),
                       "sc": self._sc_user, "tg": self.td.get(), "tc": self.cv2.get(),
                       "li": self.li.get(), "rw": self._raw_work,
                       "hl": getattr(self, "_hl", _HL_DEF),
                       "dpi": float(self._dpi), "pnglvl": int(self._png_level),
                       "lut_fav": list(getattr(self, "_lut_fav", []) or []),
                       "lo": getattr(self, "_lo", "sRGB"), "loc": getattr(self, "_loc", "sRGB"),
                       "cm": bool(getattr(self, "_cm_on", True)),
                       "gw": bool(getattr(self, "_gw_on", False)),
                       "files": [str(p) for p in self._files][:200]},
                      open(_CFG, "w"))
        except Exception: pass

    # ---------- 预览 ----------
    def _make_small_lin(self, lin):
        """预览用的小图（线性 float32）。缩放在线性域做，避免 gamma 域平均压暗。"""
        h, w = lin.shape[0], lin.shape[1]
        sc = min(1.0, 2560.0/w, 2560.0/h)
        if sc >= 1.0:
            return lin
        return _down_lin(lin, (max(1, int(w*sc)), max(1, int(h*sc))))

    def _set_preview(self, path):
        try:
            rp = str(Path(path).resolve())
            hit = self._pv_cache.pop(rp, None)   # 预加载缓存命中则秒切
            if hit:
                lin, small, curve = hit
            else:
                a = _load_float(rp, half=True, work=self._raw_work,
                                wb=self._wb_args(), encode=False)
                lin = np.ascontiguousarray(a, dtype=np.float32)
                small = self._make_small_lin(lin)
                curve = None
            if not _is_raw(rp):
                self._apply_file_icc(rp)
            self._preview_file = rp
            self._pv_lin = lin
            self._pv_lin_small = small
            # 预载缓存里带着曲线，命中时不能用当前的 UI 选择覆盖它
            self._pv_src_curve = curve if hit else ("Linear" if _is_raw(rp) else self._sc_user)
            self._pv_gen += 1
            self._pv_prod = (lin.shape[1], lin.shape[0])
            self._pv_full = self._get_full_size(rp, lin.shape[1], lin.shape[0])
            self._pv_has = True
            self._pv_fit = True
            self._pv_show_orig = False
            self._pv_orig_ck.deselect()
            self._update_prod_lbl()
            self._schedule_preload()
            self._dirty()
        except Exception as e:
            self._pv_msg(f"预览失败: {e}")

    def _apply_file_icc(self, rp):
        """把文件嵌入的 ICC 变成源空间声明。

        此前**从不读 ICC**：源空间全靠用户手动选，而 `sd` 又会被 RAW 预览覆盖成
        解码层的工作空间 —— 于是 sRGB JPEG 被当成 ACES2065-1 解释。

        行为：
          · 读到且能对上标准描述 → 设定源色域/曲线（并同步到用户声明，切图不再漂）
          · 读不到 ICC → 按通行约定当 sRGB，但**说明是"未标记"**，不假装读到了
          · 认不出描述 → **不猜**，保留用户手动选的值，并如实提示
        """
        g, c, note = icc_to_space(rp)
        self._icc_note = note
        if not g:
            return                                  # 认不出：保留用户声明，不猜
        self._sg_user = g
        self._sc_user = c or _CURVE_MATCH.get(g, "sRGB")
        try:
            self.sd.configure(command=None); self.sd.set(g)
            self.sd.configure(command=self._sd_changed)
            self.cv.configure(command=None); self.cv.set(self._sc_user)
            self.cv.configure(command=self._cv_changed)
        except Exception:
            pass
        self._save_cfg()

    def _pv_warn_update(self):
        """管线输出是**场景参考**（Log/Linear）时，显示端做不到正确渲染，必须明说。"""
        try:
            _g, _c = display_leg_out_space(self._snapshot())
            if display_cannot_judge(_c):
                self._pv_warn.configure(
                    text=f"⚠ 当前是场景参考（{_c}）——监看不能据此判断成片；"
                         f"要看渲染结果，渲染必须进输出管线（这样才会写进文件）")
            else:
                self._pv_warn.configure(text="")
        except Exception:
            pass

    def _get_full_size(self, rp, w, h):
        """获取源文件真实全尺寸（方向与预览一致），用于生产画幅/打印显示。
        对 RAW：预览是 half 解码（尺寸减半），全尺寸 = 预览尺寸×2。
        对普通图：预览即原尺寸，直接读文件。失败时退回预览尺寸。"""
        try:
            _sx = Path(rp).suffix.lower()
            if _RAW is not None and _sx in _RAW_EXTS:
                return (w * 2, h * 2)
            with Image.open(rp) as im:
                return im.size
        except Exception:
            return (w, h)

    def _schedule_preload(self):
        """延迟预载相邻文件到内存缓存（大胆吃内存，换切换即时）"""
        if getattr(self, "_preload_after", None):
            try: self.after_cancel(self._preload_after)
            except Exception: pass
        self._preload_after = self.after(300, self._do_preload)

    def _do_preload(self):
        self._preload_after = None
        if not self._preview_file: return
        idx = None
        for i, f in enumerate(self._files):
            if str(f.resolve()) == self._preview_file:
                idx = i; break
        if idx is None: return
        targets = []
        for j in (idx + 1, idx - 1):
            if 0 <= j < len(self._files):
                targets.append(self._files[j])
        if not targets: return
        gen = self._preload_gen
        # 限制缓存条数。4.0 起每条目是 **两份 float32 线性**（源 + 小图，约 150MB），
        # 比 3.3 的两份 8bit PIL（约 38MB）大得多，故从 3 条降到 2 条，
        # 让"切图即时"与"导出时峰值内存"之间留出余量（本机可用内存只有 3.2GB）。
        while len(self._pv_cache) >= 2:
            self._pv_cache.pop(next(iter(self._pv_cache)))
        def work():
            for f in targets:
                if gen != self._preload_gen:
                    return   # 列表已变，丢弃过期预载
                rp = str(f.resolve())
                if rp in self._pv_cache or rp == self._preview_file:
                    continue
                try:
                    a = _load_float(rp, half=True, work=self._raw_work,
                                    wb=self._wb_args(), encode=False)
                    lin = np.ascontiguousarray(a, dtype=np.float32)
                    cur = "Linear" if _is_raw(rp) else self._sc_user
                    self._pv_cache[rp] = (lin, self._make_small_lin(lin), cur)
                except Exception:
                    pass
        threading.Thread(target=work, daemon=True).start()

    def _refresh_preview(self):
        self._pv_after = None
        if not self._pv_has: return
        try:
            self._render()          # 预览先出，减少切换卡顿
            self._schedule_scope()  # 示波器延后渲染
        except Exception as e:
            self._log_preview_error(e)
            self._pv_msg(f"预览错误: {type(e).__name__}: {e}\n\n完整堆栈已写入 preview_error.log")

    def _log_preview_error(self, e):
        """预览出错时落一份带上下文的完整堆栈 —— 只显示一句 {e} 是没法排查的。"""
        import traceback, time as _t
        tb = traceback.format_exc()
        try:
            ctx = (f"后端={backend_name()}  GPU状态={gpu_status()}  "
                   f"生产={getattr(self, '_pv_prod', None)}  源="
                   f"{None if self._pv_lin is None else self._pv_lin.shape}  "
                   f"fit={getattr(self, '_pv_fit', None)} zoom={getattr(self, '_pv_zoom_val', None)}")
        except Exception:
            ctx = "(上下文收集失败)"
        msg = (f"\n===== {_t.strftime('%Y-%m-%d %H:%M:%S')} =====\n{ctx}\n{tb}")
        try:
            with open(Path(__file__).parent / "preview_error.log", "a", encoding="utf-8") as f:
                f.write(msg)
        except Exception:
            pass
        print(f"[preview] {type(e).__name__}: {e}\n{msg}")

    def _pv_toggle(self):
        if not self._pv_has: return
        self._pv_show_orig = not self._pv_show_orig
        self._render()
        self._schedule_scope()

    def _cm_toggle(self):
        """色彩管理开关：只影响监看，不改变导出。"""
        self._cm_on = bool(self._cm_ck.get())
        self._sync_gw_state()
        self._render()
        self._schedule_scope()

    def _sync_gw_state(self):
        """色域警告依赖色彩管理：CM 关掉时它无从计算，置灰而不是给个假开关。"""
        try:
            if self._cm_on:
                self._gw_ck.configure(state="normal")
            else:
                self._gw_ck.configure(state="disabled")
        except Exception:
            pass

    def _gw_toggle(self):
        """色域警告开关：只影响监看。"""
        self._gw_on = bool(self._gw_ck.get()) and bool(self._cm_on)
        self._gw_ratio = None
        self._render()

    def _dims_text(self, pw, ph, s):
        t = f"生产 {pw}×{ph} | {int(round(s*100))}%"
        if self._gw_on and self._gw_ratio is not None:
            t += f"  | 超显示器色域 {self._gw_ratio*100:.2f}%"
        return t

    def _schedule_scope(self):
        if getattr(self, "_scope_after", None):
            try: self.after_cancel(self._scope_after)
            except Exception: pass
        self._scope_after = self.after(150, self._render_scope)

    def _pv_show(self):
        if not self._pv_has: return
        self._render()

    # ---------- GPU 渲染（4.0）----------
    def _real_t(self, v):
        """与当前 GPU 后端同精度的标量（float64/float32 由 backend 决定）。"""
        be = gpu_backend()
        t = getattr(be, "np_real", np.float32) if be is not None else np.float32
        return t(v)

    def _gpu_ready(self):
        """显存驻留：按代次把预览源上传一次。返回 backend 或 None。"""
        be = gpu_backend()
        if be is None or self._pv_lin is None:
            return None
        if self._g_gen != self._pv_gen:
            try:
                self._g_lin = be.to_gpu(self._pv_lin)
                self._g_lin_small = (be.to_gpu(self._pv_lin_small)
                                     if self._pv_lin_small is not None else None)
                self._g_gen = self._pv_gen
            except Exception as e:
                _note_gpu_error(e)
                return None
        return be

    def _gpu_lut(self, be, snap):
        """LUT 上显存。返回 (gL, N, scale)；GPU 不支持的 LUT 类型返回 None（调用方退回 CPU）。"""
        inf = snap.get("cd")
        if not (snap.get("lut") and inf is not None):
            return None
        if inf.get("t") != "3d":
            return None                       # 1D LUT 暂未做 GPU 内核 → 整块走 CPU
        key = id(inf)
        if self._g_lut_key != key:
            L32, sc_ = _nb._prep(inf)
            self._g_lut = be.to_gpu(np.ascontiguousarray(L32, dtype=np.float32).reshape(-1))
            self._g_lut_n = int(inf["n"]); self._g_lut_sc = float(sc_)
            self._g_lut_key = key
        return self._g_lut, self._g_lut_n, self._g_lut_sc

    def _gpu_region(self, be, sx0, sy0, sx1, sy1, oh, ow, use_small, snap):
        """在显存里做 裁切 → 降采样 → 色彩链 → 显示转换 → uint8。失败返回 None（退回 CPU）。"""
        cp = _gpu.cp
        try:
            gsrc = self._g_lin_small if use_small else self._g_lin
            if gsrc is None:
                return None
            g = gsrc[sy0:sy1, sx0:sx1]
            if not g.flags["C_CONTIGUOUS"]:
                g = cp.ascontiguousarray(g)
            if g.shape[0] != oh or g.shape[1] != ow:
                g = be.box_down(g, oh, ow)
            lut = None
            if snap.get("lut") and snap.get("cd") is not None:
                lut = self._gpu_lut(be, snap)
                if lut is None:
                    return None                       # 该 LUT 类型 GPU 不支持 → 整块退回 CPU
            # 「原图」走一份**旁路了 LUT** 的快照，且整条后续链都用它，
            # 避免 transform 与显示转换对"输出空间"的判断不一致。
            eff = self._orig_snap(snap) if self._pv_show_orig else snap
            args = lut if (lut is not None and not self._pv_show_orig) else (None, 0, 1.0)
            g = gpu_transform(g, eff, be, *args)
            gm = None
            if self._cm_on:
                # 与 CPU 路径（以及导出 _export_one 的 np.clip）保持一致：
                # 显示转换前先把**目标空间**码值夹到 [0,1]。不夹的话曲线解码会作用在越界值上，
                # GPU/CPU 两条路就会分叉（实测 fit 视图最大差 6 LSB）。
                g = cp.clip(g, self._real_t(0.0), self._real_t(1.0))
                g = gpu_display_leg(g, eff, be)       # 目标空间 → 显示器空间（**先不夹**）
                if self._gw_on:
                    th = self._real_t(_GW_TH)
                    gm = cp.any((g < -th) | (g > self._real_t(1.0) + th), axis=2)
            u8 = be.to_u8(g)                          # 内部才夹取
            if gm is not None:
                self._gw_ratio = float(gm.mean())
                u8[gm] = cp.asarray(_GW_RGB)
            return cp.asnumpy(u8)
        except Exception as e:
            _note_gpu_error(e)
            return None

    def _orig_snap(self, snap):
        """「原图」用的中性快照：只把**调色**参数归零，CST / LUT 与当前完全一致。

        为什么不再直接显示解码层数据：RAW 解出来的是**宽色域线性**（ACES AP0），
        把它当 sRGB 显示必然发灰、发闷 —— 那不是任何真实观感，也不是"未调色的成片"。
        业界（如 Resolve 的 bypass grade）的「原图」= 同样的输出变换 + 不调色，
        这样切换才是"我的调色到底做了什么"的有意义对照。
        """
        s0 = dict(snap)
        s0["ev"] = 0.0; s0["ct"] = 0.0; s0["sat"] = 0.0
        # **LUT 也要旁路**：LUT 是调色的一部分，Resolve 的 bypass grade 同样会跳过它。
        # 此前只归零 ev/ct/sat，于是勾了 LUT 时「原图」显示的是"套了 LUT 但不调色"，
        # 根本不是原图。旁路后输出空间也随之变回 (tg, tc)（见 display_leg_out_space）。
        s0["lut"] = False
        s0["cd"] = None
        if not snap.get("raw"):
            s0["tp"] = 6500.0        # 非 RAW：色温的中性值是 6500K（RAW 的 tp 由解码层烘焙，本就为 0）
        return s0

    def _process_region(self, sx0, sy0, sx1, sy1, dw, dh, snap):
        """按需处理可见生产区域（色彩运算逐像素，局部结果与整图一致）

        4.0：源是 **float32 线性**，管线声明 sc='Linear'（由 resolve_source 统一给出），
        与导出走**完全相同的精度**；8bit 只在最后上屏时生成。
        降采样改在线性域做（面积平均），修掉此前在 8bit sRGB 上 LANCZOS 的 gamma 域错误。
        """
        self._gw_ratio = None          # 每次渲染重置；只有真正算过才填
        pw, ph = self._pv_prod
        cw, ch = sx1 - sx0, sy1 - sy0
        small = self._pv_lin_small
        # 区域几乎覆盖整图（低倍/适应）→ 用小图源，避免从大图裁剪再重采样
        if small is not None and (cw >= small.shape[1] or ch >= small.shape[0]):
            sc = small.shape[1] / pw
            use_small = True
            sx0, sy0, sx1, sy1 = int(sx0*sc), int(sy0*sc), int(sx1*sc), int(sy1*sc)
        else:
            use_small = False
        src = small if use_small else self._pv_lin
        sx0 = max(0, min(sx0, src.shape[1]-1)); sx1 = max(sx0+1, min(sx1, src.shape[1]))
        sy0 = max(0, min(sy0, src.shape[0]-1)); sy1 = max(sy0+1, min(sy1, src.shape[0]))
        # 处理分辨率：取三者的最小值 ——
        #   ① 显示尺寸（比屏幕还多的像素没有意义）
        #   ② 像素预算 _PV_PX_BUDGET（把每帧耗时钉在预算内）
        #   ③ 1.0（只缩不放；高倍放大时必须保留源的全部像素）
        # 旧版固定按 0.6×显示尺寸 处理，再用 NEAREST 放大回去 —— 边缘发虚/锯齿。
        cwid, chgt = sx1 - sx0, sy1 - sy0
        k = min(1.0, dw/cwid, dh/chgt, (_PV_PX_BUDGET / max(1.0, cwid*chgt)) ** 0.5)
        ow, oh = max(1, int(cwid*k)), max(1, int(chgt*k))
        # ---- GPU 优先（显存驻留，裁切/降采样都在显存里做）----
        u8 = None
        be = self._gpu_ready()
        if be is not None:
            u8 = self._gpu_region(be, sx0, sy0, sx1, sy1, oh, ow, use_small, snap)
        # ---- CPU 回退（也是 GPU 不可用时的唯一路径）----
        if u8 is None:
            crop = np.ascontiguousarray(src[sy0:sy1, sx0:sx1])
            if k < 1.0:
                crop = _down_lin(crop, (ow, oh))
            eff = self._orig_snap(snap) if self._pv_show_orig else snap
            a = self._transform_tiled(crop, eff)
            a = np.clip(a, 0, 1)                      # 目标空间码值夹到 [0,1]（与导出一致）
            gm = None
            if self._cm_on:
                # 必须传 **eff**：原图旁路了 LUT，输出空间是 (tg,tc) 而不是 LUT 的 sRGB，
                # 传 snap 会让显示转换按错误的空间来 —— 这正是勾 LUT 时原图不对的原因之一。
                a = display_leg_nb(a, eff)            # 目标空间 → 显示器空间（**先不夹**）
                if self._gw_on:
                    gm = gamut_mask(a)                # 夹取前判越界，才是真·超色域
                a = np.clip(a, 0, 1)
            u8 = (a*255 + 0.5).astype(np.uint8)
            if gm is not None:
                self._gw_ratio = float(gm.mean())
                u8[gm] = _GW_RGB
        if u8.shape[1] != dw or u8.shape[0] != dh:
            rs = Image.NEAREST if u8.shape[1]*u8.shape[0] <= dw*dh else Image.LANCZOS
            u8 = np.asarray(Image.fromarray(u8).resize((dw, dh), rs), dtype=np.uint8)
        return u8

    def _render_scope(self):
        self._scope_after = None
        if not self._pv_has: return
        # 自适应：同时受预览画布与窗口宽度约束，避免右半边被窗口边缘切掉
        try:
            avail = min(self._pv_canvas.winfo_width(), self.winfo_width() - 40) - 70
            if avail > 200 and self._scope_cv.winfo_width() > avail:
                self._scope_cv.configure(width=int(avail))
        except Exception:
            pass
        src = self._pv_lin_small if self._pv_lin_small is not None else self._pv_lin
        if src is None:
            return                      # 还没有预览源（切图/清空途中）→ 静默跳过
        try:
            sh_, sw_ = src.shape[0], src.shape[1]
            # 分辨率要高：波形是**按列统计电平直方图**，样本越少每个 bin 越空，
            # 画出来就是一堆孤立点、密度信息全丢。1400 长边实测已足够密（每 bin 20+ 样本）。
            sc = min(1.0, 1400.0/sw_, 1400.0/sh_)
            crop = _down_lin(src, (max(1, int(sw_*sc)), max(1, int(sh_*sc)))) if sc < 1.0 else src
            snap = self._snapshot()
            a = np.clip(self._transform_tiled(
                crop, self._orig_snap(snap) if self._pv_show_orig else snap), 0, 1)
            # 波形按**目标空间**取值（与导出像素一致，才是有效的 QC 依据），
            # 所以这里刻意**不套**显示转换。
            u8 = (a*255 + 0.5).astype(np.uint8)
            self._draw_waveform(u8)
        except Exception as e:
            # 波形在 after 回调里跑，异常只会进 Tk 的控制台、界面上什么都看不到 —— 必须自己兜住
            self._log_preview_error(e)

    def _draw_waveform(self, u8):
        """波形监视器。Y = 单条亮度波形；RGB = **三面板 Parade**。

        三处关键做法（都是被实测逼出来的）：

        · **RGB 用 Parade 而不是叠加**：叠加而且各通道**各自归一化**，等于把 R/G/B
          各拉满量程，通道间的相对电平被抹掉 —— 而"哪个通道过曝"正是示波器该回答的。
          Parade 三面板并排、**共用同一套归一化**，高低可以直接比。

        · **列不缩、全聚合**：把图缩到面板宽度会让每个 (列,电平) 格只剩零点几个样本，
          画出来是孤立点，而 99.5% 分位在这种稀疏数据下约等于 1 →
          h/hi 恒为 1 → 密度信息全丢。改为**所有源列聚合进输出列**，只按行抽样控成本。

        · **整幅在 PIL 里合成成 RGB 再交给 Tk**：Tk 处理带 alpha 的 PhotoImage
          是 Tcl 层逐像素，实测 RGBA 12.8ms vs RGB 0.4ms（32×）；稀疏 alpha 时更会退化到
          2130ms，放大到 900×460 要 24 秒。所以背景+网格+波形在 PIL 里叠好，Tk 只贴一张图。

        配色：**保持高饱和，浓度只提亮度**（旧的 t+(1-t)*a 浓度一高就趋白，等于褪色）。
        """
        cv = self._scope_cv
        cw = max(200, cv.winfo_width()); ch = max(110, cv.winfo_height())
        axis_w = 32
        gw = max(60, cw - axis_w - 2)
        nbins = ch
        ih, iw = u8.shape[:2]
        gap = 5
        BG = (15, 15, 15)

        # 行抽样：控制 bincount 的规模。列一个都不丢。
        step = max(1, ih // 1600)
        sub_ = u8[::step, :, :]
        rows, iw2 = sub_.shape[:2]
        xb = np.arange(iw2)

        def _hist(chan, pw):
            """把**所有源列**聚合进 pw 个输出列，得到 (nbins, pw) 电平直方图。"""
            cm = (xb*pw//iw2).astype(np.int32)
            cm = np.broadcast_to(cm, (rows, iw2)).ravel()
            b = np.clip((chan.ravel()/255.0*(nbins-1) + 0.5).astype(np.int32), 0, nbins-1)
            h = np.bincount(cm*nbins + b, minlength=pw*nbins).reshape(pw, nbins)
            return h.T.astype(np.float32)

        def _dens(h, hi):
            # 浓度 → 强度。
            # 归一化分母取**约 p75**（不是 p99.5）：实测有值 bin 的中位数只有 3~8，
            # 而 p99.5 是 108~340、峰值 1487~3878 —— 分布极度偏斜，
            # 用高分位当分母会让中位数只剩 0.13 的 alpha，整屏几近全黑。
            # p75 + 指数 0.45 使**中位数落在约 0.55**、前 10% 饱和，Y 与 Parade 两个模式一致。
            return np.clip((h/max(hi, 1e-6))**0.45, 0.0, 1.0)

        def _paint(tint, al):
            """高饱和色 × 亮度：浓度高→更亮，**不往白走**。"""
            t = np.array(tint, dtype=np.float32)
            br = (0.12 + 0.88*al).astype(np.float32)[..., None]
            return t[None, None, :]*br

        if self._scope_rgb:
            pw = max(24, (gw - 2*gap)//3)
            hs = [_hist(sub_[..., c], pw) for c in range(3)]
            _pos = [h[h > 0] for h in hs]
            _pos = [p for p in _pos if p.size]
            hi = float(np.percentile(np.concatenate(_pos), 75)) if _pos else 1.0
            # 纯色、高饱和；亮度由浓度决定
            prim = ((1.00, 0.06, 0.06), (0.06, 1.00, 0.22), (0.12, 0.30, 1.00))
            W = 3*pw + 2*gap
            wave = np.zeros((nbins, W, 3), dtype=np.float32)
            alpha = np.zeros((nbins, W), dtype=np.float32)
            for c in range(3):
                al = _dens(hs[c], hi)
                x = c*(pw + gap)
                wave[:, x:x+pw] = _paint(prim[c], al)
                alpha[:, x:x+pw] = al
        else:
            pw = gw
            Y = (0.2126*sub_[..., 0].astype(np.float32) + 0.7152*sub_[..., 1].astype(np.float32)
                 + 0.0722*sub_[..., 2].astype(np.float32))
            h = _hist(Y, pw)
            _pos = h[h > 0]
            hi = float(np.percentile(_pos, 75)) if _pos.size else 1.0
            alpha = _dens(h, hi)
            W = pw
            wave = _paint((0.10, 1.00, 0.45), alpha)          # 荧光绿，高饱和

        # ---- 在 PIL 里把 背景 + 网格 + 波形 合成成一张 RGB ----
        wave = np.ascontiguousarray(wave)
        panel = Image.new("RGB", (W, ch), BG)
        d = ImageDraw.Draw(panel)

        def _y(L):
            return int(round((nbins - 1) - (L/255.0*(nbins - 1))))   # 与 bin 映射严格一致

        for L in (0, 64, 128, 192, 255):        # 8bit 码值刻度
            col = (122, 59, 59) if L == 255 else ((58, 58, 58) if L in (64, 192) else (40, 40, 40))
            d.line([(0, _y(L)), (W, _y(L))], fill=col)
        if self._scope_rgb:
            for c in (1, 2):
                xx = c*(pw + gap) - gap//2
                d.line([(xx, 0), (xx, ch)], fill=(61, 61, 61))
        else:
            for fx in (0.25, 0.5, 0.75):
                xx = int(W*fx)
                d.line([(xx, 0), (xx, ch)], fill=(27, 27, 27))
        # 波形叠在网格**上面**，但只在有密度的地方（alpha 混合），网格不被整片盖掉
        a8 = np.clip(alpha*255 + 0.5, 0, 255).astype(np.uint8)
        # wave 是 0~1 的浮点，必须 ×255 —— 早先写成 np.clip(wave,0,255) 再转 uint8，
        # 于是几乎所有值都被截成 0/1，波形黑得几乎看不见。
        w8 = np.clip(wave, 0.0, 1.0)*255.0
        wv = Image.fromarray(np.ascontiguousarray(
            np.concatenate([w8.astype(np.uint8), a8[..., None]], axis=2)), "RGBA")
        # **必须垂直翻转**：直方图的行 0 = 电平 0（黑），而图像的行 0 在**顶部**。
        # 早先重写时把原来的 FLIP_TOP_BOTTOM 弄丢了，于是亮度轴整个反过来
        # ——实测输入 100 画在第 117 行、输入 160 画在第 188 行，亮的反而在下面。
        # 网格是按 _y() 画的（255 在顶），本来就对，所以只翻波形。
        wv = wv.transpose(Image.FLIP_TOP_BOTTOM)
        # 边框画在波形**下面**：电平 0/255 的波形正好落在第 0 / 最后一行，
        # 若边框在波形之上会把它整条盖掉 —— 而 0 与 255 恰恰是最该看到的裁切位置。
        d.rectangle([0, 0, W - 1, ch - 1], outline=(61, 61, 61))
        panel = Image.alpha_composite(panel.convert("RGBA"), wv).convert("RGB")

        self._scope_pilimg = panel
        self._scope_photo = ImageTk.PhotoImage(panel)      # RGB → 快 30 倍

        # ---- 画布：只贴图 + 画文字（文字用 Tk 字体更好看）----
        cv.delete("all")
        cv.create_image(axis_w, 0, anchor="nw", image=self._scope_photo)
        # 刻度用 **8bit 码值 0-255**（这个工具的像素就是 8bit，导出也是 8bit；
        # 早先写的 IRE 百分比对照片工作流是多余的换算）。字号也加大。
        for L in (0, 64, 128, 192, 255):
            yy = min(max(_y(L), 9), ch - 9)
            cv.create_text(axis_w - 6, yy, text=str(L), fill="#c8c8c8", anchor="e",
                           font=("Consolas", 11))
        if self._scope_rgb:
            for c, nm in enumerate(("R", "G", "B")):
                px = axis_w + c*(pw + gap)
                cv.create_text(px + 4, 4, text=nm, anchor="nw",
                               fill=("#ff3a3a", "#3aff70", "#4a80ff")[c],
                               font=("Consolas", 9, "bold"))
                pc = float((u8[..., c] >= 254).mean()*100.0)
                if pc > 0.05:
                    cv.create_text(px + pw - 4, 4, text=f"\u25b2{pc:.1f}%", anchor="ne",
                                   fill="#ff8080", font=("Consolas", 7))
        else:
            hi_p = float((u8 >= 254).mean()*100.0); lo_p = float((u8 <= 1).mean()*100.0)
            if hi_p > 0.05:
                cv.create_text(axis_w + W - 5, 4, text=f"\u25b2{hi_p:.2f}%", fill="#ff8080",
                               anchor="ne", font=("Consolas", 8))
            if lo_p > 0.05:
                cv.create_text(axis_w + W - 5, ch - 18, text=f"\u25bc{lo_p:.2f}%",
                               fill="#7090e0", anchor="ne", font=("Consolas", 8))

        # ---- 右下角缩放把手 ----
        for o in (0, 5, 10):
            cv.create_line(cw - 3 - o, ch - 3, cw - 3, ch - 3 - o, fill="#5a5a5a")

    def _scope_toggle_mode(self):
        self._scope_rgb = not self._scope_rgb
        self._scope_mode.configure(text="RGB Parade" if self._scope_rgb else "Y 亮度")
        self._render_scope()

    def _scope_set_size(self, w, h, render=True):
        """设定波形画布尺寸（并按预览区可容纳宽度夹住）。render=False 用于拖动中。"""
        try:
            avail = max(240, self._scope.master.winfo_width() - 40)
        except Exception:
            avail = 1100
        w = int(max(240, min(min(1100, avail), w)))
        h = int(max(130, min(760, h)))
        self._scope_cv.configure(width=w, height=h)
        if render:
            self._render_scope()

    def _scope_wheel(self, e):
        """滚轮缩放（等比，便于微调）。"""
        k = 1.10 if getattr(e, "delta", 0) > 0 else 1.0/1.10
        self._scope_set_size(self._scope_cv.winfo_width()*k, self._scope_cv.winfo_height()*k)

    def _scope_reset_size(self):
        self._scope_set_size(560, 300)

    def _scope_resize(self, d):
        """保留旧接口（乘法步进），以防还有别处调用。"""
        k = 1.15 if d > 0 else 1.0/1.15
        self._scope_set_size(self._scope_cv.winfo_width()*k, self._scope_cv.winfo_height()*k)

    # ---- 波形示波器拖动 ----
    def _scope_drag_start(self, e):
        # 右下角 20×20 区域 = 缩放把手（波形画布上的点击才判定）
        if e.widget is self._scope_cv:
            cw0, ch0 = self._scope_cv.winfo_width(), self._scope_cv.winfo_height()
            if e.x >= cw0 - 20 and e.y >= ch0 - 20:
                self._scope_drag_data = ("resize", e.x_root, e.y_root, cw0, ch0)
                self._scope_cv.configure(cursor="size_nw_se")
                return
        p = self._scope.place_info()
        try:
            relx = float(p.get("relx", 0.0)); x = float(p.get("x", 0)); y = float(p.get("y", 0))
            anchor = p.get("anchor", "nw")
        except Exception:
            relx, x, y, anchor = 0.0, 0.0, 0.0, "nw"
        self._scope_drag_data = (e.x_root, e.y_root, relx, x, y, anchor)
        self._scope.configure(cursor="fleur")

    def _scope_drag(self, e):
        d = getattr(self, "_scope_drag_data", None)
        if not d:
            return
        if d[0] == "resize":                      # 拖把手：只改几何，松手才重绘（否则拖动会卡）
            _, rx0, ry0, w0, h0 = d
            self._scope_set_size(w0 + (e.x_root - rx0), h0 + (e.y_root - ry0), render=False)
            return
        x0, y0, relx, x, y, anchor = d
        dx, dy = e.x_root - x0, e.y_root - y0
        nx, ny = x + dx, y + dy
        par = self._scope.master
        pw, ph = max(1, par.winfo_width()), max(1, par.winfo_height())
        w, h = self._scope.winfo_width(), self._scope.winfo_height()
        SNAP = 24          # 边缘吸附阈值
        RIGHT = -18        # 右缘吸附位（避开右侧滚动条）
        if anchor == "ne":
            nx = max(-w, min(0, nx)); ny = max(0, min(ph - h, ny))
            if abs(nx - RIGHT) < SNAP: nx = RIGHT              # 吸右
            if abs(pw - w + nx) < SNAP: nx = -(pw - w)         # 吸左
            if ny < SNAP: ny = 0                               # 吸上
            if ph - h - ny < SNAP: ny = ph - h                 # 吸下
        else:
            nx = max(0, min(pw - w, nx)); ny = max(0, min(ph - h, ny))
            if abs(nx) < SNAP: nx = 0                          # 吸左
            if abs(pw - w - nx) < SNAP: nx = pw - w            # 吸右
            if ny < SNAP: ny = 0                               # 吸上
            if ph - h - ny < SNAP: ny = ph - h                 # 吸下
        self._scope.place(relx=relx, x=nx, y=ny, anchor=anchor)

    def _scope_drag_end(self, _e):
        d = getattr(self, "_scope_drag_data", None)
        self._scope_drag_data = None
        self._scope.configure(cursor="")
        try:
            self._scope_cv.configure(cursor="crosshair")
        except Exception:
            pass
        if d and d[0] == "resize":
            self._render_scope()                  # 拖动结束才真正重绘

    def _render(self):
        if not self._pv_has: return
        cv = self._pv_canvas
        vw = max(1, cv.winfo_width()); vh = max(1, cv.winfo_height())
        if vw < 120 or vh < 120:
            vw, vh = 900, 640
        pw, ph = self._pv_prod
        snap = self._snapshot()
        if self._pv_fit:
            s = min(vw/pw, vh/ph)
            self._pv_s = s
            dw, dh = max(1, int(pw*s)), max(1, int(ph*s))
            u8 = self._process_region(0, 0, pw, ph, dw, dh, snap)
            photo = ImageTk.PhotoImage(Image.fromarray(u8))
            self._pv_photo = photo
            cv.delete("all")
            ox, oy = max(0, (vw - dw)//2), max(0, (vh - dh)//2)
            cv.create_image(ox, oy, anchor="nw", image=photo)
            cv.configure(scrollregion=(0, 0, vw, vh))
            cv.xview_moveto(0); cv.yview_moveto(0)
            self._pv_img_ox, self._pv_img_oy = ox, oy
            self._pv_dims.configure(text=self._dims_text(pw, ph, s))
            return
        # 缩放模式：按需渲染可见区域（只处理视口内生产像素，高倍放大保持全像素）
        s = self._pv_zoom_val
        self._pv_s = s
        log_w, log_h = pw*s, ph*s
        ox = max(0.0, (vw - log_w)/2); oy = max(0.0, (vh - log_h)/2)
        self._pv_img_ox, self._pv_img_oy = ox, oy
        sw_, sh_ = max(log_w, vw), max(log_h, vh)
        cv.configure(scrollregion=(0, 0, sw_, sh_))
        if log_w <= vw: cv.xview_moveto(0)
        if log_h <= vh: cv.yview_moveto(0)
        x0 = cv.canvasx(0); y0 = cv.canvasy(0)
        xa = max(x0, ox); ya = max(y0, oy)
        xb = min(x0 + vw, ox + log_w); yb = min(y0 + vh, oy + log_h)
        if xb <= xa or yb <= ya:
            cv.delete("all"); return
        # 可见生产区域（画布坐标 - 图像原点）÷ 缩放
        sx0 = int(max(0, (xa - ox)/s)); sy0 = int(max(0, (ya - oy)/s))
        sx1 = int(min(pw, (xb - ox)/s + 1)); sy1 = int(min(ph, (yb - oy)/s + 1))
        dw = max(1, int(round(xb - xa))); dh = max(1, int(round(yb - ya)))
        u8 = self._process_region(sx0, sy0, sx1, sy1, dw, dh, snap)
        photo = ImageTk.PhotoImage(Image.fromarray(u8))
        self._pv_photo = photo
        cv.delete("all")
        cv.create_image(int(round(xa)), int(round(ya)), anchor="nw", image=photo)
        self._pv_dims.configure(text=self._dims_text(pw, ph, s))

    def _pv_msg(self, text):
        cv = self._pv_canvas
        cv.delete("all")
        w = max(300, cv.winfo_width()); h = max(200, cv.winfo_height())
        cv.create_text(w//2, h//2, text=text, fill="#aaa", font=("Microsoft YaHei", 13), justify="center")
        cv.configure(scrollregion=(0, 0, w, h))

    # ---- 生产画幅 / DPI ----
    def _update_prod_lbl(self):
        if not getattr(self, "_pv_prod", None):
            self._prod_lbl.configure(text="未加载图片"); return
        pw, ph = self._pv_full if getattr(self, "_pv_full", None) else self._pv_prod
        self._prod_lbl.configure(text=f"生产画幅 {pw}×{ph} px")
        self._on_dpi(False)

    def _on_dpi(self, save=True):
        try:
            dpi = float(self._dpi_ent.get())
            if dpi <= 0: dpi = 300
        except Exception:
            dpi = 300
        self._dpi = dpi
        if getattr(self, "_pv_prod", None):
            pw, ph = self._pv_full if getattr(self, "_pv_full", None) else self._pv_prod
            self._print_lbl.configure(text=f"打印 {pw/dpi:.1f}×{ph/dpi:.1f} 英寸")
        if save:
            self._save_cfg()

    # ---- 连续缩放 ----
    def _update_zoom_lbl(self):
        if getattr(self, "_pv_fit", True):
            self._pv_zlbl.configure(text="适应")
        else:
            self._pv_zlbl.configure(text=f"{int(round(self._pv_zoom_val*100))}%")

    def _zoom_fit(self):
        if not self._pv_has: return
        self._pv_fit = True
        self._update_zoom_lbl()
        self._render()

    def _zoom_btn(self, d):
        if not self._pv_has: return
        cv = self._pv_canvas
        vw, vh = max(1, cv.winfo_width()), max(1, cv.winfo_height())
        pw, ph = self._pv_prod
        base = min(vw/pw, vh/ph)
        s = self._pv_s if not self._pv_fit else base
        factor = 1.5 if d > 0 else 1/1.5
        s = max(base*0.95, min(64.0, s*factor))
        self._pv_fit = False; self._pv_zoom_val = s
        self._update_zoom_lbl()
        self._center_view()
        self._render()

    def _set_zoom(self, s, anchor=None):
        self._pv_fit = False
        self._pv_zoom_val = s
        self._update_zoom_lbl()
        if anchor is not None:
            self._anchor_view(anchor[0], anchor[1], anchor[2], anchor[3], s)
        self._render()

    def _anchor_view(self, ix, iy, ex, ey, s):
        cv = self._pv_canvas
        pw, ph = self._pv_prod
        vw, vh = max(1, cv.winfo_width()), max(1, cv.winfo_height())
        log_w, log_h = pw*s, ph*s
        ox = max(0.0, (vw - log_w)/2); oy = max(0.0, (vh - log_h)/2)
        sw_, sh_ = max(log_w, vw), max(log_h, vh)
        cv.configure(scrollregion=(0, 0, sw_, sh_))
        tx = ox + ix*s - ex; ty = oy + iy*s - ey
        def fx(v, size, vp):
            if size <= vp: return 0.0
            return max(0.0, min((size - vp)/size, v/size))
        cv.xview_moveto(fx(tx, sw_, vw))
        cv.yview_moveto(fx(ty, sh_, vh))

    def _center_view(self):
        cv = self._pv_canvas
        vw = max(1, cv.winfo_width()); vh = max(1, cv.winfo_height())
        pw, ph = self._pv_prod
        s = self._pv_zoom_val
        log_w, log_h = pw*s, ph*s
        sw_, sh_ = max(log_w, vw), max(log_h, vh)
        ox = max(0.0, (vw - log_w)/2); oy = max(0.0, (vh - log_h)/2)
        cv.configure(scrollregion=(0, 0, sw_, sh_))
        tx = ox + log_w/2 - vw/2; ty = oy + log_h/2 - vh/2
        def fx(v, size, vp):
            if size <= vp: return 0.0
            return max(0.0, min((size - vp)/size, v/size))
        cv.xview_moveto(fx(tx, sw_, vw))
        cv.yview_moveto(fx(ty, sh_, vh))

    # ---- 鼠标缩放 / 双击 ----
    def _proc_under_cursor(self, ex, ey):
        cv = self._pv_canvas
        s = max(1e-6, self._pv_s)
        ox = getattr(self, "_pv_img_ox", 0.0); oy = getattr(self, "_pv_img_oy", 0.0)
        return (cv.canvasx(ex) - ox)/s, (cv.canvasy(ey) - oy)/s

    def _pv_motion(self, e):
        """鼠标悬停实时读取生产像素值（处理后）"""
        if not self._pv_has:
            self._pv_pxl.configure(text=""); return
        pt = self._proc_under_cursor(e.x, e.y)
        px, py = int(pt[0]), int(pt[1])
        pw, ph = self._pv_prod
        if not (0 <= px < pw and 0 <= py < ph):
            self._pv_pxl.configure(text=""); return
        try:
            # 取色直接从线性源读（比原来经 8bit sRGB PIL 更准，且与导出一致）
            lin = self._pv_lin
            px = max(0, min(px, lin.shape[1]-1)); py = max(0, min(py, lin.shape[0]-1))
            src = np.asarray(lin[py, px], dtype=np.float32)
            snap = self._snapshot()
            eff = self._orig_snap(snap) if self._pv_show_orig else snap
            a = self._transform(src[None, None, :], eff)
            a = np.clip(np.asarray(a, dtype=np.float32), 0, 1)
            oog = False
            if self._cm_on:
                a = np.asarray(display_leg_nb(a, eff), dtype=np.float32)
                oog = bool(((a < -_GW_TH) | (a > 1.0 + _GW_TH)).any())
                a = np.clip(a, 0, 1)
            out = a[0, 0]
            r, g, b = (out*255 + 0.5).astype(int)
            _txt = f"({px},{py})  R{r} G{g} B{b}"
            if oog:
                _txt += "   ⚠ 超出显示器色域"
            self._pv_pxl.configure(text=_txt)
        except Exception:
            pass

    def _wheel(self, e):
        if not self._pv_has: return
        cv = self._pv_canvas
        ex, ey = e.x, e.y
        pt = self._proc_under_cursor(ex, ey)
        if pt is None: return
        pw, ph = self._pv_prod
        # 用「生产点是否在图内」判断光标是否落在图像上（勿用 bbox，屏幕/画布坐标会错位）
        if 0 <= pt[0] < pw and 0 <= pt[1] < ph:
            pass   # 光标在图像内：锚定 pt
        else:
            pt = (pw/2, ph/2)   # 光标在图像外空白区：锚定图像中心
        vw, vh = max(1, cv.winfo_width()), max(1, cv.winfo_height())
        base = min(vw/pw, vh/ph)
        factor = 1.5 if e.delta > 0 else 1/1.5
        s = self._pv_s * factor
        s = max(base*0.95, min(64.0, s))
        self._set_zoom(s, anchor=(pt[0], pt[1], ex, ey))

    def _dc(self, e):
        if not self._pv_has: return
        if not self._pv_fit and abs(self._pv_zoom_val - 2.5) < 0.01:
            self._zoom_fit()               # 已在 250% → 回到适应
        else:
            # 双击放大到 250%，并追随点击位置（像素留在光标下）
            pw, ph = self._pv_prod
            pt = self._proc_under_cursor(e.x, e.y)
            if pt is None or not (0 <= pt[0] < pw and 0 <= pt[1] < ph):
                pt = (pw/2, ph/2)
            self._set_zoom(2.5, anchor=(pt[0], pt[1], e.x, e.y))

    # ---- 拖拽平移 ----
    def _pv_drag_start(self, e):
        if not self._pv_has or self._pv_fit: return
        cv = self._pv_canvas
        cv.configure(cursor="fleur")
        self._pv_drag_data = (e.x, e.y)
        self._pv_drag_off = (cv.canvasx(0), cv.canvasy(0))

    def _pv_drag(self, e):
        if not getattr(self, "_pv_drag_data", None): return
        x0, y0 = self._pv_drag_data
        offx, offy = self._pv_drag_off
        self._set_view_off(offx - (e.x - x0), offy - (e.y - y0))

    def _pv_drag_end(self, _e):
        self._pv_drag_data = None
        self._pv_canvas.configure(cursor="")

    def _set_view_off(self, nx, ny):
        cv = self._pv_canvas
        try:
            r = cv.cget("scrollregion").split()
            sw, sh = float(r[2]), float(r[3])
        except Exception:
            return
        vw, vh = max(1, cv.winfo_width()), max(1, cv.winfo_height())
        def fx(v, size, vp):
            if size <= vp: return 0.0
            return max(0.0, min((size - vp)/size, v/size))
        cv.xview_moveto(fx(nx, sw, vw))
        cv.yview_moveto(fx(ny, sh, vh))
        self._render()

    def _bind_resize(self):
        def h(_e):
            if getattr(self, "_rz_after", None):
                try: self.after_cancel(self._rz_after)
                except Exception: pass
            self._rz_after = self.after(150, self._on_resized)
        self.bind("<Configure>", h)
        self.after(250, self._on_resized)   # 窗口落定后校正一次预览

    def _on_resized(self):
        self._rz_after = None
        if not self._pv_has: return
        self._pv_show()
        self._schedule_scope()

    # ---------- 文件列表 ----------
    def _af(self):
        fs = filedialog.askopenfilenames(title="选图", initialdir=self._id or None)
        if not fs: return
        for f in fs:
            p = Path(f)
            if p.suffix.lower() in _INPUT_EXTS: self._files.append(p)
        self._id = str(Path(fs[0]).parent)
        self._save_cfg()
        self._rb()

    def _ad(self):
        d = filedialog.askdirectory(title="选文件夹", initialdir=self._id or None)
        if not d: return
        for p in sorted(Path(d).iterdir()):
            if p.suffix.lower() in _INPUT_EXTS and not p.name.startswith("."):
                self._files.append(p)
        self._id = d
        self._save_cfg()
        self._rb()

    def _cl(self):
        self._files.clear(); self._rb()

    def _restore(self):
        """恢复上次用户添加的文件列表（不自动扫描整个文件夹，避免误开全部照片）"""
        self._files = []
        for f in (_cl().get("files") or []):
            p = Path(f)
            if p.is_file() and p.suffix.lower() in _INPUT_EXTS:
                self._files.append(p)
        if self._files:
            self._rb()

    def _rm(self, i):
        if 0 <= i < len(self._files):
            self._files.pop(i); self._rb()

    def _sel_file(self, path, frame):
        """选中文件：设预览 + 高亮格子"""
        if getattr(self, "_sel_row", None) is not None:
            try: self._sel_row.configure(fg_color="#252525")
            except Exception: pass
        frame.configure(fg_color="#3f5a4f")
        self._sel_row = frame
        self._set_preview(path)

    def _mk_thumb(self, i, f):
        cell = ctk.CTkFrame(self.sf, fg_color="#252525", corner_radius=6)
        cv = Canvas(cell, width=150, height=100, bg="#191919", highlightthickness=0, bd=0)
        cv.pack(padx=3, pady=(3, 0))
        cv.create_text(75, 50, text="…", fill="#555", font=("Microsoft YaHei", 10))
        bar = ctk.CTkFrame(cell, fg_color="transparent")
        bar.pack(fill="x", padx=4, pady=(1, 3))
        nm = ctk.CTkLabel(bar, text=_trunc(f.name, 16), anchor="w", font=("Consolas", 8), cursor="hand2")
        nm.pack(side="left", fill="x", expand=True)
        ctk.CTkButton(bar, text="X", width=20, height=18, font=("Microsoft YaHei", 9),
                      fg_color="#c42", hover_color="#e55",
                      command=lambda idx=i: self._rm(idx)).pack(side="right")
        sel = lambda e, p=str(f), c=cell: self._sel_file(p, c)
        for w in (cell, cv, nm): w.bind("<Button-1>", sel)
        ic = ctk.CTkLabel(cell, text="", font=("Microsoft YaHei", 9))
        ic.place(relx=1.0, x=-4, y=2, anchor="ne")
        return cell, cv, ic

    def _rb(self):
        seen = set(); tmp = []
        for f in self._files:
            k = str(f.resolve())
            if k not in seen: seen.add(k); tmp.append(f)
        self._files = tmp
        self.cn.configure(text=f"{len(self._files)} 个")
        self.gb.configure(state="normal" if self._files else "disabled")
        self._preload_gen += 1; self._pv_cache.clear(); self._thumb_cache.clear()   # 列表变了，旧缓存失效
        for w in self.sf.winfo_children(): w.destroy()
        n = len(self._files)
        self._rows = []; self._row_frames = []; self._thumb_cv = []; self._thumb_photos = [None]*n
        for i, f in enumerate(self._files):
            cell, cv, ic = self._mk_thumb(i, f)
            cell.grid(row=i//2, column=i%2, padx=3, pady=3, sticky="nsew")
            self._row_frames.append(cell); self._rows.append(ic); self._thumb_cv.append(cv)
        self._schedule_thumbs()
        self.pb.set(0); self.st.configure(text="就绪")
        if self._files:
            if not (self._preview_file and any(str(f.resolve()) == self._preview_file for f in self._files)):
                self._set_preview(self._files[0])
            # 高亮当前预览格
            self._sel_row = None
            for i, f in enumerate(self._files):
                if str(f.resolve()) == str(self._preview_file):
                    self._row_frames[i].configure(fg_color="#3f5a4f")
                    self._sel_row = self._row_frames[i]
                    break
        else:
            self._preview_file = None; self._pv_lin = None
            self._pv_has = False; self._pv_prod = None; self._pv_full = None; self._pv_lin_small = None
            self._pv_gen += 1
            self._sel_row = None; self._pv_cache.clear()
            self._pv_msg("无预览\n· 添加图片后自动加载第一张\n· 或点击缩略图切换预览")

    def _schedule_thumbs(self):
        """后台线程生成缩略图写入缓冲，主线程 after 轮询绘制（避免跨线程 Tk 调用）"""
        self._thumb_gen += 1
        gen = self._thumb_gen
        self._thumb_pending = []; self._thumb_busy = True
        def work():
            try:
                for i, f in enumerate(self._files):
                    if gen != self._thumb_gen: return
                    rp = str(f.resolve())
                    pil = self._thumb_cache.get(rp)
                    if pil is None:
                        try:
                            with Image.open(rp) as im:
                                im = ImageOps.exif_transpose(im)
                                pil = ImageOps.fit(im.convert("RGB"), (150, 100), Image.LANCZOS)
                            self._thumb_cache[rp] = pil
                        except Exception:
                            continue
                    self._thumb_pending.append((i, pil))
            finally:
                self._thumb_busy = False
        threading.Thread(target=work, daemon=True).start()
        self.after(120, self._drain_thumbs)

    def _drain_thumbs(self):
        pend, self._thumb_pending = self._thumb_pending, []
        for i, pil in pend:
            self._set_thumb(i, pil)
        if getattr(self, "_thumb_busy", False):
            self.after(120, self._drain_thumbs)

    def _set_thumb(self, i, pil):
        if i >= len(self._thumb_cv): return
        try:
            photo = ImageTk.PhotoImage(pil)
            self._thumb_photos[i] = photo
            cv = self._thumb_cv[i]
            cv.delete("all")
            cv.create_image(0, 0, anchor="nw", image=photo)
        except Exception:
            pass

    # ---------- LUT ----------
    def _tg(self):
        if self.lv.get() == "on":
            if self._cd is None:
                self._pk()          # 勾选时确保 LUT 已加载
            self.td.configure(state="disabled"); self.cv2.configure(state="disabled")
        else:
            self.td.configure(state="normal"); self.cv2.configure(state="normal")
        self._on_disc()

    def _pk(self):
        n = self.ld.get(); m = [l for l in self._luts if l["disp"] == n]
        if not m: return
        l = m[0]; self._cd = None
        self._cd_path = l["path"]
        self._lut_path.configure(text="lut/" + l["name"])
        if l["li"]:
            self.li.set(l["li"])   # 期望输入唯一由「色域-曲线」文件夹决定，不猜文件名
        try:
            d = _cp(l["path"]); self._cd = d
            t = f"{d['t'].upper()} | {d['n']}x{d['n']}"
            if d["ti"]: t += f" | {d['ti']}"
            self._info.configure(text=t)
        except Exception as e:
            self._info.configure(text=f"解析失败: {e}")
        self._on_disc()

    def _import_lut_into(self, fp, li, c, overwrite=False):
        """把 fp 复制进 lut/<li>-<c>/ —— **文件夹名就是这份 LUT 期望输入空间的记录**，
        以后 _sl() 靠它自动识别，不再靠猜文件名。返回目标 Path，被跳过时返回 None。
        （抽成独立方法是为了能脱离模态框直接测试。）"""
        import shutil
        src = Path(fp)
        dst_dir = Path(__file__).parent / "lut" / f"{li}-{c}"
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / src.name
        if dst.exists() and not overwrite:
            if dst.resolve() == src.resolve():
                return dst
            return None
        shutil.copy2(str(src), str(dst))
        return dst

    def _decorate_luts(self):
        """收藏（★）排在最前，并在显示名上加标记。显示名只在本模块内部匹配，改动安全。"""
        fav = set(self._lut_fav or [])
        for l in self._luts:
            base = l.get("_base", l["disp"])
            l["_base"] = base
            l["disp"] = ("\u2605 " + base) if base in fav else base
        self._luts.sort(key=lambda l: (l["_base"] not in fav, l["_base"]))

    def _lut_toggle_fav(self):
        l = self._cur_lut()
        if not l:
            messagebox.showinfo("收藏", "先选一个 LUT")
            return
        base = l.get("_base", l["disp"])
        fav = list(self._lut_fav or [])
        if base in fav:
            fav.remove(base)
        else:
            fav.insert(0, base)
        self._lut_fav = fav
        cur = base
        self._decorate_luts()
        self.ld.configure(values=[x["disp"] for x in self._luts] or ["(无)"])
        self.ld.set(("\u2605 " + cur) if cur in fav else cur)
        self._save_cfg()
        self._lut_fav_btn.configure(text=("\u2605 已收藏" if cur in fav else "\u2606 收藏"))

    def _refresh_lut_list(self, select=None):
        self._luts = _sl()
        self._decorate_luts()
        self.ld.configure(values=[l["disp"] for l in self._luts] or ["(无)"])
        if select:
            self.ld.set(select)
        elif self._luts:
            self.ld.set(self._luts[0]["disp"])
        else:
            self.ld.set("(无)")
        self._pk()

    def _ask_lut_space(self, name):
        """导入时问清期望输入空间。返回 (色域, 曲线) 或 None。"""
        dlg = ctk.CTkToplevel(self)
        dlg.title("导入 LUT")
        dlg.geometry("430x185")
        dlg.transient(self)
        try: dlg.grab_set()
        except Exception: pass
        ctk.CTkLabel(dlg, text=f"「{name}」期望哪种输入色彩空间？",
                     font=("Microsoft YaHei", 11)).pack(pady=(14, 2))
        ctk.CTkLabel(dlg, text="决定它放进哪个「色域-曲线」文件夹；文件夹名就是以后的识别依据",
                     font=("Microsoft YaHei", 9), text_color="#777").pack()
        var = ctk.StringVar(value=self.li.get() or (GAMUT_NAMES[0] if GAMUT_NAMES else ""))
        om = ctk.CTkOptionMenu(dlg, values=GAMUT_NAMES, variable=var, width=210)
        om.pack(pady=6)
        hint = ctk.CTkLabel(dlg, text="", font=("Consolas", 10), text_color="#8b8")
        hint.pack()
        res = {}

        def _upd(_=None):
            c = _CURVE_MATCH.get(var.get(), "")
            hint.configure(text=(f"→ lut/{var.get()}-{c}" if c else "该色彩空间没有匹配曲线，请另选"))
            return c
        om.configure(command=_upd)
        _upd()

        def _ok():
            c = _CURVE_MATCH.get(var.get(), "")
            if not c:
                return
            res["v"] = (var.get(), c)
            dlg.destroy()

        row = ctk.CTkFrame(dlg, fg_color="transparent")
        row.pack(pady=10)
        ctk.CTkButton(row, text="确定", width=92, command=_ok).pack(side="left", padx=6)
        ctk.CTkButton(row, text="取消", width=92, fg_color="#555", hover_color="#666",
                      command=dlg.destroy).pack(side="left", padx=6)
        try:
            self.wait_window(dlg)
        except Exception:
            pass
        return res.get("v")

    def _im(self):
        fp = filedialog.askopenfilename(title="导入 LUT", initialdir=self._ld or None,
                                        filetypes=[("cube LUT", "*.cube"), ("所有文件", "*.*")])
        if not fp:
            return
        p = Path(fp)
        if p.suffix.lower() != ".cube":
            messagebox.showerror("导入 LUT", "只支持 .cube 文件")
            return
        self._ld = str(p.parent)
        space = self._ask_lut_space(p.stem)
        if not space:
            return
        li, c = space
        dst = self._import_lut_into(str(p), li, c)
        if dst is None:                       # 目标已有同名文件
            target = Path(__file__).parent / "lut" / f"{li}-{c}" / p.name
            if not messagebox.askyesno("导入 LUT",
                                       f"「{li}-{c}」下已有同名文件：\n{target.name}\n\n覆盖？"):
                return
            dst = self._import_lut_into(str(p), li, c, overwrite=True)
        if dst is None:
            return
        self._refresh_lut_list(select=p.stem)
        self._lut_path.configure(text="lut/" + f"{li}-{c}/" + p.name)
        self._info.configure(text=f"已导入到 {li}-{c}")

    def _cur_lut(self):
        """当前下拉选中的 LUT 条目（含 path/name/li），没有则 None。"""
        n = self.ld.get()
        return next((l for l in self._luts if l["disp"] == n), None)

    def _lut_reveal(self):
        import subprocess
        l = self._cur_lut()
        if not l:
            messagebox.showinfo("在文件夹中显示", "当前没有选中已存在的 LUT")
            return
        p = Path(l["path"])
        try:
            subprocess.Popen(["explorer.exe", "/select,", str(p)])
        except Exception:
            try:
                os.startfile(str(p.parent))
            except Exception as e:
                messagebox.showerror("定位失败", str(e))

    def _lut_delete(self):
        l = self._cur_lut()
        if not l:
            messagebox.showinfo("删除 LUT", "当前没有选中已存在的 LUT")
            return
        p = Path(l["path"])
        if not messagebox.askyesno("删除 LUT",
                                   f"确定删除这个 LUT 文件？\n\n{l['name']}\n{p}\n\n此操作不可撤销。"):
            return
        try:
            p.unlink()
        except Exception as e:
            messagebox.showerror("删除失败", str(e))
            return
        self._tidy_lut_dir(p.parent)
        self._cd = None
        self._cd_path = ""
        self._refresh_lut_list()
        self._info.configure(text=f"已删除 {l['name']}")

    def _tidy_lut_dir(self, d):
        """删完/移完把空掉的「色域-曲线」文件夹收掉，但不动 lut 根目录。"""
        try:
            root = Path(__file__).parent / "lut"
            if d.resolve() != root.resolve() and d.is_dir() and not any(d.iterdir()):
                d.rmdir()
        except Exception:
            pass

    # ---------- 输出目录 ----------
    def _short_path(self, p, n=40):
        p = str(p)
        return p if len(p) <= n else "…" + p[-(n-1):]

    def _po(self):
        d = filedialog.askdirectory(title="输出", initialdir=self._od or None)
        if d:
            self._od = d; self.ol.configure(text=self._short_path(d)); self._save_cfg()

    # ---------- 批量转换 ----------
    def _export_one(self, fp, snap):
        """导出单个文件（fp: Path），复用批量转换同一套处理/编码逻辑。
        返回 True/False。不操作 UI 行状态，由调用方负责。"""
        ext = snap["ext"]
        try:
            s = dict(snap); s["raw"] = _is_raw(str(fp))
            # encode=False 是**必须**的：管线（resolve_source）对 RAW 宣称源曲线为 Linear，
            # 期望的是线性数据。此前这里用了默认的 encode=True，等于把 sRGB 编码后的数据
            # 再当线性送进管线 —— **双重编码**，导出比预览明显偏亮、丢对比度。
            # 三条预览路径（2078/2197/2267）一直都传了 encode=False，唯独导出漏了。
            a = _load_float(str(fp), work=self._raw_work,
                            wb=self._wb_args(), encode=False)
            a = np.clip(self._transform_tiled(a, s), 0, 1)
            op = os.path.join(self._od, fp.stem + ext)
            # ICC 必须描述**像素实际所处的空间**：套了 LUT 时那就是 LUT 的输出空间，
            # 不是 tg/tc。原来写死 tg/tc，等于让导出文件对色彩空间说谎。
            _og, _oc = display_leg_out_space(s)
            icc = _icc_bytes(_og, _oc)
            exif = _extract_exif(str(fp))
            if ext == ".heic":                      # HEIF 10-bit：按格式选 chroma/quality
                i10 = np.clip(a, 0, 1)
                h = (np.round(i10*1023).astype(np.uint16) << 6).astype(np.uint16)
                h = np.ascontiguousarray(h)
                if _HEIF:
                    _f = snap.get("fmt", "HEIF 444 无损")
                    if "444" in _f:      # 无损：444 + RGB 空间
                        kw = dict(quality=-1, chroma=444, matrix_coefficients=0)
                    elif "420" in _f:    # 紧凑
                        kw = dict(quality=100, chroma=420, matrix_coefficients=5)
                    else:                # 422 高质量
                        kw = dict(quality=100, chroma=422, matrix_coefficients=5)
                    if icc:
                        kw["icc_profile"] = icc        # 4.0：此前 HEIF 完全没嵌色彩空间
                    if exif:
                        kw["exif"] = exif
                    if max(a.shape[0], a.shape[1]) > 4096:   # 超大图 tiled 编码（x265 单帧有上限）
                        kw["tile_size"] = 2048
                    pillow_heif.encode("RGB;16", (a.shape[1], a.shape[0]), h.tobytes(), op, **kw)
            elif ext == ".tiff":
                i16 = np.ascontiguousarray((a*65535 + 0.5).astype(np.uint16))
                if _TIFF is not None:
                    et = [(34675, 7, len(icc), icc, False)] if icc else []
                    et += _exif_tiff_extratags(str(fp))
                    try:
                        _TIFF.imwrite(op, i16, photometric="rgb", compression="deflate",
                                      extratags=(et or None))
                    except Exception:
                        # 元数据写失败绝不能毁掉像素：退回只带 ICC，再退回裸写
                        _TIFF.imwrite(op, i16, photometric="rgb", compression="deflate",
                                      extratags=([(34675, 7, len(icc), icc, False)] if icc else None))
                else:
                    op = os.path.splitext(op)[0] + ".png"
                    _k2 = {"icc_profile": icc, "compress_level": self._png_level}
                    if exif:
                        _k2["exif"] = exif
                    Image.fromarray(_dither(a) if snap.get("dither") else (a*255 + 0.5).astype(np.uint8), "RGB").save(op, **_k2)
            else:
                _u8 = Image.fromarray(_dither(a) if snap.get("dither") else (a*255 + 0.5).astype(np.uint8), "RGB")
                _kw = {"quality": 95 if ext == ".jpg" else None, "icc_profile": icc}
                if ext != ".jpg":
                    # PNG 压缩级别（4.0）：33MP 实测 6→3 快 1.9×（3.15s→1.65s），体积仅 +11%（34.9→36.3MB）
                    _kw["compress_level"] = self._png_level
                if exif:
                    _kw["exif"] = exif
                _u8.save(op, **_kw)
            return True
        except PermissionError as e:
            _hint = ("输出目录不可写（ACL 正常却仍被拒，通常是 Windows「受控文件夹访问」"
                     "拦截了 python.exe）。在 App 里换一个输出目录，或在 Windows 安全中心允许本程序。")
            try:
                with open(Path(__file__).parent / "export_error.log", "a", encoding="utf-8") as _lf:
                    _lf.write(f"[{fp.name}] PermissionError: {e}\n    → {_hint}\n")
            except Exception:
                pass
            print(f"[{fp.name}] PermissionError: {e}\n    → {_hint}")
            return False
        except Exception as e:
            try:
                with open(Path(__file__).parent / "export_error.log", "a", encoding="utf-8") as _lf:
                    _lf.write(f"[{fp.name}] {type(e).__name__}: {e}\n")
            except Exception:
                pass
            print(f"[{fp.name}] {e}")
            return False

    def _out_writable(self, d):
        """导出前可写性预检：返回 (可写, 说明)。

        桌面 / 文档 / 图片 是 Windows「受控文件夹访问」的默认保护目录，
        非白名单程序（python.exe）写入会被拒（Errno 13），而目录 ACL 看起来一切正常。
        先探一次，就能立刻给一句人话，而不是让人去翻日志。"""
        try:
            os.makedirs(d, exist_ok=True)
            t = os.path.join(d, ".colorlab_write_test")
            with open(t, "wb") as f:
                f.write(b"0")
            os.remove(t)
            return True, ""
        except PermissionError:
            return False, ("输出目录写不进去：\n" + str(d) + "\n\n"
                           "桌面 / 文档 / 图片 是 Windows「受控文件夹访问」的默认保护目录，"
                           "非白名单程序会被拦截（ACL 看起来正常，但仍是 Access denied）。\n\n"
                           "两种解法：\n"
                           "  1) 换一个输出目录，例如项目内的 output 文件夹；\n"
                           "  2) 想在桌面输出：Windows 安全中心 → 病毒和威胁防护 → 勒索软件防护 → "
                           "受控文件夹访问 → 允许应用通过 → 加入本程序的 python.exe。")
        except Exception as e:
            return False, f"输出目录不可用：\n{d}\n\n{type(e).__name__}: {e}"

    def _export_current(self):
        """导出当前预览的那一张图（单张，不依赖批量列表）。"""
        if not self._preview_file:
            messagebox.showwarning("", "请先打开一张图片/RAW")
            return
        if not self._od:
            d = filedialog.askdirectory(title="选择输出文件夹", initialdir=self._od or None)
            if d:
                self._od = d; self.ol.configure(text=self._short_path(d))
            else:
                return
        _ok, _why = self._out_writable(self._od)
        if not _ok:
            self.st.configure(text="导出失败：输出目录不可写")
            messagebox.showerror("导出", _why); return
        snap = self._snapshot()
        snap["fmt"] = self.fd.get()
        snap["ext"] = _OUT_FMTS[snap["fmt"]]
        if snap["lut"] and snap["cd"] is None:
            messagebox.showwarning("", "LUT 未加载（勾选启用前请先选一个可用的 .cube）"); return
        fp = Path(self._preview_file)
        self.st.configure(text="导出中...")
        self.gb.configure(state="disabled")
        def work():
            ok = self._export_one(fp, snap)
            self.after(0, lambda: self.gb.configure(state="normal"))
            self.after(0, lambda: self.st.configure(
                text=f"已导出 {fp.name} → {self._od}" if ok else "导出失败，见 export_error.log"))
        threading.Thread(target=work, daemon=True).start()

    def _go(self):
        if self._running or not self._files: return
        if not self._od:
            d = filedialog.askdirectory(title="选择输出文件夹", initialdir=self._od or None)
            if d: self._od = d; self.ol.configure(text=self._short_path(d))
            else: messagebox.showwarning("", "请先选择输出文件夹"); return
        _ok, _why = self._out_writable(self._od)
        if not _ok:
            self.st.configure(text="批量转换失败：输出目录不可写")
            messagebox.showerror("批量输出", _why); return
        snap = self._snapshot()
        snap["fmt"] = self.fd.get()
        snap["ext"] = _OUT_FMTS[snap["fmt"]]
        if snap["lut"] and snap["cd"] is None:
            messagebox.showwarning("", "LUT 未加载（勾选启用前请先选一个可用的 .cube）"); return
        self._running = True
        self.gb.configure(state="disabled", text="转换中...")
        threading.Thread(target=self._wk, args=(snap,), daemon=True).start()

    def _wk(self, snap):
        total = len(self._files); ok = 0
        for i, fp in enumerate(self._files):
            if not self._running: break
            ri = i; r = self._rows
            self.after(0, lambda i=ri: i < len(r) and r[i].configure(text="..."))
            if self._export_one(fp, snap):
                ok += 1
                self.after(0, lambda i=ri: i < len(r) and r[i].configure(text="ok", text_color="#4caf50"))
            else:
                self.after(0, lambda i=ri: i < len(r) and r[i].configure(text="err", text_color="#f44336"))
            self.after(0, lambda p=(i+1)/total: self.pb.set(p))
            self.after(0, lambda c=ok, t=total: self.st.configure(text=f"{c}/{t}"))
        self._running = False
        self.after(0, lambda: self.gb.configure(state="normal", text="开始批量转换"))
        self.after(0, lambda: self.st.configure(text=f"完成 {ok}/{total}"))

if __name__ == "__main__":
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme(str(Path(__file__).parent / "assets" / "themes" / "orange.json"))
    App().mainloop()
