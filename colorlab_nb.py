#!/usr/bin/env python3
# colorlab_nb —— numba 加速内核（BETA）
# 渐进式优化：把逐元素运算（曲线编解码、shoulder、roll、grade、pipe）改写为
# numba 并行，避免 numpy 多次分配 1 亿像素中间数组。数值与 numpy 版逐位一致。
import numpy as np
from numba import njit, prange

try:
    import colorlab as _cl
    _HAVE_PARENT = True
except Exception:
    _HAVE_PARENT = False


# ---------------- 曲线 EOTF/OETF（标量，字符串分支，供并行循环内联） ----------------
@njit
def _eotf(v, name):
    """解码：显示编码值 → 线性。name 为曲线名。返回 float32。"""
    if name == "sRGB":
        return (v / 12.92) if v <= 0.04045 else (((v + 0.055) / 1.055) ** 2.4)
    elif name == "Rec.709":
        return (v / 4.5) if v < 0.081 else (((v + 0.099) / 1.099) ** (1 / 0.45))
    elif name == "S-Log3":
        th = (171.2102946929 / 1023.0)
        if v >= th:
            return ((10.0 ** ((v * 1023 - 420) / 261.5)) * (0.18 + 0.01) - 0.01)
        return ((v * 1023 - 95) * 0.01125 / (171.2102946929 - 95))
    elif name == "ARRI LogC3":
        _LC3c, _LC3a, _LC3b, _LC3c2, _LC3d, _LC3e, _LC3f = 0.010591, 5.555556, 0.052272, 0.247190, 0.385537, 5.367655, 0.092809
        et = (_LC3e * _LC3c + _LC3f)          # 官方：线性段 y = e*x + f
        if v <= et:
            return ((v - _LC3f) / _LC3e)
        return ((10.0 ** ((v - _LC3d) / _LC3c2) - _LC3b) / _LC3a)
    elif name == "V-Log":
        if v < 0.181:
            return ((v - 0.125) / 5.6)
        return (10.0 ** ((v - 0.598206) / 0.241514) - 0.00873)
    elif name == "RED Log3G10":
        _RGb, _RGc, _RGa, _RGl = 0.155954, 0.386231, 0.224282, 2.3025851
        if v >= _RGc:
            return (_RGb * 10.0 ** ((v - _RGc) / _RGa))
        return (_RGb + (v - _RGc) * _RGb * _RGl / _RGa)
    elif name == "ACEScct":
        if v <= 0.155251141552511:
            return ((v - 0.0729055341958355) / 10.5402377416545)
        return (2.0 ** (v * 17.52 - 9.72))
    elif name == "Gamma 1.8":
        return (v ** 1.8)
    elif name == "Gamma 2.2":
        return (v ** 2.2)
    elif name == "Gamma 2.4":
        return (v ** 2.4)
    elif name == "Gamma 2.6":
        return (v ** 2.6)
    else:  # Linear 及未知
        return (v)


@njit
def _oetf(v, name):
    """编码：线性 → 显示编码值。name 为曲线名。返回 float32。"""
    if name == "sRGB":
        return (v * 12.92) if v <= 0.0031308 else (1.055 * v ** (1 / 2.4) - 0.055)
    elif name == "Rec.709":
        return (v * 4.5) if v < 0.018 else (1.099 * v ** 0.45 - 0.099)
    elif name == "S-Log3":
        if v >= 0.01125:
            return ((420 + np.log10((v + 0.01) / (0.18 + 0.01)) * 261.5) / 1023.0)
        return ((v * (171.2102946929 - 95) / 0.01125 + 95) / 1023.0)
    elif name == "ARRI LogC3":
        _LC3c, _LC3a, _LC3b, _LC3c2, _LC3d, _LC3e, _LC3f = 0.010591, 5.555556, 0.052272, 0.247190, 0.385537, 5.367655, 0.092809
        if v <= _LC3c:
            return (_LC3e * v + _LC3f)        # 官方：线性段 y = e*x + f
        return (_LC3c2 * np.log10(_LC3a * v + _LC3b) + _LC3d)
    elif name == "V-Log":
        if v < 0.01:
            return (5.6 * v + 0.125)
        return (0.241514 * np.log10(v + 0.00873) + 0.598206)
    elif name == "RED Log3G10":
        _RGb, _RGc, _RGa, _RGl = 0.155954, 0.386231, 0.224282, 2.3025851
        if v >= _RGb:
            return (_RGc + _RGa * np.log10(v / _RGb))
        return (_RGc + _RGa * (v - _RGb) / (_RGb * _RGl))
    elif name == "ACEScct":
        if v <= 0.0078125:
            return (10.5402377416545 * v + 0.0729055341958355)
        return ((np.log2(v if v > 1e-12 else 1e-12) + 9.72) / 17.52)
    elif name == "Gamma 1.8":
        return (np.power(np.maximum(v, 0.0), 1.0 / 1.8))
    elif name == "Gamma 2.2":
        return (np.power(np.maximum(v, 0.0), 1.0 / 2.2))
    elif name == "Gamma 2.4":
        return (np.power(np.maximum(v, 0.0), 1.0 / 2.4))
    elif name == "Gamma 2.6":
        return (np.power(np.maximum(v, 0.0), 1.0 / 2.6))
    else:  # Linear 及未知
        return (v)


# ---------------- 曲线名 → 整数 ID（4A：去掉逐像素字符串比较） ----------------
# 精度承诺：下面的 _eotf_id/_oetf_id 与上面的 _eotf/_oetf 是**同一套表达式与常量**，
# 只把"比较字符串"换成"比较整数"。回归测试对每条曲线、每个采样点做**逐位相等**断言；
# 两套并存是刻意的——字符串版保留为参考实现，任何数值漂移都会被测试抓住。
_CID = {
    "sRGB": 0, "Rec.709": 1, "S-Log3": 2, "ARRI LogC3": 3, "V-Log": 4,
    "RED Log3G10": 5, "ACEScct": 6,
    "Gamma 1.8": 7, "Gamma 2.2": 8, "Gamma 2.4": 9, "Gamma 2.6": 10,
    "Linear": 11,
}
_CID_LINEAR = 11     # 未知曲线与 Linear 同义（与字符串版 else 分支一致）


def curve_id(name):
    """曲线名 → 整数 ID。未知名字回落 Linear，与 _eotf/_oetf 的 else 分支一致。"""
    return _CID.get(name, _CID_LINEAR)


@njit(cache=True)
def _eotf_id(v, cid):
    """解码：与 _eotf 逐位等价，分派改用整数。"""
    if cid == 0:
        return (v / 12.92) if v <= 0.04045 else (((v + 0.055) / 1.055) ** 2.4)
    elif cid == 1:
        return (v / 4.5) if v < 0.081 else (((v + 0.099) / 1.099) ** (1 / 0.45))
    elif cid == 2:
        th = (171.2102946929 / 1023.0)
        if v >= th:
            return ((10.0 ** ((v * 1023 - 420) / 261.5)) * (0.18 + 0.01) - 0.01)
        return ((v * 1023 - 95) * 0.01125 / (171.2102946929 - 95))
    elif cid == 3:
        _LC3c, _LC3a, _LC3b, _LC3c2, _LC3d, _LC3e, _LC3f = 0.010591, 5.555556, 0.052272, 0.247190, 0.385537, 5.367655, 0.092809
        et = (_LC3e * _LC3c + _LC3f)
        if v <= et:
            return ((v - _LC3f) / _LC3e)
        return ((10.0 ** ((v - _LC3d) / _LC3c2) - _LC3b) / _LC3a)
    elif cid == 4:
        if v < 0.181:
            return ((v - 0.125) / 5.6)
        return (10.0 ** ((v - 0.598206) / 0.241514) - 0.00873)
    elif cid == 5:
        _RGb, _RGc, _RGa, _RGl = 0.155954, 0.386231, 0.224282, 2.3025851
        if v >= _RGc:
            return (_RGb * 10.0 ** ((v - _RGc) / _RGa))
        return (_RGb + (v - _RGc) * _RGb * _RGl / _RGa)
    elif cid == 6:
        if v <= 0.155251141552511:
            return ((v - 0.0729055341958355) / 10.5402377416545)
        return (2.0 ** (v * 17.52 - 9.72))
    elif cid == 7:
        return (v ** 1.8)
    elif cid == 8:
        return (v ** 2.2)
    elif cid == 9:
        return (v ** 2.4)
    elif cid == 10:
        return (v ** 2.6)
    else:
        return (v)


@njit(cache=True)
def _oetf_id(v, cid):
    """编码：与 _oetf 逐位等价，分派改用整数。"""
    if cid == 0:
        return (v * 12.92) if v <= 0.0031308 else (1.055 * v ** (1 / 2.4) - 0.055)
    elif cid == 1:
        return (v * 4.5) if v < 0.018 else (1.099 * v ** 0.45 - 0.099)
    elif cid == 2:
        if v >= 0.01125:
            return ((420 + np.log10((v + 0.01) / (0.18 + 0.01)) * 261.5) / 1023.0)
        return ((v * (171.2102946929 - 95) / 0.01125 + 95) / 1023.0)
    elif cid == 3:
        _LC3c, _LC3a, _LC3b, _LC3c2, _LC3d, _LC3e, _LC3f = 0.010591, 5.555556, 0.052272, 0.247190, 0.385537, 5.367655, 0.092809
        if v <= _LC3c:
            return (_LC3e * v + _LC3f)
        return (_LC3c2 * np.log10(_LC3a * v + _LC3b) + _LC3d)
    elif cid == 4:
        if v < 0.01:
            return (5.6 * v + 0.125)
        return (0.241514 * np.log10(v + 0.00873) + 0.598206)
    elif cid == 5:
        _RGb, _RGc, _RGa, _RGl = 0.155954, 0.386231, 0.224282, 2.3025851
        if v >= _RGb:
            return (_RGc + _RGa * np.log10(v / _RGb))
        return (_RGc + _RGa * (v - _RGb) / (_RGb * _RGl))
    elif cid == 6:
        if v <= 0.0078125:
            return (10.5402377416545 * v + 0.0729055341958355)
        return ((np.log2(v if v > 1e-12 else 1e-12) + 9.72) / 17.52)
    elif cid == 7:
        return (np.power(np.maximum(v, 0.0), 1.0 / 1.8))
    elif cid == 8:
        return (np.power(np.maximum(v, 0.0), 1.0 / 2.2))
    elif cid == 9:
        return (np.power(np.maximum(v, 0.0), 1.0 / 2.4))
    elif cid == 10:
        return (np.power(np.maximum(v, 0.0), 1.0 / 2.6))
    else:
        return (v)


# ---------------- 数组级曲线编解码（并行） ----------------
@njit(cache=True, parallel=True)
def _decode_arr(a, name):
    flat = a.ravel()
    n = flat.size
    out = np.empty_like(flat)
    for i in prange(n):
        out[i] = _eotf(flat[i], name)
    return out.reshape(a.shape)


@njit(cache=True, parallel=True)
def _encode_arr(a, name):
    flat = a.ravel()
    n = flat.size
    out = np.empty_like(flat)
    for i in prange(n):
        out[i] = _oetf(flat[i], name)
    return out.reshape(a.shape)


@njit(cache=True, parallel=True)
def _decode_arr_id(a, cid):
    flat = a.ravel()
    n = flat.size
    out = np.empty_like(flat)
    for i in prange(n):
        out[i] = _eotf_id(flat[i], cid)
    return out.reshape(a.shape)


@njit(cache=True, parallel=True)
def _encode_arr_id(a, cid):
    flat = a.ravel()
    n = flat.size
    out = np.empty_like(flat)
    for i in prange(n):
        out[i] = _oetf_id(flat[i], cid)
    return out.reshape(a.shape)


# ---------------- sRGB 专用快速路径（RAW 主场景，避免字符串分支） ----------------
@njit(cache=True, parallel=True)
def _decode_srgb(a):
    h, w, c = a.shape
    out = np.empty_like(a)
    for i in prange(h):
        for j in range(w):
            for k in range(c):
                v = a[i, j, k]
                out[i, j, k] = v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4
    return out


@njit(cache=True, parallel=True)
def _encode_srgb(a):
    h, w, c = a.shape
    out = np.empty_like(a)
    for i in prange(h):
        for j in range(w):
            for k in range(c):
                v = a[i, j, k]
                out[i, j, k] = v * 12.92 if v <= 0.0031308 else 1.055 * v ** (1 / 2.4) - 0.055
    return out


# ---------------- 降采样（面积平均）----------------
# 必须在**线性域**做。对 gamma 编码值做平均会系统性压暗（Jensen 不等式）：
# 实测 sRGB 域平均再解码回线性，相对误差中位 13%，且 100% 的像素偏低。
# 用面积平均而不是 PIL 的 float32 LANCZOS，因为后者实测 97ms/帧（1706x2560→440x660），
# 拖滑块时完全不可用；面积平均每个输入像素只读一次，且对"缩小"本身就是合格的抗锯齿低通。
@njit(cache=True, parallel=True)
def _box_down(a, oh, ow):
    """面积平均降采样到 (oh, ow)。只在缩小时使用（调用方保证 oh<=h, ow<=w）。"""
    h, w, c = a.shape
    out = np.empty((oh, ow, c), dtype=np.float32)
    for oy in prange(oh):
        y0 = oy * h // oh
        y1 = (oy + 1) * h // oh
        if y1 <= y0:
            y1 = y0 + 1
        for ox in range(ow):
            x0 = ox * w // ow
            x1 = (ox + 1) * w // ow
            if x1 <= x0:
                x1 = x0 + 1
            s0 = 0.0; s1 = 0.0; s2 = 0.0      # float64 累加，避免大面积平均掉精度
            n = 0
            for yy in range(y0, y1):
                for xx in range(x0, x1):
                    s0 += a[yy, xx, 0]; s1 += a[yy, xx, 1]; s2 += a[yy, xx, 2]
                    n += 1
            inv = 1.0 / n
            out[oy, ox, 0] = s0 * inv
            out[oy, ox, 1] = s1 * inv
            out[oy, ox, 2] = s2 * inv
    return out


# ---------------- shoulder（高光软肩，标量） ----------------
@njit
def _shoulder_v(x, k):
    knee = 0.9 - 0.12 * max(0.0, k)
    knee = max(0.35, min(0.95, knee))
    if x <= knee:
        return x
    over = x - knee
    c = 1.0 / (1.0 - knee)
    comp = (over / (over + 1.0 / c)) * (1.0 - knee)
    return min(knee + comp, 1.0)


# ---------------- roll（色域缩限，逐像素） ----------------
@njit(cache=True, parallel=True)
def _roll_arr(a, k):
    h, w, c = a.shape
    out = np.empty_like(a)
    for i in prange(h):
        for j in range(w):
            r = a[i, j, 0]; g = a[i, j, 1]; b = a[i, j, 2]
            mx = max(r, g, b); mn = min(r, g, b)
            if mx <= 1.0 and mn >= 0.0:
                out[i, j, 0] = r; out[i, j, 1] = g; out[i, j, 2] = b
                continue
            Y = 0.2126 * r + 0.7152 * g + 0.0722 * b
            Ys = _shoulder_v(Y, k)
            s_mx = (1.0 - Ys) / max(mx - Y, 1e-6) if mx > 1.0 else 1.0
            s_mn = Ys / max(Y - mn, 1e-6) if mn < 0.0 else 1.0
            s = min(s_mx, s_mn)
            s = max(0.0, min(1.0, s))
            t = 1.0 - s
            s = 1.0 - t * t * (3 - 2 * t)
            out[i, j, 0] = min(max(Ys + (r - Y) * s, 0.0), 1.0)
            out[i, j, 1] = min(max(Ys + (g - Y) * s, 0.0), 1.0)
            out[i, j, 2] = min(max(Ys + (b - Y) * s, 0.0), 1.0)
    return out


# ---------------- grade（基础调色，并行，输出仍在线性域） ----------------
@njit(cache=True, parallel=True)
def _grade_lin(arr, ev, contrast, sat, temp, kgr, kgg, kgb):
    """在线性域做调色（输入已是线性）。返回调色后的线性值。

    kgr/kgg/kgb：色温三通道增益，由调用方用 colorlab._kelvin_gain(temp) 预先算好传入
    （njit 内核里调不了 numpy 版 _kelvin_gain）。
    注：早期的「亮度 + 滚降（shoulder 软肩）」已整条删除。"""
    h, w, c = arr.shape
    out = np.empty_like(arr)
    k_ev = 2.0 ** ev
    for i in prange(h):
        for j in range(w):
            r = arr[i, j, 0]; g = arr[i, j, 1]; b = arr[i, j, 2]
            if ev:
                r *= k_ev; g *= k_ev; b *= k_ev
            if contrast:
                kc = 1.0 + contrast
                r = (r - 0.18) * kc + 0.18
                g = (g - 0.18) * kc + 0.18
                b = (b - 0.18) * kc + 0.18
            if sat:
                Y = 0.2126 * r + 0.7152 * g + 0.0722 * b
                ks = 1.0 + sat
                r = Y + (r - Y) * ks
                g = Y + (g - Y) * ks
                b = Y + (b - Y) * ks
            if temp:
                r *= kgr
                g *= kgg
                b *= kgb
            r = max(r, 0.0); g = max(g, 0.0); b = max(b, 0.0)
            out[i, j, 0] = r; out[i, j, 1] = g; out[i, j, 2] = b
    return out


# ---------------- 色域缩限内核：ACES RGC（与 colorlab.gamut_compress 等价） ----------------
# 参数与 colorlab.py 是同一套（规范 l/t/p）；这里作为 njit 的编译期常量，
# 并由回归测试断言两边数值一致，防止两处漂移。
_RGC_LIM_R, _RGC_LIM_G, _RGC_LIM_B = 1.147, 1.264, 1.312
_RGC_THR_R, _RGC_THR_G, _RGC_THR_B = 0.815, 0.803, 0.880
_RGC_PWR = 1.2


def _rgc_scale(l, t, p=_RGC_PWR):
    return (l - t) / (((1.0 - t) / (l - t)) ** (-p) - 1.0) ** (1.0 / p)


_RGC_S_R = _rgc_scale(_RGC_LIM_R, _RGC_THR_R)
_RGC_S_G = _rgc_scale(_RGC_LIM_G, _RGC_THR_G)
_RGC_S_B = _rgc_scale(_RGC_LIM_B, _RGC_THR_B)


@njit(cache=True, inline='always')
def _rgc_d(d, t, s):
    """ACES RGC 规范 Equation 4：压缩归一化距离（d < t 原样返回 = zone of trust）。"""
    if d < t:
        return d
    nd = (d - t) / s
    return t + s * nd / (1.0 + nd ** _RGC_PWR) ** (1.0 / _RGC_PWR)


@njit(cache=True, inline='always')
def _gc(r, g, b, gc_mode, k, use_roll):
    """色域缩限。gc_mode: 0 = ACES RGC（默认），1 = 旧的自研 _roll，2 = 不缩限。

    2 是给**显示器转换**用的：目标空间 → 显示器空间是纯色度操作，
    图像进入这一步之前已按目标色域缩限过一次，再压一次就是双重映射。"""
    if gc_mode == 2:
        return r, g, b
    if gc_mode == 0:
        A = max(r, g, b)
        Aa = abs(A)
        if Aa == 0.0:
            return 0.0, 0.0, 0.0                      # 规范 Step 3：A == 0 时 dn = 0
        return (A - _rgc_d((A - r) / Aa, _RGC_THR_R, _RGC_S_R) * Aa,
                A - _rgc_d((A - g) / Aa, _RGC_THR_G, _RGC_S_G) * Aa,
                A - _rgc_d((A - b) / Aa, _RGC_THR_B, _RGC_S_B) * Aa)
    if not use_roll:
        return r, g, b
    mx = max(r, g, b); mn = min(r, g, b)
    if mx <= 1.0 and mn >= 0.0:
        return r, g, b
    Y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    Ys = _shoulder_v(Y, k)
    s_mx = (1.0 - Ys) / max(mx - Y, 1e-6) if mx > 1.0 else 1.0
    s_mn = Ys / max(Y - mn, 1e-6) if mn < 0.0 else 1.0
    s = max(0.0, min(1.0, min(s_mx, s_mn)))
    t = 1.0 - s
    s = 1.0 - t * t * (3 - 2 * t)
    return (max(min(Ys + (r - Y) * s, 1.0), 0.0),
            max(min(Ys + (g - Y) * s, 1.0), 0.0),
            max(min(Ys + (b - Y) * s, 1.0), 0.0))


@njit(cache=True, parallel=True)
def rgc_fast(a):
    """ACES RGC 的 numba 版（等价于 colorlab.gamut_compress）。a: (H,W,3) float32。"""
    h, w, _ = a.shape
    out = np.empty_like(a)
    for i in prange(h):
        for j in range(w):
            r, g, b = _gc(a[i, j, 0], a[i, j, 1], a[i, j, 2], 0, 0.9, True)
            out[i, j, 0] = r
            out[i, j, 1] = g
            out[i, j, 2] = b
    return out


# ---------------- pipe（CST：曲线解码→矩阵→CAT→矩阵→缩限(RGC)→OOTF→编码） ----------------
@njit(cache=True, parallel=True)
def _pipe_go(arr, M, Mi, Cat, sc_id, tc_id, sg_display, tg_display, ootf_mode, gam_sg, gam_tg,
             k, use_roll, gc_mode):
    """完整色彩空间转换。M: sg→XYZ, Mi: XYZ→tg, Cat: 白点适配矩阵(或单位)。
    ootf_mode: 0=关/1=reverse/2=forward/3=auto(由调用方解析)。"""
    h, w, c = arr.shape
    out = np.empty_like(arr)
    for i in prange(h):
        for j in range(w):
            r = arr[i, j, 0]; g = arr[i, j, 1]; b = arr[i, j, 2]
            # 曲线解码（sc → 线性）
            r = _eotf_id(r, sc_id); g = _eotf_id(g, sc_id); b = _eotf_id(b, sc_id)
            # OOTF reverse（display→scene，剥离补偿，负值先夹 0 与 numpy 一致）
            if ootf_mode == 1 or (ootf_mode == 3 and sg_display and not tg_display):
                gam = gam_sg
                r = max(r, 0.0) ** (1.0 / gam); g = max(g, 0.0) ** (1.0 / gam); b = max(b, 0.0) ** (1.0 / gam)
            # sg RGB → XYZ
            x = M[0, 0] * r + M[0, 1] * g + M[0, 2] * b
            y = M[1, 0] * r + M[1, 1] * g + M[1, 2] * b
            z = M[2, 0] * r + M[2, 1] * g + M[2, 2] * b
            # CAT（白点适配）
            if Cat[0, 0] != 1.0 or Cat[1, 1] != 1.0 or Cat[2, 2] != 1.0:
                x2 = Cat[0, 0] * x + Cat[0, 1] * y + Cat[0, 2] * z
                y2 = Cat[1, 0] * x + Cat[1, 1] * y + Cat[1, 2] * z
                z2 = Cat[2, 0] * x + Cat[2, 1] * y + Cat[2, 2] * z
                x, y, z = x2, y2, z2
            # XYZ → tg RGB（线性）
            r = Mi[0, 0] * x + Mi[0, 1] * y + Mi[0, 2] * z
            g = Mi[1, 0] * x + Mi[1, 1] * y + Mi[1, 2] * z
            b = Mi[2, 0] * x + Mi[2, 1] * y + Mi[2, 2] * z
            r, g, b = _gc(r, g, b, gc_mode, k, use_roll)
            # OOTF forward（scene→display，加补偿，负值先夹 0 与 numpy 一致）
            if ootf_mode == 2 or (ootf_mode == 3 and tg_display):
                gam = gam_tg
                r = max(r, 0.0) ** gam; g = max(g, 0.0) ** gam; b = max(b, 0.0) ** gam
            # 曲线编码（→ tc）
            out[i, j, 0] = _oetf_id(r, tc_id)
            out[i, j, 1] = _oetf_id(g, tc_id)
            out[i, j, 2] = _oetf_id(b, tc_id)
    return out


# ---------------- sRGB→sRGB 专用 pipe（RAW 主场景：工作空间→sRGB 输出） ----------------
@njit(cache=True, parallel=True)
def _pipe_go_srgb_out(arr, M, Mi, Cat, do_reverse, do_forward, gam_sg, gam_tg, k, use_roll, gc_mode):
    """sc=tc=sRGB 特化：sRGB 解码→(矩阵→CAT→矩阵)→roll→OOTF→sRGB 编码。
    无字符串分支，面向 RAW 工作空间→sRGB 目标的最常见导出。
    do_reverse/do_forward 由调用方按 OOTF 模式与 display-ref 判定。"""
    h, w, c = arr.shape
    out = np.empty_like(arr)
    for i in prange(h):
        for j in range(w):
            r = arr[i, j, 0]; g = arr[i, j, 1]; b = arr[i, j, 2]
            # sRGB decode
            r = r / 12.92 if r <= 0.04045 else ((r + 0.055) / 1.055) ** 2.4
            g = g / 12.92 if g <= 0.04045 else ((g + 0.055) / 1.055) ** 2.4
            b = b / 12.92 if b <= 0.04045 else ((b + 0.055) / 1.055) ** 2.4
            # OOTF reverse（display→scene）
            if do_reverse:
                r = max(r, 0.0) ** (1.0 / gam_sg)
                g = max(g, 0.0) ** (1.0 / gam_sg)
                b = max(b, 0.0) ** (1.0 / gam_sg)
            # sg RGB → XYZ → CAT → XYZ → sRGB
            x = M[0, 0] * r + M[0, 1] * g + M[0, 2] * b
            y = M[1, 0] * r + M[1, 1] * g + M[1, 2] * b
            z = M[2, 0] * r + M[2, 1] * g + M[2, 2] * b
            if Cat[0, 0] != 1.0 or Cat[1, 1] != 1.0 or Cat[2, 2] != 1.0:
                x2 = Cat[0, 0] * x + Cat[0, 1] * y + Cat[0, 2] * z
                y2 = Cat[1, 0] * x + Cat[1, 1] * y + Cat[1, 2] * z
                z2 = Cat[2, 0] * x + Cat[2, 1] * y + Cat[2, 2] * z
                x, y, z = x2, y2, z2
            r = Mi[0, 0] * x + Mi[0, 1] * y + Mi[0, 2] * z
            g = Mi[1, 0] * x + Mi[1, 1] * y + Mi[1, 2] * z
            b = Mi[2, 0] * x + Mi[2, 1] * y + Mi[2, 2] * z
            # 缩限（ACES RGC / 旧 _roll，由 gc_mode 决定）
            r, g, b = _gc(r, g, b, gc_mode, k, use_roll)
            # OOTF forward（scene→display，sRGB 显示补偿）
            if do_forward:
                r = max(r, 0.0) ** gam_tg
                g = max(g, 0.0) ** gam_tg
                b = max(b, 0.0) ** gam_tg
            # sRGB encode
            out[i, j, 0] = r * 12.92 if r <= 0.0031308 else 1.055 * r ** (1 / 2.4) - 0.055
            out[i, j, 1] = g * 12.92 if g <= 0.0031308 else 1.055 * g ** (1 / 2.4) - 0.055
            out[i, j, 2] = b * 12.92 if b <= 0.0031308 else 1.055 * b ** (1 / 2.4) - 0.055
    return out


# ---------------- 便捷入口（供 colorlab.py 调用） ----------------
def grade_fast(a, work, ev=0.0, contrast=0.0, sat=0.0, temp=0.0):
    """numba 版调色：输入 sRGB 曲线编码（显示域），输出调色后显示域。与 colorlab._grade 数值一致。"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    # 色温增益在 Python 侧算好（njit 内核无法调用 _kelvin_gain）
    kg = _cl._kelvin_gain(float(temp)) if temp else (1.0, 1.0, 1.0)
    kgr, kgg, kgb = float(kg[0]), float(kg[1]), float(kg[2])
    if work == "sRGB":                   # RAW 主路径：sRGB 专用，避免字符串分支
        lin = _decode_srgb(a)
        lin2 = _grade_lin(lin, ev, contrast, sat, temp, kgr, kgg, kgb)
        return _encode_srgb(lin2)
    if work == "Linear":
        # 输入已是线性（RAW 解码层的原生输出）：直接在线性域调色，输出保持线性。
        # 此前会走"恒等解码 + 调色 + 恒等编码"三趟，而恒等两趟是纯粹的整图拷贝白跑
        # （33MP 实测每趟约 240ms）。线性域是唯一不该往返编码的域。
        return _grade_lin(a, ev, contrast, sat, temp, kgr, kgg, kgb)
    if work not in _cl.CURVE:            # 保持"未知曲线立即报错"，不静默回落
        raise KeyError(f'未知曲线: {work}')
    cid = curve_id(work)                 # 曲线名 → 整数 ID（一次解析，不进逐像素循环）
    lin = _decode_arr_id(a, cid)         # 解码到线性
    lin2 = _grade_lin(lin, ev, contrast, sat, temp, kgr, kgg, kgb)
    return _encode_arr_id(lin2, cid)


def pipe_params(sg, sc, tg, tc, k=0.9, ootf="auto", gc=None):
    """把「色域/曲线/OOTF 选项」解析成内核要用的全部参数。

    **CPU 与 GPU 两条路都必须走这里**，否则参数解析逻辑会出现第二份实现，
    两边悄悄漂移（本项目已经吃过"两处参数不一致"的亏）。
    返回 dict；曲线名未知时抛 KeyError（与 numpy 版行为一致，不静默回落）。
    """
    M, _, W = _cl._mat(sg)
    _, Mi, _ = _cl._mat(tg)
    _, _, W2 = _cl._mat(tg)
    Cat = np.eye(3, dtype=np.float32) if np.allclose(W, W2, atol=1e-6) else _cl._cat(W, W2)
    # OOTF 模式解析。auto 看**曲线**不看色域 —— 参考类型由曲线唯一决定
    # （Rec.2020+Gamma2.4 是显示参考，Rec.2020+V-Log 是场景参考，同名色域两种参考）。
    if ootf == "reverse" or (ootf == "auto" and _cl._ref_type(sc) == "display"
                             and _cl._ref_type(tc) != "display"):
        mode = 1
    elif ootf == "forward" or (ootf == "auto" and _cl._ref_type(tc) == "display"
                               and _cl._ref_type(sc) != "display"):
        mode = 2
    elif ootf == "off":
        mode = 0
    else:
        mode = 3
    for _nm in (sc, tc):                 # 未知曲线立即报错，不静默回落
        if _nm not in _cl.CURVE:
            raise KeyError(f'未知曲线: {_nm}')
    return dict(
        M=np.ascontiguousarray(M, dtype=np.float32),
        Mi=np.ascontiguousarray(Mi, dtype=np.float32),
        Cat=np.ascontiguousarray(Cat, dtype=np.float32),
        sc_id=curve_id(sc), tc_id=curve_id(tc),
        sg_disp=sg in _cl._DISPLAY_REF, tg_disp=tg in _cl._DISPLAY_REF,
        ootf_mode=mode,
        gam_sg=_cl._OOTF_GAMMA.get(sg, 1.2), gam_tg=_cl._OOTF_GAMMA.get(tg, 1.2),
        use_roll=tc not in _cl._LOG_CURVES,
        gc_mode=(int(gc) if gc is not None
                 else (1 if getattr(_cl, "_GC_MODE", "rgc") == "roll" else 0)),
        srgb_out=(sc == "sRGB" and tc == "sRGB"),
        same=(sg == tg and sc == tc),
        k=k,
    )


def pipe_fast(r, sg, sc, tg, tc, k=0.9, ootf="auto", gc=None):
    """numba 版 CST。与 colorlab.Pipe.go 数值一致。"""
    r = np.ascontiguousarray(r, dtype=np.float32)
    P = pipe_params(sg, sc, tg, tc, k=k, ootf=ootf, gc=gc)
    if P["same"]:
        return r
    M, Mi, Cat = P["M"], P["Mi"], P["Cat"]
    gam_sg, gam_tg = P["gam_sg"], P["gam_tg"]
    use_roll, gc_mode, mode = P["use_roll"], P["gc_mode"], P["ootf_mode"]
    if P["srgb_out"]:
        do_reverse = (mode == 1) or (mode == 3 and P["sg_disp"] and not P["tg_disp"])
        do_forward = (mode == 2) or (mode == 3 and P["tg_disp"])
        return _pipe_go_srgb_out(r, M, Mi, Cat, do_reverse, do_forward, gam_sg, gam_tg, k, use_roll, gc_mode)
    return _pipe_go(r, M, Mi, Cat, P["sc_id"], P["tc_id"],
                    P["sg_disp"], P["tg_disp"], mode,
                    gam_sg, gam_tg, k, use_roll, gc_mode)

# ---------------- LUT 内核（由 colorlab-opt/nb_lut.py 并入） ----------------
# 等价于 colorlab._al / _al3d / _al1d，但一次读入→计算→写出，不造 8 个角点全图数组。
@njit(cache=True, inline='always')
def _tetra(fr, fg, fb, c000, c100, c010, c110, c001, c101, c011, c111):
    """四面体插值，分支与 colorlab._al3d 的 6 个掩码一一对应（并列时权重为 0，结果一致）。"""
    if fr >= fg and fr >= fb:                 # r 最大
        if fg >= fb:
            w1 = fr - fg; w2 = fg - fb; w3 = fb; w0 = 1.0 - fr
            return (w0 * c000[0] + w1 * c100[0] + w2 * c110[0] + w3 * c111[0],
                    w0 * c000[1] + w1 * c100[1] + w2 * c110[1] + w3 * c111[1],
                    w0 * c000[2] + w1 * c100[2] + w2 * c110[2] + w3 * c111[2])
        else:
            w1 = fr - fb; w2 = fb - fg; w3 = fg; w0 = 1.0 - fr
            return (w0 * c000[0] + w1 * c100[0] + w2 * c101[0] + w3 * c111[0],
                    w0 * c000[1] + w1 * c100[1] + w2 * c101[1] + w3 * c111[1],
                    w0 * c000[2] + w1 * c100[2] + w2 * c101[2] + w3 * c111[2])
    elif fg >= fr and fg >= fb:               # g 最大
        if fr >= fb:
            w1 = fg - fr; w2 = fr - fb; w3 = fb; w0 = 1.0 - fg
            return (w0 * c000[0] + w1 * c010[0] + w2 * c110[0] + w3 * c111[0],
                    w0 * c000[1] + w1 * c010[1] + w2 * c110[1] + w3 * c111[1],
                    w0 * c000[2] + w1 * c010[2] + w2 * c110[2] + w3 * c111[2])
        else:
            w1 = fg - fb; w2 = fb - fr; w3 = fr; w0 = 1.0 - fg
            return (w0 * c000[0] + w1 * c010[0] + w2 * c011[0] + w3 * c111[0],
                    w0 * c000[1] + w1 * c010[1] + w2 * c011[1] + w3 * c111[1],
                    w0 * c000[2] + w1 * c010[2] + w2 * c011[2] + w3 * c111[2])
    else:                                     # b 最大
        if fr >= fg:
            w1 = fb - fr; w2 = fr - fg; w3 = fg; w0 = 1.0 - fb
            return (w0 * c000[0] + w1 * c001[0] + w2 * c101[0] + w3 * c111[0],
                    w0 * c000[1] + w1 * c001[1] + w2 * c101[1] + w3 * c111[1],
                    w0 * c000[2] + w1 * c001[2] + w2 * c101[2] + w3 * c111[2])
        else:
            w1 = fb - fg; w2 = fg - fr; w3 = fr; w0 = 1.0 - fb
            return (w0 * c000[0] + w1 * c001[0] + w2 * c011[0] + w3 * c111[0],
                    w0 * c000[1] + w1 * c001[1] + w2 * c011[1] + w3 * c111[1],
                    w0 * c000[2] + w1 * c001[2] + w2 * c011[2] + w3 * c111[2])


@njit(cache=True, parallel=True)
def _lut3d(a, L, N, scale):
    h, w, _ = a.shape
    out = np.empty_like(a)
    s = float(N - 1)
    n1 = N - 1
    for i in prange(h):
        for j in range(w):
            r = a[i, j, 0]; g = a[i, j, 1]; b = a[i, j, 2]
            if r < 0.0: r = 0.0
            elif r > 1.0: r = 1.0
            if g < 0.0: g = 0.0
            elif g > 1.0: g = 1.0
            if b < 0.0: b = 0.0
            elif b > 1.0: b = 1.0
            cr = r * s; cg = g * s; cb = b * s
            r0 = int(cr); g0 = int(cg); b0 = int(cb)
            if r0 > n1 - 1: r0 = n1 - 1
            if g0 > n1 - 1: g0 = n1 - 1
            if b0 > n1 - 1: b0 = n1 - 1
            if r0 < 0: r0 = 0
            if g0 < 0: g0 = 0
            if b0 < 0: b0 = 0
            fr = cr - r0; fg = cg - g0; fb = cb - b0
            r1 = r0 + 1; g1 = g0 + 1; b1 = b0 + 1
            o0, o1, o2 = _tetra(fr, fg, fb,
                                L[r0, g0, b0], L[r1, g0, b0], L[r0, g1, b0], L[r1, g1, b0],
                                L[r0, g0, b1], L[r1, g0, b1], L[r0, g1, b1], L[r1, g1, b1])
            if scale != 1.0:
                o0 *= scale; o1 *= scale; o2 *= scale
            if o0 < 0.0: o0 = 0.0
            elif o0 > 1.0: o0 = 1.0
            if o1 < 0.0: o1 = 0.0
            elif o1 > 1.0: o1 = 1.0
            if o2 < 0.0: o2 = 0.0
            elif o2 > 1.0: o2 = 1.0
            out[i, j, 0] = o0
            out[i, j, 1] = o1
            out[i, j, 2] = o2
    return out


@njit(cache=True, parallel=True)
def _lut1d(a, d, scale):
    h, w, _ = a.shape
    out = np.empty_like(a)
    N = d.shape[1]
    sl = float(N - 1)
    for i in prange(h):
        for j in range(w):
            for c in range(3):
                v = a[i, j, c]
                if v < 0.0: v = 0.0
                elif v > 1.0: v = 1.0
                idx = v * sl
                i0 = int(idx)
                if i0 > N - 2: i0 = N - 2
                if i0 < 0: i0 = 0
                f = idx - i0
                o = d[c, i0] + (d[c, i0 + 1] - d[c, i0]) * f
                if scale != 1.0: o *= scale
                if o < 0.0: o = 0.0
                elif o > 1.0: o = 1.0
                out[i, j, c] = o
    return out


def _prep(inf):
    """把 LUT 数据准备成 numba 能直接吃的 C 连续 float32（缓存到 inf 里，只做一次）。"""
    L = inf.get("_nb_c")
    if L is None:
        d = np.ascontiguousarray(inf["d"], dtype=np.float32)
        m = float(d.max()) if d.size else 1.0
        scale = (1.0 / m) if m > 1.01 else 1.0   # 与 colorlab._al 的整数域回缩一致
        inf["_nb_c"] = d
        inf["_nb_scale"] = scale
        L = d
    return L, float(inf.get("_nb_scale", 1.0))


def lut_fast(a, inf):
    """等价于 colorlab._al(a, inf)。a: float32/任意 (H,W,3)，返回 float32 0~1。"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    L, scale = _prep(inf)
    if inf["t"] == "3d":
        return _lut3d(a, L, int(inf["n"]), scale)
    return _lut1d(a, L, scale)


if __name__ == "__main__":
    import time
    x = np.random.RandomState(0).rand(4373, 5831, 3).astype(np.float32)
    t0 = time.perf_counter()
    r = grade_fast(x, "sRGB", 0.5, 0.2, 0.3, 0.0, 0.3, 0.6)
    print(f"grade_fast: {time.perf_counter()-t0:.2f}s")
    t0 = time.perf_counter()
    r2 = pipe_fast(x, "ACES2065-1", "sRGB", "sRGB", "sRGB")
    print(f"pipe_fast: {time.perf_counter()-t0:.2f}s")
