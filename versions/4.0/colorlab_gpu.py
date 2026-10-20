#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Color-Lab 4.0 · CUDA 后端（CuPy + RawKernel）

设计要点
--------
1. **与 colorlab_nb 逐条镜像**：_eotf_id / _oetf_id / _gc / _rgc_d / _tetra / _grade_lin /
   _pipe_go / _lut3d 全部一一对应，只把「数组循环」换成 CUDA 的线程索引。
   任何一处数学改动都必须两边同时改，并由验收脚本比对。
2. **曲线 ID 是一致（uniform）参数**：整次 launch 所有线程走同一个分支，
   所以那个大 switch 在 GPU 上不花代价，不需要为每种曲线单独编译内核。
3. **精度**：numba 版因为字面量是 float64，逐像素数学实际在 **float64** 里算、
   只在存取数组时落到 float32。这里用 REAL typedef 复刻同一选择，
   并提供 float32 变体用于对照（见 precision 参数）。
4. **不使用 cuBLAS/cuDNN**：3×3 矩阵直接展开，避免多一个依赖、也避免 GEMM 的额外拷贝。

本模块不 import colorlab（避免循环依赖）；矩阵/LUT/曲线参数一律由调用方传入。
"""
import os
import numpy as np

# kernel 缓存放进项目内，保持"不污染系统"的项目约定
_HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("CUPY_CACHE_DIR", os.path.join(_HERE, ".cupy_cache"))

try:
    import cupy as cp
    _HAS_CUPY = True
except Exception:                                    # 没装/驱动不可用 → 静默降级
    cp = None
    _HAS_CUPY = False

_DEVICE_OK = None
_LAST_ERR = ""


def available():
    """CUDA 是否真的可用（导入成功 ≠ 能跑）。结果缓存。"""
    global _DEVICE_OK, _LAST_ERR
    if _DEVICE_OK is not None:
        return _DEVICE_OK
    if not _HAS_CUPY:
        _LAST_ERR = "未安装 cupy"; _DEVICE_OK = False; return False
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            _LAST_ERR = "没有可见的 CUDA 设备"; _DEVICE_OK = False; return False
        cp.zeros(1, dtype=cp.float32)                # 真跑一次，避免只探测到驱动
        _DEVICE_OK = True
    except Exception as e:
        _LAST_ERR = f"{type(e).__name__}: {e}"; _DEVICE_OK = False
    return _DEVICE_OK


def last_error():
    return _LAST_ERR


# ----------------------------------------------------------------------------
# CUDA 源码
# ----------------------------------------------------------------------------
_SRC_TMPL = r"""
typedef %(REAL)s real;

#define RGC_PWR 1.2

// ---------------- 曲线解码（镜像 colorlab_nb._eotf_id） ----------------
__device__ __forceinline__ real eotf_id(real v, int cid) {
    if (cid == 0) {
        return (v <= (real)0.04045) ? (v / (real)12.92)
                                    : pow((v + (real)0.055) / (real)1.055, (real)2.4);
    } else if (cid == 1) {
        return (v < (real)0.081) ? (v / (real)4.5)
                                 : pow((v + (real)0.099) / (real)1.099, (real)(1.0/0.45));
    } else if (cid == 2) {
        real th = (real)(171.2102946929 / 1023.0);
        if (v >= th)
            return pow((real)10.0, (v * (real)1023 - (real)420) / (real)261.5) * (real)(0.18 + 0.01) - (real)0.01;
        return (v * (real)1023 - (real)95) * (real)0.01125 / (real)(171.2102946929 - 95);
    } else if (cid == 3) {
        const real c = (real)0.010591, a = (real)5.555556, b = (real)0.052272;
        const real c2 = (real)0.247190, d = (real)0.385537, e = (real)5.367655, f = (real)0.092809;
        real et = e * c + f;
        if (v <= et) return (v - f) / e;
        return (pow((real)10.0, (v - d) / c2) - b) / a;
    } else if (cid == 4) {
        if (v < (real)0.181) return (v - (real)0.125) / (real)5.6;
        return pow((real)10.0, (v - (real)0.598206) / (real)0.241514) - (real)0.00873;
    } else if (cid == 5) {
        const real gb = (real)0.155954, gc_ = (real)0.386231, ga = (real)0.224282, gl = (real)2.3025851;
        if (v >= gc_) return gb * pow((real)10.0, (v - gc_) / ga);
        return gb + (v - gc_) * gb * gl / ga;
    } else if (cid == 6) {
        if (v <= (real)0.155251141552511) return (v - (real)0.0729055341958355) / (real)10.5402377416545;
        return pow((real)2.0, v * (real)17.52 - (real)9.72);
    } else if (cid == 7) {
        return pow(v, (real)1.8);
    } else if (cid == 8) {
        return pow(v, (real)2.2);
    } else if (cid == 9) {
        return pow(v, (real)2.4);
    } else if (cid == 10) {
        return pow(v, (real)2.6);
    }
    return v;                                   // Linear / 未知
}

// ---------------- 曲线编码（镜像 colorlab_nb._oetf_id） ----------------
__device__ __forceinline__ real oetf_id(real v, int cid) {
    if (cid == 0) {
        return (v <= (real)0.0031308) ? (v * (real)12.92)
                                      : ((real)1.055 * pow(v, (real)(1.0/2.4)) - (real)0.055);
    } else if (cid == 1) {
        return (v < (real)0.018) ? (v * (real)4.5)
                                 : ((real)1.099 * pow(v, (real)0.45) - (real)0.099);
    } else if (cid == 2) {
        if (v >= (real)0.01125)
            return ((real)420 + log10((v + (real)0.01) / (real)(0.18 + 0.01)) * (real)261.5) / (real)1023.0;
        return (v * (real)(171.2102946929 - 95) / (real)0.01125 + (real)95) / (real)1023.0;
    } else if (cid == 3) {
        const real c = (real)0.010591, a = (real)5.555556, b = (real)0.052272;
        const real c2 = (real)0.247190, d = (real)0.385537, e = (real)5.367655, f = (real)0.092809;
        if (v <= c) return e * v + f;
        return c2 * log10(a * v + b) + d;
    } else if (cid == 4) {
        if (v < (real)0.01) return (real)5.6 * v + (real)0.125;
        return (real)0.241514 * log10(v + (real)0.00873) + (real)0.598206;
    } else if (cid == 5) {
        const real gb = (real)0.155954, gc_ = (real)0.386231, ga = (real)0.224282, gl = (real)2.3025851;
        if (v >= gb) return gc_ + ga * log10(v / gb);
        return gc_ + ga * (v - gb) / (gb * gl);
    } else if (cid == 6) {
        if (v <= (real)0.0078125) return (real)10.5402377416545 * v + (real)0.0729055341958355;
        real vv = (v > (real)1e-12) ? v : (real)1e-12;
        return (log2(vv) + (real)9.72) / (real)17.52;
    } else if (cid == 7) {
        return pow(fmax(v, (real)0.0), (real)(1.0/1.8));
    } else if (cid == 8) {
        return pow(fmax(v, (real)0.0), (real)(1.0/2.2));
    } else if (cid == 9) {
        return pow(fmax(v, (real)0.0), (real)(1.0/2.4));
    } else if (cid == 10) {
        return pow(fmax(v, (real)0.0), (real)(1.0/2.6));
    }
    return v;
}

// ---------------- RGC（镜像 _rgc_d 与 _gc 的 gc_mode==0 分支） ----------------
__device__ __forceinline__ real rgc_d(real d, real t, real s) {
    if (d < t) return d;
    real nd = (d - t) / s;
    return t + s * nd / pow((real)1.0 + pow(nd, (real)RGC_PWR), (real)(1.0 / RGC_PWR));
}

// 旧的自研 shoulder（仅 COLORLAB_GC=roll 的退回路径用）
__device__ __forceinline__ real shoulder_v(real x, real k) {
    real knee = (real)0.9 - (real)0.12 * fmax((real)0.0, k);
    knee = fmax((real)0.35, fmin((real)0.95, knee));
    if (x <= knee) return x;
    real over = x - knee;
    real c = (real)1.0 / ((real)1.0 - knee);
    real comp = (over / (over + (real)1.0 / c)) * ((real)1.0 - knee);
    return fmin(knee + comp, (real)1.0);
}

__device__ __forceinline__ void gc3(real r, real g, real b, int gc_mode, real k, int use_roll,
                                    real tr, real tg, real tb, real sr, real sg, real sb,
                                    real* o) {
    if (gc_mode == 2) { o[0] = r; o[1] = g; o[2] = b; return; }   // 不缩限：显示器转换用
    if (gc_mode == 0) {
        real A = fmax(r, fmax(g, b));
        real Aa = fabs(A);
        if (Aa == (real)0.0) { o[0] = o[1] = o[2] = (real)0.0; return; }
        o[0] = A - rgc_d((A - r) / Aa, tr, sr) * Aa;
        o[1] = A - rgc_d((A - g) / Aa, tg, sg) * Aa;
        o[2] = A - rgc_d((A - b) / Aa, tb, sb) * Aa;
        return;
    }
    if (!use_roll) { o[0] = r; o[1] = g; o[2] = b; return; }
    real mx = fmax(r, fmax(g, b)), mn = fmin(r, fmin(g, b));
    if (mx <= (real)1.0 && mn >= (real)0.0) { o[0] = r; o[1] = g; o[2] = b; return; }
    real Y = (real)0.2126 * r + (real)0.7152 * g + (real)0.0722 * b;
    real Ys = shoulder_v(Y, k);
    real s_mx = (mx > (real)1.0) ? ((real)1.0 - Ys) / fmax(mx - Y, (real)1e-6) : (real)1.0;
    real s_mn = (mn < (real)0.0) ? Ys / fmax(Y - mn, (real)1e-6) : (real)1.0;
    real s = fmax((real)0.0, fmin((real)1.0, fmin(s_mx, s_mn)));
    real t = (real)1.0 - s;
    s = (real)1.0 - t * t * ((real)3 - (real)2 * t);
    o[0] = fmax(fmin(Ys + (r - Y) * s, (real)1.0), (real)0.0);
    o[1] = fmax(fmin(Ys + (g - Y) * s, (real)1.0), (real)0.0);
    o[2] = fmax(fmin(Ys + (b - Y) * s, (real)1.0), (real)0.0);
}

// ---------------- 四面体插值（镜像 _tetra） ----------------
__device__ __forceinline__ void tetra(real fr, real fg, real fb,
        const real* c000, const real* c100, const real* c010, const real* c110,
        const real* c001, const real* c101, const real* c011, const real* c111, real* o) {
    real w0, w1, w2, w3;
    const real *a, *b, *c;
    if (fr >= fg && fr >= fb) {
        if (fg >= fb) { w1 = fr - fg; w2 = fg - fb; w3 = fb; a = c100; b = c110; c = c111; }
        else          { w1 = fr - fb; w2 = fb - fg; w3 = fg; a = c100; b = c101; c = c111; }
        w0 = (real)1.0 - fr;
    } else if (fg >= fr && fg >= fb) {
        if (fr >= fb) { w1 = fg - fr; w2 = fr - fb; w3 = fb; a = c010; b = c110; c = c111; }
        else          { w1 = fg - fb; w2 = fb - fr; w3 = fr; a = c010; b = c011; c = c111; }
        w0 = (real)1.0 - fg;
    } else {
        if (fr >= fg) { w1 = fb - fr; w2 = fr - fg; w3 = fg; a = c001; b = c101; c = c111; }
        else          { w1 = fb - fg; w2 = fg - fr; w3 = fr; a = c001; b = c011; c = c111; }
        w0 = (real)1.0 - fb;
    }
    o[0] = w0 * c000[0] + w1 * a[0] + w2 * b[0] + w3 * c[0];
    o[1] = w0 * c000[1] + w1 * a[1] + w2 * b[1] + w3 * c[1];
    o[2] = w0 * c000[2] + w1 * a[2] + w2 * b[2] + w3 * c[2];
}

// 二维网格的公共索引
#define GIDX(H, W) \
    int j = blockIdx.x * blockDim.x + threadIdx.x; \
    int i = blockIdx.y * blockDim.y + threadIdx.y; \
    if (i >= (H) || j >= (W)) return; \
    long long p = ((long long)i * (W) + j) * 3

// ---------------- grade（逐位镜像 _grade_lin 的调用方式） ----------------
// grade_fast 的实际组合是：解码 sc → 线性域调色 → 再编码回 sc。
// 这里把两端曲线 ID 也作为参数，做到**任意曲线组合**都与 CPU 等价，
// 而不只是 sc='Linear'（RAW 主路径）那一种。Linear 时两端都是恒等，代价为零。
extern "C" __global__ void k_grade(const real* __restrict__ src, real* __restrict__ dst,
        int H, int W, real ev, real contrast, real sat, real kgr, real kgg, real kgb,
        int pre_id, int post_id) {
    GIDX(H, W);
    real r = eotf_id(src[p], pre_id), g = eotf_id(src[p+1], pre_id), b = eotf_id(src[p+2], pre_id);
    if (ev != (real)0.0) { real ke = pow((real)2.0, ev); r *= ke; g *= ke; b *= ke; }
    if (contrast != (real)0.0) {
        real kc = (real)1.0 + contrast;
        r = (r - (real)0.18) * kc + (real)0.18;
        g = (g - (real)0.18) * kc + (real)0.18;
        b = (b - (real)0.18) * kc + (real)0.18;
    }
    if (sat != (real)0.0) {
        real Y = (real)0.2126*r + (real)0.7152*g + (real)0.0722*b;
        real ks = (real)1.0 + sat;
        r = Y + (r - Y) * ks; g = Y + (g - Y) * ks; b = Y + (b - Y) * ks;
    }
    if (kgr != (real)1.0 || kgg != (real)1.0 || kgb != (real)1.0) { r *= kgr; g *= kgg; b *= kgb; }
    dst[p]   = oetf_id(fmax(r, (real)0.0), post_id);
    dst[p+1] = oetf_id(fmax(g, (real)0.0), post_id);
    dst[p+2] = oetf_id(fmax(b, (real)0.0), post_id);
}

// ---------------- pipe / CST（镜像 _pipe_go） ----------------
extern "C" __global__ void k_pipe(const real* __restrict__ src, real* __restrict__ dst,
        int H, int W, const real* __restrict__ M, const real* __restrict__ Mi,
        const real* __restrict__ Cat, int sc_id, int tc_id,
        int sg_display, int tg_display, int ootf_mode, real gam_sg, real gam_tg,
        real k, int use_roll, int gc_mode,
        real tr, real tg_, real tb, real sr, real sg_, real sb) {
    GIDX(H, W);
    real r = eotf_id(src[p], sc_id), g = eotf_id(src[p+1], sc_id), b = eotf_id(src[p+2], sc_id);
    if (ootf_mode == 1 || (ootf_mode == 3 && sg_display && !tg_display)) {
        real ig = (real)1.0 / gam_sg;
        r = pow(fmax(r, (real)0.0), ig); g = pow(fmax(g, (real)0.0), ig); b = pow(fmax(b, (real)0.0), ig);
    }
    real x = M[0]*r + M[1]*g + M[2]*b;
    real y = M[3]*r + M[4]*g + M[5]*b;
    real z = M[6]*r + M[7]*g + M[8]*b;
    if (!(Cat[0] == (real)1.0 && Cat[4] == (real)1.0 && Cat[8] == (real)1.0)) {
        real x2 = Cat[0]*x + Cat[1]*y + Cat[2]*z;
        real y2 = Cat[3]*x + Cat[4]*y + Cat[5]*z;
        real z2 = Cat[6]*x + Cat[7]*y + Cat[8]*z;
        x = x2; y = y2; z = z2;
    }
    r = Mi[0]*x + Mi[1]*y + Mi[2]*z;
    g = Mi[3]*x + Mi[4]*y + Mi[5]*z;
    b = Mi[6]*x + Mi[7]*y + Mi[8]*z;
    real o[3];
    gc3(r, g, b, gc_mode, k, use_roll, tr, tg_, tb, sr, sg_, sb, o);
    r = o[0]; g = o[1]; b = o[2];
    if (ootf_mode == 2 || (ootf_mode == 3 && tg_display)) {
        r = pow(fmax(r, (real)0.0), gam_tg); g = pow(fmax(g, (real)0.0), gam_tg); b = pow(fmax(b, (real)0.0), gam_tg);
    }
    dst[p]   = oetf_id(r, tc_id);
    dst[p+1] = oetf_id(g, tc_id);
    dst[p+2] = oetf_id(b, tc_id);
}

// ---------------- 3D LUT 四面体（镜像 _lut3d） ----------------
extern "C" __global__ void k_lut3d(const real* __restrict__ src, const real* __restrict__ L,
        real* __restrict__ dst, int H, int W, int N, real scale) {
    GIDX(H, W);
    real r = src[p], g = src[p+1], b = src[p+2];
    r = r < (real)0.0 ? (real)0.0 : (r > (real)1.0 ? (real)1.0 : r);
    g = g < (real)0.0 ? (real)0.0 : (g > (real)1.0 ? (real)1.0 : g);
    b = b < (real)0.0 ? (real)0.0 : (b > (real)1.0 ? (real)1.0 : b);
    real s = (real)(N - 1);
    real cr = r * s, cg = g * s, cb = b * s;
    int r0 = (int)cr, g0 = (int)cg, b0 = (int)cb;
    int n1 = N - 1;
    if (r0 > n1-1) r0 = n1-1;  if (g0 > n1-1) g0 = n1-1;  if (b0 > n1-1) b0 = n1-1;
    if (r0 < 0) r0 = 0;        if (g0 < 0) g0 = 0;        if (b0 < 0) b0 = 0;
    real fr = cr - r0, fg = cg - g0, fb = cb - b0;
    int r1 = r0+1, g1 = g0+1, b1 = b0+1;
    #define LIDX(R,G,B) ((long long)(((R)*N + (G))*N + (B)) * 3)
    const real* c000 = L + LIDX(r0,g0,b0); const real* c100 = L + LIDX(r1,g0,b0);
    const real* c010 = L + LIDX(r0,g1,b0); const real* c110 = L + LIDX(r1,g1,b0);
    const real* c001 = L + LIDX(r0,g0,b1); const real* c101 = L + LIDX(r1,g0,b1);
    const real* c011 = L + LIDX(r0,g1,b1); const real* c111 = L + LIDX(r1,g1,b1);
    #undef LIDX
    real o[3];
    tetra(fr, fg, fb, c000, c100, c010, c110, c001, c101, c011, c111, o);
    for (int q = 0; q < 3; q++) {
        real v = o[q] * scale;
        dst[p+q] = v < (real)0.0 ? (real)0.0 : (v > (real)1.0 ? (real)1.0 : v);
    }
}

// ---------------- 线性域面积平均降采样（镜像 colorlab_nb._box_down） ----------------
extern "C" __global__ void k_box_down(const real* __restrict__ src, real* __restrict__ dst,
        int H, int W, int OH, int OW) {
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= OW || oy >= OH) return;
    int y0 = oy * H / OH, y1 = (oy + 1) * H / OH;
    int x0 = ox * W / OW, x1 = (ox + 1) * W / OW;
    if (y1 <= y0) y1 = y0 + 1;
    if (x1 <= x0) x1 = x0 + 1;
    double s0 = 0.0, s1 = 0.0, s2 = 0.0;              // float64 累加，与 CPU 版一致
    int n = 0;
    for (int yy = y0; yy < y1; yy++) {
        long long base = (long long)yy * W * 3;
        for (int xx = x0; xx < x1; xx++) {
            long long q = base + (long long)xx * 3;
            s0 += (double)src[q]; s1 += (double)src[q+1]; s2 += (double)src[q+2];
            n++;
        }
    }
    double inv = 1.0 / (double)n;
    long long o = ((long long)oy * OW + ox) * 3;
    dst[o]   = (real)(s0 * inv);
    dst[o+1] = (real)(s1 * inv);
    dst[o+2] = (real)(s2 * inv);
}

// ---------------- 裁切 + 量化到 uint8（管线末端的 clip + *255 + 0.5） ----------------
extern "C" __global__ void k_to_u8(const real* __restrict__ src, unsigned char* __restrict__ dst,
        long long n) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    real v = src[idx];
    v = v < (real)0.0 ? (real)0.0 : (v > (real)1.0 ? (real)1.0 : v);
    dst[idx] = (unsigned char)(v * (real)255.0 + (real)0.5);
}

// ---------------- 线性 → sRGB 编码（镜像 _encode_srgb，供 8bit 显示代理用） ----------------
extern "C" __global__ void k_encode_srgb(const real* __restrict__ src, real* __restrict__ dst,
        long long n) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    real v = src[idx];
    dst[idx] = (v <= (real)0.0031308) ? (v * (real)12.92)
                                      : ((real)1.055 * pow(v, (real)(1.0/2.4)) - (real)0.055);
}
"""

_BLOCK = (32, 8)


class Backend:
    """一个具体精度（float32 / float64）的 CUDA 后端实例。"""

    def __init__(self, precision="float64"):
        if not available():
            raise RuntimeError(f"CUDA 不可用: {_LAST_ERR}")
        self.precision = precision
        self.real_t = cp.float64 if precision == "float64" else cp.float32
        self.np_real = np.float64 if precision == "float64" else np.float32
        src = _SRC_TMPL % {"REAL": "double" if precision == "float64" else "float"}
        self._k = {n: cp.RawKernel(src, n) for n in (
            "k_grade", "k_pipe", "k_lut3d", "k_box_down", "k_to_u8", "k_encode_srgb")}
        for k in self._k.values():
            k.compile()                      # 提前编译，避免首帧卡顿

    # ---- 工具 ----
    def _grid2d(self, h, w):
        return ((w + _BLOCK[0] - 1) // _BLOCK[0], (h + _BLOCK[1] - 1) // _BLOCK[1])

    def _grid1d(self, n):
        t = 256
        return ((n + t - 1) // t,), (t,)

    def to_gpu(self, a):
        return cp.asarray(np.ascontiguousarray(a, dtype=self.np_real))

    # ---- 单步 ----
    def grade(self, g, ev, contrast, sat, kg, pre_id=None, post_id=None):
        h, w = g.shape[0], g.shape[1]
        out = cp.empty_like(g)
        _lin = 11                                   # _CID_LINEAR
        self._k["k_grade"](self._grid2d(h, w), _BLOCK,
                           (g, out, np.int32(h), np.int32(w),
                            self.np_real(ev), self.np_real(contrast), self.np_real(sat),
                            self.np_real(kg[0]), self.np_real(kg[1]), self.np_real(kg[2]),
                            np.int32(_lin if pre_id is None else pre_id),
                            np.int32(_lin if post_id is None else post_id)))
        return out

    def pipe(self, g, M, Mi, Cat, sc_id, tc_id, sg_disp, tg_disp, ootf_mode,
             gam_sg, gam_tg, k, use_roll, gc_mode, rgc):
        h, w = g.shape[0], g.shape[1]
        out = cp.empty_like(g)
        Mg = self.to_gpu(np.ascontiguousarray(M, dtype=self.np_real).ravel())
        Mig = self.to_gpu(np.ascontiguousarray(Mi, dtype=self.np_real).ravel())
        Cg = self.to_gpu(np.ascontiguousarray(Cat, dtype=self.np_real).ravel())
        tr, tg_, tb, sr, sg_, sb = rgc
        self._k["k_pipe"](self._grid2d(h, w), _BLOCK,
                          (g, out, np.int32(h), np.int32(w), Mg, Mig, Cg,
                           np.int32(sc_id), np.int32(tc_id),
                           np.int32(1 if sg_disp else 0), np.int32(1 if tg_disp else 0),
                           np.int32(ootf_mode), self.np_real(gam_sg), self.np_real(gam_tg),
                           self.np_real(k), np.int32(1 if use_roll else 0), np.int32(gc_mode),
                           self.np_real(tr), self.np_real(tg_), self.np_real(tb),
                           self.np_real(sr), self.np_real(sg_), self.np_real(sb)))
        return out

    def lut3d(self, g, Lg, N, scale):
        h, w = g.shape[0], g.shape[1]
        out = cp.empty_like(g)
        self._k["k_lut3d"](self._grid2d(h, w), _BLOCK,
                           (g, Lg, out, np.int32(h), np.int32(w), np.int32(N),
                            self.np_real(scale)))
        return out

    def box_down(self, g, oh, ow):
        h, w = g.shape[0], g.shape[1]
        out = cp.empty((oh, ow, 3), dtype=self.real_t)
        self._k["k_box_down"](self._grid2d(oh, ow), _BLOCK,
                              (g, out, np.int32(h), np.int32(w), np.int32(oh), np.int32(ow)))
        return out

    def to_u8(self, g):
        n = int(g.size)
        out = cp.empty(n, dtype=cp.uint8)
        (grid,), (t,) = self._grid1d(n)
        self._k["k_to_u8"]((grid,), (t,), (g, out, np.int64(n)))
        return out.reshape(g.shape)

    def encode_srgb(self, g):
        n = int(g.size)
        out = cp.empty_like(g).ravel()
        (grid,), (t,) = self._grid1d(n)
        self._k["k_encode_srgb"]((grid,), (t,), (g.ravel(), out, np.int64(n)))
        return out.reshape(g.shape)
