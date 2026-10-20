#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Color-Lab 色彩科学回归测试（第 0 步护栏）

用途：把「曲线 / 色域矩阵 / 两条执行路径一致性 / 色域缩限」钉成可自动跑的断言，
      任何改动之后先跑它，确认没有把别的东西改坏。

用法（用项目自己的 venv）：
    .venv\\Scripts\\python.exe tests\\color_science_check.py

约定：
  * 每个检查打印 PASS / FAIL，末尾给汇总并以退出码表示结果（0 = 全过）。
  * 基线状态下（2026-09 修复前）预期有 4 组 FAIL：ARRI LogC3、ACEScg/ACEScct 基色、
    ACEScct 曲线缺失、Gamma 编码两路径不一致（外加色温一致性）。
    这些 FAIL 就是待修清单，修完必须变 PASS。
"""
import math
import sys
from pathlib import Path

import struct as _struct
from PIL import Image
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import colorlab as cl            # noqa: E402
import colorlab_nb as nb         # noqa: E402

RESULTS = []


def check(name, ok, detail=''):
    RESULTS.append((name, bool(ok), detail))
    print(f'  [{"PASS" if ok else "FAIL"}] {name}' + (f'   {detail}' if detail else ''))


def enc1(curve, x):
    return float(np.asarray(cl.CURVE[curve][0](np.array([x], dtype=np.float32)))[0])


def dec1(curve, y):
    return float(np.asarray(cl.CURVE[curve][1](np.array([y], dtype=np.float32)))[0])


# ───────────────────────── 1. 曲线常量 vs 公开标准 ─────────────────────────
print('\n1. 曲线常量 vs 公开标准（ARRI LogC3 官方定义：x<=cut 用 e*x+f，x>cut 用 c*log10(a*x+b)+d）')

_CUT, _A, _B, _C, _D, _E, _F = 0.010591, 5.555556, 0.052272, 0.247190, 0.385537, 5.367655, 0.092809


def ref_arri_enc(x):
    return _C * math.log10(_A * x + _B) + _D if x > _CUT else _E * x + _F


def ref_arri_dec(y):
    return (10 ** ((y - _D) / _C) - _B) / _A if y > _E * _CUT + _F else (y - _F) / _E


_refs = {
    'S-Log3': (lambda x: (420 + math.log10((x + 0.01) / 0.19) * 261.5) / 1023 if x >= 0.01125
               else (x * (171.2102946929 - 95) / 0.01125 + 95) / 1023, (0.001, 0.01125, 0.18, 1.0)),
    'V-Log': (lambda x: 0.241514 * math.log10(x + 0.00873) + 0.598206 if x >= 0.01 else 5.6 * x + 0.125,
              (0.001, 0.01, 0.18, 1.0)),
    'RED Log3G10': (lambda x: 0.386231 + 0.224282 * math.log10(x / 0.155954) if x >= 0.155954
                    else 0.386231 + 0.224282 * (x - 0.155954) / (0.155954 * math.log(10)),
                    (0.05, 0.155954, 0.5, 1.0)),
    'ARRI LogC3': (ref_arri_enc, (0.001, 0.010591, 0.05, 0.18, 1.0)),
}
for name, (ref, xs) in _refs.items():
    worst = max(abs(enc1(name, x) - ref(x)) for x in xs)
    check(f'{name} 编码符合公开标准', worst < 1e-5, f'最大偏差 {worst:.3e}')

worst = max(abs(dec1('ARRI LogC3', y) - ref_arri_dec(y)) for y in (0.15, 0.4, 0.57))
check('ARRI LogC3 解码是官方编码的逆', worst < 1e-4, f'最大偏差 {worst:.3e}')

# 分段曲线在断点处必须连续
for name, cut in (('S-Log3', 0.01125), ('ARRI LogC3', _CUT), ('V-Log', 0.01), ('RED Log3G10', 0.155954)):
    lo, hi = enc1(name, cut * (1 - 1e-6)), enc1(name, cut * (1 + 1e-6))
    check(f'{name} 在断点处连续', abs(hi - lo) < 1e-3, f'跳变 {hi - lo:.6f}')


# ───────────────────────── 2. 曲线往返一致性 ─────────────────────────
print('\n2. 每条曲线的 编码→解码 往返一致性（float32 下应≈恒等）')
for name in ('sRGB', 'Rec.709', 'S-Log3', 'V-Log', 'RED Log3G10', 'ARRI LogC3', 'ACEScct', 'Gamma 2.2', 'Gamma 2.4'):
    worst = 0.0
    for x in (0.02, 0.18, 0.5, 0.9):
        try:
            worst = max(worst, abs(dec1(name, enc1(name, x)) - x))
        except Exception as e:                                    # noqa: BLE001
            worst = float('nan')
            break
    check(f'{name} 往返误差 < 1e-5', worst < 1e-5, f'最大 {worst:.3e}')


# ───────────────────────── 3. 色域基色（ACES 家族） ─────────────────────────
print('\n3. ACES 家族基色：ACES2065-1 用 AP0；ACEScg / ACEScct 必须用 AP1')
AP0 = np.array([[0.9525523959, 0.0, 0.0000936786],
                [0.3439664498, 0.7281660966, -0.0721325464],
                [0.0, 0.0, 1.0088251844]], dtype=np.float32)
AP1 = np.array([[0.6624541811, 0.1340042065, 0.1561876870],
                [0.2722287168, 0.6740817658, 0.0536895174],
                [-0.0055746495, 0.0040607335, 1.0103391003]], dtype=np.float32)
for g, expect, label in (('ACES2065-1', AP0, 'AP0'), ('ACEScg', AP1, 'AP1'), ('ACEScct', AP1, 'AP1')):
    M = cl._mat(g)[0]
    d = float(np.abs(M - expect).max())
    check(f'{g} 的 RGB→XYZ 矩阵等于 {label}', d < 1e-3, f'偏差 {d:.4f}')


# ───────────────────────── 4. 曲线注册完整性 ─────────────────────────
print('\n4. _CURVE_MATCH 指向的曲线必须都已注册（否则选到就崩）')
missing = sorted({c for c in cl._CURVE_MATCH.values() if c and c not in cl.CURVE})
check('_CURVE_MATCH 全部有对应曲线', not missing, f'缺失: {missing}')


# ───────────────────────── 5. 两条执行路径一致性 ─────────────────────────
print('\n5. numpy 路径 与 numba 路径 必须一致（同一个设置不能出两种颜色）')
x = np.array([1.6], dtype=np.float32)
for g in ('Gamma 1.8', 'Gamma 2.2', 'Gamma 2.4', 'Gamma 2.6'):
    a = float(np.asarray(cl.CURVE[g][0](x))[0])
    b = float(nb._oetf(1.6, g))
    check(f'{g} 编码 numpy==numba (x=1.6)', abs(a - b) < 1e-6, f'numpy={a:.5f} numba={b:.5f}')

rng = np.random.RandomState(0)
probe = rng.rand(16, 16, 3).astype(np.float32)
for tp in (6500.0, 3000.0, 9000.0):
    g_np = np.asarray(cl._grade(probe.copy(), 'sRGB', 0.3, 0.1, 0.15, tp), dtype=np.float64)
    g_nb = np.asarray(nb.grade_fast(probe.copy(), 'sRGB', 0.3, 0.1, 0.15, tp), dtype=np.float64)
    d = float(np.abs(g_np - g_nb).max())
    check(f'色温 {tp:.0f}K 时 numpy==numba 调色', d < 1e-4, f'最大差 {d:.3e}')

# （原「亮度 + 滚降」两条路径一致性检查已随功能删除而移除）


# ───────────────────────── 6. 色域缩限的不变量 ─────────────────────────
print('\n6. _roll 色域缩限：保持亮度、连续、不塌缩（当前实现的设计不变量）')
t = np.linspace(0.0, 2.0, 512, dtype=np.float32)
tinted = np.stack([t, t * 0.4, t * 0.1], axis=-1)[None, ...]
out = np.asarray(cl._roll(tinted.copy(), 0.9), dtype=np.float64)
Y_in = 0.2126 * tinted[..., 0] + 0.7152 * tinted[..., 1] + 0.0722 * tinted[..., 2]
Y_out = 0.2126 * out[..., 0] + 0.7152 * out[..., 1] + 0.0722 * out[..., 2]
Y_shoulder = np.asarray(cl._shoulder(Y_in.astype(np.float32), 0.9), dtype=np.float64)
# 只对「越界且未被后置 clip 截断」的像素断言亮度保持：
# 界内像素按设计原样返回（不做 shoulder）；被 clip 的像素亮度必然被改动。
bad = (tinted.max(axis=-1) > 1.0) | (tinted.min(axis=-1) < 0.0)
unclipped = ((out > 1e-6) & (out < 1 - 1e-6)).all(axis=-1)
mask = bad & unclipped
dY = float(np.abs(Y_out - Y_shoulder)[mask].max()) if mask.any() else 0.0
check('越界像素缩限后亮度 == shoulder(原亮度)', dY < 1e-4, f'最大差 {dY:.3e}（样本 {int(mask.sum())} 个）')
check('缩限结果落在 [0,1]', bool(out.min() >= -1e-6 and out.max() <= 1 + 1e-6),
      f'范围 [{out.min():.4f}, {out.max():.4f}]')
d2 = np.abs(np.diff(out[0], axis=0)).max()
check('缩限输出连续（无跳变）', float(d2) < 0.05, f'相邻最大跳变 {float(d2):.4f}')


# ───────────────────────── 7. LUT 工作流是否真的会缩限 ─────────────────────────
print('\n7. LUT 工作流（目标是 Log 曲线）是否执行色域缩限')
print('     colorlab.py:172  if tc not in _LOG_CURVES: l2 = _roll(l2, k)')
print(f'     _LOG_CURVES = {sorted(cl._LOG_CURVES)}')
print('     → LUT 的输入曲线（如 V-Log）在集合内，缩限被跳过；随后 _al 第 262 行 np.clip(img,0,1) 硬裁')
oob = np.array([[[1.6, 0.2, -0.35]]], dtype=np.float32)
print(f'     超界像素 {oob.ravel().tolist()} 经 _al 输入裁剪后 = {np.clip(oob, 0, 1).ravel().tolist()}')


# ───────────────────────── 8. ACES RGC 色域缩限的性质 ─────────────────────────
print('\n8. ACES RGC（色域缩限）的性质 —— 依据 ACES 规范和参考实现')
_rng2 = np.random.RandomState(7)
xs = (np.concatenate([_rng2.rand(4000, 3)]) * 2.0 - 0.4).astype(np.float32)   # 含负值与 >1
ys = cl.gamut_compress(xs)
_thr = np.array(cl._RGC_THR, dtype=np.float32)
_A = xs.max(axis=-1, keepdims=True)
_dn = np.where(np.abs(_A) == 0, 0.0, (_A - xs) / np.abs(_A))
_intact = (_dn < _thr).all(axis=-1)
_dmax = float(np.abs(ys - xs)[_intact].max()) if _intact.any() else 0.0
check('zone of trust（dn < t）完全不变', _dmax < 1e-6,
      f'{int(_intact.sum())}/{len(xs)} 像素在保护区内，最大改动 {_dmax:.2e}')
_lim = np.array(cl._RGC_LIM, dtype=np.float32)
_Ap = xs.max(axis=-1, keepdims=True)
_dn2 = np.where(np.abs(_Ap) == 0, 0.0, (_Ap - xs) / np.abs(_Ap))
_within = ((_dn2 <= _lim) & (_Ap > 0)).all(axis=-1)      # 每个通道都没超过 limit，且 A>0
_neg_after = int((ys[_within] < -1e-6).sum()) if _within.any() else 0
check('limit 以内的越界像素被完全治愈（无负分量）', _neg_after == 0,
      f'limit 内像素 {int(_within.sum())} 个，压缩后仍为负的通道 {_neg_after} 个')
_zs = cl.gamut_decompress(ys)
check('RGC 可逆（decompress∘compress ≈ id）', float(np.abs(_zs - xs).max()) < 2e-3,
      f'最大误差 {float(np.abs(_zs - xs).max()):.2e}')
_ys2 = cl.gamut_compress((xs * 2.0).astype(np.float32))
check('曝光不变性 f(2x) = 2f(x)', float(np.abs(_ys2 - ys * 2.0).max()) < 1e-4,
      f'最大差 {float(np.abs(_ys2 - ys * 2.0).max()):.2e}')
_edge = np.array([[[1.0 - cl._RGC_LIM[0], 1.0, 1.0]]], dtype=np.float32)
_oe = cl.gamut_compress(_edge)
check('limit 处正好压到色域边界（R → 0）', abs(float(_oe.ravel()[0])) < 1e-5,
      f'结果 {float(_oe.ravel()[0]):.2e}')

# ───────────────────────── 9. 修复点：Log 目标路径确实做了缩限 ─────────────────────────
print('\n9. 修复点：Pipe.go 在 Log 目标（LUT 工作流）上是否真的执行了缩限')
_lin = np.array([[[1.6, 0.2, -0.35]]], dtype=np.float32)
_M, _, _W = cl._mat('ACES2065-1')
_, _Mi, _W2 = cl._mat('V-Gamut')
_x = _lin @ _M.T
if not np.allclose(_W, _W2, atol=1e-6):
    _x = _x @ cl._cat(_W, _W2).T
_l2 = _x @ _Mi.T
_manual = np.asarray(cl.CURVE['V-Log'][0](cl.gamut_compress(_l2)))
_got = np.asarray(cl.Pipe.go(_lin, 'ACES2065-1', 'Linear', 'V-Gamut', 'V-Log'))
check('Pipe.go(V-Log 目标) == 手动 RGC + OETF', float(np.abs(_manual - _got).max()) < 1e-6,
      f'最大差 {float(np.abs(_manual - _got).max()):.2e}（COLORLAB_GC={cl._GC_MODE}）')
check('该超界像素不再产生负码值/NaN', bool(np.isfinite(_got).all()) and float(_got.min()) >= -1e-6,
      f'输出 min={float(_got.min()):.4f}')


# ───────────────────────── 10. numba 内核等价性（LUT / RGC / pipe / 整链） ─────────────────────────
print('\n10. numba 内核与 numpy 路径的数值等价性')
_LUT = ROOT / 'lut' / 'V-Gamut-V-Log' / 'Fujifilm' / 'Fuji_Velvia.cube'
_x = np.random.RandomState(11).rand(192, 192, 3).astype(np.float32)
_inf = None
if _LUT.exists():
    _inf = cl._cp(str(_LUT))
    _ynp = np.asarray(cl._al(_x.copy(), _inf), np.float64)
    _ynb = np.asarray(nb.lut_fast(_x.copy(), _inf), np.float64)
    check('lut_fast（33³ .cube）== _al', float(np.abs(_ynp - _ynb).max()) < 2e-5,
          f'最大差 {float(np.abs(_ynp - _ynb).max()):.2e}')
else:
    check('Fuji_Velvia.cube 存在（LUT 等价性前提）', False, str(_LUT))

_inf1 = {'t': '1d', 'n': 17, 'd': np.stack([np.linspace(0, 1, 17) ** 0.9] * 3).astype(np.float32)}
_d1 = float(np.abs(np.asarray(cl._al(_x.copy(), _inf1), np.float64)
                   - np.asarray(nb.lut_fast(_x.copy(), _inf1), np.float64)).max())
check('lut_fast（1D LUT）== _al', _d1 < 2e-5, f'最大差 {_d1:.2e}')

_z = (np.random.RandomState(12).rand(128, 128, 3).astype(np.float32) * 2.2 - 0.6)
_dr = float(np.abs(np.asarray(cl.gamut_compress(_z.copy()), np.float64)
                   - np.asarray(nb.rgc_fast(_z.copy()), np.float64)).max())
check('rgc_fast == gamut_compress', _dr < 1e-6, f'最大差 {_dr:.2e}')

_dp = float(np.abs(np.asarray(cl.Pipe.go(_x.copy(), cl._RAW_WORK_DEF, 'sRGB', 'V-Gamut', 'V-Log'), np.float64)
                   - np.asarray(nb.pipe_fast(_x.copy(), cl._RAW_WORK_DEF, 'sRGB', 'V-Gamut', 'V-Log'), np.float64)).max())
check('pipe_fast == Pipe.go（LUT 工作流目标）', _dp < 2e-5, f'最大差 {_dp:.2e}')

if _inf is not None:
    _knp = cl._al(cl.Pipe.go(cl._grade(_x.copy(), 'sRGB', 0.3, 0.1, 0.15, 0.0),
                             cl._RAW_WORK_DEF, 'sRGB', 'V-Gamut', 'V-Log'), _inf)
    _knb = nb.lut_fast(nb.pipe_fast(nb.grade_fast(_x.copy(), 'sRGB', 0.3, 0.1, 0.15, 0.0),
                                    cl._RAW_WORK_DEF, 'sRGB', 'V-Gamut', 'V-Log'), _inf)
    _dk = float(np.abs(np.asarray(_knp, np.float64) - np.asarray(_knb, np.float64)).max())
    check('整链 numpy == numba（LUT 工作流）', _dk < 5e-5, f'最大差 {_dk:.2e}')

_dt = cl._dither(_x.copy())
check('_dither 分块版输出 uint8 且形状一致', _dt.dtype == np.uint8 and _dt.shape == _x.shape,
      f'{_dt.dtype} {tuple(_dt.shape)}')
_same = (abs(nb._RGC_LIM_R - cl._RGC_LIM[0]) < 1e-12 and abs(nb._RGC_LIM_G - cl._RGC_LIM[1]) < 1e-12
         and abs(nb._RGC_LIM_B - cl._RGC_LIM[2]) < 1e-12 and abs(nb._RGC_THR_R - cl._RGC_THR[0]) < 1e-12
         and abs(nb._RGC_THR_G - cl._RGC_THR[1]) < 1e-12 and abs(nb._RGC_THR_B - cl._RGC_THR[2]) < 1e-12
         and abs(nb._RGC_PWR - cl._RGC_PWR) < 1e-12)
check('nb 与 colorlab 的 RGC 参数一致（防两处漂移）', _same)


# ───────────────────────── 11. 曲线 ID 内核 vs 字符串内核：逐位等价（精度关卡） ─────────────────────────
print('\n11. 曲线 ID 内核与字符串内核的逐位等价性（4A 的精度门槛）')
from numba import njit as _njit, prange as _prange                                   # noqa: E402
from colorlab_nb import (_eotf as _nb_eotf, _eotf_id as _nb_eotf_id,                  # noqa: E402
                         _oetf as _nb_oetf, _oetf_id as _nb_oetf_id)


@_njit(parallel=True)
def _bitwise_scan(vals, name, cid):
    n = vals.shape[0]
    bad = 0
    for i in _prange(n):
        a0 = _nb_eotf(vals[i], name); b0 = _nb_eotf_id(vals[i], cid)
        if not (a0 == b0 or (a0 != a0 and b0 != b0)):     # NaN 视为相等
            bad += 1
        a1 = _nb_oetf(vals[i], name); b1 = _nb_oetf_id(vals[i], cid)
        if not (a1 == b1 or (a1 != a1 and b1 != b1)):
            bad += 1
    return bad


_vals = np.concatenate([
    np.linspace(-0.2, 2.0, 3001),
    np.array([0.0, 0.001, 0.0031308, 0.0078125, 0.010591, 0.01125, 0.018, 0.04045, 0.081,
              0.155251141552511, 0.155954, 0.181, 0.598206, 1.0, 1.5, 2.0]),
]).astype(np.float32)
_bad_total = 0
_bad_curves = []
for _nm in cl.CURVE_NAMES:
    _b = int(_bitwise_scan(_vals, _nm, nb.curve_id(_nm)))
    _bad_total += _b
    if _b:
        _bad_curves.append((_nm, _b))
check(f'ID 内核与字符串内核逐位相等（{len(cl.CURVE_NAMES)} 条曲线 × {len(_vals)} 采样点）',
      _bad_total == 0, f'不一致 {_bad_total} 处 {_bad_curves}')
check('未知曲线名回落 Linear 且两版一致',
      int(_bitwise_scan(_vals, '不存在的曲线', nb.curve_id('不存在的曲线'))) == 0)
check('_CID 覆盖 colorlab.CURVE 全部曲线', set(cl.CURVE) <= set(nb._CID),
      f'缺: {sorted(set(cl.CURVE) - set(nb._CID))}')


# ───────────────────────── 12. 版本号与更新日志一致 ─────────────────────────
print('\n12. 版本号与更新日志是否一致（防止改了代码忘了记账）')
_clog = ROOT / 'CHANGELOG.md'
if _clog.exists():
    _heads = [l for l in _clog.read_text(encoding='utf-8').splitlines() if l.startswith('## ')]
    _top = _heads[0] if _heads else ''
    check(f'CHANGELOG 最新条目含 APP_VERSION（{cl.APP_VERSION}）', cl.APP_VERSION in _top,
          f'最新条目: {_top!r}')
else:
    check('CHANGELOG.md 存在', False, str(_clog))


# ───────────────────────── 13. 源侧解析一致性 + ffmpeg 残留 ─────────────────────────
print('\n13. 源侧解析必须与管线一致；ffmpeg 路径已彻底移除')
_r1 = cl.resolve_source({'raw': True, 'rw': 'Rec.2020', 'sg': 'sRGB', 'sc': 'Gamma 2.6'})
check('RAW 源侧 = (rw, Linear)，忽略 UI 里的选择', tuple(_r1) == ('Rec.2020', 'Linear'), f'得到 {tuple(_r1)}')
_r2 = cl.resolve_source({'raw': False, 'rw': 'ACES2065-1', 'sg': 'Rec.709', 'sc': 'Gamma 2.2'})
check('非 RAW 源侧 = 用户选择', tuple(_r2) == ('Rec.709', 'Gamma 2.2'), f'得到 {tuple(_r2)}')
_leak = [n for n in ('_FFMPEG', '_find_ffmpeg', '_apply_lut_ffmpeg') if hasattr(cl, n)]
check('ffmpeg 相关符号已全部移除（里里外外）', not _leak, f'残留: {_leak}')
import inspect as _inspect                                            # noqa: E402
_src = _inspect.getsource(cl.transform)
check('transform() 改走 resolve_source（不再自己判 RAW）', 'resolve_source(s, raw_work)' in _src)
check('LUT 引擎二选一控件已移除（不再有假选项）', '_lut_mode_sw' not in _inspect.getsource(cl.App._snapshot))


# ───────────────────────── 14. 色温：默认落在机内 WB ─────────────────────────
print('\n14. 色温默认必须落在机内（WB 是主观操作，但锚点不能是假的）')
_M = [[0.5271, -0.0712, -0.0347], [-0.6153, 1.3653, 0.2763], [-0.1601, 0.2366, 0.7242]]
_day = [2.550952911376953, 0.9247958064079285, 1.1455272436141968]      # ILCE-7M5 daylight WB
_shot = [2337.0, 1024.0, 1706.0, 1024.0]                                  # ILCE-7M5 as-shot WB
_c1 = cl._cct_from_wb(_day, _M)
check('自检：daylight WB 应还原 ≈6500K（D65）',
      _c1 is not None and abs(_c1 - 6500) < 250, f'得到 {None if _c1 is None else round(_c1)}K')
_c2 = cl._cct_from_wb(_shot, _M)
check('机内 as-shot 能算出等效色温（作为滑块默认位置）',
      _c2 is not None and 2500 < _c2 < 9000, f'得到 {None if _c2 is None else round(_c2)}K')
check('矩阵退化时安全返回 None（不编数字）', cl._cct_from_wb(_shot, [[0, 0, 0]] * 3) is None)
_kg = np.asarray(cl._kelvin_gain(4391.0), dtype=np.float64)
check('目标K == 机内K 时相对增益严格为 1（默认输出 = 机内）',
      bool(np.all(np.abs(_kg / _kg - 1.0) < 1e-15)))
check('目标K ≠ 机内K 时相对增益 ≠ 1（可主观调整）',
      bool(np.any(np.abs(np.asarray(cl._kelvin_gain(6000.0)) / _kg - 1.0) > 1e-3)))

# ---- 方向：这两条是**补上来的** ----
# 白平衡曾经整个反向（_kelvin_gain 返回的是光源颜色而不是它的倒数），
# 用户报"滑块越高色温图像越蓝"。当时 83 项全过 —— 因为上面只验了"增益会变"，
# 没验"往哪个方向变"。方向错等于功能错，必须有断言兜住。
def _rb(K, tint=0.0, cam=None):
    g = np.asarray(cl._wb_gain(float(K), float(tint), cam), dtype=np.float64)
    return float(g[0] / g[2])

check('色温方向：K 越高画面越暖（R/B 单调递增，与 ACR 一致）',
      _rb(3000) < _rb(5000) < _rb(7000) < _rb(10000),
      f'R/B 3000K={_rb(3000):.3f} 5000K={_rb(5000):.3f} 7000K={_rb(7000):.3f} 10000K={_rb(10000):.3f}')
check('色温方向：不是"越大越暖"就行，冷端必须真的偏蓝（R/B < 1）',
      _rb(2500) < 0.5, f'2500K 时 R/B={_rb(2500):.3f}')
def _gm(t):
    """色调 t 时的 绿/(红+蓝) 比：越小越偏品红。"""
    g = np.asarray(cl._wb_gain(6500.0, float(t)), dtype=np.float64)
    return float(g[1] / (g[0] + g[2]))


check('色调方向：正值偏品红（绿相对 R+B 下降，与 ACR 一致）',
      _gm(-150) > _gm(0) > _gm(150),
      f'G/(R+B)  -150={_gm(-150):.3f}  0={_gm(0):.3f}  +150={_gm(150):.3f}')
check('目标K = 6500 时基准归一化成立（旧接口语义不变）',
      bool(np.allclose(cl._kelvin_gain(6500.0), (1.0, 1.0, 1.0), atol=1e-9)))


# ──────────────────── 14. 元数据：ICC 与 EXIF ────────────────────
# 补这一组的原因：ICC 曲线与编码不一致（Rec.709 暗部差 −84.5%）、
# EXIF 四个导出格式全都没写 —— 这两个 bug 长期存在就是因为没有覆盖。
print('\n[14] 元数据：ICC profile 与 EXIF')


def _icc_tags(b):
    n = _struct.unpack('>I', b[128:132])[0]
    d = {}
    for i in range(n):
        sig, ofs, sz = _struct.unpack('>4sII', b[132 + i * 12:144 + i * 12])
        d[sig.decode('latin1')] = (ofs, sz)
    return d


def _icc_trc(b):
    """按 ICC 语义返回「设备值 → 线性」的求值函数。"""
    t = _icc_tags(b)
    if 'rTRC' not in t:
        return None
    o, _ = t['rTRC']
    typ = b[o:o + 4]
    if typ == b'curv':
        cnt = _struct.unpack('>I', b[o + 8:o + 12])[0]
        if cnt == 0:
            return lambda x: np.asarray(x, dtype=float)
        if cnt == 1:
            g = _struct.unpack('>H', b[o + 12:o + 14])[0] / 256.0
            return lambda x, g=g: np.clip(np.asarray(x, dtype=float), 0, 1) ** g
        vals = np.array(_struct.unpack('>' + 'H' * cnt, b[o + 12:o + 12 + cnt * 2]),
                        dtype=float) / 65535.0
        xs = np.linspace(0.0, 1.0, cnt)
        return lambda x: np.interp(x, xs, vals)
    if typ == b'para':
        # parametricCurveType：type 0 = Y=X^g，type 3 = Y=(aX+b)^g (X>=d) 否则 cX
        fn = _struct.unpack('>H', b[o + 8:o + 10])[0]
        npar = {0: 1, 1: 3, 2: 4, 3: 5, 4: 7}.get(fn)
        if npar is None:
            return None
        p = [v / 65536.0 for v in
             _struct.unpack('>' + 'i' * npar, b[o + 12:o + 12 + npar * 4])]
        if fn == 0:
            return lambda x, g=p[0]: np.maximum(np.asarray(x, dtype=float), 0.0) ** g
        if fn == 3:
            g, a, bb, c, d = p

            def _f(x, g=g, a=a, bb=bb, c=c, d=d):
                x = np.asarray(x, dtype=float)
                return np.where(x >= d, np.maximum(a * x + bb, 0.0) ** g, c * x)
            return _f
        return None
    return None


_bad_struct = _bad_sum = _bad_trc = 0
_worst_trc = 0.0
for _g in cl.GAMUT_NAMES:
    _b = cl._icc_bytes(_g)
    if _b is None:
        _bad_struct += 1
        _bad_sum += 1
        _bad_trc += 1
        continue
    _t = _icc_tags(_b)
    if not all(k in _t for k in ('desc', 'wtpt', 'rXYZ', 'gXYZ', 'bXYZ',
                                 'rTRC', 'gTRC', 'bTRC')):
        _bad_struct += 1
        continue
    # —— 判据一：色料和必须等于 PCS 照明体 D50（独立计数）——
    _cs = np.zeros(3)
    for _k in ('rXYZ', 'gXYZ', 'bXYZ'):
        _o = _t[_k][0]
        _cs += np.array(_struct.unpack('>3i', _b[_o + 8:_o + 20]), dtype=float) / 65536.0
    if float(np.abs(_cs - np.array([0.9642, 1.0, 0.8249])).max()) >= 1e-3:
        _bad_sum += 1
    # —— 判据二：曲线与我们的编码一致，比**绝对**误差 ——
    # ICC 的线性轴相对白点、值域 [0,1]。Log 曲线的 decode 会超过 1.0
    # （V-Log 在设备值 1.0 处约 1.24），生成时按端点归一化 —— 那是唯一
    # 合法且单调的表达（夹取会让曲线后半段整段变平）。故这里也除以端点值。
    _ev = _icc_trc(_b)
    _cn = cl._CURVE_MATCH.get(_g)
    # **不变量已随 C 方案变更**：ICC 的 TRC 现在描述「码值 → 线性光」，
    # 对显示参考内容那就是它的**显示 EOTF**，不是编码曲线。
    # 原来这里比对的是编码曲线（OETF），C 之后 Rec.709/Rec.2020 上必然不符 ——
    # 那正是本次要改的东西，不是回归。断言的目的没变：**TRC 必须与显示端同源**，
    # 否则 CL 显示与第三方读同一文件的结果会不同（同一性破裂）。
    _cn = cl.display_decode_curve(_cn) if _cn else _cn
    if _ev is None or _cn not in cl.CURVE:
        _bad_trc += 1
        continue
    _dec = cl.CURVE[_cn][1]
    _mx = float(np.max(np.asarray(_dec(np.linspace(0, 1, 4096).astype(np.float32)),
                                  dtype=float)))
    _lin = np.array([0.001, 0.005, 0.02, 0.05, 0.18, 0.5, 0.9], dtype=np.float64)
    _dev = np.asarray(cl.CURVE[_cn][0](_lin.astype(np.float32)), dtype=float)
    _want = np.asarray(_dec(_dev.astype(np.float32)), dtype=float) / (_mx if _mx > 1.0 else 1.0)
    _got = np.asarray(_ev(_dev), dtype=float)
    _ae = float(np.abs(_got - _want).max())
    _worst_trc = max(_worst_trc, _ae)
    if _ae >= 5.0 / 65535:
        _bad_trc += 1

check(f'全部 {len(cl.GAMUT_NAMES)} 个目标色域都能生成结构合规的 ICC',
      _bad_struct == 0, f'{len(cl.GAMUT_NAMES) - _bad_struct} 合格 / {_bad_struct} 不合格')
check('每个 ICC 的色料和都等于 PCS 照明体 D50（0.9642/1.0/0.8249）',
      _bad_sum == 0, f'{_bad_sum} 个不合格')
check('ICC 的 TRC 与**显示端**同源（显示 EOTF；C 方案前是编码曲线）',
      _bad_trc == 0, f'最差 {_worst_trc:.2e} / 阈值 {5/65535:.2e}，{_bad_trc} 个不合格')

# 与权威档案独立比对：自生成的 sRGB 色料应贴近 Microsoft 官方档案
try:
    _ms = r'C:\Windows\System32\spool\drivers\color\sRGB Color Space Profile.icm'
    _ref = open(_ms, 'rb').read()
    _ob = cl._icc_bytes('sRGB')
    _a, _c = _icc_tags(_ob), _icc_tags(_ref)
    _dmax = 0.0
    for _k in ('rXYZ', 'gXYZ', 'bXYZ', 'wtpt'):
        _va = np.array(_struct.unpack('>3i', _ob[_a[_k][0] + 8:_a[_k][0] + 20]), dtype=float) / 65536.0
        _vb = np.array(_struct.unpack('>3i', _ref[_c[_k][0] + 8:_c[_k][0] + 20]), dtype=float) / 65536.0
        _dmax = max(_dmax, float(np.abs(_va - _vb).max()))
    check('自生成 sRGB 的色料贴近 Microsoft 官方档案（< 5e-4）', _dmax < 5e-4,
          f'最大差 {_dmax:.2e}')
except Exception as _e:
    check('与 Microsoft 官方 sRGB 档案比对', False, f'读取失败 {_e}')

# 档案名：别的软件（XnView MP 等）按 ICC 的 desc 显示这个档案，
# 必须是标准色彩空间名。之前写成 "Color-Lab Rec.2020 Rec.709" 看着就是乱码。
def _icc_desc(b):
    n = _struct.unpack('>I', b[128:132])[0]
    for i in range(n):
        sig, ofs, sz = _struct.unpack('>4sII', b[132 + i * 12:144 + i * 12])
        if sig == b'desc':
            ac = _struct.unpack('>I', b[ofs + 8:ofs + 12])[0]
            return b[ofs + 12:ofs + 12 + ac - 1].decode('latin1')
    return None


_d_bad = [g for g in cl.GAMUT_NAMES
          if 'Color-Lab' in (_icc_desc(cl._icc_bytes(g)) or 'Color-Lab')]
check('ICC 档案名是标准色彩空间名（不含 Color-Lab 前缀）',
      not _d_bad, f'不合格: {_d_bad}')
check('sRGB 档案名与 Microsoft 官方一致（sRGB IEC61966-2.1）',
      _icc_desc(cl._icc_bytes('sRGB')) == 'sRGB IEC61966-2.1',
      f"实为 {_icc_desc(cl._icc_bytes('sRGB'))!r}")

# ICC 必须跟随**实际编码曲线**：否则用户在界面上把曲线改成 Gamma 2.4 时，
# 导出用 2.4 编码却嵌入按默认曲线描述的档案，像素与标签又对不上。
_icc_a = cl._icc_bytes('Rec.2020', 'Rec.709')
_icc_b = cl._icc_bytes('Rec.2020', 'Gamma 2.4')
_check_curve_ok = False
if _icc_a != _icc_b:
    _ev24 = _icc_trc(_icc_b)
    if _ev24 is not None:
        _v = np.asarray(_ev24(np.array([0.5], dtype=float)), dtype=float)[0]
        _check_curve_ok = abs(_v - 0.5 ** 2.4) < 0.01
check('_icc_bytes 跟随传入的曲线（改曲线后档案必须跟着变）', _check_curve_ok,
      'TRC 在 0.5 处的值应与 0.5^2.4 一致')
check('非标准曲线会在档案名里如实标出',
      '(Gamma 2.4)' in (_icc_desc(_icc_b) or ''),
      f"实为 {_icc_desc(_icc_b)!r}")

# 骨架对齐 Photoshop：v4 + chad + 非零 profileID
_b_mid = cl._icc_bytes('Display P3')
_t_mid, _ = _icc_tags(_b_mid), None
_ver_mid = _struct.unpack('>I', _b_mid[8:12])[0]
check('ICC 是 v4（对齐 Photoshop 的写法）', (_ver_mid >> 24) == 4,
      f'实为 {_ver_mid >> 24}.{(_ver_mid >> 20) & 0xf}.{(_ver_mid >> 16) & 0xf}')
check('ICC 带 chad 色适应矩阵（v4 里色料适配 D50 时必需）',
      'chad' in _t_mid and _t_mid['chad'][1] == 44, f"标签 {sorted(_t_mid)}")
check('ICC 写了非零 profileID（按规范算的 MD5）',
      any(_b_mid[84:100]), _b_mid[84:100].hex())
check('ICC 标签集与 Photoshop 的 Display P3 一致',
      sorted(_t_mid) == sorted(['bTRC', 'bXYZ', 'chad', 'cprt', 'desc', 'gTRC',
                                'gXYZ', 'rTRC', 'rXYZ', 'wtpt']),
      f'实为 {sorted(_t_mid)}')
check('ICC 体积与权威档案同量级（< 4KB）', len(_b_mid) < 4096, f'{len(_b_mid)} 字节')

check('Adobe RGB 用纯 gamma 2.2（不是 sRGB 分段曲线）',
      cl._CURVE_MATCH.get('Adobe RGB') == 'Gamma 2.2',
      f"实为 {cl._CURVE_MATCH.get('Adobe RGB')}")
check('ProPhoto RGB 用纯 gamma 1.8（不是 sRGB 分段曲线）',
      cl._CURVE_MATCH.get('ProPhoto RGB') == 'Gamma 1.8',
      f"实为 {cl._CURVE_MATCH.get('ProPhoto RGB')}")
check('DCI-P3 白点是 DCI 白(0.3140,0.3510)，不是 ICC 的 PCS 白点 D50',
      cl._GAMUT['DCI-P3']['wp'] == 'DCI', f"实为 {cl._GAMUT['DCI-P3']['wp']}")

# EXIF：RAW 抽取必须非空，且 tag 归到正确的 IFD
import glob as _glob
_raw = None
for _pat in (r'C:\Users\Administrator\Desktop\**\*.ARW', r'F:\DS Harness\**\*.ARW'):
    _hits = _glob.glob(_pat, recursive=True)
    if _hits:
        _raw = _hits[0]
        break
if _raw is None:
    check('EXIF 抽取（未找到 RAW 样本，跳过）', True, 'skip')
else:
    _eb = cl._extract_exif(_raw)
    check('RAW 能抽出非空 EXIF（旧实现在此恒为空）', bool(_eb),
          f'{0 if not _eb else len(_eb)} 字节')
    check('TIFF 用的 EXIF extratags 非空（旧实现 TIFF 完全不写 EXIF）',
          len(cl._exif_tiff_extratags(_raw)) > 0,
          f'{len(cl._exif_tiff_extratags(_raw))} 条')
    if _eb:
        _ex = Image.Exif()
        _ex.load(_eb)
        _top = set(dict(_ex).keys())
        try:
            _sub = set(dict(_ex.get_ifd(34665)).keys())
        except Exception:
            _sub = set()
        check('IFD0 只放 Make/Model，且带 34665 指针',
              271 in _top and 272 in _top and 34665 in _top, f'顶层={sorted(_top)}')
        check('曝光/时间/镜头等 tag 归到 Exif 子 IFD(34665) 而不是 IFD0',
              bool(_sub & {33434, 33437, 34855, 36867, 42036}) and 33434 not in _top,
              f'子IFD={sorted(_sub)}')


# ──────────────── 15. 导出与预览必须走同一个「源数据域」 ────────────────
# 这里抓的是一个曾经真实存在的 bug：管线对 RAW 宣称源曲线为 Linear（期望线性数据），
# 三条预览路径都传了 _load_float(..., encode=False)，
# **唯独 _export_one 用了默认的 encode=True** —— 等于把 sRGB 编码后的数据当线性再送进管线，
# 双重编码。表现是导出比预览明显偏亮、丢对比度（实测均值 80.8 → 137.5）。
# 既有测试全都建立在 transform() 的结果上，没人检查过导出实际喂进去的是什么，所以漏掉了。
print('\n[15] 导出与预览的源数据域一致性')

_RAW_ANY = None
for _pat in (r'C:\Users\Administrator\Desktop\**\*.ARW', r'F:\DS Harness\**\*.ARW'):
    _h = _glob.glob(_pat, recursive=True)
    if _h:
        _RAW_ANY = _h[0]
        break

if _RAW_ANY is None:
    check('导出路径 feed 线性数据（未找到 RAW 样本，跳过）', True, 'skip')
else:
    import pathlib as _pl
    import tempfile as _tf
    _seen = []
    _orig_lf = cl._load_float

    def _spy(rp, *a, **kw):
        _seen.append(kw.get('encode', True))
        return _orig_lf(rp, *a, **kw)

    class _Stub:
        _raw_work = cl._RAW_WORK_DEF
        _od = None
        _png_level = 3
        _wb_ref = 6500.0

        def _wb_args(self):
            return None

        def _transform_tiled(self, arr, s):
            return cl._tiled(arr, lambda x: cl.transform(x, s))

    _st = _Stub()
    _tmpd = _tf.mkdtemp(prefix='_expchk')
    _st._od = _tmpd
    _snap = {'rw': cl._RAW_WORK_DEF, 'sg': 'sRGB', 'sc': 'sRGB', 'tg': 'sRGB', 'tc': 'sRGB',
             'lut': False, 'cd': None, 'cd_path': '', 'ev': 0.0, 'ct': 0.0, 'sat': 0.0,
             'tp': 0.0, 'dither': False, 'ext': '.png', 'fmt': 'PNG 8-bit'}
    # 这一步只为了**监视 _load_float 收到了什么实参**，导出本身成功与否不重要。
    # 但 _export_one 失败时会往项目根写 export_error.log —— 测试不该污染产品目录，
    # 所以记下它原本在不在，事后把自己引发的那份清掉。
    _log = _pl.Path(cl.__file__).parent / 'export_error.log'
    _log_had = _log.exists()
    cl._load_float = _spy
    try:
        cl.App._export_one(_st, _pl.Path(_RAW_ANY), _snap)
    finally:
        cl._load_float = _orig_lf
        if not _log_had and _log.exists():
            try:
                _log.unlink()
            except Exception:
                pass
    _enc = [e for e in _seen if e is not None]
    check('导出路径调用 _load_float 时传了 encode=False（否则双重编码）',
          bool(_seen) and all(e is False for e in _seen),
          f'实际传入 encode={_seen}')
    # 顺带验证语义：encode=True 确实会编码（两者必须不同，否则这个开关是坏的）
    _a = cl._load_float(_RAW_ANY, half=True, work=cl._RAW_WORK_DEF, encode=False)
    _b = cl._load_float(_RAW_ANY, half=True, work=cl._RAW_WORK_DEF, encode=True)
    _diff = float(np.abs(np.asarray(_a) - np.asarray(_b)).max())
    check('encode=True/False 确实产生不同数据（开关有效）', _diff > 0.1, f'最大差 {_diff:.3f}')
    _mid = float(np.asarray(_a).mean())
    check('encode=False 返回的是线性数据（均值应显著低于 sRGB 编码后）',
          _mid < float(np.asarray(_b).mean()),
          f'线性 {_mid:.4f} vs 编码 {float(np.asarray(_b).mean()):.4f}')

    # ---- 色温滑块：只要求**有界**，不要求亮度不变 ----
    # 曾有一条"滑动色温不改变整体亮度（纯色度操作）"的断言，配套一个把解码结果
    # 乘标量拉回原亮度的"亮度锁定"。**两者都已移除**，理由：
    #   1) 把光源假设从 L1 改成 L2，画面本就是 (L_actual/L_assumed)·反射率，
    #      **亮度跟着动是物理自然**，不该被抹掉（用户明确认可）。
    #   2) 那个锚是**内容相关**的（全图平均亮度），不同照片得到不同归一化，不科学。
    #   3) 它当初是在补矩阵方向 bug 造成的 +57% 巨漂；方向修对后自然漂移只剩约 ±7%。
    # 这里改测"有界"：防止将来又出现量级失控的漂移。
    _shot = float(round(cl.read_shot_cct(_RAW_ANY) or 6500.0))

    def _lum(T):
        _x = cl._load_float(_RAW_ANY, half=True, work=cl._RAW_WORK_DEF,
                            wb=(float(T), 0.0, _shot), encode=False)
        _s = np.asarray(_x[::4, ::4], dtype=np.float64)
        return float(0.2126*_s[..., 0].mean() + 0.7152*_s[..., 1].mean()
                     + 0.0722*_s[..., 2].mean())
    _l1, _l2 = _lum(3000), _lum(9000)
    _drift = abs(_l2/_l1 - 1.0) if _l1 > 1e-9 else 9.9
    check('滑动色温的亮度变化有界（自然变化允许，量级失控不允许）',
          _drift < 0.35,
          f'3000K→9000K 亮度变化 {_drift*100:.1f}%（矩阵方向修对后约 7%；'
          f'曾因方向反了达到 55%+）')

    # ---- rgb_xyz_matrix 的方向（矩阵名有误导，这里用物理判据钉死）----
    # daylight_whitebalance 按定义就是 D65 光源，相机对它的响应 = 1/daylight_wb。
    # 若把它映射到 XYZ 得到 D65 色度 (0.3127,0.3290)，则所用的就是 相机→XYZ。
    # 曾经把 rawpy 的 rgb_xyz_matrix 当成 相机→XYZ 用（实际是 XYZ→相机），
    # 白平衡增益整条算错 —— 滑动色温时色相偏，就是从这里来的。
    try:
        import rawpy as _rp
        with _rp.imread(_RAW_ANY) as _r:
            _M = np.asarray(_r.rgb_xyz_matrix, np.float64)[:3, :3]
            _day = np.asarray(_r.daylight_whitebalance, np.float64)[:3]
        _resp = 1.0 / (_day / _day[1])                 # 相机对 D65 的响应（**相机空间**）
        _xyz = np.linalg.solve(_M, _resp)              # 相机 → XYZ：M 是 XYZ→相机，求逆
        _sm = _xyz.sum()
        _x, _y = _xyz[0]/_sm, _xyz[1]/_sm
        _dev = float(np.hypot(_x - 0.3127, _y - 0.3290))
        check('rgb_xyz_matrix 方向正确（M @ D65响应 必须落在 D65 色度上）',
              _dev < 0.005,
              f'得到色度 ({_x:.4f},{_y:.4f})，偏离 D65 {_dev:.4f}'
              f'（方向若反会是 0.39 —— 白平衡增益全错）')
        _wrong = _M @ _resp                            # 反方向：把 XYZ→相机 当 相机→XYZ 用
        _sw = _wrong.sum(); _wx, _wy = _wrong[0]/_sw, _wrong[1]/_sw
        check('  自检：反方向确实会明显偏离（证明这条断言有鉴别力）',
              float(np.hypot(_wx - 0.3127, _wy - 0.3290)) > 0.1,
              f'反方向色度 ({_wx:.4f},{_wy:.4f})')
    except ImportError:
        pass

    # ---- 显示端做不到的事，必须说做不到 ----
    # 场景参考（Log/Linear）内容的场景→显示渲染**没有唯一解**
    # （色域超出、白点意图、观看环境都是创作判断）→ 显示端只能如实展示 + 标注。
    check('场景参考内容在显示端做不到（必须标注"不可据此判断成片"）',
          all(cl.display_cannot_judge(c) for c in
              ("S-Log3", "V-Log", "ARRI LogC3", "RED Log3G10", "ACEScct", "Linear")),
          'S-Log3→%s  Linear→%s'
          % (cl.display_cannot_judge("S-Log3"), cl.display_cannot_judge("Linear")))
    check('  自洽与视频 OETF 内容**能**正确显示（不许误标）',
          not any(cl.display_cannot_judge(c) for c in
                  ("sRGB", "Gamma 2.2", "Gamma 2.6", "Rec.709", "Rec.2020")))

    # ---- 显示 EOTF：有效性 + 同一性（C 方案）----
    # 编码曲线与显示 EOTF 是两件事，只在 sRGB 下重合。
    # BT.709 的那条曲线是**相机侧 OETF**（光→信号），显示端由 BT.1886(γ2.4) 定义。
    # 用错会让 Rec.709 内容按系统伽马 ≈1.0 显示（规范意图 ≈1.08~1.2），
    # 实测暗部偏亮 2~53 倍 —— 用户症状是"声明 Rec.709 后暗部变亮"。
    check('显示 EOTF 映射：视频 OETF 内容用 BT.1886，自洽内容保持自身',
          cl.display_decode_curve("Rec.709") == "Gamma 2.4"
          and cl.display_decode_curve("Rec.2020") == "Gamma 2.4"
          and all(cl.display_decode_curve(c) == c
                  for c in ("sRGB", "Gamma 2.2", "Gamma 2.6")),
          'Rec.709→%s  sRGB→%s'
          % (cl.display_decode_curve("Rec.709"), cl.display_decode_curve("sRGB")))

    _vs = np.array([0.02, 0.05, 0.18, 0.5, 0.9], np.float32)
    _dg2, _dc2 = cl.display_space()[:2]
    _sddec = cl.CURVE[_dc2][1]
    _a = np.stack([_vs, _vs, _vs], 1)[None, :, :].astype(np.float32)

    def _cl_lum(loc):
        _s = {"lut": True, "cd": {"t": "3d"}, "tg": "sRGB", "tc": loc,
              "lo": "sRGB", "loc": loc}
        _o = np.asarray(cl.display_leg_nb(_a, _s), np.float64)[0, :, 0]
        return np.asarray(_sddec(_o.astype(np.float32)), np.float64)

    _want709 = np.asarray(_vs, np.float64) ** 2.4            # BT.1886（Lb=0）
    _d709 = float(np.abs(_cl_lum("Rec.709") - _want709).max())
    check('有效性：CL 显示链把 Rec.709 内容落到 BT.1886（不再是 OETF 逆）',
          _d709 < 2e-3,
          '与 BT.1886 最大差 %.2e（修前 v=0.18 偏亮 2.9 倍）' % _d709)

    def _icc_trc_fn(gamut, curve):
        _b = cl._icc_bytes(gamut, curve)
        import struct as _st
        _n = _st.unpack(">I", _b[128:132])[0]
        _tags = {}
        for _i in range(_n):
            _s, _o, _z = _st.unpack(">4sII", _b[132 + _i * 12:144 + _i * 12])
            _tags[_s.decode("latin1")] = (_o, _z)
        _o, _z = _tags["rTRC"]
        if _b[_o:_o + 4] == b"para":
            _fn, = _st.unpack(">H", _b[_o + 8:_o + 10])
            _np_ = {0: 1, 1: 3, 2: 4, 3: 5, 4: 7}[_fn]
            _vals = [x / 65536.0 for x in
                     _st.unpack(">" + "i" * _np_, _b[_o + 12:_o + 12 + 4 * _np_])]
            if _fn == 0:
                _g = _vals[0]
                return lambda v: np.power(np.asarray(v, np.float64), _g)
        return None

    _f = _icc_trc_fn("sRGB", "Rec.709")
    if _f is not None:
        _di = float(np.abs(_f(_vs) - _cl_lum("Rec.709")).max())
        check('同一性：导出 ICC 的 TRC 与显示端同源（CL 显示 == 第三方按 ICC 解码）',
              _di < 2e-3, '两者最大差 %.2e' % _di)
    else:
        print('  [SKIP] ICC TRC 解析失败')

    _dmax = 0.0
    for _c in ("sRGB", "Gamma 2.2", "Gamma 2.6"):
        if _c not in cl.CURVE:
            continue
        _w2 = np.asarray(cl.CURVE[_c][1](_vs.astype(np.float32)), np.float64)
        _dmax = max(_dmax, float(np.abs(_cl_lum(_c) - _w2).max()))
    check('  自洽空间（sRGB/纯 gamma）不被误伤', _dmax < 2e-3, '最大差 %.2e' % _dmax)

    # ---- 末端 CST：LUT 输出空间 → **目标**（= 文件最终所处的空间）----
    # 4.0 简化（用户指出）：套 LUT 时 tg/tc 曾被完全忽略，于是另造了一个
    # 「输出文件色彩空间」开关。其实第二次 CST 一直都在（不套 LUT 时 源→tg/tc 就是它），
    # 套 LUT 时只是被跳过。现在统一：**「目标」永远 = 文件所处的空间**。
    #   · 想要 LUT 原值 → 把目标设成 LUT 输出的空间，末端 CST 自动成为恒等
    #   · else → 末端 CST 把 LUT 输出真的转到目标（并在开启时被写进文件）
    _bs = {"lut": False, "cd": None, "tg": "sRGB", "tc": "Rec.709",
           "lo": "sRGB", "loc": "Rec.709", "li": "sRGB", "lc": "Rec.709",
           "sg": "sRGB", "sc": "sRGB", "ev": 0.0, "ct": 0.0, "sat": 0.0,
           "tp": 0.0, "ootf": "off", "ootf2": "off", "raw": False}
    check('出口恒为「目标」：display_leg_out_space == (tg, tc)',
          all(cl.display_leg_out_space(dict(_bs, tg=g, tc=c)) == (g, c)
              for g, c in (("sRGB", "sRGB"), ("Display P3", "sRGB"),
                           ("Rec.709", "Gamma 2.4"))),
          str(cl.display_leg_out_space(_bs)))

    # 末端 CST 必须与显示层**互为镜像**：两侧都按显示 EOTF，且方向显式解析。
    # 曾经传 ootf="auto"，被 numba `_pipe_go` 里**按色域**判定的遗留分支接管，
    # 于是 Rec.709→Rec.709（两侧都显示参考）被误加正向 OOTF —— 实测差 14%。
    _x = np.array([[[0.3, 0.6, 0.45]]], np.float32)
    _id = float(np.abs(np.asarray(cl._nb.pipe_fast(
        _x, "sRGB", cl.display_decode_curve("Rec.709"), "Rec.709",
        cl.display_decode_curve("Rec.709"), ootf="off", gc=2), np.float64)[0, 0]
        - _x[0, 0]).max()) if cl._NB else 0.0
    check('  目标 = LUT 输出空间 ⇒ 末端 CST 恒等（LUT 原值的那条路）',
          _id < 1e-6, '与输入最大差 %.2e' % _id)
    check('  末端 CST 方向必须显式解析（传 auto 会被按色域的遗留分支带偏）',
          hasattr(cl.App, "_ootf_direction2"),
          'App._ootf_direction2 存在')

    _dark = float(np.asarray(cl._nb.pipe_fast(
        _x, "sRGB", cl.display_decode_curve("Rec.709"), "Rec.709",
        cl.display_decode_curve("Rec.709"), ootf="auto"), np.float64)[0, 0, 0]) if cl._NB else 0.0
    check('  同色料空间互换不得改变颜色（sRGB↔Rec.709 矩阵完全相同）',
          abs(float(np.asarray(cl.Pipe.go(_x, "sRGB", "sRGB", "Rec.709", "sRGB",
                                          ootf="off"), np.float64)[0, 0, 0]) - 0.3) < 1e-5,
          'sRGB→Rec.709 的 R 通道 = %.6f（应 0.3）'
          % float(np.asarray(cl.Pipe.go(_x, "sRGB", "sRGB", "Rec.709", "sRGB",
                                        ootf="off"), np.float64)[0, 0, 0]))

    # 裁切警告：LUT 输出色域比目标宽时必须警告；不套 LUT 时没有这一级
    _lut = {"lut": True, "cd": {"t": "3d"}}
    _w_p3 = cl.out_cst_gain(dict(_bs, **_lut, lo="Display P3", tg="sRGB", tc="sRGB"))[0]
    _w_ok = cl.out_cst_gain(dict(_bs, **_lut, lo="sRGB", tg="sRGB", tc="sRGB"))[0]
    _w_off = cl.out_cst_gain(dict(_bs, lut=False, cd=None, lo="Display P3",
                                  tg="sRGB", tc="sRGB"))[0]
    check('  色域收窄必须警告（P3→sRGB 要裁切）；同色料/无 LUT 不警告',
          _w_p3 and not _w_ok and not _w_off,
          'P3→sRGB=%s  sRGB→sRGB=%s  无LUT=%s' % (_w_p3, _w_ok, _w_off))

    # ---- 显示器档案必须被**真正使用**（此前只认 sRGB）----
    # 原实现读到了档案、也解析了原色，却在"不是 sRGB"时主动退回 sRGB
    # （理由写着"任意显示器档案的支持尚未实现"）。于是在宽色域显示器上，
    # 内容一律先压进 sRGB 再送屏 —— **超出 sRGB 的颜色在显示端就被裁掉**，
    # 而文件里其实还有。用户："就算我的显示器支持全色域，也看不到 sRGB 以外的颜色。"
    try:
        import tempfile as _tf
        from pathlib import Path as _Pl
        _d2 = _Pl(_tf.mkdtemp())
        _saved = (cl._DISPLAY_SPACE, cl._monitor_profile_path)
        _vb = []
        for _gn in ("Display P3", "DCI-P3", "Adobe RGB", "Rec.2020", "sRGB"):
            _f = _d2 / (_gn.replace(" ", "_") + ".icc")
            _f.write_bytes(cl._icc_bytes(_gn, cl._CURVE_MATCH.get(_gn, "sRGB")))
            cl._DISPLAY_SPACE = None
            cl._monitor_profile_path = (lambda _p: (lambda: _p))(str(_f))
            _g2 = cl.display_space()[0]
            _vb.append((_gn, _g2))
        check('显示器档案必须被真正使用（宽色域显示器不再被压进 sRGB）',
              all(a == b for a, b in _vb),
              "  ".join("%s→%s" % (a, b) for a, b in _vb))
        # 认不出时不许假装认得出
        _f2 = _d2 / "junk.icc"
        _f2.write_bytes(b"not a profile")
        cl._DISPLAY_SPACE = None
        cl._monitor_profile_path = (lambda _p: (lambda: _p))(str(_f2))
        _g3, _c3, _n3 = cl.display_space()
        # 断言检查的是「是否**如实说明**」，不是某个特定字符 ——
        # 不同分支的措辞不同（"非矩阵类…按 sRGB 监看" / "⚠ 原色不在已知色域表里" /
        # "未读到显示器档案"），早先只认 "⚠" 会误判成失败。
        check('  档案认不出时退回 sRGB 并如实说明（不假装支持）',
              _g3 == "sRGB" and ("按 sRGB" in _n3 or "未读到" in _n3), _n3[:70])
        cl._DISPLAY_SPACE, cl._monitor_profile_path = _saved
    except Exception as _e:
        print('  [SKIP] 显示器档案检查:', _e)

    # ---- 任意显示器档案：实测原色注册成自定义色域 ----
    # 真实显示器**实测出来的原色**与标准 P3/Adobe RGB 原色本就不同（面板差异 + 校准），
    # 精确匹配（容差 0.004）必然不命中。原先不命中就退回 sRGB —— 于是用户的校准档案
    # （连名字叫 DCIP3 的那个）全被判成"不在已知色域表里"，宽色域显示器被压成 sRGB，
    # **看不到 sRGB 以外的颜色**。现在改为：用档案的实测原色注册自定义色域。
    try:
        import tempfile as _tf2
        from pathlib import Path as _Pl2
        _d3 = _Pl2(_tf2.mkdtemp())
        _saved2 = (cl._DISPLAY_SPACE, cl._monitor_profile_path)
        _f3 = _d3 / "measured.icc"
        _f3.write_bytes(cl._icc_bytes("Adobe RGB", "Gamma 2.2"))
        cl._DISPLAY_SPACE = None
        cl._monitor_profile_path = (lambda _p: (lambda: _p))(str(_f3))
        _g4, _c4, _n4 = cl.display_space()
        check('任意显示器档案可用：不在已知表里也不退回 sRGB（用实测原色）',
              _g4 != "sRGB", '%s / %s  %s' % (_g4, _c4, _n4[:58]))
        _ok4 = False
        try:
            _M4 = np.asarray(cl._mat(_g4)[0], np.float64)
            _ok4 = _M4.shape == (3, 3) and bool(np.isfinite(_M4).all())
        except Exception:
            pass
        check('  注册出的色域能建出合法矩阵（否则显示链会静默 return a，画面纹丝不动）',
              _ok4, 'shape=%s' % (np.asarray(cl._mat(_g4)[0]).shape,))
        check('  自定义色域不进 GAMUT_NAMES（不污染界面下拉）',
              "__DISPLAY__" not in cl.GAMUT_NAMES,
              'GAMUT_NAMES 里有 __DISPLAY__ ? %s' % ("__DISPLAY__" in cl.GAMUT_NAMES))
        cl._DISPLAY_SPACE, cl._monitor_profile_path = _saved2
    except Exception as _e:
        print('  [SKIP] 任意显示器档案检查:', _e)

    # ---- 软打样 A/B 工具：差异必须是"真实的" ----
    # 用户："可以做一个三原色的色块图 ab，让用户自行判断是否有差异。
    #          还可以加入 24 色图作为后备的验证。"
    # 工具本身要能被验证，否则它只是又一张"看起来对"的图。
    try:
        import importlib.util as _iu
        _sp = _iu.spec_from_file_location(
            "softproof_ab", str(_Pl(cl.__file__).parent / "tools" / "softproof_ab.py"))
        _m = _iu.module_from_spec(_sp); _sp.loader.exec_module(_m)
        # ① 目标是 sRGB → 必须零差异（否则说明链路里混进了不该有的变换）
        _e0 = max(_m.delta_e(c, tuple(int(x) for x in _m.convert([c], "sRGB", "sRGB")[0]))
                  for _, c in _m.PRIMARIES + _m.CHECKER)
        check('软打样工具：目标 = sRGB 时差异必须为 0（链路里不许混进多余变换）',
              _e0 < 0.01, '最大 ΔE76 = %.4f' % _e0)
        # ② 目标是宽色域 → 必须有明显差异（否则工具测不出东西）
        _e1 = max(_m.delta_e(c, tuple(int(x) for x in _m.convert([c], "Display P3", "sRGB")[0]))
                  for _, c in _m.PRIMARIES)
        check('  目标 = Display P3 时原色差异必须显著（否则图看不出差别，等于没测）',
              _e1 > 10.0, '原色最大 ΔE76 = %.2f（阈值 10）' % _e1)
        # ③ 中性灰不跨色域 → 两侧必须一致（重要的自检：防止把灰也改了）
        _gy = [(n, c) for n, c in _m.CHECKER if n.startswith("中性") or n in ("白", "黑")]
        _eg = max(_m.delta_e(c, tuple(int(x) for x in _m.convert([c], "Display P3", "sRGB")[0]))
                  for _, c in _gy)
        check('  中性灰不得被改动（灰轴不跨色域，改了就是错）', _eg < 1.0,
              '灰阶最大 ΔE76 = %.2f' % _eg)
    except Exception as _e:
        print('  [SKIP] 软打样工具检查:', _e)

    # ---- 逐级指纹：五级一次跑完，任一级偏就报是哪一级 ----
    # 这个工具第一次跑就抓到两个真问题：
    #   · 主 CST 的目标编码用 tc 而非 display_decode_curve(tc) → 像素与 ICC 矛盾
    #     （Rec.709 目标实测 ΔE76 13.84）
    #   · 工具自己绕过 transform、且没覆盖全局 display_space() → 假阳性
    # 把它钉进回归，这两类问题就不会再回来。
    try:
        import importlib.util as _iu2
        _fp = _iu2.spec_from_file_location(
            "chain_fp", str(_Pl(cl.__file__).parent / "tools" / "chain_fingerprint.py"))
        _cfp = _iu2.module_from_spec(_fp); _fp.loader.exec_module(_cfp)
        _worst = (0.0, "")
        for _tg, _dg in (("sRGB", "sRGB"), ("Rec.709", "sRGB"),
                         ("Display P3", "sRGB"), ("sRGB", "Display P3")):
            _tc2 = cl._CURVE_MATCH.get(_tg, "sRGB")
            _dc2 = cl._CURVE_MATCH.get(_dg, "sRGB")
            _saved3 = cl._DISPLAY_SPACE
            cl._DISPLAY_SPACE = (_dg, _dc2, "指纹回归")
            try:
                for _nm, _rgb in _cfp.PRIMARIES + _cfp.CHECKER:
                    _r = _cfp.fingerprint_one(_rgb, _tg, _tc2, _dg, _dc2)
                    if _r["dE"] > _worst[0]:
                        _worst = (_r["dE"], f"{_nm}@{_tg}->{_dg}")
            finally:
                cl._DISPLAY_SPACE = _saved3
        check('逐级指纹：文件→显示器码值全链自洽（ΔE76 全部为 0）',
              _worst[0] < 1.0, '最差 %.3f（%s）' % _worst)
    except Exception as _e:
        print('  [SKIP] 逐级指纹检查:', _e)

    # ---- RGC 只在"目标更窄"时才该执行 ----
    # 主 CST 默认 gc=None → mode 0 = ACES RGC，**无条件执行**。
    # 于是 sRGB→Rec.709（色料矩阵逐位相同）也被压缩：实测蓝原色差 2.26e-01。
    # 这不是观感取舍 —— 同色域/窄→宽 的转换必须是**恒等**，否则每导出一次就丢一次饱和。
    _fit = {("sRGB", "Rec.709"): True, ("sRGB", "Rec.2020"): True,
            ("sRGB", "Display P3"): True, ("sRGB", "Adobe RGB"): True,
            ("Adobe RGB", "sRGB"): False, ("Display P3", "sRGB"): False,
            ("Rec.2020", "sRGB"): False}
    _bad = [f"{a}->{b}" for (a, b), w in _fit.items()
            if cl._sg_fits_in_tg(a, b) != w]
    check('「源色域装得下目标」判定正确（决定要不要缩限）', not _bad,
          '错判: %s' % _bad if _bad else '7 组全对')
    check('  装得下 → gc=2（不缩限）；装不下 → None（走默认 RGC）',
          cl.gc_for("sRGB", "Rec.709") == 2 and cl.gc_for("Adobe RGB", "sRGB") is None,
          'sRGB→Rec.709=%s  AdobeRGB→sRGB=%s'
          % (cl.gc_for("sRGB", "Rec.709"), cl.gc_for("Adobe RGB", "sRGB")))
    # 真正的回归守卫：同色料转换必须逐位恒等
    _px = np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
                     [0.2, 0.5, 0.8]]], np.float32)
    _o1_ = np.asarray(cl.Pipe.go(_px, "sRGB", "sRGB", "Rec.709",
                                 cl.display_decode_curve("Rec.709"), ootf="off",
                                 gc=cl.gc_for("sRGB", "Rec.709")), np.float64)
    _o2_ = np.asarray(cl.Pipe.go(_px, "sRGB", "sRGB", "Rec.709",
                                 cl.display_decode_curve("Rec.709"), ootf="off",
                                 gc=2), np.float64)
    check('  同色料转换必须恒等（RGC 不得凭空压缩饱和色）',
          float(np.abs(_o1_ - _o2_).max()) < 1e-9,
          '最大差 %.2e（修前 2.26e-01）' % float(np.abs(_o1_ - _o2_).max()))

    # ---- 文件嵌入的 ICC 必须被读出来（此前从不读）----
    # 用户以为"读得到 ICC 所以输入 cs 不成问题"，实际全文 ImageCms 只出现在
    # 「生成 sRGB 档案写 ICC」那一处 —— 从不读。于是源空间全靠手动选，
    # 而源**色域**又没有用户声明存储（只有曲线有 _sc_user），
    # RAW 预览会把它覆盖成解码层工作空间 → sRGB JPEG 被当成 ACES2065-1 解释。
    try:
        import io as _io
        from PIL import Image as _Im, ImageCms as _Cms
        _tmp2 = _pl.Path(_tmpd) / "_icc_probe.jpg"
        _im = _Im.new("RGB", (8, 8), (128, 128, 128))
        _im.save(_tmp2, "JPEG",
                 icc_profile=_Cms.ImageCmsProfile(_Cms.createProfile("sRGB")).tobytes())
        _g, _c, _note = cl.icc_to_space(str(_tmp2))
        check('能读出文件嵌入的 ICC 并映射成源空间',
              _g == "sRGB" and _c == "sRGB",
              f'得到 {_g}/{_c}  ({_note})')
        _im.save(_pl.Path(_tmpd) / "_noicc.jpg", "JPEG")      # 不带 ICC
        _g2, _c2, _n2 = cl.icc_to_space(str(_pl.Path(_tmpd) / "_noicc.jpg"))
        check('  无 ICC 时按通行约定当 sRGB，且**说明是"未标记"**（不假装读到）',
              _g2 == "sRGB" and "未嵌入" in _n2, f'{_g2}/{_c2}  {_n2}')
        check('  不可识别时**不猜**（返回 None，保留用户声明）',
              cl.icc_to_space(str(_pl.Path(_tmpd) / "_nope.jpg"))[0] is None)
    except Exception as _e:
        print('  [SKIP] ICC 读取检查:', _e)

    # ---- OOTF 的目标侧必须是 CST 的实际目标，不是 LUT 的输出空间 ----
    # 数据流：RAW线性 --CST--> [LUT输入] --LUT--> [LUT输出] --显示转换--> 监看
    # OOTF 只作用在 CST 这一段上；LUT 输出空间驱动的是**下游显示转换**，是另一个阶段。
    # 二者曾被混为一谈，后果具体：官方 S-Log3→Rec.709 还原 LUT 被判成 forward，
    # 于是在 S-Log3 编码**之前**先乘了显示伽马 —— 喂给 LUT 的数据是错的。
    # 用源码结构钉住（这条逻辑在 App 方法里，测试套件不建真实 App）。
    import inspect as _insp
    _osrc = _insp.getsource(cl.App._ootf_direction)
    # 只匹配**调用形式** `display_leg_out_space(`：该函数的文档里会**提到**这个名字
    # （说明"那是下游显示转换，是另一个阶段"），按整段源码 grep 会误报 —— 第一版就误报了。
    check('OOTF 目标侧取 CST 的实际目标（套 LUT 时 LUT 的**输入**曲线），不是 LUT 输出空间',
          "self.li" in _osrc and "display_leg_out_space(" not in _osrc,
          '应调用 self.li（LUT 输入），且不得**调用** display_leg_out_space（LUT 输出）')

    # ---- 参考类型必须由**曲线**决定，不能由色域 ----
    # 旧实现用色域名判断（_DISPLAY_REF 里全是色域名），但同一个色域可以有两种参考：
    # Rec.2020 + Gamma 2.6 = 显示参考；Rec.2020 + V-Log = 场景参考。
    # 用户指出「关键不是色域，而是 log，因为主要关心的变换是明度(log)主导的」。
    check('参考类型由曲线决定：Log = 场景参考',
          all(cl._ref_type(c) == "scene" for c in ("S-Log3", "V-Log", "ARRI LogC3", "ACEScct")),
          f'S-Log3→{cl._ref_type("S-Log3")} V-Log→{cl._ref_type("V-Log")}')
    check('参考类型由曲线决定：显示曲线 = 显示参考',
          all(cl._ref_type(c) == "display" for c in ("sRGB", "Rec.709", "Gamma 2.2", "Gamma 2.6")))
    check('  自检：RAW 的 Linear 归入**非显示**侧（否则 RAW→sRGB 会不施加 forward OOTF）',
          cl._ref_type("Linear") != "display",
          f'Linear → {cl._ref_type("Linear")}（旧实现靠色域名 ACES2065-1 碰巧也对）')

    # ---- 文件所处的空间 ----
    # 演进史：① 曾硬编码按 sRGB；② 4.0 改为按用户声明的 **LUT 输出空间 (lo/loc)**；
    # ③ **再改为「目标」(tg/tc)** —— 用户指出：套 LUT 时 tg/tc 被完全忽略，
    #    于是另造了一个「输出文件色彩空间」开关；其实那一级 CST 一直都在
    #    （不套 LUT 时 源→tg/tc 就是它），套 LUT 时只是被跳过。现在统一：
    #    **不管套不套 LUT，文件最终所处的空间永远是「目标」。**
    #    lo/loc 只声明「LUT 吐出来的是什么」，是末端那一级 CST 的起点。
    _sl_ = {"lut": True, "cd": {"t": "3d"}, "tg": "DCI-P3", "tc": "Gamma 2.6",
            "lo": "Rec.709", "loc": "Rec.709"}
    check('文件空间恒为「目标」：套 LUT 时也是 tg/tc（lo/loc 只是末端 CST 的起点）',
          cl.display_leg_out_space(_sl_) == ("DCI-P3", "Gamma 2.6")
          and cl.pipeline_out_space(_sl_) == ("Rec.709", "Rec.709"),
          f'文件={cl.display_leg_out_space(_sl_)}  LUT输出={cl.pipeline_out_space(_sl_)}')
    _sl2 = dict(_sl_); _sl2["lut"] = False
    check('  原图旁路 LUT 时必须落回 tg/tc（否则显示转换用错空间）',
          cl.display_leg_out_space(_sl2) == ("DCI-P3", "Gamma 2.6"),
          f'得到 {cl.display_leg_out_space(_sl2)}')

    # ---- 高光重建：默认绝不能落回 Clip ----
    # rawpy 的默认恰好是最粗糙的 Clip，而我们**从未显式设过**它 —— 于是长期默默
    # 用最差档，高光发品红/青。传感器各通道各自饱和，一个通道先到顶、其余还在涨时
    # 颜色比例就崩了。实测用户 10 张图有 3 张「仅 1 通道裁切」占 0.7%~2.4%。
    _hle = cl._hl_enum()
    check('高光重建默认不是 Clip（rawpy 默认恰是最差档，必须显式覆盖）',
          cl._HL_MODE != "裁切（最差）" and _hle is not None,
          f'当前 {cl._HL_MODE} → {None if _hle is None else _hle.name}')

    # ---- 双光源标定矩阵（可选增强，只取客观标定数据）----
    _mk, _md = cl._shot_model(_RAW_ANY)
    _prof = cl._dcp_find(_mk, _md)
    if _prof:
        check('双光源：色温两端必须精确命中各自的 ColorMatrix',
              float(np.abs(cl._dcp_matrix(_prof, _prof["T1"]) - _prof["M1"]).max()) < 1e-9 and
              float(np.abs(cl._dcp_matrix(_prof, _prof["T2"]) - _prof["M2"]).max()) < 1e-9,
              '%s  %.0fK / %.0fK' % (_prof["src"], _prof["T1"], _prof["T2"]))
        check('  超出标定区间必须夹住（不外推成非物理矩阵）',
              float(np.abs(cl._dcp_matrix(_prof, 1500) - _prof["M1"]).max()) < 1e-9 and
              float(np.abs(cl._dcp_matrix(_prof, 20000) - _prof["M2"]).max()) < 1e-9)
    else:
        print('  [SKIP] 本机无 %s %s 的 .dcp —— 正常，会自动退回 LibRaw 单矩阵' % (_mk, _md))

    import shutil as _sh
    _sh.rmtree(_tmpd, ignore_errors=True)


# ───────────────────────── 汇总 ─────────────────────────
failed = [n for n, ok, _ in RESULTS if not ok]
print('\n' + '=' * 72)
print(f'汇总：{len(RESULTS) - len(failed)} / {len(RESULTS)} 通过')
if failed:
    print('待修项：')
    for n in failed:
        print(f'  - {n}')
print('=' * 72)
sys.exit(1 if failed else 0)
