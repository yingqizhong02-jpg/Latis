#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""路线 B：由 camera_whitebalance + LibRaw 色矩阵反算光源 CCT（只读，零新依赖）。

要点：rawpy 的 color_matrix 是 3x4（4 个相机通道 → 3 个输出通道）。
      Bayer 的两个绿通道响应一致，降维时应把 G1/G2 两列**相加**，才能得到 3x3 的 相机→XYZ。
验证：同一套算法作用在 daylight_whitebalance 上，应给出接近该机标称日光（约 5000–6000K）的值。
"""
import sys
import numpy as np

ARW = sys.argv[1] if len(sys.argv) > 1 else r'C:\Users\Administrator\Desktop\新建文件夹 (4)\DSC03492.ARW'
import rawpy


def cct_from_mul(mul4, M):
    """mul4: 4 通道白平衡系数（rawpy camera_whitebalance 原值）；M: 3x4 color_matrix。"""
    mul = np.array(mul4[:3], dtype=np.float64)
    mul = mul / (mul[1] if mul[1] else 1.0)          # 归一到 G=1
    illum_rgb = 1.0 / mul                             # 相机对光源的响应 ∝ 1/白平衡系数
    M3 = np.column_stack([M[:, 0], M[:, 1] + M[:, 3], M[:, 2]])   # 绿通道合并 → 3x3
    try:
        Minv = np.linalg.inv(M3)
    except np.linalg.LinAlgError:
        return None, None, None, M3
    XYZ = Minv @ illum_rgb
    s = XYZ.sum()
    if s == 0:
        return None, None, None, M3
    x, y = XYZ[0] / s, XYZ[1] / s
    n = (x - 0.3320) / (0.1858 - y)
    cct = 449 * n ** 3 + 3525 * n ** 2 + 6823.3 * n + 5520.33     # McCamy 近似
    return cct, (x, y), XYZ, M3


with rawpy.imread(ARW) as raw:
    M = np.array(raw.color_matrix, dtype=np.float64)
    cam = list(raw.camera_whitebalance)
    day = list(raw.daylight_whitebalance)
    print('color_matrix (3x4):')
    print(np.round(M, 4))
    M3 = np.column_stack([M[:, 0], M[:, 1] + M[:, 3], M[:, 2]])
    print('合并绿通道后的 3x3（相机→XYZ）:')
    print(np.round(M3, 4))
    print('  该矩阵对 (1,1,1) 的作用 =', np.round(M3 @ np.ones(3), 4), '（相机中性应回到 D65 附近）')

    for label, mul in (('机内 as-shot', cam), ('机内 daylight', day)):
        cct, xy, XYZ, _ = cct_from_mul(mul, M)
        if cct is None:
            print(f'  {label}: 不可解')
        else:
            print(f'  {label}: WB={[round(v,4) for v in mul[:3]]}  →  xy=({xy[0]:.4f},{xy[1]:.4f})'
                  f'  CCT ≈ {cct:.0f} K')

    # 与 _kelvin_gain 的关系：_kelvin_gain 是"相对 6500K 的中性增益"，与相机 WB 不是同一坐标系，
    # 因此这里不比较绝对值，只看"目标 K / 机内 K"的比值是否可用作相对偏移。
    sys.path.insert(0, r'F:\DS Harness\Color-Lab3.2')
    import colorlab as cl
    cct_shot, _, _, _ = cct_from_mul(cam, M)
    if cct_shot:
        print(f'\n  说明：_kelvin_gain 以 6500K 为基准（K=6500 → (1,1,1)）。')
        print(f'  若把滑块语义改为"目标 K"、默认取机内 {cct_shot:.0f}K，则默认增益 = '
              f'_kelvin_gain({cct_shot:.0f})/_kelvin_gain({cct_shot:.0f}) = 1 → 与现状完全一致。')
        print('  用户拖到 3200K 时增益 = _kelvin_gain(3200)/_kelvin_gain(%.0f) = %s'
              % (cct_shot, np.round(np.array(cl._kelvin_gain(3200.0)) / np.array(cl._kelvin_gain(cct_shot)), 4)))
print('\n（只读验证结束）')
