#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""数值等价性检查：4.0 的改动到底把输出改了多少？

分两步（colorlab 模块名会冲突，两个版本必须跑在两个进程里）：
    # 1) 各自 dump
    .venv\\Scripts\\python.exe tests\\numeric_equiv.py dump --root <树> --out <npz> --file <ARW>
    # 2) 对比
    .venv\\Scripts\\python.exe tests\\numeric_equiv.py compare --a <npz33> --b <npz40>
"""
import argparse, hashlib, sys
from pathlib import Path

import numpy as np


def build_snap(cl, arw):
    lut = Path(cl.__file__).parent / 'lut' / 'V-Gamut-V-Log' / 'Fujifilm' / 'Fuji_Velvia.cube'
    li, lc = cl._folder_input('V-Gamut-V-Log')
    return {'sg': cl._RAW_WORK_DEF, 'sc': 'sRGB', 'raw': True, 'rw': cl._RAW_WORK_DEF,
            'tg': 'sRGB', 'tc': 'sRGB', 'lut': True, 'ootf': 'auto', 'li': li, 'lc': lc,
            'cd': cl._cp(str(lut)), 'cd_path': str(lut), 'ev': 0.3, 'ct': 0.10, 'sat': 0.15,
            'tp': 6500.0, 'dither': False, 'ext': '.png', 'fmt': 'PNG 8-bit'}


def do_dump(args):
    root = Path(args.root).resolve()
    sys.path.insert(0, str(root))
    import colorlab as cl
    arw = Path(args.file)
    shot = cl.read_shot_cct(arw) or 6500.0
    snap = build_snap(cl, arw); snap['tp'] = shot

    w = np.random.RandomState(0).rand(64, 64, 3).astype(np.float32)
    cl.transform(w, snap)                                   # JIT 预热

    a = cl._load_float(str(arw), half=args.half, work=cl._RAW_WORK_DEF, wb=(shot, 0.0, shot))
    f32_lin = np.ascontiguousarray(a, dtype=np.float32)
    lin_sha = hashlib.sha1(f32_lin.tobytes()).hexdigest()

    a = np.clip(cl._tiled(a, lambda x: cl.transform(x, snap)), 0, 1)
    f32 = np.ascontiguousarray(a, dtype=np.float32)
    fin_sha = hashlib.sha1(f32.tobytes()).hexdigest()

    np.savez(args.out,
             u16=(f32 * 65535 + 0.5).astype(np.uint16),
             u8=(f32 * 255 + 0.5).astype(np.uint8),
             meta=np.array([cl.APP_VERSION, fin_sha, lin_sha, str(f32.shape),
                            str([round(float(f32[..., k].mean()), 8) for k in range(3)])],
                           dtype=object))
    print(f'dump 完成: {args.out}')
    print(f'  形状        {f32.shape}   half={args.half}')
    print(f'  decode sha1 {lin_sha}')
    print(f'  最终 sha1   {fin_sha}')
    print(f'  逐通道 mean {[round(float(f32[..., k].mean()), 8) for k in range(3)]}')
    return 0


def do_compare(args):
    A = np.load(args.a, allow_pickle=True)
    B = np.load(args.b, allow_pickle=True)
    print(f'A = {args.a}\n    version={A["meta"][0]}  final_sha1={A["meta"][1]}\n    mean={A["meta"][4]}')
    print(f'B = {args.b}\n    version={B["meta"][0]}  final_sha1={B["meta"][1]}\n    mean={B["meta"][4]}')
    print()
    for k, label in (('u8', '8bit 输出'), ('u16', '16bit 输出')):
        x, y = A[k], B[k]
        if x.shape != y.shape:
            print(f'{label}: 形状不同 {x.shape} vs {y.shape}'); continue
        d = np.abs(x.astype(np.int32) - y.astype(np.int32))
        n = int((d != 0).sum())
        print(f'{label:<12} 不同 {n:,}/{x.size:,} ({n/x.size*100:.4f}%)  '
              f'最大差 {int(d.max())}  平均差 {d.mean():.5f}')
    print()
    print(f'decode 层 sha1 相同 (_load_float 输出): {A["meta"][2] == B["meta"][2]}')
    print(f'最终 float32 sha1 相同                : {A["meta"][1] == B["meta"][1]}')
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='mode', required=True)
    d = sub.add_parser('dump')
    d.add_argument('--root', required=True)
    d.add_argument('--out', required=True)
    d.add_argument('--file', required=True)
    d.add_argument('--half', action='store_true')
    c = sub.add_parser('compare')
    c.add_argument('--a', required=True)
    c.add_argument('--b', required=True)
    args = ap.parse_args()
    return do_dump(args) if args.mode == 'dump' else do_compare(args)


if __name__ == '__main__':
    sys.exit(main())
