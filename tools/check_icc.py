#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查一张图里嵌入的 ICC / EXIF 到底对不对。

用法（把一个或多个图片拖到 check_icc.bat 上，或直接命令行传）：
    python check_icc.py <图片路径> [更多路径...]

输出：
  · 从文件字节里直接解析 ICC 与 EXIF（不依赖应用层）
  · 交给 Windows 自己的色彩引擎（mscms）验证
  · 与系统里的权威档案做对照
"""
import sys
import os
import io
import struct
import ctypes
import zlib
from ctypes import wintypes

D = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "System32", "spool", "drivers", "color")


# ---------------------------------------------------------------- 文件解析
def png_chunks(b):
    out, pos = [], 8
    while pos + 8 <= len(b):
        ln = struct.unpack(">I", b[pos:pos + 4])[0]
        typ = b[pos + 4:pos + 8].decode("latin1")
        out.append((typ, b[pos + 8:pos + 8 + ln]))
        pos += 12 + ln
        if typ == "IEND":
            break
    return out


def icc_from_png(b):
    for typ, data in png_chunks(b):
        if typ == "iCCP":
            z = data.index(b"\x00")
            return zlib.decompress(data[z + 2:]), data[:z].decode("latin1")
    return None, None


def exif_from_png(b):
    for typ, data in png_chunks(b):
        if typ == "eXIf":
            return data
    return None


def icc_from_jpeg(b):
    pos, parts = 2, {}
    while pos + 4 <= len(b):
        if b[pos] != 0xFF:
            pos += 1
            continue
        m = b[pos + 1]
        if m in (0xD8, 0x01) or 0xD0 <= m <= 0xD7:
            pos += 2
            continue
        if m == 0xDA:
            break
        ln = struct.unpack(">H", b[pos + 2:pos + 4])[0]
        seg = b[pos + 4:pos + 2 + ln]
        if m == 0xE2 and seg[:12] == b"ICC_PROFILE\x00":
            parts[seg[12]] = seg[14:]
        pos += 2 + ln
    return b"".join(parts[k] for k in sorted(parts)) if parts else None


def icc_tags(b):
    n = struct.unpack(">I", b[128:132])[0]
    t = {}
    for i in range(n):
        sig, ofs, sz = struct.unpack(">4sII", b[132 + i * 12:144 + i * 12])
        t[sig.decode("latin1")] = (ofs, sz)
    return t, n


def icc_desc(b):
    t, _ = icc_tags(b)
    if "desc" not in t:
        return None
    o = t["desc"][0]
    try:
        ac = struct.unpack(">I", b[o + 8:o + 12])[0]
        return b[o + 12:o + 12 + ac - 1].decode("latin1")
    except Exception:
        return None


# ---------------------------------------------------------------- Windows WCS
def wcs_check(data):
    try:
        mscms = ctypes.WinDLL("mscms")
    except Exception as e:
        return f"无法加载 mscms: {e}"

    class PROFILE(ctypes.Structure):
        _fields_ = [("dwType", wintypes.DWORD), ("pProfileData", ctypes.c_void_p),
                    ("cbDataSize", wintypes.DWORD)]

    mscms.OpenColorProfileW.argtypes = [ctypes.POINTER(PROFILE), wintypes.DWORD,
                                        wintypes.DWORD, wintypes.DWORD]
    mscms.OpenColorProfileW.restype = wintypes.HANDLE
    mscms.CloseColorProfile.argtypes = [wintypes.HANDLE]
    mscms.IsColorProfileValid.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL)]
    mscms.IsColorProfileValid.restype = wintypes.BOOL

    buf = ctypes.create_string_buffer(bytes(data), len(data))
    pr = PROFILE(2, ctypes.cast(buf, ctypes.c_void_p), len(data))
    h = mscms.OpenColorProfileW(ctypes.byref(pr), 1, 1, 0)
    if not h:
        return f"❌ Windows 打不开（GetLastError={ctypes.GetLastError()}）"
    v = wintypes.BOOL(False)
    mscms.IsColorProfileValid(h, ctypes.byref(v))
    mscms.CloseColorProfile(h)
    return f"{'✅ 有效' if v.value else '❌ 无效'}"


# ---------------------------------------------------------------- 主流程
def check(path):
    print("=" * 76)
    print(f"文件：{path}")
    print("=" * 76)
    if not os.path.exists(path):
        print("  ❌ 文件不存在")
        return
    b = open(path, "rb").read()
    print(f"  大小 {len(b)/1024:.1f} KB")

    ext = os.path.splitext(path)[1].lower()
    icc = iccname = exif = None
    if b[:8] == b"\x89PNG\r\n\x1a\n":
        icc, iccname = icc_from_png(b)
        exif = exif_from_png(b)
        chunks = [c[0] for c in png_chunks(b)]
        print(f"  格式 PNG   块顺序: IHDR {'iCCP' if icc else '(无iCCP)'} "
              f"{'eXIf' if exif else '(无eXIf)'} ... {len(chunks)} 个块")
    elif b[:2] == b"\xff\xd8":
        icc = icc_from_jpeg(b)
        exif = None
        pos = 2
        while pos + 4 <= len(b):
            if b[pos] != 0xFF:
                pos += 1
                continue
            m = b[pos + 1]
            if m in (0xD8, 0x01) or 0xD0 <= m <= 0xD7:
                pos += 2
                continue
            if m == 0xDA:
                break
            ln = struct.unpack(">H", b[pos + 2:pos + 4])[0]
            seg = b[pos + 4:pos + 2 + ln]
            if m == 0xE1 and seg[:6] == b"Exif\x00\x00":
                exif = seg[6:]
            pos += 2 + ln
        print(f"  格式 JPEG   ICC={'有' if icc else '无'}  EXIF={'有' if exif else '无'}")
    else:
        print("  格式 其它（TIFF/HEIF 请用 exiftool 或系统属性查看）")

    print()
    print("  ── ICC ──")
    if not icc:
        print("    ❌ 文件里**没有**嵌入 ICC profile")
        return
    print(f"    ✅ 嵌入 {len(icc)} 字节" + (f"（PNG iCCP 名: {iccname!r}）" if iccname else ""))
    d = icc_desc(icc)
    print(f"    档案名（软件会显示这个）: {d!r}")
    t, n = icc_tags(icc)
    print(f"    标签 {n} 个: {sorted(t)}")
    ver = struct.unpack(">I", icc[8:12])[0]
    print(f"    版本 {ver>>24}.{(ver>>20)&0xf}.{(ver>>16)&0xf}   "
          f"类别 {icc[12:16]!r}  空间 {icc[16:20]!r}  PCS {icc[20:24]!r}")
    if all(k in t for k in ("rXYZ", "gXYZ", "bXYZ")):
        s = [0.0, 0.0, 0.0]
        for k in ("rXYZ", "gXYZ", "bXYZ"):
            v = struct.unpack(">3i", icc[t[k][0] + 8:t[k][0] + 20])
            for j in range(3):
                s[j] += v[j] / 65536.0
        print(f"    色料和 {[round(x, 5) for x in s]}  (PCS 照明体 D50 应为 0.9642/1.0/0.8249)")
    if "rTRC" in t:
        o = t["rTRC"][0]
        typ = icc[o:o + 4].decode("latin1")
        if typ == "curv":
            cnt = struct.unpack(">I", icc[o + 8:o + 12])[0]
            extra = "恒等" if cnt == 0 else (f"{cnt} 点表" if cnt > 1 else
                                          f"gamma={struct.unpack(">H", icc[o+12:o+14])[0]/256.0}")
        elif typ == "para":
            fn = struct.unpack(">H", icc[o + 8:o + 10])[0]
            npar = {0: 1, 1: 3, 2: 4, 3: 5, 4: 7}.get(fn, 0)
            p = [v / 65536.0 for v in struct.unpack(">" + "i" * npar,
                                                    icc[o + 12:o + 12 + npar * 4])] if npar else []
            extra = f"类型{fn} 参数 {[round(x, 5) for x in p]}"
        else:
            extra = ""
        print(f"    TRC 类型 {typ!r}  {extra}")
    if "chad" in t:
        o, z = t["chad"]
        m = struct.unpack(">9i", icc[o + 8:o + 44])
        print(f"    chad 色适应矩阵 {z} 字节: " +
              " ".join(f"{v/65536.0:.4f}" for v in m[:3]) + " ...")
    print(f"    profileID {icc[84:100].hex() or '(全0)'}")
    print()
    print("  ── Windows 色彩引擎的判断 ──")
    print(f"    {wcs_check(icc)}")
    print()
    print("  ── EXIF ──")
    if not exif:
        print("    ❌ 没有 EXIF")
    else:
        try:
            from PIL import Image
            e = Image.Exif()
            e.load(exif)
            top = sorted(dict(e).keys())
            sub = sorted(dict(e.get_ifd(34665)).keys()) if e.get_ifd(34665) else []
            print(f"    ✅ {len(exif)} 字节  IFD0={top}")
            print(f"       子IFD(34665)={sub}")
            allt = dict(e)
            allt.update(dict(e.get_ifd(34665)) if e.get_ifd(34665) else {})
            for code, nm in ((271, "厂商"), (272, "机型"), (33434, "快门"), (33437, "光圈"),
                             (34855, "ISO"), (36867, "拍摄时间"), (42036, "镜头")):
                if code in allt:
                    print(f"       {nm}: {allt[code]}")
        except Exception as ex:
            print(f"    读取失败: {type(ex).__name__}: {ex}")

    print()
    print("  ── 与系统权威档案对照（littleCMS 互转，应接近恒等）──")
    try:
        from PIL import Image, ImageCms
        import numpy as np
        ours = ImageCms.getOpenProfile(io.BytesIO(icc))
        M = {"sRGB": "sRGB Color Space Profile.icm", "Adobe RGB": "AdobeRGB1998.icc",
             "ProPhoto RGB": "ProPhoto.icm"}
        hit = None
        for k, fn in M.items():
            p = os.path.join(D, fn)
            if d and k.lower().replace(" ", "") in d.lower().replace(" ", "") and os.path.exists(p):
                hit = (k, p)
                break
        if hit:
            ref = ImageCms.getOpenProfile(hit[1])
            tf = ImageCms.buildTransformFromOpenProfiles(ref, ours, "RGB", "RGB")
            pts = [(v, v, v) for v in range(0, 256, 17)]
            pts += [(255, 0, 0), (0, 255, 0), (0, 0, 255), (128, 64, 200)]
            im = Image.new("RGB", (len(pts), 1))
            im.putdata(pts)
            out = ImageCms.applyTransform(im, tf)
            a = np.frombuffer(im.tobytes(), np.uint8).astype(int)
            c = np.frombuffer(out.tobytes(), np.uint8).astype(int)
            print(f"    与 {hit[0]} 权威档案互转：最大差 {int(np.abs(a-c).max())} LSB")
        else:
            print("    该档案名在系统里没有对应的权威档案可对照")
    except Exception as e:
        print(f"    对照失败: {type(e).__name__}: {e}")
    print()


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print("用法：把图片文件拖到 check_icc.bat 上，或  python check_icc.py <图片>")
        print()
        for p in (r"F:\DS Harness\Color-Lab4.0\output",):
            if os.path.isdir(p):
                hits = [os.path.join(p, f) for f in os.listdir(p)
                        if os.path.splitext(f)[1].lower() in (".png", ".jpg", ".tif", ".tiff")]
                if hits:
                    print(f"（在 {p} 里找到 {len(hits)} 个文件，检查第一个）")
                    args = [hits[0]]
    if not args:
        sys.exit(1)
    for a in args:
        check(a)
