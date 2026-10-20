#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""版本管理与回退。项目没有 VCS，所以用**快照 + 清单**保证每次改动都能退回。

    python tools/version.py list                 # 看有哪些快照
    python tools/version.py bump 4.1 --note "…"  # 改 APP_VERSION 并插入 CHANGELOG 标题
    python tools/version.py snapshot             # 按当前 APP_VERSION 打快照
    python tools/version.py rollback 4.1         # 回退（会先给当前状态打 safety 快照）
    python tools/version.py verify 4.1           # 校验快照完整性（sha256）

快照只收**应用代码**（colorlab*.py + 两个 bat）。测试与工具脚本不回退 ——
它们只会越改越好，回退反而会丢掉新断言。
"""
import sys
import json
import shutil
import hashlib
import argparse
import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VERS = ROOT / "versions"
CODE = ["colorlab.py", "colorlab_nb.py", "colorlab_gpu.py",
        "color_tool.bat", "setup.bat"]


def _sha(p):
    h = hashlib.sha256()
    h.update(Path(p).read_bytes())
    return h.hexdigest()[:16]


def _app_version():
    for line in (ROOT / "colorlab.py").read_text(encoding="utf-8").splitlines():
        if line.startswith("APP_VERSION"):
            return line.split("=", 1)[1].split("#")[0].strip().strip('"\'')
    return "?"


def cmd_list(_):
    if not VERS.exists():
        print("  还没有任何快照。")
        return
    print(f"  {'版本':<10}{'日期':<14}{'文件数':>6}  说明")
    print("  " + "-" * 74)
    for d in sorted(VERS.iterdir()):
        if not d.is_dir():
            continue
        mf = d / "manifest.json"
        if not mf.exists():
            print(f"  {d.name:<10}{'（无清单）':<14}")
            continue
        m = json.loads(mf.read_text(encoding="utf-8"))
        print(f"  {m['version']:<10}{m['date']:<14}{len(m['files']):>6}  {m.get('note','')[:40]}")
    cur = _app_version()
    print(f"\n  当前 APP_VERSION = {cur}")


def cmd_bump(a):
    new = a.version
    src = ROOT / "colorlab.py"
    s = src.read_text(encoding="utf-8")
    old = _app_version()
    lines = s.splitlines()
    for i, ln in enumerate(lines):
        if ln.startswith("APP_VERSION"):
            lines[i] = f'APP_VERSION = "{new}"   # 改动请同步写进 CHANGELOG.md（回归测试第 12 组会校验两者一致）'
            break
    src.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # CHANGELOG：新版本标题必须紧跟文档标题、**在第一个 `## ` 之前**。
    # 早先插在「第一条 ### 之前」—— 而那条 ### 属于**上一版**，
    # 于是新标题落进上一版的段落里，顺序成了 4.1, 4.2, …，
    # 「最新条目含 APP_VERSION」断言必然失败（4.1/4.2 两次都踩）。
    cl = ROOT / "CHANGELOG.md"
    c = cl.read_text(encoding="utf-8").splitlines()
    top1 = next((i for i, l in enumerate(c) if l.startswith("# ")), None)
    top2 = next((i for i, l in enumerate(c) if l.startswith("## ")), None)
    if top2 is None:
        raise SystemExit("CHANGELOG 里找不到 ## 标题")
    # 若已有同名版本标题，先移除（可重复 bump）
    if any(l.startswith(f"## {new} ") or l == f"## {new}" for l in c):
        j = next(i for i, l in enumerate(c)
                 if l.startswith(f"## {new} ") or l == f"## {new}")
        k = next((i for i in range(j + 1, len(c)) if c[i].startswith("## ")), len(c))
        c = c[:j] + c[k:]
        top2 = next(i for i, l in enumerate(c) if l.startswith("## "))
    insert_at = top2
    date = datetime.date.today().isoformat()
    hdr = [f"## {new} — {date}", ""]
    if a.note:
        hdr += [a.note, ""]
    c = c[:insert_at] + hdr + c[insert_at:]
    cl.write_text("\n".join(c) + "\n", encoding="utf-8")
    print(f"  APP_VERSION: {old} → {new}")
    print(f"  CHANGELOG: 已在文档标题之后、第一个 ## 之前插入 ## {new} — {date}")
    print(f"  接着跑： python tools/version.py snapshot")


def cmd_snapshot(a):
    ver = a.version or _app_version()
    d = VERS / ver
    if d.exists() and not a.force:
        raise SystemExit(f"  快照 {ver} 已存在（要覆盖加 --force）")
    d.mkdir(parents=True, exist_ok=True)
    files = {}
    for f in CODE:
        p = ROOT / f
        if not p.exists():
            continue
        shutil.copy2(p, d / f)
        files[f] = _sha(p)
    m = {"version": ver, "date": datetime.date.today().isoformat(),
         "note": a.note or "", "files": files}
    (d / "manifest.json").write_text(json.dumps(m, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
    print(f"  已打快照 versions/{ver}/  （{len(files)} 个文件）")
    for f, h in files.items():
        print(f"    {f:<20} {h}")


def cmd_verify(a):
    ver = a.version
    d = VERS / ver
    if not d.exists():
        raise SystemExit(f"  没有快照 {ver}")
    m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    bad = []
    for f, h in m["files"].items():
        p = d / f
        if not p.exists():
            bad.append((f, "缺失")); continue
        if _sha(p) != h:
            bad.append((f, "sha256 不符"))
    print(f"  versions/{ver}/  {len(m['files'])} 个文件")
    if bad:
        for f, why in bad:
            print(f"    ✗ {f}: {why}")
        raise SystemExit(1)
    print("    ✅ 全部一致")


def cmd_rollback(a):
    ver = a.version
    d = VERS / ver
    if not d.exists():
        raise SystemExit(f"  没有快照 {ver}")
    cur = _app_version()
    # 先给当前状态打一个 safety 快照，**绝不制造不可逆的改动**
    safety = f"{cur}-before-rollback-{datetime.datetime.now():%Y%m%d%H%M%S}"
    cmd_snapshot(argparse.Namespace(version=safety, note=f"回退到 {ver} 之前的状态",
                                    force=True))
    for f in CODE:
        p = d / f
        if p.exists():
            shutil.copy2(p, ROOT / f)
            print(f"  已恢复 {f}")
    print(f"\n  已从 {safety} 回退到 {ver}。若回退错了，可用：")
    print(f"      python tools/version.py rollback {safety}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("list"); p.set_defaults(fn=cmd_list)

    p = sub.add_parser("bump")
    p.add_argument("version")
    p.add_argument("--note", default="")
    p.set_defaults(fn=cmd_bump)

    p = sub.add_parser("snapshot")
    p.add_argument("version", nargs="?", default=None)
    p.add_argument("--note", default="")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_snapshot)

    p = sub.add_parser("rollback")
    p.add_argument("version")
    p.set_defaults(fn=cmd_rollback)

    p = sub.add_parser("verify")
    p.add_argument("version")
    p.set_defaults(fn=cmd_verify)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
