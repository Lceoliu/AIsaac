"""只读核对 isaac-ng.exe 上的 Hook 目标字节与 PE 身份，不写游戏目录。

用法: python scripts/verify_targets.py [<isaac-ng.exe 路径>]
返回 0 = 全部匹配；1 = 不匹配；2 = 找不到文件。
"""
from __future__ import annotations

import json
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT_EXE = r"D:\Steam\steamapps\common\The Binding of Isaac Rebirth\isaac-ng.exe"


def read_pe(path: str):
    with open(path, "rb") as f:
        data = f.read()
    if data[:2] != b"MZ":
        raise ValueError("not a PE file")
    e_lfanew = struct.unpack_from("<I", data, 0x3C)[0]
    if data[e_lfanew:e_lfanew + 4] != b"PE\0\0":
        raise ValueError("bad NT signature")
    timestamp = struct.unpack_from("<I", data, e_lfanew + 8)[0]
    num_sections = struct.unpack_from("<H", data, e_lfanew + 6)[0]
    opt_size = struct.unpack_from("<H", data, e_lfanew + 20)[0]
    size_of_image = struct.unpack_from("<I", data, e_lfanew + 24 + 56)[0]
    sec_off = e_lfanew + 24 + opt_size
    sections = []
    for i in range(num_sections):
        off = sec_off + i * 40
        vsize, vaddr, rawsize, rawptr = struct.unpack_from("<IIII", data, off + 8)
        sections.append((vaddr, vsize, rawptr, rawsize))
    return data, timestamp, size_of_image, sections


def rva_to_bytes(data, sections, rva: int, n: int) -> bytes:
    for vaddr, vsize, rawptr, rawsize in sections:
        if vaddr <= rva < vaddr + max(vsize, rawsize):
            off = rawptr + (rva - vaddr)
            return data[off:off + n]
    raise ValueError(f"rva 0x{rva:08X} not in any section")


def main() -> int:
    exe = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_EXE
    if not os.path.isfile(exe):
        print(f"SKIP exe not found: {exe}")
        return 2
    with open(os.path.join(ROOT, "j460_targets.json"), encoding="utf-8") as f:
        spec = json.load(f)
    data, timestamp, size_of_image, sections = read_pe(exe)
    ok = True
    want_ts = int(spec["pe"]["timestamp"], 16)
    want_size = int(spec["pe"]["size_of_image"], 16)
    if timestamp != want_ts or size_of_image != want_size:
        print(f"FAIL pe identity: timestamp=0x{timestamp:08X} size=0x{size_of_image:08X} "
              f"(want 0x{want_ts:08X}/0x{want_size:08X})")
        ok = False
    for name, fn in spec["functions"].items():
        rva = int(fn["rva"], 16)
        tokens = fn["prologue"].split()
        got = rva_to_bytes(data, sections, rva, len(tokens))
        mismatch = [i for i, tok in enumerate(tokens) if tok != "??" and got[i] != int(tok, 16)]
        wild = sum(1 for tok in tokens if tok == "??")
        if mismatch:
            print(f"FAIL {name} @0x{rva:08X}: got {got.hex(' ')} mismatch_at={mismatch}")
            ok = False
        else:
            print(f"OK   {name} @0x{rva:08X} ({len(tokens)} bytes, {wild} relocatable skipped)")
    print("PASS all targets match" if ok else "FAIL target mismatch")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
