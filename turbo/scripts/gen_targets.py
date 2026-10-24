"""j460_targets.json -> src/generated/j460_targets.hpp

prologue 里的 `??` 表示含绝对地址、会被 ASLR 重定位的字节：生成掩码 0x00，比对时跳过。

用法: python scripts/gen_targets.py [--check]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
JSON_PATH = os.path.join(ROOT, "j460_targets.json")
OUT_PATH = os.path.join(ROOT, "src", "generated", "j460_targets.hpp")


def parse_prologue(text: str):
    values, mask = [], []
    for tok in text.split():
        if tok == "??":
            values.append(0)
            mask.append(0)
        else:
            values.append(int(tok, 16))
            mask.append(0xFF)
    return values, mask


def render(spec: dict) -> str:
    pe = spec["pe"]
    lines = [
        "// 由 scripts/gen_targets.py 从 j460_targets.json 生成，不要手改。",
        "#pragma once",
        "#include <cstddef>",
        "#include <cstdint>",
        "",
        "namespace isaac_turbo::j460 {",
        "",
        f"constexpr std::uint32_t kPeTimestamp = {pe['timestamp']}u;",
        f"constexpr std::uint32_t kSizeOfImage = {pe['size_of_image']}u;",
        "",
        "struct FunctionTarget {",
        "  const char* name;",
        "  std::uint32_t rva;",
        "  const std::uint8_t* prologue;",
        "  const std::uint8_t* mask;  // 0x00 = 该字节含重定位地址，不比对",
        "  std::size_t prologue_bytes;",
        "};",
        "",
    ]
    names = []
    for key, fn in spec["functions"].items():
        values, mask = parse_prologue(fn["prologue"])
        ident = "k" + "".join(part.capitalize() for part in key.split("_"))
        names.append((ident, key))
        lines.append(f"// {fn['abi']}")
        lines.append(f"constexpr std::uint8_t {ident}Prologue[{len(values)}] = {{"
                     + ", ".join(f"0x{b:02X}" for b in values) + "};")
        lines.append(f"constexpr std::uint8_t {ident}Mask[{len(mask)}] = {{"
                     + ", ".join(f"0x{b:02X}" for b in mask) + "};")
        lines.append(f"constexpr std::uint32_t {ident}Rva = {fn['rva']}u;")
        lines.append("")
    lines.append("constexpr FunctionTarget kFunctionTargets[] = {")
    for ident, key in names:
        lines.append(f'    {{"{key}", {ident}Rva, {ident}Prologue, {ident}Mask, sizeof({ident}Prologue)}},')
    lines.append("};")
    lines.append("")
    for key, d in spec["data"].items():
        ident = "k" + "".join(part.capitalize() for part in key.split("_"))
        if "rva" in d:
            lines.append(f"constexpr std::uint32_t {ident}Rva = {d['rva']}u;  // {d['note']}")
        else:
            lines.append(f"constexpr std::uint32_t {ident}Offset = {d['offset']}u;  // {d['note']}")
    lines.append("")
    lines.append("}  // namespace isaac_turbo::j460")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只比较，不写文件")
    args = ap.parse_args()
    with open(JSON_PATH, encoding="utf-8") as f:
        spec = json.load(f)
    text = render(spec)
    if args.check:
        try:
            with open(OUT_PATH, encoding="utf-8") as f:
                current = f.read()
        except FileNotFoundError:
            current = ""
        if current != text:
            print("generated header drifted:", OUT_PATH, file=sys.stderr)
            return 1
        print("generated header up to date")
        return 0
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print("wrote", OUT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
