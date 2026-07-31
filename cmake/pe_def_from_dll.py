"""Emit an MSVC module-definition (.def) file from a PE32+ DLL's export table.

Feeds the import-library synthesis in CunibsWindowsImplib: this script produces the
.def, and lib.exe turns it into a .lib.

The image is parsed directly rather than through dumpbin, which cannot report whether
an export is code or data, formats its output per version and locale, and is not a
sibling of CMAKE_AR in every Visual Studio layout.
"""

from __future__ import annotations

import argparse
import re
import struct
import sys
from pathlib import Path
from typing import NamedTuple

IMAGE_FILE_MACHINE_AMD64 = 0x8664
PE32PLUS_MAGIC = 0x20B
IMAGE_SCN_MEM_EXECUTE = 0x20000000
# Data directory 0 is the export table, at this offset into a PE32+ optional header.
EXPORT_DIRECTORY_OFFSET = 112

# Only plain C identifiers become .def entries. A DLL that also exports mangled C++
# internals ("?Foo@bar@@YA...") is not offering public API there, and '?' and '@'
# collide with the ordinal syntax of a .def file.
_C_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class Section(NamedTuple):
    vaddr: int
    vspan: int  # max(VirtualSize, SizeOfRawData): tolerate either convention
    rawptr: int
    chars: int


class Export(NamedTuple):
    name: str
    is_data: bool


def _sections(buf: bytes, opt: int, opt_size: int, count: int) -> list[Section]:
    table = opt + opt_size
    out = []
    for i in range(count):
        off = table + i * 40
        vsize, vaddr, rawsize, rawptr = struct.unpack_from("<IIII", buf, off + 8)
        (chars,) = struct.unpack_from("<I", buf, off + 36)
        out.append(Section(vaddr, max(vsize, rawsize), rawptr, chars))
    return out


def read_exports(buf: bytes) -> tuple[str, list[Export]]:
    """Return (the DLL name recorded in the image, its exported symbols)."""
    (e_lfanew,) = struct.unpack_from("<I", buf, 0x3C)
    if buf[e_lfanew : e_lfanew + 4] != b"PE\0\0":
        raise ValueError("not a PE image")

    coff = e_lfanew + 4
    machine, n_sections = struct.unpack_from("<HH", buf, coff)
    (opt_size,) = struct.unpack_from("<H", buf, coff + 16)
    if machine != IMAGE_FILE_MACHINE_AMD64:
        raise ValueError(f"machine 0x{machine:04x} is not x64")

    opt = coff + 20
    (magic,) = struct.unpack_from("<H", buf, opt)
    if magic != PE32PLUS_MAGIC:
        raise ValueError(f"optional header magic 0x{magic:04x} is not PE32+")

    exp_rva, exp_size = struct.unpack_from("<II", buf, opt + EXPORT_DIRECTORY_OFFSET)
    if not exp_rva:
        raise ValueError("no export directory")

    sections = _sections(buf, opt, opt_size, n_sections)

    def section_of(rva: int) -> Section | None:
        return next((s for s in sections if s.vaddr <= rva < s.vaddr + s.vspan), None)

    def offset_of(rva: int) -> int:
        section = section_of(rva)
        if section is None:
            raise ValueError(f"RVA 0x{rva:x} falls outside every section")
        return section.rawptr + (rva - section.vaddr)

    def cstr(rva: int) -> str:
        start = offset_of(rva)
        return buf[start : buf.index(b"\0", start)].decode("ascii")

    # IMAGE_EXPORT_DIRECTORY has eleven fields. Unpacking ten shifts every RVA by
    # one slot and yields plausible-looking garbage rather than an error.
    (
        _characteristics,
        _timestamp,
        _major,
        _minor,
        name_rva,
        _ordinal_base,
        _n_functions,
        n_names,
        addr_rva,
        names_rva,
        ordinals_rva,
    ) = struct.unpack_from("<IIHHIIIIIII", buf, offset_of(exp_rva))

    addr_off = offset_of(addr_rva)
    names_off = offset_of(names_rva)
    ordinals_off = offset_of(ordinals_rva)

    exports: list[Export] = []
    for i in range(n_names):
        (symbol_rva,) = struct.unpack_from("<I", buf, names_off + 4 * i)
        (index,) = struct.unpack_from("<H", buf, ordinals_off + 2 * i)  # already 0-based
        (func_rva,) = struct.unpack_from("<I", buf, addr_off + 4 * index)
        if exp_rva <= func_rva < exp_rva + exp_size:
            continue  # a forwarder, not a real export
        section = section_of(func_rva)
        is_data = not (section and section.chars & IMAGE_SCN_MEM_EXECUTE)
        exports.append(Export(cstr(symbol_rva), is_data))

    return cstr(name_rva), exports


def render_def(dll_name: str, exports: list[Export]) -> str:
    # LIBRARY stamps the imported DLL name into the .lib. Without it lib.exe derives
    # the name from the .lib basename, so cudart.lib would import from a nonexistent
    # cudart.dll rather than from cudart64_13.dll.
    lines = [f"LIBRARY {dll_name}", "EXPORTS"]
    for export in sorted(exports, key=lambda e: e.name):
        if not _C_IDENT.fullmatch(export.name):
            continue
        # A data export declared without DATA gets a synthesised code thunk, so a
        # reference to it would read the thunk's instructions instead of the value.
        lines.append(f"    {export.name}{' DATA' if export.is_data else ''}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dll", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    try:
        dll_name, exports = read_exports(args.dll.read_bytes())
    except (OSError, ValueError, struct.error) as exc:
        print(f"could not read the exports of {args.dll}: {exc}", file=sys.stderr)
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_def(dll_name, exports), encoding="ascii")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
