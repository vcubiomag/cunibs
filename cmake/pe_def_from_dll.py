"""Emit an MSVC module-definition (.def) file from a PE32+ DLL's export table.

Feeds the import-library synthesis in CunibsWindowsImplib: this script produces the .def, and
lib.exe turns it into a .lib. dumpbin cannot report whether an export is code or data.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pefile

# Only plain C identifiers become .def entries. A DLL that also exports mangled C++ internals
# ("?Foo@bar@@YA...") is not offering public API there, and '?' and '@' collide with the ordinal
# syntax of a .def file.
_C_IDENT = re.compile(rb"[A-Za-z_][A-Za-z0-9_]*")
_EXECUTE = pefile.SECTION_CHARACTERISTICS["IMAGE_SCN_MEM_EXECUTE"]


def render_def(dll: Path) -> str:
    pe = pefile.PE(str(dll), fast_load=True)
    if pe.FILE_HEADER.Machine != pefile.MACHINE_TYPE["IMAGE_FILE_MACHINE_AMD64"]:
        raise ValueError("not an x64 image")
    pe.parse_data_directories([pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_EXPORT"]])
    table = pe.DIRECTORY_ENTRY_EXPORT

    # LIBRARY stamps the imported DLL name into the .lib. Without it lib.exe derives the name from
    # the .lib basename, so cudart.lib would import from a nonexistent cudart.dll rather than from
    # cudart64_13.dll.
    lines = [f"LIBRARY {table.name.decode()}", "EXPORTS"]
    exports = [
        s for s in table.symbols if s.name and not s.forwarder and _C_IDENT.fullmatch(s.name)
    ]
    for symbol in sorted(exports, key=lambda s: s.name):
        section = pe.get_section_by_rva(symbol.address)
        is_code = section is not None and section.Characteristics & _EXECUTE
        # A data export declared without DATA gets a synthesised code thunk, so a reference to it
        # would read the thunk's instructions instead of the value.
        lines.append(f"    {symbol.name.decode()}{'' if is_code else ' DATA'}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dll", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    try:
        text = render_def(args.dll)
    except (OSError, ValueError, AttributeError, pefile.PEFormatError) as exc:
        print(f"could not read the exports of {args.dll}: {exc}", file=sys.stderr)
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="ascii")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
