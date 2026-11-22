"""Install nvcc and cudart from NVIDIA's redistributable archives, checked against the manifest."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path
from urllib.request import urlopen

REDIST = "https://developer.download.nvidia.com/compute/cuda/redist"
COMPONENTS = ("cuda_nvcc", "cuda_cudart")


def unpack(data: bytes, prefix: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        if data.startswith(b"PK"):
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                z.extractall(tmp)  # noqa: S202
        else:
            with tarfile.open(fileobj=io.BytesIO(data)) as t:
                t.extractall(tmp, filter="data")
        (root,) = Path(tmp).iterdir()
        shutil.copytree(root, prefix, symlinks=True, dirs_exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", help="CTK release, e.g. 12.9.1")
    parser.add_argument("prefix", type=Path)
    args = parser.parse_args()

    platform = "windows-x86_64" if sys.platform == "win32" else "linux-x86_64"
    manifest = json.load(urlopen(f"{REDIST}/redistrib_{args.version}.json"))  # noqa: S310
    for name in COMPONENTS:
        entry = manifest[name][platform]
        data = urlopen(f"{REDIST}/{entry['relative_path']}").read()  # noqa: S310
        if hashlib.sha256(data).hexdigest() != entry["sha256"]:
            sys.exit(f"{entry['relative_path']}: sha256 does not match the manifest")
        unpack(data, args.prefix)
        print(f"{name} {manifest[name]['version']} -> {args.prefix}")
    # The archives ship lib/, but nvcc.profile on Linux links from lib64/.
    if platform == "linux-x86_64":
        (args.prefix / "lib").rename(args.prefix / "lib64")


if __name__ == "__main__":
    main()
