#!/usr/bin/env python3
"""Generate release-level SHA-256 checksums."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


EXCLUDED_NAMES = {"SHA256SUMS", ".DS_Store"}
EXCLUDED_PARTS = {"__pycache__", ".git", ".venv"}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.root.resolve()
    paths = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.name not in EXCLUDED_NAMES
        and not EXCLUDED_PARTS.intersection(path.relative_to(root).parts)
    ]
    lines = [f"{digest(path)}  {path.relative_to(root).as_posix()}" for path in sorted(paths)]
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(lines)} checksums")


if __name__ == "__main__":
    main()
