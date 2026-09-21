#!/usr/bin/env python3
"""Validate the public EvoPathBench dataset without network access."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
    return rows


def validate(root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    episodes = read_jsonl(root / "episodes.jsonl")
    streams = read_jsonl(root / "streams.jsonl")

    episode_ids = [row["episode_id"] for row in episodes]
    stream_ids = [row["stream_id"] for row in streams]
    if len(episode_ids) != len(set(episode_ids)):
        raise ValueError("duplicate episode_id values")
    if len(stream_ids) != len(set(stream_ids)):
        raise ValueError("duplicate stream_id values")
    if len(episodes) != int(manifest["scenario_count"]):
        raise ValueError("scenario_count does not match episodes.jsonl")
    if len(streams) != int(manifest["stream_count"]):
        raise ValueError("stream_count does not match streams.jsonl")

    known = set(episode_ids)
    missing: list[str] = []
    for stream in streams:
        for event in stream["events"]:
            if event["episode_id"] not in known:
                missing.append(event["episode_id"])
        for checkpoint in stream["checkpoints"]:
            for episode_id in checkpoint["probe_episode_ids"]:
                if episode_id not in known:
                    missing.append(episode_id)
    if missing:
        raise ValueError(f"{len(missing)} stream references are missing episodes")

    mismatches = []
    for relative, expected in manifest.get("sha256", {}).items():
        path = root / relative
        observed = sha256(path)
        if observed != expected:
            mismatches.append(relative)
    if mismatches:
        raise ValueError("checksum mismatch: " + ", ".join(mismatches))

    return {
        "status": "ok",
        "episodes": len(episodes),
        "streams": len(streams),
        "families": dict(sorted(Counter(row["family_id"] for row in episodes).items())),
        "templates": dict(sorted(Counter(row["template"] for row in streams).items())),
        "checked_files": len(manifest.get("sha256", {})),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/finance"))
    args = parser.parse_args()
    print(json.dumps(validate(args.dataset), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
