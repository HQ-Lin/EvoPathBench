#!/usr/bin/env python3
"""Print structural counts for an EvoPathBench release."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/finance"))
    args = parser.parse_args()
    episodes = list(rows(args.dataset / "episodes.jsonl"))
    streams = list(rows(args.dataset / "streams.jsonl"))
    summary = {
        "episode_count": len(episodes),
        "stream_count": len(streams),
        "episodes_by_family": Counter(item["family_id"] for item in episodes),
        "episodes_by_layer": Counter(item["layer"] for item in episodes),
        "episodes_by_role": Counter(item["role"] for item in episodes),
        "streams_by_template": Counter(item["template"] for item in streams),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
