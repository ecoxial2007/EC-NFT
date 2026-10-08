#!/usr/bin/env python3
"""Construct evidence weights from saved attention without loading a model."""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ec_nft.evidence import build_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-ratio", type=float, default=0.1)
    parser.add_argument("--border-ratio", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--weight-min", type=float, default=0.5)
    parser.add_argument("--weight-max", type=float, default=1.5)
    args = parser.parse_args()
    cache = build_cache(args.sqlite, top_ratio=args.top_ratio, border_ratio=args.border_ratio,
                        beta=args.beta, minimum=args.weight_min, maximum=args.weight_max)
    target = Path(args.output).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + f".{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(cache, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"[EC-NFT] {cache['group_count']} mixed groups; {len(cache['records'])} trajectories")
    print(f"[EC-NFT] CPU attention cache: {target}")


if __name__ == "__main__":
    main()
