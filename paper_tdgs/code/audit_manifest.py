"""Build a read-only audit manifest for the TDGS result artifacts.

The manifest is deliberately derived from files on disk. It does not query
Slurm and it does not infer completion from a worklist. A final paper should
archive its JSON output beside the exact harvest files used for the tables.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rows(path):
    with path.open(encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        raise ValueError(f"expected a list of rows: {path}")
    return rows


def summarize(rows):
    by_ratio = Counter(str(row.get("ratio")) for row in rows)
    by_arm = Counter(str(row.get("arm")) for row in rows)
    by_group = Counter(
        (str(row.get("ds")), str(row.get("ratio")), str(row.get("arm")))
        for row in rows
    )
    by_cell_seed = Counter(
        (str(row.get("ds")), str(row.get("ratio")), str(row.get("arm")),
         str(row.get("seed")))
        for row in rows
    )
    duplicate_keys = [list(key) + [count]
                      for key, count in by_cell_seed.items() if count > 1]
    return {
        "rows": len(rows),
        "by_ratio": dict(sorted(by_ratio.items())),
        "by_arm": dict(sorted(by_arm.items())),
        "dataset_ratio_arm_groups": len(by_group),
        "seeds_per_dataset_ratio_arm": {
            "|".join(key): count for key, count in sorted(by_group.items())
        },
        "duplicate_dataset_ratio_arm_seed": duplicate_keys,
        "seed_range": sorted({row.get("seed") for row in rows}),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--round2", required=True, type=Path)
    parser.add_argument("--round1", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    paths = [args.round2] + ([args.round1] if args.round1 else [])
    manifest = {
        "schema": "tdgs-audit-manifest-v1",
        "sources": [{"path": str(path), "sha256": sha256(path)}
                    for path in paths],
        "summaries": {path.name: summarize(load_rows(path)) for path in paths},
    }
    payload = json.dumps(manifest, indent=2, sort_keys=True)
    if args.out:
        args.out.write_text(payload + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(payload)


if __name__ == "__main__":
    main()
