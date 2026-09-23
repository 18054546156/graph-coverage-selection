"""Merge one-seed real-screen JSONL outputs without duplicate selections."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--exclude-job-id", action="append", default=[],
        help="Ignore a stale job-id directory; may be repeated.",
    )
    args = p.parse_args()

    rows = {}
    excluded = set(args.exclude_job_id)
    files = [
        path for path in sorted(args.root.rglob("*.jsonl"))
        if not any(f"_{job_id}" in str(path.parent) for job_id in excluded)
    ]
    for path in files:
        with path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                dataset = row["dataset"]
                block = int(row["block"])
                local_index = int(row["screen_index"])
                global_index = int(row.get("global_screen_index", block * 225 + local_index))
                key = (dataset, global_index)
                if key in rows and rows[key]["_source"] != str(path):
                    raise RuntimeError(f"duplicate {key}: {rows[key]['_source']} and {path}")
                row["global_screen_index"] = global_index
                row["_source"] = str(path)
                rows[key] = row

    counts = defaultdict(int)
    for dataset, _ in rows:
        counts[dataset] += 1
    expected = 450
    bad = {dataset: n for dataset, n in counts.items() if n != expected}
    if bad:
        raise RuntimeError(f"incomplete datasets: {bad}; found {len(rows)} rows")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for key in sorted(rows):
            row = dict(rows[key])
            row.pop("_source", None)
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    print(json.dumps({"files": len(files), "rows": len(rows), "datasets": dict(sorted(counts.items())), "output": str(args.output)}))


if __name__ == "__main__":
    main()
