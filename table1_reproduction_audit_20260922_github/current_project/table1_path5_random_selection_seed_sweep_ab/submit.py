#!/usr/bin/env python3
"""Idempotently submit preparation, five GPU runs, and aggregation."""

from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def run(command: list[str]) -> str:
    completed = subprocess.run(command, cwd=ROOT, check=True, text=True, capture_output=True)
    if completed.stderr.strip():
        print(completed.stderr.strip(), file=sys.stderr)
    return completed.stdout.strip()


def parse_job_id(output: str) -> str:
    match = re.search(r"Submitted batch job (\d+)", output)
    if not match:
        raise ValueError(f"could not parse Slurm job id from: {output!r}")
    return match.group(1)


def write_record(path: Path, record: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    shutil.move(temporary, path)


def main() -> int:
    record_path = ROOT / "submitted_jobs.json"
    if record_path.exists():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("summary"):
            print(json.dumps(record, indent=2, sort_keys=True))
            return 0
    else:
        record = {
            "schema": "graphcov-table1/path5-random-selection-seed-sweep-jobs-v1",
            "prepare": None,
            "train_array": None,
            "summary": None,
        }
    prepare_id = record.get("prepare")
    if not prepare_id:
        prepare_id = parse_job_id(run(["sbatch", "slurm/prepare.slurm"]))
        record["prepare"] = prepare_id
        write_record(record_path, record)
    train_id = record.get("train_array")
    if not train_id:
        train_id = parse_job_id(run(["sbatch", f"--dependency=afterok:{prepare_id}", "slurm/random_array.slurm"]))
        record["train_array"] = train_id
        write_record(record_path, record)
    summary_id = record.get("summary")
    if not summary_id:
        summary_id = parse_job_id(run(["sbatch", f"--dependency=afterok:{train_id}", "slurm/summary.slurm"]))
        record["summary"] = summary_id
        write_record(record_path, record)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
