#!/usr/bin/env python3
"""Idempotently freeze inputs and submit the paired array plus aggregation."""

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
    output = completed.stdout.strip()
    if completed.stderr.strip():
        print(completed.stderr.strip(), file=sys.stderr)
    return output


def parse_job_id(output: str) -> str:
    match = re.search(r"Submitted batch job (\d+)", output)
    if not match:
        raise ValueError(f"could not parse Slurm job id from: {output!r}")
    return match.group(1)


def main() -> int:
    record_path = ROOT / "submitted_jobs.json"
    if record_path.exists():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("summary"):
            print(json.dumps(record, indent=2, sort_keys=True))
            return 0
    else:
        record = {
            "schema": "graphcov-table1/path5-fixed-selection-jobs-v1",
            "paired_array": None,
            "summary": None,
        }

    run(
        [
            sys.executable,
            "freeze_inputs.py",
            "--protocol",
            "protocol.json",
            "--output",
            "artifacts",
        ]
    )
    array_id = record.get("paired_array")
    if not array_id:
        array_id = parse_job_id(run(["sbatch", "slurm/paired_array.slurm"]))
        record["paired_array"] = array_id
        temporary = record_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        shutil.move(temporary, record_path)
    summary_id = parse_job_id(
        run(["sbatch", f"--dependency=afterok:{array_id}", "slurm/summary.slurm"])
    )
    record["summary"] = summary_id
    temporary = record_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    shutil.move(temporary, record_path)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
