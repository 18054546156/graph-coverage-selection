"""Evaluate existing final checkpoints on clean and corrupted test data."""

from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path = [entry for entry in sys.path if Path(entry or ".").resolve() != SCRIPT_DIR]
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from reliability_medmnistc.scripts.run_pipeline import main


if __name__ == "__main__":
    raise SystemExit(main("evaluate"))
