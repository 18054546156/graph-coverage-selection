"""Pin Table 1 jobs to the vendored upstream GraphCov implementation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENDOR_ROOT = Path(__file__).resolve().parent / "vendor"
MANIFEST_PATH = VENDOR_ROOT / "official_manifest.json"
SOURCE_COMMIT = "8cf757adc4c333dc1427d511f0de2f246d15ebac"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_official_vendor() -> None:
    """Fail closed if the Table 1 implementation snapshot was changed."""
    if not MANIFEST_PATH.exists():
        raise RuntimeError(f"missing Table 1 vendor manifest: {MANIFEST_PATH}")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("source_commit") != SOURCE_COMMIT:
        raise RuntimeError(
            "Table 1 vendor source commit mismatch: "
            f"{manifest.get('source_commit')} != {SOURCE_COMMIT}"
        )

    expected = {
        str(item["path"]): str(item["sha256"])
        for item in manifest.get("files", [])
    }
    actual_paths = {
        path.relative_to(VENDOR_ROOT).as_posix()
        for path in VENDOR_ROOT.rglob("*")
        if (
            path.is_file()
            and path.name != MANIFEST_PATH.name
            and "__pycache__" not in path.parts
            and path.suffix != ".pyc"
        )
    }
    if actual_paths != set(expected):
        missing = sorted(set(expected) - actual_paths)
        extra = sorted(actual_paths - set(expected))
        raise RuntimeError(f"Table 1 vendor file set mismatch: missing={missing}, extra={extra}")

    mismatches = []
    for relative_path, expected_hash in expected.items():
        path = VENDOR_ROOT / relative_path
        actual_hash = _sha256(path)
        if actual_hash != expected_hash:
            mismatches.append(f"{relative_path}: {actual_hash} != {expected_hash}")
    if mismatches:
        raise RuntimeError("Table 1 vendor hash mismatch:\n" + "\n".join(mismatches))


def configure_official_runtime() -> Path:
    """Verify and prepend the immutable vendor root before importing graphcov."""
    verify_official_vendor()
    loaded = sys.modules.get("graphcov")
    if loaded is not None:
        raise RuntimeError("graphcov was imported before Table 1 vendor isolation was configured")
    vendor_root = str(VENDOR_ROOT)
    if vendor_root in sys.path:
        sys.path.remove(vendor_root)
    sys.path.insert(0, vendor_root)
    return VENDOR_ROOT / "graphcov"
