"""Persist and verify selection and downstream-model artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


ARTIFACT_FORMAT_VERSION = 1


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, default=_json_default))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json_dump(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)
            handle.write('\n')
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_numpy_save(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix='.npy', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            np.save(handle, array, allow_pickle=False)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_torch_save(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix='.pt', dir=path.parent)
    os.close(fd)
    try:
        torch.save(payload, tmp_name)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def save_selected_indices(
    indices: np.ndarray,
    output_dir: Path,
    metadata: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
) -> Path:
    """Save the ordered train-split indices and provenance metadata."""
    output_dir = Path(output_dir)
    selected = np.asarray(indices, dtype=np.int64).reshape(-1)
    if selected.size and selected.min() < 0:
        raise ValueError('Selected indices must be non-negative')
    if np.unique(selected).size != selected.size:
        raise ValueError('Selected indices contain duplicates')

    indices_path = output_dir / 'selected_indices.npy'
    _atomic_numpy_save(selected, indices_path)

    payload = dict(metadata or {})
    payload.update({
        'artifact_format_version': ARTIFACT_FORMAT_VERSION,
        'artifact_type': 'selected_train_indices',
        'index_semantics': 'zero-based indices into the official MedMNIST train split',
        'n_selected': int(selected.size),
        'indices_dtype': str(selected.dtype),
        'indices_sha256': _sha256(indices_path),
        'saved_at_utc': datetime.now(timezone.utc).isoformat(),
    })
    _atomic_json_dump(payload, output_dir / 'selection_metadata.json')

    if verbose:
        print(f"  [Artifacts] Saved {selected.size} selected indices to {indices_path}")
    return indices_path


def load_selected_indices(indices_path: Path, expected_sha256: Optional[str] = None) -> np.ndarray:
    """Load selected indices and optionally verify their file checksum."""
    indices_path = Path(indices_path)
    if expected_sha256 is not None and _sha256(indices_path) != expected_sha256:
        raise ValueError(f"Checksum mismatch for {indices_path}")
    return np.load(indices_path, allow_pickle=False)


def load_selection_metadata(metadata_path: Path) -> Dict[str, Any]:
    with Path(metadata_path).open('r', encoding='utf-8') as handle:
        return json.load(handle)


def save_final_model_checkpoint(
    model: torch.nn.Module,
    output_dir: Path,
    metadata: Dict[str, Any],
    verbose: bool = False,
) -> Path:
    """Save the final model used for the reported final test metrics.

    This is deliberately not called a best checkpoint: the released training
    loop records periodic test maxima but does not snapshot those weights.
    """
    output_dir = Path(output_dir)
    checkpoint_path = output_dir / 'final_model.pt'
    payload = {
        'artifact_format_version': ARTIFACT_FORMAT_VERSION,
        'artifact_type': 'final_inference_checkpoint',
        'model_state_dict': model.state_dict(),
        'metadata': _json_safe(metadata),
        'saved_at_utc': datetime.now(timezone.utc).isoformat(),
    }
    _atomic_torch_save(payload, checkpoint_path)

    checkpoint_metadata = dict(metadata)
    checkpoint_metadata.update({
        'artifact_format_version': ARTIFACT_FORMAT_VERSION,
        'artifact_type': 'final_inference_checkpoint',
        'checkpoint_sha256': _sha256(checkpoint_path),
        'contains_optimizer_state': False,
        'contains_scheduler_state': False,
        'saved_at_utc': datetime.now(timezone.utc).isoformat(),
    })
    _atomic_json_dump(checkpoint_metadata, output_dir / 'checkpoint_metadata.json')

    if verbose:
        print(f"  [Artifacts] Saved final model checkpoint to {checkpoint_path}")
    return checkpoint_path


def load_model_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Path,
    device: Optional[torch.device] = None,
) -> torch.nn.Module:
    """Load a structured checkpoint or a legacy raw state_dict."""
    map_location = device if device is not None else torch.device('cpu')
    try:
        checkpoint = torch.load(Path(checkpoint_path), map_location=map_location, weights_only=True)
    except TypeError:  # PyTorch versions before weights_only was introduced
        checkpoint = torch.load(Path(checkpoint_path), map_location=map_location)
    state_dict = checkpoint.get('model_state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    return model


def load_checkpoint_metadata(metadata_path: Path) -> Dict[str, Any]:
    with Path(metadata_path).open('r', encoding='utf-8') as handle:
        return json.load(handle)


def create_artifact_dir(
    base_dir: Path,
    dataset: str,
    ratio: float,
    method: str,
    embedding: str,
    seed: int,
    trial: int,
    include_importance: bool = False,
    importance: Optional[str] = None,
) -> Path:
    """Return a unique per-configuration directory inside one run directory."""
    ratio_basis_points = int(round(ratio * 10000))
    ratio_str = f"r{ratio_basis_points:04d}bp"
    method_str = f"{method}_{importance}" if include_importance and importance else method
    config_str = f"{method_str}_{embedding}_seed{seed}_trial{trial + 1:02d}"
    artifact_dir = Path(base_dir) / dataset / ratio_str / config_str
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir


def get_artifact_paths(artifact_dir: Path) -> Dict[str, Path]:
    artifact_dir = Path(artifact_dir)
    return {
        'indices': artifact_dir / 'selected_indices.npy',
        'selection_metadata': artifact_dir / 'selection_metadata.json',
        'model': artifact_dir / 'final_model.pt',
        'checkpoint_metadata': artifact_dir / 'checkpoint_metadata.json',
    }


def verify_artifacts(artifact_dir: Path) -> Dict[str, Any]:
    """Verify required files, checksums, and basic index invariants."""
    paths = get_artifact_paths(artifact_dir)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing artifact files: {missing}")

    selection_metadata = load_selection_metadata(paths['selection_metadata'])
    checkpoint_metadata = load_checkpoint_metadata(paths['checkpoint_metadata'])
    indices = load_selected_indices(paths['indices'], selection_metadata['indices_sha256'])
    if int(selection_metadata['n_selected']) != len(indices):
        raise ValueError('n_selected does not match selected_indices.npy')
    if checkpoint_metadata['selected_indices_sha256'] != selection_metadata['indices_sha256']:
        raise ValueError('Checkpoint and selection metadata refer to different selected indices')
    if _sha256(paths['model']) != checkpoint_metadata['checkpoint_sha256']:
        raise ValueError(f"Checksum mismatch for {paths['model']}")
    return {
        'valid': True,
        'n_selected': int(len(indices)),
        'indices_sha256': selection_metadata['indices_sha256'],
        'checkpoint_sha256': checkpoint_metadata['checkpoint_sha256'],
    }


def artifacts_exist(artifact_dir: Path) -> bool:
    return all(path.exists() for path in get_artifact_paths(artifact_dir).values())
