import json
from pathlib import Path

import numpy as np
import pytest
import torch

from graphcov.run.artifacts import (
    create_artifact_dir,
    load_model_checkpoint,
    load_selected_indices,
    save_final_model_checkpoint,
    save_selected_indices,
    verify_artifacts,
)


def test_artifact_paths_are_unique_by_run_and_trial(tmp_path: Path):
    run_a = tmp_path / 'runs' / 'run_a' / 'artifacts'
    run_b = tmp_path / 'runs' / 'run_b' / 'artifacts'
    kwargs = dict(dataset='pathmnist', ratio=0.05, method='graph_a2', embedding='uni')

    a1 = create_artifact_dir(run_a, seed=42, trial=0, **kwargs)
    a2 = create_artifact_dir(run_a, seed=43, trial=1, **kwargs)
    b1 = create_artifact_dir(run_b, seed=42, trial=0, **kwargs)

    assert a1 != a2
    assert a1 != b1
    assert a1.name.endswith('seed42_trial01')
    assert a2.name.endswith('seed43_trial02')


def test_indices_and_checkpoint_round_trip(tmp_path: Path):
    artifact_dir = create_artifact_dir(
        tmp_path / 'artifacts', 'bloodmnist', 0.02, 'random', 'none', 42, 0
    )
    indices = np.array([7, 2, 11], dtype=np.int64)
    save_selected_indices(indices, artifact_dir, {'dataset': 'bloodmnist'})
    selection_metadata = json.loads((artifact_dir / 'selection_metadata.json').read_text())

    torch.manual_seed(42)
    model = torch.nn.Linear(4, 3)
    inputs = torch.randn(2, 4)
    expected = model(inputs).detach()
    save_final_model_checkpoint(
        model,
        artifact_dir,
        {
            'selected_indices_sha256': selection_metadata['indices_sha256'],
            'model': {'architecture': 'torch.nn.Linear', 'in_features': 4, 'out_features': 3},
            'final_test_metrics': {'balanced_accuracy': 0.5},
            'full_config': {'output_dir': tmp_path},
        },
    )

    restored = torch.nn.Linear(4, 3)
    load_model_checkpoint(restored, artifact_dir / 'final_model.pt')

    assert np.array_equal(load_selected_indices(artifact_dir / 'selected_indices.npy'), indices)
    assert torch.equal(restored(inputs), expected)
    assert verify_artifacts(artifact_dir)['valid'] is True


def test_duplicate_indices_are_rejected(tmp_path: Path):
    with pytest.raises(ValueError, match='duplicates'):
        save_selected_indices(np.array([1, 1]), tmp_path)
