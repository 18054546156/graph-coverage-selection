# Artifact loading and verification

## Verify one configuration

```python
from pathlib import Path
from graphcov.run.artifacts import verify_artifacts

artifact_dir = Path(
    'results/runs/<run_id>/artifacts/pathmnist/r0500bp/'
    'graph_a2_uni_seed42_trial01'
)
print(verify_artifacts(artifact_dir))
```

Verification checks that all four files exist, both SHA256 values match, the
index count is correct, and the checkpoint refers to the same selected subset.

## Load the exact subset

```python
from graphcov.run.artifacts import load_selected_indices, load_selection_metadata

metadata = load_selection_metadata(artifact_dir / 'selection_metadata.json')
indices = load_selected_indices(
    artifact_dir / 'selected_indices.npy',
    expected_sha256=metadata['indices_sha256'],
)
subset = torch.utils.data.Subset(train_dataset, indices.tolist())
```

The indices address the official MedMNIST `train` split in its original order.
The metadata records package versions, class counts, full CLI/config, seed,
budget rounding, graph settings, and repository commit.

## Load the final ResNet-18

```python
import torch
from graphcov.run.artifacts import load_checkpoint_metadata, load_model_checkpoint
from graphcov.run.embeddings import ResNet18WithFeatures

meta = load_checkpoint_metadata(artifact_dir / 'checkpoint_metadata.json')
model = ResNet18WithFeatures(
    num_classes=meta['model']['num_classes'],
    in_channels=meta['model']['in_channels'],
    pretrained=False,
)
load_model_checkpoint(model, artifact_dir / 'final_model.pt', device=torch.device('cpu'))
model.eval()
```

The model expects images resized by MedMNIST+ to `image_size`, converted to a
tensor, and normalized with the means/stds in checkpoint metadata.

## Checkpoint semantics

`final_model.pt` is an inference checkpoint at the end of training. It contains
the state dict and reconstruction metadata, but not optimizer or scheduler
state, so it is not a bit-exact training-resume checkpoint.

The released code evaluates the test set every 10 epochs and records a periodic
maximum. Those weights are not saved here because calling them a validation-best
model would be incorrect. `periodic_test_maxima` remains in metadata only as an
audit field; `final_test_metrics` corresponds to the saved final weights.

## Remaining limits

- Exact UNI Hub revision and embedding-cache hash are not present in the
  authors' code. Sharing selected indices removes the need to regenerate the
  selection, but regenerating it may still depend on the downloaded UNI revision.
- The paper does not fully disclose `k` for every Table 1 dataset or the table's
  checkpoint policy. These are protocol-provenance gaps, not fixed by artifact
  persistence.
- MedMNIST data are not copied into the artifact directory; the official dataset
  must be installed separately.

