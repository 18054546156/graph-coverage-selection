# Changes relative to the released repository

The extension changes artifact persistence only. It does not change sample
scores, selected-index order, optimizer steps, scheduler steps, or test metrics.

## Code changes

- `graphcov/run/artifacts.py`
  - atomically saves ordered `int64` indices and SHA256 metadata;
  - saves a self-describing final-model checkpoint;
  - loads legacy raw state dicts and new structured checkpoints;
  - verifies index/checkpoint checksums and their linkage.
- `graphcov/run/evaluation.py`
  - adds opt-in `return_model=False`;
  - preserves the official five-item return value by default, so
    `compare_k.py`, `compare_global.py`, and `compare_sizes.py` still work.
- `graphcov/run/experiment.py`
  - requests the final model only in the main runner;
  - writes artifacts under the current run directory;
  - records dataset, selection, model, training, software, and git provenance;
  - links every result row to its artifact directory.
- `graphcov/run/results.py`
  - uses microseconds in run IDs to prevent same-second run collisions.
- `graphcov/run/evaluation.py`, `embeddings.py`, and `eva.py`
  - cast MedMNIST labels to `torch.int64` before CrossEntropy/one-hot indexing;
  - this fixes a runtime incompatibility with MedMNIST versions returning
    `int32` labels and does not change the mathematical objective.

## Deliberate semantics

The saved file is `final_model.pt`, not `best_model.pt`. The official loop tracks
the best balanced accuracy observed on test but never snapshots best weights.
Saving the live model under a best-model name would make the checkpoint and
metadata disagree.

Artifact write or checksum failures fail that configuration before a result row
is accepted. This is intentional for formal reproducibility runs: a metric
without its promised artifacts is incomplete.

## Protocol correction

The formal commands are split because `--global` also changes Facility. Graph-A2
is run globally; the Facility baseline is run without `--global`. See
`TABLE1_REPRODUCTION.md`.
