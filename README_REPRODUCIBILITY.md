# Graph Coverage Selection: reproducibility extension

This directory is a clone of the authors' released repository with artifact
persistence added. The selection objective and downstream training algorithm
are unchanged.

Each completed configuration now stores:

```text
<output>/runs/<run_id>/artifacts/<dataset>/r<ratio-bp>/
  <method>_<embedding>_seed<seed>_trial<trial>/
    selected_indices.npy
    selection_metadata.json
    final_model.pt
    checkpoint_metadata.json
```

`selected_indices.npy` contains ordered, zero-based indices into the official
MedMNIST training split. `final_model.pt` contains the ResNet-18 weights after
the configured training budget. It is intentionally not named `best_model.pt`:
the released code records periodic test maxima but does not retain those
weights.

Start with a one-epoch smoke test:

```bash
python -m graphcov.run \
  --datasets bloodmnist \
  --methods random \
  --ratios 0.02 \
  --trials 1 --seed 42 \
  --training-paradigm epoch --epochs 1 \
  --test-every-n-epochs 1 --batch-size 256 \
  --size 28 --num-workers 0 \
  --output results/artifact_smoke
```

Then verify the generated files:

```bash
python -m pytest -q tests/test_artifacts.py
```

See `TABLE1_REPRODUCTION.md` for the formal command split and unresolved paper
configuration ambiguities. Do not launch the full 400-training matrix until the
smoke checkpoint reloads and its checksum passes.

