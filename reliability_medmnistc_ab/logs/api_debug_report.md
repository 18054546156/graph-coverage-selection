# MedMNIST-C API debug record

## Passed

- Clone at `8acfd2710c6e0e8b2745be8b1fa1c17b94b8f8a7` imported successfully.
- `DatasetManager`, `CorruptedMedMNIST`, and `Evaluator` imported successfully.
- A synthetic smoke package made from two official 28-pixel PathMNIST test images resized to 224 pixels exercised the unmodified `DatasetManager`.
- PathMNIST smoke generation produced all 11 configured corruption files, each with five severity blocks.
- `CorruptedMedMNIST` read every generated file and returned the expected RGB tensor shape.
- The source clone was restored to a clean Git worktree after execution.

The executed notebook and smoke summary are the authoritative records for this smoke test. They are not scientific robustness results.

## Compatibility findings

- The GraphCov Python environment had `medmnist==3.0.2`, no `opencv`, no `wand`, and no ImageMagick shared library. It was not modified.
- The isolated MedMNIST-C environment uses `medmnist==3.0.1`, NumPy `1.26.4`, OpenCV `4.10.0.84`, and Wand `0.7.2`.
- A NumPy 2.4.6 runtime fails in the unmodified repository at `np.fromstring(bytes, ...)` during MotionBlur. NumPy 1.26.4 runs the original call with a deprecation warning.
- ImageMagick is supplied by the isolated conda prefix; the login node has no system `MagickWand` library.

## Current status after the debug run

The earlier Zenodo download failure was later resolved. The official 224-pixel files were generated into the HPC experiment package, and Slurm job `31176` completed with exit code 0 and produced all 56 MedMNIST-C corruption files. This debug report still describes only the API smoke test; it is not a downstream robustness result.
