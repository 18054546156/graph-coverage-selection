"""S2 wrapper: run the unmodified pipeline with a colour-only training transform.

graphcov/ must stay byte-identical (the pipeline asserts it), so the colour
augmentation is injected at runtime: graphcov.run.data.get_train_transform is
replaced in sys.modules BEFORE pipeline.py does
`from graphcov.run.data import get_train_transform`. With `augment: true` the
pipeline wraps the train set in AugmentedDataset(get_train_transform(...)), so
this transform is the ONLY change vs the original runs. No crop/flip: the
original runs used augment=false, so adding spatial augmentation would confound
colour with geometry.

Usage (same as run_pipeline.py): python s2_colour_pipeline.py --config X.yaml --phase full
"""
import os
import runpy
import sys

SOURCE_ROOT = os.environ["GRAPH_ROOT"]
sys.path.insert(0, SOURCE_ROOT)

from torchvision import transforms  # noqa: E402
import graphcov.run.data as graphcov_data  # noqa: E402

COLOUR_JITTER = dict(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)


def colour_train_transform(in_channels, size=224):
    if in_channels == 3:
        jitter = transforms.ColorJitter(**COLOUR_JITTER)
    else:
        jitter = transforms.ColorJitter(brightness=COLOUR_JITTER["brightness"],
                                        contrast=COLOUR_JITTER["contrast"])
    return transforms.Compose([
        jitter,
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * in_channels, std=[0.5] * in_channels),
    ])


graphcov_data.get_train_transform = colour_train_transform
print("[s2] colour-only train transform installed: ColorJitter({})".format(COLOUR_JITTER), flush=True)

runner = os.path.join(SOURCE_ROOT, "reliability_medmnistc", "scripts", "run_pipeline.py")
sys.argv = [runner] + sys.argv[1:]
runpy.run_path(runner, run_name="__main__")
