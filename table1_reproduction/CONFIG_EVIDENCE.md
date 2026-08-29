# Table 1 configuration evidence

This file separates paper-proven settings, public-code behavior, and local
evaluation safeguards. The README demo does not encode the complete Table 1
protocol.

## Paper-proven Graph-A2 configuration

```text
embedding = UNI, 224x224, 1024 dimensions
graph scope = global over all training embeddings
k = 50
H = 2
kernel = A_sym + A_sym^2
budgets = balanced per class
ratios = 0.02, 0.05
downstream = ResNet-18 from scratch
optimizer = SGD(lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = cosine annealing
augmentation = none
batch size = 256
training = 1000 epochs
trials = 5
metric = test balanced accuracy
```

The decisive evidence for `k=50` is the paper's Table 2 k-ablation. Its `k=50`
row reports:

| Dataset | 2% | 5% |
| --- | ---: | ---: |
| BloodMNIST | 84.5 +/- 1.7 | 93.4 +/- 0.7 |
| OrganSMNIST | 63.7 +/- 0.7 | 68.4 +/- 1.0 |

All four values exactly equal the corresponding Graph-A2/Ours entries in
Table 1. The previous `k=10` interpretation came from public parser and method
defaults; those defaults are not evidence of the paper-run setting.

The method section directly establishes a global k-NN graph and the two-hop
kernel `K = A_sym + A_sym^2`. The experimental setup establishes the remaining
embedding and downstream-training settings above.

## Public-code behavior retained

The vendored author snapshot is pinned to commit
`8cf757adc4c333dc1427d511f0de2f246d15ebac` and remains unmodified. Local
orchestration retains these observable runner behaviors:

- deterministic selectors reuse one frozen subset across training trials;
- Random and FPS vary the selection seed with the trial;
- the effective balanced quota is
  `floor(floor(n_train * ratio) / n_classes)` per class;
- dynamics methods share the repository's 28x28, 200-epoch dynamics cache.

The author README command uses 1000 iterations and default `k=10`. It is a demo,
not the paper's 1000-epoch Table 1 protocol.

## Facility scope is unresolved

The paper describes Facility Location on full pairwise cosine similarity, but
does not unambiguously say whether balanced Facility selection is global or
per-class. The public runner defaults to per-class Facility. Its CLI `--global`
flag applies to both Graph-A2 and Facility, so a shared flag would silently
change two methods at once.

The canonical config therefore sets global scope only on `graph_a2` and keeps
Facility per-class. This is an explicit implementation assumption, not a
paper-proven fact. A future global Facility reproduction requires a separate,
scalable implementation and a separately named output root.

## Strict local evaluation safeguard

The public author runtime repeatedly evaluates test during training and can
report the best observed test score. The isolated Job 2 intentionally uses a
stricter protocol:

1. train while reading validation only;
2. save `best_val_checkpoint.pt` at the best validation balanced accuracy;
3. reload that checkpoint after training;
4. load and evaluate test exactly once;
5. save `test_result.json`, `history.json`, and per-class diagnostics.

This prevents test-set model selection. It is a deliberate local correction,
not a claim that the public runner already behaved this way. Results from the
strict protocol must be reported as such when compared with the paper.

## Output isolation

Corrected outputs use:

```text
table1_reproduction/outputs/job1_selection_k50_global/
table1_reproduction/outputs/job2_table1_k50_global_valckpt/
```

Any older `job1_selection_clean`, `job2_table1_valckpt`, or Graph-A2 pilot
output was generated against the prior k=10 assumption and must not be merged
into the corrected summary.
