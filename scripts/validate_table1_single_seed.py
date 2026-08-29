"""Validate the PathMNIST/BloodMNIST single-seed Table 1 reproduction."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from graphcov.run.artifacts import verify_artifacts


PAPER = {
    ('pathmnist', 0.02): {
        'random': (77.5, 4.6), 'el2n_top': (36.8, 1.0),
        'forgetting': (53.5, 3.9), 'eva': (53.5, 5.7),
        'facility': (77.0, 1.6), 'fps': (73.0, 3.8),
        'herding': (78.0, 2.1), 'graph_a2': (80.9, 0.9),
    },
    ('pathmnist', 0.05): {
        'random': (84.0, 2.4), 'el2n_top': (49.9, 2.4),
        'forgetting': (58.9, 4.2), 'eva': (58.7, 3.5),
        'facility': (82.5, 2.2), 'fps': (83.0, 1.6),
        'herding': (84.6, 2.7), 'graph_a2': (85.9, 1.7),
    },
    ('bloodmnist', 0.02): {
        'random': (83.2, 1.5), 'el2n_top': (48.3, 6.9),
        'forgetting': (67.5, 3.3), 'eva': (75.9, 2.5),
        'facility': (82.7, 1.6), 'fps': (78.8, 2.1),
        'herding': (83.1, 3.0), 'graph_a2': (84.5, 1.7),
    },
    ('bloodmnist', 0.05): {
        'random': (90.3, 0.5), 'el2n_top': (72.5, 1.7),
        'forgetting': (81.4, 2.1), 'eva': (86.6, 1.7),
        'facility': (88.2, 2.7), 'fps': (89.0, 2.5),
        'herding': (92.3, 0.6), 'graph_a2': (93.4, 0.7),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_path = args.output / 'results.csv'
    if not results_path.exists():
        raise FileNotFoundError(results_path)

    results = pd.read_csv(results_path)
    results = results[
        results['dataset'].isin(['pathmnist', 'bloodmnist'])
        & results['ratio'].isin([0.02, 0.05])
        & (results['seed'] == args.seed)
    ].copy()
    expected_keys = {
        (dataset, ratio, method)
        for (dataset, ratio), methods in PAPER.items()
        for method in methods
    }
    actual_counts = results.groupby(['dataset', 'ratio', 'base_method']).size()
    duplicate_keys = [key for key, count in actual_counts.items() if count != 1]
    if duplicate_keys:
        raise ValueError(f'Duplicate configurations in results.csv: {duplicate_keys}')

    actual_keys = set(actual_counts.index)
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)

    rows = []
    artifact_errors = []
    for row in results.itertuples(index=False):
        key = (row.dataset, float(row.ratio))
        if key not in PAPER or row.base_method not in PAPER[key]:
            continue
        paper_mean, paper_std = PAPER[key][row.base_method]
        reproduced = float(row.balanced_accuracy) * 100.0
        delta = reproduced - paper_mean
        artifact_dir = args.output / Path(row.artifact_dir)
        artifact_ok = True
        try:
            verify_artifacts(artifact_dir)
        except Exception as exc:
            artifact_ok = False
            artifact_errors.append(f'{artifact_dir}: {exc}')
        rows.append({
            'dataset': row.dataset,
            'ratio': float(row.ratio),
            'method': row.base_method,
            'seed': int(row.seed),
            'reproduced_ba_pct': reproduced,
            'paper_mean_pct': paper_mean,
            'paper_std_pct': paper_std,
            'delta_pp': delta,
            'abs_delta_pp': abs(delta),
            'z_from_paper_mean': delta / paper_std,
            'within_paper_1std': abs(delta) <= paper_std,
            'artifact_ok': artifact_ok,
            'artifact_dir': str(artifact_dir),
        })

    comparison = pd.DataFrame(rows).sort_values(['dataset', 'ratio', 'method'])
    comparison_path = args.output / 'single_seed_comparison.csv'
    comparison.to_csv(comparison_path, index=False)

    complete = not missing and not unexpected and len(comparison) == 32
    artifacts_ok = len(artifact_errors) == 0 and len(comparison) == 32
    mean_abs_delta = comparison['abs_delta_pp'].mean() if len(comparison) else float('nan')
    inside = int(comparison['within_paper_1std'].sum()) if len(comparison) else 0

    report_lines = [
        '# Table 1 single-seed forensic reproduction',
        '',
        f'- Seed: {args.seed}',
        f'- Complete configurations: {len(comparison)}/32 ({complete})',
        f'- Verified artifact sets: {int(comparison.artifact_ok.sum()) if len(comparison) else 0}/32',
        f'- Mean absolute BA difference: {mean_abs_delta:.3f} pp',
        f'- Cells inside paper mean +/- one reported std: {inside}/32',
        '',
        'A single seed is a diagnostic comparison, not a reproduction of the paper mean or std.',
        '',
        '```text',
        comparison.to_string(index=False) if len(comparison) else 'No matching results.',
        '```',
    ]
    if missing:
        report_lines.extend(['', f'Missing: {missing}'])
    if unexpected:
        report_lines.extend(['', f'Unexpected: {unexpected}'])
    if artifact_errors:
        report_lines.extend(['', 'Artifact errors:', *[f'- {item}' for item in artifact_errors]])
    report_path = args.output / 'SINGLE_SEED_REPORT.md'
    report_path.write_text('\n'.join(report_lines) + '\n', encoding='utf-8')

    print(comparison[['dataset', 'ratio', 'method', 'reproduced_ba_pct',
                      'paper_mean_pct', 'paper_std_pct', 'delta_pp', 'artifact_ok']]
          .to_string(index=False))
    print(f'\nWrote {comparison_path}')
    print(f'Wrote {report_path}')
    if not complete or not artifacts_ok:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
