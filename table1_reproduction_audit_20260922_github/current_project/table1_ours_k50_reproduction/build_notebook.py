#!/usr/bin/env python3
"""Build a compact, data-driven Table 1 Ours reproduction notebook."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "reports" / "table1_ours_reproduction.ipynb"


def main() -> int:
    report = json.loads((ROOT / "reports" / "summary.json").read_text(encoding="utf-8"))
    protocol = json.loads((ROOT / "protocol.json").read_text(encoding="utf-8"))
    def markdown_cell(source: str) -> dict:
        return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}

    def code_cell(source: str) -> dict:
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source.splitlines(True),
        }

    cells = [
        markdown_cell(
            "# Graph-A2 / Ours: Table 1 复现\n\n"
            "本 Notebook 只补论文 Table 1 的 Ours 列。Ours 在作者代码中对应 `graph_a2`，协议固定为 "
            "UNI 224、global graph、k=50、H=2、equal class quota、ResNet-18 from scratch、1000 epochs。"
        ),
        code_cell(
            "from pathlib import Path\nimport json\nimport pandas as pd\n\n"
            f"ROOT = Path(r'{ROOT}')\n"
            "protocol = json.loads((ROOT / 'protocol.json').read_text(encoding='utf-8'))\n"
            "summary = json.loads((ROOT / 'reports' / 'summary.json').read_text(encoding='utf-8'))\n"
            "table = pd.read_csv(ROOT / 'reports' / 'ours_table1.csv')\n"
            "table"
        ),
        markdown_cell(
            "## 结果解释\n\n"
            "主指标是作者 runner 的 final `balanced_accuracy` 字段；`best_balanced_accuracy` 只作为审计字段，" 
            "不用于 Table 1 主表。每个条件应有 seeds 42--46 共 5 次训练。"
        ),
        code_cell(
            "table.assign(\n"
            "    reproduction = table.apply(lambda r: f\"{r.reproduction_mean_pct:.2f}+/-{r.reproduction_std_pct:.2f}\", axis=1),\n"
            "    paper = table.apply(lambda r: f\"{r.paper_mean_pct:.1f}+/-{r.paper_std_pct:.1f}\", axis=1),\n"
            ")[['dataset','ratio','n_trials','reproduction','paper','delta_pp','within_4pp']]"
        ),
        markdown_cell(
            "## 完成性审计\n\n"
            f"期望 `{report['expected_trials']}` 个 Graph-A2 trial，当前记录 `{report['actual_graph_a2_rows']}` 个；"
            f"缺失 `{len(report['missing_trials'])}` 个，重复行 `{report['duplicate_row_count']}` 个。\n\n"
            "选样数组保存在同目录 `selection/`，原始作者 runner 输出保存在 `results/raw/`。"
        ),
        code_cell(
            "pd.DataFrame(summary['selection_artifacts']).head(20)"
        ),
        markdown_cell(
            "## 与论文的边界\n\n"
            "这份结果用于闭合 Table 1 的 Ours 列，不改变七个非 Graph-A2 方法的复现结果。"
            "论文原始制表脚本未公开，因此仍应同时保留 `final` 与 `best-test` 字段；本表只使用 final。"
        ),
    ]
    nb = {
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK.parent.mkdir(exist_ok=True)
    NOTEBOOK.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(NOTEBOOK)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
