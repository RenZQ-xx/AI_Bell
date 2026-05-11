#!/usr/bin/env python3
"""Render the grouped 46x46 orbit-occupancy matrix HTML from its JSON summary."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = PROJECT_ROOT / "data" / "class46x46_grouped_min_occupancy_matrix_summary.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "class46x46_grouped_min_occupancy_matrix.html"

CLASS_ORDER = [
    1, 2, 3, 4, 5, 6,
    7, 8, 9, 10, 11, 12, 15, 19, 20, 22,
    13, 16, 24,
    14, 17, 21, 18, 23,
    28, 29, 30, 31,
    26, 27,
    32, 35,
    36, 37, 40,
    38, 39, 42, 43, 41, 44, 45, 46,
    25, 33, 34,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render data/class46x46_grouped_min_occupancy_matrix.html from its JSON summary."
    )
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def signature(groups: list[str]) -> str:
    return "<br>".join(html.escape(group) for group in groups)


def entry_text(entries: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for entry in entries:
        size = int(entry["orbit_size"])
        count = int(entry["orbit_count"])
        full = int(entry["full_occupied_orbits"])
        partial = int(entry["partial_occupied_orbits"])
        lines.append(
            f'{size}x{count}: <span class="full">F{full}</span> '
            f'<span class="partial">P{partial}</span>'
        )
    return "<br>".join(lines)


def is_pure_full(entries: list[dict[str, Any]]) -> bool:
    return all(int(entry["partial_occupied_orbits"]) == 0 for entry in entries)


def render(summary: dict[str, Any]) -> str:
    matrix = summary["matrix"]
    row_groups = summary["row_groups"]
    yellow_components = [set(int(value) for value in group) for group in summary["yellow_components"]]

    def same_yellow_component(row_class: int, col_class: int) -> bool:
        return any(row_class in group and col_class in group for group in yellow_components)

    headers = "".join(f"<th>{class_id}</th>" for class_id in CLASS_ORDER)
    rows: list[str] = []
    for row_class in CLASS_ORDER:
        cells = [
            f'<td class="row-label"><strong>{row_class}</strong><br>{signature(row_groups[str(row_class)])}</td>'
        ]
        for col_class in CLASS_ORDER:
            entries = matrix[str(row_class)][str(col_class)]
            classes = []
            if row_class == col_class:
                classes.append("diagonal")
            elif same_yellow_component(row_class, col_class):
                classes.append("same-block")
            elif is_pure_full(entries):
                classes.append("pure-full")
            class_attr = f' class="{" ".join(classes)}"' if classes else ""
            cells.append(f'<td{class_attr}><div class="cell">{entry_text(entries)}</div></td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")

    order_text = (
        "1-6 | 7-8-9-10-11-12-15-19-20-22 | 13-16-24 | "
        "14-17-21-18-23 | 28-29-30-31 | 26-27 | 32-35 | "
        "36-37-40 | 38-39-42-43-41-44-45-46 | 25-33-34"
    )
    semantics = summary.get("semantics", {})
    selection = html.escape(str(semantics.get("selection", "")))
    row_semantics = html.escape(str(semantics.get("rows", "")))
    col_semantics = html.escape(str(semantics.get("columns", "")))

    return f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>46×46 轨道分组最小占据矩阵</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 16px; }}
h1 {{ margin-bottom: 8px; }}
p {{ max-width: 1280px; }}
table {{ border-collapse: collapse; font-size: 12px; }}
th, td {{ border: 1px solid #999; padding: 4px 6px; vertical-align: top; }}
th {{ background: #f2f2f2; position: sticky; top: 0; z-index: 2; }}
th:first-child, td:first-child {{ position: sticky; left: 0; background: #fafafa; z-index: 1; }}
th:first-child {{ z-index: 3; }}
.cell {{ line-height: 1.35; white-space: nowrap; }}
.row-label {{ min-width: 120px; }}
.diagonal {{ background: #fff2a8; }}
.same-block {{ background: #fff2a8; }}
.pure-full {{ background: #e9f7e9; }}
.diagonal .cell {{ font-weight: 700; }}
.full {{ color: #1b7f3b; font-weight: 700; }}
.partial {{ color: #1d5fa7; font-weight: 700; }}
.legend {{ margin: 8px 0 12px; }}
.legend span {{ margin-right: 14px; }}
</style>
</head>
<body>
<h1>46×46 轨道分组最小占据矩阵</h1>
<p>行固定为各 class 的 rep1 stabilizer 轨道划分。列遍历该 class 在全群作用下的全部不同实例。每个格子优先追求 P0；在此基础上再最小化占据轨道数；若仍有并列，再优先整轨道占满数 F 更大。</p>
<p><strong>JSON semantics:</strong> rows = {row_semantics}; columns = {col_semantics}; selection = {selection}</p>
<div class="legend"><span><strong>F</strong> = 整轨道占满</span><span><strong>P</strong> = 仅部分占据</span><span>黄色 = 对角线或确认同块</span><span>浅绿 = P0</span><span>黄色非对角块: 1,2,3,4,5,6；7,8,9,10,11,12,15,19,20,22；13,16,24；14,17,21；28,29,30,31</span></div>
<p><strong>当前顺序:</strong> {order_text}</p>
<table>
<tr><th class="row-label">row \\ col</th>{headers}</tr>
{"\n".join(rows)}
</table>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(summary), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
