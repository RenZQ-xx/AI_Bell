#!/usr/bin/env python3
"""Generate the static-baseline pattern discovery matrix HTML."""

from __future__ import annotations

import html
import json
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RUNS_DIR = ROOT / "src" / "baseline" / "runs"
OUT_PATH = ROOT / "src" / "baseline" / "summary" / "static_baseline_pattern_matrix.html"

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

GROUP_BREAK_AFTER = {6, 22, 24, 23, 31, 27, 35, 40, 46}


@dataclass(frozen=True)
class Experiment:
    pattern: str
    run_json: str
    target_classes: tuple[int, ...]
    rare_classes: tuple[int, ...]
    note: str = ""


EXPERIMENTS = [
    Experiment(
        "class1 pattern0 all46 focus1-6",
        "modular_strict_log_seed20260501_20260502_probe8_cf.json",
        tuple(CLASS_ORDER),
        (1, 2, 3, 4, 5, 6),
        "Restored class1 baseline probe8; rare reward focuses classes 1-6 while target_classes spans all 46 classes.",
    ),
    Experiment(
        "class7 pattern0 broad no43/44",
        "modular_strict_log_class7_pattern0_broad_no43_44_probe16.json",
        (7, 8, 9, 10, 11, 12, 15, 19, 20, 22, 25, 28, 29, 30, 31, 32, 34, 35, 38, 39, 40, 42, 43, 44, 45, 46),
        (7, 8, 9, 10, 11, 12, 15, 19, 20, 22, 25, 28, 29, 30, 31, 32, 34, 35, 38, 39, 40, 42, 45, 46),
        "Direct static reproduction target for the old supportability-gate class7 probe; 43/44 remain target classes but are excluded from rare reward.",
    ),
    Experiment(
        "class13 pattern0 static",
        "modular_strict_log_class13_pattern0_probe16.json",
        (13, 16, 24, 25, 33, 34, 40),
        (13, 16, 24, 25, 33, 34, 40),
        "Static baseline record. The old dynamic-terminal record hit 7/7; this static row hits 6/7 and misses class34.",
    ),
    Experiment(
        "class14 pattern0 static",
        "modular_strict_log_class14_pattern0_probe16.json",
        (14, 17, 21, 26, 27, 32, 33, 36, 37, 39, 43, 45),
        (14, 17, 21, 26, 27, 32, 33, 36, 37, 39, 43, 45),
    ),
    Experiment(
        "class18 pattern0 static",
        "modular_strict_log_class18_pattern0_probe16.json",
        (18, 26, 35, 36, 37, 39, 45, 46),
        (18, 26, 35, 36, 37, 39, 45, 46),
        "Both old dynamic and restored static runs hit 7/8 and miss class36.",
    ),
    Experiment(
        "class23 pattern0 static",
        "modular_strict_log_class23_pattern0_probe16.json",
        (23, 35, 38, 39, 41, 43, 44, 45, 46),
        (35, 38, 39, 41, 45, 46),
        "Class23 rare-tail baseline comparison row; 43/44 are target classes but not rare classes.",
    ),
    Experiment(
        "class28/29/30/31 pattern0 static",
        "modular_strict_log_class28_pattern0_probe16.json",
        (28, 29, 30, 31, 38, 42, 44, 46),
        (28, 29, 30, 31, 38, 42, 44, 46),
        "Classes 28, 29, 30, and 31 share this pattern0 orbit partition.",
    ),
    Experiment("class26 pattern0 static", "modular_strict_log_class26_pattern0_probe16.json", (26, 36, 37, 39), (26, 36, 37, 39)),
    Experiment("class27 pattern0 static", "modular_strict_log_class27_pattern0_probe16.json", (27, 45), (27, 45)),
    Experiment("class32 pattern0 static", "modular_strict_log_class32_pattern0_probe16.json", (32, 45), (32, 45)),
    Experiment("class35 pattern0 static", "modular_strict_log_class35_pattern0_probe16.json", (35, 45, 46), (35, 45, 46)),
    Experiment("class36 pattern0 static", "modular_strict_log_class36_pattern0_probe16.json", (36,), (36,)),
    Experiment("class37 pattern0 static", "modular_strict_log_class37_pattern0_probe16.json", (37,), (37,)),
    Experiment("class40 pattern0 static", "modular_strict_log_class40_pattern0_probe16.json", (40,), (40,)),
    Experiment("class38 pattern0 static", "modular_strict_log_class38_pattern0_probe16.json", (38, 44, 46), (38, 44, 46)),
    Experiment("class39 pattern0 static", "modular_strict_log_class39_pattern0_probe16.json", (39,), (39,)),
    Experiment("class42 pattern0 static", "modular_strict_log_class42_pattern0_probe16.json", (42, 46), (42, 46)),
    Experiment("class43 pattern0 static", "modular_strict_log_class43_pattern0_probe16.json", (43,), (43,)),
    Experiment("class41 pattern0 static", "modular_strict_log_class41_pattern0_probe16.json", (41,), (41,)),
    Experiment("class44 pattern0 static", "modular_strict_log_class44_pattern0_probe16.json", (44,), (44,)),
    Experiment("class45 pattern0 static", "modular_strict_log_class45_pattern0_probe16.json", (45,), (45,)),
    Experiment("class46 pattern0 static", "modular_strict_log_class46_pattern0_probe16.json", (46,), (46,)),
    Experiment("class25 pattern0 static", "modular_strict_log_class25_pattern0_probe16.json", (25, 40), (25, 40)),
    Experiment("class33 pattern0 static", "modular_strict_log_class33_pattern0_probe16.json", (33,), (33,)),
    Experiment("class34 pattern0 static", "modular_strict_log_class34_pattern0_probe16.json", (34,), (34,)),
]


def load_run(filename: str) -> dict:
    path = RUNS_DIR / filename
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def encountered_counts(summary: dict) -> dict[int, int]:
    counts: dict[int, int] = {}
    for label, value in summary.get("encountered_label_counts", {}).items():
        if not label.startswith("exact:"):
            continue
        class_text = label.split(":", 1)[1].replace("class", "")
        counts[int(class_text)] = int(value)
    return counts


def final_labels(summary: dict) -> str:
    label_counts = summary.get("label_counts", {})
    parts = []
    def sort_key(item: tuple[str, int]) -> tuple[int, str]:
        label, _ = item
        clean = label.replace("exact:class", "").replace("exact:", "")
        return (int(clean), label) if clean.isdigit() else (999, label)

    for label, count in sorted(label_counts.items(), key=sort_key):
        clean = label.replace("exact:class", "exact:").replace("exact:", "")
        parts.append(f"{clean}x{count}")
    return ", ".join(parts) if parts else "-"


def cell_html(exp: Experiment, cls: int, counts: dict[int, int]) -> str:
    target = cls in exp.target_classes
    rare = cls in exp.rare_classes
    hit_count = counts.get(cls, 0)
    group_class = " group-end" if cls in GROUP_BREAK_AFTER else ""
    if not target:
        return f'<td class="outside{group_class}"></td>'
    if hit_count:
        rare_mark = '<span class="rare">*</span>' if rare else ""
        title = f"class {cls}: hit {hit_count}; rare={str(rare).lower()}"
        return f'<td class="hit{group_class}" title="{html.escape(title)}"><div class="cell">{hit_count}{rare_mark}</div></td>'
    rare_mark = '<span class="rare">*</span>' if rare else ""
    title = f"class {cls}: missed; rare={str(rare).lower()}"
    return f'<td class="miss{group_class}" title="{html.escape(title)}"><div class="cell">miss{rare_mark}</div></td>'


def render() -> str:
    rows = []
    notes = []
    for exp in EXPERIMENTS:
        data = load_run(exp.run_json)
        counts = encountered_counts(data["summary"])
        hit_targets = [cls for cls in exp.target_classes if counts.get(cls, 0) > 0]
        coverage = f"{len(hit_targets)}/{len(exp.target_classes)}"
        label = html.escape(exp.pattern)
        final = html.escape(final_labels(data["summary"]))
        title = html.escape(f"final labels: {final_labels(data['summary'])}")
        row_cells = [
            f'<td class="pattern-col" title="{title}"><strong>{label}</strong></td>',
            f'<td class="coverage-col">{coverage}</td>',
        ]
        row_cells.extend(cell_html(exp, cls, counts) for cls in CLASS_ORDER)
        rows.append("<tr>" + "".join(row_cells) + "</tr>")
        note = exp.note or "Static baseline reproduction row."
        notes.append(
            "<tr>"
            f"<td>{html.escape(exp.pattern)}</td>"
            f'<td><code>../runs/{html.escape(exp.run_json)}</code></td>'
            f"<td>{html.escape(final_labels(data['summary']))}</td>"
            f"<td>{html.escape(note)}</td>"
            "</tr>"
        )

    class_headers = []
    for cls in CLASS_ORDER:
        group_class = " class=\"group-end\"" if cls in GROUP_BREAK_AFTER else ""
        class_headers.append(f"<th{group_class}>{cls}</th>")

    return f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>Static Baseline Pattern Discovery Matrix</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 16px; color: #222; }}
h1 {{ margin: 0 0 8px; }}
p {{ max-width: 1280px; line-height: 1.45; }}
.legend {{ margin: 8px 0 12px; }}
.legend span {{ margin-right: 14px; white-space: nowrap; }}
.table-wrap {{ overflow: auto; max-height: 78vh; border: 1px solid #999; }}
table {{ border-collapse: collapse; font-size: 12px; }}
th, td {{ border: 1px solid #999; padding: 4px 6px; text-align: center; vertical-align: middle; }}
th {{ background: #f2f2f2; position: sticky; top: 0; z-index: 3; }}
td.pattern-col, th.pattern-col {{ position: sticky; left: 0; z-index: 2; text-align: left; min-width: 220px; max-width: 220px; background: #fafafa; }}
td.coverage-col, th.coverage-col {{ position: sticky; left: 233px; z-index: 2; min-width: 74px; background: #fafafa; font-weight: 700; }}
th.pattern-col, th.coverage-col {{ z-index: 4; background: #ececec; }}
.cell {{ line-height: 1.25; white-space: nowrap; }}
.hit {{ background: #dff2df; color: #17672e; font-weight: 700; }}
.miss {{ background: #f8d7de; color: #9a2636; font-weight: 700; }}
.outside {{ background: #fff; color: #aaa; }}
.rare {{ color: #6f3fb0; font-weight: 800; padding-left: 1px; }}
.group-end {{ border-right: 3px solid #555; }}
.notes {{ margin-top: 18px; max-width: 1400px; }}
.notes table {{ width: 100%; font-size: 12px; }}
.notes th, .notes td {{ text-align: left; vertical-align: top; }}
code {{ background: #f5f5f5; padding: 1px 3px; border-radius: 3px; }}
</style>
</head>
<body>
<h1>Static Baseline Pattern Discovery Matrix</h1>
<p>行是合并后的代表 pattern 实验；列是 46 个 class，顺序沿用严格 46x46 矩阵。浅绿色表示该 class 在该 static baseline run 中被 exact encountered，粉色表示该 class 在该 pattern 的可表示/target 范围内但未命中，空白表示不在该行的可表示范围内。cell 数值为 <code>encountered_label_counts</code> 中的 exact 命中次数；<span class="rare">*</span> 表示该 class 在该 run 中被设置为 rare class。</p>
<div class="legend">
  <span><strong class="hit">绿色</strong> = hit</span>
  <span><strong class="miss">粉色</strong> = missed target</span>
  <span><span class="rare">*</span> = rare class</span>
  <span>当前顺序: 1-6 | 7-8-9-10-11-12-15-19-20-22 | 13-16-24 | 14-17-21-18-23 | 28-29-30-31 | 26-27 | 32-35 | 36-37-40 | 38-39-42-43-41-44-45-46 | 25-33-34</span>
</div>
<div class="table-wrap">
<table>
<tr><th class="pattern-col">Pattern</th><th class="coverage-col">Hit coverage</th>{"".join(class_headers)}</tr>
{"\n".join(rows)}
</table>
</div>
<div class="notes">
<h2>Run JSON 与说明</h2>
<table>
<tr><th>Pattern</th><th>Run JSON</th><th>Final labels</th><th>Note</th></tr>
{"\n".join(notes)}
</table>
</div>
</body>
</html>
"""


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(render(), encoding="utf-8")
    print(OUT_PATH)


if __name__ == "__main__":
    main()
