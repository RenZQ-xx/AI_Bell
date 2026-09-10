"""Compare compatibility-prior and uniform expansion outcomes in v4.1-v4.3."""
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "bucket_base", HERE / "analyze_expansion_buckets.py"
)
BASE_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE_MODULE)

OUT = HERE / "runs" / "v4_1_v4_2_v4_3_bucket_quality"


def main():
    OUT.mkdir(exist_ok=True)
    data = {version: BASE_MODULE.load(version) for version in ("v4_1", "v4_2", "v4_3")}
    (OUT / "data.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# v4.1 / v4.2 / v4.3 expansion bucket结果质量",
        "",
        "每轮只发生一次expansion，因此将该轮terminal结果归因于该次bucket。"
        "prior=bucket0；uniform=bucket1。该归因包含后续rollout影响。",
        "",
        "|版本|bucket组|样本|exact|exact率（95% CI）|类别数|非44 exact|新类别|boundary|non-coplanar|平均父rank|平均树深度|平均节点扩展序号|",
        "|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for version, result in data.items():
        for group in ("prior", "uniform"):
            stats = result["groups"][group]
            ci = stats["exact_ci95"]
            lines.append(
                f"|{version}|{group}|{stats['n']}|{stats['exact']}|"
                f"{stats['exact_rate']:.1%} ({ci[0]:.1%}–{ci[1]:.1%})|"
                f"{stats['class_count']}|{stats['non44']}|{stats['new_classes']}|"
                f"{stats['boundary']}|{stats['non_coplanar']}|"
                f"{stats['mean_parent_rank']:.2f}|{stats['mean_depth']:.2f}|"
                f"{stats['mean_node_expansion_ordinal']:.2f}|"
            )
        comparison = result["groups"]["comparison"]
        paired = result["paired_first_second"]
        lines += [
            "",
            f"{version}: prior-uniform exact率差 `{comparison['risk_difference']:.1%}`，"
            f"比值 `{comparison['risk_ratio']:.2f}`，Fisher双侧 `p={comparison['fisher_p']:.4f}`。",
            f"非class44 exact率为 `{comparison['non44_prior_rate']:.1%}` vs "
            f"`{comparison['non44_uniform_rate']:.1%}`，Fisher双侧 "
            f"`p={comparison['non44_fisher_p']:.4f}`。",
            f"同一父节点首个prior与第二个uniform配对：{paired['n']}对，"
            f"exact为{paired['prior_exact']} vs {paired['uniform_exact']}；"
            f"仅prior成功{paired['discordant_prior_only']}、仅uniform成功"
            f"{paired['discordant_uniform_only']}，McNemar精确p="
            f"{paired['mcnemar_exact_p']:.4f}。",
            "",
            "类别分布：",
            f"- prior: `{result['groups']['prior']['classes']}`",
            f"- uniform: `{result['groups']['uniform']['classes']}`",
            "",
        ]

    prior_n = sum(x["groups"]["prior"]["n"] for x in data.values())
    uniform_n = sum(x["groups"]["uniform"]["n"] for x in data.values())
    prior_exact = sum(x["groups"]["prior"]["exact"] for x in data.values())
    uniform_exact = sum(x["groups"]["uniform"]["exact"] for x in data.values())
    prior_non44 = sum(x["groups"]["prior"]["non44"] for x in data.values())
    uniform_non44 = sum(x["groups"]["uniform"]["non44"] for x in data.values())
    prior_only = sum(
        x["paired_first_second"]["discordant_prior_only"] for x in data.values()
    )
    uniform_only = sum(
        x["paired_first_second"]["discordant_uniform_only"] for x in data.values()
    )
    discordant = prior_only + uniform_only
    paired_p = min(
        1,
        2
        * sum(
            __import__("math").comb(discordant, k)
            for k in range(min(prior_only, uniform_only) + 1)
        )
        / (2**discordant),
    )
    lines += [
        "## 三版本合并",
        "",
        f"prior为{prior_exact}/{prior_n}（{prior_exact/prior_n:.1%}），uniform为"
        f"{uniform_exact}/{uniform_n}（{uniform_exact/uniform_n:.1%}）；Fisher双侧"
        f"p={BASE_MODULE.fisher_two_sided(prior_exact, prior_n-prior_exact, uniform_exact, uniform_n-uniform_exact):.4f}。",
        f"非class44 exact为{prior_non44}/{prior_n}（{prior_non44/prior_n:.1%}）对"
        f"{uniform_non44}/{uniform_n}（{uniform_non44/uniform_n:.1%}）；Fisher双侧"
        f"p={BASE_MODULE.fisher_two_sided(prior_non44, prior_n-prior_non44, uniform_non44, uniform_n-uniform_non44):.4f}。",
        f"合并同父节点配对中，仅prior成功{prior_only}、仅uniform成功{uniform_only}，"
        f"McNemar精确p={paired_p:.4f}。",
        "",
    ]

    lines += [
        "## 解释边界",
        "",
        "bucket并非随机处理：bucket0通常是节点第一次扩展，而uniform发生得更晚；"
        "两组父rank、树深度及节点扩展序号不同。Fisher检验只描述未分层汇总差异，"
        "不能证明prior本身造成差异。配对比较控制父节点，但仍把第一次与第二次扩展"
        "混在处理差异中。",
    ]
    report = "\n".join(lines) + "\n"
    (OUT / "README.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
