"""
Phase 15: Compute paired statistical tests on Phase 12 main scale results.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.statistical_tests import compare_paired_methods

def main() -> None:
    seeds = [42, 43, 44, 45, 46]
    methods = [
        "random",
        "utility_greedy",
        "sliding_window",
        "fixed_exploration",
        "page_hinckley_adaptive",
        "fedqual_cpx",
    ]
    finals: dict[str, list[float]] = {m: [] for m in methods}
    ginis: dict[str, list[float]] = {m: [] for m in methods}

    for m in methods:
        for s in seeds:
            path = Path(f"results/raw/main_class_swap_{m}_seed{s}/summary.json")
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    d = json.load(f)
                finals[m].append(float(d["final_accuracy"]))
                ginis[m].append(float(d["participation"]["gini"]))

    print("Loaded seeds count:", {m: len(v) for m, v in finals.items()})

    results = []
    for m in methods:
        if m == "fedqual_cpx":
            continue
        r_acc = compare_paired_methods(
            finals["fedqual_cpx"], finals[m],
            method_a_name="FedQual-CPX", method_b_name=m,
        )
        r_gini = compare_paired_methods(
            ginis["fedqual_cpx"], ginis[m],
            method_a_name="FedQual-CPX (Gini)", method_b_name=f"{m} (Gini)",
        )
        results.append({"metric": "final_accuracy", **r_acc})
        results.append({"metric": "gini_starvation", **r_gini})

    out_json = Path("results/tables/statistical_significance_main.json")
    out_csv = Path("results/tables/statistical_significance_main.csv")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    print("\n" + "=" * 80)
    print("PAIRED STATISTICAL SIGNIFICANCE TESTS (N_seeds=5, T=100 rounds)")
    print("=" * 80)
    for r in results:
        sig = "YES (p < 0.05)" if r["statistically_significant_05"] else "NO (p >= 0.05)"
        print(
            f"{r['metric']:<16} | {r['method_a']} vs {r['method_b']:<22} | "
            f"diff={r['mean_difference']:+.4f} | Cohen's d={r['cohens_d']:>6.2f} | "
            f"p={r['wilcoxon_p_value']:.4f} | Sig: {sig}"
        )
    print("=" * 80)
    print(f"Saved to {out_csv} and {out_json}")


if __name__ == "__main__":
    main()
