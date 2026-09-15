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

import argparse

def main() -> None:
    parser = argparse.ArgumentParser(description="Compute paired statistical tests on main scale results.")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Explicit list of seeds to evaluate (e.g. 42..51)")
    parser.add_argument("--drift-type", type=str, default="class_swap", choices=["class_swap", "feature_shift", "gradual_drift"])
    parser.add_argument("--methods", type=str, nargs="+", default=None, help="Methods to compare against FedQual-CPX")
    args = parser.parse_args()

    default_methods = [
        "random",
        "utility_greedy",
        "sliding_window",
        "fixed_exploration",
        "page_hinckley_adaptive",
        "fedqual_cpx",
        "oort",
    ]
    methods = args.methods if args.methods else default_methods
    
    # Auto-detect all valid seeds if not specified
    if args.seeds:
        target_seeds = list(args.seeds)
    else:
        found_seeds = set()
        for p in Path("results/raw").glob(f"main_{args.drift_type}_fedqual_cpx_seed*"):
            try:
                s_val = int(p.name.split("seed")[-1])
                found_seeds.add(s_val)
            except ValueError:
                pass
        target_seeds = sorted(list(found_seeds)) if found_seeds else [42, 43, 44, 45, 46]

    finals: dict[str, list[float]] = {m: [] for m in methods}
    ginis: dict[str, list[float]] = {m: [] for m in methods}

    for m in methods:
        for s in target_seeds:
            path = Path(f"results/raw/main_{args.drift_type}_{m}_seed{s}/summary.json")
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    d = json.load(f)
                # Guard: refuse to load short pilot runs
                if d.get("total_rounds", 0) < 50:
                    print(f"Warning: Skipping {path} because total_rounds={d.get('total_rounds', 0)} < 50.")
                    continue
                finals[m].append(float(d["final_accuracy"]))
                ginis[m].append(float(d.get("participation", {}).get("gini", 0.0)))

    print(f"Loaded seeds count for {args.drift_type} (Target seeds={target_seeds}):", {m: len(v) for m, v in finals.items()})

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
