"""Experiments 1-5 for each shortlist (top, banded, balanced).

Tables + plots go to results/<shortlist>/, plus runs_log.csv with every
single GA / random search run.

Tuning is one factor at a time from the defaults (pop 50, 100 gens, 5%
mutation, £3m) - basically the simplest version of what Eiben & Smit
(2011) recommend.

Every stochastic setup runs N_RUNS = 30 times, seeds 0..29, fresh rng per
run, so runs can be paired across settings. Greedy and exhaustive are
deterministic so they only run once.

NB the 30 repeats in exp 5 are about how reliable the GA is, not about how
uncertain the land prices / GDVs are. The scenarios are for that.

python experiments.py [top|banded|balanced]   (nothing = all three)
"""
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from comparison_methods import exhaustive_search, greedy_selection, random_search, random_search_repair
from genetic_algorithm import run_ga
from prepare_data import load_projects

BUDGET = 3_000_000
N_RUNS = 30

# colours always in this order so a method keeps its colour across charts
COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK = "#0b0b0b"
INK_SOFT = "#52514e"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": INK_SOFT,
    "axes.labelcolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#e4e3df",
    "grid.linewidth": 0.8,
    "xtick.color": INK_SOFT,
    "ytick.color": INK_SOFT,
    "lines.linewidth": 2,
    "font.size": 10,
})

run_log = []


def get_arrays(projects):
    return projects["expected_profit"].to_numpy(), projects["total_investment"].to_numpy()


def reached(profit, optimum):   # within a penny
    return profit >= optimum - 0.01


def gap_pct(optimum, profit):
    # clipped at 0: float sums in a different order can come out a tiny bit
    # above the optimum and then it prints -0.0%
    if optimum <= 0:
        return 0.0
    return max(0.0, 100 * (optimum - profit) / optimum)


def log_run(experiment, method, setting, seed, result, optimum):
    run_log.append({
        "experiment": experiment,
        "method": method,
        "setting": setting,
        "seed": seed,
        "profit": result["profit"],
        "reached_optimum": reached(result["profit"], optimum),
        "best_generation": result.get("best_generation"),
        "evaluations_to_best": result["evaluations_to_best"],
        "total_evaluations": result["evaluations"],
    })


def summarise(runs, optimum):
    final = np.array([r["profit"] for r in runs])
    return {
        "n_runs": len(runs),
        "mean_profit": final.mean(),
        "std_profit": final.std(ddof=1) if len(runs) > 1 else 0.0,
        "best_profit": final.max(),
        "worst_profit": final.min(),
        "hit_rate": np.mean([reached(p, optimum) for p in final]),
        "mean_gap_pct": gap_pct(optimum, final.mean()),
        "mean_evals_to_best": np.mean([r["evaluations_to_best"] for r in runs]),
        "evaluations": runs[0]["evaluations"],
    }


def repeat(method, experiment, method_name, setting, optimum):
    """method(seed) for every seed, logs each run, returns (summary, runs)"""
    runs = []
    for seed in range(N_RUNS):
        result = method(seed)
        log_run(experiment, method_name, setting, seed, result, optimum)
        runs.append(result)
    return summarise(runs, optimum), runs


def repeat_ga(profits, costs, budget, optimum, experiment, setting, **settings):
    summary, runs = repeat(lambda seed: run_ga(profits, costs, budget, seed=seed, **settings),
                           experiment, "GA", setting, optimum)
    summary["mean_best_generation"] = np.mean([r["best_generation"] for r in runs])
    return summary, np.array([r["history"] for r in runs])


def plot_convergence(curves, labels, title, path, x_values=None, xlabel="Generation"):
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (curve, label) in enumerate(zip(curves, labels)):
        x = x_values[i] if x_values is not None else np.arange(len(curve))
        ax.plot(x, curve / 1000, color=COLOURS[i], label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Mean best profit (£k)")
    ax.set_title(title, loc="left", color=INK)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_table(rows, path, decimals=3):
    table = pd.DataFrame(rows)
    table.to_csv(path, index=False)
    print(table.round(decimals).to_string(index=False))
    return table


# exp 1 and 2 run with both repair and penalty. With repair the GA has the
# optimum within a generation or two anyway, so the penalty runs are where
# the parameters actually show

def experiment_1(profits, costs, optimum, out):
    print("\nExperiment 1 - mutation rate")
    rows = []
    for constraint in ["repair", "penalty"]:
        curves = []
        for rate in [0.01, 0.05, 0.10]:
            summary, hist = repeat_ga(profits, costs, BUDGET, optimum, "exp1",
                                      f"{constraint} mutation={rate}",
                                      mutation_rate=rate, constraint=constraint)
            rows.append({"constraint": constraint, "mutation_rate": rate, **summary})
            curves.append(hist.mean(axis=0))
        plot_convergence(curves, ["1%", "5%", "10%"],
                         f"Exp 1: mutation rate per gene ({constraint})",
                         f"{out}/exp1_mutation_rate_{constraint}.png")
    save_table(rows, f"{out}/exp1_mutation_rate.csv")


def experiment_2(profits, costs, optimum, out):
    """Population size, twice: once with 100 gens for everyone (unfair, big
    pops get more evaluations) and once with ~5000 evaluations each."""
    print("\nExperiment 2 - population size")
    rows = []
    eval_budget = 5000

    for constraint in ["repair", "penalty"]:
        curves, x_values = [], []
        for pop in [20, 50, 100]:
            summary, _ = repeat_ga(profits, costs, BUDGET, optimum, "exp2",
                                   f"{constraint} pop={pop} gens=100",
                                   pop_size=pop, generations=100, constraint=constraint)
            rows.append({"constraint": constraint, "pop_size": pop, "generations": 100,
                         "setup": "fixed generations", **summary})

            # children per generation = pop - 2 elites
            gens = (eval_budget - pop) // (pop - 2)
            summary, hist = repeat_ga(profits, costs, BUDGET, optimum, "exp2",
                                      f"{constraint} pop={pop} gens={gens}",
                                      pop_size=pop, generations=gens, constraint=constraint)
            rows.append({"constraint": constraint, "pop_size": pop, "generations": gens,
                         "setup": "fixed evaluations", **summary})
            curves.append(hist.mean(axis=0))
            x_values.append(pop + np.arange(gens + 1) * (pop - 2))

        plot_convergence(curves, ["pop 20", "pop 50", "pop 100"],
                         f"Exp 2: population size, ~5,000 evaluations ({constraint})",
                         f"{out}/exp2_population_size_{constraint}.png", x_values=x_values,
                         xlabel="Fitness evaluations")
    save_table(rows, f"{out}/exp2_population_size.csv")


def experiment_3(profits, costs, optimum, out):
    print("\nExperiment 3 - penalty vs repair")
    rows, curves = [], []
    for method in ["penalty", "repair"]:
        start = time.perf_counter()
        summary, hist = repeat_ga(profits, costs, BUDGET, optimum, "exp3", method,
                                  constraint=method)
        elapsed = (time.perf_counter() - start) / N_RUNS
        rows.append({"constraint": method, **summary, "mean_time_s": elapsed})
        curves.append(hist.mean(axis=0))
    save_table(rows, f"{out}/exp3_constraint_handling.csv")
    plot_convergence(curves, ["penalty", "repair"],
                     "Exp 3: constraint handling", f"{out}/exp3_constraint_handling.png")


def experiment_4(profits, costs, out):
    print("\nExperiment 4 - algorithm comparison")
    rows = []
    for budget in [1_750_000, 2_750_000, 3_000_000, 4_000_000]:
        start = time.perf_counter()
        exact = exhaustive_search(profits, costs, budget)
        exact_time = time.perf_counter() - start
        optimum = exact["profit"]

        start = time.perf_counter()
        greedy = greedy_selection(profits, costs, budget)
        greedy_time = time.perf_counter() - start

        for name, result, elapsed in [("Exhaustive", exact, exact_time),
                                      ("Greedy", greedy, greedy_time)]:
            result = {**result, "evaluations_to_best": result["evaluations"]}
            rows.append({"budget": budget, "method": name, **summarise([result], optimum),
                         "mean_time_s": elapsed})

        # random searches get the same evals as a default GA run: 50 + 100*48
        ga_evals = 50 + 100 * 48
        stochastic = [
            ("GA + repair", lambda seed: run_ga(profits, costs, budget, seed=seed)),
            ("Random", lambda seed: random_search(profits, costs, budget, ga_evals, seed)),
            ("Random + repair", lambda seed: random_search_repair(profits, costs, budget, ga_evals, seed)),
        ]
        for name, method in stochastic:
            start = time.perf_counter()
            summary, _ = repeat(method, "exp4", name, f"budget={budget}", optimum)
            rows.append({"budget": budget, "method": name, **summary,
                         "mean_time_s": (time.perf_counter() - start) / N_RUNS})

    table = save_table(rows, f"{out}/exp4_algorithm_comparison.csv", decimals=4)

    # exhaustive not plotted, its gap is 0 by definition
    methods = ["GA + repair", "Random", "Random + repair", "Greedy"]
    budgets = table["budget"].unique()
    x = np.arange(len(budgets))
    width = 0.2
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, method in enumerate(methods):
        gaps = table[table.method == method].set_index("budget").loc[budgets, "mean_gap_pct"]
        bars = ax.bar(x + (i - 1.5) * width, gaps, width - 0.02, color=COLOURS[i], label=method)
        # otherwise 0% bars are invisible
        ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=8, color=INK_SOFT)
    ax.set_xticks(x, [f"£{b / 1e6:.2f}m" for b in budgets])
    ax.set_xlabel("Budget")
    ax.set_ylabel("Optimality gap vs exhaustive (%)")
    ax.set_title("Exp 4: mean gap from the exact optimum", loc="left", color=INK)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(f"{out}/exp4_algorithm_comparison.png", dpi=150)
    plt.close(fig)


def experiment_5(shortlist, out):
    """Same 20 projects, re-costed under each scenario."""
    print("\nExperiment 5 - sensitivity")
    scenarios = [
        ("baseline", "baseline", {}),
        ("allowance", "allowance 5%", {"allowance_rate": 0.05}),
        ("allowance", "allowance 15%", {"allowance_rate": 0.15}),
        ("NB build rate", "NB £2,500/m²", {"new_build_rate": 2500}),
        ("NB build rate", "NB £2,800/m²", {"new_build_rate": 2800}),
        ("land price", "land ×1.25", {"land_price_multiplier": 1.25}),
        ("land price", "land ×1.50", {"land_price_multiplier": 1.50}),
        ("land price", "land ×2.00", {"land_price_multiplier": 2.00}),
        ("sale value", "GDV −10%", {"sale_value_multiplier": 0.90}),
        ("sale value", "GDV +10%", {"sale_value_multiplier": 1.10}),
        ("sale value", "GDV top quartile", {"sale_value": "top_quartile"}),
    ]
    rows = []
    for group, label, settings in scenarios:
        projects = load_projects(shortlist, **settings)
        profits, costs = get_arrays(projects)

        exact = exhaustive_search(profits, costs, BUDGET)
        optimum = exact["profit"]
        greedy = greedy_selection(profits, costs, BUDGET)
        summary, _ = repeat_ga(profits, costs, BUDGET, optimum, "exp5", label)

        chosen = projects[exact["selection"] == 1]
        quality = valuation_quality(chosen)
        rows.append({
            "group": group,
            "scenario": label,
            "optimal_profit": optimum,
            "optimal_cost": exact["cost"],
            "n_refurb": int((chosen.strategy == "REFURB").sum()),
            "n_new_build": int((chosen.strategy == "NEW_BUILD").sum()),
            "refurb_capital": chosen.loc[chosen.strategy == "REFURB", "total_investment"].sum(),
            "new_build_capital": chosen.loc[chosen.strategy == "NEW_BUILD", "total_investment"].sum(),
            "selected": " ".join(chosen.project_id),
            "greedy_gap_pct": gap_pct(optimum, greedy["profit"]),
            "ga_mean_profit": summary["mean_profit"],
            "ga_std_profit": summary["std_profit"],
            "ga_best_profit": summary["best_profit"],
            "ga_worst_profit": summary["worst_profit"],
            "ga_mean_gap_pct": summary["mean_gap_pct"],
            "ga_hit_rate": summary["hit_rate"],
            "ga_mean_evals_to_best": summary["mean_evals_to_best"],
            "optimum_low_confidence": quality["low_confidence"],
        })

    table = save_table(rows, f"{out}/exp5_sensitivity.csv", decimals=2)

    fig, ax = plt.subplots(figsize=(8, 5))
    y = np.arange(len(table))[::-1]
    profit_k = table["optimal_profit"] / 1000
    ax.barh(y, profit_k, 0.6, color=COLOURS[0])
    baseline = profit_k.iloc[0]
    ax.axvline(baseline, color=INK_SOFT, linewidth=1, linestyle="--")
    for yi, value, row in zip(y, profit_k, table.itertuples()):
        ax.text(value + baseline * 0.01, yi, f"£{value:,.0f}k  ({row.n_refurb}R + {row.n_new_build}N)",
                va="center", color=INK_SOFT, fontsize=8)
    ax.set_yticks(y, table["scenario"])
    ax.set_xlim(0, profit_k.max() * 1.35)
    ax.set_xlabel("Optimal modelled profit at £3m (£k)")
    ax.set_title("Exp 5: optimal profit under each cost / value scenario\n"
                 "dashed line = baseline; R / N = refurbs / new builds chosen",
                 loc="left", color=INK, fontsize=10)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(f"{out}/exp5_sensitivity.png", dpi=150)
    plt.close(fig)


WARNING_TYPES = {
    "no_postcode_district": "no postcode district",
    "borough_comparables": "borough-wide comparables",
    "all_type_comparables": "comparables of all property types",
    "few_comparables": "fewer than 20 comparable sales",
    "ndss_floor_area": "floor area is an NDSS minimum",
}


def valuation_quality(projects):
    warnings = projects["valuation_warnings"].fillna("")
    counts = {"projects": len(projects), "low_confidence": int((warnings != "").sum())}
    for column, text in WARNING_TYPES.items():
        counts[column] = int(warnings.str.contains(text, regex=False).sum())
    return counts


def valuation_report(projects, out):
    """Warnings in the shortlist and in the £3m optimum. Also re-solves with
    the warned projects removed - just a check, nothing gets drawn to replace
    them so it's not a proper benchmark."""
    print("\nValuation quality")
    profits, costs = get_arrays(projects)
    exact = exhaustive_search(profits, costs, BUDGET)
    warned = projects["valuation_warnings"].fillna("") != ""
    no_district = projects["postcode_district"].isna()

    rows = []
    for label, subset in [("shortlist", projects), ("optimum at £3m", projects[exact["selection"] == 1])]:
        rows.append({"group": label, **valuation_quality(subset),
                     "project_ids": " ".join(subset.loc[subset.valuation_warnings.fillna("") != "", "project_id"]),
                     "optimal_profit": exact["profit"] if label != "shortlist" else np.nan})
    for label, drop in [("optimum without warned projects", warned),
                        ("optimum without no-postcode projects", no_district)]:
        kept = projects[~drop]
        result = exhaustive_search(*get_arrays(kept), BUDGET)
        rows.append({"group": label, **valuation_quality(kept[result["selection"] == 1]),
                     "project_ids": " ".join(projects.loc[drop, "project_id"]) + " removed",
                     "optimal_profit": result["profit"]})
    save_table(rows, f"{out}/valuation_quality.csv", decimals=0)


def run_all(shortlist):
    out = f"results/{shortlist}"
    os.makedirs(out, exist_ok=True)
    run_log.clear()

    projects = load_projects(shortlist)
    profits, costs = get_arrays(projects)
    optimum = exhaustive_search(profits, costs, BUDGET)["profit"]
    print(f"\n######## {shortlist} shortlist: {len(projects)} projects, "
          f"budget £{BUDGET:,}, exact optimum £{optimum:,.0f}")

    experiment_1(profits, costs, optimum, out)
    experiment_2(profits, costs, optimum, out)
    experiment_3(profits, costs, optimum, out)
    experiment_4(profits, costs, out)
    experiment_5(shortlist, out)
    valuation_report(projects, out)

    pd.DataFrame(run_log).to_csv(f"{out}/runs_log.csv", index=False)
    print(f"\n{len(run_log)} stochastic runs logged, results saved in {out}/")


if __name__ == "__main__":
    shortlists = sys.argv[1:] or ["top", "banded", "balanced"]
    for name in shortlists:
        run_all(name)
