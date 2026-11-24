"""The methods the GA is compared against.

random_search     - baseline, samples random portfolios and keeps the best feasible one
random_search_repair - same, but over-budget samples are fixed with the GA's
                    repair() instead of thrown away
greedy_selection  - take projects in order of expected profit / cost
exhaustive_search - checks every subset, gives the true optimum for ~20 projects

All of them return the same dict shape as run_ga so experiments.py can
treat them the same way.
"""
import numpy as np

from genetic_algorithm import repair


def random_search(profits, costs, budget, n_samples=5000, seed=None):
    """Sample random portfolios (each project in with p=0.5).

    n_samples defaults to 5000 = 50 x 100, the same number of fitness
    evaluations the GA gets with its default settings.
    """
    rng = np.random.default_rng(seed)
    n_projects = len(profits)
    best = np.zeros(n_projects, dtype=int)
    best_profit = 0.0
    n_feasible = 0
    evaluations_to_best = 0

    for i in range(n_samples):
        candidate = rng.integers(0, 2, size=n_projects)
        if candidate @ costs > budget:
            continue
        n_feasible += 1
        profit = candidate @ profits
        if profit > best_profit:
            best, best_profit, evaluations_to_best = candidate, profit, i + 1

    return {
        "selection": best,
        "profit": float(best_profit),
        "cost": float(best @ costs),
        "evaluations": n_samples,
        "evaluations_to_best": evaluations_to_best,
        "n_feasible": n_feasible,
    }


def random_search_repair(profits, costs, budget, n_samples=5000, seed=None):
    """Random search where every sample is repaired before it's scored.

    Gives random search the same constraint handling the GA has, so the
    comparison is about the search itself.
    """
    rng = np.random.default_rng(seed)
    n_projects = len(profits)
    best = np.zeros(n_projects, dtype=int)
    best_profit = 0.0
    evaluations_to_best = 0

    for i in range(n_samples):
        candidate = repair(rng.integers(0, 2, size=n_projects), profits, costs, budget)
        profit = candidate @ profits
        if profit > best_profit:
            best, best_profit, evaluations_to_best = candidate, profit, i + 1

    return {
        "selection": best,
        "profit": float(best_profit),
        "cost": float(best @ costs),
        "evaluations": n_samples,
        "evaluations_to_best": evaluations_to_best,
    }


def greedy_selection(profits, costs, budget):
    """Add projects by descending profit/cost ratio while they still fit.

    Keeps scanning after a project doesn't fit, since a cheaper one further
    down the list might. Loss-making projects are never picked.
    """
    selection = np.zeros(len(profits), dtype=int)
    spent = 0.0
    for i in np.argsort(-profits / costs):
        if profits[i] <= 0:
            break
        if spent + costs[i] <= budget:
            selection[i] = 1
            spent += costs[i]

    return {
        "selection": selection,
        "profit": float(selection @ profits),
        "cost": float(spent),
        "evaluations": len(profits),
    }


def exhaustive_search(profits, costs, budget, chunk_bits=16):
    """Brute force over all 2^n portfolios.

    Fine for n around 20 (about a million subsets). Pisinger (1997) gives
    far better exact methods for big instances, but at this size brute
    force is quick and obviously correct, which is what we want from the
    reference optimum.
    """
    n = len(profits)
    if n > 26:
        raise ValueError(f"{n} projects is too many for brute force")

    bit_values = 1 << np.arange(n)
    best_mask, best_profit = 0, 0.0
    chunk = 1 << min(chunk_bits, n)

    for start in range(0, 1 << n, chunk):
        masks = np.arange(start, start + chunk)
        bits = (masks[:, None] & bit_values) > 0
        total_cost = bits @ costs
        total_profit = np.where(total_cost <= budget, bits @ profits, -np.inf)
        i = np.argmax(total_profit)
        if total_profit[i] > best_profit:
            best_profit, best_mask = total_profit[i], masks[i]

    selection = ((best_mask & bit_values) > 0).astype(int)
    return {
        "selection": selection,
        "profit": float(best_profit),
        "cost": float(selection @ costs),
        "evaluations": 1 << n,
    }
