"""Baselines to compare the GA with: random search (with and without
repair), greedy, and brute force for the true optimum.

They all return the same dict as run_ga so experiments.py doesn't care which is which.
"""
import numpy as np

from genetic_algorithm import repair


def random_search(profits, costs, budget, n_samples=5000, seed=None):
    """Each project in with p=0.5, over-budget samples just get thrown away.
    (experiments.py sets n_samples to match the GA's evaluations)"""
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
    """Same but each sample goes through the GA's repair() first, so the only
    difference from the GA is the search."""
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
    """Best profit/cost first. Doesn't stop at the first one that won't fit
    (a cheaper one further down might). Never picks loss-makers."""
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
    """Brute force, all 2^n subsets. ~1M for n=20 so it's fine here, and it's
    obviously correct. For big n you'd want a proper method (Pisinger 1997)."""
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
