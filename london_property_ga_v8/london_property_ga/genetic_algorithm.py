"""GA for choosing projects under the budget.

One bit per project (1 = buy), like Chu & Beasley (1998), with the usual
Goldberg (1989) operators.

Over budget -> either repair it (drop worst profit/cost until it fits) or
penalise the overspend. Both from He, Zhou & Yao (2008).
"""
import numpy as np


def create_population(pop_size, n_projects, rng):
    return rng.integers(0, 2, size=(pop_size, n_projects))


def penalty_weight(profits, costs):
    """rho = best profit/cost ratio (He et al.), so every £ over budget
    costs at least as much as the best project earns per £.

    Doesn't actually guarantee infeasible < feasible, hence run_ga keeping
    the best feasible one on the side. If nothing is profitable the max
    ratio is negative and would *reward* overspending -> use abs instead.
    """
    ratios = profits / costs
    rho = np.max(ratios)
    if rho <= 0:
        rho = np.max(np.abs(ratios)) or 1.0
    return rho


def fitness(chromosome, profits, costs, budget, rho=None):
    """Profit, minus rho x overspend if over budget (-inf if no rho)."""
    total_profit = chromosome @ profits
    total_cost = chromosome @ costs
    if total_cost <= budget:
        return total_profit
    if rho is None:
        return -np.inf
    return total_profit - rho * (total_cost - budget)


def tournament_selection(population, fitnesses, rng, tournament_size=3):
    contestants = rng.choice(len(population), size=tournament_size, replace=False)
    winner = contestants[np.argmax(fitnesses[contestants])]
    return population[winner].copy()


def crossover(parent1, parent2, rng, crossover_rate=0.8, two_point=False):
    child1, child2 = parent1.copy(), parent2.copy()
    if rng.random() > crossover_rate:
        return child1, child2

    n = len(parent1)
    if n < 2:
        return child1, child2
    if two_point and n >= 3:   # 2 cuts need 3+ genes
        a, b = sorted(rng.choice(np.arange(1, n), size=2, replace=False))
    else:
        a, b = rng.integers(1, n), n
    child1[a:b] = parent2[a:b]
    child2[a:b] = parent1[a:b]
    return child1, child2


def mutation(chromosome, rng, mutation_rate=0.05):
    flips = rng.random(len(chromosome)) < mutation_rate
    chromosome[flips] = 1 - chromosome[flips]
    return chromosome


def repair(chromosome, profits, costs, budget):
    chromosome = chromosome.copy()
    ratio_order = np.argsort(profits / costs)  # worst first
    total_cost = chromosome @ costs
    for i in ratio_order:
        if total_cost <= budget:
            break
        if chromosome[i] == 1:
            chromosome[i] = 0
            total_cost -= costs[i]
    return chromosome


def run_ga(profits, costs, budget, pop_size=50, generations=100, tournament_size=3,
           crossover_rate=0.8, mutation_rate=0.05, elite_count=2,
           constraint="repair", two_point=False, seed=None):
    """Returns the best *feasible* portfolio seen. Tracked separately because
    with the penalty the fittest chromosome can be a bit over budget."""
    profits = np.asarray(profits, dtype=float)
    costs = np.asarray(costs, dtype=float)
    if constraint not in ("repair", "penalty"):
        raise ValueError("constraint must be 'repair' or 'penalty'")
    if len(profits) != len(costs):
        raise ValueError(f"profits ({len(profits)}) and costs ({len(costs)}) must be the same length")
    if not (np.isfinite(profits).all() and np.isfinite(costs).all()):
        raise ValueError("profits and costs must all be finite numbers")
    if (costs <= 0).any():
        raise ValueError("every project cost must be positive")
    for name, value in [("pop_size", pop_size), ("generations", generations),
                        ("tournament_size", tournament_size), ("elite_count", elite_count)]:
        if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
            raise ValueError(f"{name} must be an integer, got {value!r}")
    if generations < 0 or tournament_size < 1:
        raise ValueError("generations must be >= 0 and tournament_size >= 1")
    if not np.isfinite(budget) or budget <= 0:
        raise ValueError(f"budget must be finite and positive, got {budget}")
    if pop_size <= tournament_size:
        raise ValueError(f"pop_size ({pop_size}) must be bigger than tournament_size ({tournament_size})")
    if not 0 <= mutation_rate <= 1:
        raise ValueError(f"mutation_rate must be between 0 and 1, got {mutation_rate}")
    if not 0 <= crossover_rate <= 1:
        raise ValueError(f"crossover_rate must be between 0 and 1, got {crossover_rate}")
    if not 0 <= elite_count < pop_size:
        raise ValueError(f"elite_count must be between 0 and pop_size - 1, got {elite_count}")

    rng = np.random.default_rng(seed)
    n_projects = len(profits)
    rho = penalty_weight(profits, costs) if constraint == "penalty" else None

    # empty portfolio is always feasible
    best = np.zeros(n_projects, dtype=int)
    best_profit = 0.0
    evaluations = 0
    generation = 0
    # gen 0 = initial population
    best_generation, evaluations_to_best = 0, 0

    def evaluate(pop):
        nonlocal best, best_profit, evaluations, best_generation, evaluations_to_best
        if constraint == "repair":
            pop = np.array([repair(c, profits, costs, budget) for c in pop])
        scores = np.array([fitness(c, profits, costs, budget, rho) for c in pop])
        evaluations += len(pop)

        feasible = pop @ costs <= budget
        if feasible.any():
            idx = np.flatnonzero(feasible)[np.argmax(scores[feasible])]
            if scores[idx] > best_profit:
                best_profit = scores[idx]
                best = pop[idx].copy()
                best_generation = generation
                # exact position, not end of batch
                evaluations_to_best = evaluations - len(pop) + idx + 1
        return pop, scores

    population, scores = evaluate(create_population(pop_size, n_projects, rng))
    history = [best_profit]

    for generation in range(1, generations + 1):
        # careful: [-0:] is the whole array
        elite_idx = np.argsort(scores)[-elite_count:] if elite_count else np.array([], dtype=int)
        elites = population[elite_idx]
        elite_scores = scores[elite_idx]

        children = []
        while len(children) < pop_size - elite_count:
            p1 = tournament_selection(population, scores, rng, tournament_size)
            p2 = tournament_selection(population, scores, rng, tournament_size)
            c1, c2 = crossover(p1, p2, rng, crossover_rate, two_point)
            children.append(mutation(c1, rng, mutation_rate))
            children.append(mutation(c2, rng, mutation_rate))
        children = np.array(children[:pop_size - elite_count])

        children, child_scores = evaluate(children)
        population = np.vstack([elites, children])
        scores = np.concatenate([elite_scores, child_scores])
        history.append(best_profit)

    return {
        "selection": best,
        "profit": float(best_profit),
        "cost": float(best @ costs),
        "evaluations": evaluations,
        "best_generation": best_generation,
        "evaluations_to_best": evaluations_to_best,
        "history": history,
    }


if __name__ == "__main__":
    import pandas as pd

    projects = pd.read_csv("data/projects_top.csv")
    profits = projects["expected_profit"].to_numpy()
    costs = projects["total_investment"].to_numpy()

    result = run_ga(profits, costs, budget=3_000_000, seed=1)
    chosen = projects.loc[result["selection"] == 1, "project_id"].tolist()
    print(f"profit £{result['profit']:,.0f}  cost £{result['cost']:,.0f}")
    print("selected:", chosen)
    print(f"best found in generation {result['best_generation']} "
          f"after {result['evaluations_to_best']} of {result['evaluations']} evaluations")
