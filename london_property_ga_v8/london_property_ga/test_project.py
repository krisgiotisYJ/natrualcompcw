"""python -m unittest test_project -v"""
import unittest

import numpy as np
import pandas as pd

from calculate_costs import calculate_project
from comparison_methods import exhaustive_search, random_search, random_search_repair
from genetic_algorithm import fitness, penalty_weight, repair, run_ga
from experiments import N_RUNS, gap_pct
from prepare_data import (RAW_PATH, SHORTLISTS, abnormal_works, bedroom_conflict,
                          build_projects, classify_plot, description_bedrooms, drop_duplicate_units, eligible, load_projects,
                          ndss_size, stated_dwellings)


def small_instance(n=12, seed=0):
    """Small random instance, quick to brute force."""
    rng = np.random.default_rng(seed)
    costs = rng.uniform(300_000, 900_000, size=n)
    profits = costs * rng.uniform(-0.1, 0.6, size=n)
    budget = costs.sum() * 0.35
    return profits, costs, budget


class TestCostModel(unittest.TestCase):

    def test_hand_calculation(self):
        # £200k acquisition + £100k works + 10% = £330k
        project = {"strategy": "REFURB", "floor_area_m2": 100,
                   "acquisition_price": 200_000, "local_price_per_m2": 4000}
        total, profit = calculate_project(project, refurb_rate=1000)
        self.assertEqual(project["building_cost"], 100_000)
        self.assertEqual(project["additional_costs"], 30_000)
        self.assertEqual(total, 330_000)
        self.assertEqual(profit, 400_000 - 330_000)

    def test_land_multiplier_only_affects_new_build(self):
        refurb = {"strategy": "REFURB", "floor_area_m2": 100,
                  "acquisition_price": 200_000, "local_price_per_m2": 5000}
        plot = dict(refurb, strategy="NEW_BUILD")
        refurb_cost, _ = calculate_project(refurb, land_price_multiplier=2.0)
        plot_cost, _ = calculate_project(plot, land_price_multiplier=2.0)
        self.assertAlmostEqual(refurb_cost, (200_000 + 180_000) * 1.1)
        self.assertAlmostEqual(plot_cost, (400_000 + 220_000) * 1.1)

    def test_missing_floor_area_raises(self):
        project = {"strategy": "NEW_BUILD", "floor_area_m2": float("nan"),
                   "acquisition_price": 200_000, "local_price_per_m2": 5000}
        with self.assertRaisesRegex(ValueError, "floor_area_m2"):
            calculate_project(project)

    def test_non_finite_and_out_of_range_inputs_raise(self):
        base = {"strategy": "REFURB", "floor_area_m2": 90,
                "acquisition_price": 300_000, "local_price_per_m2": 5000}
        with self.assertRaisesRegex(ValueError, "acquisition_price"):
            calculate_project(dict(base, acquisition_price=float("inf")))
        with self.assertRaisesRegex(ValueError, "abnormal_costs"):
            calculate_project(dict(base, abnormal_costs=-1))
        with self.assertRaisesRegex(ValueError, "land_price_multiplier"):
            calculate_project(dict(base), land_price_multiplier=0)
        with self.assertRaisesRegex(ValueError, "allowance_rate"):
            calculate_project(dict(base), allowance_rate=float("nan"))

    def test_bad_price_raises(self):
        for bad in [0, -5, None, float("nan")]:
            project = {"strategy": "REFURB", "floor_area_m2": 90,
                       "acquisition_price": bad, "local_price_per_m2": 5000}
            with self.assertRaisesRegex(ValueError, "acquisition_price"):
                calculate_project(project)


class TestPortfolio(unittest.TestCase):

    def setUp(self):
        self.profits, self.costs, self.budget = small_instance()

    def test_empty_portfolio(self):
        empty = np.zeros(len(self.costs), dtype=int)
        self.assertEqual(empty @ self.costs, 0)
        self.assertEqual(fitness(empty, self.profits, self.costs, self.budget), 0)

    def test_over_budget_without_rho_is_worst(self):
        everything = np.ones(len(self.costs), dtype=int)
        self.assertEqual(fitness(everything, self.profits, self.costs, self.budget), -np.inf)

    def test_over_budget_penalised(self):
        everything = np.ones(len(self.costs), dtype=int)
        rho = np.max(self.profits / self.costs)
        score = fitness(everything, self.profits, self.costs, self.budget, rho)
        self.assertLess(score, everything @ self.profits)

    def test_repair_always_feasible(self):
        rng = np.random.default_rng(1)
        for _ in range(500):
            chromosome = rng.integers(0, 2, size=len(self.costs))
            fixed = repair(chromosome, self.profits, self.costs, self.budget)
            self.assertLessEqual(fixed @ self.costs, self.budget)
            # repair only removes projects, never adds them
            self.assertTrue(np.all(fixed <= chromosome))

    def test_ga_never_beats_exhaustive(self):
        optimum = exhaustive_search(self.profits, self.costs, self.budget)["profit"]
        for seed in range(5):
            for constraint in ["repair", "penalty"]:
                result = run_ga(self.profits, self.costs, self.budget, generations=30,
                                constraint=constraint, seed=seed)
                self.assertLessEqual(result["profit"], optimum + 1e-6)
                self.assertLessEqual(result["cost"], self.budget)

    def test_random_searches_feasible(self):
        for method in [random_search, random_search_repair]:
            result = method(self.profits, self.costs, self.budget, n_samples=300, seed=3)
            self.assertLessEqual(result["cost"], self.budget)

    def test_same_seed_same_result(self):
        a = run_ga(self.profits, self.costs, self.budget, generations=20, seed=7)
        b = run_ga(self.profits, self.costs, self.budget, generations=20, seed=7)
        self.assertEqual(a["profit"], b["profit"])
        self.assertTrue(np.array_equal(a["selection"], b["selection"]))
        self.assertEqual(a["history"], b["history"])
        self.assertEqual(a["best_generation"], b["best_generation"])

    def test_evaluations_to_best_is_exact(self):
        # generations=0 -> only the 30 initial chromosomes, so 1..30
        # (the old end-of-batch count always said 30)
        result = run_ga(self.profits, self.costs, self.budget, pop_size=30, generations=0, seed=4)
        self.assertGreaterEqual(result["evaluations_to_best"], 1)
        self.assertLessEqual(result["evaluations_to_best"], 30)

    def test_zero_elites_keeps_population_size(self):
        result = run_ga(self.profits, self.costs, self.budget, pop_size=20, generations=10,
                        elite_count=0, seed=0)
        # 20 + 10 x 20. The old [-0:] slice kept everyone as "elites" so this was off
        self.assertEqual(result["evaluations"], 20 + 10 * 20)
        self.assertLessEqual(result["cost"], self.budget)

    def test_single_project(self):
        for constraint in ["repair", "penalty"]:
            fits = run_ga(np.array([50.0]), np.array([100.0]), 150, generations=5,
                          constraint=constraint, seed=0)
            self.assertEqual(fits["selection"].tolist(), [1])
            too_big = run_ga(np.array([50.0]), np.array([100.0]), 80, generations=5,
                             constraint=constraint, seed=0)
            self.assertEqual(too_big["selection"].tolist(), [0])

    def test_penalty_positive_when_all_projects_lose(self):
        profits = -self.costs * 0.1
        self.assertGreater(penalty_weight(profits, self.costs), 0)
        self.assertGreater(penalty_weight(self.profits, self.costs), 0)


class TestRunGaValidation(unittest.TestCase):

    def setUp(self):
        self.profits, self.costs, self.budget = small_instance()

    def test_population_must_exceed_tournament(self):
        with self.assertRaisesRegex(ValueError, "pop_size"):
            run_ga(self.profits, self.costs, self.budget, pop_size=3, tournament_size=3)

    def test_mutation_rate_range(self):
        with self.assertRaisesRegex(ValueError, "mutation_rate"):
            run_ga(self.profits, self.costs, self.budget, mutation_rate=1.5)

    def test_array_lengths_match(self):
        with self.assertRaisesRegex(ValueError, "same length"):
            run_ga(self.profits, self.costs[:-1], self.budget)

    def test_costs_finite_and_positive(self):
        for bad in [np.inf, np.nan, 0.0, -1.0]:
            costs = self.costs.copy()
            costs[3] = bad
            with self.assertRaises(ValueError):
                run_ga(self.profits, costs, self.budget)

    def test_integer_parameters(self):
        with self.assertRaisesRegex(ValueError, "pop_size"):
            run_ga(self.profits, self.costs, self.budget, pop_size=50.0)
        with self.assertRaisesRegex(ValueError, "generations"):
            run_ga(self.profits, self.costs, self.budget, generations=10.5)

    def test_budget_positive(self):
        with self.assertRaisesRegex(ValueError, "budget"):
            run_ga(self.profits, self.costs, 0)

    def test_budget_finite(self):
        for bad in [float("nan"), float("inf")]:
            with self.assertRaisesRegex(ValueError, "budget"):
                run_ga(self.profits, self.costs, bad)


class TestExperimentRules(unittest.TestCase):

    def test_thirty_runs(self):
        self.assertEqual(N_RUNS, 30)

    def test_gap_clipped_at_zero(self):
        self.assertEqual(gap_pct(100.0, 100.0 + 1e-9), 0.0)
        self.assertAlmostEqual(gap_pct(200.0, 150.0), 25.0)

    def test_saved_runs_use_seeds_0_to_29(self):
        for name in SHORTLISTS:
            log = pd.read_csv(f"results/{name}/runs_log.csv")
            for _, runs in log.groupby(["experiment", "method", "setting"]):
                self.assertEqual(sorted(runs.seed), list(range(N_RUNS)))


class TestData(unittest.TestCase):

    def test_saved_shortlists_match_generated(self):
        # if these don't match, rerunning the code gives different results to the ones saved
        for name in SHORTLISTS:
            saved = pd.read_csv(f"data/projects_{name}.csv")
            generated = load_projects(name)
            self.assertEqual(list(saved.candidate_id), list(generated.candidate_id), name)
            np.testing.assert_allclose(saved.expected_profit, generated.expected_profit)

    def test_saved_results_match_shortlists(self):
        for name in SHORTLISTS:
            projects = load_projects(name)
            optimum = exhaustive_search(projects.expected_profit.to_numpy(),
                                        projects.total_investment.to_numpy(), 3_000_000)["profit"]
            table = pd.read_csv(f"results/{name}/exp4_algorithm_comparison.csv")
            saved = table[(table.budget == 3_000_000) & (table.method == "Exhaustive")].mean_profit
            self.assertAlmostEqual(float(saved.iloc[0]), optimum, places=2)

    def test_new_vehicular_access(self):
        excluded, cost, note = abnormal_works(
            "Construction of a new dwelling with new vehicular access.")
        self.assertFalse(excluded)
        self.assertEqual(cost, 10_000)

    def test_vacant_land_beats_adjacent(self):
        self.assertEqual(classify_plot("Construction of a two bedroom dwellinghouse on the land "
                                       "adjacent to 6 Tanner Street.",
                                       "Vacant Land Between, 6 - 7 Tanner Street"), "VACANT_LAND")
        self.assertEqual(classify_plot("New house in the rear garden", "Vacant land at 5 High Street"),
                         "GARDEN_PLOT")

    def test_ndss_sizes(self):
        self.assertEqual(ndss_size(4, house=True), 106)
        self.assertEqual(ndss_size(5, house=True), 119)
        self.assertEqual(ndss_size(6, house=False), 125)
        self.assertIsNone(ndss_size(7, house=True))

    def test_stated_dwellings(self):
        self.assertEqual(stated_dwellings("Proposed two storey two bedroom detached house."), 1)
        self.assertEqual(stated_dwellings("2 x Two storey, 2-bed semi-detached dwellings"), 2)
        self.assertEqual(stated_dwellings("construction of 4no. 2-bedroom, two-storey houses"), 4)
        self.assertIsNone(stated_dwellings("Erection of 2 semi-detached dwellings and 3 flats, with "
                                           "landscaping and new homes"))

    def test_description_bedrooms(self):
        self.assertEqual(description_bedrooms("Erection of 2x4 bedroom dwelling houses", 2), [4, 4])
        self.assertEqual(description_bedrooms("one single storey (2 bed) and one two storey "
                                              "(3 bed) semi-detached dwellings", 2), [2, 3])
        self.assertEqual(description_bedrooms("Proposed two storey two bedroom detached house", 1), [2])
        self.assertIsNone(description_bedrooms("Erection of a detached dwelling", 1))

    def test_bedroom_conflict_flagged(self):
        # 113 Gregory Crescent: description 1 x 2 bed, unit record 3 bedrooms
        unit = {"unit_type": "House or Bungalow", "no_bedrooms": 3, "gia": None}
        description = "Erection of two storey dwelling house (C3 Use) (1 x 2 bed) with rear roof dormer"
        self.assertIn("[2]", bedroom_conflict([unit], description))
        self.assertEqual(bedroom_conflict([dict(unit, no_bedrooms=2)], description), "")

    def test_conflicting_records_not_shortlisted(self):
        pool = build_projects(pd.read_csv(RAW_PATH))
        self.assertTrue((pool.bedroom_conflict != "").any())
        self.assertTrue((eligible(pool).bedroom_conflict == "").all())

    def test_site_check_sites_not_shortlisted(self):
        pool = build_projects(pd.read_csv(RAW_PATH))
        self.assertEqual(int(pool.site_check.sum()), 14)
        for name in SHORTLISTS:
            self.assertFalse(load_projects(name).site_check.any(), name)

    def test_balanced_shortlist_is_balanced(self):
        projects = load_projects("balanced")
        for strategy, n in [("REFURB", 6), ("NEW_BUILD", 4)]:
            group = projects[projects.strategy == strategy]
            self.assertEqual((group.expected_profit > 0).sum(), n)
            self.assertEqual((group.expected_profit <= 0).sum(), n)

    def test_repeated_unit_rows_dropped(self):
        row = {"unit_type": "House or Bungalow", "no_bedrooms": 2, "gia": 72}
        self.assertEqual(len(drop_duplicate_units([row, dict(row)], "Proposed detached house")), 1)
        # different homes are kept even if the description seems to say one
        other = dict(row, no_bedrooms=1, gia=50)
        self.assertEqual(len(drop_duplicate_units([row, other], "1 x two bed and 1 x one bed")), 2)


if __name__ == "__main__":
    unittest.main()
