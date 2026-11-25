# London property portfolio selection with a genetic algorithm

The project picks which of 20 London housing projects to fund with £3m, so that total modelled profit is as high as possible. That makes it a 0-1 knapsack problem. I solve it with a binary GA and then check the GA against some simpler methods and an exhaustive search.

There are two kinds of project. A REFURB buys an existing house, renovates it at £1,800/m² and sells it. A NEW_BUILD buys a plot with planning permission (vacant land, a garden plot or a garage site), builds at £2,200/m² and sells. All the money is worked out before the optimiser runs:

```
works            = rate × floor area (+ abnormal costs, new builds only)
additional_costs = 10% × (acquisition + works)
total_investment = acquisition + works + additional_costs
estimated_gdv    = floor area × local £/m²
expected_profit  = estimated_gdv − total_investment
```

Every project is a hypothetical appraisal. A refurb starts from the price a house really sold for in 2024, but the works and the resale are modelled. New-build plot prices are modelled from MHCLG land values. So the profits are model outputs, not what a developer would actually make. That's good enough for testing an optimiser, but I wouldn't buy anything on the strength of them.

## Running it

```
pip install -r requirements.txt
python -m unittest test_project -v   # 38 tests, a few seconds
python prepare_data.py               # the three shortlists -> data/projects_<name>.csv
python experiments.py                # all experiments, all shortlists, about 10 min
python experiments.py top            # just one shortlist
```

`data/raw_candidates.csv` and the Datahub extract are included, so you don't need to download anything to rerun it. Results go to `results/<shortlist>/`. The code is split into `calculate_costs.py` (cost model), `prepare_data.py` (candidate pool, data checks, shortlists), `genetic_algorithm.py`, `comparison_methods.py` (random search, random search + repair, greedy, exhaustive), `experiments.py` and `test_project.py`. To rebuild from scratch, run `python download_data.py` (about 1.5 GB; `data/raw/` isn't included) and then `python prepare_data.py --rebuild`. The Datahub changes daily, so the saved extract is kept unless you pass `--refresh-datahub`.

## Data

**Refurbs.** These are 2024 HM Land Registry sales linked to EPC floor areas (Chi et al. 2025). A sale qualifies if it's a freehold house, a standard market sale, not new, 50-250 m², and in the cheapest quarter of at least 5 comparable sales (same postcode district, house type and age band). Sales below half the comparable median are dropped as likely data errors. GDV is the group's median £/m². That gives 3,482 candidates, of which only 307 (9%) make a profit, because the £/m² has to rise about 40% just to cover £1,800/m² plus 10%. Picking houses by poor EPC rating instead worked worse (53 of 4,464 profitable), since many London period houses rate E even after renovation. Note that the comparables include sales made after the purchase date.

**New builds.** These are approved full permissions from the Planning London Datahub, decided 2022-24 for 1-4 new homes on a site with no existing homes, not yet started: 147 applications. `classify_plot` keeps garage sites (51), garden plots (32) and vacant land (25). It drops extensions/conversions (32) and mixed-use schemes (7), which the model can't price. Also dropped or fixed:

- duplicate unit rows (some boroughs, mostly Havering, enter each home twice) and older permissions on the same address
- sites over 0.5 ha, LLDC/OPDC sites (no MHCLG value) and basement schemes
- 14 schemes that also alter the house next door (`site_check`), and 113 Gregory Crescent, where the description and unit record disagree on bedrooms (`bedroom_conflict`). Both stay in `raw_candidates.csv` but never reach a shortlist.

Floor area is the application GIA, or the NDSS minimum (DCLG 2015) if that's missing or outside 37-250 m². Demolition gets a flat £20,000 and new access £10,000, both my own estimates (42 sites have one). New-build GDV is the £/m² of 2024 EPC C+ sales of the same property type, at district level if there are 10+ sales, otherwise borough level. This is the weakest input: 20 sites have no postcode district, and 55 of 82 carry a `valuation_warnings` entry. Plot price is MHCLG's 2023 borough £/ha at the nearest density band, converted to £ per dwelling and scaled by district vs borough prices (factor 0.64-1.22). MHCLG says these aren't site valuations, so treat them as a rough proxy. That leaves 82 sites in 17 boroughs, 63 of them profitable.

**Shortlists.** Projects over £1m, `site_check` sites and bedroom conflicts are removed first.
- `top`: the 12 refurbs and 8 new builds with the best return on cost. All 20 are profitable.
- `banded`: each strategy is split into thirds by return on cost, then 4/4/4 refurbs and 2/3/3 new builds are drawn with seed 42. The thirds are cut by position, because `pd.qcut` put the bin edges in different places in pandas 2.2 and 3.0 and changed the draw. All 12 banded refurbs lose money.
- `balanced`: 6 profitable and 6 loss-making refurbs, plus 4 and 4 new builds, with seed 42. I added it because banded turned out too easy (an algorithm can just ignore the refurbs). I fixed the rule before running anything on it. It's a stress test, not a sample of the market.

## Method

The GA uses the binary knapsack representation from Chu and Beasley (1998) and standard operators from Goldberg (1989):
- one bit per project, fitness = total profit
- population 50, 100 generations (4,850 evaluations)
- tournament size 3, one-point crossover at 0.8, mutation 0.05 per gene, 2 elites

Over-budget portfolios are either repaired or penalised (He, Zhou and Yao 2008). Repair drops the chosen project with the worst profit/cost ratio until the portfolio fits. Penalty subtracts ρ × overspend, with ρ the best profit/cost ratio; since that can still let an infeasible portfolio score well (Deb 2000), the GA tracks the best feasible one separately.

Baselines: random search (p = 0.5 per project, over-budget samples discarded), random search + repair, greedy by profit/cost, and exhaustive search over all 2^20 subsets, which takes about 0.2 s and gives the true optimum. Each stochastic setting runs 30 times (seeds 0-29), changing one parameter at a time (Eiben and Smit 2011). Every run is in `runs_log.csv`. The repeats measure the GA's reliability, not the uncertainty in land prices or GDVs; experiment 5 covers that.

## Results

| Shortlist | Optimum at £3m | Capital used | Projects |
|---|---|---|---|
| top | £2,270,035 | £2,902,981 | N02 to N06, all new builds |
| banded | £764,565 | £2,252,525 | N03 to N08, all new builds |
| balanced | £878,921 | £2,812,513 | R01 (113 Cortis Road) + N01 to N03 |

Figures are for the top list at £3m unless stated, and "29/30" means 29 of 30 runs hit the exact optimum. Top refurbs are all profitable but return less per £ than the new builds, so none are picked. Banded leaves £0.75m unspent because all its refurbs lose money. The optima rely on uncertain valuations: dropping every project with a valuation warning lowers them by 18.7% (top), 53.7% (banded) and 21.5% (balanced); see `valuation_quality.csv`.

**Exp 1, mutation rate**

| Constraint | Mutation | top | Mean gap | Worst run | Evals to best | balanced |
|---|---|---|---|---|---|---|
| repair | 1% | 23/30 | 0.15% | £2,238,110 | 124 | 30/30 |
| repair | 5% | 30/30 | 0% | £2,270,035 | 212 | 30/30 |
| penalty | 1% | 16/30 | 1.31% | £1,967,418 | 1,365 | 13/30 |
| penalty | 5% | 30/30 | 0% | £2,270,035 | 981 | 30/30 |

1% mutation is too low, especially with the penalty: its worst balanced run (£614,115) is 30% below the optimum, the worst GA result overall. 5% and 10% always reach it (10% with repair in only 69 evaluations), and banded hits the optimum at every setting.

**Exp 2, population size.** With 100 generations, a penalty GA with a population of 20 (only 1,820 evaluations) got 27/30 on top and 24/30 on balanced. Given 276 generations (about 5,000 evaluations) it got 30/30 on both, so the problem was the number of evaluations, not the population size. With repair, population 20 misses the same run whether it gets 100 or 276 generations; that run gets stuck £709 short early on.

**Exp 3, repair vs penalty.** Both reach the optimum in all 30 runs on every list; repair just gets there faster, after 212 evaluations against 981 on top, 75 against 569 on banded and 10 against 816 on balanced. On balanced, 29 of the 30 repair runs already had the optimum in generation 0, because repairing 50 random chromosomes gives near-greedy portfolios. Per run it's slightly slower (0.186 s vs 0.166 s).

**Exp 4, comparison.** Mean gap to the optimum over 30 runs (greedy runs once):

| Budget | GA + repair | Random | Random + repair | Greedy |
|---|---|---|---|---|
| top £1.75m | 0% | 42.9% | 0% | 9.2% |
| top £3.00m | 0% | 20.8% | 0% | 1.9% |
| banded £4.00m | 0% | 25.2% | 3.0% (0/30) | 0% |
| balanced £2.75m | 0% | 31.5% | 0% | 4.0% |

Random search + repair matches the GA almost everywhere, so on 20 projects repair does most of the work. Plain random search is 16-69% short because most samples go over budget. The GA only wins on banded at £4m: the budget covers every profitable project, so repair rarely triggers, and only GA selection removes the loss-makers. Balanced turned out easier than expected.

**Exp 5, sensitivity** (30 GA runs per scenario)

| Scenario | top optimum | Greedy gap | GA at optimum | balanced optimum | balanced mix |
|---|---|---|---|---|---|
| baseline | £2,270,035 | 1.9% | 30/30 | £878,921 | 1 R + 3 NB |
| allowance 5% / 15% | £2,588,866 / £2,109,116 | 9.1% / 0.3% | 27/30 / 30/30 | £1,006,763 / £751,080 | 1 R + 3 NB |
| NB build £2,500 / £2,800 | £1,970,385 / £1,587,836 | 3.4% / 0.02% | 30/30 / 30/30 | £741,097 / £622,794 | 1R+3NB / 3R+1NB |
| land ×1.25 / ×1.5 / ×2 | £1,932,455 / £1,581,715 / £1,138,024 | 2.3% / 5.2% / 0% | 30 / 29 / 30 | £725,078 / £610,584 / £516,534 | 2R+2NB / 3R+1NB / 3R+1NB |
| GDV −10% / +10% | £1,752,733 / £2,793,853 | 1.4% / 2.4% | 30/30 / 10/30 | £509,778 / £1,248,064 | 1 R + 3 NB |
| GDV top quartile | £3,049,090 | 5.6% | 15/30 | £1,334,959 | 1 R + 3 NB |

Land price matters most: doubling it halves the top optimum and cuts balanced by 41%. The top optimum never includes a refurb, whatever the scenario, but on balanced dearer land or building costs swap new builds for refurbs. GDV +10% is the hardest case (10/30), but the mean gap is only 0.16%: the second-best portfolio is £5,106 below the optimum and repair tends to settle there. Greedy missed in 10 of the 11 top scenarios.

## Limitations / things I would improve

- The plot prices are the biggest problem. MHCLG's figures are borough averages made for policy appraisal, and my density band and location factor are steps I added on top. Combined with a fairly cheap £2,200/m² build cost, I think they flatter new builds, which is probably why the top optimum is all new builds. Real land sale prices would be much better, but I couldn't find a usable source.
- New-build GDV comes from second-hand EPC C+ sales, not new-build sales, and 20 sites only have borough values. The allowances are flat guesses too, and the 10% doesn't really cover stamp duty, finance, fees, CIL/S106 and VAT on a London scheme. A cheap refurb might be cheap for reasons the data doesn't show (structural problems, a bad street). Datahub statuses also lag, so some "not started" sites have probably been built or sold by now.
- 20 projects is too small to show much. Exhaustive search takes 0.2 s, and repaired random search keeps up with the GA. Next time I'd try 100+ projects or extra constraints (cash flow over time, a cap per borough), where exhaustive search isn't possible. Profit is also a single number with no risk attached. Xu et al. (2025) optimise on risk as well, but they have price histories, which these projects don't.

## References

- Chu, P.C. and Beasley, J.E. (1998) 'A genetic algorithm for the multidimensional knapsack problem', *Journal of Heuristics*, 4, pp. 63-86. https://doi.org/10.1023/A:1009642405419
- Chi, B., Dennett, A., Oléron-Evans, T. and Morphet, R. (2025) *House Price Per Square Metre in England and Wales, 1995-2024*. UK Data Service. https://doi.org/10.5255/UKDA-SN-857911
- Deb, K. (2000) 'An efficient constraint handling method for genetic algorithms', *Computer Methods in Applied Mechanics and Engineering*, 186(2-4), pp. 311-338. https://doi.org/10.1016/S0045-7825(99)00389-8
- DCLG (2015) *Technical housing standards: nationally described space standard*. https://www.gov.uk/government/publications/technical-housing-standards-nationally-described-space-standard
- Eiben, A.E. and Smit, S.K. (2011) 'Parameter tuning for configuring and analyzing evolutionary algorithms', *Swarm and Evolutionary Computation*, 1(1), pp. 19-31. https://doi.org/10.1016/j.swevo.2011.02.001
- Goldberg, D.E. (1989) *Genetic Algorithms in Search, Optimization, and Machine Learning*. Addison-Wesley.
- Greater London Authority, *Planning London Datahub*. https://www.london.gov.uk/programmes-strategies/planning/digital-planning/planning-london-datahub
- He, J., Zhou, Y. and Yao, X. (2008) 'A comparison of GAs penalizing infeasible solutions and repairing infeasible solutions on the 0-1 knapsack problem', *Dagstuhl Seminar Proceedings*. https://doi.org/10.4230/DagSemProc.08051.3
- HM Land Registry, *Price Paid Data*. https://www.gov.uk/government/statistical-data-sets/price-paid-data-downloads
- MHCLG (2026) *Land value estimates for policy appraisal 2023*. https://www.gov.uk/government/publications/land-value-estimates-for-policy-appraisal-2023
- MHCLG, *Brownfield land* dataset. https://www.planning.data.gov.uk/dataset/brownfield-land
- Pisinger, D. (1997) 'A minimal algorithm for the 0-1 knapsack problem', *Operations Research*, 45(5), pp. 758-767. https://doi.org/10.1287/opre.45.5.758
- Xu et al. (2025) 'Utilizing the real estate investment trusts for portfolio optimisation by application of genetic algorithm', *Humanities and Social Sciences Communications*, 12, Article 476. https://doi.org/10.1057/s41599-025-04715-0

Contains HM Land Registry data © Crown copyright and database right 2024, licensed under the Open Government Licence v3.0.
