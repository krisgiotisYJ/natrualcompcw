# London property portfolio selection with a genetic algorithm

This project picks which London residential projects to fund from a shortlist of 20, so that total expected profit is as high as possible without going over a fixed capital budget (£3m by default). Each project is one of two kinds:

- REFURB: buy an existing house, renovate it and sell it. Building cost is £1,800 per m².
- NEW_BUILD: buy a development plot (vacant land, a garden plot or a garage site), build on it and sell. Building cost is £2,200 per m².

Both are hypothetical appraisals. A refurb starts from the price a house really sold for in 2024, but the renovation and the resale are modelled: the project doesn't use houses that were bought, refurbished and sold again, so nothing here is a developer's actual profit. That is enough for the purpose, which is to give the optimiser realistic numbers to choose between.

All the money is worked out before any optimisation happens:

```
works            = rate × floor area (+ abnormal costs, new builds only)
additional_costs = 10% × (acquisition + works)
total_investment = acquisition + works + additional_costs
estimated_gdv    = floor area × local £/m²
expected_profit  = estimated_gdv − total_investment
```

`calculate_project()` rejects any input that isn't a finite number in range: prices, areas and £/m² must be positive, abnormal costs and the allowance rate can't be negative, and build rates and multipliers must be above zero. `building_cost` (rate × area), `abnormal_costs`, `additional_costs` and `total_investment` are separate columns. The sale value is an estimated GDV, not a valuation, so every profit figure in this project is a modelled profit.

After that the problem is a 0-1 knapsack: profit is the value, total investment the weight and the budget the capacity. The optimiser only decides which projects go in.

## Running it

```
pip install -r requirements.txt
python -m unittest test_project -v   # 38 tests, a few seconds
python prepare_data.py               # the three shortlists -> data/projects_<name>.csv
python experiments.py                # all experiments on all three shortlists, about 10 minutes
python experiments.py top            # or just one shortlist
```

`data/raw_candidates.csv` is included, so none of the commands above need the raw source files. Rebuilding the candidate pool from scratch is covered under "Source files and rebuilding" below. Results go to `results/top/`, `results/banded/` and `results/balanced/`.

## Files

| File | What it does |
|---|---|
| `calculate_costs.py` | `calculate_project()`: costs and profit for one project, with the sensitivity multipliers |
| `prepare_data.py` | builds the candidate pool from the raw sources, runs the data checks, classifies each candidate and makes the three shortlists |
| `genetic_algorithm.py` | `create_population`, `fitness`, `tournament_selection`, `crossover`, `mutation`, `repair`, `run_ga` |
| `comparison_methods.py` | `random_search`, `random_search_repair`, `greedy_selection`, `exhaustive_search` |
| `experiments.py` | experiments 1 to 5 plus a log of every GA and random search run |
| `test_project.py` | unit tests (unittest) |
| `download_data.py` | downloads the raw source files into `data/raw/` |

## Data

| Source | Used for |
|---|---|
| HM Land Registry Price Paid Data, 2024 | address, PPD category and new/old flag for each sale |
| Chi, Dennett, Oléron-Evans and Morphet (2025), house price per m², London borough files | the same sales linked to EPC certificates: floor area, £/m², EPC score |
| Planning London Datahub (GLA) | small new-build permissions in every borough: description, site area, and each new home's type, bedrooms and floor area |
| Brownfield land registers (planning.data.gov.uk) | cross-check only |
| MHCLG Land value estimates for policy appraisal 2023 | residential land value per hectare by borough |

Not every input is a measurement, and the projects table says which ones are estimates.

| Column | Values |
|---|---|
| `price_source_type` | `recorded sale (Price Paid 2024)` for REFURB, `estimated land value (MHCLG 2023)` for NEW_BUILD |
| `floor_area_source` | `EPC` for REFURB. For NEW_BUILD it is `Datahub unit GIA` where every new home has a floor area on the application, and `Datahub bedrooms + NDSS minimum` where at least one is missing or implausible and the NDSS minimum for its bedroom count is used instead. If a home has no bedroom count either, the area is read from the description (`planning description + NDSS minimum` or `assumed minimum (2-bed NDSS)`). No site in the current extract needs that |
| `abnormal_costs` and `notes` | any allowance added for extra works, and why |
| `site_type` and `plot_type` | `existing_dwelling` (REFURB) or `development_plot` (NEW_BUILD). `plot_type` says which kind of plot: `VACANT_LAND`, `GARDEN_PLOT` or `GARAGE_REDEVELOPMENT` (see below) |
| `valuation_basis` and `comparable_count` | which sales the GDV £/m² comes from and how many there were. REFURB: same district, house type and age band. NEW_BUILD: for example `district, type T` or `borough, type house` |
| `dwellings`, `site_hectares`, `density_band`, `land_value_per_plot`, `location_factor` | NEW_BUILD only: every input to the plot price, so `acquisition_price` = dwellings × land_value_per_plot × location_factor (rounded to £1,000) can be checked by hand |
| `bedroom_conflict` | NEW_BUILD only: set when the planning description and the unit records disagree on bedrooms (see below) |
| `site_check` | NEW_BUILD only: `True` when the scheme also alters the host or neighbouring house. These sites aren't shortlisted |
| `valuation_warnings` | reasons the GDV is less reliable than usual, empty if none |

### REFURB candidates

A 2024 London sale is considered if it is a house (detached, semi or terrace), freehold, a standard market sale, not a new build and 50 to 250 m². It counts as a refurb candidate if it sold in the cheapest quarter of comparable sales, meaning houses of the same type and construction age band in the same postcode district, with at least 5 such sales. Sales below half the comparable median are dropped as likely floor-area mismatches or non-market sales. The acquisition price is the price actually paid.

The estimated GDV is the median £/m² of the same comparable group. Experiment 5 also runs the top quarter (75th percentile) of the group as a sensitivity case. The comparables include sales from later in 2024 than the purchase, which is fine for an appraisal but means this isn't a simulation of a decision made with only the information available on the purchase date.

EPC rating isn't used to pick candidates. Selecting houses rated E or worse and valuing them at the median of EPC C+ sales in the district gives 4,464 candidates, of which only 53 (1.2%) make a profit. A London period house with solid walls rates E even after a full renovation, and the median certificate in that group was 10 years old at the time of sale. Those houses sold at the 59th percentile of their comparables, so most were ordinary houses bought at full price. The EPC score is kept in the `notes` column for reference.

That gives 3,482 candidates in 32 boroughs, of which 307 (9%) make a modelled profit at the median GDV and 1,132 at the top-quarter GDV. With £1,800/m² plus the 10% allowance, a refurb needs its £/m² to rise about 40% above the purchase price to break even, which most houses in the cheapest quarter still don't manage.

### NEW_BUILD candidates

The candidates come from the Planning London Datahub. They are approved full permissions decided in 2022 to 2024 for 1 to 4 new-build homes on a site with no existing homes, where work hasn't started (status still "Approved", not "Commenced", "Completed" or "Lapsed"). That gives 147 applications. The query is in `download_data.py`, and the extract is saved as `data/pld_small_new_build.json`.

The model prices a new build as a plot bought at land value, so each permission is classified from its description into a `plot_type`, and only the first three types below are kept:

| plot_type | Meaning | Applications |
|---|---|---|
| `GARAGE_REDEVELOPMENT` | garages, sheds, stables or other structures are demolished first | 51 |
| `GARDEN_PLOT` | built in the rear or side garden of a house, or on land next to one | 32 |
| `VACANT_LAND` | none of the other types, or a site the application calls vacant land without mentioning a garden | 25 |
| `EXTENSION_CONVERSION` (dropped) | the home is made by extending or converting a building, or is an annexe. 35-36 Dorset Square ("single storey rear extension at lower ground floor level to create a self-contained residential unit") is one | 32 |
| `MIXED_USE` (dropped) | the scheme also has shop, office or garage space that the model neither costs nor values, such as a tyre workshop replaced by offices with 4 flats above | 7 |

The rules are regular expressions in `prepare_data.py` (`classify_plot`). They aren't perfect: Bromley 22/02409, a detached bungalow behind a dental practice, is dropped because the same application extends the practice. A site named as vacant land is `VACANT_LAND` even if the description says "adjacent", unless it also mentions a garden: "Vacant Land Between 6 - 7 Tanner Street" (a 2-bed house "on the land adjacent to 6 Tanner Street") had been labelled `GARDEN_PLOT`. The label doesn't enter the cost or price formulas.

Then:

- Some boroughs enter each home twice. Havering does it on most applications, so "Proposed two storey two bedroom detached house" has two identical 72 m² rows and even the application's unit total says 2. Where the description gives a number of homes ("2 x … dwellings", "4no. … houses", or only the singular "house"), repeated rows beyond that number are dropped. A row counts as a repeat only if it has the same type, bedrooms and floor area as one already kept, or has no floor area. If the description gives two counts ("2 houses and 3 flats") nothing is dropped. This changed 12 applications, 9 of them in Havering.
- Permissions for the same address are alternatives for one piece of land. 159 Monkhams Lane has one permission for three houses and three for a single house each. Only the most recently decided permission is kept (3 dropped).
- Sites over 0.5 ha (a data error for four homes), sites in the LLDC and OPDC (not boroughs, so no MHCLG land value) and basement schemes (12; the description never gives the basement's size) are dropped.

Two allowances cover works the £/m² rate doesn't: £20,000 where structures are demolished and £10,000 for new or resurfaced access. Both are my own assumptions, not quotes, and the `notes` column records them. 42 sites carry at least one. The demolition allowance is meant for clearing small structures on the plot, such as the single storey side extension and garage at 26 Hardie Road; it isn't sized for a whole house or for asbestos or contamination. Because the 10% allowance is applied on top, the access allowance adds £11,000 to a project's investment. Until v6 the pattern missed "new vehicular access", so 27 Princes Park Avenue, Hayes had no access cost; with it, its modelled profit falls from £229,476 to £218,476, which is enough to drop it from the top shortlist (see Results).

Some permissions in the extract are variations of an earlier consent rather than the first approval. 34 Allen Road, Rainham is one ("Variation of Condition Nos 2, 3, and 7 to amend arrangement of materials, position and sizing of doors ... (Two storey, 2-bed, ..."). It still describes a new 2-bed house, so it stays in as a new build.

Some schemes also change the house next to the plot. 2a Devon Road demolishes a side utility room and moves a conservatory door; 63 Thurlow Gardens adds a rear extension and roof alterations to the host house. The flat allowance wasn't sized for that, so instead of guessing a cost the 14 such sites get `site_check = True`. They stay in `raw_candidates.csv` but can't be shortlisted, because uncosted work on another building would otherwise feed straight into the optimum. In v4 two of them (1 Valley Road and 135 Woodbridge Road) were in the banded optimum.

Floor area is the sum of the new homes' GIA from the application. A home with no GIA, or one outside 37 to 250 m², gets the minimum for its bedroom count from the Nationally Described Space Standard (DCLG 2015): single storey sizes for flats and bungalows, two storey sizes for houses, and the beds + 2 persons occupancy (2b4p, 3b5p up to 6b8p). The NDSS stops at 6 bedrooms, so a bigger home without a usable GIA is dropped rather than sized as a smaller one. None are in the current extract. 159 Monkhams Lane's 5-bed houses get 119 m² (5b7p, two storey).

The bedrooms in the description are compared with the unit records. One site disagrees: 113 Gregory Crescent's description says "1 x 2 bed", but its unit record says 3 bedrooms and gives no floor area. The code had used the unit record, so the home was sized at 93 m² (3b5p) when a 2-bed would be 79 m² (2b4p). At 79 m² the project's modelled profit falls from £101,477 to £61,031, and since it was in the v3 banded shortlist, the banded optimum would have fallen from £934,994 to £894,548. Only the planning drawings can say which is right, so the site stays in `raw_candidates.csv` with the conflict in `bedroom_conflict` but can't be shortlisted.

GDV uses 2024 EPC C+ standard-market sales (Price Paid category A) of the same property type: terraced, semi-detached or detached for those unit types, all houses for "House or Bungalow", and flats for flats. The most local group with at least 10 sales is used, falling back from postcode district + type to borough + type, then district and borough over all types. 51 of the 82 sites are valued at district level and 2 fall back to all property types. 20 have no postcode district in the Datahub, so they get borough values, and 20 rest on fewer than 20 sales. (The category A filter changes nothing at present, because the linked sales file only contains category A sales, but it keeps repossessions and other non-market sales out if the source changes.) An EPC rating of C or better shows a home in reasonable condition. It doesn't show that the home is new or finished to the standard a developer would sell at.

The fallbacks stay, but each candidate's `valuation_warnings` lists what makes its GDV less certain: no postcode district (20 new-build sites), borough-wide comparables (31), comparables of all property types (2), fewer than 20 comparable sales (20) and a floor area that is an NDSS minimum rather than a measurement (9). 55 of the 82 new-build sites have at least one warning. Among refurbs the only warning that can apply is the small comparable group, and 1,401 of 3,482 have it.

Plot prices are a model. MHCLG publishes its land value estimates for policy appraisal and says they aren't valuations of individual sites. The estimate takes the borough's MHCLG £/ha at the density band nearest the site's own density (69 sites use the low band, 10 medium, 2 high and 1 with no site area defaults to low) and converts it to a value per dwelling. That figure is then scaled by the postcode district's £/m² (all property types) relative to the borough's: the factor runs from 0.64 to 1.22, and sites without a postcode district get 1. Each step is a column in the table. For example, Land To Rear Of 43 Henry Road, Barnet: 1 dwelling × £163,818 per plot (low band) × 1.0438 = £171,000.

That leaves 82 sites in 17 boroughs, 63 of them profitable. The largest counts are Redbridge (16), Croydon (13), Havering (11) and Bromley (11). Each site's planning reference was also searched for in the brownfield register, and none were found. The register's small sites are mostly older permissions, and only Haringey describes them, while none of the Datahub sites are in Haringey.

### The three shortlists

Projects costing over £1m, sites with a bedroom conflict and `site_check` sites are removed first in all three.

- `top` holds the 12 REFURB and 8 NEW_BUILD projects with the highest return on cost. All 20 are profitable. The refurbs are in 9 boroughs and the new builds in 5.
- `banded` sorts each strategy's pool by return on cost (ties broken by `candidate_id`), cuts it into thirds by position (low, middle, high) and draws from each third with a fixed seed (42): 4/4/4 refurbs and 2/3/3 new builds. Cutting by position gives the same thirds in every pandas version. `pd.qcut` placed its bin edges differently in pandas 2.2 and 3.0, which moved two sites between thirds and changed the draw. Only 9% of the refurb pool is profitable, so all 12 banded refurbs are loss-makers, while 6 of the 8 banded new builds are profitable. The banded list shows how the methods cope with a table that is mostly poor projects. Removing ineligible sites from the pool moves the new-build thirds, so the banded new builds differ from earlier versions.
- `balanced` is a controlled benchmark added because the banded list is easy: every banded refurb loses money, so an algorithm can ignore that half of the problem. It draws 6 profitable and 6 loss-making refurbs and 4 profitable and 4 loss-making new builds, with the same fixed seed (42). The rule was written down before any algorithm was run on it, and it wasn't adjusted afterwards. The half-and-half profitability mix is chosen on purpose, so the balanced list is a stress test, not a random sample of the London market.

`test_project.py` checks that the three `data/projects_<name>.csv` files match what `prepare_data.py` generates, and that the optimum in `results/` matches the generated shortlist. I checked the shortlists under pandas 3.0.3 with numpy 2.4.6 and under pandas 2.2.3 with numpy 1.26.4.

## Method

The GA follows Chu and Beasley (1998). A chromosome is one bit per project and the fitness is total profit. The operators come from Goldberg (1989): random initial population, tournament selection of size 3, one-point crossover at 0.8, bit-flip mutation at 0.05 per gene, and elitism keeping the best 2 (0 is allowed and keeps none). With a single project there is nothing to cross over, so crossover is skipped. It runs with a population of 50 for 100 generations, which comes to 4,850 fitness evaluations. `run_ga()` rejects invalid settings with a ValueError, for example a population no bigger than the tournament, or a budget that is zero, negative, NaN or infinite. It also rejects costs that are zero, negative or not finite, and non-integer population, generation, tournament and elite counts. It records the generation and the exact evaluation (counted one chromosome at a time) at which its final best solution first appeared. Called on its own without a penalty weight, `fitness()` scores an over-budget portfolio as minus infinity, so reusing it outside `run_ga()` can't rank an infeasible portfolio above a feasible one.

There are two ways to handle portfolios that go over budget, taken from He, Zhou and Yao (2008):

- Repair drops the chosen project with the lowest profit/cost ratio, then the next lowest, until the portfolio fits, and writes the repaired chromosome back into the population.
- Penalty subtracts ρ × overspend, where ρ is the largest profit/cost ratio among the projects, or the largest absolute ratio if no project makes a profit (otherwise ρ would be negative and reward overspending). This is a heuristic: it doesn't guarantee that an over-budget portfolio scores below every feasible one. Deb (2000) gives the general case for keeping feasibility separate from the objective.

Under the penalty method the GA keeps a separate record of the best feasible portfolio it has seen, and returns that.

There are four comparison methods:

- Random search draws portfolios with each project in at p = 0.5 and throws away any that go over budget.
- Random search + repair draws the same way but fixes each sample with the GA's `repair()`.
- Greedy adds projects in profit/cost order while they fit.
- Exhaustive search checks all 2²⁰ subsets, which takes about 0.2 s. Pisinger (1997) gives faster exact methods for large instances.

Both random searches get the same number of evaluations as the GA.

### Experimental setup

| Setting | Value |
|---|---|
| Shortlists | top, banded, balanced |
| Independent runs | 30 per stochastic configuration |
| Seeds | 0 to 29, each run with its own freshly seeded generator |
| Population / generations | 50 / 100 |
| Mutation | 5% per gene |
| Budget | £3m (experiment 4 also tries £1.75m, £2.75m and £4m) |

Every stochastic configuration is run 30 times: each GA setting, random search and random search + repair. All of them use the same seeds, so runs can be paired across settings, although different algorithms use the random numbers differently. Greedy and exhaustive search are deterministic and run once. Parameters are changed one at a time from the defaults, following the systematic tuning approach of Eiben and Smit (2011), and experiment 2 also compares population sizes at a fixed number of evaluations.

Each summary table reports, over the 30 runs: mean profit, standard deviation, best and worst profit, the number of runs that reached the exact optimum, the mean gap to the optimum and the mean number of evaluations to the best solution. Convergence charts plot the mean best-so-far profit over the 30 runs, and `runs_log.csv` has every individual run.

Thirty runs show how reliable a setting is in a way one run can't. At GDV +10% on the top list the GA with default settings reaches the optimum in 10 of 30 runs (33%). A single run would have hit or missed by chance and suggested either that it always finds the optimum or that it never does. A 30/30 result is an observed rate on these seeds, not a guarantee.

In experiment 5 the 30 GA runs per scenario measure how reliably the GA finds that scenario's optimum. They don't make the land prices or sale values any more certain; that uncertainty is what the scenarios themselves test.

## Results

Figures are for the `top` shortlist at £3m unless stated. The full tables are in `results/<shortlist>/`, and every GA and random search run is listed in `runs_log.csv` there (1,290 runs per shortlist). Run counts like 29/30 are runs that reached the exact optimum out of 30.

| Shortlist | Exact optimum at £3m | Capital used | Projects |
|---|---|---|---|
| top | £2,270,035 | £2,902,981 | N02 Henry Road, N03 Hollyview Close, N04 Martindale Road, N05 Cherry Trees, N06 High Street Richmond |
| banded | £764,565 | £2,252,525 | six new builds (N03 to N08) |
| balanced | £878,921 | £2,812,513 | R01 113 Cortis Road and three new builds (N01 to N03) |

Every top-shortlist refurb is profitable, but each one returns less on its capital than these new builds, so the top optimum has none. The banded optimum leaves £0.75m unspent because every refurb in that list loses money at baseline costs. The balanced optimum is the only baseline optimum with a refurb in it.

Taking the `site_check` sites out of the pool left the top shortlist unchanged but moved the new-build draws for the other two. In v4 the banded optimum was £898,381 and included two flagged sites. It now has none, and neither does any other shortlist. In v6 the access fix moved 27 Princes Park Avenue out of the top shortlist and 1A Church Avenue in as N08. N08 isn't in the optimum, so the optimum is the same, but the GA runs below are on the new list and their figures changed.

### Valuation quality of the selected projects

`valuation_quality.csv` counts the valuation warnings in each shortlist and in its optimum. As a diagnostic it also re-solves with the warned projects removed. No replacements are drawn, so these are not benchmark results.

| Shortlist | Optimum projects with a warning | Warnings | Optimum without warned projects |
|---|---|---|---|
| top | 2 of 5 (N03, N06) | N06 has no postcode district and borough comparables; N03 uses borough comparables | £1,844,490 (18.7% lower) |
| banded | 4 of 6 (N03, N04, N05, N08) | 3 rest on fewer than 20 sales, 1 on comparables of all property types | £353,930 (53.7% lower) |
| balanced | 2 of 4 (R01, N02) | both rest on fewer than 20 sales | £689,910 (21.5% lower) |

Removing only the projects with no postcode district costs the top optimum £42,320 (to £2,227,715) and doesn't change the other two. The optimum leans on lower-confidence valuations in all three lists, most of all in the banded one, so the profit figures are only as good as the comparables behind them. Experiment 5 also records how many of each scenario's chosen projects carry a warning (`optimum_low_confidence`).

### How early the GA finds the optimum

With repair, the GA's final best appears after 212 fitness evaluations on average out of 4,850 (mean generation 3.9), counting one chromosome at a time. Repairing 50 random chromosomes produces near-greedy portfolios, so most runs have it within the first few generations. Under the penalty method it takes 981 evaluations (generation 19.9). On the banded list the figures are 75 and 569, and on the balanced list 10 and 816.

### Experiment 1: mutation rate

| Constraint | Mutation | top | Mean gap | Std dev | Worst run | Evaluations to best | balanced |
|---|---|---|---|---|---|---|---|
| repair | 1% | 23/30 | 0.15% | £9,705 | £2,238,110 | 124 | 30/30 |
| repair | 5% | 30/30 | 0% | £0 | £2,270,035 | 212 | 30/30 |
| repair | 10% | 30/30 | 0% | £0 | £2,270,035 | 69 | 30/30 |
| penalty | 1% | 16/30 | 1.31% | £62,036 | £1,967,418 | 1,365 | 13/30 |
| penalty | 5% | 30/30 | 0% | £0 | £2,270,035 | 981 | 30/30 |
| penalty | 10% | 30/30 | 0% | £0 | £2,270,035 | 1,646 | 30/30 |

At 1% mutation both methods converge too early, the penalty GA most: its worst top-list run is 13% short. On the balanced list the penalty GA at 1% has a mean gap of 6.7% and its worst run (£614,115) is 30% below the optimum, the worst result of any GA setting in the project. With repair, 5% and 10% always reach the optimum on the top list, and 10% gets there in about a third of the evaluations (69 against 212). With the penalty method, 5% and 10% always reach the optimum and 5% gets there in fewer evaluations. On the banded list every setting reaches the optimum in every run.

### Experiment 2: population size

| Constraint | Population | 100 generations | About 5,000 evaluations |
|---|---|---|---|
| repair | 20 | 29/30 (1,820 evaluations) | 29/30 (276 generations) |
| repair | 50 | 30/30 | 30/30 (103 generations) |
| repair | 100 | 30/30 (9,900 evaluations) | 30/30 (50 generations) |
| penalty | 20 | 27/30 (1,820 evaluations) | 30/30 |
| penalty | 50 / 100 | 30/30 | 30/30 |

With the penalty method, population 20 falls short only because it gets fewer evaluations. Given the same budget of about 5,000 it always reaches the optimum. The balanced list shows the same thing (24/30 at 100 generations, 30/30 at 276). With repair the bigger population helps beyond the extra evaluations: population 50 or 100 reaches the optimum every time, and population 20 misses in the same run whether it gets 100 or 276 generations, because that run settles on a portfolio £709 short early and never leaves it. Every other banded and balanced setting reaches the optimum in every run.

### Experiment 3: penalty vs repair

Both methods reach the optimum in all 30 runs on every list. Repair finds its best sooner: on the top list after 212 evaluations against 981, about a fifth as many. It takes about 12% longer per run (0.186 s against 0.166 s), because every chromosome is repaired.

### Experiment 4: algorithm comparison

Mean optimality gap over 30 runs (greedy is one run):

| Budget | GA + repair | Random | Random + repair | Greedy |
|---|---|---|---|---|
| top £1.75m | 0% | 42.9% | 0% | 9.2% |
| top £2.75m | 0% | 26.7% | 0% | 0% |
| top £3.00m | 0% | 20.8% | 0% | 1.9% |
| top £4.00m | 0% | 15.8% | 0% | 0% |
| banded £4.00m | 0% | 25.2% | 3.0% (0/30) | 0% |
| balanced £2.75m | 0% | 31.5% | 0% | 4.0% |

At every other banded and balanced budget the GA, random + repair and greedy all reach the optimum, and random search is 24% to 69% short. Mean run times are about 0.2 s for the GA, 0.05 s for random, 0.11 s for random + repair, a few milliseconds at most for greedy and 0.18 to 0.32 s for exhaustive.

Random search + repair reaches the optimum in every run on the top and balanced lists, so on 20 projects the repair heuristic does nearly all the work. Plain random search trails by 16% to 69%, because most of its samples go over budget and are thrown away. Its standard deviation is £53,000 to £428,000, while the GA's is £0 at every budget on every list.

The banded list at £4m is still the one case where the GA beats repaired random search. The budget covers every profitable project, so repair rarely triggers, and random samples keep loss-making projects that repair never removes. Random + repair ends 3.0% short in every run, while the GA, whose fitness pushes loss-makers out, reaches the optimum every time.

The balanced list didn't turn out much harder. Its optimum takes a refurb, so the algorithms can't ignore that half of the problem, but the GA with repair and random + repair still reach the optimum in every run at every budget, and greedy misses only at £2.75m. Only the penalty GA at 1% mutation struggles. A single budget constraint over 20 projects suits profit/cost sorting well, so the simple methods are competitive here, and separating them from the GA would take a larger candidate list.

### Experiment 5: sensitivity (30 GA runs per scenario)

| Scenario | top optimum | Mix | Greedy gap | GA at optimum | balanced optimum | Mix |
|---|---|---|---|---|---|---|
| baseline | £2,270,035 | 5 NB | 1.9% | 30/30 | £878,921 | 1 R + 3 NB |
| allowance 5% | £2,588,866 | 6 NB | 9.1% | 27/30 | £1,006,763 | 1 R + 3 NB |
| allowance 15% | £2,109,116 | 5 NB | 0.3% | 30/30 | £751,080 | 1 R + 3 NB |
| NB build £2,500/m² | £1,970,385 | 5 NB | 3.4% | 30/30 | £741,097 | 1 R + 3 NB |
| NB build £2,800/m² | £1,587,836 | 5 NB | 0.02% | 30/30 | £622,794 | 3 R + 1 NB |
| land price ×1.25 | £1,932,455 | 5 NB | 2.3% | 30/30 | £725,078 | 2 R + 2 NB |
| land price ×1.50 | £1,581,715 | 5 NB | 5.2% | 29/30 | £610,584 | 3 R + 1 NB |
| land price ×2.00 | £1,138,024 | 4 NB | 0% | 30/30 | £516,534 | 3 R + 1 NB |
| GDV −10% | £1,752,733 | 5 NB | 1.4% | 30/30 | £509,778 | 1 R + 3 NB |
| GDV +10% | £2,793,853 | 6 NB | 2.4% | 10/30 | £1,248,064 | 1 R + 3 NB |
| GDV top quartile | £3,049,090 | 6 NB | 5.6% | 15/30 | £1,334,959 | 1 R + 3 NB |

The chart (`exp5_sensitivity.png`) shows the optimal modelled profit for each scenario against the baseline. The CSV now also has the GA's best profit and mean evaluations to best for each scenario.

No scenario puts a refurb in the top optimum. Even with plot prices doubled or a £2,800/m² build cost, the best new builds return more on cost than the best refurb. The balanced list responds much more: dearer plots or building costs swap new builds for refurbs, and from land ×1.50 the optimum is three refurbs and one new build. On the banded list one refurb (R10) enters the optimum at a 5% allowance and at both higher GDVs. Plot prices move profit the most: doubling them halves the top optimum and cuts the balanced one by 41%.

GDV +10% is the hardest instance in the project. The GA with default settings reaches the optimum in only 10 of 30 runs on the top list, though its mean gap is 0.16% and its worst run is £6,516 short, so a low success rate can go with a small loss in quality. The second-best portfolio is only £5,106 below the optimum and the third £6,516, and the cheapest-ratio-first repair tends to settle on one of those. The top-quartile GDV scenario is similar: 15 of 30 runs, a mean gap of 0.58%, and the worst run £35,162 short, which is the third-best portfolio. Greedy misses in 10 of the 11 top scenarios, by up to 9.1%. On the banded and balanced lists the GA and greedy reach the optimum in every scenario.

## Source files and rebuilding

Everything needed to run the experiments is included: `data/raw_candidates.csv`, both shortlists and the Datahub extract. Anyone cloning the project can reproduce the results with the commands under "Running it" and doesn't need to download anything.

To rebuild the candidate pool from the original sources:

```
python download_data.py          # about 1.5 GB downloaded, roughly 700 MB kept in data/raw/
python prepare_data.py --rebuild # recreates data/raw_candidates.csv and both shortlists
```

`data/raw/` isn't included because of its size. The sources are:

| File | Source |
|---|---|
| `data/raw/price_paid/pp-2024.csv` | HM Land Registry Price Paid Data, 2024 yearly file: https://www.gov.uk/government/statistical-data-sets/price-paid-data-downloads |
| `data/raw/hpm_london/hpm_la_2024/<Borough>_link_26122024.csv` (33 files) | Chi et al. (2025), DOI 10.5255/UKDA-SN-857911. The full release is on UK Data Service ReShare. The script takes the London Datastore copy (`hpm_la_2024.zip`, https://data.london.gov.uk/dataset/house-price-per-square-metre-in-england-and-wales) and keeps only the London borough files |
| `data/raw/brownfield/brownfield-land.csv` | Brownfield land registers, all English LPAs: https://www.planning.data.gov.uk/dataset/brownfield-land |
| `data/raw/land_values/Land_value_estimates_for_policy_appraisal_2023.xlsx` | MHCLG: https://www.gov.uk/government/publications/land-value-estimates-for-policy-appraisal-2023 |
| `data/pld_small_new_build.json` (included) | Planning London Datahub, https://www.london.gov.uk/programmes-strategies/planning/digital-planning/planning-london-datahub |

The Datahub is queried through its guest read-only API at `https://planningdata.london.gov.uk/api-guest/applications/_search`. It needs the `X-API-AllowRequest` header, and the value is published in the GLA's API connection guide (https://www.london.gov.uk/sites/default/files/planninglondondatahub_api_connection_technical_documentation_v1.pdf), so no account or key is needed. The Datahub is updated daily: application statuses change and new decisions come in, so a fresh download won't match the included extract. `download_data.py` therefore keeps the included file and only replaces it with `--refresh-datahub`. The extract records its download date and the exact query.

## Limitations

- Refurb prices are real 2024 sales, used as if each house could be bought at that price, while plot prices are modelled, so the two strategies are priced on different bases. MHCLG says its land value estimates are for policy appraisal and don't represent the market value of individual sites. They are used here as hypothetical inputs, so new-build profits are model outputs, not developer returns anyone could expect, and experiment 5 shows how much they depend on the land figure. Combined with the low £2,200/m² build cost, the estimates probably flatter new-build margins.
- New-build GDV uses EPC C+ sales of the property type in the area, not new builds, and 20 of the 82 sites have no postcode district and get borough values. `valuation_warnings` marks these.
- Plot prices are estimates, not purchase prices. MHCLG's figures are borough averages for policy appraisal, and the density band and location factor are my own steps on top.
- The demolition and access allowances are flat assumptions, not site estimates. Sites that also alter a neighbouring house are flagged (`site_check`) and kept out of the shortlists rather than costed.
- The bedroom check only catches disagreements the description states explicitly. It found one; a description that doesn't mention bedrooms can't be checked.
- New-build floor areas come from the planning application forms, which applicants fill in themselves. Obvious errors are caught (homes under 37 m² or over 250 m², duplicated rows), but the rest are taken as given.
- "Not started" relies on the Datahub status, which boroughs don't always update when work begins. A permission granted in 2022 may also have been sold on or built out by now.
- Refurb candidates are houses that sold cheaply against their comparables. A low price can also reflect things the data doesn't show, such as a short lease on part of the site, a bad location within the district, or structural problems that would cost more than £1,800/m².
- The cost model leaves out stamp duty, sale costs, finance, professional fees, CIL/S106, utility connections and VAT, except through the single 10% allowance, which probably doesn't cover them all for a London scheme.
- The refurb projects are appraisals, not observed outcomes: each starts from a real purchase but its works cost and resale are modelled, and the comparables include sales after the purchase date.
- On 20 projects repaired random sampling does as well as the GA on all three shortlists, except where the budget is loose enough that repair never removes the loss-makers (banded list at £4m). The balanced benchmark didn't change that. A clearer GA advantage would need a larger candidate list.
- Profit is a single expected value. Xu et al. (2025) optimise REIT portfolios on return and risk using price history, which physical projects don't have.

## References

Chu, P.C. and Beasley, J.E. (1998) 'A genetic algorithm for the multidimensional knapsack problem', *Journal of Heuristics*, 4, pp. 63–86. https://doi.org/10.1023/A:1009642405419

Chi, B., Dennett, A., Oléron-Evans, T. and Morphet, R. (2025) *House Price Per Square Metre in England and Wales, 1995–2024*. UK Data Service, University College London. https://doi.org/10.5255/UKDA-SN-857911

Deb, K. (2000) 'An efficient constraint handling method for genetic algorithms', *Computer Methods in Applied Mechanics and Engineering*, 186(2–4), pp. 311–338. https://doi.org/10.1016/S0045-7825(99)00389-8

Eiben, A.E. and Smit, S.K. (2011) 'Parameter tuning for configuring and analyzing evolutionary algorithms', *Swarm and Evolutionary Computation*, 1(1), pp. 19–31. https://doi.org/10.1016/j.swevo.2011.02.001

Goldberg, D.E. (1989) *Genetic Algorithms in Search, Optimization, and Machine Learning*. Addison-Wesley. ISBN 9780201157673

He, J., Zhou, Y. and Yao, X. (2008) 'A comparison of GAs penalizing infeasible solutions and repairing infeasible solutions on the 0-1 knapsack problem', *Dagstuhl Seminar Proceedings*. https://doi.org/10.4230/DagSemProc.08051.3

HM Land Registry, *Price Paid Data*. https://www.gov.uk/government/statistical-data-sets/price-paid-data-downloads

Department for Communities and Local Government (2015) *Technical housing standards: nationally described space standard*. https://www.gov.uk/government/publications/technical-housing-standards-nationally-described-space-standard

Ministry of Housing, Communities and Local Government (2026) *Land value estimates for policy appraisal 2023*. https://www.gov.uk/government/publications/land-value-estimates-for-policy-appraisal-2023

Ministry of Housing, Communities and Local Government, *Brownfield land* dataset. https://www.planning.data.gov.uk/dataset/brownfield-land

Greater London Authority, *Planning London Datahub*. https://www.london.gov.uk/programmes-strategies/planning/digital-planning/planning-london-datahub

Pisinger, D. (1997) 'A minimal algorithm for the 0-1 knapsack problem', *Operations Research*, 45(5), pp. 758–767. https://doi.org/10.1287/opre.45.5.758

Xu et al. (2025) 'Utilizing the real estate investment trusts for portfolio optimisation by application of genetic algorithm', *Humanities and Social Sciences Communications*, 12, Article 476. https://doi.org/10.1057/s41599-025-04715-0

Contains HM Land Registry data © Crown copyright and database right 2024, licensed under the Open Government Licence v3.0.
