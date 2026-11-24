"""Cost and profit model for a single project.

Total Investment = (Acquisition + Works) x (1 + allowance_rate)
  Works      = construction (rate x floor area) + any abnormal costs
  Allowance  = allowance_rate x (Acquisition + Works), 10% by default

The sale value is an estimated GDV: floor area times the local price per
m2 for homes in good condition (see prepare_data.py). It isn't a valuation.
"""
import math

REFURB_RATE = 1800        # £ per m2
NEW_BUILD_RATE = 2200     # £ per m2
ALLOWANCE_RATE = 0.10     # fees, contingency, finance etc.


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _check_positive(project, field):
    value = project.get(field)
    if not _is_number(value) or value <= 0:
        name = project.get("project_id", project.get("candidate_id", "project"))
        raise ValueError(f"{name}: {field} must be a positive number, got {value!r}")
    return value


def _check_setting(name, value, minimum, allow_equal=True):
    """Sensitivity settings: finite and at least (or above) minimum."""
    if not _is_number(value) or value < minimum or (value == minimum and not allow_equal):
        bound = ">=" if allow_equal else ">"
        raise ValueError(f"{name} must be a finite number {bound} {minimum}, got {value!r}")


def calculate_project(project, allowance_rate=ALLOWANCE_RATE, refurb_rate=REFURB_RATE,
                      new_build_rate=NEW_BUILD_RATE, land_price_multiplier=1.0,
                      sale_value_multiplier=1.0):
    """Fill in the cost and profit fields for one project (a dict).

    Returns (total_investment, expected_profit). building_cost and
    additional_costs are written back onto the dict as well so they can
    be kept as separate columns in the projects table.

    land_price_multiplier only applies to NEW_BUILD plots, whose prices are
    estimates. sale_value_multiplier scales the estimated GDV for both.
    Every input must be a finite number in a sensible range, otherwise
    ValueError.
    """
    _check_setting("allowance_rate", allowance_rate, 0)
    for name, value in [("refurb_rate", refurb_rate), ("new_build_rate", new_build_rate),
                        ("land_price_multiplier", land_price_multiplier),
                        ("sale_value_multiplier", sale_value_multiplier)]:
        _check_setting(name, value, 0, allow_equal=False)
    if project["strategy"] == "REFURB":
        rate = refurb_rate
        acquisition = _check_positive(project, "acquisition_price")
    elif project["strategy"] == "NEW_BUILD":
        rate = new_build_rate
        acquisition = _check_positive(project, "acquisition_price") * land_price_multiplier
    else:
        raise ValueError(f"unknown strategy: {project['strategy']}")

    floor_area = _check_positive(project, "floor_area_m2")
    price_per_m2 = _check_positive(project, "local_price_per_m2")
    abnormal = project.get("abnormal_costs", 0)
    if not _is_number(abnormal) or abnormal < 0:
        raise ValueError(f"abnormal_costs must be a finite number >= 0, got {abnormal!r}")

    building_cost = rate * floor_area
    works = building_cost + abnormal
    additional_costs = allowance_rate * (acquisition + works)
    total_investment = acquisition + works + additional_costs

    estimated_gdv = floor_area * price_per_m2 * sale_value_multiplier
    expected_profit = estimated_gdv - total_investment

    project["acquisition_cost"] = round(acquisition, 2)   # after any land multiplier
    project["building_cost"] = round(building_cost, 2)
    project["additional_costs"] = round(additional_costs, 2)
    project["total_investment"] = round(total_investment, 2)
    project["estimated_gdv"] = round(estimated_gdv, 2)
    project["expected_profit"] = round(expected_profit, 2)

    return project["total_investment"], project["expected_profit"]


if __name__ == "__main__":
    # quick sanity check against a hand calculation
    test = {"strategy": "NEW_BUILD", "floor_area_m2": 130,
            "acquisition_price": 220000, "local_price_per_m2": 5400}
    cost, profit = calculate_project(test)
    # (220000 + 286000) * 1.1 = 556600, sale = 702000
    print(cost, profit)
    assert cost == 556600 and profit == 145400
