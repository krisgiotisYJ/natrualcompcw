"""Build the projects table from the real data sources.

Stage 1 (python prepare_data.py --rebuild, slow, needs data/raw/ - see
download_data.py):
  REFURB candidates   - 2024 London house sales from HM Land Registry Price
                        Paid Data, linked to EPC floor areas by Chi, Dennett,
                        Oleron-Evans & Morphet (2025). A house counts as a
                        refurb opportunity if it sold in the cheapest quarter
                        of comparable sales (same postcode district, house
                        type and construction age band). The purchase price
                        is real; the refurbishment and the resale are
                        modelled. Nothing here is a record of a house that
                        was actually refurbished and resold.
  NEW_BUILD candidates - permissions for 1-4 new-build homes, decided
                        2022-2024 and not started, from the Planning London
                        Datahub (data/pld_small_new_build.json). Only vacant
                        land, garden plots and garage/structure sites are
                        kept (plot_type); extensions, conversions and
                        mixed-use schemes are dropped.
  Sale values (GDV)    - estimated. Refurbs: median £/m2 of the same
                        comparable group. New builds: median £/m2 of 2024
                        EPC C+ sales of the same property type in the
                        postcode district, falling back to the borough and
                        then to all types. The 75th percentile is kept as
                        well for the sensitivity run.
  Plot prices          - estimated: MHCLG Land value estimates for policy
                        appraisal 2023, converted to a value per plot and
                        scaled by a location factor. The site area, density
                        band, per-plot value and factor are all kept in the
                        table so each estimate can be checked by hand.
  Data checks          - bedroom_conflict, site_check and
                        valuation_warnings columns (see below).
  Writes data/raw_candidates.csv (the whole screened pool).

Stage 2 (python prepare_data.py, quick):
  Classifies data/raw_candidates.csv, works out costs/profit and picks a
  20-project shortlist. Three shortlists are available:
    "top"      - 12 REFURB + 8 NEW_BUILD with the highest return on cost
    "banded"   - the same counts drawn evenly from the low, middle and high
                 return thirds of each pool, with a fixed seed
    "balanced" - the same counts, half profitable and half loss-making in
                 each strategy, with the same fixed seed
  All three are written out (data/projects_<name>.csv).

Classification rules:
  existing_dwelling with a floor area      -> REFURB
  development_plot with a proposed area    -> NEW_BUILD
(plot_type says which kind of development plot: vacant land, garden plot or
garage redevelopment.)

Two kinds of site stay in raw_candidates.csv but are left out of the
shortlists: sites whose planning description and unit records disagree on
the number of bedrooms (the two readings give different floor areas, and
only the drawings could settle which is right), and site_check sites, which
also alter the host or a neighbouring house in ways the model doesn't cost.

Several inputs are estimates rather than measurements, and the table says
which: price_source_type, floor_area_source, valuation_basis,
comparable_count and the notes column.
"""
import glob
import json
import re
import sys

import numpy as np
import pandas as pd

from calculate_costs import calculate_project

RAW_PATH = "data/raw_candidates.csv"
PLD_PATH = "data/pld_small_new_build.json"

PRICE_PAID_PATH = "data/raw/price_paid/pp-2024.csv"
LINKED_GLOB = "data/raw/hpm_london/hpm_la_2024/*.csv"
BROWNFIELD_PATH = "data/raw/brownfield/brownfield-land.csv"
LAND_VALUE_PATH = "data/raw/land_values/Land_value_estimates_for_policy_appraisal_2023.xlsx"

N_REFURB = 12
N_NEW_BUILD = 8
# a project costing more than a third of the £3m budget would leave room for
# only one or two others, so those are screened out of the shortlist
MAX_PROJECT_COST = 1_000_000
SHORTLIST_SEED = 42   # fixed before any algorithm was compared on the shortlists

EPC_C_OR_BETTER = 69   # SAP score threshold for EPC band C
MIN_SALES_FOR_VALUE = 10
# below this many comparable sales a median £/m2 moves a lot with one sale
FEW_COMPARABLES = 20

# Refurb screen. A poor EPC rating doesn't mean a house needs work: old
# London houses rate E even after a full renovation, and most certificates
# in the linked data were 5-15 years old at the time of sale. Selling well
# below comparable houses is a better sign, so that is what is used.
COMPARABLE_GROUP = ["postcode_district", "propertytype", "CONSTRUCTION_AGE_BAND"]
MIN_COMPARABLE_SALES = 5
CHEAP_QUANTILE = 0.25

# Demolishing garages / sheds and clearing the site. Not in the £/m2 build
# rate, so added as a lump sum. This is my own allowance, not a quote.
DEMOLITION_ALLOWANCE = 20_000
# New or resurfaced access to a backland plot, same idea
ACCESS_ALLOWANCE = 10_000

# Nationally Described Space Standard minimum sizes (m2) by bedrooms, for
# single storey homes (flats, bungalows) and two storey houses. The NDSS
# sets several occupancies per bedroom count; these are the 2-person 1-bed
# and beds + 2 persons from 2 beds up (2b4p, 3b5p, ... 6b8p). The NDSS
# stops at 6 bedrooms, so a bigger home needs a floor area from the
# application or it is left out.
NDSS_FLAT = {0: 39, 1: 50, 2: 70, 3: 86, 4: 99, 5: 112, 6: 125}
NDSS_HOUSE = {1: 58, 2: 79, 3: 93, 4: 106, 5: 119, 6: 132}
DEFAULT_BEDS = 2

# Datahub unit types that get the house NDSS sizes
HOUSE_UNIT_TYPES = {"Terraced Home", "Semi Detached Home", "Detached Home", "House or Bungalow"}
# A unit GIA outside this range is a typing error on the application form
# (the extract has 1 m2 homes and a 6,700 m2 flat)
PLAUSIBLE_UNIT_GIA = (37, 250)

PRICE_PAID_COLUMNS = ["transactionid", "pp_price", "date", "pp_postcode", "ptype", "old_new",
                      "pp_duration", "paon", "saon", "street", "locality", "town", "district",
                      "county", "ppd_category", "record_status"]

POSTCODE_RE = re.compile(r"\b([A-Z]{1,2}\d[A-Z\d]?)\s*\d[A-Z]{2}\b")
NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}

# Existing structures to clear. Only demolition / removal counts: a scheme
# "with an associated garage" is building one, not knocking one down.
DEMOLITION_RE = (r"demoli|\bremov(?:al|e) of\b|take down|scrap yard|disused|"
                 r"existing (?:\w+ ){0,3}(?:garages?|outbuildings?|sheds?|structures?|stores?|buildings?)")

# What kind of site each Datahub permission is (plot_type column). Only
# VACANT_LAND, GARDEN_PLOT and GARAGE_REDEVELOPMENT fit the model of buying
# a plot at land value and building on it.
#   EXTENSION_CONVERSION - the new home is made by extending or converting
#                          an existing building, or is an annexe
#   MIXED_USE            - the scheme includes shop / office / garage space
#                          that the model neither costs nor values
#   GARAGE_REDEVELOPMENT - garages, sheds, stables or other structures are
#                          demolished first (gets the demolition allowance)
#   GARDEN_PLOT          - built in the garden or side land of a house
#   VACANT_LAND          - none of the above, or a site the application
#                          calls vacant land with no mention of a garden
PLOT_TYPES_KEPT = {"VACANT_LAND", "GARDEN_PLOT", "GARAGE_REDEVELOPMENT"}
EXTENSION_CONVERSION_RE = (
    r"roof ?(?:top|extension|level|storey)|mansard|upward extension|additional (?:floor|stor)|"
    r"extra (?:floor|stor)|loft|change of use|conver|sub-?divi|knock through|amalgamat|"
    r"join (?:two|the) (?:dwellings|properties)|annexe|"
    r"extension[^.;]{0,80}?\b(?:to|and) (?:create|provide|contain|form)\b")
MIXED_USE_RE = (r"commercial (?:use|unit)|office (?:on|space|floorspace)|additional office|"
                r"extension to existing (?:shop|dental|office|commercial)|garage units at ground")
GARDEN_PLOT_RE = (r"garden|rear of|to the rear|adjacent|adjoining|attached|to the side|"
                  r"side elevation|end (?:of )?terrace|land part of|curtilage")
# Works to the house next door or the host house (a new extension, roof
# changes, moving a door). The flat demolition allowance wasn't meant to
# cover these, so the site gets a site_check flag instead of a guessed cost
# and isn't shortlisted.
SITE_CHECK_RE = (r"host property|existing dwelling|conservatory|utility room|relocation of|"
                 r"rear extensions?\b(?! to (?:create|provide))|first floor rear|hip to gable|"
                 r"roof (?:alterations|enlargement)|alterations to (?:roof|fenestration)|"
                 r"porch to \d|^alterations")

# Datahub unit types -> property type groups in the sales data
UNIT_TYPE_GROUPS = {"Terraced Home": "T", "Semi Detached Home": "S", "Detached Home": "D",
                    "House or Bungalow": "house", "Flat Apartment Maisonette": "F"}
SALES_TYPE_GROUPS = {"D": ["D"], "S": ["S"], "T": ["T"], "F": ["F"],
                     "house": ["D", "S", "T"], "any": ["D", "S", "T", "F"]}


# ---- stage 1: real sources -> raw_candidates.csv ----

def load_linked_sales():
    """2024 London sales with EPC floor area, joined to Price Paid for address."""
    price_paid = pd.read_csv(PRICE_PAID_PATH, header=None, names=PRICE_PAID_COLUMNS)
    price_paid = price_paid[price_paid.county == "GREATER LONDON"]

    linked = pd.concat(pd.read_csv(f, low_memory=False) for f in sorted(glob.glob(LINKED_GLOB)))
    linked = linked[linked.year == 2024]

    sales = linked.merge(price_paid, on="transactionid")
    sales["postcode_district"] = sales.postcode.str.split().str[0]
    print(f"{len(sales)} linked 2024 London sales", file=sys.stderr)
    return sales


def q75(s):
    return s.quantile(0.75)


def good_condition_values(sales):
    """Median, 75th percentile and count of £/m2 for EPC C+ sales, by postcode
    district and by borough, for each property type group.

    Only standard market sales (Price Paid category A) are used; category B
    covers repossessions, buy-to-let and transfers to companies, which
    aren't open-market prices. EPC C or better is a proxy for a home in good
    condition, not evidence that it is new or finished to a developer's
    resale standard."""
    good = sales[(sales.CURRENT_ENERGY_EFFICIENCY >= EPC_C_OR_BETTER)
                 & (sales.ppd_category == "A")]
    tables = {}
    for level, key in [("district", "postcode_district"), ("borough", "lad23cd")]:
        frames = []
        for group, types in SALES_TYPE_GROUPS.items():
            stats = good[good.propertytype.isin(types)].groupby(key).priceper.agg(
                ["median", q75, "count"])
            stats.index = pd.MultiIndex.from_product([stats.index, [group]])
            frames.append(stats)
        tables[level] = pd.concat(frames)
    return tables


def new_build_value(tables, district, la_code, type_group):
    """£/m2 for new homes of one type group: the most local group with at
    least MIN_SALES_FOR_VALUE sales, falling back from district + type to
    borough + type, then district and borough over all property types.

    Returns (median, q75, count, basis)."""
    for level, place, group in [("district", district, type_group),
                                ("borough", la_code, type_group),
                                ("district", district, "any"),
                                ("borough", la_code, "any")]:
        table = tables[level]
        if (place, group) in table.index:
            row = table.loc[(place, group)]
            if row["count"] >= MIN_SALES_FOR_VALUE:
                label = "all types" if group == "any" else f"type {group}"
                return row["median"], row["q75"], int(row["count"]), f"{level}, {label}"
    raise ValueError(f"no sales to value {district} / {la_code}")


def site_type_group(units):
    groups = {UNIT_TYPE_GROUPS.get(u.get("unit_type"), "any") for u in units}
    return groups.pop() if len(groups) == 1 else "any"


def refurb_candidates(sales):
    houses = sales[
        sales.propertytype.isin(["D", "S", "T"])
        & (sales.duration == "F")
        & (sales.ppd_category == "A")       # standard market sales only
        & (sales.old_new == "N")
        & sales.tfarea.between(50, 250)
        & sales.CONSTRUCTION_AGE_BAND.fillna("").str.startswith("England and Wales")
    ].copy()

    group = houses.groupby(COMPARABLE_GROUP).priceper
    houses["n_comparable"] = group.transform("size")
    houses["price_rank"] = group.rank(pct=True)
    houses["local_price_per_m2"] = group.transform("median")
    houses["local_price_q75"] = group.transform(q75)

    houses = houses[
        (houses.n_comparable >= MIN_COMPARABLE_SALES)
        & (houses.price_rank <= CHEAP_QUANTILE)
        # under half the group median is more likely a bad floor-area match
        # or a non-market sale than a real bargain
        & (houses.priceper >= 0.5 * houses.local_price_per_m2)
    ]

    address = (houses.paon.astype(str) + " " + houses.street.fillna("")).str.title()
    age = houses.CONSTRUCTION_AGE_BAND.str.replace("England and Wales: ", "")
    notes = ("sold " + houses.dateoftransfer.astype(str)
             + "; cheapest " + (100 * houses.price_rank).round(0).astype(int).astype(str)
             + "% of " + houses.n_comparable.astype(str) + " comparable sales (built " + age
             + "); EPC " + houses.CURRENT_ENERGY_EFFICIENCY.astype(str))
    return pd.DataFrame({
        "reference": houses.transactionid,
        "address": address,
        "borough": houses.district.str.title(),
        "postcode_district": houses.postcode_district,
        "site_type": "existing_dwelling",
        "existing_floor_area_m2": houses.tfarea,
        "proposed_floor_area_m2": None,
        "floor_area_source": "EPC",
        "acquisition_price": houses.price,
        "price_source_type": "recorded sale (Price Paid 2024)",
        "abnormal_costs": 0,
        "local_price_per_m2": houses.local_price_per_m2.round(0),
        "local_price_q75": houses.local_price_q75.round(0),
        "comparable_count": houses.n_comparable,
        "valuation_basis": "district, type and age band",
        "plot_type": None,
        "bedroom_conflict": "",
        "site_check": False,
        "epc_score": houses.CURRENT_ENERGY_EFFICIENCY,
        "notes": notes,
        "source": "Price Paid Data 2024 + EPC (Chi et al. 2025)",
    })


def land_value_sheet():
    sheet = pd.read_excel(LAND_VALUE_PATH, sheet_name="Residential", header=None)
    return sheet[sheet[1] == "London"]


def load_land_values(london):
    """London residential land value per dwelling plot, by borough code.

    The MHCLG sheet gives £/ha at low, medium and high density, each with a
    typical dwellings/ha. Dividing one by the other gives a value per plot.
    Median house price column is used throughout.
    """
    values = pd.DataFrame({
        "la_code": london[2],
        "dph_low": london[4], "dph_medium": london[5], "dph_high": london[6],
        "per_ha_low": london[9], "per_ha_medium": london[14], "per_ha_high": london[19],
    })
    return values.set_index("la_code").astype(float)


def la_code_for(lpa_name, codes):
    """Datahub names are short ('Kingston', 'Barking & Dagenham')."""
    name = lpa_name.replace("&", "and")
    for full, code in codes.items():
        if full == name or full.startswith(name + " upon"):
            return code
    return None   # LLDC and OPDC aren't boroughs


def plot_value(row, land_values):
    """MHCLG density band and land value per plot for one site.

    The band is the one whose typical dwellings/ha is closest to the site's
    own density. Sites with no site area get the low band, the one that
    suits 1-4 homes on a garden or garage plot best.
    Returns (band, value per plot)."""
    lv = land_values.loc[row["la_code"]]
    bands = ["low", "medium", "high"]
    if pd.notna(row["hectares"]) and row["hectares"] > 0:
        density = row["dwellings"] / row["hectares"]
        band = min(bands, key=lambda b: abs(lv[f"dph_{b}"] - density))
    else:
        band = "low (no site area)"
    key = band.split()[0]
    return band, lv[f"per_ha_{key}"] / lv[f"dph_{key}"]


def ndss_size(beds, house):
    """NDSS minimum for a home with this many bedrooms, or None above 6."""
    sizes = NDSS_HOUSE if house else NDSS_FLAT
    if beds == 0:
        return NDSS_FLAT[0]   # studios are always single storey
    return sizes.get(beds)


def description_bedrooms(description, dwellings):
    """Bedrooms of each new home as the planning description gives them,
    sorted, or None if it doesn't say.

    Handles '3 x 1-bed flats', '1 x three bed, 1 x two bed', 'one single
    storey (2 bed) and one two storey (3 bed)', '3-bed house' (applied to
    every home) and 'studio'.
    """
    text = str(description).lower() if pd.notna(description) else ""
    num = r"(\d|one|two|three|four|five|six)"

    def to_int(s):
        return int(s) if s.isdigit() else NUMBER_WORDS[s]

    pairs = re.findall(num + r"\s*(?:no\.?)?\s*x\s*" + num + r"[\s-]*bed", text)
    if pairs:
        return sorted(b for n, beds in pairs for b in [to_int(beds)] * to_int(n))
    singles = [to_int(b) for b in re.findall(r"\b" + num + r"[\s-]*bed", text)]
    if len(singles) == 1:
        return [singles[0]] * dwellings
    if singles and len(singles) == dwellings:
        return sorted(singles)
    if singles:
        return None
    if "studio" in text:
        return [0] * dwellings
    return None


def proposed_floor_area(notes, dwellings):
    """Floor area from the planning description, using NDSS minimum sizes.

    Returns (area, source); area is None for a home bigger than the NDSS
    covers. If no bedroom count can be read every dwelling is assumed to be
    a 2-bed.
    """
    text = str(notes).lower() if pd.notna(notes) else ""
    house = bool(re.search(r"house|bungalow|detached|terrace", text))
    beds = description_bedrooms(notes, dwellings)
    source = "planning description + NDSS minimum"
    if beds is None:
        beds, source = [DEFAULT_BEDS] * dwellings, "assumed minimum (2-bed NDSS)"
    sizes = [ndss_size(b, house) for b in beds]
    if None in sizes:
        return None, "excluded: more bedrooms than the NDSS covers"
    return sum(sizes), source


def bedroom_conflict(units, description):
    """Text describing a disagreement between the bedrooms in the planning
    description and in the Datahub unit records, or "" if they agree or
    either one is silent.

    113 Gregory Crescent is the case that prompted this: the description
    says '1 x 2 bed', the unit record says 3 bedrooms and gives no floor
    area, so the NDSS size depends on which one is believed (79 or 93 m2).
    """
    recorded = [u.get("no_bedrooms") for u in units]
    if not recorded or None in recorded:
        return ""
    recorded = sorted(int(b) for b in recorded)
    described = description_bedrooms(description, len(units))
    if described is None or described == recorded:
        return ""
    return f"description says {described} bedrooms, unit records say {recorded}"


def unit_floor_area(units, description):
    """Floor area of the new homes on a Datahub application.

    Returns (area, source). Uses each unit's GIA from the application form.
    A unit whose GIA is missing or implausible gets the NDSS minimum for its
    bedroom count instead; if a unit has no bedroom count either, the whole
    site falls back to reading the description. area is None when a home
    has no usable GIA and more bedrooms than the NDSS covers.
    """
    area, estimated = 0, False
    for unit in units:
        gia = unit.get("gia") or 0
        if PLAUSIBLE_UNIT_GIA[0] <= gia <= PLAUSIBLE_UNIT_GIA[1]:
            area += gia
            continue
        if unit.get("no_bedrooms") is None:
            return proposed_floor_area(description, len(units))
        size = ndss_size(int(unit["no_bedrooms"]), unit.get("unit_type") in HOUSE_UNIT_TYPES)
        if size is None:
            return None, "excluded: more bedrooms than the NDSS covers"
        area += size
        estimated = True
    return area, "Datahub bedrooms + NDSS minimum" if estimated else "Datahub unit GIA"


def abnormal_works(notes):
    """Look for works the £/m2 rate doesn't cover.

    Returns (exclude, extra_cost, note). Basements are excluded because the
    description never says how big they are, so there is nothing to price.
    Demolition / clearance and access works get lump-sum allowances.
    """
    text = str(notes).lower()
    if "basement" in text:
        return True, 0, "excluded: basement excavation, cost unknown"
    extra, notes = 0, []
    if re.search(DEMOLITION_RE, text):
        extra += DEMOLITION_ALLOWANCE
        notes.append(f"£{DEMOLITION_ALLOWANCE:,} demolition/clearance")
    if re.search(r"access road|re-?surfacing|new (?:vehicular )?access|formation of (?:a |an )?(?:new )?"
                 r"(?:means of |vehicular )?access", text):
        extra += ACCESS_ALLOWANCE
        notes.append(f"£{ACCESS_ALLOWANCE:,} access works")
    if notes:
        return False, extra, "includes " + " + ".join(notes) + " allowance (assumed)"
    return False, 0, ""


def site_address(site_name, number, street):
    """One-line address from the Datahub's free-text site name and street.

    Site names are often multi-line with the postcode on the end, or just a
    fragment like 'Land To Rear Of' or 'Development At 63' with the street
    held separately.
    """
    def tidy(text):
        return text.title() if text.isupper() else text

    # postcodes and 'London' can be on their own line or after a comma
    parts = re.split(r"[\r\n]+|,", str(site_name))
    lines = [tidy(l.strip()) for l in parts
             if l.strip() and l.strip().upper() != "LONDON"
             and not POSTCODE_RE.fullmatch(l.strip().upper())]
    # a line ending 'Rear Of' / 'Adjacent To' runs on into the next one
    address = re.sub(r"\b(at|of|to|r/o), ", r"\1 ", ", ".join(lines), flags=re.I)
    street = tidy(str(street)) if street else ""
    if street and street.upper() not in address.upper():
        if re.search(r"\d+[a-z]?$", address, re.I):         # 'Development At 63'
            address = f"{address} {street}"
        else:
            sep = " " if re.search(r"\b(at|of|to|r/o)$", address, re.I) else ", "
            street_part = " ".join(str(x) for x in [number, street] if x)
            address = sep.join(x for x in [address, street_part] if x)
    return address


def stated_dwellings(description):
    """Number of new homes the description gives, or None if it doesn't say
    or gives more than one count.

    '2 x two storey ... dwellings' -> 2, 'Erection of 4no. 2-bedroom houses'
    -> 4, 'a pair of ... dwellings' -> 2, and a description that only ever
    uses the singular ('detached house') -> 1.
    """
    text = description.lower()
    plural = r"(?:dwellings|dwellinghouses|dwelling houses|houses|homes|bungalows|flats|units|apartments)"
    if re.search(r"\bpair of\b(?:[\w,-]+\s+){0,6}?" + plural, text):
        return 2
    # '4no.', '2x4 bedroom', '3 x' all give the count; '2-bed', 'two storey'
    # and 'three story building' don't
    counts = re.findall(r"\b(\d|two|three|four)(?:\s*(?:no\.?|x)(?=[\s\d])|\b)"
                        r"(?![\s-]*(?:bed|stor))\s*(?:[\w,-]+\s+){0,6}?" + plural + r"\b", text)
    if len(counts) > 1:   # '2 houses and 3 flats': too easy to misread
        return None
    if counts:
        n = counts[0]
        return int(n) if n.isdigit() else NUMBER_WORDS[n]
    if re.search(r"\b" + plural + r"\b", text):
        return None
    if re.search(r"\b(?:dwelling|dwellinghouse|house|bungalow|flat|unit)\b", text):
        return 1
    return None


def drop_duplicate_units(units, description):
    """Remove repeated unit rows when the description gives fewer homes.

    Some boroughs enter every home twice (Havering does it on most
    applications, e.g. 'Proposed two storey two bedroom detached house' with
    two identical 72 m2 rows), and some list a home once with its floor
    area and once without. A row only counts as a repeat if an earlier row
    has the same type and bedrooms and the same GIA, or the repeat has no
    GIA. If that doesn't bring the count down to the stated number the rows
    are left alone.
    """
    stated = stated_dwellings(description)
    if stated is None or len(units) <= stated:
        return units
    kept = []
    for unit in sorted(units, key=lambda u: not u.get("gia")):   # measured rows first
        repeat = any(k.get("unit_type") == unit.get("unit_type")
                     and k.get("no_bedrooms") == unit.get("no_bedrooms")
                     and (k.get("gia") == unit.get("gia") or not unit.get("gia"))
                     for k in kept)
        if not repeat or len(kept) < stated:
            kept.append(unit)
    return kept if len(kept) == stated else units


def classify_plot(description, address):
    text = f"{description} {address}".lower()
    if re.search(EXTENSION_CONVERSION_RE, text):
        return "EXTENSION_CONVERSION"
    if re.search(MIXED_USE_RE, text):
        return "MIXED_USE"
    if re.search(DEMOLITION_RE, text):
        return "GARAGE_REDEVELOPMENT"
    # "Vacant land between 6 and 7 Tanner Street ... land adjacent to 6":
    # the site name beats a generic "adjacent"
    if "vacant land" in text and "garden" not in text:
        return "VACANT_LAND"
    if re.search(GARDEN_PLOT_RE, text):
        return "GARDEN_PLOT"
    return "VACANT_LAND"


def load_datahub_sites(path=PLD_PATH):
    """One row per application in the Datahub extract (see download_data.py)."""
    with open(path) as f:
        snapshot = json.load(f)
    rows = []
    for app in snapshot["applications"]:
        details = app.get("application_details") or {}
        resi = details.get("residential_details") or {}
        units = [u for u in resi.get("residential_units") or []
                 if u.get("change_type") == "Gain" and u.get("unit_development_type") == "New Build"]
        units = drop_duplicate_units(units, app.get("description") or "")
        site_text = app.get("site_name") or ""
        postcode = POSTCODE_RE.search(f"{app.get('postcode') or ''} {site_text}".upper())
        rows.append({
            "reference": app["id"],
            "lpa_app_no": app.get("lpa_app_no") or "",
            "lpa_name": app["lpa_name"],
            "address": site_address(site_text, app.get("site_number"), app.get("street_name")),
            "postcode_district": postcode.group(1) if postcode else None,
            "hectares": details.get("site_area"),
            "units": units,
            "dwellings": len(units),
            "description": " ".join((app.get("description") or "").split()),
            "decision_date": app.get("decision_date"),
        })
    return pd.DataFrame(rows)


def on_brownfield_register(app_numbers):
    """Cross-check against the brownfield register: does the planning
    reference turn up anywhere in its reference or notes columns?"""
    register = pd.read_csv(BROWNFIELD_PATH, low_memory=False, usecols=["reference", "notes"])
    text = " ".join(register.reference.astype(str)) + " " + " ".join(register.notes.dropna())
    text = text.upper()
    return app_numbers.map(lambda ref: len(ref) >= 6 and ref.upper() in text)


def new_build_candidates(sales):
    sites = load_datahub_sites()
    sites["plot_type"] = [classify_plot(d, a) for d, a in zip(sites.description, sites.address)]
    print("  plot types: " + ", ".join(f"{k} {v}" for k, v in sites.plot_type.value_counts().items()),
          file=sys.stderr)
    sites = sites[
        sites.plot_type.isin(PLOT_TYPES_KEPT)
        & sites.dwellings.between(1, 4)
        & ~(sites.hectares > 0.5)   # 4 homes on half a hectare+ is a data error
    ].copy()

    # Several permissions for the same address are alternatives for one piece
    # of land (159 Monkhams Lane has one for three houses and three for a
    # single house each), so only the most recently decided one is kept.
    decided = pd.to_datetime(sites.decision_date, format="%d/%m/%Y", errors="coerce")
    sites = sites.assign(decided=decided).sort_values(["decided", "reference"], ascending=[False, True])
    repeats = sites.address.str.lower().duplicated()
    print(f"  {repeats.sum()} older permissions dropped for an address already in the pool",
          file=sys.stderr)
    sites = sites[~repeats].sort_index()

    london = land_value_sheet()
    codes = dict(zip(london[3], london[2]))
    sites["la_code"] = sites.lpa_name.map(lambda n: la_code_for(n, codes))
    sites = sites.dropna(subset=["la_code"])

    sites["bedroom_conflict"] = [bedroom_conflict(u, d) for u, d in zip(sites.units, sites.description)]
    sites["site_check"] = sites.description.str.contains(SITE_CHECK_RE, case=False, regex=True)

    works = [abnormal_works(d) for d in sites.description]
    sites["exclude"] = [w[0] for w in works]
    sites["abnormal_costs"] = [w[1] for w in works]
    sites["works_note"] = [w[2] for w in works]
    print(f"  {sites.exclude.sum()} sites dropped for basement works", file=sys.stderr)
    sites = sites[~sites.exclude]

    areas = [unit_floor_area(u, d) for u, d in zip(sites.units, sites.description)]
    sites["proposed_floor_area_m2"] = [a[0] for a in areas]
    sites["floor_area_source"] = [a[1] for a in areas]
    too_big = sites.proposed_floor_area_m2.isna()
    print(f"  {too_big.sum()} sites dropped: homes bigger than the NDSS covers, no floor area given",
          file=sys.stderr)
    sites = sites[~too_big]

    tables = good_condition_values(sales)
    values = [new_build_value(tables, d, la, site_type_group(u))
              for d, la, u in zip(sites.postcode_district, sites.la_code, sites.units)]
    sites["local_price_per_m2"] = [v[0] for v in values]
    sites["local_price_q75"] = [v[1] for v in values]
    sites["comparable_count"] = [v[2] for v in values]
    sites["valuation_basis"] = [v[3] for v in values]

    # MHCLG values are borough averages; scale them by how the postcode
    # district's prices (all property types) compare with the borough's,
    # otherwise plots in the expensive parts of a borough come out far too
    # cheap. Sites with no postcode district get a factor of 1.
    district_all = [new_build_value(tables, d, la, "any")[0]
                    for d, la in zip(sites.postcode_district, sites.la_code)]
    borough_all = [new_build_value(tables, None, la, "any")[0] for la in sites.la_code]
    sites["location_factor"] = np.array(district_all) / np.array(borough_all)
    land_values = load_land_values(london)
    bands = [plot_value(row, land_values) for _, row in sites.iterrows()]
    sites["density_band"] = [b[0] for b in bands]
    sites["land_value_per_plot"] = [b[1] for b in bands]
    sites["acquisition_price"] = sites.dwellings * sites.land_value_per_plot * sites.location_factor

    brownfield = on_brownfield_register(sites.lpa_app_no)
    print(f"  {brownfield.sum()} of {len(sites)} sites also on the brownfield register",
          file=sys.stderr)
    print(f"  {(sites.bedroom_conflict != '').sum()} sites with conflicting bedroom counts, "
          f"{sites.site_check.sum()} flagged for works to a neighbouring house", file=sys.stderr)
    notes_out = (sites.works_note
                 + np.where(brownfield, "; also on brownfield register", "")
                 + "; decided " + sites.decision_date.fillna("?")
                 + "; " + sites.description.str.slice(0, 160)).str.lstrip("; ")

    return pd.DataFrame({
        "reference": sites.reference,
        "address": sites.address,
        "borough": sites.lpa_name,
        "postcode_district": sites.postcode_district,
        "site_type": "development_plot",
        "existing_floor_area_m2": None,
        "proposed_floor_area_m2": sites.proposed_floor_area_m2,
        "floor_area_source": sites.floor_area_source,
        "dwellings": sites.dwellings,
        "site_hectares": sites.hectares,
        "density_band": sites.density_band,
        "land_value_per_plot": sites.land_value_per_plot.round(0),
        "location_factor": sites.location_factor.round(4),
        "acquisition_price": sites.acquisition_price.round(-3),
        "price_source_type": "estimated land value (MHCLG 2023)",
        "abnormal_costs": sites.abnormal_costs,
        "local_price_per_m2": sites.local_price_per_m2.round(0),
        "local_price_q75": sites.local_price_q75.round(0),
        "comparable_count": sites.comparable_count,
        "valuation_basis": sites.valuation_basis,
        "plot_type": sites.plot_type,
        "bedroom_conflict": sites.bedroom_conflict,
        "site_check": sites.site_check,
        "epc_score": None,
        "notes": notes_out,
        "source": "Planning London Datahub + MHCLG land values 2023",
    })


def valuation_warnings(row):
    """Reasons a candidate's GDV estimate is less reliable than usual,
    joined with '; ', or "" if there are none."""
    reasons = []
    if pd.isna(row["postcode_district"]):
        reasons.append("no postcode district")
    if row["valuation_basis"].startswith("borough"):
        reasons.append("borough-wide comparables")
    if "all types" in row["valuation_basis"]:
        reasons.append("comparables of all property types")
    if row["comparable_count"] < FEW_COMPARABLES:
        reasons.append(f"fewer than {FEW_COMPARABLES} comparable sales")
    if "NDSS" in row["floor_area_source"]:
        reasons.append("floor area is an NDSS minimum, not a measured area")
    return "; ".join(reasons)


def rebuild_raw_candidates():
    sales = load_linked_sales()
    refurbs = refurb_candidates(sales)
    plots = new_build_candidates(sales)
    print(f"{len(refurbs)} refurb candidates, {len(plots)} new-build sites", file=sys.stderr)

    raw = pd.concat([refurbs, plots], ignore_index=True)
    raw["valuation_warnings"] = raw.apply(valuation_warnings, axis=1)
    raw.insert(0, "candidate_id", [f"C{i:04d}" for i in range(len(raw))])
    raw.to_csv(RAW_PATH, index=False)
    print(f"saved {RAW_PATH}", file=sys.stderr)


# ---- stage 2: raw_candidates.csv -> projects table ----

def classify(row):
    if row["site_type"] == "existing_dwelling" and pd.notna(row["existing_floor_area_m2"]):
        return "REFURB"
    if row["site_type"] == "development_plot" and pd.notna(row["proposed_floor_area_m2"]):
        return "NEW_BUILD"
    return None


SALE_VALUE_COLUMNS = {"median": "local_price_per_m2", "top_quartile": "local_price_q75"}


def text_or_blank(value):
    return value if pd.notna(value) else ""


def build_projects(raw, sale_value="median", **cost_settings):
    """Classify every candidate and work out its costs and profit.

    sale_value picks which local £/m2 the GDV is based on: the median of the
    comparable sales (baseline) or their 75th percentile. cost_settings get
    passed straight to calculate_project, which is how the sensitivity
    experiments change the allowance, build rate, land price or sale values.
    """
    value_column = SALE_VALUE_COLUMNS[sale_value]
    projects = []
    for _, row in raw.iterrows():
        strategy = classify(row)
        if strategy is None:
            continue
        if strategy == "REFURB":
            floor_area = row["existing_floor_area_m2"]
        else:
            floor_area = row["proposed_floor_area_m2"]

        project = {
            "candidate_id": row["candidate_id"],
            "address": row["address"],
            "borough": row["borough"],
            "postcode_district": row["postcode_district"],
            "strategy": strategy,
            "floor_area_m2": float(floor_area),
            "floor_area_source": row["floor_area_source"],
            "acquisition_price": float(row["acquisition_price"]),
            "price_source_type": row["price_source_type"],
            "abnormal_costs": float(row["abnormal_costs"]),
            "local_price_per_m2": float(row[value_column]),
            "comparable_count": int(row["comparable_count"]),
            "valuation_basis": row["valuation_basis"],
            "plot_type": text_or_blank(row["plot_type"]),
            "dwellings": row["dwellings"],
            "site_hectares": row["site_hectares"],
            "density_band": text_or_blank(row["density_band"]),
            "land_value_per_plot": row["land_value_per_plot"],
            "location_factor": row["location_factor"],
            "bedroom_conflict": text_or_blank(row["bedroom_conflict"]),
            "site_check": bool(row["site_check"]),
            "valuation_warnings": text_or_blank(row["valuation_warnings"]),
            "notes": row["notes"],
            "source": row["source"],
            "reference": row["reference"],
        }
        calculate_project(project, **cost_settings)
        projects.append(project)

    return pd.DataFrame(projects)


def eligible(projects):
    """Projects a shortlist may draw from: no more than MAX_PROJECT_COST, no
    unresolved bedroom conflict and no uncosted works to a neighbouring
    house (site_check)."""
    projects = projects[(projects.total_investment <= MAX_PROJECT_COST)
                        & (projects.bedroom_conflict == "")
                        & ~projects.site_check].copy()
    projects["return_on_cost"] = projects.expected_profit / projects.total_investment
    return projects


def shortlist_top(projects):
    """Original shortlist: best N of each strategy by return on cost."""
    projects = eligible(projects)
    keep = []
    for strategy, n in [("REFURB", N_REFURB), ("NEW_BUILD", N_NEW_BUILD)]:
        pool = projects[projects.strategy == strategy]
        keep.append(pool.nlargest(n, "return_on_cost"))
    return pd.concat(keep).drop(columns="return_on_cost")


def shortlist_banded(projects, seed=SHORTLIST_SEED):
    """Varied shortlist: same counts, spread over low/middle/high returns.

    Each strategy's pool is sorted by return on cost (candidate_id breaks
    ties) and cut into thirds by position, and projects are drawn at random
    (fixed seed) from each third in turn, so the table has weak and
    loss-making projects in it as well as the best ones. Cutting by position
    rather than with pd.qcut matters: qcut's bin edges differ slightly
    between pandas 2 and 3, which moved two sites between thirds and changed
    the draw.
    """
    projects = eligible(projects)
    rng = np.random.default_rng(seed)
    keep = []
    for strategy, n in [("REFURB", N_REFURB), ("NEW_BUILD", N_NEW_BUILD)]:
        pool = projects[projects.strategy == strategy].sort_values(
            ["return_on_cost", "candidate_id"], kind="stable")
        # 12 -> 4/4/4, 8 -> 2/3/3 (leftover goes to the higher bands)
        per_band = [n // 3 + (1 if i >= 3 - n % 3 else 0) for i in range(3)]
        for positions, k in zip(np.array_split(np.arange(len(pool)), 3), per_band):
            members = pool.iloc[positions]
            picked = rng.choice(len(members), size=k, replace=False)
            keep.append(members.iloc[np.sort(picked)].sort_values("return_on_cost", ascending=False))
    return pd.concat(keep).drop(columns="return_on_cost")


def shortlist_balanced(projects, seed=SHORTLIST_SEED):
    """Controlled benchmark: half profitable and half loss-making projects
    in each strategy (6 + 6 refurbs, 4 + 4 new builds), drawn at random with
    the fixed seed.

    In the banded shortlist every refurb loses money, so an algorithm can
    ignore that half of the problem. Here both strategies have projects worth
    buying. The rule was set before any results on this shortlist were seen.
    """
    projects = eligible(projects)
    rng = np.random.default_rng(seed)
    keep = []
    for strategy, n in [("REFURB", N_REFURB), ("NEW_BUILD", N_NEW_BUILD)]:
        pool = projects[projects.strategy == strategy].sort_values("candidate_id")
        picked = []
        for half in [pool[pool.expected_profit > 0], pool[pool.expected_profit <= 0]]:
            rows = rng.choice(len(half), size=n // 2, replace=False)
            picked.append(half.iloc[np.sort(rows)])
        keep.append(pd.concat(picked).sort_values("return_on_cost", ascending=False))
    return pd.concat(keep).drop(columns="return_on_cost")


SHORTLISTS = {"top": shortlist_top, "banded": shortlist_banded, "balanced": shortlist_balanced}


def load_projects(shortlist="top", raw_path=RAW_PATH, sale_value="median", **cost_settings):
    """The 20-project table, with costs under the given settings.

    The shortlist is always picked at the default cost settings, so the
    sensitivity runs change the costs of the same 20 projects rather than
    swapping in different ones.
    """
    raw = pd.read_csv(raw_path)
    shortlist_ids = list(SHORTLISTS[shortlist](build_projects(raw)).candidate_id)

    raw = raw[raw.candidate_id.isin(shortlist_ids)]
    projects = build_projects(raw, sale_value, **cost_settings)
    projects = projects.set_index("candidate_id").loc[shortlist_ids].reset_index()

    ids = [f"R{i:02d}" for i in range(1, N_REFURB + 1)] + \
          [f"N{i:02d}" for i in range(1, N_NEW_BUILD + 1)]
    projects.insert(0, "project_id", ids[:len(projects)])
    return projects


if __name__ == "__main__":
    if "--rebuild" in sys.argv:
        rebuild_raw_candidates()

    pool = build_projects(pd.read_csv(RAW_PATH))
    for strategy in ["REFURB", "NEW_BUILD"]:
        sub = pool[pool.strategy == strategy]
        print(f"{strategy}: {len(sub)} candidates, {(sub.expected_profit > 0).sum()} profitable "
              f"at default costs", file=sys.stderr)

    cols = ["project_id", "address", "borough", "strategy", "floor_area_m2", "acquisition_price",
            "abnormal_costs", "total_investment", "estimated_gdv", "expected_profit"]
    for name in SHORTLISTS:
        projects = load_projects(name)
        path = f"data/projects_{name}.csv"
        projects.to_csv(path, index=False)
        print(f"\n== {name} shortlist -> {path}")
        print(projects[cols].to_string(index=False))
        print(f"total investment if everything was bought: £{projects.total_investment.sum():,.0f}")
