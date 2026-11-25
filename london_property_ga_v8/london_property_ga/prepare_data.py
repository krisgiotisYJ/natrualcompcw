"""Builds the project tables.

python prepare_data.py --rebuild   slow, needs data/raw/ -> data/raw_candidates.csv
python prepare_data.py             quick, costs + the three shortlists -> data/projects_<name>.csv

Refurb purchase prices are real 2024 sales; works, resale (GDV) and new-build
plot prices are all estimates. Bedroom conflicts and site_check sites stay in
raw_candidates.csv but never go on a shortlist.
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
# anything over a third of the budget leaves room for only one or two others
MAX_PROJECT_COST = 1_000_000
SHORTLIST_SEED = 42   # fixed before any algorithm was compared on the shortlists

EPC_C_OR_BETTER = 69   # SAP score threshold for EPC band C
MIN_SALES_FOR_VALUE = 10
# fewer sales than this and one sale moves the median a lot
FEW_COMPARABLES = 20

# Refurb screen = cheap vs comparables, NOT a bad EPC. Old London houses
# rate E even after a full renovation, and most certs were 5-15 yrs old at sale
COMPARABLE_GROUP = ["postcode_district", "propertytype", "CONSTRUCTION_AGE_BAND"]
MIN_COMPARABLE_SALES = 5
CHEAP_QUANTILE = 0.25

# demolition / clearing the site, not in the build rate. My own number, not a quote
DEMOLITION_ALLOWANCE = 20_000
# new access to a backland plot, also a guess
ACCESS_ALLOWANCE = 10_000

# NDSS minimum sizes (m2) by bedrooms. Flats/bungalows = single storey,
# houses = two storey. Occupancy beds + 2 (2b4p, 3b5p...), 1-bed is 2 person.
# NDSS stops at 6 beds, so bigger homes need a GIA or they get dropped
NDSS_FLAT = {0: 39, 1: 50, 2: 70, 3: 86, 4: 99, 5: 112, 6: 125}
NDSS_HOUSE = {1: 58, 2: 79, 3: 93, 4: 106, 5: 119, 6: 132}
DEFAULT_BEDS = 2

# Datahub unit types that get the house NDSS sizes
HOUSE_UNIT_TYPES = {"Terraced Home", "Semi Detached Home", "Detached Home", "House or Bungalow"}
# GIA outside this = typo on the form (there's a 1 m2 home and a 6,700 m2 flat!)
PLAUSIBLE_UNIT_GIA = (37, 250)

PRICE_PAID_COLUMNS = ["transactionid", "pp_price", "date", "pp_postcode", "ptype", "old_new",
                      "pp_duration", "paon", "saon", "street", "locality", "town", "district",
                      "county", "ppd_category", "record_status"]

POSTCODE_RE = re.compile(r"\b([A-Z]{1,2}\d[A-Z\d]?)\s*\d[A-Z]{2}\b")
NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}

# only knocking things down counts - "with an associated garage" means building one
DEMOLITION_RE = (r"demoli|\bremov(?:al|e) of\b|take down|scrap yard|disused|"
                 r"existing (?:\w+ ){0,3}(?:garages?|outbuildings?|sheds?|structures?|stores?|buildings?)")

# plot_type. Only the first three fit "buy a plot at land value and build":
#   GARAGE_REDEVELOPMENT - garages/sheds etc. demolished first (gets the demolition allowance)
#   GARDEN_PLOT          - garden or side land of a house
#   VACANT_LAND          - everything else, incl. anything called vacant land with no garden mentioned
# dropped:
#   EXTENSION_CONVERSION - extending/converting an existing building, or an annexe
#   MIXED_USE            - also has shop/office/garage space I can't cost or value
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
# works to the host house or next door (extension, roof, moving a door...).
# A flat allowance can't cover that, so these get site_check and stay off the shortlists
SITE_CHECK_RE = (r"host property|existing dwelling|conservatory|utility room|relocation of|"
                 r"rear extensions?\b(?! to (?:create|provide))|first floor rear|hip to gable|"
                 r"roof (?:alterations|enlargement)|alterations to (?:roof|fenestration)|"
                 r"porch to \d|^alterations")

# Datahub unit types -> property type groups in the sales data
UNIT_TYPE_GROUPS = {"Terraced Home": "T", "Semi Detached Home": "S", "Detached Home": "D",
                    "House or Bungalow": "house", "Flat Apartment Maisonette": "F"}
SALES_TYPE_GROUPS = {"D": ["D"], "S": ["S"], "T": ["T"], "F": ["F"],
                     "house": ["D", "S", "T"], "any": ["D", "S", "T", "F"]}


# stage 1: raw sources -> raw_candidates.csv

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
    """EPC C+ £/m2 (median, q75, count) by district and by borough, per type group.

    Category A only - B is repossessions, buy-to-let, company transfers etc.
    C+ is just a proxy for decent condition, doesn't mean new-build standard."""
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
    """Most local £/m2 with at least MIN_SALES_FOR_VALUE sales: district + type,
    then borough + type, then district / borough any type.
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
        # under half the median = probably a bad floor-area match, not a bargain
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
    """Land value per plot by borough code. MHCLG gives £/ha and typical
    dwellings/ha for low/medium/high density, so divide. Median house price
    column throughout."""
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
    """Returns (density band, value per plot). Band = whichever typical
    dwellings/ha is closest to the site's own density. No site area -> low,
    which suits 1-4 homes on a garden/garage plot anyway."""
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
    """Bedrooms per home from the description (sorted), or None."""
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
    """(area, source) from the description and NDSS sizes. No bedrooms
    given -> assume 2-bed. area is None if a home is over 6 beds."""
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
    # e.g. 113 Gregory Crescent: description 2 bed, unit record 3, no GIA -> 79 or 93 m2
    recorded = [u.get("no_bedrooms") for u in units]
    if not recorded or None in recorded:
        return ""
    recorded = sorted(int(b) for b in recorded)
    described = description_bedrooms(description, len(units))
    if described is None or described == recorded:
        return ""
    return f"description says {described} bedrooms, unit records say {recorded}"


def unit_floor_area(units, description):
    """(area, source): unit GIA, else NDSS for the bedrooms, else the description."""
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
    """Returns (exclude, extra_cost, note). Basements get excluded, the
    description never says how big so I can't price them. Demolition and
    access get the lump sums."""
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
    """Number of homes in the description, or None."""
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
    """Drop repeated unit rows (Havering enters most homes twice). A repeat =
    same type, beds and GIA as an earlier row, or no GIA. If that doesn't get
    down to the stated count, change nothing."""
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
    """Does the planning ref turn up anywhere in the brownfield register?"""
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

    # same address = alternative schemes for one plot (159 Monkhams Lane has
    # four), keep the latest
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

    # MHCLG is a borough average, so scale by district £/m2 vs borough or plots
    # in the expensive end of a borough come out far too cheap. No district -> 1
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
    """Why this GDV is shakier than usual, '; '-joined ("" if nothing)."""
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


# stage 2: raw_candidates.csv -> projects table

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
    """Costs + profit for every candidate. sale_value picks the median or
    top-quartile £/m2 for GDV, cost_settings go straight to calculate_project
    (that's how exp 5 changes the assumptions)."""
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
    """What a shortlist can draw from: max MAX_PROJECT_COST, no bedroom
    conflict, no site_check."""
    projects = projects[(projects.total_investment <= MAX_PROJECT_COST)
                        & (projects.bedroom_conflict == "")
                        & ~projects.site_check].copy()
    projects["return_on_cost"] = projects.expected_profit / projects.total_investment
    return projects


def shortlist_top(projects):
    """Best N of each strategy by return on cost."""
    projects = eligible(projects)
    keep = []
    for strategy, n in [("REFURB", N_REFURB), ("NEW_BUILD", N_NEW_BUILD)]:
        pool = projects[projects.strategy == strategy]
        keep.append(pool.nlargest(n, "return_on_cost"))
    return pd.concat(keep).drop(columns="return_on_cost")


def shortlist_banded(projects, seed=SHORTLIST_SEED):
    """Same counts, spread over low/middle/high returns: random draw (fixed
    seed) from each third.

    Thirds are cut by position after sorting (candidate_id breaks ties). I had
    pd.qcut here originally but its bin edges are slightly different in
    pandas 2 and 3, which moved two sites and changed the draw.
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
    """6 + 6 refurbs and 4 + 4 new builds, half profitable and half not,
    fixed seed. Banded turned out too easy because all its refurbs lose
    money. Rule decided before I looked at any results on this one."""
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
    """The 20 projects, costed under the given settings. Always picked at the
    default costs, so exp 5 re-costs the same 20 rather than picking new ones."""
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
