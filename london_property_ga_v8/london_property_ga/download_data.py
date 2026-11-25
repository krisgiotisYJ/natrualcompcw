"""Downloads the raw data for `python prepare_data.py --rebuild`.

~1.5 GB in total, mostly the 1.3 GB house price per m2 zip. Only the 33
London borough files are kept from that (~430 MB) and the zip gets deleted.

The Datahub extract (data/pld_small_new_build.json) is already in the repo.
The Datahub changes every day so a new download won't match it - use
--refresh-datahub if you really want to replace it.

python download_data.py [--refresh-datahub]
"""
import datetime
import json
import os
import shutil
import sys
import urllib.request
import zipfile

SOURCES = {
    "data/raw/price_paid/pp-2024.csv":
        "http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com/pp-2024.csv",
    # brownfield registers - only used as a cross-check
    "data/raw/brownfield/brownfield-land.csv":
        "https://files.planning.data.gov.uk/dataset/brownfield-land.csv",
    "data/raw/land_values/Land_value_estimates_for_policy_appraisal_2023.xlsx":
        "https://assets.publishing.service.gov.uk/media/69b2d95f912f3f96bf687b1c/"
        "Land_value_estimates_for_policy_appraisal_2023.xlsx",
}

# Chi et al. (2025), from the London Datastore copy
HPM_ZIP_URL = ("https://data.london.gov.uk/download/epo9w/e4a4f81e-d922-4d14-b6f7-9a4d8c09e9bc/"
               "hpm_la_2024.zip")
HPM_DIR = "data/raw/hpm_london"

LONDON_BOROUGHS = [
    "Barking_and_Dagenham", "Barnet", "Bexley", "Brent", "Bromley", "Camden", "City_of_London",
    "Croydon", "Ealing", "Enfield", "Greenwich", "Hackney", "Hammersmith_and_Fulham", "Haringey",
    "Harrow", "Havering", "Hillingdon", "Hounslow", "Islington", "Kensington_and_Chelsea",
    "Kingston_upon_Thames", "Lambeth", "Lewisham", "Merton", "Newham", "Redbridge",
    "Richmond_upon_Thames", "Southwark", "Sutton", "Tower_Hamlets", "Waltham_Forest",
    "Wandsworth", "Westminster",
]

# London Datastore blocks python's default user agent
HEADERS = {"User-Agent": "Mozilla/5.0 (research data download)"}


def download(url, path):
    if os.path.exists(path):
        print(f"already have {path}")
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"downloading {url}")
    request = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(request, timeout=120) as response, open(path + ".part", "wb") as f:
        shutil.copyfileobj(response, f, length=1 << 20)
    os.rename(path + ".part", path)


# guest read-only access - header + value from the GLA's API guide:
# https://www.london.gov.uk/sites/default/files/planninglondondatahub_api_connection_technical_documentation_v1.pdf
PLD_SEARCH_URL = "https://planningdata.london.gov.uk/api-guest/applications/_search"
PLD_HEADERS = {"X-API-AllowRequest": "be2rmRnt&", "Content-Type": "application/json", **HEADERS}
PLD_PATH = "data/pld_small_new_build.json"
RESI = "application_details.residential_details."

# approved full permissions, 1-4 new homes, no existing homes on site,
# decided 2022-24 and status still "Approved" (= not started)
PLD_QUERY = {"bool": {"must": [
    {"term": {"decision.raw": "Approved"}},
    {"term": {"status.raw": "Approved"}},
    {"terms": {"application_type_full.raw": [
        "Full planning permission", "Full planning & demolition in a conservation area"]}},
    {"range": {"decision_date": {"gte": "01/01/2022", "lte": "31/12/2024"}}},
    {"range": {RESI + "total_no_proposed_residential_units": {"gte": 1, "lte": 4}}},
    {"term": {RESI + "total_no_existing_residential_units": 0}},
    {"nested": {"path": RESI + "residential_units", "query": {
        "term": {RESI + "residential_units.unit_development_type.raw": "New Build"}}}},
]}}
PLD_FIELDS = ["id", "lpa_name", "lpa_app_no", "description", "decision_date", "status",
              "site_name", "site_number", "street_name", "postcode", "centroid",
              "application_details.site_area", RESI + "total_no_proposed_residential_units",
              RESI + "residential_units"]


def download_datahub(refresh=False):
    if os.path.exists(PLD_PATH) and not refresh:
        print(f"already have {PLD_PATH} (use --refresh-datahub to replace it)")
        return
    applications, page = [], 500
    while True:
        body = {"query": PLD_QUERY, "_source": PLD_FIELDS, "size": page,
                "from": len(applications), "sort": [{"_doc": "asc"}]}
        request = urllib.request.Request(PLD_SEARCH_URL, data=json.dumps(body).encode(),
                                         headers=PLD_HEADERS, method="POST")
        with urllib.request.urlopen(request, timeout=120) as response:
            hits = json.load(response)["hits"]["hits"]
        applications += [h["_source"] for h in hits]
        if len(hits) < page:
            break
    snapshot = {"downloaded": datetime.date.today().isoformat(), "url": PLD_SEARCH_URL,
                "query": PLD_QUERY, "applications": applications}
    # write to a temp file then rename, so a crash can't leave half a file
    with open(PLD_PATH + ".part", "w") as f:
        json.dump(snapshot, f, indent=1)
    os.replace(PLD_PATH + ".part", PLD_PATH)
    print(f"saved {len(applications)} Datahub applications to {PLD_PATH}")


def download_london_hpm():
    wanted = {f"hpm_la_2024/{name}_link_26122024.csv" for name in LONDON_BOROUGHS}
    if all(os.path.exists(os.path.join(HPM_DIR, n)) for n in wanted):
        print("already have the London house price per m2 files")
        return

    zip_path = os.path.join(HPM_DIR, "hpm_la_2024.zip")
    download(HPM_ZIP_URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        missing = wanted - set(zf.namelist())
        if missing:
            raise RuntimeError(f"not in the zip: {sorted(missing)}")
        for name in sorted(wanted):
            zf.extract(name, HPM_DIR)
    os.remove(zip_path)
    print(f"extracted {len(wanted)} borough files to {HPM_DIR}")


if __name__ == "__main__":
    for path, url in SOURCES.items():
        download(url, path)
    download_london_hpm()
    download_datahub(refresh="--refresh-datahub" in sys.argv)
    print("done - now run: python prepare_data.py --rebuild")
