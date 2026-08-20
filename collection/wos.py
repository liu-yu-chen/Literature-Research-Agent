import os
import time
import requests
import pandas as pd

# ================= 1. Configuration =================
API_KEY = "your own api key".strip()
URL = "https://api.clarivate.com/apis/wos-starter/v1/documents"  #
OUTPUT_PATH = "../database/wos.parquet"  #
# Add missing configurations
headers = {
    "X-ApiKey": API_KEY,
    "Accept": "application/json"
}
DEFAULT_START_YEAR = 1900
CURRENT_YEAR = 2026
MAX_DAILY_REQUESTS = 50000
PAGE_SIZE = 50
MAX_RETRIES = 5
TIMEOUT_SECONDS = 30
REQUEST_DELAY = 0.5

# Broadened Search Query (Covering ABM, IBM, MAS, and variations)
KEYWORDS_QUERY = """
TS=
(
    (
        "agent-based model*"
        OR "agent based model*"
        OR "agent-based modeling"
        OR "agent based modeling"
        OR "agent-based simulation*"
        OR "agent based simulation*"
        OR "agent-based framework*"
        OR "agent-based approach*"
        OR "individual-based model*"
        OR "individual based model*"
        OR "individual-based simulation*"
        OR "social simulation"
        OR "artificial societ*"
        OR "agent-based microsimulation"
        OR "ABM"
        OR "IBM"
        OR "agent-based system*"
        OR "agent based system*"
    )

    OR

    (
        (
            "agent-based"
            OR "individual-based"
            OR "agent-driven"
            OR "agent-centric"
        )
        AND
        (
            urban
            OR city
            OR cities
            OR spatial
            OR geographic*
            OR GIS
            OR "land use"
            OR "land-use"
            OR transport*
            OR mobility
            OR migration
            OR population
            OR epidemi*
            OR disease
            OR infection
            OR health
            OR ecological
            OR ecosystem
            OR environment*
            OR economic*
            OR market*
            OR financial
            OR agricultural
            OR climate
            OR disaster
            OR evacuation
            OR policy
            OR policies
            OR behavior*
            OR behaviour*
            OR social
            OR network*
            OR "complex system*"
        )
    )
)

NOT TS=
(
    "multi-agent system*"
    OR
    "multi agent system*"
    OR
    "multi-agent reinforcement learning"
    OR
    "distributed artificial intelligence"
    OR
    "software agent*"
)
"""

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)  #

# ================= 2. Load Local Database for Deduplication =================
existing_dois = set()  #
existing_titles = set()  #
existing_uids = set()  #


def normalize_title(title):
    """Normalize title for exact deduplication: lowercase and strip spaces."""
    return title.strip().lower() if title else ""  #


if os.path.exists(OUTPUT_PATH):  #
    try:
        existing_df = pd.read_parquet(OUTPUT_PATH)  #

        if "doi" in existing_df.columns:  #
            existing_dois = set(existing_df["doi"].dropna().str.lower().str.strip().tolist())  #
        if "title" in existing_df.columns:  #
            existing_titles = set(existing_df["title"].dropna().apply(normalize_title).tolist())  #
        if "uid" in existing_df.columns:  #
            existing_uids = set(existing_df["uid"].dropna().tolist())  #

        print(f"📦 Local database loaded successfully: {len(existing_df)} total records found.")  #
        print(f"   ├─ Indexed DOIs: {len(existing_dois)}")  #
        print(f"   ├─ Indexed Titles: {len(existing_titles)}")  #
        print(f"   └─ Indexed UIDs: {len(existing_uids)}")  #
    except Exception as e:
        print(f"⚠️ Failed to read local database: {e}")  #
else:
    print("ℹ️ No local database found. A new wos.parquet file will be created upon completion.")  #

# ================= 3. Detect Earliest Publication Year Dynamically =================
print("\n🔍 Detecting publication time span from Web of Science...")
start_year = DEFAULT_START_YEAR

try:
    init_params = {"q": KEYWORDS_QUERY, "limit": 1, "page": 1}
    init_res = requests.get(URL, headers=headers, params=init_params, timeout=TIMEOUT_SECONDS)
    if init_res.status_code == 200:
        meta = init_res.json().get("metadata", {})
        total_found = meta.get("total", 0)
        print(f"📊 Total matching records in WoS database: {total_found}")

        # If hits returned, attempt to extract year from first/oldest or set wide range
        hits = init_res.json().get("hits", [])
        if hits:
            pub_year = hits[0].get("source", {}).get("publishYear")
            if pub_year:
                # Set a safe early bound, e.g., 1900 to present
                start_year = 1900
except Exception as e:
    print(f"⚠️ Could not fetch metadata dynamically, falling back to START_YEAR = {DEFAULT_START_YEAR}: {e}")

# ================= 4. Year-by-Year Incremental Retrieval Loop =================
new_records = []  #
request_counter = 0  #

print(f"\n🚀 Starting unbounded year-segmented literature collection ({start_year} - {CURRENT_YEAR})...")
print(f"🔍 Base Query: {KEYWORDS_QUERY}\n")  #

for year in range(CURRENT_YEAR, start_year - 1, -1):
    if request_counter >= MAX_DAILY_REQUESTS:  #
        print("⚠️ Reached max daily request limit. Stopping gracefully.")  #
        break

    yearly_query = f"({KEYWORDS_QUERY}) AND PY={year}"  #
    page = 1  #
    total_records_year = None  #
    fetched_count_year = 0  #

    # Quick pre-check for year existence to avoid empty iterations
    print(f"📅 --- Processing Year: {year} ---")  #

    while True:
        params = {
            "q": yearly_query,
            "limit": PAGE_SIZE,
            "page": page
        }  #

        response_data = None  #

        # Exponential Backoff Retry Mechanism for Network & 500/429 Errors
        for attempt in range(1, MAX_RETRIES + 1):  #
            try:
                request_counter += 1  #
                response = requests.get(
                    URL,
                    headers=headers,
                    params=params,
                    timeout=TIMEOUT_SECONDS
                )  #

                if response.status_code == 200:  #
                    response_data = response.json()  #
                    break

                elif response.status_code in [429, 500, 502, 503, 504]:  #
                    backoff_time = (2 ** attempt) * 2  #
                    print(
                        f"⚠️ HTTP {response.status_code} error on Year {year}, Page {page}. "
                        f"Retrying in {backoff_time}s (Attempt {attempt}/{MAX_RETRIES})..."
                    )  #
                    time.sleep(backoff_time)  #

                elif response.status_code == 401:  #
                    print("❌ API Authentication Failed (401). Please check your API key.")  #
                    break

                else:
                    print(f"❌ Unexpected API Response {response.status_code}: {response.text}")  #
                    break

            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):  #
                backoff_time = attempt * 5  #
                print(
                    f"🚨 Connection error on Year {year}, Page {page}. "
                    f"Retrying in {backoff_time}s..."
                )  #
                time.sleep(backoff_time)  #
            except Exception as e:
                print(f"🚨 Unexpected Exception: {e}")  #
                break

        if response_data is None:  #
            print(f"⚠️ Could not fetch Year {year}, Page {page}. Moving to next chunk/saving progress.")  #
            break

        hits = response_data.get("hits", [])  #
        metadata = response_data.get("metadata", {})  #

        if total_records_year is None:  #
            total_records_year = metadata.get("total", 0)  #
            print(f"📊 Year {year} total matching records: {total_records_year}")  #

        if not hits:  #
            break

        skipped_count = 0  #
        added_in_page = 0  #

        for doc in hits:  #
            uid = doc.get("uid", "")  #
            title = doc.get("title", "")  #
            norm_title = normalize_title(title)  #

            identifiers = doc.get("identifiers", {})  #
            doi = identifiers.get("doi", "").lower().strip()  #

            is_in_uid = uid in existing_uids if uid else False  #
            is_in_doi = doi in existing_dois if doi else False  #
            is_in_title = norm_title in existing_titles if norm_title else False  #

            if is_in_uid or is_in_doi or is_in_title:  #
                skipped_count += 1  #
                continue

            if uid: existing_uids.add(uid)  #
            if doi: existing_dois.add(doi)  #
            if norm_title: existing_titles.add(norm_title)  #

            names = doc.get("names", {})  #
            authors_list = [a.get("displayName") for a in names.get("authors", []) if a.get("displayName")]  #
            authors_str = "; ".join(authors_list) if authors_list else ""  #

            source = doc.get("source", {})  #
            pub_year = source.get("publishYear", "")  #
            pub_date = source.get("publishDate", "")  #
            publish_time = str(pub_date) if pub_date else str(pub_year)  #

            citations = doc.get("citations", [])  #
            times_cited = citations[0].get("count", 0) if citations else 0  #

            abstract = doc.get("abstract") or doc.get("summary", {}).get("abstract", "")  #

            new_records.append({
                "uid": uid,
                "title": title,
                "authors": authors_str,
                "abstract": abstract,
                "doi": doi,
                "publish_time": publish_time,
                "publish_year": pub_year,
                "journal": source.get("sourceTitle", ""),
                "times_cited": times_cited,
                "doc_type": ", ".join(doc.get("types", []))
            })  #
            added_in_page += 1  #

        fetched_count_year += len(hits)  #
        print(
            f"   ✓ [{year}] Page {page} ({fetched_count_year}/{total_records_year}) | "
            f"⏭️ Skipped: {skipped_count} | ✨ Added: +{added_in_page}"
        )  #

        if fetched_count_year >= total_records_year:  #
            break

        page += 1  #
        time.sleep(REQUEST_DELAY)  #

# ================= 5. Append and Save to Parquet =================
if new_records:  #
    new_df = pd.DataFrame(new_records)  #

    if os.path.exists(OUTPUT_PATH):  #
        old_df = pd.read_parquet(OUTPUT_PATH)  #
        final_df = pd.concat([old_df, new_df], ignore_index=True)  #
    else:
        final_df = new_df  #

    final_df = final_df.drop_duplicates(subset=["uid"], keep="first")  #
    if "doi" in final_df.columns:  #
        has_doi = final_df[final_df["doi"].str.len() > 0]  #
        no_doi = final_df[final_df["doi"].str.len() == 0]  #
        has_doi = has_doi.drop_duplicates(subset=["doi"], keep="first")  #
        final_df = pd.concat([has_doi, no_doi], ignore_index=True)  #

    final_df.to_parquet(OUTPUT_PATH, index=False, engine="pyarrow")  #

    print("\n" + "=" * 60)
    print(f"🎉 Extraction complete! New records added: {len(new_df)}")  #
    print(f"📁 Parquet database updated at: {OUTPUT_PATH}")  #
    print(f"📊 Total unique records in database: {len(final_df)}")  #
    print("=" * 60)
else:
    print("\nℹ️ All fetched records already exist in the local database. No updates performed.")  #