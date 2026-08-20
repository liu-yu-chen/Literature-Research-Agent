import json
import os
import time
from datetime import datetime
import pandas as pd
from Bio import Entrez
from tqdm import tqdm

# Entrez configuration
Entrez.email = "perosonal e-mail"
Entrez.tool = "MyPubMedScript"
Entrez.api_key = "your own api key"

# Path configuration
OUTPUT_FILE = "../database/pubmed.parquet"
CHECKPOINT_FILE = "../database/pubmed_checkpoint.json"
TEMP_DIR = "../database/pubmed_temp"

SLEEP_TIME = 0.35
BATCH_SIZE = 500

os.makedirs("../database", exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)


def load_existing_dois():
    # Load existing DOIs from the parquet file to avoid redundant downloads.
    if not os.path.exists(OUTPUT_FILE):
        return set()

    try:
        df = pd.read_parquet(OUTPUT_FILE)
        if "doi" in df.columns:
            return set(df["doi"].dropna().astype(str).tolist())
    except Exception as e:
        print("Load parquet error:", e)

    return set()


def search_pubmed_with_history(query_string):
    # Execute search on PubMed using WebEnv history server.
    try:
        with Entrez.esearch(
            db="pubmed",
            term=query_string,
            retmax=0,
            usehistory="y",
            sort="relevance",
        ) as handle:
            result = Entrez.read(handle)
            return (
                result.get("WebEnv"),
                result.get("QueryKey"),
                int(result.get("Count", 0)),
            )
    except Exception as e:
        print("Search error:", e)
        return None, None, 0


def fetch_pubmed_batch(webenv, query_key, retstart, retmax, retries=3):
    # Fetch article records in batches from PubMed.
    for i in range(retries):
        try:
            with Entrez.efetch(
                db="pubmed",
                rettype="xml",
                retmode="xml",
                retstart=retstart,
                retmax=retmax,
                webenv=webenv,
                query_key=query_key,
            ) as handle:
                data = Entrez.read(handle)
                return data.get("PubmedArticle", [])
        except Exception as e:
            print(f"Fetch error at retstart {retstart} (retry {i+1}):", e)
            time.sleep(5)

    return []


def parse_article(article):
    # Parse XML article record into a flat dictionary structure.
    try:
        medline = article["MedlineCitation"]
        art = medline["Article"]

        title = str(art.get("ArticleTitle", ""))

        abstract = ""
        if art.get("Abstract"):
            texts = []
            for x in art["Abstract"]["AbstractText"]:
                if hasattr(x, "attributes") and "Label" in x.attributes:
                    texts.append(x.attributes["Label"] + ": " + str(x))
                else:
                    texts.append(str(x))
            abstract = " ".join(texts)

        authors = []
        affiliations = []

        for a in art.get("AuthorList", []):
            name = (a.get("LastName", "") + " " + a.get("ForeName", "")).strip()
            if name:
                authors.append(name)

            for aff in a.get("AffiliationInfo", []):
                txt = str(aff.get("Affiliation", "")).strip()
                if txt:
                    affiliations.append(txt)

        journal = str(art.get("Journal", {}).get("Title", ""))

        year = ""
        try:
            year = str(
                art["Journal"]["JournalIssue"]["PubDate"].get("Year", "")
            )
        except Exception:
            pass

        doi = ""
        for item in article.get("PubmedData", {}).get("ArticleIdList", []):
            if item.attributes.get("IdType") == "doi":
                doi = str(item)
                break

        keywords = []
        for klist in medline.get("KeywordList", []):
            for k in klist:
                keywords.append(str(k))

        mesh = []
        for m in medline.get("MeshHeadingList", []):
            mesh.append(str(m["DescriptorName"]))

        return {
            "pmid": str(medline["PMID"]),
            "title": title,
            "abstract": abstract,
            "core_text": "Title: " + title + "\nAbstract: " + abstract,
            "authors": json.dumps(authors, ensure_ascii=False),
            "affiliations": json.dumps(
                list(set(affiliations)), ensure_ascii=False
            ),
            "journal": journal,
            "year": year,
            "doi": doi,
            "keywords": json.dumps(keywords, ensure_ascii=False),
            "mesh_terms": json.dumps(mesh, ensure_ascii=False),
            "source": "PubMed",
        }

    except Exception as e:
        print("Parse error:", e)
        return None


def save_batch_parquet(batch, file_tag):
    # Save parsed batch result to a temporary parquet file.
    if not batch:
        return

    df = pd.DataFrame(batch)
    path = os.path.join(TEMP_DIR, f"batch_{file_tag}.parquet")
    df.to_parquet(path, index=False)


def merge_parquet():
    # Merge all temporary batch files with existing dataset and deduplicate by DOI.
    files = [
        os.path.join(TEMP_DIR, x)
        for x in os.listdir(TEMP_DIR)
        if x.endswith(".parquet")
    ]

    if not files:
        return

    dfs = []
    if os.path.exists(OUTPUT_FILE):
        dfs.append(pd.read_parquet(OUTPUT_FILE))

    for f in files:
        dfs.append(pd.read_parquet(f))

    df = pd.concat(dfs, ignore_index=True)

    if "doi" in df.columns:
        df = df.drop_duplicates(subset=["doi"], keep="first")

    df.to_parquet(OUTPUT_FILE, index=False)
    print("Merged and saved total papers:", len(df))


def load_checkpoint():
    # Load checkpoint for resume capability.
    if os.path.exists(CHECKPOINT_FILE):
        return json.load(open(CHECKPOINT_FILE, encoding="utf-8"))
    return {}


def save_checkpoint(data):
    # Save current progress checkpoint.
    json.dump(data, open(CHECKPOINT_FILE, "w", encoding="utf-8"), indent=2)


if __name__ == "__main__":

    existing_dois = load_existing_dois()
    print("Existing DOI count:", len(existing_dois))

    # Build keywords and exclude terms
    EXCLUDE_TYPES = [
        "Editorial",
        "Letter",
        "Book Review",
        "Erratum",
        "Short Communication",
        "Comment",
        "Response",
        "Correction",
        "Notes",
    ]
    pt_exclude = "NOT (" + " OR ".join([f"{x}[pt]" for x in EXCLUDE_TYPES]) + ")"

    EXCLUDE_TERMS = [
        "robot*[Title/Abstract]",
        "robotics[MeSH Terms]",
        '"multi-agent system*"[Title/Abstract]',
        '"multiagent system*"[Title/Abstract]',
        '"multi-agent control"[Title/Abstract]',
        '"contrast agent*"[Title/Abstract]',
        '"imaging agent*"[Title/Abstract]',
    ]
    term_exclude = "NOT (" + " OR ".join(EXCLUDE_TERMS) + ")"

    abm_keywords = [
        '"agent-based model"[Title/Abstract]',
        '"agent-based models"[Title/Abstract]',
        '"agent-based modeling"[Title/Abstract]',
        '"agent-based modelling"[Title/Abstract]',
        '"agent-based simulation"[Title/Abstract]',
        '"agent-based simulations"[Title/Abstract]',
        '"agent based model"[Title/Abstract]',
        '"agent based models"[Title/Abstract]',
        '"agent based modeling"[Title/Abstract]',
        '"agent based simulation"[Title/Abstract]',
        '"individual-based model"[Title/Abstract]',
        '"individual-based models"[Title/Abstract]',
        '"individual-based modeling"[Title/Abstract]',
        '"individual-based simulation"[Title/Abstract]',
        '"individual based model"[Title/Abstract]',
        '"individual based models"[Title/Abstract]',
        '"individual based modeling"[Title/Abstract]',
        '"individual based simulation"[Title/Abstract]',
        '"agent-based epidemic model"[Title/Abstract]',
        '"agent-based infectious disease model"[Title/Abstract]',
        '"individual-based epidemic model"[Title/Abstract]',
        '"individual-based infectious disease model"[Title/Abstract]',
        "(simulation[Title/Abstract] AND (agent*[Title/Abstract] OR individual*[Title/Abstract]))",
        '("Computer Simulation"[Mesh] AND (agent*[Title/Abstract] OR individual*[Title/Abstract]))',
    ]

    health_keywords = """
    (
        "Public Health"[Mesh] OR "Epidemiology"[Mesh] OR "Communicable Diseases"[Mesh] OR
        "Infectious Disease Transmission"[Mesh] OR "Disease Outbreaks"[Mesh] OR
        epidemiology[Title/Abstract] OR epidemic*[Title/Abstract] OR pandemic*[Title/Abstract] OR
        outbreak*[Title/Abstract] OR infection*[Title/Abstract] OR infectious disease*[Title/Abstract] OR
        disease transmission[Title/Abstract] OR vaccination[Title/Abstract] OR vaccine*[Title/Abstract] OR
        immunization[Title/Abstract]
    )
    """

    base_query = (
        "("
        + " OR ".join(abm_keywords)
        + ") AND "
        + health_keywords
        + " "
        + pt_exclude
        + " "
        + term_exclude
    )

    checkpoint = load_checkpoint()

    # Define years chunk to process (e.g. from 1990 to current year)
    current_year = datetime.now().year
    years = list(range(1900, current_year + 1))

    for year in years:
        year_key = f"year_{year}"
        year_query = f"({base_query}) AND ({year}[dp])"

        print(f"\n--- Processing Year: {year} ---")
        webenv, key, total = search_pubmed_with_history(year_query)
        print(f"Total matched papers in {year}: {total}")

        if total == 0:
            continue

        # If a single year has over 9999 papers, notify user (unlikely for this niche domain)
        if total > 9999:
            print(
                f"Warning: Year {year} has {total} records (>9999), consider splitting by month."
            )

        start = checkpoint.get(year_key, 0)
        if start >= total:
            print(f"Year {year} already completed.")
            continue

        print(f"Resuming year {year} from record: {start} / {total}")

        with tqdm(
            total=total - start, desc=f"Fetching {year} Literature"
        ) as bar:
            for retstart in range(start, total, BATCH_SIZE):
                records = fetch_pubmed_batch(webenv, key, retstart, BATCH_SIZE)

                if not records:
                    break

                batch = []
                for r in records:
                    p = parse_article(r)
                    if p:
                        doi = p["doi"]
                        if not doi or doi not in existing_dois:
                            batch.append(p)
                            if doi:
                                existing_dois.add(doi)

                file_tag = f"{year}_{retstart}"
                save_batch_parquet(batch, file_tag)

                checkpoint[year_key] = retstart + len(records)
                save_checkpoint(checkpoint)

                bar.update(len(records))
                time.sleep(SLEEP_TIME)

    merge_parquet()
    print("Process Finished Successfully!")