import os
import json
import orjson
from tqdm import tqdm

INPUT_FILE = r"stdlit.json"
AUTHOR_FILE = r"author_master.json"
OUTPUT_FILE = r"paper_master.json"

def load_json(path):
    # Read raw JSON file using built-in json library
    with open(path, encoding="utf8") as f:
        return json.load(f)

def build_author_lookup(author_master):
    # Construct a fast lookup table indexed by author ID
    lookup = {}
    for aid, author in author_master.items():
        lookup[aid] = {
            "id": aid,
            "name": author.get("canonical_name", ""),
            "institutions": author.get("institutions", []),
            "countries": author.get("countries", []),
            "fields": author.get("fields", [])
        }
    return lookup

def normalize_paper(paper, author_lookup):
    # Clean and match paper metadata with author information
    authors = []
    institutions = set()
    countries = set()

    for auth in paper.get("authorships", []):
        author = auth.get("author", {})
        aid = author.get("id", "")

        if aid:
            aid = aid.replace("https://openalex.org/", "")

        if aid in author_lookup:
            info = author_lookup[aid]
            authors.append({
                "id": aid,
                "name": info["name"]
            })
            institutions.update(info["institutions"])
            countries.update(info["countries"])
        else:
            name = author.get("display_name", "")
            if name:
                authors.append({
                    "id": None,
                    "name": name
                })

    topics = [t["display_name"] for t in paper.get("topics", []) if t.get("display_name")]

    keywords = []
    for k in paper.get("keywords", []):
        if isinstance(k, dict):
            keywords.append(k.get("keyword", ""))
        else:
            keywords.append(str(k))

    return {
        "id": paper.get("id", ""),
        "doi": paper.get("doi", ""),
        "title": paper.get("title", ""),
        "year": paper.get("publication_year", paper.get("year", None)),
        "journal": paper.get("journal", ""),
        "authors": authors,
        "institutions": list(institutions),
        "countries": list(countries),
        "topics": topics,
        "keywords": keywords,
        "citation": paper.get("cited_by_count", paper.get("citation_count", 0)),
        "abstract": paper.get("abstract", ""),
        "source": paper.get("source", "OpenAlex")
    }

print("Loading data")
papers = load_json(INPUT_FILE)
authors = load_json(AUTHOR_FILE)

author_lookup = build_author_lookup(authors)

print("Standardizing papers")
paper_master = [normalize_paper(paper, author_lookup) for paper in tqdm(papers)]

print("Saving")
with open(OUTPUT_FILE, "wb") as f:
    f.write(orjson.dumps(paper_master, option=orjson.OPT_INDENT_2))

print("Finished")
print("Papers:", len(paper_master))