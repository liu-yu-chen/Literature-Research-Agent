import os
# expandable_segments is not supported on Windows; removed env var to avoid warning
import re
import torch
import faiss
import numpy as np
import orjson

from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

INPUT_FILE = r"processed_literature.json"
OUTPUT_FILE = r"stdlit.json"
AUTHOR_FILE = r"author_master.json"
INSTITUTION_FILE = r"institution_master.json"

VECTOR_DIR = r"vector_store"
INDEX_FILE = os.path.join(VECTOR_DIR, "author.index")
EMBED_FILE = os.path.join(VECTOR_DIR, "author_embeddings.npy")
CACHE_FILE = r"D:\GeoAgent\repository\cache\qwen_cache.json"

BGE_PATH = r"..\models\bge-large-en-v1.5"
QWEN_PATH = r"..\models\Qwen2.5-3B-Instruct"

os.makedirs(VECTOR_DIR, exist_ok=True)
os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)

# Global model pointers for lazy loading
qwen_model = None
qwen_tokenizer = None


def get_qwen_model():
    """Lazy load Qwen model explicitly onto GPU to prevent CPU/Disk offloading."""
    global qwen_model, qwen_tokenizer
    if qwen_model is None:
        print("Initializing Qwen2.5 on GPU (cuda:0)...")
        qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_PATH)
        qwen_tokenizer.padding_side = "left"
        if qwen_tokenizer.pad_token is None:
            qwen_tokenizer.pad_token = qwen_tokenizer.eos_token

        qwen_model = AutoModelForCausalLM.from_pretrained(
            QWEN_PATH,
            device_map="cuda:0",  # Lock strictly to primary GPU
            dtype=torch.float16,
            attn_implementation="sdpa"
        )
        qwen_model.eval()
    return qwen_model, qwen_tokenizer


def clean_text(text):
    if not text:
        return ""
    text = str(text).lower()
    return " ".join(text.split())


def normalize_name(name):
    name = clean_text(name)
    name = name.replace(".", "")
    return name.title()


def split_first_last_name(full_name):
    if not full_name:
        return {"first_name": "", "last_name": ""}

    cleaned = re.sub(r'[^\w\s-]', '', full_name).strip()
    parts = cleaned.split()

    if len(parts) == 0:
        return {"first_name": "", "last_name": ""}
    elif len(parts) == 1:
        return {"first_name": parts[0].title(), "last_name": ""}
    else:
        first_name = " ".join(parts[:-1]).title()
        last_name = parts[-1].title()
        return {"first_name": first_name, "last_name": last_name}


def normalize_institution(name):
    name = clean_text(name)
    replace_map = {
        "univ": "university",
        "university of hong kong": "the university of hong kong",
        "hku": "the university of hong kong",
        "ust": "hong kong university of science and technology",
    }
    return replace_map.get(name, name).title()


def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as f:
            return orjson.loads(f.read())
    return {}


def save_cache(cache):
    with open(CACHE_FILE, "wb") as f:
        f.write(orjson.dumps(cache, option=orjson.OPT_INDENT_2))


def parse_json(text):
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return orjson.loads(text[start:end])
    except Exception:
        pass
    return None


# Step 1: Filter Literature for ABM Relevance (Ultra-Fast Hybrid Pipeline)
print("Loading papers...")
with open(INPUT_FILE, "rb") as f:
    papers = orjson.loads(f.read())

cache = load_cache()
cache_updated = False

print("Filtering papers for Agent-Based Model (ABM) relevance...")

# Extended keyword list to increase early hit rate
abm_keywords = [
    "agent-based", "agent based", "multi-agent", "multiagent",
    "individual-based", "individual based", "abm", "abms", "mas",
    "cellular automata", "cellular automaton", "microsimulation",
    "complex adaptive system", "spatial agent", "pedestrian dynamics",
    "crowd simulation", "social simulation", "artificial society",
    "netlogo", "repast", "gama platform", "mesa framework"
]

filtered_papers = []
papers_for_bge = []

# Level 1: Keyword Fast Matching
for paper in papers:
    title = paper.get("title", "")
    abstract = paper.get("abstract", "") or paper.get("abstract_inverted_index", "")
    if isinstance(abstract, dict):
        abstract = " ".join(abstract.keys())

    combined_text = f"{title} {abstract}".lower()

    if any(kw in combined_text for kw in abm_keywords):
        filtered_papers.append(paper)
    else:
        papers_for_bge.append((paper, title, abstract))

print(f"Level 1 - Keyword Matched ABM papers: {len(filtered_papers)}")
print(f"Level 1 - Remaining papers for BGE filtering: {len(papers_for_bge)}")

# Level 2: BGE Semantic Similarity Screening
papers_for_llm = []

if papers_for_bge:
    print("Loading BGE model for Level 2 Semantic Filtering...")
    bge = SentenceTransformer(BGE_PATH, device="cuda")

    # Define ABM Semantic Anchor
    abm_anchor_text = "Agent-based modeling, multi-agent spatial simulation, individual-based interaction rules, pedestrian and agent dynamics, microsimulation."
    anchor_emb = bge.encode([abm_anchor_text], normalize_embeddings=True)

    # Prepare candidate texts
    candidate_texts = [f"Title: {title}\nAbstract: {abstract[:300]}" for _, title, abstract in papers_for_bge]

    print("Encoding paper abstracts via BGE...")
    candidate_embs = bge.encode(
        candidate_texts,
        batch_size=128,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    # Compute Cosine Similarities via Dot Product (since embeddings are normalized)
    similarities = np.dot(candidate_embs, anchor_emb.T).squeeze(-1)

    # Set Vector Thresholds
    # >= 0.60: High confidence ABM paper
    # < 0.35: Irrelevant paper
    # 0.35 ~ 0.60: Borderline paper to be checked by Qwen
    for idx, (paper, title, abstract) in enumerate(papers_for_bge):
        sim = float(similarities[idx])
        paper_id = paper.get("id", "")
        key = f"abm_check_{paper_id}"

        if key in cache:
            if cache[key].get("is_abm", False):
                filtered_papers.append(paper)
        elif sim >= 0.60:
            cache[key] = {"is_abm": True, "method": "bge_high_confidence"}
            cache_updated = True
            filtered_papers.append(paper)
        elif sim < 0.35:
            cache[key] = {"is_abm": False, "method": "bge_low_confidence"}
            cache_updated = True
        else:
            papers_for_llm.append((paper, key, f"Title: {title}\nAbstract: {abstract[:300]}"))

print(f"Level 2 - BGE Bypassed (High Sim) Papers Added: {len(filtered_papers)}")
print(f"Level 2 - Borderline papers pending LLM verification: {len(papers_for_llm)}")

if 'bge' in locals():
    del bge
    torch.cuda.empty_cache()

# Level 3: Qwen Logits Fast Classification
if papers_for_llm:
    qwen, tokenizer = get_qwen_model()

    token_1_id = tokenizer.encode("1", add_special_tokens=False)[-1]
    token_0_id = tokenizer.encode("0", add_special_tokens=False)[-1]

    llm_batch_size = 16
    print(f"Level 3 - Running Logits classification for {len(papers_for_llm)} items...")

    with torch.inference_mode():
        for i in tqdm(range(0, len(papers_for_llm), llm_batch_size)):
            batch = papers_for_llm[i: i + llm_batch_size]
            batch_prompts = []
            for item in batch:
                text_content = item[2]
                prompt = (
                    'Determine if this academic paper uses Agent-Based Modeling (ABM) or Multi-Agent/Individual-Based simulation.\n'
                    'Even if terms like "ABM" or "Agent" are NOT explicitly mentioned, reply 1 if the study models macroscopic phenomena through discrete individual interaction rules (e.g., pedestrian dynamics, microscopic traffic/economic simulation, cell models). Otherwise reply 0.\n\n'
                    f'{text_content}\n'
                    'Answer:'
                )
                batch_prompts.append(prompt)

            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(qwen.device)

            outputs = qwen(**inputs)
            next_token_logits = outputs.logits[:, -1, :]

            logits_1 = next_token_logits[:, token_1_id]
            logits_0 = next_token_logits[:, token_0_id]
            is_abm_flags = logits_1 > logits_0

            for (paper, key, _), is_abm_tensor in zip(batch, is_abm_flags):
                is_abm = bool(is_abm_tensor.item())

                cache[key] = {"is_abm": is_abm, "method": "qwen_logits"}
                cache_updated = True

                if is_abm:
                    filtered_papers.append(paper)

            del inputs, outputs, next_token_logits
            torch.cuda.empty_cache()

papers = filtered_papers
print(f"Total valid ABM papers retained: {len(papers)}")


# Step 2: Extract author profiles and restore names
def extract_authors(papers):
    authors = {}
    institutions = {}

    for paper in papers:
        topics = [
            t.get("display_name", "")
            for t in paper.get("topics", [])
            if t.get("display_name")
        ]

        for auth in paper.get("authorships", []):
            author_info = auth.get("author", {})
            author_id = author_info.get("id")
            if not author_id:
                continue

            author_id = str(author_id).replace("https://openalex.org/", "")
            if not author_id:
                continue

            raw_name = author_info.get("display_name", "")
            parsed_name = split_first_last_name(raw_name)
            auth["first_name"] = parsed_name["first_name"]
            auth["last_name"] = parsed_name["last_name"]

            if author_id not in authors:
                authors[author_id] = {
                    "author_id": author_id,
                    "names": set(),
                    "institutions": set(),
                    "countries": set(),
                    "fields": set(),
                    "departments": set(),
                    "papers": [],
                }

            author = authors[author_id]
            if raw_name:
                author["names"].add(raw_name)

            for ins in auth.get("institutions", []):
                raw_inst_name = ins.get("display_name", "")
                std_name = normalize_institution(raw_inst_name)

                if std_name:
                    author["institutions"].add(std_name)
                    if std_name not in institutions:
                        institutions[std_name] = {
                            "name": std_name,
                            "countries": set(),
                            "authors": set(),
                            "papers": 0,
                        }
                    institutions[std_name]["authors"].add(author_id)
                    institutions[std_name]["papers"] += 1

                country = ins.get("country_code", "")
                if country:
                    author["countries"].add(country)
                    if std_name:
                        institutions[std_name]["countries"].add(country)

            author["fields"].update(topics)
            author["papers"].append(paper.get("id", ""))

    for author in authors.values():
        author["names"] = list(author["names"])
        author["institutions"] = list(author["institutions"])
        author["countries"] = list(author["countries"])
        author["fields"] = list(author["fields"])
        author["papers"] = list(set(author["papers"]))
        author["paper_count"] = len(author["papers"])

        canonical_name = max(author["names"], key=len) if author["names"] else ""
        author["canonical_name"] = canonical_name

        parsed_canonical = split_first_last_name(canonical_name)
        author["first_name"] = parsed_canonical["first_name"]
        author["last_name"] = parsed_canonical["last_name"]

    for ins in institutions.values():
        ins["authors"] = list(ins["authors"])
        ins["countries"] = list(ins["countries"])

    return authors, institutions


def author_text(author):
    return f"{' '.join(author.get('names', []))} {' '.join(author.get('institutions', []))} {' '.join(author.get('countries', []))} {' '.join(author.get('fields', []))}"


def json_serializable(obj):
    if isinstance(obj, set):
        return list(obj)
    if isinstance(obj, dict):
        return {k: json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_serializable(x) for x in obj]
    return obj


print("Extracting author profiles...")
author_master, institution_master = extract_authors(papers)
print("Authors:", len(author_master))
print("Institutions:", len(institution_master))

with open(AUTHOR_FILE, "wb") as f:
    f.write(orjson.dumps(json_serializable(author_master)))

with open(INSTITUTION_FILE, "wb") as f:
    f.write(orjson.dumps(json_serializable(institution_master)))

# Step 3: Build FAISS Index and Disambiguate Authors
print("Loading BGE model...")
bge = SentenceTransformer(BGE_PATH, device="cuda")

author_ids = list(author_master.keys())
texts = [author_text(author_master[x]) for x in author_ids]

print("Encoding authors...")
embeddings = bge.encode(
    texts, batch_size=128, normalize_embeddings=True, show_progress_bar=True
)
embeddings = np.asarray(embeddings, dtype="float32")

index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, INDEX_FILE)
np.save(EMBED_FILE, embeddings)
print("FAISS index ready")


def rule_score(source, candidate):
    score = 0
    if set(source.get("countries", [])) & set(candidate.get("countries", [])):
        score += 0.25
    if set(source.get("institutions", [])) & set(candidate.get("institutions", [])):
        score += 0.45
    if set(source.get("fields", [])) & set(candidate.get("fields", [])):
        score += 0.30
    return score


print("Resolving unknown authors...")
unknown = []
for paper in papers:
    for auth in paper.get("authorships", []):
        if not auth.get("author", {}).get("id", ""):
            unknown.append(auth)

print("Unknown authors:", len(unknown))

valid_unknown = [
    item for item in unknown if item.get("author", {}).get("display_name", "")
]
unknown_texts = [item["author"]["display_name"] for item in valid_unknown]

print("Encoding unknown authors in batch...")
unknown_embs = bge.encode(
    unknown_texts,
    batch_size=128,
    normalize_embeddings=True,
    show_progress_bar=True,
)
unknown_embs = np.asarray(unknown_embs, dtype="float32")

print("Batch searching FAISS index...")
all_scores, all_ids = index.search(unknown_embs, 5)

auto_match = 0
qwen_match = 0
new_author = 0

qwen_pending_items = []
qwen_prompts = []
qwen_keys = []
qwen_bests = []

for idx, item in enumerate(valid_unknown):
    name = unknown_texts[idx]
    source = {
        "name": name,
        "institution": "",
        "country": "",
        "countries": [],
        "institutions": [],
        "fields": [],
    }

    candidates = []
    for s, i in zip(all_scores[idx], all_ids[idx]):
        candidate = author_master[author_ids[i]]
        candidate_score = (float(s) + rule_score(source, candidate)) / 2
        candidates.append((candidate_score, candidate))

    candidates.sort(key=lambda x: x[0], reverse=True)
    best_score, best = candidates[0]

    if best_score >= 0.90:
        item["standard_author"] = best["author_id"]
        item["confidence"] = best_score
        item["method"] = "faiss_rule"
        auto_match += 1
    else:
        key = f"disambig_{normalize_name(name)}_{best['author_id']}"
        if key in cache:
            result = cache[key]
            if result.get("same_person", False):
                item["standard_author"] = best["author_id"]
                item["confidence"] = result.get("confidence", 0)
                item["method"] = "qwen"
                qwen_match += 1
            else:
                item["standard_author"] = None
                item["confidence"] = 0
                item["method"] = "new"
                new_author += 1
        else:
            prompt = f"""You are an academic author disambiguation system. Determine whether two author records refer to the same person.

Unknown author:
Name: {name}

Candidate author:
Name: {best["canonical_name"]}
Institution: {best["institutions"]}
Country: {best["countries"]}
Research fields: {best["fields"]}

Return JSON only:
{{"same_person":true,"confidence":0.0}}"""

            qwen_pending_items.append(item)
            qwen_prompts.append(prompt)
            qwen_keys.append(key)
            qwen_bests.append(best)

if qwen_pending_items:
    qwen, tokenizer = get_qwen_model()
    llm_batch_size = 32
    print("Running batch LLM verification for disambiguation...")

    with torch.inference_mode():
        for i in tqdm(range(0, len(qwen_prompts), llm_batch_size)):
            batch_prompts = qwen_prompts[i: i + llm_batch_size]
            batch_items = qwen_pending_items[i: i + llm_batch_size]
            batch_keys = qwen_keys[i: i + llm_batch_size]
            batch_bests = qwen_bests[i: i + llm_batch_size]

            inputs = tokenizer(
                batch_prompts, return_tensors="pt", padding=True
            ).to(qwen.device)
            outputs = qwen.generate(
                **inputs, max_new_tokens=60, do_sample=False
            )

            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            for item, key, best, text in zip(
                    batch_items, batch_keys, batch_bests, decoded
            ):
                result = parse_json(text)
                if result is None:
                    result = {
                        "same_person": False,
                        "confidence": 0,
                    }

                cache[key] = result
                cache_updated = True

                if result.get("same_person", False):
                    item["standard_author"] = best["author_id"]
                    item["confidence"] = result.get("confidence", 0)
                    item["method"] = "qwen"
                    qwen_match += 1
                else:
                    item["standard_author"] = None
                    item["confidence"] = 0
                    item["method"] = "new"
                    new_author += 1

if cache_updated:
    save_cache(cache)

print("FAISS matched:", auto_match)
print("Qwen matched:", qwen_match)
print("New authors:", new_author)

print("Saving stdlit.json...")
with open(OUTPUT_FILE, "wb") as f:
    f.write(orjson.dumps(papers, option=orjson.OPT_INDENT_2))

print("Finished successfully.")