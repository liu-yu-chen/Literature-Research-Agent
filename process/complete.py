import json
import logging
import os
import time
from typing import Any, Dict, List, Set, Union
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("literature_enrichment.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)


class OpenAlexFetcher:
    def __init__(self, mailto: str = "", api_key: str = ""):
        self.mailto = mailto
        self.api_key = api_key
        self.base_url = "https://api.openalex.org/works"
        self.session = self._create_robust_session()

    @staticmethod
    def _create_robust_session() -> requests.Session:
        """
        Create a robust requests session with retries and connection limits.
        """
        session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=2.0,
            status_forcelist=[429, 500, 502, 503, 504],
            raise_on_status=False
        )
        adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=10)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def fetch_batch(self, id_list: List[str], id_type: str = "doi") -> List[Dict[str, Any]]:
        """
        Fetch literature metadata in batches with tight execution timeouts.
        """
        if not id_list:
            return []

        if id_type == "doi":
            formatted_ids = [f"https://doi.org/{i.replace('https://doi.org/', '').strip()}" for i in id_list]
            filter_str = f"doi:{'|'.join(formatted_ids)}"
        else:
            filter_str = f"{id_type}:{'|'.join([str(i).strip() for i in id_list])}"

        params = {
            "filter": filter_str,
            "per_page": 100
        }
        if self.mailto:
            params["mailto"] = self.mailto
        if self.api_key:
            params["api_key"] = self.api_key

        headers = {
            "User-Agent": f"LiteratureFetcher/1.0 (mailto:{self.mailto})",
            "Connection": "close"
        }

        try:
            response = self.session.get(self.base_url, params=params, headers=headers, timeout=(5, 10))
            if response.status_code == 200:
                data = response.json()
                return data.get("results", [])
            else:
                logging.warning(f"Batch query failed for {id_type} with HTTP status {response.status_code}")
        except Exception as err:
            logging.error(f"Network error or timeout during {id_type} batch: {err}")

        # Bisect recursively on failure
        if len(id_list) > 1:
            mid = len(id_list) // 2
            logging.info(f"Splitting batch of {len(id_list)} into two smaller subsets ({mid} and {len(id_list) - mid})")
            left_results = self.fetch_batch(id_list[:mid], id_type)
            right_results = self.fetch_batch(id_list[mid:], id_type)
            return left_results + right_results
        else:
            logging.warning(f"Failed to retrieve single item after split: {id_list[0]}")
            return []


def reconstruct_abstract(inverted_index: Dict[str, List[int]]) -> str:
    """
    Reconstruct plaintext abstract from OpenAlex abstract_inverted_index.
    """
    if not inverted_index or not isinstance(inverted_index, dict):
        return ""

    word_positions = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_positions.append((pos, word))

    word_positions.sort(key=lambda x: x[0])
    return " ".join([word for _, word in word_positions])


def load_input_files(input_paths: Union[str, List[str]]) -> pd.DataFrame:
    """
    Load data from Parquet, CSV, or Excel files into a unified pandas DataFrame.
    """
    if isinstance(input_paths, str):
        input_paths = [input_paths]

    dfs = []
    for file_path in input_paths:
        if not os.path.exists(file_path):
            logging.warning(f"File not found: {file_path}")
            continue

        logging.info(f"Loading input file: {file_path}")
        if file_path.endswith(".parquet"):
            df = pd.read_parquet(file_path)
        elif file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif file_path.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file_path)
        else:
            logging.error(f"Unsupported file format: {file_path}")
            continue

        dfs.append(df)

    if not dfs:
        raise ValueError("No valid data loaded from the provided file paths.")

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.columns = combined_df.columns.str.lower()
    return combined_df


def load_checkpoint(checkpoint_file: str) -> tuple[List[Dict[str, Any]], Set[str]]:
    """
    Load intermediate raw results and extracted processed batch IDs from checkpoint file.
    """
    if os.path.exists(checkpoint_file):
        logging.info(f"Checkpoint file '{checkpoint_file}' found. Resuming progress...")
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                raw_results = data.get("raw_results", [])
                processed_input_ids = set(data.get("processed_input_ids", []))
                logging.info(
                    f"Resumed {len(raw_results)} records from {len(processed_input_ids)} previously processed input IDs.")
                return raw_results, processed_input_ids
        except Exception as e:
            logging.error(f"Failed to load checkpoint file: {e}. Starting fresh.")
    return [], set()


def save_checkpoint(checkpoint_file: str, raw_results: List[Dict[str, Any]], processed_input_ids: Set[str]):
    """
    Save current fetched raw results and processed input IDs to checkpoint file.
    """
    temp_file = f"{checkpoint_file}.tmp"
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump({
            "raw_results": raw_results,
            "processed_input_ids": list(processed_input_ids)
        }, f, ensure_ascii=False)

    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
    os.rename(temp_file, checkpoint_file)


def process_raw_literature(
        input_paths: Union[str, List[str]],
        output_csv: str = "raw_literature.csv",
        output_json: str = "raw_literature.json",
        checkpoint_file: str = "raw_literature_checkpoint.json",
        mailto: str = "user@example.com",
        api_key: str = "",
        batch_size: int = 50
):
    """
    Main processing pipeline with Checkpoint/Resume mechanism.
    """
    df = load_input_files(input_paths)
    fetcher = OpenAlexFetcher(mailto=mailto, api_key=api_key)

    raw_results, processed_input_ids = load_checkpoint(checkpoint_file)
    openalex_processed_ids = {res.get("id") for res in raw_results if res.get("id")}

    if "doi" in df.columns:
        all_dois = df["doi"].dropna().astype(str).str.strip().unique().tolist()
        unprocessed_dois = [d for d in all_dois if d not in processed_input_ids]

        total_batches = (len(unprocessed_dois) - 1) // batch_size + 1 if unprocessed_dois else 0
        logging.info(
            f"Total DOIs: {len(all_dois)} | Already processed: {len(all_dois) - len(unprocessed_dois)} | Remaining: {len(unprocessed_dois)}")

        for i in range(0, len(unprocessed_dois), batch_size):
            batch_num = i // batch_size + 1
            batch = unprocessed_dois[i:i + batch_size]
            logging.info(f"Processing DOI batch {batch_num}/{total_batches}")

            results = fetcher.fetch_batch(batch, id_type="doi")
            for res in results:
                if res.get("id") not in openalex_processed_ids:
                    openalex_processed_ids.add(res.get("id"))
                    raw_results.append(res)

            processed_input_ids.update(batch)
            time.sleep(0.1)

            if batch_num % 20 == 0 or batch_num == total_batches:
                logging.info(f"Saving checkpoint at batch {batch_num}/{total_batches}...")
                save_checkpoint(checkpoint_file, raw_results, processed_input_ids)

    if "pmid" in df.columns:
        all_pmids = df["pmid"].dropna().astype(str).str.strip().unique().tolist()
        unprocessed_pmids = [p for p in all_pmids if p not in processed_input_ids]

        total_batches = (len(unprocessed_pmids) - 1) // batch_size + 1 if unprocessed_pmids else 0
        logging.info(
            f"Total PMIDs: {len(all_pmids)} | Already processed: {len(all_pmids) - len(unprocessed_pmids)} | Remaining: {len(unprocessed_pmids)}")

        for i in range(0, len(unprocessed_pmids), batch_size):
            batch_num = i // batch_size + 1
            batch = unprocessed_pmids[i:i + batch_size]
            logging.info(f"Processing PMID batch {batch_num}/{total_batches}")

            results = fetcher.fetch_batch(batch, id_type="pmid")
            for res in results:
                if res.get("id") not in openalex_processed_ids:
                    openalex_processed_ids.add(res.get("id"))
                    raw_results.append(res)

            processed_input_ids.update(batch)
            time.sleep(0.1)

            if batch_num % 20 == 0 or batch_num == total_batches:
                logging.info(f"Saving checkpoint at batch {batch_num}/{total_batches}...")
                save_checkpoint(checkpoint_file, raw_results, processed_input_ids)

    logging.info(f"Saving final raw literature JSON output to {output_json}")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(raw_results, f, ensure_ascii=False, indent=2)

    flattened_data = []
    for item in raw_results:
        abstract_text = reconstruct_abstract(item.get("abstract_inverted_index"))
        primary_loc = item.get("primary_location") or {}
        source_info = primary_loc.get("source") or {}

        authors = [
            a.get("author", {}).get("display_name", "")
            for a in item.get("authorships", [])
            if a.get("author")
        ]

        concepts = [
            c.get("display_name", "")
            for c in item.get("concepts", [])
        ]

        flattened_data.append({
            "openalex_id": item.get("id"),
            "doi": item.get("doi"),
            "title": item.get("title"),
            "publication_year": item.get("publication_year"),
            "publication_date": item.get("publication_date"),
            "type": item.get("type"),
            "cited_by_count": item.get("cited_by_count"),
            "venue_name": source_info.get("display_name"),
            "authors": "; ".join(authors),
            "concepts": "; ".join(concepts),
            "abstract": abstract_text,
            "raw_payload": json.dumps(item, ensure_ascii=False)
        })

    raw_df = pd.DataFrame(flattened_data)

    logging.info(f"Saving final raw literature CSV output to {output_csv}")
    raw_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

    logging.info("Literature extraction pipeline executed successfully.")


if __name__ == "__main__":
    INPUT_PARQUET_FILES = [
        r"..\database\dblp_agent.parquet",
        r"..\database\pubmed.parquet",
        r"..\database\wos.parquet"
    ]

    USER_EMAIL = "liuyc091502@outlook.com"
    API_KEY = "6n8PKhioAwAdLT97xMxb4b"  # Add OpenAlex Premium API key here if available

    process_raw_literature(
        input_paths=INPUT_PARQUET_FILES,
        output_csv="raw_literature.csv",
        output_json="raw_literature.json",
        checkpoint_file="raw_literature_checkpoint.json",
        mailto=USER_EMAIL,
        api_key=API_KEY,
        batch_size=100
    )