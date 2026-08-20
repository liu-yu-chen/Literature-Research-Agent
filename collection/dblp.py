import gzip
import os
import re
import ssl
import urllib.request
import pandas as pd
from lxml import etree
from tqdm import tqdm

XML_PATH = r"..\database\dblp.xml.gz"
OUTPUT = r"..\database\dblp_agent.parquet"
DBLP_DOWNLOAD_URL = "https://dblp.org/xml/dblp.xml.gz"

POSITIVE_PATTERNS = [
    # Core ABM / IBM variants with tight word distance (0 to 3 words between)
    r"\bagent[- ]based(?:\s+\w+){0,3}\s+(?:model|modeling|modelling|simulation|simulator|framework|approach)s?\b",
    r"\bindividual[- ]based(?:\s+\w+){0,3}\s+(?:model|modeling|modelling|simulation|simulator|framework)s?\b",
    r"\bagent[- ]oriented(?:\s+\w+){0,3}\s+(?:simulation|simulator|modeling|modelling)s?\b",

    # Standard field-specific acronyms
    r"\b(ABM|ABMs|ABMS|IBM|IBMs)\b",

    # Spatial / Land Use ABM variants
    r"\bspatial(?:ly)?\s+agent[- ]based\b",
    r"\bspatially\s+explicit\s+agent\b",

    # Specialized ABM tools, libraries, and frameworks
    r"\b(netlogo|starlogo|sugarscape|ascape|swarm)\b",
    r"\b(mesa|mesa[- ]geo)\b",
    r"\b(repast|mason)(?:\s+\w+){0,2}\s+(?:framework|toolkit|platform|simulation|simulator)?\b",
    r"\bgama(?:\s+(?:platform|framework|simulation))?\b",
    r"\bmatsim\b",

    # Domain-specific terms (keep if broad coverage is needed)
    r"\bsocial simulation\b",
    r"\bgenerative social science\b"
]

# Exclusion patterns to filter out robotics, hardware, and physical agents
NEGATIVE_PATTERNS = [
    r"\brobot",
    r"\brobotics\b",
    r"\bmulti-robot\b",
    r"\bmultiagent control\b",
    r"\bmobile agent\b",
    r"\bcontrast agent\b",
]

# Compile regular expressions for performance
POS_REGEXES = [re.compile(p, re.IGNORECASE) for p in POSITIVE_PATTERNS]
NEG_REGEXES = [re.compile(p, re.IGNORECASE) for p in NEGATIVE_PATTERNS]


class DownloadProgressBar(tqdm):
    # Custom tqdm bar for urllib download updates.
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_dblp_if_missing(file_path, url, max_retries=5):
    # Check if DBLP XML gz file exists and is fully downloaded, otherwise re-download with retries.
    target_dir = os.path.dirname(file_path)
    if target_dir:
        os.makedirs(target_dir, exist_ok=True)

    ssl_context = ssl._create_unverified_context()

    # Check if existing file is complete
    if os.path.exists(file_path):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "Mozilla/5.0"}, method="HEAD"
            )
            with urllib.request.urlopen(req, context=ssl_context) as response:
                expected_size = int(response.headers.get("content-length", 0))
                actual_size = os.path.getsize(file_path)

            if expected_size > 0 and actual_size == expected_size:
                print(f"Found complete DBLP file: {file_path}")
                return
            else:
                print(
                    f"Incomplete file detected ({actual_size} / {expected_size} bytes). Removing and re-downloading..."
                )
                os.remove(file_path)
        except Exception as e:
            print(f"Failed to verify existing file size: {e}. Re-downloading...")
            os.remove(file_path)

    # Retry loop for download
    for attempt in range(1, max_retries + 1):
        print(
            f"Starting download from DBLP official source (Attempt {attempt}/{max_retries}): {url}"
        )
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "Mozilla/5.0"}
            )
            with urllib.request.urlopen(
                req, context=ssl_context
            ) as response, open(file_path, "wb") as out_file:
                total_size = int(response.headers.get("content-length", 0))
                downloaded_size = 0

                with DownloadProgressBar(
                    unit="B",
                    unit_scale=True,
                    miniters=1,
                    desc="Downloading dblp.xml.gz",
                    total=total_size,
                ) as t:
                    block_size = 1024 * 64  # Increased buffer size for efficiency
                    while True:
                        buffer = response.read(block_size)
                        if not buffer:
                            break
                        out_file.write(buffer)
                        downloaded_size += len(buffer)
                        t.update(len(buffer))

            # Validate full download completion
            if total_size > 0 and downloaded_size < total_size:
                raise IOError(
                    f"Incomplete download: {downloaded_size}/{total_size} bytes received."
                )

            print("\nDownload completed successfully!")
            return

        except Exception as e:
            print(f"\nDownload failed: {e}")
            if os.path.exists(file_path):
                os.remove(file_path)
            if attempt == max_retries:
                raise RuntimeError(
                    f"Failed to download DBLP dataset after {max_retries} attempts."
                )


def match_title(title):
    # Match title against positive regex patterns while filtering out negative ones.
    if not title:
        return False, []

    # Check negative filters first
    for neg_re in NEG_REGEXES:
        if neg_re.search(title):
            return False, []

    # Match positive keywords
    matched_tags = []
    for pos_re in POS_REGEXES:
        match = pos_re.search(title)
        if match:
            matched_tags.append(match.group(0).lower())

    if matched_tags:
        return True, list(set(matched_tags))

    return False, []


def parse_dblp():
    # Parse DBLP XML file and extract articles matching ABM keywords.
    papers = []

    with gzip.open(XML_PATH, "rb") as f:

        context = etree.iterparse(
            f,
            events=("end",),
            tag=("article", "inproceedings", "incollection", "phdthesis"),
            recover=True,
        )

        for _, elem in tqdm(context, desc="Parsing DBLP"):

            title = elem.findtext("title")
            is_match, matched_kw = match_title(title)

            if is_match:

                authors = [a.text for a in elem.findall("author") if a.text]

                affiliations = []
                for a in elem.findall("author"):
                    affil = (
                        a.get("aux")
                        or a.get("orcid")
                        or a.findtext("address")
                    )
                    if affil:
                        affiliations.append(affil)

                xml_keywords = [
                    kw.text for kw in elem.findall("keyword") if kw.text
                ]

                all_keywords = list(set(matched_kw + xml_keywords))

                year = elem.findtext("year")
                journal = elem.findtext("journal")
                venue = elem.findtext("booktitle")

                doi = ""
                for ee in elem.findall("ee"):
                    if ee.text and "doi.org" in ee.text:
                        doi = ee.text
                        break

                papers.append(
                    {
                        "title": title,
                        "authors": authors,
                        "affiliations": affiliations,
                        "keywords": all_keywords,
                        "year": year,
                        "journal": journal,
                        "venue": venue,
                        "doi": doi,
                        "type": elem.tag,
                        "source": "DBLP",
                    }
                )

            elem.clear()

            while elem.getprevious() is not None:
                del elem.getparent()[0]

    return papers


if __name__ == "__main__":

    # Ensure file exists and is completely downloaded before parsing
    download_dblp_if_missing(XML_PATH, DBLP_DOWNLOAD_URL)

    print("File:", XML_PATH)
    print("Size:", round(os.path.getsize(XML_PATH) / 1024 / 1024, 2), "MB")

    data = parse_dblp()

    print("Matched papers:", len(data))

    df = pd.DataFrame(data)
    df.to_parquet(OUTPUT, index=False)

    print("Saved:", OUTPUT)