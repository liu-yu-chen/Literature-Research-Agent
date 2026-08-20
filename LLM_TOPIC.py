import os
import json
import re
import gc

import torch
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = r"P:\Literature-Research-Agent-main\models\Llama-3.2-3B-Instruct"

INPUT_FILE = "repository/filtered.json"

# IMPORTANT:
# Use a new output file for v2.
# This prevents old v1 classifications from being silently reused.
OUTPUT_FILE = "llama_topics_v2.json"

CLASSIFICATION_VERSION = "v2.0"

MAX_NEW_TOKENS = 96

SAVE_INTERVAL = 2000

MIN_BATCH = 4
MAX_BATCH = 16

MAX_INPUT_TOKENS = 1536

ABSTRACT_MAX_CHARS = 1600

# COVID-19 should only be considered historically possible
# from publication year 2020 onward.
COVID_START_YEAR = 2020


# ============================================================
# Model
# ============================================================

print("Loading model...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)

tokenizer.padding_side = "left"

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.clean_up_tokenization_spaces = False

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.float16
)

model.eval()
model.config.use_cache = True

print("Model loaded")


# ============================================================
# Utility
# ============================================================

def get_year(paper):
    """
    Safely extract publication year.
    """
    year = paper.get("publication_year", "")

    try:
        return int(str(year)[:4])
    except Exception:
        return None


def normalize_bool(value):
    """
    Normalize LLM boolean output.
    """
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.strip().lower() in {
            "true",
            "yes",
            "1"
        }

    if isinstance(value, (int, float)):
        return bool(value)

    return False


def extract_json(text):
    """
    Extract JSON object from LLM output.
    """
    if not isinstance(text, str):
        return None

    text = text.strip()

    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"\s*```$",
        "",
        text
    )

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(
        r"\{.*\}",
        text,
        flags=re.DOTALL
    )

    if not match:
        return None

    candidate = match.group(0)

    try:
        return json.loads(candidate)
    except Exception:
        return None


def normalize_string(value):
    """
    Normalize a single textual field.
    """
    if value is None:
        return ""

    if not isinstance(value, str):
        value = str(value)

    value = re.sub(
        r"\s+",
        " ",
        value
    ).strip()

    return value


def normalize_list(value):
    """
    Normalize a list-valued field.
    """
    if value is None:
        return []

    if isinstance(value, str):
        value = [value]

    if not isinstance(value, list):
        return []

    result = []

    for item in value:
        item = normalize_string(item)

        if item:
            result.append(item)

    return result


# ============================================================
# COVID Detection
# ============================================================

COVID_PATTERNS = [
    r"\bcovid[- ]?19\b",
    r"\bcovid19\b",
    r"\bsars[- ]cov[- ]?2\b",
    r"\bsarscov2\b",
    r"\b2019[- ]ncov\b",
    r"\bcoronavirus disease 2019\b",
    r"\bcoronavirus disease[- ]19\b"
]


def contains_explicit_covid(text):
    """
    Detect explicit COVID-19 terminology.

    This function is intentionally conservative.
    Generic terms such as coronavirus, pandemic, epidemic,
    infectious disease, or epidemiology are NOT sufficient.
    """
    if not isinstance(text, str):
        return False

    return any(
        re.search(
            pattern,
            text,
            flags=re.IGNORECASE
        )
        for pattern in COVID_PATTERNS
    )


def paper_explicitly_mentions_covid(paper):
    """
    Detect whether title or abstract explicitly mentions COVID-19.
    """
    title = normalize_string(
        paper.get("title", "")
    )

    abstract = normalize_string(
        paper.get("abstract", "")
    )

    text = f"{title} {abstract}"

    return contains_explicit_covid(text)


# ============================================================
# Output Normalization
# ============================================================

def normalize_output(
    paper,
    raw_output
):
    """
    Normalize LLM output into a stable schema.

    COVID classification is controlled by both:
    1. LLM classification
    2. deterministic temporal/content validation
    """

    year = get_year(paper)

    data = extract_json(raw_output)

    if data is None:
        return {
            "is_covid19": False,
            "main_topic": "",
            "research_field": "",
            "methodology": [],
            "application_domain": "",
            "keywords": [],
            "json_valid": False,
            "temporal_violation": False
        }

    data.setdefault(
        "is_covid19",
        False
    )

    data.setdefault(
        "main_topic",
        ""
    )

    data.setdefault(
        "research_field",
        ""
    )

    data.setdefault(
        "methodology",
        []
    )

    data.setdefault(
        "application_domain",
        ""
    )

    data.setdefault(
        "keywords",
        []
    )

    data["is_covid19"] = normalize_bool(
        data.get("is_covid19")
    )

    data["main_topic"] = normalize_string(
        data.get("main_topic")
    )

    data["research_field"] = normalize_string(
        data.get("research_field")
    )

    data["application_domain"] = normalize_string(
        data.get("application_domain")
    )

    data["methodology"] = normalize_list(
        data.get("methodology")
    )

    data["keywords"] = normalize_list(
        data.get("keywords")
    )

    # --------------------------------------------------------
    # Deterministic COVID validation
    # --------------------------------------------------------

    explicit_covid = paper_explicitly_mentions_covid(
        paper
    )

    temporal_violation = False

    # COVID-19 did not exist as a named disease before 2020.
    if year is not None and year < COVID_START_YEAR:

        if data["is_covid19"]:
            temporal_violation = True

        data["is_covid19"] = False

    # If the paper explicitly contains COVID-19 terminology
    # and is published in 2020 or later, preserve the signal.
    elif year is not None and year >= COVID_START_YEAR:

        if explicit_covid:
            data["is_covid19"] = True

    # If publication year is unavailable, do not force a
    # COVID classification from the LLM alone.
    elif year is None:

        if not explicit_covid:
            data["is_covid19"] = False

    data["json_valid"] = True

    data["temporal_violation"] = temporal_violation

    return data


# ============================================================
# Prompt
# ============================================================

def build_prompt(paper):

    title = normalize_string(
        paper.get("title", "")
    )

    abstract = normalize_string(
        paper.get("abstract", "")
    )

    year = paper.get(
        "publication_year",
        ""
    )

    abstract = abstract[:ABSTRACT_MAX_CHARS]

    topics = []

    for topic in paper.get(
        "topics",
        []
    )[:5]:

        if isinstance(topic, dict):

            name = topic.get(
                "display_name"
            )

            if name:
                topics.append(
                    normalize_string(name)
                )

    prompt = f"""
Classify this scientific paper for large-scale bibliometric analysis.

The classification has FIVE separate conceptual dimensions.

1. main_topic

Identify the general scientific research problem or research topic.

It should describe WHAT the study investigates scientifically.

Examples:
- Epidemiology
- Disease Transmission Modeling
- Social Network Analysis
- Migration Studies
- Urban Planning
- Environmental Health
- Health Inequality
- Population Dynamics

Do NOT use a specific disease, country, outbreak, population group,
or historical event as main_topic.

2. research_field

Identify the broad academic discipline.

Examples:
- Public Health
- Epidemiology
- Medicine
- Sociology
- Geography
- Economics
- Political Science
- Computer Science

3. methodology

Identify the major research methods used.

Examples:
- Agent-Based Modeling
- Regression Analysis
- Network Analysis
- Machine Learning
- System Dynamics
- Spatial Analysis
- Survey Research

4. application_domain

Identify the specific empirical object, disease, population,
geographic setting, or application context.

Examples:
- COVID-19
- Influenza
- HIV
- Urban Population
- Migration
- Public Transportation

This field is independent from main_topic and research_field.

5. is_covid19

Return true ONLY when the paper explicitly studies COVID-19
or SARS-CoV-2.

Do NOT infer COVID-19 from:

- Epidemiology
- Infectious Disease
- Coronavirus
- Pandemic
- Epidemic
- Disease Transmission
- Public Health
- Respiratory Disease
- Viral Infection

A paper about epidemiology is NOT automatically a COVID-19 paper.

A paper about disease transmission is NOT automatically a COVID-19 paper.

A paper about coronavirus is NOT automatically a COVID-19 paper.

Publication year is:

{year}

If publication year is before 2020,
is_covid19 MUST be false.

Historical diseases such as SARS, MERS, influenza, or other
coronavirus research before 2020 must NOT be classified as COVID-19.

IMPORTANT:

Do not allow the existence of many COVID-19 papers in the
literature to influence classification of unrelated papers.

Classify each paper independently using only its title,
abstract, publication year, and supplied metadata.

Do not use COVID-19 as a synonym for epidemiology.

Do not use COVID-19 as a synonym for infectious disease.

Do not use COVID-19 as a synonym for pandemic research.

The scientific topic and application domain are different variables.

Publication year:
{year}

Title:
{title}

Abstract:
{abstract}

OpenAlex Topics:
{topics}

Return JSON only.

Required schema:

{{
    "is_covid19": false,
    "main_topic": "",
    "research_field": "",
    "methodology": [],
    "application_domain": "",
    "keywords": []
}}

Output rules:

- Return valid JSON only.
- No explanation.
- Academic terminology only.
- main_topic should be concise.
- Do not include disease names in main_topic.
- Do not include COVID-19 in main_topic.
- Do not include COVID-19 in research_field unless it is genuinely part
  of a named academic field, which is unlikely.
- COVID-19 should normally appear only in application_domain and is_covid19.
"""

    return prompt


# ============================================================
# Token Estimation
# ============================================================

def estimate_tokens(prompts):

    lengths = []

    for prompt in prompts:

        length = len(
            tokenizer(
                prompt,
                add_special_tokens=False
            )["input_ids"]
        )

        lengths.append(length)

    if not lengths:
        return 0

    return max(lengths)


# ============================================================
# Dynamic Batch Size
# ============================================================

def choose_batch_size(prompts):

    max_tokens = estimate_tokens(
        prompts
    )

    if max_tokens < 400:
        return MAX_BATCH

    elif max_tokens < 800:
        return min(
            8,
            MAX_BATCH
        )

    else:
        return MIN_BATCH


# ============================================================
# Single Batch Generation
# ============================================================

def generate_sub_batch(
    papers
):

    prompts = [
        build_prompt(paper)
        for paper in papers
    ]

    messages = []

    for prompt in prompts:

        messages.append(
            [
                {
                    "role": "system",
                    "content":
                    "You are a scientific literature classifier. Return valid JSON only."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

    texts = [
        tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True
        )
        for message in messages
    ]

    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(
            model.device
        )
        for key, value in inputs.items()
    }

    with torch.no_grad():

        result = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id
        )

    outputs = []

    # Important:
    # Because left padding is used, each sample has the same
    # input length after batch padding.
    input_len = inputs[
        "input_ids"
    ].shape[1]

    for generated in result:

        text = tokenizer.decode(
            generated[input_len:],
            skip_special_tokens=True
        )

        outputs.append(
            text
        )

    return outputs


# ============================================================
# Batch Generation With OOM Recovery
# ============================================================

def generate_batch(
    papers
):

    prompts = [
        build_prompt(paper)
        for paper in papers
    ]

    batch_size = choose_batch_size(
        prompts
    )

    outputs = []

    start = 0

    while start < len(papers):

        sub_papers = papers[
            start:start + batch_size
        ]

        try:

            sub_outputs = generate_sub_batch(
                sub_papers
            )

            outputs.extend(
                sub_outputs
            )

            start += len(
                sub_papers
            )

        except RuntimeError as e:

            error_text = str(e)

            if (
                "CUDA" in error_text
                or "out of memory" in error_text.lower()
            ):

                print(
                    f"CUDA memory error. "
                    f"Reducing batch size: {batch_size} -> "
                    f"{max(1, batch_size // 2)}"
                )

                torch.cuda.empty_cache()
                gc.collect()

                if batch_size > 1:

                    batch_size = max(
                        1,
                        batch_size // 2
                    )

                    continue

                print(
                    "Falling back to single-paper processing."
                )

                for paper in sub_papers:

                    try:

                        single_output = generate_sub_batch(
                            [paper]
                        )

                        outputs.extend(
                            single_output
                        )

                    except Exception as single_error:

                        print(
                            "Single paper failed:",
                            single_error
                        )

                        outputs.append(
                            json.dumps(
                                {
                                    "is_covid19": False,
                                    "main_topic": "",
                                    "research_field": "",
                                    "methodology": [],
                                    "application_domain": "",
                                    "keywords": []
                                },
                                ensure_ascii=False
                            )
                        )

                    torch.cuda.empty_cache()
                    gc.collect()

                start += len(
                    sub_papers
                )

            else:

                raise

    return outputs


# ============================================================
# Result Validation
# ============================================================

def validate_results(results):

    temporal_violations = 0

    json_failures = 0

    covid_count = 0

    pre_2020_covid = 0

    application_covid_count = 0

    main_topic_covid_count = 0

    for item in results:

        year = item.get(
            "year"
        )

        try:

            year = int(
                str(year)[:4]
            )

        except Exception:

            year = None

        data = extract_json(
            item.get(
                "topic_analysis",
                ""
            )
        )

        if data is None:

            json_failures += 1

            continue

        if data.get(
            "temporal_violation",
            False
        ):

            temporal_violations += 1

        if data.get(
            "is_covid19",
            False
        ):

            covid_count += 1

            if (
                year is not None
                and year < COVID_START_YEAR
            ):

                pre_2020_covid += 1

        application_domain = normalize_string(
            data.get(
                "application_domain",
                ""
            )
        )

        main_topic = normalize_string(
            data.get(
                "main_topic",
                ""
            )
        )

        if contains_explicit_covid(
            application_domain
        ):

            application_covid_count += 1

        if contains_explicit_covid(
            main_topic
        ):

            main_topic_covid_count += 1

    print()
    print("=" * 60)
    print("V2 Validation")
    print("=" * 60)

    print(
        f"Total results: {len(results)}"
    )

    print(
        f"JSON failures: {json_failures}"
    )

    print(
        f"Raw temporal violations detected: {temporal_violations}"
    )

    print(
        f"COVID-19 papers: {covid_count}"
    )

    print(
        f"Pre-2020 COVID-19 violations after correction: "
        f"{pre_2020_covid}"
    )

    print(
        f"COVID-19 in application_domain: "
        f"{application_covid_count}"
    )

    print(
        f"COVID-19 in main_topic: "
        f"{main_topic_covid_count}"
    )

    print("=" * 60)

    if pre_2020_covid == 0:

        print(
            "Temporal validation PASSED."
        )

    else:

        print(
            "Temporal validation FAILED."
        )

    if main_topic_covid_count == 0:

        print(
            "Main-topic contamination check PASSED."
        )

    else:

        print(
            "Main-topic contamination detected."
        )


# ============================================================
# Main
# ============================================================

def main():

    print(
        f"Loading input file: {INPUT_FILE}"
    )

    with open(
        INPUT_FILE,
        encoding="utf-8"
    ) as f:

        papers = json.load(
            f
        )

    print(
        f"Input papers: {len(papers)}"
    )

    # --------------------------------------------------------
    # Resume
    # --------------------------------------------------------

    if os.path.exists(
        OUTPUT_FILE
    ):

        print(
            f"Loading existing v2 results: {OUTPUT_FILE}"
        )

        with open(
            OUTPUT_FILE,
            encoding="utf-8"
        ) as f:

            results = json.load(
                f
            )

    else:

        results = []

    finished_ids = {
        item.get("id")
        for item in results
        if item.get("id") is not None
    }

    papers = [
        paper
        for paper in papers
        if paper.get("id") not in finished_ids
    ]

    print(
        f"Remaining papers: {len(papers)}"
    )

    if not papers:

        print(
            "All papers have already been processed."
        )

        validate_results(
            results
        )

        return

    count = 0

    index = 0

    with tqdm(
        total=len(papers),
        desc="Processing Papers"
    ) as pbar:

        while index < len(papers):

            batch = papers[
                index:index + MAX_BATCH
            ]

            outputs = generate_batch(
                batch
            )

            # ------------------------------------------------
            # Safety check
            # ------------------------------------------------

            if len(outputs) != len(batch):

                print(
                    "Warning: output count does not match batch count."
                )

                while len(outputs) < len(batch):

                    outputs.append(
                        json.dumps(
                            {
                                "is_covid19": False,
                                "main_topic": "",
                                "research_field": "",
                                "methodology": [],
                                "application_domain": [],
                                "keywords": []
                            },
                            ensure_ascii=False
                        )
                    )

                outputs = outputs[
                    :len(batch)
                ]

            # ------------------------------------------------
            # Save results
            # ------------------------------------------------

            for paper, raw_output in zip(
                batch,
                outputs
            ):

                normalized = normalize_output(
                    paper,
                    raw_output
                )

                results.append(
                    {
                        "id": paper.get(
                            "id"
                        ),
                        "title": paper.get(
                            "title"
                        ),
                        "year": paper.get(
                            "publication_year"
                        ),
                        "classification_version":
                            CLASSIFICATION_VERSION,
                        "raw_output":
                            raw_output,
                        "topic_analysis":
                            json.dumps(
                                normalized,
                                ensure_ascii=False
                            )
                    }
                )

            batch_len = len(
                batch
            )

            index += batch_len

            count += batch_len

            pbar.update(
                batch_len
            )

            # ------------------------------------------------
            # Periodic save
            # ------------------------------------------------

            if count >= SAVE_INTERVAL:

                with open(
                    OUTPUT_FILE,
                    "w",
                    encoding="utf-8"
                ) as f:

                    json.dump(
                        results,
                        f,
                        ensure_ascii=False,
                        indent=2
                    )

                tqdm.write(
                    f"Saved: {len(results)}"
                )

                count = 0

    # --------------------------------------------------------
    # Final save
    # --------------------------------------------------------

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2
        )

    print()
    print(
        "Finished all papers."
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    validate_results(
        results
    )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()
