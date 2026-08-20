import os

import json

import gc

import re

import torch

from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig



# ==========================

# 1. Config & Data Prep

# ==========================



BASE = r"P:\Literature-Research-Agent-main"

INPUT = BASE + r"\repository\processed_literature.json"

OUTPUT = BASE + r"\repository\filtered.json"

CHECKPOINT_YES = BASE + r"\repository\llama_yes.json"

CHECKPOINT_LOG = BASE + r"\repository\processed_titles.txt"



LLAMA = BASE + r"\models\Llama-3.2-3B-Instruct"





def clean(x):

    return x if isinstance(x, str) else ""





with open(INPUT, "r", encoding="utf-8") as f:

    papers = json.load(f)



print(f"Total input candidate papers: {len(papers)}")



# Pre-compile Hard Exclusion Pattern

EXCLUDE_PATTERN = re.compile(

    r"\b(particle swarm optimization|pso|ant colony optimization|large language model agent|llm agent)\b",

    re.IGNORECASE

)



valid_papers = []

for p in papers:

    title, abstract = clean(p.get("title")), clean(p.get("abstract"))

    if not (title or abstract):

        continue



    if EXCLUDE_PATTERN.search(title):

        continue



    valid_papers.append(p)



print(f"Papers remaining for LLM verification after pre-cleaning: {len(valid_papers)}")



# ==========================

# 2. Checkpoint Recovery

# ==========================



filtered_result = []

processed_titles = set()



if os.path.exists(CHECKPOINT_YES):

    try:

        with open(CHECKPOINT_YES, "r", encoding="utf-8") as f:

            filtered_result = json.load(f)

    except Exception:

        print("Could not load positive checkpoint, starting from scratch.")



if os.path.exists(CHECKPOINT_LOG):

    try:

        with open(CHECKPOINT_LOG, "r", encoding="utf-8") as f:

            processed_titles = {line.strip() for line in f if line.strip()}

        print(f"Recovered {len(processed_titles)} total evaluated titles from checkpoint log.")

    except Exception:

        print("Could not load log checkpoint, starting from scratch.")



unprocessed_papers = [p for p in valid_papers if p.get("title") not in processed_titles]

print(f"Pending LLM processing count: {len(unprocessed_papers)}")



# ==========================

# 3. LLM Screening Pipeline

# ==========================



if unprocessed_papers:

    tokenizer = AutoTokenizer.from_pretrained(LLAMA, local_files_only=True, padding_side="left")

    if tokenizer.pad_token is None:

        tokenizer.pad_token = tokenizer.eos_token



    # Build token variants for YES, NO, and WAITLIST

    yes_tokens = [tokenizer.encode(tok, add_special_tokens=False)[0] for tok in ["YES", "Yes", " YES", " Yes"]]

    no_tokens = [tokenizer.encode(tok, add_special_tokens=False)[0] for tok in ["NO", "No", " NO", " No"]]

    waitlist_tokens = [tokenizer.encode(tok, add_special_tokens=False)[0] for tok in

                       ["WAITLIST", "Waitlist", " WAITLIST", " Waitlist"]]



    quant_config = BitsAndBytesConfig(

        load_in_4bit=True,

        bnb_4bit_quant_type="nf4",

        bnb_4bit_compute_dtype=torch.float16,

        bnb_4bit_use_double_quant=True

    )



    model = AutoModelForCausalLM.from_pretrained(

        LLAMA,

        quantization_config=quant_config,

        device_map={"": "cuda:0"},

        local_files_only=True,

        low_cpu_mem_usage=True,

        attn_implementation="sdpa"

    )

    model.eval()





    # ----------------------------------------------------

    # Prompt 1: First-pass Tri-Label Screening Prompt (Updated with Software Platforms)

    # ----------------------------------------------------

    def build_first_pass_prompt(p):

        messages = [

            {

                "role": "system",

                "content": "You are an expert reviewer classifying literature for an Agent-Based Modeling database. Answer ONLY YES, NO, or WAITLIST."

            },

            {

                "role": "user",

                "content": f"""

Determine whether this paper belongs to Agent-Based Modeling (ABM) or closely related bottom-up simulation research.



Classify as YES if there is evidence that the paper uses or develops:

- Agent-Based Modeling (ABM) / Agent-Based Simulation (ABS)

- Individual-Based Modeling (IBM)

- Multi-agent simulation of real-world entities (individuals, households, firms, vehicles, animals, cells)

- Spatial agent models or dynamic microsimulation with behavioral micro-entities

- Standard ABM software platforms or toolkits, including but not limited to:

  * Early/Classic: Sugarscape, Swarm, StarLogo, NetLogo, Ascape

  * Java/Object-Oriented: MASON, Repast (Repast Simphony), MATSim (large-scale transportation)

  * Modern/Python/GIS: GAMA, Mesa, Mesa Geo



Classify as NO if the paper is mainly about:

- LLM agents, software agents, robot agents, or game AI

- Multi-agent reinforcement learning without domain simulation

- Particle Swarm Optimization (PSO), Ant Colony Optimization, Genetic Algorithms

- Pure machine learning prediction or regression analysis

- Differential equations or System Dynamics without individual agents



Classify as WAITLIST if:

- The methodology seems like a bottom-up simulation or cellular spatial model, but it is ambiguous whether individual behavioral units exist.

- You are uncertain whether it is ABM or a non-agent simulation.



Title:

{clean(p.get("title"))}



Abstract:

{clean(p.get("abstract"))[:1000]}



Question:

Is this paper ABM, non-ABM, or borderline?



Answer ONLY YES, NO, or WAITLIST.

"""

            }

        ]

        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)





    # ----------------------------------------------------

    # Prompt 2: Strict Double Check Prompt for Waitlisted Papers (Updated with Software Toolkits)

    # ----------------------------------------------------

    def build_second_pass_prompt(p):

        messages = [

            {

                "role": "system",

                "content": "You are a highly rigorous methodologist evaluating borderline papers for Agent-Based Modeling (ABM). Answer ONLY YES or NO."

            },

            {

                "role": "user",

                "content": f"""

Perform a strict secondary verification for this borderline paper. 



To classify as YES, the paper MUST satisfy ALL THREE conditions OR explicitly utilize a recognized ABM platform:



1. Micro-Level Discrete Entities: Explicitly models discrete individuals, agents, actors, firms, cells, or vehicles (NOT continuous aggregates or populations).

2. Heterogeneous / Autonomous Behavior: Entities possess individual decision rules, behaviors, adaptive strategies, or state transitions.

3. Interaction / Emergence: Entities interact with each other or a shared spatial environment to generate system-level outcomes.



(Note: Explicit usage or extension of ABM frameworks like NetLogo, MASON, Repast, MATSim, GAMA, Mesa, Mesa Geo, or Swarm is strong evidence for YES).



If it is purely a mathematical system dynamics model, equation-based aggregate model, pure statistical regression, or algorithmic optimization (PSO/GA), classify as NO.



Title:

{clean(p.get("title"))}



Abstract:

{clean(p.get("abstract"))[:1200]}



Question:

Does this paper strictly fulfill ABM methodology or utilize an ABM framework?



Answer ONLY YES or NO.

"""

            }

        ]

        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)





    # Prepare first pass batch items

    paper_prompts = [(p, build_first_pass_prompt(p)) for p in unprocessed_papers]

    paper_prompts.sort(key=lambda x: len(x[1]))  # Smart Batching



    llama_batch_size = 8

    log_file = open(CHECKPOINT_LOG, "a", encoding="utf-8")

    waitlist_papers = []



    # ====================================================

    # Phase 1: First-pass Tri-Label Screening

    # ====================================================

    print("\n--- Phase 1: Tri-Label Screening (YES / NO / WAITLIST) ---")

    with torch.inference_mode():

        for i in tqdm(range(0, len(paper_prompts), llama_batch_size), desc="Phase 1 Screening"):

            batch_items = paper_prompts[i:i + llama_batch_size]

            batch_papers = [item[0] for item in batch_items]

            prompts = [item[1] for item in batch_items]



            inputs = tokenizer(

                prompts,

                return_tensors="pt",

                padding=True,

                truncation=True,

                max_length=1024

            ).to("cuda:0")



            outputs = model(**inputs, logits_to_keep=1)

            next_logits = outputs.logits[:, -1, :]

            probs = torch.softmax(next_logits, dim=-1)



            yes_probs = probs[:, yes_tokens].sum(dim=-1)

            no_probs = probs[:, no_tokens].sum(dim=-1)

            wait_probs = probs[:, waitlist_tokens].sum(dim=-1)



            decision_matrix = torch.stack([yes_probs, no_probs, wait_probs], dim=-1)

            decisions = torch.argmax(decision_matrix, dim=-1).cpu().tolist()



            for p, dec in zip(batch_papers, decisions):

                title = p.get("title", "")

                if dec == 0:  # Direct YES

                    p["_method"] = "Llama-3.2-3B-Phase1-DirectYES"

                    filtered_result.append(p)

                    if title:

                        log_file.write(title + "\n")

                elif dec == 1:  # Direct NO

                    if title:

                        log_file.write(title + "\n")

                else:  # WAITLIST -> To be evaluated in Phase 2

                    waitlist_papers.append(p)



            if (i // llama_batch_size) % 1000 == 0:

                log_file.flush()

                with open(CHECKPOINT_YES, "w", encoding="utf-8") as f:

                    json.dump(filtered_result, f, ensure_ascii=False)

                torch.cuda.empty_cache()



    # ====================================================

    # Phase 2: Secondary Verification for WAITLIST Papers

    # ====================================================

    if waitlist_papers:

        print(f"\n--- Phase 2: Strict Double-Check for {len(waitlist_papers)} Waitlisted Papers ---")

        second_prompts = [(p, build_second_pass_prompt(p)) for p in waitlist_papers]

        second_prompts.sort(key=lambda x: len(x[1]))



        with torch.inference_mode():

            for i in tqdm(range(0, len(second_prompts), llama_batch_size), desc="Phase 2 Double Check"):

                batch_items = second_prompts[i:i + llama_batch_size]

                batch_papers = [item[0] for item in batch_items]

                prompts = [item[1] for item in batch_items]



                inputs = tokenizer(

                    prompts,

                    return_tensors="pt",

                    padding=True,

                    truncation=True,

                    max_length=1024

                ).to("cuda:0")



                outputs = model(**inputs, logits_to_keep=1)

                next_logits = outputs.logits[:, -1, :]

                probs = torch.softmax(next_logits, dim=-1)



                yes_probs = probs[:, yes_tokens].sum(dim=-1)

                no_probs = probs[:, no_tokens].sum(dim=-1)



                is_yes = (yes_probs > no_probs).cpu().tolist()



                for p, flag in zip(batch_papers, is_yes):

                    title = p.get("title", "")

                    if flag:

                        p["_method"] = "Llama-3.2-3B-Phase2-VerifiedYES"

                        filtered_result.append(p)



                    if title:

                        log_file.write(title + "\n")



                if (i // llama_batch_size) % 50 == 0:

                    log_file.flush()

                    with open(CHECKPOINT_YES, "w", encoding="utf-8") as f:

                        json.dump(filtered_result, f, ensure_ascii=False)

                    torch.cuda.empty_cache()



    log_file.close()



    del model, tokenizer

    gc.collect()

    torch.cuda.empty_cache()



# Clean up temp checkpoint files after full completion

if os.path.exists(CHECKPOINT_YES):

    os.remove(CHECKPOINT_YES)

if os.path.exists(CHECKPOINT_LOG):

    os.remove(CHECKPOINT_LOG)



# ==========================

# 4. Save Final Results

# ==========================



print(f"\nFinal Filtered Papers Count: {len(filtered_result)}")



with open(OUTPUT, "w", encoding="utf-8") as f:

    json.dump(filtered_result, f, ensure_ascii=False, indent=2)



print("Saved final results to:", OUTPUT)