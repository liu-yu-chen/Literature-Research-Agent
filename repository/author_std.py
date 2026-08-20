import os,json,gc,re,requests,torch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor,as_completed
from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig


# ================= Configuration =================

LLAMA_PATH=r"P:\Literature-Research-Agent-main\models\Llama-3.2-3B-Instruct"
INPUT_PATH=(r"P:\Literature-Research-Agent-main\repository\filtered.json")
OUTPUT_PATH=r"P:\Literature-Research-Agent-main\repository\standardized_literature.json"

OPENALEX_CACHE=r"P:\Literature-Research-Agent-main\repository\openalex_cache.json"
LLM_CACHE=r"P:\Literature-Research-Agent-main\repository\llm_author_cache.json"
CHECKPOINT=r"P:\Literature-Research-Agent-main\repository\author_checkpoint.json"

OPENALEX_API_KEY="6n8PKhioAwAdLT97xMxb4b"

torch.backends.cuda.matmul.allow_tf32=True
torch.backends.cudnn.allow_tf32=True


# ================= OpenAlex =================

class OpenAlexClient:

    def __init__(self,key,cache):
        self.key=key
        self.cache_file=cache
        if os.path.exists(cache):
            self.cache=json.load(open(cache,"r",encoding="utf-8"))
        else:
            self.cache={}

    def clean_id(self,x):
        return x.replace("https://openalex.org/","") if x else ""

    def save(self):
        json.dump(self.cache,open(self.cache_file,"w",encoding="utf-8"),ensure_ascii=False)

    def parse(self,a):

        parsed=a.get("parsed_longest_name",{})
        institutions=[]

        for item in a.get("affiliations",[]):
            inst=item.get("institution",{})
            institutions.append({
                "institution_id":self.clean_id(inst.get("id")),
                "institution":inst.get("display_name",""),
                "country":inst.get("country_code","")
            })

        return {
            "author_id":self.clean_id(a.get("id")),
            "name":a.get("display_name",""),
            "first_name":parsed.get("first",""),
            "last_name":parsed.get("last",""),
            "orcid":a.get("orcid",""),
            "institutions":institutions,
            "source":"OpenAlex"
        }


    def request_batch(self,ids):

        ids=[self.clean_id(x) for x in ids if self.clean_id(x) not in self.cache]

        if not ids:
            return

        params={
            "filter":"id:"+"|".join(ids),
            "per-page":100,
            "api_key":self.key
        }

        try:
            r=requests.get(
                "https://api.openalex.org/authors",
                params=params,
                timeout=30
            )

            if r.status_code!=200:
                print(r.text)
                return

            for a in r.json().get("results",[]):
                p=self.parse(a)
                self.cache[p["author_id"]]=p

        except Exception as e:
            print(e)


    def resolve(self,ids,batch=100,workers=8):

        batches=[
            ids[i:i+batch]
            for i in range(0,len(ids),batch)
        ]

        with ThreadPoolExecutor(max_workers=workers) as ex:
            tasks=[
                ex.submit(self.request_batch,b)
                for b in batches
            ]

            for _ in tqdm(
                as_completed(tasks),
                total=len(tasks),
                desc="OpenAlex"
            ):
                pass

        self.save()


# ================= Llama =================

def load_llama():

    tokenizer=AutoTokenizer.from_pretrained(
        LLAMA_PATH,
        local_files_only=True,
        padding_side="left"
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token=tokenizer.eos_token

    config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    model=AutoModelForCausalLM.from_pretrained(
        LLAMA_PATH,
        quantization_config=config,
        device_map="auto",
        local_files_only=True,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa"
    )

    model.eval()

    return tokenizer,model



def build_prompt(author):

    return f"""
Resolve academic author information.
Name:{author.get("name","")}
ORCID:{author.get("orcid","")}

Return JSON only:
{{
"first_name":"",
"last_name":"",
"confidence":0
}}
"""


def llm_batch(authors,tokenizer,model,batch_size=16):

    results=[]

    for i in tqdm(
        range(0,len(authors),batch_size),
        desc="LLM fallback"
    ):

        batch=authors[i:i+batch_size]

        texts=[
            tokenizer.apply_chat_template(
                [
                    {
                        "role":"system",
                        "content":"Return JSON only."
                    },
                    {
                        "role":"user",
                        "content":build_prompt(a)
                    }
                ],
                tokenize=False,
                add_generation_prompt=True
            )
            for a in batch
        ]

        inputs=tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        ).to(model.device)


        with torch.inference_mode():

            out=model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False
            )


        for j in range(len(batch)):

            text=tokenizer.decode(
                out[j][inputs.input_ids[j].shape[0]:],
                skip_special_tokens=True
            )

            try:
                m=re.search(r"\{.*\}",text,re.DOTALL)
                results.append(json.loads(m.group()) if m else {})
            except:
                results.append({})

    return results



# ================= Main =================


papers=json.load(open(INPUT_PATH,"r",encoding="utf-8"))

author_ids=set()

for p in papers:
    for a in p.get("authorships",[]):
        aid=a.get("author",{}).get("id")
        if aid:
            author_ids.add(aid)

author_ids=list(author_ids)

print("Unique authors:",len(author_ids))


openalex=OpenAlexClient(
    OPENALEX_API_KEY,
    OPENALEX_CACHE
)

openalex.resolve(author_ids)


resolved=set(openalex.cache.keys())

failed=[]

for aid in author_ids:
    cid=openalex.clean_id(aid)
    if cid not in resolved:
        failed.append(aid)


print("OpenAlex resolved:",len(resolved))
print("LLM fallback:",len(failed))


# LLM cache

if os.path.exists(LLM_CACHE):
    llm_cache=json.load(open(LLM_CACHE,"r",encoding="utf-8"))
else:
    llm_cache={}


need=[]

for aid in failed:
    name=aid
    if name not in llm_cache:
        need.append({
            "id":aid,
            "name":aid
        })


if need:

    tokenizer,model=load_llama()

    llm_results=llm_batch(
        need,
        tokenizer,
        model
    )

    for a,r in zip(need,llm_results):
        llm_cache[a["id"]]=r


    json.dump(
        llm_cache,
        open(LLM_CACHE,"w",encoding="utf-8"),
        ensure_ascii=False
    )

    del model,tokenizer
    gc.collect()
    torch.cuda.empty_cache()



# merge

for idx,p in enumerate(tqdm(papers,desc="Merge")):

    authors=[]

    for item in p.get("authorships",[]):

        a=item.get("author",{})
        aid=openalex.clean_id(a.get("id"))

        if aid in openalex.cache:
            authors.append(openalex.cache[aid])

        else:
            r=llm_cache.get(aid,{})
            r["name"]=a.get("display_name","")
            r["source"]="LLM"
            authors.append(r)


    p["standardized_authors"]=authors

    if idx % 50000 == 0 and idx > 0:
        with open(
                CHECKPOINT,
                "w",
                encoding="utf-8"
        ) as f:
            json.dump(
                papers[:idx],
                f,
                ensure_ascii=False
            )


json.dump(
    papers,
    open(OUTPUT_PATH,"w",encoding="utf-8"),
    ensure_ascii=False,
    indent=2
)


print("Finished:",len(papers))