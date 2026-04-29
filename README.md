# mtp_NUMERICAL_CLAIM_VERIFICATION

## INTRODUCTION
We started with literature review and this are the helpful links. 
Quantemp Repository: https://github.com/factiverse/QuanTemp

Quantemp Trained models: https://drive.google.com/drive/folders/1FmaelDhJ7QwsRTs8H0B4vYliw_qjL7P-

Quantemp Dataset: https://github.com/factiverse/QuanTemp/tree/main/data/raw_data

## STANDARD PIPELINE
<img width="1660" height="186" alt="image" src="https://github.com/user-attachments/assets/7d3ac29d-24d9-448b-8642-2c2de3b42575" />

## CLAIM DECOMPOSITION
It is a method where we decompose a complex claim into sub-questions. Quantmep has decomposed claims into YES-NO type questions. Entity mentioned in question so when we do retrieval as the retrieval is keywords matching based retrieval it will retrieve the evidences which have keyword match from question. This does not bring contrastive evidences. Example: "Did 11.3% work in informal economy?" → snippets with "11.3%" regardless of context


### WH-QUESTION GENERATION
We proposed the strategy of decomposing claims into Wh-Questions. 
```python
import torch
from transformers import T5ForConditionalGeneration, T5TokenizerFast
tokenizer = T5TokenizerFast.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base").to('cuda')
model.load_state_dict(torch.load("/content/drive/MyDrive/NUMERICAL CLAIM VERIFICATION/Finetune_Question_Generation_22kData.pth"))

def run_model(input_string, **generator_args):
  generator_args = {
  "max_length": 512, #256,
  "num_beams": 20,
  "length_penalty": 1.5,
  "no_repeat_ngram_size": 3,
  "early_stopping": True,
  "num_return_sequences" : 20
  }
  input_string = "generate questions: " + input_string + " </s>"
  input_ids = tokenizer.encode(input_string, return_tensors="pt").to('cuda')
  res = model.generate(input_ids, **generator_args)
  output = tokenizer.batch_decode(res, skip_special_tokens=True)
  output = [item.split("<sep>") for item in output]
  return output
sentence = 'Twitter owner Elon Musk tweeted that Democrats paid former Twitter CEO Jack Dorsey "millions of dollars" to block negative information about the Bidens in 2020.'
questions = run_model(sentence)

```

### MMR BASED RERANKING
After we have the generated questions, we will need to keep only questions which are diverse and relevent to claim. For this purpose we have used MMR strategy which keeps the most relevant items while down-weighting those too similar to what's already picked. The parameter diversity in diversify method controls the balance between diversity and relevance. We kept it 0.5. Documentation of this library can be found at this [github repository.](https://github.com/Pringled/pyversity)

```python
!pip install pyversity
import numpy as np
from pyversity import diversify, Strategy
from sentence_transformers import SentenceTransformer
model_name='all-MiniLM-L6-v2'
model = SentenceTransformer(model_name)

import pandas as pd
import json

try:
    with open('comparison_claims.json', 'r') as f: #as we were working on comparison claims earlier. This file can be found in the datasets folder.
        comparison_claims_data = json.load(f)
    df_comparison = pd.DataFrame(comparison_claims_data)
    print("DataFrame created from JSON file:")
    display(df_comparison.head())
except FileNotFoundError:
    print("Error: 'comparison_claims.json' not found. Please make sure the file exists.")
except Exception as e:
    print(f"An error occurred while reading the JSON file: {e}")


indices_collected = []
for index, claim in enumerate(comparison_claims_data):
  query = claim['claim']
  gayathri_questions = claim['gayathri_generated_questions']
  # Define embeddings and scores (e.g. cosine similarities of a query result)
  # Generate embeddings
  query_embedding = model.encode([query])[0]
  question_embeddings = model.encode(gayathri_questions)

  # Calculate relevance scores (cosine similarity)
  query_norm = query_embedding / np.linalg.norm(query_embedding)
  question_norms = question_embeddings / np.linalg.norm(question_embeddings, axis=1, keepdims=True)
  scores = np.dot(question_norms, query_norm)
  # Diversify the result
  diversified_result = diversify(
      embeddings=question_embeddings,
      scores=scores,
      k=3, # Number of questions to select
      strategy=Strategy.MMR, # Diversification strategy to use
      diversity=0.5 # Diversity parameter (higher values prioritize diversity)
  )

  # Get the indices of the diversified result
  diversified_indices = diversified_result.indices
  indices_collected.append({'claim': claim['claim'], 'indices': diversified_indices})
  print(index)
```

### TEMPORAL QUERY REFINEMENT
We found that the question generation model is trained on numerical data but it does not necessarily preserve temporal constraints (years, periods) in the query. So we came up with a strategy of temporal query refinement, where we detect temporal entities in the claim then inject them into the generated WH question and rewrite query before evidence retrieval. The temporal entity detection part was done by [heideltime](https://github.com/HeidelTime/heideltime) library in python. This library works so slowly so I splitted the dataset across multiple notebooks which ran parallely.

```python
pip install py-heideltime
!pip install gdown
!gdown --id 1jU264txlUGx4n0lWhb4VP41qBbPk-u_N #input dataset file in drive which contains all the claims, generated and reranked questions from the above stages.
print("File downloaded successfully!")

# For each claim get the temporal tags.

import json
import pandas as pd
df = pd.read_json('Processed_complete_dataset.json')
df = df[df['taxonomy' == 'Temporal']]
from py_heideltime import heideltime
def process_chunk(text_chunk):
    return heideltime(text_chunk)
from tqdm import tqdm
tqdm.pandas()

df["result"] = df['claim'].progress_apply(process_chunk)

''' Do same for questions of each claim and update only those questions which don't have any heidaltime result i.e heidaltime returned an empty dictionary.
For example:
Claim: Estimates by the City of Cape Town found that 161,000 individuals, or 11.3% of the total workforce, were employed by the informal economy in 2015.
generated question: "Who employed 161,000 people in Cape Town?"
-- No temporal reference in the query
HeidalTime results: [{'text': '2015', 'tid': 't1', 'type': 'DATE', 'value': '2015', 'span': [141, 145]}]

After query refinement: "Who employed 161,000 people in Cape Town? 2015"
-- Injected temporal reference from the result of heidaltime dictionary['value']'''
```


## EVIDENCE RETRIEVAL 
Now we have refined queries we are good to go for evidence retrieval. The api we used to evidence retrieval is Google Custom Search. The Google Custom Search JSON API allows you to programmatically retrieve search results from specific websites or the entire web using RESTful requests. You can configure the engine to search only your own website, a specific list of domains, or the entire web.  Results are returned in JSON format, following the OpenSearch 1.1 specification. You can filter results by language, country, date range, or file type (e.g., PDF). Setup Requirements to use the API, you must obtain two identifiers from the Google Cloud Console, API Key  & Search Engine ID (cx) - Created in the [Programmable Search Engine Control Panel](https://programmablesearchengine.google.com/controlpanel/all). Free Tier allows only 100 requests per day so created multiple apis or if you can afford go for paid version which allows 10,000 requests per day. We have 15478 claims and 3 queries per claim so around 46000 queries. This can be a bottleneck so with paid version also it will take atleast 5 days.

### From Google Custom Search APIS
```python
import requests
import json
from typing import List, Dict
import time

class SimpleGoogleRetriever:
    def __init__(self, api_keys: List[str], search_engine_id: str):
        """
        Initialize with multiple API keys for rotation

        Args:
            api_keys: List of Google API keys
            search_engine_id: Custom Search Engine ID
        """
        self.api_keys = api_keys
        self.current_key_index = 0
        self.search_engine_id = search_engine_id
        self.base_url = "https://www.googleapis.com/customsearch/v1"

    def get_current_api_key(self):
        """Get current API key"""
        return self.api_keys[self.current_key_index]

    def rotate_api_key(self):
        """Rotate to next API key"""
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        print(f"Rotated to API key {self.current_key_index + 1}/{len(self.api_keys)}")

    def search_google(self, query: str, num_results: int = 10) -> List[Dict]:
        """
        Search Google and return results with title, snippet, link

        Args:
            query: Search query
            num_results: Number of results (max 10 per request)

        Returns:
            List of search results
        """
        params = {
            'key': self.get_current_api_key(),
            'cx': self.search_engine_id,
            'q': query,
            'num': min(num_results, 10)
        }

        try:
            response = requests.get(self.base_url, params=params, timeout=10)

            # Check for quota exceeded
            if response.status_code == 429:
                print("Quota exceeded, rotating API key...")
                self.rotate_api_key()
                time.sleep(1)
                return self.search_google(query, num_results)  # Retry with new key

            response.raise_for_status()
            results = response.json()

            documents = []
            if 'items' in results:
                for item in results['items']:
                    documents.append({
                        'title': item.get('title', ''),
                        'snippet': item.get('snippet', ''),
                        'link': item.get('link', '')
                    })

            return documents

        except Exception as e:
            print(f"Error during search: {e}")
            # Try rotating key on error
            self.rotate_api_key()
            return []

    def retrieve_for_claim(self, claim: str, questions: List[str],
                          num_results_per_question: int = 10) -> Dict:
        """
        Retrieve evidence for all questions of a claim

        Args:
            claim: The claim text
            questions: List of questions for this claim
            num_results_per_question: How many results per question

        Returns:
            Dictionary with claim, questions, and evidence
        """
        print(f"\n{'='*80}")
        print(f"Processing claim: {claim[:100]}...")
        print(f"{'='*80}")

        evidence_data = []

        for i, question in enumerate(questions, 1):
            print(f"\nQuestion {i}/{len(questions)}: {question}")

            # Search Google
            results = self.search_google(question, num_results_per_question)
            print(f"  Retrieved {len(results)} results")

            evidence_data.append({
                'question': question,
                'search_results': results
            })

            # Small delay to avoid rate limiting
            time.sleep(0.5)

        return {
            'claim': claim,
            'evidence': evidence_data
        }


# Your API keys
API_KEYS = [
    "AIzaSyBaRpS1F0r-NcpgH2Nc7DJSKmKjQeSE",
    "AIzaSyDCEL0pSQoY7PYMuvr2ckwhsGgw45i",
    "AIzaSyCeK6qyP6z4kBTkdHJo83XQUnUffvF",
    "AIzaSyCUmAY1grugj7BcLgH09OCv_fMj3sLk",
    "AIzaSyBCcbrBRsBvBnDbkOygI4iQslbkZ_E",
    "AIzaSyBzyo_3rphj8-ezIO6LEzNOo1Fv2A"
]

SEARCH_ENGINE_ID = "351648df707cb49be" #this is sample id use correct one or contact me. 

# Initialize retriever
retriever = SimpleGoogleRetriever(API_KEYS, SEARCH_ENGINE_ID)

with open("claims_with_mmr_and_selected_questions.json", 'r', encoding='utf-8') as f:
    all_claims = json.load(f)

output_file = 'claims_with_evidence.json'

# Load existing progress if file exists
if os.path.exists(output_file):
    with open(output_file, 'r', encoding='utf-8') as f:
        all_claims_evidence = json.load(f)
    print(f"Resuming from existing file with {len(all_claims_evidence)} claims already processed")
else:
    all_claims_evidence = []

# Start from where we left off
start_index = len(all_claims_evidence)

# Loop through claim_identifiers
for jind in range(start_index, min(1000, len(all_cat_claims))):
    item = all_cat_claims[jind]
    claim = item['claim']
    questions = [q['question'] for q in item['selected_questions']] #you might have different name of the key but idea is list all the questions/queries generated for which you want to retrieve evidences

    # Retrieve evidence
    claim_evidence = retriever.retrieve_for_claim(
        claim=claim,
        questions=questions, 
        num_results_per_question=10 # max 10 results can be obtained per query. Pagination will be considered as new call to api.
    )

    all_claims_evidence.append(claim_evidence)
    print("index", jind)
    jind += 1
# Save to JSON
output_file = 'claims_with_evidence2.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(all_claims_evidence, f, ensure_ascii=False, indent=2)

print(f"\n{'='*80}")
print(f"✓ Saved evidence for {len(all_claims_evidence)} claims to {output_file}")
print(f"{'='*80}")
```
### Rerank the evidence to get top snippet
``` python
from sentence_transformers import SentenceTransformer, util

# Load a sentence transformer model
# Using a smaller model for efficiency
model = SentenceTransformer('all-MiniLM-L6-v2')

def rerank_and_select_top_evidence(claim, questions_with_evidence, model):
    """
    Rerank evidence snippets for each question based on relevance to the claim
    and select the top 1 snippet for each question.
    """
    reranked_evidence = []

    # Encode the claim once
    claim_embedding = model.encode(claim, convert_to_tensor=True)

    for q_evidence in questions_with_evidence:
        question = q_evidence.get('question')
        search_results = q_evidence.get('search_results')

        if not question or not search_results:
            reranked_evidence.append({
                'question': question,
                'top_evidence': None
            })
            continue

        # Extract snippets for encoding
        snippets = [res.get('snippet', '') for res in search_results]

        # Encode snippets
        snippet_embeddings = model.encode(snippets, convert_to_tensor=True)

        # Calculate cosine similarity between claim embedding and snippet embeddings
        claim_similarity_scores = util.cos_sim(claim_embedding, snippet_embeddings)[0]

        # Combine with original search result data and sort by claim similarity
        scored_snippets = []
        for i, score in enumerate(claim_similarity_scores):
             # Ensure index is within bounds of search_results
            if i < len(search_results):
                scored_snippets.append({
                    'snippet': snippets[i],
                    'title': search_results[i].get('title', ''),
                    'link': search_results[i].get('link', ''),
                    'claim_similarity': score.item() # Convert tensor to float
                })

        # Sort by claim similarity in descending order
        sorted_snippets = sorted(scored_snippets, key=lambda x: x['claim_similarity'], reverse=True)

        # Select the top 1 evidence snippet
        top_evidence = sorted_snippets[0] if sorted_snippets else None


        reranked_evidence.append({
            'question': question,
            'top_evidence': top_evidence
        })

    return reranked_evidence
```

### PAGE CONTENT RETRIEVAL
Now at this stage we have dataset with claim temporal refined queries, evidences for each of the queries. Observe that search results documents follow structure like
```
documents.append({
                        'title': item.get('title', ''),
                        'snippet': item.get('snippet', ''),
                        'link': item.get('link', '')
                    })
```
Here we have the link of page source of the snippet. From this link idea is to get visit that page and retrieve whole page content and then create a vector database with content chunks and query those chunks to get the top chunks from the page. Queries are 3 types of queries, question, snippet split  by '...' and target entities. 
Here is the full pipeline:

**Pipeline steps :**
1. Load data
2. Match our questions -> closest QuantTemp question (SBERT cosine similarity)
3. Extract target entities from matched QT question (GLiNER)
4. Fetch article; if inaccessible, walk fallback snippets
5. Build per-article FAISS vector DB
6. Multi-query retrieval -- snippet splits (by `...`) + individual target entities as queries
7. Threshold logic -- >0.7 swap, 0.5-0.7 keep both, <0.5 snippet only
8. Entity extraction from chunks(score > 0.5) + set comparison for TRUE/CONFLICTING signal
9. Checkpoint saving -- auto-saves after every N records; resume safely
10. Empirical analysis -- inspect 10-20 cases >= 0.6 sim: snippet vs chunk
11. Export enriched JSON

---
**Decision Logic :**
```
For each question:
  -> Multi-query FAISS (snippet splits + individual entities)
    |- Best chunk score > 0.7  -> swap snippet with chunk as evidence
    |- Best chunk score 0.5-0.7 -> keep chunk + keep original snippet
    '- Best chunk score < 0.5  -> fall back to snippet only

  -> Collect entities from all chunks with score > 0.5
    -> Compare against target entities
      |- Entities match + good chunk -> likely TRUE
      |- Good chunk + missing entities -> CONFLICTING / possibly FALSE
      '- No good chunk + missing entities -> CONFLICTING
```
This whole pipeline is implemented in the claim_verification_analysis_updated_v2.ipynb. 


## NLI
### QWEN MODEL TRAINING AND TESTING
#### HPC: Download the qwen model 
On most HPC clusters (including Slurm environments):

Compute nodes do not have internet access
Direct calls to Hugging Face (e.g., from_pretrained("repo_id")) fail during jobs

Therefore, the model must be:

1. Downloaded beforehand
2. Stored on a shared filesystem
3. Loaded in offline mode

1. Create Conda Environment
```bash
conda create -n roberta-hpc python=3.10 -y
```
2. Activate Environment 
```bash
conda activate roberta-hpc
```
3. Install Dependencies
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate datasets peft scikit-learn pandas numpy huggingface_hub
```
4. Download Model
Run this on a login node (with internet).
```bash
python - <<EOF
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="TheFinAI/Fino1-8B",
    local_dir="/home/m142402008-kpal/roberta_factcheck/models/fin-o1-8b-cache/model_files",
    local_dir_use_symlinks=False,
    resume_download=True
)

print("Download complete.")
EOF
```
5. Verify Download
```bash
ls /home/m142402008-kpal/roberta_factcheck/models/fin-o1-8b-cache/model_files
```
You should see:
```
config.json
tokenizer.json
*.safetensors
```

#### Script
```python
# -*- coding: utf-8 -*-
"""
TheFinAI/Fin-o1-8B LoRA fine-tuning for fact checking
HPC-ready script (SLURM + GPU) - OFFLINE MODE
Save this file with name- train_fin_o1_8b.py
"""

import json
import os
import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import classification_report, accuracy_score
from dataclasses import dataclass
from typing import Any, Dict, List
import transformers

# ============================================================
# PATHS
# ============================================================

BASE_DIR = "/home/m142402008-kpal/roberta_factcheck"
DATA_PATH = os.path.join(BASE_DIR, "data/final_reranked_temporal+page_content_refined1.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs_fin_o1_on_final_reranked_temporal+page_content_refined1")
LOG_DIR = os.path.join(BASE_DIR, "logs_fin_o1_on_final_reranked_temporal+page_content_refined1")
FINAL_MODEL_DIR = os.path.join(BASE_DIR, "models/fin-o1-factcheck-final_on_final_reranked_temporal+page_content_refined1")

# Model path
MODEL_PATH = os.path.join(BASE_DIR, "models/fin-o1-8b-cache/model_files")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(os.path.dirname(FINAL_MODEL_DIR), exist_ok=True)

# ============================================================
# GPU CHECK
# ============================================================

print("=" * 70)
print("GPU CHECK")
print("=" * 70)
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# DATA LOADING
# ============================================================

def read_json(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data)

print("\nLoading dataset...")
df = read_json(DATA_PATH)
print(f"Total samples: {len(df)}")

# Normalize labels
df["label"] = df["label"].replace({"Half True/False": "Conflicting"})

print("\nLabel distribution:")
print(df["label"].value_counts())
print(df["reranked_our_evidences"].apply(type).value_counts())
import json

def normalize_evidences(x):
    if isinstance(x, str):
        try:
            return json.loads(x)
        except:
            return []
    return x

df["reranked_our_evidences"] = df["reranked_our_evidences"].apply(normalize_evidences)
print(df["reranked_our_evidences"].apply(type).value_counts())


def build_prompt(row):
    evidence_parts = []
    for i, ev in enumerate(row["reranked_our_evidences"], start=1):
        q = ev.get("questions", "").strip()
        docs = ev.get("top_k_doc", [])
        if len(docs) > 0:
            docs = docs[0]
        a = " ".join(docs).strip()
        if q and a:
            evidence_parts.append(f"Evidence {i}:\nQuestion: {q}\nAnswer: {a}")

    evidence_text = "\n\n".join(evidence_parts) if evidence_parts else "No evidence available."

    # CHANGED: one new line
    signal_text = ENTITY_SIGNAL_TEXT.get(row.get("entity_signal", "some_present"))

    return f"""You are a fact-checking expert. Given a claim and supporting evidence, classify the claim as True, False, or Conflicting.

Claim: {row['claim']}

{evidence_text}

Based on the evidence above, classify this claim."""

LABEL_MAP = {
    "True": "True",
    "False": "False",
    "Conflicting": "Conflicting"
}

def create_chat_example(row):
    """Create chat-formatted example for Fin-o1"""
    prompt = build_prompt(row)
    label = LABEL_MAP[row["label"]]

    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": f"Classification: {label}"}
    ]

    return {
        "messages": messages,
        "label": row["label"]
    }

# ============================================================
# SPLITS
# ============================================================

train_df = df[df["category"] == "train"]
val_df   = df[df["category"] == "validation"]
test_df  = df[df["category"] == "test"]

print(f"\nTrain: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

train_examples = [create_chat_example(row) for _, row in train_df.iterrows()]
val_examples   = [create_chat_example(row) for _, row in val_df.iterrows()]
test_examples  = [create_chat_example(row) for _, row in test_df.iterrows()]

# ============================================================
# TOKENIZER & MODEL LOADING (OFFLINE MODE)
# ============================================================

print("\nLoading tokenizer and model (OFFLINE MODE)...")
print(f"Model path: {MODEL_PATH}")

# Verify model files exist
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

# Set offline mode environment variables
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    local_files_only=True
)

# Set padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print("✓ Tokenizer loaded successfully")

# Load model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    local_files_only=True
)

print("✓ Model loaded successfully")

# ============================================================
# LORA CONFIGURATION
# ============================================================

print("\nApplying LoRA configuration...")
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Enable gradient checkpointing for LoRA
model.enable_input_require_grads()

# Ensure model is in training mode
model.train()

# ============================================================
# CUSTOM DATA COLLATOR
# ============================================================

@dataclass
class DataCollatorForCausalLM:
    """
    Custom data collator that properly handles padding for causal LM with labels
    """
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Extract input_ids
        input_ids = [f["input_ids"] for f in features]

        # Pad sequences
        batch = self.tokenizer.pad(
            {"input_ids": input_ids},
            padding=True,
            return_tensors="pt"
        )

        # Create labels (same as input_ids, with padding tokens set to -100)
        labels = batch["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        batch["labels"] = labels

        return batch

# ============================================================
# TOKENIZATION
# ============================================================

def tokenize_function(examples):
    """Tokenize chat messages for training"""
    texts = []
    for msg in examples["messages"]:
        # Apply chat template
        text = tokenizer.apply_chat_template(
            msg,
            tokenize=False,
            add_generation_prompt=False
        )
        texts.append(text)

    # Tokenize without creating labels yet (collator will handle it)
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=2048,
    )

    return tokenized

print("\nTokenizing datasets...")
train_ds = Dataset.from_list(train_examples)
val_ds = Dataset.from_list(val_examples)
test_ds = Dataset.from_list(test_examples)

# Remove columns to avoid conflicts
train_ds = train_ds.map(tokenize_function, batched=True, remove_columns=["messages", "label"])
val_ds = val_ds.map(tokenize_function, batched=True, remove_columns=["messages", "label"])
test_ds = test_ds.map(tokenize_function, batched=True, remove_columns=["messages", "label"])

print("Tokenization done.")

# ============================================================
# TRAINING ARGUMENTS
# ============================================================

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,  # Reduced for 8B model
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,  # Increased to maintain effective batch size
    learning_rate=2e-4,
    num_train_epochs=3,
    weight_decay=0.01,
    warmup_ratio=0.1,
    bf16=True,
    logging_steps=10,
    logging_dir=LOG_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    dataloader_num_workers=4,
    report_to="none",
    optim="adamw_8bit",
    gradient_checkpointing=True,
)

# ============================================================
# DATA COLLATOR
# ============================================================

data_collator = DataCollatorForCausalLM(tokenizer=tokenizer)

# ============================================================
# TRAINER
# ============================================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=data_collator,
)

# ============================================================
# TRAIN
# ============================================================

print("\n" + "=" * 70)
print("STARTING TRAINING")
print("=" * 70)

trainer.train()

# ============================================================
# SAVE MODEL
# ============================================================

model.save_pretrained(FINAL_MODEL_DIR)
tokenizer.save_pretrained(FINAL_MODEL_DIR)
print(f"\nFinal LoRA model saved to: {FINAL_MODEL_DIR}")

# ============================================================
# TEST EVALUATION
# ============================================================

print("\nEvaluating on test set...")

def extract_label(text):
    """Extract classification from generated text"""
    text = text.lower()
    if "classification: true" in text:
        return "True"
    elif "classification: false" in text:
        return "False"
    elif "classification: conflicting" in text:
        return "Conflicting"
    elif "true" in text and "false" not in text:
        return "True"
    elif "false" in text:
        return "False"
    else:
        return "Conflicting"

y_true = []
y_pred = []

model.eval()
print("\nGenerating predictions on test set...")

for i, example in enumerate(test_examples):
    if i % 50 == 0:
        print(f"Processing {i}/{len(test_examples)}...")

    prompt = tokenizer.apply_chat_template(
        example["messages"][:-1],
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    predicted_label = extract_label(generated)

    y_true.append(example["label"])
    y_pred.append(predicted_label)

# ============================================================
# METRICS
# ============================================================

print("\n" + "=" * 70)
print("TEST SET RESULTS")
print("=" * 70)

print("\nCLASSIFICATION REPORT:")
print(classification_report(y_true, y_pred, target_names=["Conflicting", "False", "True"]))

accuracy = accuracy_score(y_true, y_pred)
print(f"\nTest Accuracy: {accuracy:.4f}")

results_df = pd.DataFrame({
    "true_label": y_true,
    "predicted_label": y_pred,
    "claim": [test_df.iloc[i]["claim"] for i in range(len(y_true))],
"taxonomy":[test_df.iloc[i]["taxonomy"] for i in range(len(y_true))]
})
results_path = os.path.join(OUTPUT_DIR, "test_predictions_our2evidences.csv")
results_df.to_csv(results_path, index=False)
print(f"\nPredictions saved to: {results_path}")

print("\nTraining + Evaluation completed successfully!")

# ============================================================
# TAXONOMY-WISE EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("TAXONOMY-WISE EVALUATION")
print("=" * 70)

# Attach taxonomy info to results
results_df["taxonomy"] = test_df["taxonomy"].values

LABEL_ORDER = ["True", "False", "Conflicting"]

taxonomy_results = []

for taxonomy in sorted(results_df["taxonomy"].unique()):
    taxonomy_data = results_df[results_df["taxonomy"] == taxonomy]

    y_true_tax = taxonomy_data["true_label"].tolist()
    y_pred_tax = taxonomy_data["predicted_label"].tolist()

    accuracy_tax = accuracy_score(y_true_tax, y_pred_tax)
    true_counts = taxonomy_data["true_label"].value_counts()

    print(f"\n{'='*70}")
    print(f"Taxonomy: {taxonomy}")
    print(f"{'='*70}")
    print(f"Samples: {len(taxonomy_data)}")
    print(f"Accuracy: {accuracy_tax:.4f}")

    print("\nTrue Label Distribution:")
    for label in LABEL_ORDER:
        print(f"  {label}: {true_counts.get(label, 0)}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_true_tax,
            y_pred_tax,
            labels=LABEL_ORDER,
            target_names=LABEL_ORDER,
            zero_division=0
        )
    )

    taxonomy_results.append({
        "taxonomy": taxonomy,
        "samples": len(taxonomy_data),
        "true_count": true_counts.get("True", 0),
        "false_count": true_counts.get("False", 0),
        "conflicting_count": true_counts.get("Conflicting", 0),
        "accuracy": accuracy_tax
    })

# ============================================================
# SAVE TAXONOMY SUMMARY
# ============================================================

taxonomy_summary_df = pd.DataFrame(taxonomy_results)
taxonomy_summary_path = os.path.join(OUTPUT_DIR, "taxonomy_wise_summary_for_new_Ev_temporal.csv")
taxonomy_summary_df.to_csv(taxonomy_summary_path, index=False)

print(f"\nTaxonomy-wise summary saved to: {taxonomy_summary_path}")

# ============================================================
# CONFUSION MATRICES BY TAXONOMY
# ============================================================

from sklearn.metrics import confusion_matrix

print("\n" + "=" * 70)
print("CONFUSION MATRICES BY TAXONOMY")
print("=" * 70)

for taxonomy in sorted(results_df["taxonomy"].unique()):
    taxonomy_data = results_df[results_df["taxonomy"] == taxonomy]

    y_true_tax = taxonomy_data["true_label"].tolist()
    y_pred_tax = taxonomy_data["predicted_label"].tolist()

    cm = confusion_matrix(y_true_tax, y_pred_tax, labels=LABEL_ORDER)

    print(f"\n{taxonomy}:")
    print("Confusion Matrix (rows=actual, cols=predicted):")
    print(f"{'':>12} {'True':>10} {'False':>10} {'Conflicting':>12}")
    for i, label in enumerate(LABEL_ORDER):
        print(f"{label:>12} {cm[i][0]:>10} {cm[i][1]:>10} {cm[i][2]:>12}")

print("\n" + "=" * 70)
print("TAXONOMY-WISE EVALUATION COMPLETED SUCCESSFULLY!")
print("=" * 70)
```

To run this python file in HPC we need to create a job. Job is created using below slurm script. Command to submit the job is 'sbatch run_fin_o1.slurm'.
```slurm
#!/bin/bash
#SBATCH --job-name=fin_o1_fact
#SBATCH --output=logs/fin_o1_page_content_%j.out
#SBATCH --error=logs/fin_o1_page_content_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=64G

# Print job info
echo "=========================================="
echo "SLURM_CLUSTER_NAME = $SLURM_CLUSTER_NAME"
echo "SLURM_JOB_ACCOUNT = $SLURM_JOB_ACCOUNT"
echo "SLURM_JOB_ID = $SLURM_JOB_ID"
echo "SLURM_JOB_NAME = $SLURM_JOB_NAME"
echo "SLURM_JOB_NODELIST = $SLURM_JOB_NODELIST"
echo "SLURM_JOB_USER = $SLURM_JOB_USER"
echo "SLURM_JOB_UID = $SLURM_JOB_UID"
echo "SLURM_JOB_PARTITION = $SLURM_JOB_PARTITION"
echo "SLURM_TASK_PID = $SLURM_TASK_PID"
echo "SLURM_SUBMIT_DIR = $SLURM_SUBMIT_DIR"
echo "SLURM_CPUS_ON_NODE = $SLURM_CPUS_ON_NODE"
echo "SLURM_NTASKS = $SLURM_NTASKS"
echo "SLURM_TASK_PID = $SLURM_TASK_PID"
echo "=========================================="

# Activate conda environment
source /home/apps/compilers/anaconda3/2024/etc/profile.d/conda.sh
conda activate roberta-hpc

# Run training
python train_fin_o1_8b.py
```
#### Commands
Most often I came across these commands only.

>cat logs/fin_o1_page_content_51291.err - To show the content of the file
>
>nano run_fin_o1.slurm - to open the file in editabel mode. ctrl+o for saving the update and ctrl+x for exiting
>
> sbatch run_fin_o1.slurm - submit the job to hpc node available
>
> squeue --me - check the status of the job weather it is running, pending, etc.
>
> cd roberta_factcheck/ - change the directory
>
> ls -  list the files/folders in the current directory folder
>
> history - list the previous commands in the terminal

### ROBERTA MODEL TRAINING AND TESTING
### 3 ENCODER STRATEGY TRAINING AND RESTING

## DATASETS
