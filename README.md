# 🔢 Numerical Claim Verification

> **Master's Thesis Project (MTP)** — A end-to-end pipeline for verifying numerical claims using question decomposition, temporal query refinement, web-based evidence retrieval, and Natural Language Inference (NLI).

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Pipeline Architecture](#-pipeline-architecture)
- [Module 1: Claim Decomposition](#-module-1-claim-decomposition)
  - [WH-Question Generation](#wh-question-generation)
  - [MMR-Based Reranking](#mmr-based-reranking)
  - [Temporal Query Refinement](#temporal-query-refinement)
- [Module 2: Evidence Retrieval](#-module-2-evidence-retrieval)
  - [Google Custom Search API](#google-custom-search-api)
  - [Evidence Reranking](#evidence-reranking)
  - [Page Content Retrieval](#page-content-retrieval)
- [Module 3: NLI (Verdict Classification)](#-module-3-nli-verdict-classification)
  - [RoBERTa Model](#roberta-model)
  - [Qwen / FinO1-8B on HPC](#qwen--fino1-8b-on-hpc)
  - [3-Encoder Strategy](#3-encoder-strategy)
- [Dataset](#-dataset)
- [Repository Structure](#-repository-structure)
- [Setup & Reproduction](#-setup--reproduction)
- [References](#-references)

---

## 🧠 Overview

**What is numerical claim verification?**

Numerical claims are factual statements that contain quantities, percentages, statistics, or measurements — e.g., *"11.3% of Cape Town's workforce was employed in the informal economy in 2015."* These claims are abundant in news, social media, and policy documents, and are particularly prone to misquotation or context manipulation. 

**Why is this hard?**

- Numbers can be accurate but misleading when stripped of context (wrong year, wrong population, wrong unit, wrong entity).
- Manual fact-checking cannot scale with content growth so there is need of automated fact checking system.
- We focus on claims which has multiple verifiable components within one claim and requires verification of each component independently.

**What does this project do?**

This project builds a complete automated pipeline that:
1. **Decomposes** a complex numerical claim into focused retrieval queries (WH-questions).
2. **Refines** those queries by injecting temporal context (years, dates) to improve retrieval precision.
3. **Retrieves** web evidence using Google Custom Search, then fetches full page content for deeper analysis.
4. **Classifies** the claim as *Supported*, *Refuted*, or *Conflicting* using trained NLI models. Conflicting claims are those claims which are partly true and partly false, mixture, missattribution, miscaptioned, etc.

The pipeline is evaluated on the [QuanTemp dataset](https://github.com/factiverse/QuanTemp), a benchmark specifically designed for numerical claim verification.

---

## 🏗️ Pipeline Architecture

```
INPUT CLAIM
    │
    ▼
┌─────────────────────────────┐
│   CLAIM DECOMPOSITION       │
│  • WH-Question Generation   │
│  • MMR Reranking            │
│  • Temporal Query Refinement│
└────────────┬────────────────┘
             │ 3 refined queries per claim
             ▼
┌─────────────────────────────┐
│   EVIDENCE RETRIEVAL        │
│  • Google Custom Search     │
│  • Snippet Reranking        │
│  • Page Content Retrieval   │
│  • FAISS Vector DB per page │
└────────────┬────────────────┘
             │ top evidence chunks
             ▼
┌─────────────────────────────┐
│   NLI / VERDICT             │
│  • RoBERTa-Large (FinQA)    │
│  • Qwen / FinO1-8B          │
│  • 3-Encoder Strategy       │
└────────────┬────────────────┘
             │
             ▼
     TRUE / FALSE / CONFLICTING
```

---

## 📦 Module 1: Claim Decomposition

### Why decompose claims?

A long numerical claim often embeds multiple sub-facts. Querying the entire claim directly retrieves documents that simply match the keywords in the query — not necessarily documents that *verify* or *contradict* the claim. Decomposition allows targeted retrieval for each sub-fact.

**Problem with QuanTemp's YES/NO decomposition:**  
QuanTemp decomposes claims into binary yes/no questions (e.g., *"Did 11.3% work in informal economy?"*). Since entity names are preserved, retrieval systems return snippets that keyword-match the number regardless of context. This produces *confirmatory* evidence, not *contrastive* evidence needed for verification.

---

### WH-Question Generation

**Goal:** Decompose a claim into open-ended WH-questions (who, what, when, where, how many) that are diverse and cover different aspects of the claim.

**Model Weights:** [Download Here](https://drive.google.com/file/d/1q3G-t8DEoDyEYbU05YNMmH3w9zpajun_/view?usp=sharing)

**Notebook:** [`0. Basic_Wh_Q_generation+MMR_Reranking+Evidence_Retrieval+Reranking_Evidences.ipynb`](https://colab.research.google.com/drive/1mf9hEgLoAWpY6mVYhW-7qo9M6wEu8STP)

**How to run:**

```python
import torch
from transformers import T5ForConditionalGeneration, T5TokenizerFast

tokenizer = T5TokenizerFast.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base").to('cuda')
model.load_state_dict(torch.load("/path/to/Finetune_Question_Generation_22kData.pth"))

def run_model(input_string):
    generator_args = {
        "max_length": 512,
        "num_beams": 20,
        "length_penalty": 1.5,
        "no_repeat_ngram_size": 3,
        "early_stopping": True,
        "num_return_sequences": 20
    }
    input_string = "generate questions: " + input_string + " </s>"
    input_ids = tokenizer.encode(input_string, return_tensors="pt").to('cuda')
    res = model.generate(input_ids, **generator_args)
    output = tokenizer.batch_decode(res, skip_special_tokens=True)
    return [item.split("<sep>") for item in output]
```

**Key configuration:**

| Parameter | Value | Reason |
|---|---|---|
| `num_beams` | 20 | Beam search for higher quality questions |
| `num_return_sequences` | 20 | Generate many candidates before MMR filtering |
| `length_penalty` | 1.5 | Encourage longer, more specific questions |
| `no_repeat_ngram_size` | 3 | Avoid repetitive phrasing |

**Pretrained weights:** Available via [Google Drive (QuanTemp models)](https://drive.google.com/drive/folders/1FmaelDhJ7QwsRTs8H0B4vYliw_qjL7P-)

---

### MMR-Based Reranking

**Goal:** From 20 generated questions, select the 3 best that are both *relevant to the claim* and *diverse from each other*.

**Why MMR?** If we just pick the top-3 by relevance score, we often get 3 nearly identical questions. Maximal Marginal Relevance (MMR) balances relevance and diversity — it greedily selects questions that are highly relevant to the claim but dissimilar to already-selected questions.

**Library:** [`pyversity`](https://github.com/Pringled/pyversity)

**Diversity parameter:** `0.5` — equal weight between relevance and diversity (range: 0 = pure relevance, 1 = pure diversity).

**Embeddings:** `all-MiniLM-L6-v2` (SentenceTransformers) for fast cosine similarity.

```bash
pip install pyversity sentence-transformers
```

```python
from pyversity import diversify, Strategy
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

query_embedding = model.encode([claim])[0]
question_embeddings = model.encode(questions)

# Relevance scores
query_norm = query_embedding / np.linalg.norm(query_embedding)
question_norms = question_embeddings / np.linalg.norm(question_embeddings, axis=1, keepdims=True)
scores = np.dot(question_norms, query_norm)

# Diversify
result = diversify(
    embeddings=question_embeddings,
    scores=scores,
    k=3,
    strategy=Strategy.MMR,
    diversity=0.5
)
selected_indices = result.indices
```

---

### Temporal Query Refinement

**Goal:** Inject temporal constraints (years, dates, periods) from the original claim into the generated WH-questions before retrieval.

**Why?** The T5 question generation model is trained on numeric QA data. They do NOT necessarily preserve temporal constraints (years, periods) in the query. For example:

| | Text |
|---|---|
| **Claim** | *"...employed by the informal economy **in 2015**."* |
| **Generated question** | *"Who employed 161,000 people in Cape Town?"* ← no year |
| **After refinement** | *"Who employed 161,000 people in Cape Town? **2015**"* |

Without the year, retrieval may return documents from different years, producing misleading evidence.

**Tool:** [`py-heideltime`](https://github.com/HeidelTime/heideltime) — a Python wrapper for the HeidelTime temporal tagger.

```bash
pip install py-heideltime
```

```python
from py_heideltime import heideltime

result = heideltime("...employed by the informal economy in 2015.")
# Returns: [{'text': '2015', 'tid': 't1', 'type': 'DATE', 'value': '2015', 'span': [141, 145]}]
```

**Refinement logic:** For each generated question, if HeidelTime finds no temporal expression, inject the temporal value from the claim-level HeidelTime result.

> ⚠️ **Note:** HeidelTime is slow. The dataset was processed by splitting it across multiple parallel notebooks.

---

## 🔍 Module 2: Evidence Retrieval

### Google Custom Search API

**Goal:** For each refined question/query, retrieve up to 10 web search results (title, snippet, URL).

**Why Google Custom Search?**
- Programmatic access to Google's index with JSON output
- Can be configured to search the entire web or specific domains
- Supports filtering by language, date range, and file type

**Setup:**
1. Go to [Google Cloud Console](https://console.cloud.google.com/) and create a project
2. Enable the Custom Search API and get an **API Key**
3. Go to [Programmable Search Engine Control Panel](https://programmablesearchengine.google.com/controlpanel/all) and create a search engine — get the **Search Engine ID (cx)**

**Free tier:** 100 requests/day. **Paid tier:** 10,000 requests/day ($5 per 1,000 queries beyond free tier).

**Scale of this project:** 15,478 claims × 3 queries = ~46,000 queries. At 10,000/day, this takes ~5 days with the paid plan.

**Key implementation features:**
- **API key rotation:** Multiple keys rotate automatically when quota is exceeded (429 error)
- **Resumable processing:** Saves progress after each claim; can safely restart from checkpoint

```python
class SimpleGoogleRetriever:
    def __init__(self, api_keys: List[str], search_engine_id: str):
        self.api_keys = api_keys
        self.current_key_index = 0
        self.search_engine_id = search_engine_id
        self.base_url = "https://www.googleapis.com/customsearch/v1"
    
    def rotate_api_key(self):
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
    
    def search_google(self, query: str, num_results: int = 10) -> List[Dict]:
        params = {
            'key': self.get_current_api_key(),
            'cx': self.search_engine_id,
            'q': query,
            'num': min(num_results, 10)
        }
        response = requests.get(self.base_url, params=params, timeout=10)
        if response.status_code == 429:
            self.rotate_api_key()
            return self.search_google(query, num_results)  # retry
        # ... parse and return results
```

> ⚠️ **Important:** Replace API keys and Search Engine ID with your own credentials. The ones in the code are samples.

---

### Evidence Reranking

**Goal:** From 10 retrieved snippets per query, select the single most relevant snippet to the claim.

**Method:** Cosine similarity between the claim embedding and each snippet embedding using `all-MiniLM-L6-v2`.

```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')
claim_embedding = model.encode(claim, convert_to_tensor=True)
snippet_embeddings = model.encode(snippets, convert_to_tensor=True)

scores = util.cos_sim(claim_embedding, snippet_embeddings)[0]
top_idx = scores.argmax()
top_snippet = snippets[top_idx]
```

---

### Page Content Retrieval

**Goal:** Go beyond the short Google snippet (~160 chars) and retrieve the full article content for deeper evidence.

**Why?** Google snippets are truncated. The actual numerical context (tables, surrounding paragraphs) is often in the full page. Full-page retrieval allows semantic search over a much richer evidence base.

**Full pipeline** (implemented in `2.1 Page Content Retrieval Strategy.ipynb`):

1. **Load data** — claims with questions, snippets, and source links
2. **Match questions to QuanTemp questions** — SBERT cosine similarity, Quantemp questions have entities but not ours so missing entities from our questions but present in quantemp most similar question can be our target entity.
3. **Extract target entities** — using [GLiNER](https://github.com/urchade/GLiNER) on matched QuanTemp questions
4. **Fetch article content** — HTTP request to source URL; fall back to snippets if page is inaccessible
5. **Build per-article FAISS vector DB** — chunk page content, embed with SBERT, index with FAISS
6. **Multi-query retrieval** — query FAISS with: (a) question, (b) snippet splits by `...`, (c) individual target entities
7. **Threshold-based chunk selection:**

| Similarity Score | Action |
|---|---|
| `> 0.7` | Replace snippet with chunk (chunk is better) |
| `0.5 – 0.7` | Keep both chunk and original snippet |
| `< 0.5` | Keep original snippet only (chunk not useful) |

8. **Entity comparison** — extract entities from chunks (score > 0.5) and compare against target entities:

| Condition | Signal |
|---|---|
| Entities match + good chunk | Likely **TRUE** |
| Good chunk + missing entities | Likely **CONFLICTING** or **FALSE** |
| No good chunk + missing entities | **CONFLICTING** |

9. **Checkpoint saving** — auto-saves every N records for safe resumption
10. **Export** — enriched JSON with claim, queries, evidence chunks, entity signals

---

## 🤖 Module 3: NLI (Verdict Classification)

### Goal

Given a claim and retrieved evidence, classify the claim as:
- **Supported** — evidence confirms the claim
- **Refuted** — evidence contradicts the claim
- **Conflicting** — some evidences supports and some contradicts the claim

---

### RoBERTa Model

**Base model:** `roberta-large`, fine-tuned on FinQA numerical reasoning data.

**Notebooks/Scripts:**
- Base training: [Kaggle notebook](https://www.kaggle.com/code/madhavdeshatwad/train-nli-model)
- Repository: [train_nli_model](https://github.com/sdmadhav/train_nli_model/tree/main)
- FinQA fine-tuning: [`train_finqa_roberta.py`](https://github.com/sdmadhav/mtp_NUMERICAL_CLAIM_VERIFICATION/blob/main/train_finqa_roberta.py)

---

### Qwen / FinO1-8B on HPC

**Model:** [`TheFinAI/Fino1-8B`](https://huggingface.co/TheFinAI/Fino1-8B) — a reasoning-focused financial LLM.

**Why HPC?** The 8B parameter model requires a GPU with ≥40GB VRAM. This was run on a Slurm-based HPC cluster.

> ⚠️ **HPC Constraint:** Compute nodes typically have no internet access. The model must be downloaded on a login node first.

#### Step-by-step HPC Setup

**1. Create and activate environment (login node)**
```bash
conda create -n roberta-hpc python=3.10 -y
conda activate roberta-hpc
```

**2. Install dependencies**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate datasets peft scikit-learn pandas numpy huggingface_hub
```

**3. Download model to shared filesystem (login node only)**
```bash
python - <<EOF
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="TheFinAI/Fino1-8B",
    local_dir="/path/to/shared/storage/fin-o1-8b",
    local_dir_use_symlinks=False,
    resume_download=True
)
EOF
```

**4. Verify download**
```bash
ls /path/to/shared/storage/fin-o1-8b
# Should show: config.json, tokenizer.json, *.safetensors
```

**5. Submit training job**
```bash
sbatch run_fin_o1.slurm
```

**Slurm script (`run_fin_o1.slurm`):**
```bash
#!/bin/bash
#SBATCH --job-name=fin_o1_fact
#SBATCH --output=logs/fin_o1_%j.out
#SBATCH --error=logs/fin_o1_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=64G

source /home/apps/compilers/anaconda3/2024/etc/profile.d/conda.sh
conda activate roberta-hpc
python train_fin_o1_8b.py
```

**Training script:** [`3.2 train_fin_o1_8b.py`](https://github.com/sdmadhav/mtp_NUMERICAL_CLAIM_VERIFICATION/blob/main/3.2%20train_fin_o1_8b.py)

#### Common HPC Commands

| Command | Purpose |
|---|---|
| `sbatch run_fin_o1.slurm` | Submit job to queue |
| `squeue --me` | Check your job status (RUNNING, PENDING, etc.) |
| `cat logs/fin_o1_51291.err` | View error output |
| `nano run_fin_o1.slurm` | Edit slurm script (Ctrl+O save, Ctrl+X exit) |
| `cd roberta_factcheck/` | Navigate to working directory |
| `ls` | List files in current directory |
| `history` | View previous terminal commands |

---

### 3-Encoder Strategy

A custom architecture using three separate encoders for:
1. The **claim** 
2. The **question/query**
3. The **evidence**

The three representations are combined for final verdict classification. Details and training scripts are in the repository.

---

## Dataset

**Primary benchmark:** [QuanTemp](https://github.com/factiverse/QuanTemp/tree/main/data/raw_data)

QuanTemp is a dataset of numerical claims annotated with:
- Claim text and taxonomy (Temporal, Comparative, Estimative, etc.)
- Pre-generated YES/NO decomposition questions
- Target numerical entities

**Scale:** ~15,478 claims

**Intermediate data files generated by this pipeline:**

| File | Description |
|---|---|
| `Processed_complete_dataset.json` | Full dataset with taxonomy labels |
| `claims_with_mmr_and_selected_questions.json` | Claims with top-3 MMR-selected WH-questions |
| `comparison_claims.json` | Subset of comparison-type claims |
| `claims_with_evidence.json` | Claims + retrieved Google snippets |
| `claims_with_evidence2.json` | Extended evidence file |

---

## 📁 Repository Structure

```
mtp_NUMERICAL_CLAIM_VERIFICATION/
│
├── 1. heideltime_temporal_entity_extraction.ipynb
│   └── WH-question generation + temporal entity extraction using HeidelTime
│
├── claim_verification_analysis_updated_v2.ipynb
│   └── Full page content retrieval + FAISS vector search + entity comparison pipeline
│
├── 3.2 train_fin_o1_8b.py
│   └── Fine-tuning FinO1-8B on HPC for NLI verdict classification
│
├── train_finqa_roberta.py
│   └── RoBERTa-Large fine-tuning on FinQA data
│
├── run_fin_o1.slurm
│   └── Slurm job submission script for HPC
│
├── datasets/
│   └── comparison_claims.json (and other intermediate data)
│
└── README.md
```

---

## ⚙️ Setup & Reproduction

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (≥ 16GB VRAM for RoBERTa; ≥ 40GB for Fino1-8B)
- Google Custom Search API key + Search Engine ID

### Installation

```bash
git clone https://github.com/sdmadhav/mtp_NUMERICAL_CLAIM_VERIFICATION.git
cd mtp_NUMERICAL_CLAIM_VERIFICATION

pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate datasets peft
pip install sentence-transformers faiss-cpu
pip install py-heideltime pyversity
pip install pandas numpy scikit-learn requests
```

### End-to-End Reproduction

Follow these steps in order:

**Step 1 — Question Generation & Temporal Refinement**
```bash
# Open and run notebook:
jupyter notebook "1. heideltime_temporal_entity_extraction.ipynb"
```
- Download the T5 model weights from [Google Drive](https://drive.google.com/drive/folders/1FmaelDhJ7QwsRTs8H0B4vYliw_qjL7P-)
- Input: `Processed_complete_dataset.json`
- Output: `claims_with_mmr_and_selected_questions.json`

**Step 2 — Evidence Retrieval**
- Set up your Google Custom Search API key and Search Engine ID
- Run the retrieval loop (code in the Evidence Retrieval section above)
- Input: `claims_with_mmr_and_selected_questions.json`
- Output: `claims_with_evidence.json`

**Step 3 — Page Content Retrieval**
```bash
jupyter notebook claim_verification_analysis_updated_v2.ipynb
```
- Input: `claims_with_evidence.json`
- Output: Enriched JSON with full-page evidence chunks and entity signals

**Step 4 — NLI Training (RoBERTa)**
```bash
python train_finqa_roberta.py
```

**Step 5 — NLI Training (FinO1-8B on HPC)**
```bash
# On HPC login node:
sbatch run_fin_o1.slurm
```

---

## 📎 References

- **QuanTemp Dataset & Benchmark:** [github.com/factiverse/QuanTemp](https://github.com/factiverse/QuanTemp)
- **QuanTemp Trained Models:** [Google Drive](https://drive.google.com/drive/folders/1FmaelDhJ7QwsRTs8H0B4vYliw_qjL7P-)
- **HeidelTime (Temporal Tagger):** [github.com/HeidelTime/heideltime](https://github.com/HeidelTime/heideltime)
- **pyversity (MMR):** [github.com/Pringled/pyversity](https://github.com/Pringled/pyversity)
- **Google Custom Search API:** [Programmable Search Engine](https://programmablesearchengine.google.com/controlpanel/all)
- **TheFinAI/Fino1-8B:** [HuggingFace](https://huggingface.co/TheFinAI/Fino1-8B)
- **First Semester Colab Work:** [Colab Notebook](https://colab.research.google.com/drive/1mf9hEgLoAWpY6mVYhW-7qo9M6wEu8STP?usp=sharing)
- **Question Generation Model Paper:** [Understanding Numerical Context by Asking Quantitative Questions](https://doi.org/10.1007/978-3-031-88720-8_35)

---

## Author

**Madhav Shivaji Deshatwad** — Master's Thesis Project  
Indian Institute of Technology, Palakkad  
[GitHub Profile](https://github.com/sdmadhav)
