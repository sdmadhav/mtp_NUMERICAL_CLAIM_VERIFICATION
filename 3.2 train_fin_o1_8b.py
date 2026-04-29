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
