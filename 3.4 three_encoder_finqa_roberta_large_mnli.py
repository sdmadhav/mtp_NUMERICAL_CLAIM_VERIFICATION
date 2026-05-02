# -*- coding: utf-8 -*-
"""
=============================================================================
FACT VERIFICATION — MULTI-ENCODER WITH FinQA-RoBERTa-large
=============================================================================

Architecture
------------
  3 independent FinQA-RoBERTa-large encoders, one per evidence slot:

      encoder_1 : (claim, question_1, evidence_1)
      encoder_2 : (claim, question_2, evidence_2)
      encoder_3 : (claim, question_3, evidence_3)

  Each encoder input is tokenized as a pair:
      text_a = claim
      text_b = question_i + " [SEP] " + evidence_i

  Each encoder returns pooler_output (1024-d).

  Fusion:
      concat([pooler_1, pooler_2, pooler_3])     →  (batch, 3072)
      → Linear(3072 → 1024) → ReLU → Dropout(0.1)
      → Linear(1024 → 768)  → ReLU
      → Linear(768  → 3)    → logits

Training (mirrors QuanTemp / Approach-1 notebook exactly)
  - AdamW, lr=2e-5, eps=1e-8
  - Early stopping: patience=2, monitors val loss
  - Freeze encoder layers 0-4 in ALL three encoders
  - CrossEntropyLoss, seed=42
  - LabelEncoder (alphabetical): Conflicting=0, False=1, True=2
  - num_evidences cap: 2  (slots beyond cap get empty strings, same as Approach 1)

Data format (same JSON as Approach 1):
  Each record: {
    "claim":     "...",
    "label":     "True" | "False" | "Conflicting",
    "evidences": [
      {"questions": "...", "top_k_doc": ["doc1", ...]},
      ...
    ]
  }

Setup
-----
  Place the QuanTemp weights file next to this script:
      finqa_roberta_claimdecomp_early_stop_2/model_weights.zip

  Data files: /content/drive/MyDrive/claim_verification_final/our_train_final_reranked_temporal+page_content_refined.json, /content/drive/MyDrive/claim_verification_final/our_train_final_reranked_temporal+page_content_refined.json, /content/drive/MyDrive/claim_verification_final/our_train_final_reranked_temporal+page_content_refined.json
=============================================================================
"""

import json, os, random, time, datetime
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoModel, RobertaTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
from pathlib import Path
from collections import Counter

import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"   # never try HuggingFace
os.environ["HF_DATASETS_OFFLINE"]  = "1"

SCRIPT_DIR = Path(__file__).parent.resolve()


CONFIG = {
    #"base_model":            "roberta-large-mnli",
    "base_model": os.path.expandvars("$HOME/fact_verif/roberta_large_mnli_local"),
    # QuanTemp FinQA weights — loaded into ALL 3 encoders independently
    "finqa_weights":         "finqa_roberta_claimdecomp_early_stop_2/model_weights.zip",

    "max_length":            256,     # token budget per encoder (pair encoding)
    "batch_size":            16,
    "lr":                    2e-5,
    "adam_eps":              1e-8,
    "epochs":                20,
    "early_stop_patience":   2,
    "freeze_first_n_layers": 5,       # freeze layers 0-4 in each encoder's backbone

    # Must match FinQA checkpoint exactly
    "encoder_hidden":        1024,    # roberta-large hidden size  (pooler_output dim)
    "finqa_mlp_dim":         768,     # FinQA head intermediate dim (loaded, not used in fusion)

    # Fusion MLP
    "fusion_hidden1":        1024,    # 3072 → 1024
    "fusion_hidden2":        768,     # 1024 → 768

    "dropout":               0.1,
    "num_evidences":         2,       # how many evidence slots to use (rest → empty string)
    "seed":                  42,
}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round(elapsed))))

def flat_accuracy(preds, labels):
    return (np.argmax(preds, axis=1).flatten() == labels.flatten()).mean()


# ─────────────────────────────────────────────────────────────────────────────
# EARLY STOPPING  (exact copy from QuanTemp / Approach 1)
# ─────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience=2, verbose=True, delta=0, path="checkpoint.pt"):
        self.patience = patience;  self.verbose = verbose
        self.counter = 0;          self.best_score = None
        self.early_stop = False;   self.val_loss_min = np.inf
        self.delta = delta;        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score;  self._save(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score;  self._save(val_loss, model);  self.counter = 0

    def _save(self, val_loss, model):
        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...")
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE FINQA-ROBERTA ENCODER
# ─────────────────────────────────────────────────────────────────────────────

class FinQARobertaEncoder(nn.Module):
    """
    One FinQA-RoBERTa encoder instance.

    Architecture matches the QuanTemp MultiClassClassifier exactly so that
    strict=True weight loading works:
        roberta   ← backbone (roberta-large-mnli)
        dropout   ← Dropout(0.1)
        mlp       ← Linear(1024→768) → ReLU → Linear(768→3)   [FinQA head]

    In forward() we tap roberta's pooler_output BEFORE the FinQA head,
    exactly as Approach 1 does. The head weights are loaded but never called
    during fusion — they exist only to satisfy the checkpoint structure.
    """

    def __init__(self, base_model_path: str,
                 hidden_dim: int = 1024, mlp_dim: int = 768, dropout: float = 0.1):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(
            base_model_path, output_hidden_states=True, output_attentions=True)
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, 3),       # 3 FinQA classes — kept for checkpoint compatibility
        )
        self.hidden_size = hidden_dim    # 1024

    def forward(self, input_ids, attention_mask):
        """Returns pooler_output: (batch, 1024)  — same as Approach 1."""
        out = self.roberta(input_ids, attention_mask=attention_mask)
        return out["pooler_output"]      # (batch, 1024)


def _load_finqa_weights(encoder: FinQARobertaEncoder,
                        weights_path: str,
                        device,
                        tag: str) -> FinQARobertaEncoder:
    """Load QuanTemp state_dict into one encoder. Keys: roberta.*, dropout.*, mlp.0.*, mlp.2.*"""
    print(f"  [{tag}] loading FinQA weights ...")
    sd = torch.load(weights_path, map_location=device)
    missing, unexpected = encoder.load_state_dict(sd, strict=True)
    print(f"  [{tag}] done — missing: {len(missing)}, unexpected: {len(unexpected)}")
    return encoder


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-ENCODER MODEL
# ─────────────────────────────────────────────────────────────────────────────

class MultiEncoderFinQA(nn.Module):
    """
    encoder_1(claim, q1, ev1) ──┐
    encoder_2(claim, q2, ev2) ──┼──► concat (3072) ──► fusion MLP ──► logits
    encoder_3(claim, q3, ev3) ──┘

    Fusion MLP:
        Linear(3072→1024) → ReLU → Dropout
        Linear(1024→768)  → ReLU
        Linear(768→num_classes)
    """
    def __init__(self, base_model_path, num_classes,
                 encoder_hidden=1024, mlp_dim=768,
                 fusion_hidden1=1024, fusion_hidden2=768, dropout=0.1):
        super().__init__()

        # Single shared encoder — called 3 times in forward()
        self.shared_encoder = FinQARobertaEncoder(
            base_model_path, encoder_hidden, mlp_dim, dropout)

        concat_dim = encoder_hidden * 3   # 3 × 1024 = 3072
        self.fusion_mlp = nn.Sequential(
            nn.Linear(concat_dim,     fusion_hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden1, fusion_hidden2),
            nn.ReLU(),
            nn.Linear(fusion_hidden2, num_classes),
        )

    def forward(self, ids_1, mask_1, ids_2, mask_2, ids_3, mask_3):
        # Same weights, three evidence slots
        p1 = self.shared_encoder(ids_1, mask_1)   # (B, 1024)
        p2 = self.shared_encoder(ids_2, mask_2)   # (B, 1024)
        p3 = self.shared_encoder(ids_3, mask_3)   # (B, 1024)

        fused  = torch.cat([p1, p2, p3], dim=-1)  # (B, 3072)
        return self.fusion_mlp(fused)              # (B, num_classes)

# ─────────────────────────────────────────────────────────────────────────────
# TOKENIZATION  (mirrors original 3-encoder pair encoding exactly)
# ─────────────────────────────────────────────────────────────────────────────

def tokenize_pair(tokenizer, claim: str, question: str, evidence: str,
                  max_length: int):
    """
    Pair encoding — identical logic to the original BERT 3-encoder script:

        text_a = claim
        text_b = question + " [SEP] " + evidence

    RoBERTa doesn't use token_type_ids, so only input_ids and attention_mask
    are returned (both shaped (seq_len,)).
    """
    text_b = (question + " [SEP] " + evidence).strip() if (question or evidence) else " "
    enc = tokenizer(
        claim,
        text_b,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_overflowing_tokens=False,   # silences the warning
        return_tensors="pt",
    )
    return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)


def get_slot_features(data, slot_idx: int, num_evidences: int, tokenizer, max_length: int):
    """
    Tokenize one evidence slot (0-indexed) across the whole dataset.

    For each sample:
        - Applies the num_evidences cap first (same as Approach 1)
        - Picks evidence slot `slot_idx`; if the slot doesn't exist → ("", "")
        - Calls tokenize_pair(claim, question_i, evidence_i)

    Returns (input_ids, attention_masks) shaped (N, seq_len).
    """
    all_ids, all_masks = [], []

    for fact in data:
        claim   = fact["claim"]
        capped  = fact["reranked_our_evidences"]  # apply cap before slot selection

        if slot_idx < len(capped):
            ev       = capped[slot_idx]
            question = ev.get("questions", "") or ""
            evidence = ev["top_k_doc"][0] if ev.get("top_k_doc") else ""
        else:
            question, evidence = "", ""               # pad missing slots with empty strings

        ids, mask = tokenize_pair(tokenizer, claim, question, evidence, max_length)
        all_ids.append(ids.unsqueeze(0))
        all_masks.append(mask.unsqueeze(0))

    return torch.cat(all_ids, dim=0), torch.cat(all_masks, dim=0)


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING LOOP  (mirrors Approach 1 exactly)
# ─────────────────────────────────────────────────────────────────────────────

def train(model, train_loader, val_loader, device, config, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, "model_weights.pt")

    optimizer = AdamW(model.parameters(), lr=config["lr"], eps=config["adam_eps"])

    # Defined but NEVER stepped — exact QuanTemp notebook behaviour
    _ = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_loader) * config["epochs"],
    )

    loss_func      = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(
        patience=config["early_stop_patience"], verbose=True, path=ckpt_path)

    total_t0 = time.time()

    for epoch_i in range(config["epochs"]):
        print(f"\n======== Epoch {epoch_i+1} / {config['epochs']} ========")
        print("Training...")
        t0 = time.time()
        model.train()
        total_loss, total_acc = 0.0, 0.0

        for step, batch in enumerate(train_loader):
            if step % 40 == 0 and step != 0:
                print(f"  Batch {step:>5,}  of  {len(train_loader):>5,}.    "
                      f"Elapsed: {format_time(time.time()-t0)}.")

            # Unpack: [ids_1, mask_1, ids_2, mask_2, ids_3, mask_3, labels]
            ids_1, mask_1, ids_2, mask_2, ids_3, mask_3, b_labels = [
                x.to(device) for x in batch]

            model.zero_grad()
            logits = model(ids_1, mask_1, ids_2, mask_2, ids_3, mask_3)

            loss = loss_func(logits, b_labels)
            total_loss += loss.item()
            loss.backward()
            # grad clipping DISABLED — matches QuanTemp notebook
            optimizer.step()
            # scheduler.step() NOT called — matches QuanTemp notebook
            total_acc += flat_accuracy(
                logits.detach().cpu().numpy(), b_labels.cpu().numpy())

        n = len(train_loader)
        print(f"  Train Accuracy: {total_acc/n:.4f}")
        print(f"  Average training loss: {total_loss/n:.4f}")
        print(f"  Training epoch took: {format_time(time.time()-t0)}")

        # ── Validation ────────────────────────────────────────────────────────
        print("\nRunning Validation...")
        t0 = time.time()
        model.eval()
        val_loss, val_acc = 0.0, 0.0

        with torch.no_grad():
            for batch in val_loader:
                ids_1, mask_1, ids_2, mask_2, ids_3, mask_3, b_labels = [
                    x.to(device) for x in batch]
                logits    = model(ids_1, mask_1, ids_2, mask_2, ids_3, mask_3)
                val_loss += loss_func(logits, b_labels).item()
                val_acc  += flat_accuracy(logits.cpu().numpy(), b_labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        print(f"  Accuracy: {val_acc/len(val_loader):.4f}")
        print(f"  Validation Loss: {avg_val_loss:.4f}")
        print(f"  Validation took: {format_time(time.time()-t0)}")

        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    print("\nTraining complete!")
    print(f"Total training took {format_time(time.time()-total_t0)} (h:mm:ss)")
    return ckpt_path


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model, test_loader, device, label_names):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            ids_1, mask_1, ids_2, mask_2, ids_3, mask_3, b_labels = [
                x.to(device) for x in batch]
            logits = model(ids_1, mask_1, ids_2, mask_2, ids_3, mask_3)
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_labels.extend(b_labels.cpu().numpy())

    print(f"\nAccuracy:    {accuracy_score(all_labels, all_preds):.4f}")
    print(f"Weighted F1: {f1_score(all_labels, all_preds, average='weighted'):.4f}")
    print(f"Macro F1:    {f1_score(all_labels, all_preds, average='macro'):.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=label_names))


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    random.seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    torch.manual_seed(CONFIG["seed"])
    torch.cuda.manual_seed_all(CONFIG["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    def load_json(name):
        with open(SCRIPT_DIR / name) as f:
            return json.load(f)

    print("\nLoading data...")
    train_data = load_json("claim_verification_final/our_train_final_reranked_temporal+page_content_refined.json")
    val_data   = load_json("claim_verification_final/our_val_final_reranked_temporal+page_content_refined.json")
    test_data  = load_json("claim_verification_final/our_test_final_reranked_temporal+page_content_refined.json")
    print(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")
    print(f"Train labels: {Counter(d['label'] for d in train_data)}")

    # Alphabetical LabelEncoder: Conflicting=0, False=1, True=2
    LE = LabelEncoder()
    train_labels = LE.fit_transform([d["label"] for d in train_data])
    val_labels   = LE.transform([d["label"] for d in val_data])
    test_labels  = LE.transform([d["label"] for d in test_data])
    num_classes  = len(LE.classes_)
    label_names  = list(LE.classes_)
    print(f"Label mapping: { {c: i for i, c in enumerate(LE.classes_)} }")

    # Tokenizer (RoBERTa — no token_type_ids)
    print(f"\nLoading tokenizer: {CONFIG['base_model']}")
    tokenizer = RobertaTokenizer.from_pretrained(CONFIG["base_model"])

    # ── Tokenize each slot independently ─────────────────────────────────────
    # Each slot uses pair encoding: text_a=claim, text_b=question_i+[SEP]+evidence_i
    # This exactly mirrors how the original BERT 3-encoder script builds its inputs.
    print(f"\nTokenizing slots (num_evidences={CONFIG['num_evidences']}, "
          f"max_length={CONFIG['max_length']})...")

    splits = {}
    for split_name, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        print(f"  {split_name}...")
        slot_tensors = []
        for slot_idx in range(3):
            ids, masks = get_slot_features(
                data, slot_idx, CONFIG["num_evidences"], tokenizer, CONFIG["max_length"])
            slot_tensors.extend([ids, masks])  # order: ids_1, mask_1, ids_2, mask_2, ids_3, mask_3
        splits[split_name] = slot_tensors

    # ── DataLoaders ───────────────────────────────────────────────────────────
    def make_loader(slot_tensors, labels, shuffle: bool):
        labels_t = torch.tensor(labels)
        ds       = TensorDataset(*slot_tensors, labels_t)
        sampler  = RandomSampler(ds) if shuffle else SequentialSampler(ds)
        return DataLoader(ds, sampler=sampler, batch_size=CONFIG["batch_size"])

    train_loader = make_loader(splits["train"], train_labels, shuffle=True)
    val_loader   = make_loader(splits["val"],   val_labels,   shuffle=False)
    test_loader  = make_loader(splits["test"],  test_labels,  shuffle=False)

    # ── Build model ───────────────────────────────────────────────────────────
    print(f"\nBuilding MultiEncoderFinQA (backbone: {CONFIG['base_model']})...")
    model = MultiEncoderFinQA(
        base_model_path = CONFIG["base_model"],
        num_classes     = num_classes,
        encoder_hidden  = CONFIG["encoder_hidden"],
        mlp_dim         = CONFIG["finqa_mlp_dim"],
        fusion_hidden1  = CONFIG["fusion_hidden1"],
        fusion_hidden2  = CONFIG["fusion_hidden2"],
        dropout         = CONFIG["dropout"],
    )

    # ── Load FinQA weights into each encoder independently ────────────────────
    weights_path = str(SCRIPT_DIR / CONFIG["finqa_weights"])
    # Replace the three separate _load_finqa_weights calls with:
    print(f"\nLoading FinQA weights into shared encoder from: {weights_path}")
    model.shared_encoder = _load_finqa_weights(
        model.shared_encoder, weights_path, device, "shared_encoder")

    # Freeze layers 0–4 in the single shared encoder
    n_freeze = CONFIG["freeze_first_n_layers"]
    for param in model.shared_encoder.roberta.encoder.layer[:n_freeze].parameters():
        param.requires_grad = False

    total_p     = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal params: {total_p:,} | Trainable: {trainable_p:,}")

    model.to(device)

    # ── Train ─────────────────────────────────────────────────────────────────
    save_dir  = str(SCRIPT_DIR / "models" / "multi_encoder_finqa_roberta")
    ckpt_path = train(model, train_loader, val_loader, device, CONFIG, save_dir)

    # ── Evaluate best checkpoint ──────────────────────────────────────────────
    print("\n" + "="*60)
    print("EVALUATION ON TEST SET (best checkpoint by val loss)")
    print("="*60)
    best_model = MultiEncoderFinQA(
        base_model_path = CONFIG["base_model"],
        num_classes     = num_classes,
        encoder_hidden  = CONFIG["encoder_hidden"],
        mlp_dim         = CONFIG["finqa_mlp_dim"],
        fusion_hidden1  = CONFIG["fusion_hidden1"],
        fusion_hidden2  = CONFIG["fusion_hidden2"],
        dropout         = CONFIG["dropout"],
    )
    best_model.load_state_dict(torch.load(ckpt_path, map_location=device))
    best_model.to(device)
    evaluate(best_model, test_loader, device, label_names)


if __name__ == "__main__":
    main()
