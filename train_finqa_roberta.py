"""
FinQA-RoBERTa-large NLI fine-tuning on "our" dataset.

Architecture mirrors the QuanTemp notebook exactly:
  - Base:      roberta-large-mnli  (NOT roberta-base)
  - Weights:   FinQA ELASTIC checkpoint loaded into roberta backbone
               (download from https://drive.google.com/drive/folders/1FmaelDhJ7QwsRTs8H0B4vYliw_qjL7P-)
               Place the file at:  models/checkpoint_best_0.65.pt
  - Head:      Dropout(0.1) → Linear(1024→768) → ReLU → Linear(768→3)
  - max_length: 256  (not 512 — QuanTemp uses 256)
  - batch_size: 16
  - lr:         2e-5, eps=1e-8, NO scheduler step (scheduler defined but not stepped)
  - epochs:     up to 20, with EarlyStopping(patience=2) on val loss
  - First 5 encoder layers FROZEN after model init
  - Loss:       CrossEntropyLoss
  - Seed:       42
  - num_evidences: ALL evidences per claim (set() dedup, same as QuanTemp)
    but we cap at 2 to match our experiment
"""

import json
import os
import random
import time
import datetime
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoModel, RobertaTokenizer, AdamW, get_linear_schedule_with_warmup
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
from pathlib import Path
from collections import Counter

SCRIPT_DIR = Path(__file__).parent.resolve()

# ─────────────────────────────────────────────
# CONFIG  (mirrors QuanTemp notebook exactly)
# ─────────────────────────────────────────────
CONFIG = {
    "base_model":       "roberta-large-mnli",
    "finqa_ckpt":       "models/checkpoint_best_0.65.pt",  # FinQA ELASTIC weights
    "max_length":       256,          # QuanTemp uses 256, not 512
    "batch_size":       16,           # QuanTemp uses 16
    "lr":               2e-5,
    "adam_eps":         1e-8,
    "epochs":           20,           # max epochs; early stopping kicks in
    "early_stop_patience": 2,         # patience=2, exactly as notebook
    "freeze_first_n_layers": 5,       # freeze layers 0-4 of encoder
    "hidden_dim":       1024,         # roberta-large hidden size
    "mlp_dim":          768,
    "dropout":          0.1,
    "num_evidences":    2,            # cap at 2 to match our experiment
    "seed":             42,
}


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round(elapsed))))


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    return np.sum(pred_flat == labels.flatten()) / len(labels.flatten())


# ─────────────────────────────────────────────
# EARLY STOPPING  (exact copy from notebook)
# ─────────────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience=2, verbose=True, delta=0, path="checkpoint.pt"):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self._save(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self._save(val_loss, model)
            self.counter = 0

    def _save(self, val_loss, model):
        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...")
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


# ─────────────────────────────────────────────
# MODEL  (exact architecture from notebook)
# ─────────────────────────────────────────────
class MultiClassClassifier(nn.Module):
    """
    FinQA-RoBERTa architecture from QuanTemp notebook.
    backbone  : roberta-large-mnli with FinQA ELASTIC weights loaded
    head      : Dropout → Linear(1024→768) → ReLU → Linear(768→num_classes)
    """
    def __init__(self, base_model_path, num_classes, finqa_ckpt_path,
                 hidden_dim=1024, mlp_dim=768, dropout=0.1):
        super().__init__()

        # Load FinQA checkpoint weights
        print(f"Loading FinQA ELASTIC checkpoint from: {finqa_ckpt_path}")
        state_dict = torch.load(finqa_ckpt_path, map_location="cpu")

        # The checkpoint stores weights under "plm_model.*" keys — strip that prefix
        state_dict_final = {}
        for key, value in state_dict.items():
            if "plm_model" in key:
                state_dict_final[key.split("plm_model.")[1]] = value

        print(f"  FinQA weights found: {len(state_dict_final)} tensors")

        # Load roberta-large-mnli and overwrite backbone weights with FinQA weights
        self.roberta = AutoModel.from_pretrained(
            base_model_path,
            output_hidden_states=True,
            output_attentions=True
        )
        missing, unexpected = self.roberta.load_state_dict(state_dict_final, strict=False)
        print(f"  Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")

        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, num_classes)
        )

    def forward(self, tokens, masks):
        output = self.roberta(tokens, attention_mask=masks)
        x = self.dropout(output["pooler_output"])
        return self.mlp(x)


# ─────────────────────────────────────────────
# FEATURE EXTRACTION
# (QuanTemp uses ALL evidences dedup'd with set();
#  we cap at num_evidences=2 to match our experiment)
# ─────────────────────────────────────────────
def get_features(data, num_evidences=None):
    """
    Build input strings in QuanTemp format:
    [Claim]: ... [Questions]: ... [Evidences]: ...

    num_evidences=None  → use ALL (original QuanTemp behaviour)
    num_evidences=2     → cap at first 2 evidence entries (our experiment)
    """
    features = []
    for fact in data:
        claim = fact["claim"]
        evidences, questions = [], []

        for i, ev in enumerate(fact["evidences"]):
            if num_evidences is not None and i >= num_evidences:
                break
            if ev.get("top_k_doc"):
                evidences.append(ev["top_k_doc"][0])
            questions.append(ev["questions"])

        # QuanTemp deduplicates with set()
        questions = list(set(questions))
        evidences = list(set(evidences))

        feature = (
            "[Claim]:" + claim
            + "[Questions]:" + " ".join(questions)
            + "[Evidences]:" + " ".join(evidences)
        )
        features.append(feature)
    return features


def tokenize_features(features, tokenizer, max_length=256):
    input_ids, attention_masks = [], []
    for sent in features:
        encoded = tokenizer.encode_plus(
            sent,
            add_special_tokens=True,
            max_length=max_length,
            pad_to_max_length=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids.append(encoded["input_ids"])
        attention_masks.append(encoded["attention_mask"])
    return torch.cat(input_ids, dim=0), torch.cat(attention_masks, dim=0)


# ─────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────
def train(model, train_loader, val_loader, device, config, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, "model_weights.pt")

    optimizer = AdamW(model.parameters(), lr=config["lr"], eps=config["adam_eps"])

    # NOTE: QuanTemp defines the scheduler but never calls scheduler.step()
    # We replicate that exactly — scheduler is defined but not stepped.
    total_steps = len(train_loader) * config["epochs"]
    scheduler = get_linear_schedule_with_warmup(   # noqa: F841 (intentionally unused)
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    loss_func = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(
        patience=config["early_stop_patience"],
        verbose=True,
        path=ckpt_path
    )

    total_t0 = time.time()
    training_stats = []

    for epoch_i in range(config["epochs"]):
        print(f"\n======== Epoch {epoch_i+1} / {config['epochs']} ========")
        print("Training...")
        t0 = time.time()

        model.train()
        total_train_loss = 0
        total_train_accuracy = 0

        for step, batch in enumerate(train_loader):
            if step % 40 == 0 and step != 0:
                elapsed = format_time(time.time() - t0)
                print(f"  Batch {step:>5,}  of  {len(train_loader):>5,}.    Elapsed: {elapsed}.")

            b_ids   = batch[0].to(device)
            b_mask  = batch[1].to(device)
            b_labels = batch[2].to(device)

            model.zero_grad()
            logits = model(b_ids, b_mask)
            loss = loss_func(logits, b_labels)
            total_train_loss += loss.item()
            loss.backward()
            # NOTE: QuanTemp does NOT clip gradients (line is commented out)
            optimizer.step()
            # NOTE: scheduler.step() is NOT called (matches notebook)

            total_train_accuracy += flat_accuracy(
                logits.detach().cpu().numpy(),
                b_labels.cpu().numpy()
            )

        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_acc  = total_train_accuracy / len(train_loader)
        training_time  = format_time(time.time() - t0)

        print(f" Train Accuracy: {avg_train_acc:.2f}")
        print(f"  Average training loss: {avg_train_loss:.2f}")
        print(f"  Training epoch took: {training_time}")

        # ── Validation ──
        print("\nRunning Validation...")
        t0 = time.time()
        model.eval()
        total_eval_loss = 0
        total_eval_accuracy = 0

        with torch.no_grad():
            for batch in val_loader:
                b_ids    = batch[0].to(device)
                b_mask   = batch[1].to(device)
                b_labels = batch[2].to(device)

                logits = model(b_ids, b_mask)
                loss = loss_func(logits, b_labels)
                total_eval_loss += loss.item()
                total_eval_accuracy += flat_accuracy(
                    logits.cpu().numpy(), b_labels.cpu().numpy()
                )

        avg_val_loss = total_eval_loss / len(val_loader)
        avg_val_acc  = total_eval_accuracy / len(val_loader)
        val_time     = format_time(time.time() - t0)

        print(f"  Accuracy: {avg_val_acc:.2f}")

        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        print(f"  Validation Loss: {avg_val_loss:.2f}")
        print(f"  Validation took: {val_time}")

        training_stats.append({
            "epoch": epoch_i + 1,
            "Training Loss": avg_train_loss,
            "Valid. Loss": avg_val_loss,
            "Valid. Accur.": avg_val_acc,
        })

    print("\nTraining complete!")
    print(f"Total training took {format_time(time.time() - total_t0)} (h:mm:ss)")
    return ckpt_path, training_stats


# ─────────────────────────────────────────────
# EVALUATE
# ─────────────────────────────────────────────
def evaluate(model, test_loader, device, label_names):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            b_ids    = batch[0].to(device)
            b_mask   = batch[1].to(device)
            b_labels = batch[2].to(device)

            logits = model(b_ids, b_mask)
            preds  = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(b_labels.cpu().numpy())

    acc         = accuracy_score(all_labels, all_preds)
    f1_weighted = f1_score(all_labels, all_preds, average="weighted")
    f1_macro    = f1_score(all_labels, all_preds, average="macro")

    print(f"\nAccuracy:    {acc:.4f}")
    print(f"Weighted F1: {f1_weighted:.4f}")
    print(f"Macro F1:    {f1_macro:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=label_names))

    return acc, f1_weighted, f1_macro


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    # ── Seed ──
    random.seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    torch.manual_seed(CONFIG["seed"])
    torch.cuda.manual_seed_all(CONFIG["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Load JSON data ──
    def load_json(name):
        path = SCRIPT_DIR / name
        with open(path) as f:
            return json.load(f)

    print("\nLoading data...")
    train_data = load_json("our_train.json")
    val_data   = load_json("our_val.json")
    test_data  = load_json("our_test.json")

    print(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")
    print(f"Train label dist: {Counter(d['label'] for d in train_data)}")

    # ── Label encoding (QuanTemp uses sklearn LabelEncoder) ──
    LE = LabelEncoder()
    train_labels = [d["label"] for d in train_data]
    val_labels   = [d["label"] for d in val_data]
    test_labels  = [d["label"] for d in test_data]

    train_labels_enc = LE.fit_transform(train_labels)
    val_labels_enc   = LE.transform(val_labels)
    test_labels_enc  = LE.transform(test_labels)

    num_classes  = len(LE.classes_)
    label_names  = list(LE.classes_)
    print(f"Label mapping: { {c: i for i, c in enumerate(LE.classes_)} }")

    # ── Tokenizer ──
    print(f"\nLoading tokenizer: {CONFIG['base_model']}")
    tokenizer = RobertaTokenizer.from_pretrained(CONFIG["base_model"])

    # ── Build features ──
    print(f"\nBuilding features (num_evidences={CONFIG['num_evidences']})...")
    train_features = get_features(train_data, CONFIG["num_evidences"])
    val_features   = get_features(val_data,   CONFIG["num_evidences"])
    test_features  = get_features(test_data,  CONFIG["num_evidences"])

    print("Tokenizing...")
    train_ids, train_masks = tokenize_features(train_features, tokenizer, CONFIG["max_length"])
    val_ids,   val_masks   = tokenize_features(val_features,   tokenizer, CONFIG["max_length"])
    test_ids,  test_masks  = tokenize_features(test_features,  tokenizer, CONFIG["max_length"])

    # ── Datasets & loaders ──
    train_labels_t = torch.tensor(train_labels_enc)
    val_labels_t   = torch.tensor(val_labels_enc)
    test_labels_t  = torch.tensor(test_labels_enc)

    train_dataset = TensorDataset(train_ids, train_masks, train_labels_t)
    val_dataset   = TensorDataset(val_ids,   val_masks,   val_labels_t)
    test_dataset  = TensorDataset(test_ids,  test_masks,  test_labels_t)

    train_loader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset),
                              batch_size=CONFIG["batch_size"])
    val_loader   = DataLoader(val_dataset,   sampler=SequentialSampler(val_dataset),
                              batch_size=CONFIG["batch_size"])
    test_loader  = DataLoader(test_dataset,  sampler=SequentialSampler(test_dataset),
                              batch_size=CONFIG["batch_size"])

    # ── Model ──
    finqa_ckpt = SCRIPT_DIR / CONFIG["finqa_ckpt"]
    model = MultiClassClassifier(
        base_model_path = CONFIG["base_model"],
        num_classes     = num_classes,
        finqa_ckpt_path = str(finqa_ckpt),
        hidden_dim      = CONFIG["hidden_dim"],
        mlp_dim         = CONFIG["mlp_dim"],
        dropout         = CONFIG["dropout"],
    )

    # Freeze first 5 encoder layers (exact QuanTemp setup: layers 0-4)
    print(f"\nFreezing first {CONFIG['freeze_first_n_layers']} encoder layers...")
    for param in model.roberta.encoder.layer[:CONFIG["freeze_first_n_layers"]].parameters():
        param.requires_grad = False

    model.to(device)

    # ── Train ──
    save_dir = str(SCRIPT_DIR / "models" / "finqa_roberta_our_dataset")
    ckpt_path, stats = train(model, train_loader, val_loader, device, CONFIG, save_dir)

    # ── Load best checkpoint and evaluate ──
    print("\n" + "="*60)
    print("EVALUATION ON TEST SET (best checkpoint)")
    print("="*60)

    best_model = MultiClassClassifier(
        base_model_path = CONFIG["base_model"],
        num_classes     = num_classes,
        finqa_ckpt_path = str(finqa_ckpt),
        hidden_dim      = CONFIG["hidden_dim"],
        mlp_dim         = CONFIG["mlp_dim"],
        dropout         = CONFIG["dropout"],
    )
    best_model.load_state_dict(torch.load(ckpt_path, map_location=device))
    best_model.to(device)

    evaluate(best_model, test_loader, device, label_names)


if __name__ == "__main__":
    main()
