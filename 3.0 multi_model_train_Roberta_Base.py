import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    RobertaTokenizer, 
    RobertaForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from tqdm import tqdm
import logging
import os
from pathlib import Path
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.resolve()

class ClaimVerificationDataset(Dataset):
    """Dataset for claim verification with flexible evidence selection"""
    
    def __init__(self, json_file, tokenizer, max_length=512, num_evidences=1):
        # Convert to absolute path
        json_path = SCRIPT_DIR / json_file if not Path(json_file).is_absolute() else Path(json_file)
        
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_evidences = num_evidences
        
        # Create label mapping from this dataset
        unique_labels = sorted(set(item['label'] for item in self.data))
        self.label_map = {label: idx for idx, label in enumerate(unique_labels)}
        
        print(f"Loaded {len(self.data)} examples from {json_path}")
        print(f"Using {self.num_evidences} evidence(s) per claim")
        print(f"Label distribution: {Counter([item['label'] for item in self.data])}")
        print(f"Label mapping: {self.label_map}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        claim = item['claim']
        evidences_data = item['evidences']
        label = self.label_map[item['label']]
        
        # Extract questions and evidences (first k)
        questions = []
        evidences = []
        
        for i, ev in enumerate(evidences_data):
            if i >= self.num_evidences:
                break
            
            if 'questions' in ev:
                questions.append(ev['questions'])
            
            if 'top_k_doc' in ev and ev['top_k_doc']:
                evidences.append(ev['top_k_doc'][0])
        
        # Create feature
        questions_str = ' '.join(questions) if questions else ""
        evidences_str = ' '.join(evidences) if evidences else ""
        
        feature = f"[Claim]: {claim} [Questions]: {questions_str} [Evidences]: {evidences_str}"
        
        encoding = self.tokenizer(
            feature,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long),
            'claim': claim
        }


class MergedDataset(Dataset):
    """Merged dataset that combines evidence from both approaches for the same claims"""
    
    def __init__(self, our_file, quantemp_file, tokenizer, max_length=512, 
                 label_map=None, num_evidences=3):
        """
        Merge datasets by claim - combine questions and evidences from both sources
        
        Args:
            our_file: Path to "our" dataset JSON
            quantemp_file: Path to QuantEmp dataset JSON
            tokenizer: RoBERTa tokenizer
            max_length: Maximum sequence length
            label_map: Label mapping (from "our" dataset)
            num_evidences: Number of evidences to use per claim from EACH dataset
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_evidences = num_evidences
        self.label_map = label_map
        
        # Convert to absolute paths
        our_path = SCRIPT_DIR / our_file if not Path(our_file).is_absolute() else Path(our_file)
        quantemp_path = SCRIPT_DIR / quantemp_file if not Path(quantemp_file).is_absolute() else Path(quantemp_file)
        
        # Load both datasets
        with open(our_path, 'r') as f:
            our_data = json.load(f)
        with open(quantemp_path, 'r') as f:
            quantemp_data = json.load(f)
        
        # Create claim-based lookup for both datasets
        our_claim_map = {item['claim']: item for item in our_data}
        quantemp_claim_map = {item['claim']: item for item in quantemp_data}
        
        # Find common claims
        common_claims = set(our_claim_map.keys()) & set(quantemp_claim_map.keys())
        
        print(f"\n📊 Dataset Merging Statistics:")
        print(f"  Our dataset: {len(our_claim_map)} claims")
        print(f"  QuantEmp dataset: {len(quantemp_claim_map)} claims")
        print(f"  Common claims: {len(common_claims)} claims")
        
        # Build merged data
        self.data = []
        for claim in common_claims:
            our_item = our_claim_map[claim]
            quantemp_item = quantemp_claim_map[claim]
            
            # Use label from "our" dataset
            self.data.append({
                'claim': claim,
                'label': our_item['label'],
                'our_evidences': our_item['evidences'],
                'quantemp_evidences': quantemp_item['evidences']
            })
        
        print(f"  Final merged dataset: {len(self.data)} examples")
        print(f"  Label distribution: {Counter([item['label'] for item in self.data])}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        claim = item['claim']
        label = self.label_map[item['label']]
        
        # Extract questions and evidences from BOTH datasets
        our_questions = []
        our_evidences = []
        quantemp_questions = []
        quantemp_evidences = []
        
        # From "our" dataset
        for i, ev in enumerate(item['our_evidences']):
            if i >= self.num_evidences:
                break
            if 'questions' in ev:
                our_questions.append(ev['questions'])
            if 'top_k_doc' in ev and ev['top_k_doc']:
                our_evidences.append(ev['top_k_doc'][0])
        
        # From QuantEmp dataset
        for i, ev in enumerate(item['quantemp_evidences']):
            if i >= self.num_evidences:
                break
            if 'questions' in ev:
                quantemp_questions.append(ev['questions'])
            if 'top_k_doc' in ev and ev['top_k_doc']:
                quantemp_evidences.append(ev['top_k_doc'][0])
        
        # Combine questions and evidences from both sources
        all_questions = our_questions + quantemp_questions
        all_evidences = our_evidences + quantemp_evidences
        
        questions_str = ' '.join(all_questions) if all_questions else ""
        evidences_str = ' '.join(all_evidences) if all_evidences else ""
        
        feature = f"[Claim]: {claim} [Questions]: {questions_str} [Evidences]: {evidences_str}"
        
        encoding = self.tokenizer(
            feature,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long),
            'claim': claim
        }


def train_model(model, train_loader, val_loader, device, config, model_name):
    """Train a single model with metrics tracking"""
    
    print(f"\n{'='*70}")
    print(f"Training Model: {model_name}")
    print(f"{'='*70}")
    
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'], 
                     weight_decay=config.get('weight_decay', 0.01))
    
    total_steps = len(train_loader) * config['epochs']
    warmup_steps = int(total_steps * config['warmup_ratio'])
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    best_f1 = 0
    best_epoch = 0
    
    # Metrics tracking
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'val_f1': []
    }
    
    for epoch in range(config['epochs']):
        # ============ TRAINING ============
        model.train()
        total_loss = 0
        train_preds = []
        train_labels = []
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            # Track predictions for training accuracy
            preds = torch.argmax(outputs.logits, dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = total_loss / len(train_loader)
        train_acc = accuracy_score(train_labels, train_preds)
        
        # ============ VALIDATION ============
        model.eval()
        val_preds = []
        val_labels = []
        val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                val_loss += outputs.loss.item()
                preds = torch.argmax(outputs.logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='weighted')
        
        # Store metrics
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        print(f"Epoch {epoch+1}:")
        print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
        
        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch + 1
            save_dir = SCRIPT_DIR / "models" / model_name
            os.makedirs(save_dir, exist_ok=True)
            model.save_pretrained(save_dir)
            print(f"  💾 Saved best model (F1: {best_f1:.4f})")
    
    print(f"\n✅ Training completed for {model_name}")
    print(f"   Best F1: {best_f1:.4f} at epoch {best_epoch}")
    
    return history, best_f1, best_epoch


def evaluate_model(model, test_loader, device, label_map, model_name):
    """Evaluate model on test set with detailed metrics"""
    
    print(f"\n{'='*70}")
    print(f"Evaluating Model: {model_name}")
    print(f"{'='*70}")
    
    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    all_claims = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_claims.extend(batch['claim'])
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    
    idx_to_label = {v: k for k, v in label_map.items()}
    label_names = [idx_to_label[i] for i in range(len(label_map))]
    
    print(f"\nResults for {model_name}:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Weighted F1: {f1_weighted:.4f}")
    print(f"  Macro F1: {f1_macro:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=label_names))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    return {
        'accuracy': accuracy,
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro,
        'predictions': all_preds,
        'true_labels': all_labels,
        'claims': all_claims,
        'confusion_matrix': cm,
        'label_names': label_names
    }


def plot_training_metrics(histories, model_names, save_dir='plots'):
    """Plot training and validation metrics"""
    
    save_path = SCRIPT_DIR / save_dir
    os.makedirs(save_path, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Training Loss
    ax = axes[0, 0]
    for history, name in zip(histories, model_names):
        ax.plot(history['train_loss'], marker='o', label=name)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Validation Loss
    ax = axes[0, 1]
    for history, name in zip(histories, model_names):
        ax.plot(history['val_loss'], marker='s', label=name)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Training Accuracy
    ax = axes[1, 0]
    for history, name in zip(histories, model_names):
        ax.plot(history['train_acc'], marker='o', label=name)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Training Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Validation Accuracy
    ax = axes[1, 1]
    for history, name in zip(histories, model_names):
        ax.plot(history['val_acc'], marker='s', label=name)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Validation Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path / 'training_metrics.png', dpi=300, bbox_inches='tight')
    print(f"📊 Training metrics plot saved to {save_path / 'training_metrics.png'}")
    plt.close()


def plot_confusion_matrices(results_list, model_names, save_dir='plots'):
    """Plot confusion matrices for all models"""
    
    save_path = SCRIPT_DIR / save_dir
    os.makedirs(save_path, exist_ok=True)
    
    n_models = len(results_list)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
    
    if n_models == 1:
        axes = [axes]
    
    for idx, (results, name) in enumerate(zip(results_list, model_names)):
        cm = results['confusion_matrix']
        label_names = results['label_names']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=label_names, yticklabels=label_names,
                   ax=axes[idx])
        axes[idx].set_title(f'{name}\nAccuracy: {results["accuracy"]:.4f}')
        axes[idx].set_ylabel('True Label')
        axes[idx].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig(save_path / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
    print(f"📊 Confusion matrices saved to {save_path / 'confusion_matrices.png'}")
    plt.close()


def save_predictions(results_list, model_names, save_dir='predictions'):
    """Save predictions for each model"""
    
    save_path = SCRIPT_DIR / save_dir
    os.makedirs(save_path, exist_ok=True)
    
    for results, name in zip(results_list, model_names):
        predictions_data = []
        
        for claim, pred, true_label in zip(results['claims'], 
                                           results['predictions'], 
                                           results['true_labels']):
            predictions_data.append({
                'claim': claim,
                'predicted_label_idx': int(pred),
                'true_label_idx': int(true_label),
                'predicted_label': results['label_names'][pred],
                'true_label': results['label_names'][true_label]
            })
        
        filename = save_path / f'{name}_predictions.json'
        with open(filename, 'w') as f:
            json.dump(predictions_data, f, indent=2)
        
        print(f"💾 Predictions for {name} saved to {filename}")


def main():
    # Print current working directory for debugging
    print(f"\n{'='*70}")
    print("DIRECTORY INFORMATION")
    print(f"{'='*70}")
    print(f"Script directory: {SCRIPT_DIR}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"\nLooking for JSON files in: {SCRIPT_DIR}")
    
    # List files in the script directory
    json_files = list(SCRIPT_DIR.glob('*.json'))
    if json_files:
        print("\nFound JSON files:")
        for f in json_files:
            print(f"  - {f.name}")
    else:
        print("\n⚠️ Warning: No JSON files found in script directory!")
    
    # Configuration
    CONFIG = {
        'batch_size': 8,
        'learning_rate': 2e-5,
        'epochs': 10,
        'max_length': 512,
        'warmup_ratio': 0.1,
        'weight_decay': 0.01,
        'seed': 42,
        'num_evidences': 2  # k parameter per dataset (combined will use k from each)
    }
    
    print(f"\n{'='*70}")
    print("CONFIGURATION")
    print(f"{'='*70}")
    for key, value in CONFIG.items():
        print(f"{key}: {value}")
    
    # Set seed
    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    
    # ========================================================================
    # Load "our" dataset to get label mapping
    # ========================================================================
    print("\n" + "="*70)
    print("Loading 'Our' Dataset (Reference for labels)")
    print("="*70)
    
    our_train_dataset = ClaimVerificationDataset(
        'our_train.json', 
        tokenizer, 
        max_length=CONFIG['max_length'],
        num_evidences=CONFIG['num_evidences']
    )
    
    LABEL_MAP = our_train_dataset.label_map
    print(f"\n✅ Label mapping: {LABEL_MAP}")
    
    # ========================================================================
    # MODEL 1: Train on "Our" dataset only
    # ========================================================================
    print("\n" + "="*70)
    print("MODEL 1: Training on 'Our' Dataset Only")
    print("="*70)
    
    our_val_dataset = ClaimVerificationDataset(
        'our_val.json', 
        tokenizer, 
        max_length=CONFIG['max_length'],
        num_evidences=CONFIG['num_evidences']
    )
    
    our_train_loader = DataLoader(our_train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    our_val_loader = DataLoader(our_val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    
    model1 = RobertaForSequenceClassification.from_pretrained(
        'roberta-base',
        num_labels=len(LABEL_MAP)
    )
    
    model1_history, model1_best_f1, model1_best_epoch = train_model(
        model1, our_train_loader, our_val_loader, 
        device, CONFIG, "model1_our_only"
    )
    
    # ========================================================================
    # MODEL 2: Train on Merged dataset
    # ========================================================================
    print("\n" + "="*70)
    print("MODEL 2: Training on MERGED Dataset (Our + QuantEmp by Claim)")
    print("="*70)
    
    merged_train_dataset = MergedDataset(
        'our_train.json',
        'train_claimdecomp_evidence_question_mapping.json',
        tokenizer,
        max_length=CONFIG['max_length'],
        label_map=LABEL_MAP,
        num_evidences=CONFIG['num_evidences']
    )
    
    merged_val_dataset = MergedDataset(
        'our_val.json',
        'val_claimdecomp_evidence_question_mapping.json',
        tokenizer,
        max_length=CONFIG['max_length'],
        label_map=LABEL_MAP,
        num_evidences=CONFIG['num_evidences']
    )
    
    merged_train_loader = DataLoader(merged_train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    merged_val_loader = DataLoader(merged_val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    
    model2 = RobertaForSequenceClassification.from_pretrained(
        'roberta-base',
        num_labels=len(LABEL_MAP)
    )
    
    model2_history, model2_best_f1, model2_best_epoch = train_model(
        model2, merged_train_loader, merged_val_loader,
        device, CONFIG, "model2_merged"
    )
    
    # ========================================================================
    # MODEL 3: Train on QuantEmp only (Baseline)
    # ========================================================================
    print("\n" + "="*70)
    print("MODEL 3: Training on QuantEmp Dataset Only (Baseline)")
    print("="*70)
    
    quantemp_train_dataset = ClaimVerificationDataset(
        'train_claimdecomp_evidence_question_mapping.json', 
        tokenizer, 
        max_length=CONFIG['max_length'],
        num_evidences=CONFIG['num_evidences']
    )
    
    quantemp_val_dataset = ClaimVerificationDataset(
        'val_claimdecomp_evidence_question_mapping.json', 
        tokenizer, 
        max_length=CONFIG['max_length'],
        num_evidences=CONFIG['num_evidences']
    )
    
    quantemp_train_loader = DataLoader(quantemp_train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    quantemp_val_loader = DataLoader(quantemp_val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    
    model3 = RobertaForSequenceClassification.from_pretrained(
        'roberta-base',
        num_labels=len(LABEL_MAP)
    )
    
    model3_history, model3_best_f1, model3_best_epoch = train_model(
        model3, quantemp_train_loader, quantemp_val_loader,
        device, CONFIG, "model3_quantemp_only"
    )
    
    # ========================================================================
    # EVALUATION PHASE
    # ========================================================================
    print("\n" + "="*70)
    print("EVALUATION PHASE")
    print("="*70)
    
    # Load trained models
    model1_loaded = RobertaForSequenceClassification.from_pretrained(str(SCRIPT_DIR / 'models' / 'model1_our_only'))
    model2_loaded = RobertaForSequenceClassification.from_pretrained(str(SCRIPT_DIR / 'models' / 'model2_merged'))
    model3_loaded = RobertaForSequenceClassification.from_pretrained(str(SCRIPT_DIR / 'models' / 'model3_quantemp_only'))
    
    # Create test datasets
    print("\n--- Creating Test Datasets ---")
    
    # Test on "Our" dataset
    our_test_dataset = ClaimVerificationDataset(
        'our_test.json', 
        tokenizer, 
        max_length=CONFIG['max_length'],
        num_evidences=CONFIG['num_evidences']
    )
    our_test_loader = DataLoader(our_test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    
    # Test on Merged dataset
    merged_test_dataset = MergedDataset(
        'our_test.json',
        'test_claimdecomp_evidence_question_mapping.json',
        tokenizer,
        max_length=CONFIG['max_length'],
        label_map=LABEL_MAP,
        num_evidences=CONFIG['num_evidences']
    )
    merged_test_loader = DataLoader(merged_test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    
    # Test on QuantEmp only dataset
    quantemp_test_dataset = ClaimVerificationDataset(
        'test_claimdecomp_evidence_question_mapping.json', 
        tokenizer, 
        max_length=CONFIG['max_length'],
        num_evidences=CONFIG['num_evidences']
    )
    quantemp_test_loader = DataLoader(quantemp_test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    
    # Evaluate all models
    results = {}
    
    print("\n--- Model 1 (Our) on 'Our' Test Set ---")
    results['model1_on_our'] = evaluate_model(
        model1_loaded, our_test_loader, device, LABEL_MAP, "Model1_on_Our_Test"
    )
    
    print("\n--- Model 2 (Merged) on Merged Test Set ---")
    results['model2_on_merged'] = evaluate_model(
        model2_loaded, merged_test_loader, device, LABEL_MAP, "Model2_on_Merged_Test"
    )
    
    print("\n--- Model 3 (QuantEmp) on QuantEmp Test Set ---")
    results['model3_on_quantemp'] = evaluate_model(
        model3_loaded, quantemp_test_loader, device, LABEL_MAP, "Model3_on_QuantEmp_Test"
    )
    
    # ========================================================================
    # VISUALIZATION & SAVING
    # ========================================================================
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    # Plot training metrics
    plot_training_metrics(
        [model1_history, model2_history, model3_history],
        ['Model 1 (Our)', 'Model 2 (Merged)', 'Model 3 (QuantEmp)']
    )
    
    # Plot confusion matrices
    plot_confusion_matrices(
        [results['model1_on_our'], results['model2_on_merged'], results['model3_on_quantemp']],
        ['Model 1 (Our)', 'Model 2 (Merged)', 'Model 3 (QuantEmp)']
    )
    
    # Save predictions
    save_predictions(
        [results['model1_on_our'], results['model2_on_merged'], results['model3_on_quantemp']],
        ['model1_our', 'model2_merged', 'model3_quantemp']
    )
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "="*70)
    print("FINAL COMPARISON SUMMARY")
    print("="*70)
    
    summary = {
        'configuration': CONFIG,
        'label_mapping': LABEL_MAP,
        'models': {
            'model1_our_only': {
                'training': {
                    'best_val_f1': model1_best_f1,
                    'best_epoch': model1_best_epoch
                },
                'testing_on_our': {
                    'accuracy': results['model1_on_our']['accuracy'],
                    'f1_weighted': results['model1_on_our']['f1_weighted'],
                    'f1_macro': results['model1_on_our']['f1_macro']
                }
            },
            'model2_merged': {
                'training': {
                    'best_val_f1': model2_best_f1,
                    'best_epoch': model2_best_epoch
                },
                'testing_on_merged': {
                    'accuracy': results['model2_on_merged']['accuracy'],
                    'f1_weighted': results['model2_on_merged']['f1_weighted'],
                    'f1_macro': results['model2_on_merged']['f1_macro']
                }
            },
            'model3_quantemp_only': {
                'training': {
                    'best_val_f1': model3_best_f1,
                    'best_epoch': model3_best_epoch
                },
                'testing_on_quantemp': {
                    'accuracy': results['model3_on_quantemp']['accuracy'],
                    'f1_weighted': results['model3_on_quantemp']['f1_weighted'],
                    'f1_macro': results['model3_on_quantemp']['f1_macro']
                }
            }
        }
    }
    
    # Save summary
    summary_path = SCRIPT_DIR / 'final_comparison_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print comparison table
    print("\n📊 PERFORMANCE COMPARISON")
    print("="*90)
    print(f"{'Model':<25} {'Test Set':<20} {'Accuracy':<12} {'F1-Weighted':<12} {'F1-Macro':<12}")
    print("-"*90)
    
    print(f"{'Model 1 (Our Only)':<25} {'Our Test':<20} "
          f"{results['model1_on_our']['accuracy']:<12.4f} "
          f"{results['model1_on_our']['f1_weighted']:<12.4f} "
          f"{results['model1_on_our']['f1_macro']:<12.4f}")
    
    print(f"{'Model 2 (Merged)':<25} {'Merged Test':<20} "
          f"{results['model2_on_merged']['accuracy']:<12.4f} "
          f"{results['model2_on_merged']['f1_weighted']:<12.4f} "
          f"{results['model2_on_merged']['f1_macro']:<12.4f}")
    
    print(f"{'Model 3 (QuantEmp Only)':<25} {'QuantEmp Test':<20} "
          f"{results['model3_on_quantemp']['accuracy']:<12.4f} "
          f"{results['model3_on_quantemp']['f1_weighted']:<12.4f} "
          f"{results['model3_on_quantemp']['f1_macro']:<12.4f}")
    
    print("="*90)
    
    # Determine best model
    f1_scores = {
        'Model 1 (Our)': results['model1_on_our']['f1_weighted'],
        'Model 2 (Merged)': results['model2_on_merged']['f1_weighted'],
        'Model 3 (QuantEmp)': results['model3_on_quantemp']['f1_weighted']
    }
    
    best_model = max(f1_scores, key=f1_scores.get)
    print(f"\n🏆 Best Model: {best_model} with F1-Weighted = {f1_scores[best_model]:.4f}")
    
    print("\n✅ All processes completed!")
    print(f"📊 Summary saved to '{summary_path}'")
    print(f"📈 Plots saved to '{SCRIPT_DIR / 'plots'}' directory")
    print(f"💾 Predictions saved to '{SCRIPT_DIR / 'predictions'}' directory")
    print(f"🤖 Models saved to '{SCRIPT_DIR / 'models'}' directory")


if __name__ == "__main__":
    main()
