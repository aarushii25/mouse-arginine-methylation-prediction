import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score
from imblearn.over_sampling import SMOTE

# ═══════════════════════════════════════
#        DATA LOADING
# ═══════════════════════════════════════

asym_folder = "Asymmetric_dimethylarginine"
asym_neg_path = os.path.join(asym_folder, "mouse_negative_asymmetric_dimethylarginine_sequences.csv")
asym_pos_path = os.path.join(asym_folder, "mouse_positive_asymmetric_dimethylarginine_sequences.csv")
df_asym_neg = pd.read_csv(asym_neg_path)
df_asym_pos = pd.read_csv(asym_pos_path)
df_asym_neg["label"] = 0
df_asym_pos["label"] = 1

sym_folder = "Symmetric_dimethylarginine"
sym_neg_path = os.path.join(sym_folder, "mouse_negative_symmetric_dimethylarginine_sequences.csv")
sym_pos_path = os.path.join(sym_folder, "mouse_positive_symmetric_dimethylarginine_sequences.csv")
df_sym_neg = pd.read_csv(sym_neg_path)
df_sym_pos = pd.read_csv(sym_pos_path)
df_sym_neg["label"] = 0
df_sym_pos["label"] = 1

dimethyl_folder = "Dimethyl_arginine"
dimethyl_neg_path = os.path.join(dimethyl_folder, "mouse_negative_dimethylated_arginine_sequences.csv")
dimethyl_pos_path = os.path.join(dimethyl_folder, "mouse_positive_dimethylated_arginine_sequences.csv")
df_dimethyl_neg = pd.read_csv(dimethyl_neg_path)
df_dimethyl_pos = pd.read_csv(dimethyl_pos_path)
df_dimethyl_neg["label"] = 0
df_dimethyl_pos["label"] = 1

omega_folder = "Omega-N-methylarginine"
omega_neg_path = os.path.join(omega_folder, "mouse_negative_omega-n-methylarginine_sequences.csv")
omega_pos_path = os.path.join(omega_folder, "mouse_positive_omega-n-methylarginine_sequences.csv")
df_omega_neg = pd.read_csv(omega_neg_path)
df_omega_pos = pd.read_csv(omega_pos_path)
df_omega_neg["label"] = 0
df_omega_pos["label"] = 1

df = pd.concat([df_asym_neg, df_asym_pos, df_sym_neg, df_sym_pos,
                df_dimethyl_neg, df_dimethyl_pos, df_omega_neg, df_omega_pos],
                ignore_index=True)

df["sequence"] = df["Positive_sequence"].fillna(df["Negative_sequence"])
print(f"Total sequences: {len(df)}")

# ═══════════════════════════════════════
#        DATASET CLASS
# ═══════════════════════════════════════

class ProteinDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer, max_length=15):
        self.sequences = sequences
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = " ".join(list(str(self.sequences[idx]).upper()))
        encoding = self.tokenizer(
            seq,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids"      : encoding["input_ids"].squeeze(),
            "attention_mask" : encoding["attention_mask"].squeeze(),
            "label"          : torch.tensor(self.labels[idx], dtype=torch.long)
        }

# ═══════════════════════════════════════
#   ESM2 + LoRA CLASSIFIER
# ═══════════════════════════════════════

class ESMLoRAClassifier(nn.Module):
    def __init__(self, esm_model, hidden_size=320):
        super().__init__()
        self.esm    = esm_model
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_size, 2)  # 2 classes

    def forward(self, input_ids, attention_mask):
        outputs = self.esm(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # CLS token ki embedding use karo
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        return logits

# ═══════════════════════════════════════
#        MAIN TRAINING
# ═══════════════════════════════════════

print("\nLoading ESM2 tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
base_model = AutoModel.from_pretrained("facebook/esm2_t6_8M_UR50D")

# LoRA Config
lora_config = LoraConfig(
    r=8,                    # LoRA rank — chhota = less params
    lora_alpha=16,          # scaling factor
    target_modules=["query", "value"],  # attention layers pe LoRA
    lora_dropout=0.1,
    bias="none"
)

# LoRA apply karo
esm_lora = get_peft_model(base_model, lora_config)
esm_lora.print_trainable_parameters()  # kitne params train honge

# Full classifier
model_lora = ESMLoRAClassifier(esm_lora, hidden_size=320)
print("✅ ESM2 + LoRA model ready!")

# ═══════════════════════════════════════
#        TRAIN TEST SPLIT
# ═══════════════════════════════════════

sequences = df["sequence"].values
labels    = df["label"].values

X_train_seq, X_test_seq, y_train_seq, y_test_seq = train_test_split(
    sequences, labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

print(f"Train: {len(X_train_seq)} | Test: {len(X_test_seq)}")

# Dataset aur DataLoader
train_dataset = ProteinDataset(X_train_seq, y_train_seq, tokenizer)
test_dataset  = ProteinDataset(X_test_seq,  y_test_seq,  tokenizer)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)

# ═══════════════════════════════════════
#        TRAINING LOOP
# ═══════════════════════════════════════

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model_lora = model_lora.to(device)
optimizer  = torch.optim.AdamW(model_lora.parameters(), lr=2e-4)
criterion  = nn.CrossEntropyLoss()

epochs = 20
best_acc = 0

print("\nTraining ESM2 + LoRA...")
for epoch in range(epochs):
    model_lora.train()
    total_loss = 0

    for batch in train_loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels_batch   = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model_lora(input_ids, attention_mask)
        loss   = criterion(logits, labels_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Validation
    model_lora.eval()
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_batch   = batch["label"].to(device)

            logits = model_lora(input_ids, attention_mask)
            probs  = torch.softmax(logits, dim=1)[:, 1]
            preds  = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    mcc = matthews_corrcoef(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)

    print(f"Epoch {epoch+1}/{epochs} → Loss: {total_loss:.4f} | Acc: {round(acc*100,2)}% | MCC: {round(mcc,4)} | AUC: {round(auc,4)}")

    # Best model save karo
    if acc > best_acc:
        best_acc = acc
        torch.save(model_lora.state_dict(), "best_esm_lora.pt")
        print(f"  ✅ Best model saved! Acc: {round(best_acc*100,2)}%")

# ═══════════════════════════════════════
#        FINAL RESULTS
# ═══════════════════════════════════════

print(f"\n{'='*50}")
print(f"FINAL ESM2 + LoRA RESULTS:")
print(f"Best Accuracy : {round(best_acc * 100, 2)}%")
print(f"MCC           : {round(mcc, 4)}")
print(f"AUC           : {round(auc, 4)}")
print(f"ESM2 + LoRA     : {round(best_acc*100,2)}%")