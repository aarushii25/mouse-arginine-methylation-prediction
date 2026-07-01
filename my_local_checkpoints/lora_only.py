# Fine-tuning with LoRA adapters on a lightweight base model
# Testing if LoRA alone without ESM2 can improve methylation prediction

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModel

# Loading dataset
asym_folder = "Asymmetric_dimethylarginine"
df_asym_neg = pd.read_csv(os.path.join(asym_folder, "mouse_negative_asymmetric_dimethylarginine_sequences.csv"))
df_asym_pos = pd.read_csv(os.path.join(asym_folder, "mouse_positive_asymmetric_dimethylarginine_sequences.csv"))
df_asym_neg["label"] = 0
df_asym_pos["label"] = 1

sym_folder = "Symmetric_dimethylarginine"
df_sym_neg = pd.read_csv(os.path.join(sym_folder, "mouse_negative_symmetric_dimethylarginine_sequences.csv"))
df_sym_pos = pd.read_csv(os.path.join(sym_folder, "mouse_positive_symmetric_dimethylarginine_sequences.csv"))
df_sym_neg["label"] = 0
df_sym_pos["label"] = 1

dimethyl_folder = "Dimethyl_arginine"
df_dimethyl_neg = pd.read_csv(os.path.join(dimethyl_folder, "mouse_negative_dimethylated_arginine_sequences.csv"))
df_dimethyl_pos = pd.read_csv(os.path.join(dimethyl_folder, "mouse_positive_dimethylated_arginine_sequences.csv"))
df_dimethyl_neg["label"] = 0
df_dimethyl_pos["label"] = 1

omega_folder = "Omega-N-methylarginine"
df_omega_neg = pd.read_csv(os.path.join(omega_folder, "mouse_negative_omega-n-methylarginine_sequences.csv"))
df_omega_pos = pd.read_csv(os.path.join(omega_folder, "mouse_positive_omega-n-methylarginine_sequences.csv"))
df_omega_neg["label"] = 0
df_omega_pos["label"] = 1

df = pd.concat([df_asym_neg, df_asym_pos, df_sym_neg, df_sym_pos,
                df_dimethyl_neg, df_dimethyl_pos, df_omega_neg, df_omega_pos],
                ignore_index=True)

df["sequence"] = df["Positive_sequence"].fillna(df["Negative_sequence"])
print(f"Dataset loaded — {len(df)} sequences")

# Dataset class
class ProteinDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer, max_length=15):
        self.sequences  = sequences
        self.labels     = labels
        self.tokenizer  = tokenizer
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

# LoRA only classifier — using ESM2 smallest variant
# Higher LoRA rank this time for more expressive power
class LoRAClassifier(nn.Module):
    def __init__(self, base_model, hidden_size=320):
        super().__init__()
        self.encoder    = base_model
        self.dropout    = nn.Dropout(0.2)
        self.bn         = nn.BatchNorm1d(hidden_size)  # batch norm for stability
        self.fc1        = nn.Linear(hidden_size, 128)
        self.relu       = nn.ReLU()
        self.fc2        = nn.Linear(128, 2)

    def forward(self, input_ids, attention_mask):
        outputs    = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.bn(cls_output)
        cls_output = self.dropout(cls_output)
        cls_output = self.relu(self.fc1(cls_output))
        return self.fc2(cls_output)

# Loading base model
print("\nLoading ESM2 base model...")
tokenizer  = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
base_model = AutoModel.from_pretrained("facebook/esm2_t6_8M_UR50D")

# LoRA config — higher rank than before for better expressiveness
lora_config = LoraConfig(
    r=16,                              # rank badha diya (8 → 16)
    lora_alpha=32,                     # scaling factor bhi badha
    target_modules=["query", "value", "key"],  # key bhi add kiya
    lora_dropout=0.05,                 # dropout thoda kam
    bias="none"
)

lora_model = get_peft_model(base_model, lora_config)
lora_model.print_trainable_parameters()

model = LoRAClassifier(lora_model, hidden_size=320)
print("✅ LoRA model ready!")

# Train test split
sequences = df["sequence"].values
labels    = df["label"].values

X_train_seq, X_test_seq, y_train_seq, y_test_seq = train_test_split(
    sequences, labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

train_dataset = ProteinDataset(X_train_seq, y_train_seq, tokenizer)
test_dataset  = ProteinDataset(X_test_seq,  y_test_seq,  tokenizer)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)

# Training setup
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nRunning on: {device}")

model     = model.to(device)

# Using class weights for imbalanced data
class_weights = torch.tensor([1.0, 2.76]).to(device)  # 4645/1681 ≈ 2.76
criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4,
    weight_decay=0.01
)

# Learning rate scheduler — reduces lr when plateau
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', patience=3, factor=0.5
)

# Training
epochs   = 25
best_acc = 0.0
best_mcc = 0.0
best_auc = 0.0

print("\nStarting LoRA training...\n")
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch in train_loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels_batch   = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss   = criterion(logits, labels_batch)
        loss.backward()

        # Gradient clipping — training ko stable rakhta hai
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item()

    # Evaluation
    model.eval()
    all_preds  = []
    all_probs  = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_batch   = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            probs  = torch.softmax(logits, dim=1)[:, 1]
            preds  = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    mcc = matthews_corrcoef(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)

    # Scheduler step
    scheduler.step(acc)

    print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {total_loss:.4f} | Acc: {round(acc*100,2)}% | MCC: {round(mcc,4)} | AUC: {round(auc,4)}")

    if acc > best_acc:
        best_acc = acc
        best_mcc = mcc
        best_auc = auc
        torch.save(model.state_dict(), "best_lora_only.pt")


# Final results
print(f"\n{'='*45}")

print(f"Accuracy : {round(best_acc * 100, 2)}%")
print(f"MCC      : {round(best_mcc, 4)}")
print(f"AUC      : {round(best_auc, 4)}")
