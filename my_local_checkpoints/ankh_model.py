# Methylation prediction using Ankh protein language model
# Ankh is one of the latest and most efficient protein LLMs (2023)
# Designed specifically for protein understanding tasks

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score

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

# Installing ankh if not already installed
try:
    import ankh
    print("Ankh already installed!")
except ImportError:
    print("Installing ankh...")
    os.system("pip install ankh")
    import ankh

# Dataset class for Ankh
# Ankh uses a different tokenization than ESM2
class AnkhDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer, max_length=15):
        self.sequences  = sequences
        self.labels     = labels
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = list(str(self.sequences[idx]).upper())

        encoding = self.tokenizer(
            seq,
            is_split_into_words=True,
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

# Ankh classifier
# Ankh base has hidden size of 768
class AnkhClassifier(nn.Module):
    def __init__(self, ankh_model, hidden_size=768):
        super().__init__()
        self.encoder    = ankh_model
        self.dropout    = nn.Dropout(0.2)
        self.bn         = nn.BatchNorm1d(hidden_size)
        self.fc1        = nn.Linear(hidden_size, 256)
        self.relu       = nn.ReLU()
        self.fc2        = nn.Linear(256, 64)
        self.fc3        = nn.Linear(64, 2)

    def forward(self, input_ids, attention_mask):
        outputs    = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.bn(cls_output)
        cls_output = self.dropout(cls_output)
        cls_output = self.relu(self.fc1(cls_output))
        cls_output = self.relu(self.fc2(cls_output))
        return self.fc3(cls_output)

# Loading Ankh base model
print("\nLoading Ankh base model...")
ankh_model, tokenizer = ankh.load_base_model()
ankh_model.eval()
print("✅ Ankh base model loaded!")

# Freezing first 18 layers — only training last 6 layers
# Ankh base has 24 transformer layers total
for i, layer in enumerate(ankh_model.encoder.block):
    if i < 18:
        for param in layer.parameters():
            param.requires_grad = False

model = AnkhClassifier(ankh_model, hidden_size=768)

total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters     : {total_params:,}")
print(f"Trainable parameters : {trainable_params:,}")

# Train test split
sequences = df["sequence"].values
labels    = df["label"].values

X_train_seq, X_test_seq, y_train_seq, y_test_seq = train_test_split(
    sequences, labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

print(f"\nTrain : {len(X_train_seq)} sequences")
print(f"Test  : {len(X_test_seq)} sequences")

train_dataset = AnkhDataset(X_train_seq, y_train_seq, tokenizer)
test_dataset  = AnkhDataset(X_test_seq,  y_test_seq,  tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=16, shuffle=False)

# Training setup
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nRunning on: {device}")

model     = model.to(device)

# Weighted loss for class imbalance
class_weights = torch.tensor([1.0, 2.76]).to(device)
criterion     = nn.CrossEntropyLoss(weight=class_weights)

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=5e-5,
    weight_decay=0.01
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=20,    # cosine cycle length
    eta_min=1e-6 # minimum learning rate
)

# Training loop
epochs   = 20
best_acc = 0.0
best_mcc = 0.0
best_auc = 0.0

print("\nStarting Ankh training...\n")
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

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    scheduler.step()

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

    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {total_loss:.4f} | Acc: {round(acc*100,2)}% | MCC: {round(mcc,4)} | AUC: {round(auc,4)} | LR: {current_lr:.2e}")

    if acc > best_acc:
        best_acc = acc
        best_mcc = mcc
        best_auc = auc
        torch.save(model.state_dict(), "best_ankh.pt")

# Final results

print(f"Accuracy : {round(best_acc * 100, 2)}%")
print(f"MCC      : {round(best_mcc, 4)}")
print(f"AUC      : {round(best_auc, 4)}")
