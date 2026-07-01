# Protein sequence methylation prediction using ProtBERT
# A transformer-based approach fine-tuned on mouse arginine methylation data

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score

# Loading the dataset
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
print(f"Dataset loaded — {len(df)} sequences total")

# Preparing the data for ProtBERT
# ProtBERT expects space-separated amino acids
class ProteinDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer, max_length=15):
        self.sequences  = sequences
        self.labels     = labels
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # ProtBERT needs spaces between each amino acid
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

# Building the classifier on top of ProtBERT
# Using the CLS token representation for classification
class ProtBERTClassifier(nn.Module):
    def __init__(self, bert_model, hidden_size=1024):
        super().__init__()
        self.bert       = bert_model
        self.dropout    = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        outputs    = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        return self.classifier(cls_output)

# Loading ProtBERT from HuggingFace
# This might take a few minutes on first run as it downloads the model
print("\nLoading ProtBERT model...")
tokenizer  = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
bert_model = BertModel.from_pretrained("Rostlab/prot_bert")

# Freezing the first 20 layers — only fine-tuning the last few layers
# This speeds up training significantly while still getting good results
for i, layer in enumerate(bert_model.encoder.layer):
    if i < 20:
        for param in layer.parameters():
            param.requires_grad = False

model = ProtBERTClassifier(bert_model, hidden_size=1024)
print("ProtBERT loaded successfully!")

total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters     : {total_params:,}")
print(f"Trainable parameters : {trainable_params:,}")

# Splitting data into train and test sets
sequences = df["sequence"].values
labels    = df["label"].values

X_train_seq, X_test_seq, y_train_seq, y_test_seq = train_test_split(
    sequences, labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

print(f"\nTrain set : {len(X_train_seq)} sequences")
print(f"Test set  : {len(X_test_seq)} sequences")

train_dataset = ProteinDataset(X_train_seq, y_train_seq, tokenizer)
test_dataset  = ProteinDataset(X_test_seq,  y_test_seq,  tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=16, shuffle=False)

# Setting up training
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nRunning on: {device}")

model     = model.to(device)
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=2e-5,
    weight_decay=0.01
)
criterion = nn.CrossEntropyLoss()

# Training loop
epochs   = 15
best_acc = 0.0

print("\nStarting training...\n")
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
        optimizer.step()
        total_loss += loss.item()

    # Evaluating on test set after each epoch
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

    print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {total_loss:.4f} | Acc: {round(acc*100,2)}% | MCC: {round(mcc,4)} | AUC: {round(auc,4)}")

    if acc > best_acc:
        best_acc     = acc
        best_mcc     = mcc
        best_auc     = auc
        torch.save(model.state_dict(), "best_protbert.pt")


# Final results
print(f"\n{'='*45}")
print(f"ProtBERT Fine-tuning — Final Results")
print(f"{'='*45}")
print(f"Accuracy : {round(best_acc * 100, 2)}%")
print(f"MCC      : {round(best_mcc, 4)}")
print(f"AUC      : {round(best_auc, 4)}")
print(f"{'='*45}")