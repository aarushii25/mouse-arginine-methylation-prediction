# # Comprehensive protein language model evaluation for mouse arginine methylation
# # Models: LoRA-only, ProtBERT, Ankh, SaProt, ProteinBERT
# # Run this on a powerful GPU machine for best results

# import os
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score
# import json
# from datetime import datetime

# # ═══════════════════════════════════════
# #        DATA LOADING
# # ═══════════════════════════════════════

# def load_data():
#     asym_folder = "Asymmetric_dimethylarginine"
#     df_asym_neg = pd.read_csv(os.path.join(asym_folder, "mouse_negative_asymmetric_dimethylarginine_sequences.csv"))
#     df_asym_pos = pd.read_csv(os.path.join(asym_folder, "mouse_positive_asymmetric_dimethylarginine_sequences.csv"))
#     df_asym_neg["label"] = 0
#     df_asym_pos["label"] = 1

#     sym_folder = "Symmetric_dimethylarginine"
#     df_sym_neg = pd.read_csv(os.path.join(sym_folder, "mouse_negative_symmetric_dimethylarginine_sequences.csv"))
#     df_sym_pos = pd.read_csv(os.path.join(sym_folder, "mouse_positive_symmetric_dimethylarginine_sequences.csv"))
#     df_sym_neg["label"] = 0
#     df_sym_pos["label"] = 1

#     dimethyl_folder = "Dimethyl_arginine"
#     df_dimethyl_neg = pd.read_csv(os.path.join(dimethyl_folder, "mouse_negative_dimethylated_arginine_sequences.csv"))
#     df_dimethyl_pos = pd.read_csv(os.path.join(dimethyl_folder, "mouse_positive_dimethylated_arginine_sequences.csv"))
#     df_dimethyl_neg["label"] = 0
#     df_dimethyl_pos["label"] = 1

#     omega_folder = "Omega-N-methylarginine"
#     df_omega_neg = pd.read_csv(os.path.join(omega_folder, "mouse_negative_omega-n-methylarginine_sequences.csv"))
#     df_omega_pos = pd.read_csv(os.path.join(omega_folder, "mouse_positive_omega-n-methylarginine_sequences.csv"))
#     df_omega_neg["label"] = 0
#     df_omega_pos["label"] = 1

#     df = pd.concat([df_asym_neg, df_asym_pos, df_sym_neg, df_sym_pos,
#                     df_dimethyl_neg, df_dimethyl_pos, df_omega_neg, df_omega_pos],
#                     ignore_index=True)

#     df["sequence"] = df["Positive_sequence"].fillna(df["Negative_sequence"])
#     print(f"Dataset loaded — {len(df)} sequences total")
#     return df

# # ═══════════════════════════════════════
# #        DATASET CLASS
# # ═══════════════════════════════════════

# class ProteinDataset(Dataset):
#     def __init__(self, sequences, labels, tokenizer, max_length=15, split_chars=True):
#         self.sequences   = sequences
#         self.labels      = labels
#         self.tokenizer   = tokenizer
#         self.max_length  = max_length
#         self.split_chars = split_chars

#     def __len__(self):
#         return len(self.sequences)

#     def __getitem__(self, idx):
#         seq = str(self.sequences[idx]).upper()
#         if self.split_chars:
#             seq = " ".join(list(seq))

#         encoding = self.tokenizer(
#             seq,
#             max_length=self.max_length,
#             padding="max_length",
#             truncation=True,
#             return_tensors="pt"
#         )
#         return {
#             "input_ids"      : encoding["input_ids"].squeeze(),
#             "attention_mask" : encoding["attention_mask"].squeeze(),
#             "label"          : torch.tensor(self.labels[idx], dtype=torch.long)
#         }

# class AnkhDataset(Dataset):
#     def __init__(self, sequences, labels, tokenizer, max_length=15):
#         self.sequences  = sequences
#         self.labels     = labels
#         self.tokenizer  = tokenizer
#         self.max_length = max_length

#     def __len__(self):
#         return len(self.sequences)

#     def __getitem__(self, idx):
#         seq = list(str(self.sequences[idx]).upper())
#         encoding = self.tokenizer(
#             seq,
#             is_split_into_words=True,
#             max_length=self.max_length,
#             padding="max_length",
#             truncation=True,
#             return_tensors="pt"
#         )
#         return {
#             "input_ids"      : encoding["input_ids"].squeeze(),
#             "attention_mask" : encoding["attention_mask"].squeeze(),
#             "label"          : torch.tensor(self.labels[idx], dtype=torch.long)
#         }

# # ═══════════════════════════════════════
# #        CLASSIFIER HEAD
# # ═══════════════════════════════════════

# class ProteinClassifier(nn.Module):
#     def __init__(self, encoder, hidden_size):
#         super().__init__()
#         self.encoder    = encoder
#         self.dropout    = nn.Dropout(0.2)
#         self.bn         = nn.BatchNorm1d(hidden_size)
#         self.fc1        = nn.Linear(hidden_size, 256)
#         self.relu       = nn.ReLU()
#         self.fc2        = nn.Linear(256, 64)
#         self.fc3        = nn.Linear(64, 2)

#     def forward(self, input_ids, attention_mask):
#         outputs    = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
#         cls_output = outputs.last_hidden_state[:, 0, :]
#         cls_output = self.bn(cls_output)
#         cls_output = self.dropout(cls_output)
#         cls_output = self.relu(self.fc1(cls_output))
#         cls_output = self.relu(self.fc2(cls_output))
#         return self.fc3(cls_output)

# # ═══════════════════════════════════════
# #        TRAINING FUNCTION
# # ═══════════════════════════════════════

# def train_and_evaluate(model, train_loader, test_loader, model_name, epochs=20, lr=2e-5):
#     device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model     = model.to(device)

#     class_weights = torch.tensor([1.0, 2.76]).to(device)
#     criterion     = nn.CrossEntropyLoss(weight=class_weights)

#     optimizer = torch.optim.AdamW(
#         filter(lambda p: p.requires_grad, model.parameters()),
#         lr=lr,
#         weight_decay=0.01
#     )

#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#         optimizer, T_max=epochs, eta_min=1e-6
#     )

#     best_acc = 0.0
#     best_mcc = 0.0
#     best_auc = 0.0

#     print(f"\n{'='*50}")
#     print(f"Training: {model_name}")
#     print(f"Device  : {device}")
#     print(f"{'='*50}\n")

#     for epoch in range(epochs):
#         model.train()
#         total_loss = 0

#         for batch in train_loader:
#             input_ids      = batch["input_ids"].to(device)
#             attention_mask = batch["attention_mask"].to(device)
#             labels_batch   = batch["label"].to(device)

#             optimizer.zero_grad()
#             logits = model(input_ids, attention_mask)
#             loss   = criterion(logits, labels_batch)
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#             optimizer.step()
#             total_loss += loss.item()

#         scheduler.step()

#         model.eval()
#         all_preds  = []
#         all_probs  = []
#         all_labels = []

#         with torch.no_grad():
#             for batch in test_loader:
#                 input_ids      = batch["input_ids"].to(device)
#                 attention_mask = batch["attention_mask"].to(device)
#                 labels_batch   = batch["label"].to(device)

#                 logits = model(input_ids, attention_mask)
#                 probs  = torch.softmax(logits, dim=1)[:, 1]
#                 preds  = torch.argmax(logits, dim=1)

#                 all_preds.extend(preds.cpu().numpy())
#                 all_probs.extend(probs.cpu().numpy())
#                 all_labels.extend(labels_batch.cpu().numpy())

#         acc = accuracy_score(all_labels, all_preds)
#         mcc = matthews_corrcoef(all_labels, all_preds)
#         auc = roc_auc_score(all_labels, all_probs)

#         print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {total_loss:.4f} | Acc: {round(acc*100,2)}% | MCC: {round(mcc,4)} | AUC: {round(auc,4)}")

#         if acc > best_acc:
#             best_acc = acc
#             best_mcc = mcc
#             best_auc = auc
#             torch.save(model.state_dict(), f"best_{model_name.replace(' ', '_')}.pt")
#             print(f"           ✅ Best model saved!")

#     return best_acc, best_mcc, best_auc

# # ═══════════════════════════════════════
# #        RESULTS SAVE FUNCTION
# # ═══════════════════════════════════════

# def save_results(all_results):
#     now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     output = {"date_time": now, "results": all_results}

#     with open("protein_lm_results.json", "w") as f:
#         json.dump(output, f, indent=4)

#     print(f"\n{'='*50}")
#     print(f"ALL MODEL RESULTS")
#     print(f"{'='*50}")
#     for name, metrics in all_results.items():
#         print(f"{name:<20} | Acc: {metrics['Accuracy']}% | MCC: {metrics['MCC']} | AUC: {metrics['AUC']}")
#     print(f"{'='*50}")
#     print(f"\n✅ Results saved to protein_lm_results.json")

# # ═══════════════════════════════════════
# #        MAIN
# # ═══════════════════════════════════════

# df = load_data()
# sequences = df["sequence"].values
# labels    = df["label"].values

# X_train_seq, X_test_seq, y_train_seq, y_test_seq = train_test_split(
#     sequences, labels,
#     test_size=0.2,
#     random_state=42,
#     stratify=labels
# )

# all_results = {}

# # ═══════════════════════════════════════
# #   MODEL 1 — LoRA Only (ESM2 + LoRA)
# # ═══════════════════════════════════════

# print("\n" + "="*50)
# print("MODEL 1: LoRA Only")
# print("="*50)

# try:
#     from transformers import AutoTokenizer, AutoModel
#     from peft import LoraConfig, get_peft_model

#     tokenizer  = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
#     base_model = AutoModel.from_pretrained("facebook/esm2_t6_8M_UR50D")

#     lora_config = LoraConfig(
#         r=16,
#         lora_alpha=32,
#         target_modules=["query", "value", "key"],
#         lora_dropout=0.05,
#         bias="none"
#     )
#     lora_model = get_peft_model(base_model, lora_config)
#     lora_model.print_trainable_parameters()

#     model_lora = ProteinClassifier(lora_model, hidden_size=320)

#     train_ds = ProteinDataset(X_train_seq, y_train_seq, tokenizer)
#     test_ds  = ProteinDataset(X_test_seq,  y_test_seq,  tokenizer)
#     train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
#     test_dl  = DataLoader(test_ds,  batch_size=32, shuffle=False)

#     acc, mcc, auc = train_and_evaluate(
#         model_lora, train_dl, test_dl, "LoRA_Only", epochs=25, lr=1e-4
#     )
#     all_results["LoRA Only"] = {
#         "Accuracy": round(acc * 100, 2),
#         "MCC"     : round(mcc, 4),
#         "AUC"     : round(auc, 4)
#     }
#     print(f"\nLoRA Only → Acc: {round(acc*100,2)}% | MCC: {round(mcc,4)} | AUC: {round(auc,4)}")

# except Exception as e:
#     print(f"LoRA Only failed: {e}")
#     all_results["LoRA Only"] = {"error": str(e)}

# # ═══════════════════════════════════════
# #   MODEL 2 — ProtBERT
# # ═══════════════════════════════════════

# print("\n" + "="*50)
# print("MODEL 2: ProtBERT")
# print("="*50)

# try:
#     from transformers import BertTokenizer, BertModel

#     tokenizer_bert  = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
#     bert_base       = BertModel.from_pretrained("Rostlab/prot_bert")

#     # Freeze first 20 layers
#     for i, layer in enumerate(bert_base.encoder.layer):
#         if i < 20:
#             for param in layer.parameters():
#                 param.requires_grad = False

#     model_bert = ProteinClassifier(bert_base, hidden_size=1024)

#     train_ds = ProteinDataset(X_train_seq, y_train_seq, tokenizer_bert)
#     test_ds  = ProteinDataset(X_test_seq,  y_test_seq,  tokenizer_bert)
#     train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)
#     test_dl  = DataLoader(test_ds,  batch_size=16, shuffle=False)

#     acc, mcc, auc = train_and_evaluate(
#         model_bert, train_dl, test_dl, "ProtBERT", epochs=15, lr=2e-5
#     )
#     all_results["ProtBERT"] = {
#         "Accuracy": round(acc * 100, 2),
#         "MCC"     : round(mcc, 4),
#         "AUC"     : round(auc, 4)
#     }
#     print(f"\nProtBERT → Acc: {round(acc*100,2)}% | MCC: {round(mcc,4)} | AUC: {round(auc,4)}")

# except Exception as e:
#     print(f"ProtBERT failed: {e}")
#     all_results["ProtBERT"] = {"error": str(e)}

# # ═══════════════════════════════════════
# #   MODEL 3 — Ankh
# # ═══════════════════════════════════════

# print("\n" + "="*50)
# print("MODEL 3: Ankh")
# print("="*50)

# try:
#     import ankh

#     ankh_base, tokenizer_ankh = ankh.load_base_model()
#     ankh_base.eval()

#     # Freeze first 18 layers
#     for i, layer in enumerate(ankh_base.encoder.block):
#         if i < 18:
#             for param in layer.parameters():
#                 param.requires_grad = False

#     model_ankh = ProteinClassifier(ankh_base, hidden_size=768)

#     train_ds = AnkhDataset(X_train_seq, y_train_seq, tokenizer_ankh)
#     test_ds  = AnkhDataset(X_test_seq,  y_test_seq,  tokenizer_ankh)
#     train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)
#     test_dl  = DataLoader(test_ds,  batch_size=16, shuffle=False)

#     acc, mcc, auc = train_and_evaluate(
#         model_ankh, train_dl, test_dl, "Ankh", epochs=20, lr=5e-5
#     )
#     all_results["Ankh"] = {
#         "Accuracy": round(acc * 100, 2),
#         "MCC"     : round(mcc, 4),
#         "AUC"     : round(auc, 4)
#     }
#     print(f"\nAnkh → Acc: {round(acc*100,2)}% | MCC: {round(mcc,4)} | AUC: {round(auc,4)}")

# except Exception as e:
#     print(f"Ankh failed: {e}")
#     all_results["Ankh"] = {"error": str(e)}

# # ═══════════════════════════════════════
# #   MODEL 4 — SaProt
# # ═══════════════════════════════════════

# print("\n" + "="*50)
# print("MODEL 4: SaProt")
# print("="*50)

# try:
#     from transformers import EsmTokenizer, EsmModel

#     tokenizer_saprot = EsmTokenizer.from_pretrained("westlake-repl/SaProt_650M_AF2")
#     saprot_base      = EsmModel.from_pretrained("westlake-repl/SaProt_650M_AF2")

#     # Freeze first 28 layers — SaProt has 33 layers total
#     for i, layer in enumerate(saprot_base.encoder.layer):
#         if i < 28:
#             for param in layer.parameters():
#                 param.requires_grad = False

#     model_saprot = ProteinClassifier(saprot_base, hidden_size=1280)

#     train_ds = ProteinDataset(X_train_seq, y_train_seq, tokenizer_saprot)
#     test_ds  = ProteinDataset(X_test_seq,  y_test_seq,  tokenizer_saprot)
#     train_dl = DataLoader(train_ds, batch_size=8, shuffle=True)   # batch size kam — model bada hai
#     test_dl  = DataLoader(test_ds,  batch_size=8, shuffle=False)

#     acc, mcc, auc = train_and_evaluate(
#         model_saprot, train_dl, test_dl, "SaProt", epochs=15, lr=1e-5
#     )
#     all_results["SaProt"] = {
#         "Accuracy": round(acc * 100, 2),
#         "MCC"     : round(mcc, 4),
#         "AUC"     : round(auc, 4)
#     }
#     print(f"\nSaProt → Acc: {round(acc*100,2)}% | MCC: {round(mcc,4)} | AUC: {round(auc,4)}")

# except Exception as e:
#     print(f"SaProt failed: {e}")
#     all_results["SaProt"] = {"error": str(e)}

# # ═══════════════════════════════════════
# #   MODEL 5 — ProteinBERT
# # ═══════════════════════════════════════

# print("\n" + "="*50)
# print("MODEL 5: ProteinBERT")
# print("="*50)

# try:
#     os.system("pip install -q proteinbert")
#     from proteinbert import OutputType, OutputSpec, FinetuningModelGenerator, load_pretrained_model
#     from proteinbert.conv_and_global_attention_model import get_model_with_hidden_layers_as_outputs

#     pretrained_model_generator, input_encoder = load_pretrained_model()

#     model_generator = FinetuningModelGenerator(
#         pretrained_model_generator,
#         OutputSpec(OutputType.BINARY, [0, 1]),
#         pretraining_model_manipulator=get_model_with_hidden_layers_as_outputs,
#         dropout_rate=0.5
#     )

#     from proteinbert.finetuning import finetune, evaluate_by_len

#     train_df = pd.DataFrame({
#         "seq"   : X_train_seq,
#         "label" : y_train_seq
#     })
#     test_df = pd.DataFrame({
#         "seq"   : X_test_seq,
#         "label" : y_test_seq
#     })

#     # ProteinBERT has its own training API
#     training_callbacks = finetune(
#         model_generator,
#         input_encoder,
#         train_df,
#         seq_col="seq",
#         label_col="label",
#         n_epochs=20,
#         lr=1e-4,
#         batch_size=32
#     )

#     results_df, _ = evaluate_by_len(
#         model_generator,
#         input_encoder,
#         test_df,
#         seq_col="seq",
#         label_col="label"
#     )

#     acc = results_df["accuracy"].mean()
#     mcc = results_df["mcc"].mean()
#     auc = results_df["auc"].mean()

#     all_results["ProteinBERT"] = {
#         "Accuracy": round(acc * 100, 2),
#         "MCC"     : round(mcc, 4),
#         "AUC"     : round(auc, 4)
#     }
#     print(f"\nProteinBERT → Acc: {round(acc*100,2)}% | MCC: {round(mcc,4)} | AUC: {round(auc,4)}")

# except Exception as e:
#     print(f"ProteinBERT failed: {e}")
#     all_results["ProteinBERT"] = {"error": str(e)}

# # ═══════════════════════════════════════
# #        SAVE ALL RESULTS
# # ═══════════════════════════════════════

# save_results(all_results)


# Comprehensive protein language model evaluation for mouse arginine methylation
# Models: LoRA-only (ESM2), ProtBERT, SaProt, ProteinBERT
# Run this on a powerful GPU machine for best results

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score
import json
from datetime import datetime

# ═══════════════════════════════════════
#        DATA LOADING
# ═══════════════════════════════════════

def load_data():
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
    return df

# ═══════════════════════════════════════
#        DATASET CLASS
# ═══════════════════════════════════════

class ProteinDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer, max_length=15, split_chars=True):
        self.sequences   = sequences
        self.labels      = labels
        self.tokenizer   = tokenizer
        self.max_length  = max_length
        self.split_chars = split_chars

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = str(self.sequences[idx]).upper()
        if self.split_chars:
            seq = " ".join(list(seq))

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
#        CLASSIFIER HEAD
# ═══════════════════════════════════════

class ProteinClassifier(nn.Module):
    def __init__(self, encoder, hidden_size):
        super().__init__()
        self.encoder    = encoder
        self.dropout    = nn.Dropout(0.2)
        self.bn         = nn.BatchNorm1d(hidden_size)
        self.fc1        = nn.Linear(hidden_size, 256)
        self.relu       = nn.ReLU()
        self.fc2        = nn.Linear(256, 64)
        self.fc3        = nn.Linear(64, 2)

    def forward(self, input_ids, attention_mask):
        outputs    = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.bn(cls_output)
        cls_output = self.dropout(cls_output)
        cls_output = self.relu(self.fc1(cls_output))
        cls_output = self.relu(self.fc2(cls_output))
        return self.fc3(cls_output)

# ═══════════════════════════════════════
#        TRAINING FUNCTION
# ═══════════════════════════════════════

def train_and_evaluate(model, train_loader, test_loader, model_name, epochs=20, lr=2e-5):
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = model.to(device)

    class_weights = torch.tensor([1.0, 2.76]).to(device)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=0.01
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    best_acc = 0.0
    best_mcc = 0.0
    best_auc = 0.0

    print(f"\n{'='*50}")
    print(f"Training: {model_name}")
    print(f"Device  : {device}")
    print(f"{'='*50}\n")

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
            best_acc = acc
            best_mcc = mcc
            best_auc = auc
            torch.save(model.state_dict(), f"best_{model_name.replace(' ', '_')}.pt")
            print(f"           ✅ Best model saved!")

    return best_acc, best_mcc, best_auc

# ═══════════════════════════════════════
#        RESULTS SAVE & COMPARE FUNCTION
# ═══════════════════════════════════════

def save_and_compare_results(all_results):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    output = {"date_time": now, "results": all_results}

    with open("protein_lm_results.json", "w") as f:
        json.dump(output, f, indent=4)

    print(f"\n{'='*60}")
    print(f"ALL MODEL RESULTS")
    print(f"{'='*60}")
    
    valid_results = {}
    for name, metrics in all_results.items():
        if "error" in metrics:
            print(f"{name:<15} | ERROR: {metrics['error']}")
        else:
            print(f"{name:<15} | Acc: {metrics['Accuracy']:>6.2f}% | MCC: {metrics['MCC']:>6.4f} | AUC: {metrics['AUC']:>6.4f}")
            valid_results[name] = metrics
            
    print(f"{'='*60}")
    
    # Best Model Comparison
    if valid_results:
        best_acc_model = max(valid_results, key=lambda k: valid_results[k]["Accuracy"])
        best_mcc_model = max(valid_results, key=lambda k: valid_results[k]["MCC"])
        best_auc_model = max(valid_results, key=lambda k: valid_results[k]["AUC"])
        
        print(f"\n🏆 BEST PERFORMING MODELS 🏆")
        print(f"Highest Accuracy : {best_acc_model} ({valid_results[best_acc_model]['Accuracy']}%)")
        print(f"Highest MCC      : {best_mcc_model} ({valid_results[best_mcc_model]['MCC']})")
        print(f"Highest AUC      : {best_auc_model} ({valid_results[best_auc_model]['AUC']})")
        print(f"{'='*60}")
        
    print(f"\n✅ Results saved to protein_lm_results.json")

# ═══════════════════════════════════════
#        MAIN
# ═══════════════════════════════════════

if __name__ == "__main__":
    df = load_data()
    sequences = df["sequence"].values
    labels    = df["label"].values

    X_train_seq, X_test_seq, y_train_seq, y_test_seq = train_test_split(
        sequences, labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    all_results = {}

    # ═══════════════════════════════════════
    #  MODEL 1 — LoRA Only (ESM2 + LoRA)
    # ═══════════════════════════════════════
    print("\n" + "="*50)
    print("MODEL 1: LoRA Only")
    print("="*50)

    try:
        from transformers import AutoTokenizer, AutoModel
        from peft import LoraConfig, get_peft_model

        tokenizer  = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
        base_model = AutoModel.from_pretrained("facebook/esm2_t6_8M_UR50D")

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["query", "value", "key"],
            lora_dropout=0.05,
            bias="none"
        )
        lora_model = get_peft_model(base_model, lora_config)
        lora_model.print_trainable_parameters()

        model_lora = ProteinClassifier(lora_model, hidden_size=320)

        train_ds = ProteinDataset(X_train_seq, y_train_seq, tokenizer)
        test_ds  = ProteinDataset(X_test_seq,  y_test_seq,  tokenizer)
        train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
        test_dl  = DataLoader(test_ds,  batch_size=32, shuffle=False)

        acc, mcc, auc = train_and_evaluate(
            model_lora, train_dl, test_dl, "LoRA_Only", epochs=25, lr=1e-4
        )
        all_results["LoRA Only"] = {
            "Accuracy": round(acc * 100, 2),
            "MCC"     : round(mcc, 4),
            "AUC"     : round(auc, 4)
        }
        print(f"\nLoRA Only → Acc: {round(acc*100,2)}% | MCC: {round(mcc,4)} | AUC: {round(auc,4)}")

    except Exception as e:
        print(f"LoRA Only failed: {e}")
        all_results["LoRA Only"] = {"error": str(e)}

    # ═══════════════════════════════════════
    #  MODEL 2 — ProtBERT
    # ═══════════════════════════════════════
    print("\n" + "="*50)
    print("MODEL 2: ProtBERT")
    print("="*50)

    try:
        from transformers import BertTokenizer, BertModel

        tokenizer_bert  = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
        bert_base       = BertModel.from_pretrained("Rostlab/prot_bert")

        # Freeze first 20 layers
        for i, layer in enumerate(bert_base.encoder.layer):
            if i < 20:
                for param in layer.parameters():
                    param.requires_grad = False

        model_bert = ProteinClassifier(bert_base, hidden_size=1024)

        train_ds = ProteinDataset(X_train_seq, y_train_seq, tokenizer_bert)
        test_ds  = ProteinDataset(X_test_seq,  y_test_seq,  tokenizer_bert)
        train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)
        test_dl  = DataLoader(test_ds,  batch_size=16, shuffle=False)

        acc, mcc, auc = train_and_evaluate(
            model_bert, train_dl, test_dl, "ProtBERT", epochs=15, lr=2e-5
        )
        all_results["ProtBERT"] = {
            "Accuracy": round(acc * 100, 2),
            "MCC"     : round(mcc, 4),
            "AUC"     : round(auc, 4)
        }
        print(f"\nProtBERT → Acc: {round(acc*100,2)}% | MCC: {round(mcc,4)} | AUC: {round(auc,4)}")

    except Exception as e:
        print(f"ProtBERT failed: {e}")
        all_results["ProtBERT"] = {"error": str(e)}

    # ═══════════════════════════════════════
    #  MODEL 3 — SaProt
    # ═══════════════════════════════════════
    print("\n" + "="*50)
    print("MODEL 3: SaProt")
    print("="*50)

    try:
        from transformers import EsmTokenizer, EsmModel

        tokenizer_saprot = EsmTokenizer.from_pretrained("westlake-repl/SaProt_650M_AF2")
        saprot_base      = EsmModel.from_pretrained("westlake-repl/SaProt_650M_AF2")

        # Freeze first 28 layers — SaProt has 33 layers total
        for i, layer in enumerate(saprot_base.encoder.layer):
            if i < 28:
                for param in layer.parameters():
                    param.requires_grad = False

        model_saprot = ProteinClassifier(saprot_base, hidden_size=1280)

        train_ds = ProteinDataset(X_train_seq, y_train_seq, tokenizer_saprot)
        test_ds  = ProteinDataset(X_test_seq,  y_test_seq,  tokenizer_saprot)
        train_dl = DataLoader(train_ds, batch_size=8, shuffle=True)   # batch size kam — model bada hai
        test_dl  = DataLoader(test_ds,  batch_size=8, shuffle=False)

        acc, mcc, auc = train_and_evaluate(
            model_saprot, train_dl, test_dl, "SaProt", epochs=15, lr=1e-5
        )
        all_results["SaProt"] = {
            "Accuracy": round(acc * 100, 2),
            "MCC"     : round(mcc, 4),
            "AUC"     : round(auc, 4)
        }
        print(f"\nSaProt → Acc: {round(acc*100,2)}% | MCC: {round(mcc,4)} | AUC: {round(auc,4)}")

    except Exception as e:
        print(f"SaProt failed: {e}")
        all_results["SaProt"] = {"error": str(e)}

    # # ═══════════════════════════════════════
    # #  MODEL 4 — ProteinBERT
    # # ═══════════════════════════════════════
    # print("\n" + "="*50)
    # print("MODEL 4: ProteinBERT")
    # print("="*50)

    # try:
    #     os.system("pip install -q proteinbert")
    #     from proteinbert import OutputType, OutputSpec, FinetuningModelGenerator, load_pretrained_model
    #     from proteinbert.conv_and_global_attention_model import get_model_with_hidden_layers_as_outputs

    #     pretrained_model_generator, input_encoder = load_pretrained_model()

    #     model_generator = FinetuningModelGenerator(
    #         pretrained_model_generator,
    #         OutputSpec(OutputType.BINARY, [0, 1]),
    #         pretraining_model_manipulator=get_model_with_hidden_layers_as_outputs,
    #         dropout_rate=0.5
    #     )

    #     from proteinbert.finetuning import finetune, evaluate_by_len

    #     train_df = pd.DataFrame({
    #         "seq"   : X_train_seq,
    #         "label" : y_train_seq
    #     })
    #     test_df = pd.DataFrame({
    #         "seq"   : X_test_seq,
    #         "label" : y_test_seq
    #     })

    #     # ProteinBERT has its own training API
    #     training_callbacks = finetune(
    #         model_generator,
    #         input_encoder,
    #         train_df,
    #         seq_col="seq",
    #         label_col="label",
    #         n_epochs=20,
    #         lr=1e-4,
    #         batch_size=32
    #     )

    #     results_df, _ = evaluate_by_len(
    #         model_generator,
    #         input_encoder,
    #         test_df,
    #         seq_col="seq",
    #         label_col="label"
    #     )

    #     acc = results_df["accuracy"].mean()
    #     mcc = results_df["mcc"].mean()
    #     auc = results_df["auc"].mean()

    #     all_results["ProteinBERT"] = {
    #         "Accuracy": round(acc * 100, 2),
    #         "MCC"     : round(mcc, 4),
    #         "AUC"     : round(auc, 4)
    #     }
    #     print(f"\nProteinBERT → Acc: {round(acc*100,2)}% | MCC: {round(mcc,4)} | AUC: {round(auc,4)}")

    # except Exception as e:
    #     print(f"ProteinBERT failed: {e}")
    #     all_results["ProteinBERT"] = {"error": str(e)}

    # ═══════════════════════════════════════
    #        SAVE ALL RESULTS & COMPARE
    # ═══════════════════════════════════════
    save_and_compare_results(all_results)