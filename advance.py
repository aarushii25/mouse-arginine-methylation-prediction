# Advanced Methods - Stacking + Ensemble+FeatureSelection
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ═══════════════════════════════════════
#        DATA LOADING (same as main.py)
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
#        FEATURE EXTRACTION (same)
# ═══════════════════════════════════════

amino_acids = list("ACDEFGHIKLMNPQRSTVWY")

def get_aac(sequence):
    sequence = str(sequence).upper()
    length = len(sequence)
    aac = []
    for aa in amino_acids:
        count = sequence.count(aa)
        aac.append(count / length if length > 0 else 0)
    return aac

aac_features = df["sequence"].apply(get_aac)
X_aac = np.array(aac_features.tolist())

dipeptides = [a + b for a in amino_acids for b in amino_acids]

def get_dpc(sequence):
    sequence = str(sequence).upper()
    length = len(sequence)
    n = length - 1
    dpc = []
    for dp in dipeptides:
        count = 0
        for i in range(n):
            if sequence[i:i+2] == dp:
                count += 1
        dpc.append(count / n if n > 0 else 0)
    return dpc

dpc_features = df["sequence"].apply(get_dpc)
X_dpc = np.array(dpc_features.tolist())

def get_entropy_features(sequence):
    sequence = str(sequence).upper()
    length = len(sequence)
    if length == 0:
        return [0, 0, 0, 0]
    probs = []
    for aa in amino_acids:
        p = sequence.count(aa) / length
        if p > 0:
            probs.append(p)
    probs = np.array(probs)
    SE = -np.sum(probs * np.log2(probs))
    alpha = 2
    RE = (1 / (1 - alpha)) * np.log2(np.sum(probs ** alpha))
    AE = (alpha / (alpha - 1)) * (1 - np.sum(probs ** alpha) ** (1/alpha))
    HE = (1 / (1 - alpha)) * (np.sum(probs ** alpha) - 1)
    return [SE, RE, AE, HE]

entropy_features = df["sequence"].apply(get_entropy_features)
X_entropy = np.array(entropy_features.tolist())

physicochemical = {
    'A':  [ 1.800,  0.000,  0.000,  89.09,  6.00, 1.181, 0.843, 0.360, 0.250,  0.000],
    'C':  [ 2.500,  0.000,  0.000, 121.16,  5.07, 1.461, 0.851, 0.350, 0.208,  8.330],
    'D':  [-3.500, -1.000,  1.000, 133.10,  2.77, 1.587, 0.897, 0.510, 0.208,  3.650],
    'E':  [-3.500, -1.000,  1.000, 147.13,  3.22, 1.862, 0.928, 0.500, 0.250,  4.250],
    'F':  [ 2.800,  0.000,  0.000, 165.19,  5.48, 2.228, 0.881, 0.310, 0.208,  0.000],
    'G':  [-0.400,  0.000,  0.000,  75.03,  5.97, 0.881, 0.811, 0.540, 0.208,  0.000],
    'H':  [-3.200,  0.000,  1.000, 155.16,  7.59, 2.025, 0.944, 0.320, 0.208,  6.000],
    'I':  [ 4.500,  0.000,  0.000, 131.17,  6.02, 1.810, 0.854, 0.460, 0.292,  0.000],
    'K':  [-3.900,  1.000,  1.000, 146.19,  9.74, 2.258, 0.930, 0.470, 0.333, 10.530],
    'L':  [ 3.800,  0.000,  0.000, 131.17,  5.98, 1.931, 0.854, 0.370, 0.292,  0.000],
    'M':  [ 1.900,  0.000,  0.000, 149.21,  5.74, 2.034, 0.894, 0.300, 0.208,  0.000],
    'N':  [-3.500,  0.000,  1.000, 132.12,  5.41, 1.655, 0.901, 0.460, 0.208,  0.000],
    'P':  [-1.600,  0.000,  0.000, 115.13,  6.30, 1.468, 0.858, 0.510, 0.167,  0.000],
    'Q':  [-3.500,  0.000,  1.000, 146.15,  5.65, 1.932, 0.930, 0.490, 0.208,  0.000],
    'R':  [-4.500,  1.000,  1.000, 174.20, 10.76, 2.560, 0.959, 0.530, 0.292, 12.480],
    'S':  [-0.800,  0.000,  1.000, 105.09,  5.68, 1.298, 0.883, 0.510, 0.208,  0.000],
    'T':  [-0.700,  0.000,  1.000, 119.12,  5.60, 1.525, 0.886, 0.440, 0.208,  0.000],
    'V':  [ 4.200,  0.000,  0.000, 117.15,  5.96, 1.645, 0.851, 0.390, 0.292,  0.000],
    'W':  [-0.900,  0.000,  0.000, 204.23,  5.89, 2.663, 0.912, 0.310, 0.167,  0.000],
    'Y':  [-1.300,  0.000,  1.000, 181.19,  5.66, 2.368, 0.922, 0.420, 0.208, 10.070],
}

def get_physicochemical(sequence):
    sequence = str(sequence).upper()
    length = len(sequence)
    if length == 0:
        return [0] * 10
    prop_sum = [0.0] * 10
    count = 0
    for aa in sequence:
        if aa in physicochemical:
            for i in range(10):
                prop_sum[i] += physicochemical[aa][i]
            count += 1
    if count > 0:
        prop_avg = [p / count for p in prop_sum]
    else:
        prop_avg = [0] * 10
    return prop_avg

phys_features = df["sequence"].apply(get_physicochemical)
X_phys = np.array(phys_features.tolist())

# ═══════════════════════════════════════
#        COMBINE + SPLIT + SMOTE
# ═══════════════════════════════════════

X = np.hstack([X_aac, X_dpc, X_entropy, X_phys])
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print(f"Features Shape: {X.shape}")

# ═══════════════════════════════════════
#   METHOD 1 — STACKING CLASSIFIER
# ═══════════════════════════════════════

print("\n" + "="*50)
print("METHOD 1: STACKING (RF + XGB + LGBM → LogReg)")
print("="*50)

base_models = [
    ('rf',   RandomForestClassifier(n_estimators=100, random_state=42)),
    ('xgb',  XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')),
    ('lgbm', LGBMClassifier(n_estimators=100, random_state=42, verbose=-1))
]

stacking_model = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5
)

stacking_model.fit(X_train_sm, y_train_sm)


y_pred_stack = stacking_model.predict(X_test)
y_prob_stack = stacking_model.predict_proba(X_test)[:, 1]

acc_stack = accuracy_score(y_test, y_pred_stack)
mcc_stack = matthews_corrcoef(y_test, y_pred_stack)
auc_stack = roc_auc_score(y_test, y_prob_stack)

print(f"Accuracy : {round(acc_stack * 100, 2)}%")
print(f"MCC      : {round(mcc_stack, 4)}")
print(f"AUC      : {round(auc_stack, 4)}")

# ═══════════════════════════════════════
#   METHOD 2 — ENSEMBLE + FEATURE SELECTION
# ═══════════════════════════════════════

print("\n" + "="*50)
print("METHOD 2: ENSEMBLE + FEATURE SELECTION")
print("="*50)

# Feature Selection pehle
selector = SelectFromModel(
    RandomForestClassifier(n_estimators=100, random_state=42),
    max_features=100
)
selector.fit(X_train_sm, y_train_sm)

X_train_fs = selector.transform(X_train_sm)
X_test_fs  = selector.transform(X_test)

print(f"Selected Features: {X_train_fs.shape[1]}")

# Ensemble in selected features
ensemble_fs_model = VotingClassifier(
    estimators=[
        ('rf',   RandomForestClassifier(n_estimators=100, random_state=42)),
        ('xgb',  XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')),
        ('lgbm', LGBMClassifier(n_estimators=100, random_state=42, verbose=-1))
    ],
    voting='soft'
)

ensemble_fs_model.fit(X_train_fs, y_train_sm)


y_pred_ens_fs = ensemble_fs_model.predict(X_test_fs)
y_prob_ens_fs = ensemble_fs_model.predict_proba(X_test_fs)[:, 1]

acc_ens_fs = accuracy_score(y_test, y_pred_ens_fs)
mcc_ens_fs = matthews_corrcoef(y_test, y_pred_ens_fs)
auc_ens_fs = roc_auc_score(y_test, y_prob_ens_fs)

print(f"Accuracy : {round(acc_ens_fs * 100, 2)}%")
print(f"MCC      : {round(mcc_ens_fs, 4)}")
print(f"AUC      : {round(auc_ens_fs, 4)}")

# ═══════════════════════════════════════
#        FINAL COMPARISON
# ═══════════════════════════════════════

print("\n" + "="*50)
print("FINAL COMPARISON")
print("="*50)
print(f"Stacking          : Acc={round(acc_stack*100,2)}%  MCC={round(mcc_stack,4)}  AUC={round(auc_stack,4)}")
print(f"Ensemble+FS        : Acc={round(acc_ens_fs*100,2)}%  MCC={round(mcc_ens_fs,4)}  AUC={round(auc_ens_fs,4)}")


# Logistic Regression
print("\n------Logistic Regression-----\n")
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(
    max_iter=1000,        # zyada iterations — convergence ke liye
    random_state=42
)
lr_model.fit(X_train_sm, y_train_sm)


# Prediction
y_pred_lr = lr_model.predict(X_test)
y_prob_lr = lr_model.predict_proba(X_test)[:, 1]

# Evaluation
acc_lr = accuracy_score(y_test, y_pred_lr)
mcc_lr = matthews_corrcoef(y_test, y_pred_lr)
auc_lr = roc_auc_score(y_test, y_prob_lr)

# Printing
print(f"Accuracy : {round(acc_lr * 100, 2)}%")
print(f"MCC      : {round(mcc_lr, 4)}")
print(f"AUC      : {round(auc_lr, 4)}")

# Stacking + ESM
print("\n" + "="*50)
print("METHOD 3: STACKING + ESM")
print("="*50)
import os
import numpy as np
import esm
import torch
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
if os.path.exists("X_esm.npy"):
    print("ESM embeddings loading...")
    X_esm = np.load("X_esm.npy")
    print(f"Loaded! Shape: {X_esm.shape}")
else:
    print("Loading ESM model...")
    esm_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    esm_model.eval()
    batch_converter = alphabet.get_batch_converter()
   

    def get_esm_embedding(sequence):
        data = [("protein", str(sequence))]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        with torch.no_grad():
            results = esm_model(batch_tokens, repr_layers=[6])
        embedding = results["representations"][6][0, 1:-1].mean(0)
        return embedding.numpy()

    print("Generating ESM embeddings...")
    embeddings = []
    for i, seq in enumerate(df["sequence"]):
        emb = get_esm_embedding(seq)
        embeddings.append(emb)
        if i % 100 == 0:
            print(f"Progress: {i}/{len(df)}")

    X_esm = np.array(embeddings)
    np.save("X_esm.npy", X_esm)
    print(f"Shape: {X_esm.shape}")

# ESM + Handcrafted features combine karo
X_esm_stack = np.hstack([X_aac, X_dpc, X_entropy, X_phys, X_esm])
y_esm_stack = df["label"].values

print(f"Combined Feature Shape: {X_esm_stack.shape}")

# Train Test Split
X_train_es, X_test_es, y_train_es, y_test_es = train_test_split(
    X_esm_stack, y_esm_stack,
    test_size=0.2,
    random_state=42,
    stratify=y_esm_stack
)

# SMOTE
smote_es = SMOTE(random_state=42)
X_train_es_sm, y_train_es_sm = smote_es.fit_resample(X_train_es, y_train_es)

# Stacking Model
base_models_esm = [
    ('rf',   RandomForestClassifier(n_estimators=100, random_state=42)),
    ('xgb',  XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')),
    ('lgbm', LGBMClassifier(n_estimators=100, random_state=42, verbose=-1))
]

stacking_esm_model = StackingClassifier(
    estimators=base_models_esm,
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5
)

stacking_esm_model.fit(X_train_es_sm, y_train_es_sm)


# Prediction
y_pred_stack_esm = stacking_esm_model.predict(X_test_es)
y_prob_stack_esm = stacking_esm_model.predict_proba(X_test_es)[:, 1]

# Evaluation
acc_stack_esm = accuracy_score(y_test_es, y_pred_stack_esm)
mcc_stack_esm = matthews_corrcoef(y_test_es, y_pred_stack_esm)
auc_stack_esm = roc_auc_score(y_test_es, y_prob_stack_esm)

print(f"Accuracy : {round(acc_stack_esm * 100, 2)}%")
print(f"MCC      : {round(mcc_stack_esm, 4)}")
print(f"AUC      : {round(auc_stack_esm, 4)}")

print("\n" + "="*50)
print("FINAL COMPARISON")
print("="*50)

print(f"Stacking + ESM          : Acc={round(acc_stack_esm*100,2)}%  MCC={round(mcc_stack_esm,4)}  AUC={round(auc_stack_esm,4)}")
