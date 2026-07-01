# Library
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, 
                             matthews_corrcoef, roc_auc_score,
                             precision_score, recall_score, f1_score)
from imblearn.over_sampling import SMOTE
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from xgboost import XGBClassifier                                                                    # for XGBoost model
from catboost import CatBoostClassifier                                                              # for CatBoost model
from lightgbm import LGBMClassifier                                                                  # for LightGSM model
from sklearn.svm import SVC                                                                          # for SVM
import torch                                                                                         # for esm 
import esm                                                                                           # for esm
import matplotlib.pyplot as plt

# for neural network
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import VotingClassifier                                                       # for VoteClassifier
from sklearn.feature_selection import SelectFromModel                                               # for FeatureSelection

from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
import joblib                                                                                       # for checkpoint

# DATA LOADING
# ASYMMETRIC
asym_folder = "Asymmetric_dimethylarginine"
asym_neg_path = os.path.join(asym_folder, "mouse_negative_asymmetric_dimethylarginine_sequences.csv")
asym_pos_path = os.path.join(asym_folder, "mouse_positive_asymmetric_dimethylarginine_sequences.csv")
df_asym_neg = pd.read_csv(asym_neg_path)
df_asym_pos = pd.read_csv(asym_pos_path)
df_asym_neg["label"] = 0 # NON METHYLATED
df_asym_pos["label"] = 1 # METHYLATED

# SYMMETRIC
sym_folder = "Symmetric_dimethylarginine"
sym_neg_path = os.path.join(sym_folder, "mouse_negative_symmetric_dimethylarginine_sequences.csv")
sym_pos_path = os.path.join(sym_folder, "mouse_positive_symmetric_dimethylarginine_sequences.csv")
df_sym_neg = pd.read_csv(sym_neg_path)
df_sym_pos = pd.read_csv(sym_pos_path)
df_sym_neg["label"] = 0
df_sym_pos["label"] = 1

# DIMETHYL ARGININE
dimethyl_folder = "Dimethyl_arginine"
dimethyl_neg_path = os.path.join(dimethyl_folder, "mouse_negative_dimethylated_arginine_sequences.csv")
dimethyl_pos_path = os.path.join(dimethyl_folder, "mouse_positive_dimethylated_arginine_sequences.csv")
df_dimethyl_neg = pd.read_csv(dimethyl_neg_path)
df_dimethyl_pos = pd.read_csv(dimethyl_pos_path)
df_dimethyl_neg["label"] = 0
df_dimethyl_pos["label"] = 1

# OMEGA-N-METHYLARGININE
omega_folder = "Omega-N-methylarginine"
omega_neg_path = os.path.join(omega_folder, "mouse_negative_omega-n-methylarginine_sequences.csv")
omega_pos_path = os.path.join(omega_folder, "mouse_positive_omega-n-methylarginine_sequences.csv")
df_omega_neg = pd.read_csv(omega_neg_path)
df_omega_pos = pd.read_csv(omega_pos_path)
df_omega_neg["label"] = 0
df_omega_pos["label"] = 1

# MERGE
df = pd.concat([df_asym_neg, df_asym_pos, df_sym_neg, df_sym_pos,
                df_dimethyl_neg, df_dimethyl_pos, df_omega_neg, df_omega_pos],
                ignore_index=True)

print(df.shape)
print(df["label"].value_counts())
print(df.columns)

df["sequence"] = df["Positive_sequence"].fillna(df["Negative_sequence"])
print(df["sequence"].isnull().sum())






# FEATURE EXTRACTION


amino_acids = list("ACDEFGHIKLMNPQRSTVWY")

# AAC (F1 - F20)
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
print(X_aac.shape)

# DPC (F21 - F420)
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
print(X_dpc.shape)

# ENTROPY FEATURES (F421 - F424)
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
print(X_entropy.shape)







# PHYSICOCHEMICAL PROPERTIES (F425 - F434)
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
print(X_phys.shape)


# COMBINE ALL FEATURES


X = np.hstack([X_aac, X_dpc, X_entropy, X_phys])  # (6326, 434)
y = df["label"].values


# TRAIN / TEST SPLIT


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print(X_train.shape)
print(X_test.shape)

# SMOTE


smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print("Before SMOTE:", y_train.sum(), "methylated vs", len(y_train) - y_train.sum(), "non-methylated")
print("After SMOTE :", y_train_sm.sum(), "methylated vs", len(y_train_sm) - y_train_sm.sum(), "non-methylated")











def train_or_load(model, model_name, X_train, y_train):
    checkpoint_path = f"checkpoint_{model_name}.pkl"
    
    if os.path.exists(checkpoint_path):
        print(f"Checkpoint — {model_name} loading...")
        loaded_model = joblib.load(checkpoint_path)
        return loaded_model
    else:
        print(f"Training {model_name} from scratch...")
        model.fit(X_train, y_train)
        joblib.dump(model, checkpoint_path)
        print(f"{model_name} trained and saved!")
        return model










# ALL THE MACHINE LEARNING MODEL





# ESM Combined + Random Forest
print("\n------ESM Combined + Random Forest-----\n")

if os.path.exists("X_esm.npy"):
    X_esm = np.load("X_esm.npy")
else:
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

    embeddings = []
    for i, seq in enumerate(df["sequence"]):
        emb = get_esm_embedding(seq)
        embeddings.append(emb)
        if i % 100 == 0:
            print(f"Progress: {i}/{len(df)}")

    X_esm = np.array(embeddings)
    np.save("X_esm.npy", X_esm)
    print(f"Shape: {X_esm.shape}")

# AAC + DPC + Entropy + Phys + ESM combine 
X_esm_combined = np.hstack([X_aac, X_dpc, X_entropy, X_phys, X_esm])
y_esm_combined = df["label"].values

print(f"Combined Feature Shape: {X_esm_combined.shape}")  # (6326, 754)

# Train Test Split
X_train_esm_c, X_test_esm_c, y_train_esm_c, y_test_esm_c = train_test_split(
    X_esm_combined, y_esm_combined,
    test_size=0.2,
    random_state=42,
    stratify=y_esm_combined
)

# SMOTE
smote_esm_c = SMOTE(random_state=42)
X_train_esm_c_sm, y_train_esm_c_sm = smote_esm_c.fit_resample(X_train_esm_c, y_train_esm_c)

# Train
# Train
esm_combined_model = train_or_load(
    RandomForestClassifier(n_estimators=100, random_state=42),
    "ESM_Combined_RF",
    X_train_esm_c_sm, y_train_esm_c_sm
)

# Prediction
y_pred_esm_c = esm_combined_model.predict(X_test_esm_c)
y_prob_esm_c = esm_combined_model.predict_proba(X_test_esm_c)[:, 1]

# Evaluation
acc_esm_c = accuracy_score(y_test_esm_c, y_pred_esm_c)
mcc_esm_c = matthews_corrcoef(y_test_esm_c, y_pred_esm_c)
auc_esm_c = roc_auc_score(y_test_esm_c, y_prob_esm_c)

# Printing
print(f"Accuracy : {round(acc_esm_c * 100, 2)}%")
print(f"MCC      : {round(mcc_esm_c, 4)}")
print(f"AUC      : {round(auc_esm_c, 4)}")








# ESM + SVM
print("\n------ESM + SVM-----\n")

y_esm_svm = df["label"].values

# Train Test Split
X_train_esm_svm, X_test_esm_svm, y_train_esm_svm, y_test_esm_svm = train_test_split(
    X_esm, y_esm_svm,
    test_size=0.2,
    random_state=42,
    stratify=y_esm_svm
)

# SMOTE
smote_esm_svm = SMOTE(random_state=42)
X_train_esm_svm_sm, y_train_esm_svm_sm = smote_esm_svm.fit_resample(X_train_esm_svm, y_train_esm_svm)

# Train
# Train
esm_svm_model = train_or_load(
    SVC(kernel='rbf', random_state=42, probability=True),
    "ESM_SVM",
    X_train_esm_svm_sm, y_train_esm_svm_sm
)


# Prediction
y_pred_esm_svm = esm_svm_model.predict(X_test_esm_svm)
y_prob_esm_svm = esm_svm_model.predict_proba(X_test_esm_svm)[:, 1]

# Evaluation
acc_esm_svm = accuracy_score(y_test_esm_svm, y_pred_esm_svm)
mcc_esm_svm = matthews_corrcoef(y_test_esm_svm, y_pred_esm_svm)
auc_esm_svm = roc_auc_score(y_test_esm_svm, y_prob_esm_svm)

# Printing
print(f"Accuracy : {round(acc_esm_svm * 100, 2)}%")
print(f"MCC      : {round(mcc_esm_svm, 4)}")
print(f"AUC      : {round(auc_esm_svm, 4)}")









# SVM
print("\n------SVM-----\n")
svm_model = SVC(
    kernel='rbf',       
    random_state=42,
    probability=True     
)

svm_model = train_or_load(
    SVC(n_estimators=100, random_state=42, probability=True),
    "SVM",
    X_train_sm, y_train_sm
)

# Prediction
y_pred_svm = svm_model.predict(X_test)
y_prob_svm = svm_model.predict_proba(X_test)[:, 1]

# Evaluation
acc_svm = accuracy_score(y_test, y_pred_svm)
mcc_svm = matthews_corrcoef(y_test, y_pred_svm)
auc_svm = roc_auc_score(y_test, y_prob_svm)

# Printing
print(f"Accuracy : {round(acc_svm * 100, 2)}%")
print(f"MCC      : {round(mcc_svm, 4)}")
print(f"AUC      : {round(auc_svm, 4)}")








# LIGHTGBM
print("\n-------LightGBM--------\n")
light_model = LGBMClassifier(
    n_estimators = 100,
    random_state = 42,
    verbose = -1
)

model = train_or_load(
    LGBMClassifier(n_estimators=100, random_state=42),
    "LightGBM",
    X_train_sm, y_train_sm
)

# Pediction 
y_pred_light = light_model.predict(X_test)
y_prob_light = light_model.predict_proba(X_test)[:,1]

# Evaluation
acc_light = accuracy_score(y_test, y_pred_light)
mcc_light = matthews_corrcoef(y_test, y_pred_light)
auc_light = roc_auc_score(y_test , y_prob_light)

# Printing
print(f"Accuracy : {round(acc_light * 100, 2)}%")
print(f"MCC      : {round(mcc_light, 4)}")
print(f"AUC      : {round(auc_light, 4)}")







# Naive Bayes
print("\n------Naive Bayes-----\n")
from sklearn.naive_bayes import GaussianNB

nb_model = GaussianNB()

nb_model = train_or_load(
    GaussianNB(),
    "Naive Bayes",
    X_train_sm, y_train_sm
)
# Prediction
y_pred_nb = nb_model.predict(X_test)
y_prob_nb = nb_model.predict_proba(X_test)[:, 1]

# Evaluation
acc_nb = accuracy_score(y_test, y_pred_nb)
mcc_nb = matthews_corrcoef(y_test, y_pred_nb)
auc_nb = roc_auc_score(y_test, y_prob_nb)

# Printing
print(f"Accuracy : {round(acc_nb * 100, 2)}%")
print(f"MCC      : {round(mcc_nb, 4)}")
print(f"AUC      : {round(auc_nb, 4)}")





# KNN
print("\n------KNN-----\n")
from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(
    n_neighbors=5,    # 5 nearest neighbors
    metric='euclidean'
)
knn_model = train_or_load(
    KNeighborsClassifier(n_estimators=100, metric='euclidean'),
    "KNN",
    X_train_sm, y_train_sm
)
# Prediction
y_pred_knn = knn_model.predict(X_test)
y_prob_knn = knn_model.predict_proba(X_test)[:, 1]

# Evaluation
acc_knn = accuracy_score(y_test, y_pred_knn)
mcc_knn = matthews_corrcoef(y_test, y_pred_knn)
auc_knn = roc_auc_score(y_test, y_prob_knn)

# Printing
print(f"Accuracy : {round(acc_knn * 100, 2)}%")
print(f"MCC      : {round(mcc_knn, 4)}")
print(f"AUC      : {round(auc_knn, 4)}")












#   STACKING (RF + XGB + LGBM → LogReg)

print("\n------ STACKING (RF + XGB + LGBM → LogReg)-----\n")

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

stacking_model = train_or_load(
    StackingClassifier(
        estimators=[
            ('rf',   RandomForestClassifier(n_estimators=100, random_state=42)),
            ('xgb',  XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')),
            ('lgbm', LGBMClassifier(n_estimators=100, random_state=42, verbose=-1))
        ],
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5
    ),
    "Stacking",
    X_train_sm, y_train_sm
)

y_pred_stack = stacking_model.predict(X_test)
y_prob_stack = stacking_model.predict_proba(X_test)[:, 1]

acc_stack = accuracy_score(y_test, y_pred_stack)
mcc_stack = matthews_corrcoef(y_test, y_pred_stack)
auc_stack = roc_auc_score(y_test, y_prob_stack)

print(f"Accuracy : {round(acc_stack * 100, 2)}%")
print(f"MCC      : {round(mcc_stack, 4)}")
print(f"AUC      : {round(auc_stack, 4)}")














# Stacking with 5 Base Models
print("\n------STACKING (RF + XGB + LGBM + CAT + SVM)-----\n")

base_models_v2 = [
    ('rf',    RandomForestClassifier(n_estimators=100, random_state=42)),
    ('xgb',   XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')),
    ('lgbm',  LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)),
    ('cat',   CatBoostClassifier(iterations=100, random_seed=42, verbose=0)),
    ('svm',   SVC(kernel='rbf', random_state=42, probability=True))
]

stacking_model_v2 = StackingClassifier(
    estimators=base_models_v2,
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5
)

stacking_model_v2.fit(X_train_sm, y_train_sm)

# Prediction
y_pred_stack_v2 = stacking_model_v2.predict(X_test)
y_prob_stack_v2 = stacking_model_v2.predict_proba(X_test)[:, 1]

# Evaluation
acc_stack_v2 = accuracy_score(y_test, y_pred_stack_v2)
mcc_stack_v2 = matthews_corrcoef(y_test, y_pred_stack_v2)
auc_stack_v2 = roc_auc_score(y_test, y_prob_stack_v2)

print(f"Accuracy : {round(acc_stack_v2 * 100, 2)}%")
print(f"MCC      : {round(mcc_stack_v2, 4)}")
print(f"AUC      : {round(auc_stack_v2, 4)}")

print(f"Stacking (5 models): Acc={round(acc_stack_v2*100,2)}%  MCC={round(mcc_stack_v2,4)}  AUC={round(auc_stack_v2,4)}")















#   ENSEMBLE + FEATURE SELECTION

print("\n------ENSEMBLE + FEATURE SELECTION-----\n")

selector = SelectFromModel(
    RandomForestClassifier(n_estimators=100, random_state=42),
    max_features=100
)
selector.fit(X_train_sm, y_train_sm)
X_train_fs = selector.transform(X_train_sm)
X_test_fs  = selector.transform(X_test)

ensemble_fs_model = VotingClassifier(
    estimators=[
        ('rf',   RandomForestClassifier(n_estimators=100, random_state=42)),
        ('xgb',  XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')),
        ('lgbm', LGBMClassifier(n_estimators=100, random_state=42, verbose=-1))
    ],
    voting='soft'
)


selector = SelectFromModel(
    RandomForestClassifier(n_estimators=100, random_state=42),
    max_features=100
)
selector.fit(X_train_sm, y_train_sm)
X_train_fs = selector.transform(X_train_sm)
X_test_fs  = selector.transform(X_test)

ensemble_fs_model = train_or_load(
    VotingClassifier(
        estimators=[
            ('rf',   RandomForestClassifier(n_estimators=100, random_state=42)),
            ('xgb',  XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')),
            ('lgbm', LGBMClassifier(n_estimators=100, random_state=42, verbose=-1))
        ],
        voting='soft'
    ),
    "Ensemble_FS",
    X_train_fs, y_train_sm
)

y_pred_ens_fs = ensemble_fs_model.predict(X_test_fs)
y_prob_ens_fs = ensemble_fs_model.predict_proba(X_test_fs)[:, 1]

acc_ens_fs = accuracy_score(y_test, y_pred_ens_fs)
mcc_ens_fs = matthews_corrcoef(y_test, y_pred_ens_fs)
auc_ens_fs = roc_auc_score(y_test, y_prob_ens_fs)

print(f"Accuracy : {round(acc_ens_fs * 100, 2)}%")
print(f"MCC      : {round(mcc_ens_fs, 4)}")
print(f"AUC      : {round(auc_ens_fs, 4)}")



#   LOGISTIC REGRESSION

print("\n------LOGISTIC REGRESSION-----\n")

lr_model = LogisticRegression(max_iter=1000, random_state=42)

lr_model = train_or_load(
    LogisticRegression(max_iter=1000, random_state=42),
    "LOGISTIC REGRESSION",
    X_train_sm, y_train_sm
)

y_pred_lr = lr_model.predict(X_test)
y_prob_lr = lr_model.predict_proba(X_test)[:, 1]

acc_lr = accuracy_score(y_test, y_pred_lr)
mcc_lr = matthews_corrcoef(y_test, y_pred_lr)
auc_lr = roc_auc_score(y_test, y_prob_lr)

print(f"Accuracy : {round(acc_lr * 100, 2)}%")
print(f"MCC      : {round(mcc_lr, 4)}")
print(f"AUC      : {round(auc_lr, 4)}")












# KNN + ESM
print("\n------KNN + ESM-----\n")

# X_esm already load ho chuka hai
y_esm_knn = df["label"].values

# Train Test Split
X_train_esm_knn, X_test_esm_knn, y_train_esm_knn, y_test_esm_knn = train_test_split(
    X_esm, y_esm_knn,
    test_size=0.2,
    random_state=42,
    stratify=y_esm_knn
)

# SMOTE
smote_esm_knn = SMOTE(random_state=42)
X_train_esm_knn_sm, y_train_esm_knn_sm = smote_esm_knn.fit_resample(X_train_esm_knn, y_train_esm_knn)

# Train
# Train
knn_esm_model = train_or_load(
    KNeighborsClassifier(n_neighbors=5, metric='euclidean'),
    "KNN_ESM",
    X_train_esm_knn_sm, y_train_esm_knn_sm
)

# Prediction
y_pred_esm_knn = knn_esm_model.predict(X_test_esm_knn)
y_prob_esm_knn = knn_esm_model.predict_proba(X_test_esm_knn)[:, 1]

# Evaluation
acc_esm_knn = accuracy_score(y_test_esm_knn, y_pred_esm_knn)
mcc_esm_knn = matthews_corrcoef(y_test_esm_knn, y_pred_esm_knn)
auc_esm_knn = roc_auc_score(y_test_esm_knn, y_prob_esm_knn)

# Printing
print(f"Accuracy : {round(acc_esm_knn * 100, 2)}%")
print(f"MCC      : {round(mcc_esm_knn, 4)}")
print(f"AUC      : {round(auc_esm_knn, 4)}")









#CATBOOST
print("\n-------CatBoost-------")
cat_model = CatBoostClassifier(
    iterations = 100,
    random_seed = 42,
    verbose = 0                          # it won't let the training data print
)


cat_model = train_or_load(
    CatBoostClassifier(n_estimators=100, random_state=42,verbose=0),
    "CatBoost",
    X_train_sm, y_train_sm
)
# Prediction
y_pred_cat = cat_model.predict(X_test)
y_prob_cat = cat_model.predict_proba(X_test)[:,1]

# Evaluation
acc_cat = accuracy_score(y_test, y_pred_cat)
mcc_cat = matthews_corrcoef(y_test, y_pred_cat)
auc_cat = roc_auc_score(y_test, y_prob_cat)

# Printing
print(f"Accuracy : {round(acc_cat * 100, 2)}%")
print(f"MCC      : {round(mcc_cat, 4)}")
print(f"AUC      : {round(auc_cat, 4)}")








# XGBOOST
print("\n------XGBoost-----\n")
xgb_model = XGBClassifier(
    n_estimators=100,
    random_state=42,
    eval_metric="logloss"
)
xgb_model = train_or_load(
    XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss"),
    "xgboost",
    X_train_sm, y_train_sm
)

# Prediction
y_pred_xgb = xgb_model.predict(X_test)
y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

# Evaluation
acc_xgb = accuracy_score(y_test, y_pred_xgb)
mcc_xgb = matthews_corrcoef(y_test, y_pred_xgb)
auc_xgb = roc_auc_score(y_test, y_prob_xgb)

# Printing
print(f"Accuracy : {round(acc_xgb * 100, 2)}%")
print(f"MCC      : {round(mcc_xgb, 4)}")
print(f"AUC      : {round(auc_xgb, 4)}")






# RANDOM FOREST TRAINING

print("\n------RANDOM FOREST-------\n")
model = RandomForestClassifier(n_estimators=100, random_state=42)

model = train_or_load(
    RandomForestClassifier(n_estimators=100, random_state=42),
    "random_forest",
    X_train_sm, y_train_sm
)
# PREDICTION

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# EVALUATION METRICS

acc          = accuracy_score(y_test, y_pred)
mcc          = matthews_corrcoef(y_test, y_pred)
auc          = roc_auc_score(y_test, y_prob)
precision_nm = precision_score(y_test, y_pred, pos_label=0)
precision_m  = precision_score(y_test, y_pred, pos_label=1)
recall_nm    = recall_score(y_test, y_pred, pos_label=0)
recall_m     = recall_score(y_test, y_pred, pos_label=1)
f1_nm        = f1_score(y_test, y_pred, pos_label=0)
f1_m         = f1_score(y_test, y_pred, pos_label=1)
support_nm   = int((y_test == 0).sum())
support_m    = int((y_test == 1).sum())

# Tabular Report
print("\n" + "="*65)
print(f"{'EVALUATION RESULTS':^65}")
print("="*65)
print(f"{'Metric':<20} {'Non-Methylated':>20} {'Methylated':>20}")
print("-"*65)
print(f"{'Precision':<20} {precision_nm:>20.4f} {precision_m:>20.4f}")
print(f"{'Recall':<20} {recall_nm:>20.4f} {recall_m:>20.4f}")
print(f"{'F1-Score':<20} {f1_nm:>20.4f} {f1_m:>20.4f}")
print(f"{'Support':<20} {support_nm:>20} {support_m:>20}")
print("-"*65)
print(f"{'Accuracy':<20} {round(acc * 100, 2):>20}%")
print(f"{'MCC':<20} {round(mcc, 4):>20}")
print(f"{'AUC':<20} {round(auc, 4):>20}")
print("="*65)

# 10-FOLD CROSS VALIDATION

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy")

print(f"\nCV Scores:  {cv_scores}")
print(f"CV Mean:    {cv_scores.mean() * 100:.2f}%")
print(f"CV Std:     {cv_scores.std() * 100:.2f}%")










# HYPERPARAMETER TUNING 
print("\n------Hyperparameter Tuning (GridSearchCV)-----\n")
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators' : [100, 200, 300],
    'max_depth'    : [10, 20, None],
    'max_features' : ['sqrt', 'log2'],
    'min_samples_split' : [2, 5]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,              # 5-fold CV
    scoring='accuracy',
    n_jobs=-1,        
    verbose=2          
)

grid_search.fit(X_train_sm, y_train_sm)

print(f"\nBest Parameters: {grid_search.best_params_}")
print(f"Best CV Score:   {round(grid_search.best_score_ * 100, 2)}%")

# Best model 
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)
y_prob_best = best_model.predict_proba(X_test)[:, 1]

# Evaluation
acc_best = accuracy_score(y_test, y_pred_best)
mcc_best = matthews_corrcoef(y_test, y_pred_best)
auc_best = roc_auc_score(y_test, y_prob_best)

print(f"\nAccuracy : {round(acc_best * 100, 2)}%")
print(f"MCC      : {round(mcc_best, 4)}")
print(f"AUC      : {round(auc_best, 4)}")













# GridSearchCV + ESM
print("\n------GridSearchCV + ESM-----\n")


y_esm_grid = df["label"].values

# Train Test Split
X_train_esm_grid, X_test_esm_grid, y_train_esm_grid, y_test_esm_grid = train_test_split(
    X_esm, y_esm_grid,
    test_size=0.2,
    random_state=42,
    stratify=y_esm_grid
)

# SMOTE
smote_esm_grid = SMOTE(random_state=42)
X_train_esm_grid_sm, y_train_esm_grid_sm = smote_esm_grid.fit_resample(X_train_esm_grid, y_train_esm_grid)

# GridSearchCV
param_grid_esm = {
    'n_estimators'      : [100, 200, 300],
    'max_depth'         : [10, 20, None],
    'max_features'      : ['sqrt', 'log2'],
    'min_samples_split' : [2, 5]
}

grid_search_esm = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid_esm,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2
)

grid_search_esm.fit(X_train_esm_grid_sm, y_train_esm_grid_sm)

print(f"\nBest Parameters: {grid_search_esm.best_params_}")
print(f"Best CV Score:   {round(grid_search_esm.best_score_ * 100, 2)}%")

# Best model
best_esm_model = grid_search_esm.best_estimator_
y_pred_esm_grid = best_esm_model.predict(X_test_esm_grid)
y_prob_esm_grid = best_esm_model.predict_proba(X_test_esm_grid)[:, 1]

# Evaluation
acc_esm_grid = accuracy_score(y_test_esm_grid, y_pred_esm_grid)
mcc_esm_grid = matthews_corrcoef(y_test_esm_grid, y_pred_esm_grid)
auc_esm_grid = roc_auc_score(y_test_esm_grid, y_prob_esm_grid)

print(f"Accuracy : {round(acc_esm_grid * 100, 2)}%")
print(f"MCC      : {round(mcc_esm_grid, 4)}")
print(f"AUC      : {round(auc_esm_grid, 4)}")












#   STACKING + ESM

print("\n------STACKING + ESM-----\n")

X_esm_stack = np.hstack([X_aac, X_dpc, X_entropy, X_phys, X_esm])
y_esm_stack = df["label"].values

X_train_es, X_test_es, y_train_es, y_test_es = train_test_split(
    X_esm_stack, y_esm_stack,
    test_size=0.2,
    random_state=42,
    stratify=y_esm_stack
)

smote_es = SMOTE(random_state=42)
X_train_es_sm, y_train_es_sm = smote_es.fit_resample(X_train_es, y_train_es)

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


stacking_esm_model = train_or_load(
    StackingClassifier(
        estimators=[
            ('rf',   RandomForestClassifier(n_estimators=100, random_state=42)),
            ('xgb',  XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')),
            ('lgbm', LGBMClassifier(n_estimators=100, random_state=42, verbose=-1))
        ],
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5
    ),
    "Stacking_ESM",
    X_train_es_sm, y_train_es_sm
)


y_pred_stack_esm = stacking_esm_model.predict(X_test_es)
y_prob_stack_esm = stacking_esm_model.predict_proba(X_test_es)[:, 1]

acc_stack_esm = accuracy_score(y_test_es, y_pred_stack_esm)
mcc_stack_esm = matthews_corrcoef(y_test_es, y_pred_stack_esm)
auc_stack_esm = roc_auc_score(y_test_es, y_prob_stack_esm)

print(f"Accuracy : {round(acc_stack_esm * 100, 2)}%")
print(f"MCC      : {round(mcc_stack_esm, 4)}")
print(f"AUC      : {round(auc_stack_esm, 4)}")









# NEURAL NETWORK
class MethylationNN(nn.Module):
    def __init__(self):
        super(MethylationNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(434, 256),   # Input → Hidden 1
            nn.ReLU(),             # Activation function
            nn.Dropout(0.3),       # Overfitting rokne ke liye
            nn.Linear(256, 128),   # Hidden 1 → Hidden 2
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),    # Hidden 2 → Hidden 3
            nn.ReLU(),
            nn.Linear(64, 1),      # Hidden 3 → Output
            nn.Sigmoid()           # 0 to 1 ke beech output
        )

    def forward(self, x):
        return self.network(x)
    
print("\n------Neural Network (MLP)-----\n")


X_train_tensor = torch.FloatTensor(X_train_sm)
y_train_tensor = torch.FloatTensor(y_train_sm).unsqueeze(1)
X_test_tensor  = torch.FloatTensor(X_test)
y_test_tensor  = torch.FloatTensor(y_test).unsqueeze(1)

# DataLoader 
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader  = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Model 
nn_model   = MethylationNN()
criterion  = nn.BCELoss()           # Binary Cross Entropy Loss
optimizer  = optim.Adam(nn_model.parameters(), lr=0.001)

# Train 
epochs = 50
for epoch in range(epochs):
    nn_model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = nn_model(X_batch)
        loss   = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs} → Loss: {total_loss:.4f}")



# Predict 
nn_model.eval()
with torch.no_grad():
    y_prob_nn = nn_model(X_test_tensor).squeeze().numpy()
    y_pred_nn = (y_prob_nn >= 0.5).astype(int)

# Evaluation
acc_nn = accuracy_score(y_test, y_pred_nn)
mcc_nn = matthews_corrcoef(y_test, y_pred_nn)
auc_nn = roc_auc_score(y_test, y_prob_nn)

print(f"Accuracy : {round(acc_nn * 100, 2)}%")
print(f"MCC      : {round(mcc_nn, 4)}")
print(f"AUC      : {round(auc_nn, 4)}")













# ENSEMBLE METHOD
print("\n------Ensemble (RF + XGB + LGBM)-----\n")

ensemble_model = VotingClassifier(
    estimators=[
        ('rf',   RandomForestClassifier(n_estimators=100, random_state=42)),
        ('xgb',  XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')),
        ('lgbm', LGBMClassifier(n_estimators=100, random_state=42, verbose=-1))
    ],
    voting='soft'   
)

model = train_or_load(
    VotingClassifier(n_estimators=100, random_state=42),
    "Ensemble (RF + XGB + LGBM)",
    X_train_sm, y_train_sm
)

# Prediction
y_pred_ens = ensemble_model.predict(X_test)
y_prob_ens = ensemble_model.predict_proba(X_test)[:, 1]

# Evaluation
acc_ens = accuracy_score(y_test, y_pred_ens)
mcc_ens = matthews_corrcoef(y_test, y_pred_ens)
auc_ens = roc_auc_score(y_test, y_prob_ens)

print(f"Accuracy : {round(acc_ens * 100, 2)}%")
print(f"MCC      : {round(mcc_ens, 4)}")
print(f"AUC      : {round(auc_ens, 4)}")










# FEATURE SELECTION WITH RANDOM FOREST
print("\n------Feature Selection (Top 100)-----\n")


selector = SelectFromModel(
    RandomForestClassifier(n_estimators=100, random_state=42),
    max_features=100    # top 100 features
)

selector.fit(X_train_sm, y_train_sm)

# Top 100 features select karo
X_train_fs = selector.transform(X_train_sm)
X_test_fs  = selector.transform(X_test)

print(f"Before Feature Selection: {X_train_sm.shape[1]} features")
print(f"After Feature Selection:  {X_train_fs.shape[1]} features")


fs_model = RandomForestClassifier(n_estimators=100, random_state=42)

model = train_or_load(
    SelectFromModel(n_estimators=100, random_state=42),
    "Feature Selection With Random Forest",
    X_train_fs, y_train_sm
)

# Prediction
y_pred_fs = fs_model.predict(X_test_fs)
y_prob_fs = fs_model.predict_proba(X_test_fs)[:, 1]

# Evaluation
acc_fs = accuracy_score(y_test, y_pred_fs)
mcc_fs = matthews_corrcoef(y_test, y_pred_fs)
auc_fs = roc_auc_score(y_test, y_prob_fs)

print(f"Accuracy : {round(acc_fs * 100, 2)}%")
print(f"MCC      : {round(mcc_fs, 4)}")
print(f"AUC      : {round(auc_fs, 4)}")











# ESM2 Large (650M)
print("\n------ESM2 Large (650M)-----\n")

import esm
import torch

if os.path.exists("X_esm_large.npy"):
    print("ESM Large embeddings loading...")
    X_esm_large = np.load("X_esm_large.npy")
    print(f"Loaded! Shape: {X_esm_large.shape}")
else:
    print("Loading ESM2 Large model (650M)...")
    esm_large_model, alphabet_large = esm.pretrained.esm2_t33_650M_UR50D()
    esm_large_model.eval()
    batch_converter_large = alphabet_large.get_batch_converter()

    def get_esm_large_embedding(sequence):
        data = [("protein", str(sequence))]
        batch_labels, batch_strs, batch_tokens = batch_converter_large(data)
        with torch.no_grad():
            results = esm_large_model(batch_tokens, repr_layers=[33])  # 33 layers
        embedding = results["representations"][33][0, 1:-1].mean(0)
        return embedding.numpy()

    print("Generating ESM Large embeddings... (3-4 ghante lagenge!)")
    embeddings_large = []
    for i, seq in enumerate(df["sequence"]):
        emb = get_esm_large_embedding(seq)
        embeddings_large.append(emb)
        if i % 50 == 0:
            print(f"Progress: {i}/{len(df)}")

    X_esm_large = np.array(embeddings_large)
    np.save("X_esm_large.npy", X_esm_large)
    print(f"Shape: {X_esm_large.shape}")

# Train Test Split
y_esm_large = df["label"].values
X_train_esml, X_test_esml, y_train_esml, y_test_esml = train_test_split(
    X_esm_large, y_esm_large,
    test_size=0.2,
    random_state=42,
    stratify=y_esm_large
)

# SMOTE
smote_esml = SMOTE(random_state=42)
X_train_esml_sm, y_train_esml_sm = smote_esml.fit_resample(X_train_esml, y_train_esml)

# Train
esm_large_rf = RandomForestClassifier(n_estimators=100, random_state=42)
esm_large_rf.fit(X_train_esml_sm, y_train_esml_sm)

# Prediction
y_pred_esml = esm_large_rf.predict(X_test_esml)
y_prob_esml = esm_large_rf.predict_proba(X_test_esml)[:, 1]

# Evaluation
acc_esml = accuracy_score(y_test_esml, y_pred_esml)
mcc_esml = matthews_corrcoef(y_test_esml, y_pred_esml)
auc_esml = roc_auc_score(y_test_esml, y_prob_esml)

print(f"Accuracy : {round(acc_esml * 100, 2)}%")
print(f"MCC      : {round(mcc_esml, 4)}")
print(f"AUC      : {round(auc_esml, 4)}")










# Two-Level (Deep) Stacking

print("\n------Two-Level Stacking-----\n")

from sklearn.model_selection import cross_val_predict

if os.path.exists("checkpoint_deep_level0.pkl") and \
   os.path.exists("checkpoint_deep_level1.pkl") and \
   os.path.exists("checkpoint_deep_level2.pkl"):

    print("Deep Stacking checkpoints — loading...")
    level0_models_trained = joblib.load("checkpoint_deep_level0.pkl")
    level1_model = joblib.load("checkpoint_deep_level1.pkl")
    level2_model = joblib.load("checkpoint_deep_level2.pkl")

    level0_test_preds = np.column_stack([
        m.predict_proba(X_test)[:, 1] for m in level0_models_trained
    ])
    level1_test_pred = level1_model.predict_proba(level0_test_preds)[:, 1]
    final_test_features = np.hstack([level0_test_preds, level1_test_pred.reshape(-1, 1)])

else:
    print("Training Deep Stacking from scratch...")

    level0_models_list = [
        ('rf',   RandomForestClassifier(n_estimators=100, random_state=42)),
        ('xgb',  XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')),
        ('lgbm', LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)),
        ('cat',  CatBoostClassifier(iterations=100, random_seed=42, verbose=0))
    ]

    level0_train_preds = np.zeros((X_train_sm.shape[0], len(level0_models_list)))
    level0_test_preds  = np.zeros((X_test.shape[0], len(level0_models_list)))
    level0_models_trained = []

    for i, (name, mdl) in enumerate(level0_models_list):
        print(f"Training {name}...")
        mdl.fit(X_train_sm, y_train_sm)
        level0_models_trained.append(mdl)
        level0_train_preds[:, i] = cross_val_predict(
            mdl, X_train_sm, y_train_sm, cv=5, method='predict_proba')[:, 1]
        level0_test_preds[:, i] = mdl.predict_proba(X_test)[:, 1]

    level1_model = LogisticRegression(max_iter=1000, random_state=42)
    level1_model.fit(level0_train_preds, y_train_sm)
    level1_train_pred = level1_model.predict_proba(level0_train_preds)[:, 1]
    level1_test_pred  = level1_model.predict_proba(level0_test_preds)[:, 1]

    final_train_features = np.hstack([level0_train_preds, level1_train_pred.reshape(-1, 1)])
    final_test_features  = np.hstack([level0_test_preds, level1_test_pred.reshape(-1, 1)])

    level2_model = LogisticRegression(max_iter=1000, random_state=42)
    level2_model.fit(final_train_features, y_train_sm)

    joblib.dump(level0_models_trained, "checkpoint_deep_level0.pkl")
    joblib.dump(level1_model, "checkpoint_deep_level1.pkl")
    joblib.dump(level2_model, "checkpoint_deep_level2.pkl")


# Prediction
y_pred_deep = level2_model.predict(final_test_features)
y_prob_deep = level2_model.predict_proba(final_test_features)[:, 1]

acc_deep = accuracy_score(y_test, y_pred_deep)
mcc_deep = matthews_corrcoef(y_test, y_pred_deep)
auc_deep = roc_auc_score(y_test, y_prob_deep)

print(f"Accuracy : {round(acc_deep * 100, 2)}%")
print(f"MCC      : {round(mcc_deep, 4)}")
print(f"AUC      : {round(auc_deep, 4)}")













# Weighted Voting
print("\n------Weighted Voting-----\n")

weighted_ensemble = VotingClassifier(
    estimators=[
        ('rf',   RandomForestClassifier(n_estimators=100, random_state=42)),
        ('xgb',  XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')),
        ('lgbm', LGBMClassifier(n_estimators=100, random_state=42, verbose=-1))
    ],
    voting='soft',
    weights=[2, 1, 2]   # RF aur LGBM ko zyada weight (best individual performers)
)

weighted_ensemble.fit(X_train_sm, y_train_sm)


y_pred_weighted = weighted_ensemble.predict(X_test)
y_prob_weighted = weighted_ensemble.predict_proba(X_test)[:, 1]

acc_weighted = accuracy_score(y_test, y_pred_weighted)
mcc_weighted = matthews_corrcoef(y_test, y_pred_weighted)
auc_weighted = roc_auc_score(y_test, y_prob_weighted)

print(f"Accuracy : {round(acc_weighted * 100, 2)}%")
print(f"MCC      : {round(mcc_weighted, 4)}")
print(f"AUC      : {round(auc_weighted, 4)}")



print(f"Two-Level Stacking      : Acc={round(acc_deep*100,2)}%  MCC={round(mcc_deep,4)}  AUC={round(auc_deep,4)}")
print(f"Weighted Voting          : Acc={round(acc_weighted*100,2)}%  MCC={round(mcc_weighted,4)}  AUC={round(auc_weighted,4)}")















# JSON SAVE

now = datetime.now()
date_time = now.strftime("%Y-%m-%d %H:%M:%S")
new_result = {
    "date_time"    : date_time,
    "<----Random_Forest---->": {
        "Accuracy" : round(acc * 100, 2),
        "MCC"      : round(mcc, 4),
        "AUC"      : round(auc, 4),
        "CV_Mean"  : round(cv_scores.mean() * 100, 2),
        "CV_Std"   : round(cv_scores.std() * 100, 2)
    },
    "<----XGBoost---->": {
        "Accuracy" : round(acc_xgb * 100, 2),
        "MCC"      : round(mcc_xgb, 4),
        "AUC"      : round(auc_xgb, 4)
    },
    "<----CatBoost---->": {
        "Accuracy" : round(acc_cat * 100, 2),
        "MCC"      : round(mcc_cat, 4),
        "AUC"      : round(auc_cat, 4)
    },
    "<----LightGBM---->": {
        "Accuracy" : round(acc_light * 100, 2),
        "MCC"      : round(mcc_light, 4),
        "AUC"      : round(auc_light, 4)
    },
    "<----SVM---->": {
        "Accuracy" : round(acc_svm * 100, 2),
        "MCC"      : round(mcc_svm, 4),
        "AUC"      : round(auc_svm, 4)
    },
    "<----ESM + RF---->": {
        "Accuracy" : round(acc_esm_c * 100, 2),
        "MCC"      : round(mcc_esm_c, 4),
        "AUC"      : round(auc_esm_c, 4)
    },
    "<----ESM + SVM---->": {
        "Accuracy" : round(acc_esm_svm * 100, 2),
        "MCC"      : round(mcc_esm_svm, 4),
        "AUC"      : round(auc_esm_svm, 4)
    },
    "<----Hyperparameter Tuned RF---->": {
        "Accuracy" : round(acc_best * 100, 2),
        "MCC"      : round(mcc_best, 4),
        "AUC"      : round(auc_best, 4)
    },
    "<----Naive Bayes---->": {
        "Accuracy" : round(acc_nb * 100, 2),
        "MCC"      : round(mcc_nb, 4),
        "AUC"      : round(auc_nb, 4)
    },
    "<----KNN---->": {
        "Accuracy" : round(acc_knn * 100, 2),
        "MCC"      : round(mcc_knn, 4),
        "AUC"      : round(auc_knn, 4)
    },
    "<----ESM + KNN---->": {
        "Accuracy" : round(acc_esm_knn * 100, 2),
        "MCC"      : round(mcc_esm_knn, 4),
        "AUC"      : round(auc_esm_knn, 4)
    },
    "<----ESM + GridSearchCV---->": {
        "Accuracy" : round(acc_esm_grid * 100, 2),
        "MCC"      : round(mcc_esm_grid, 4),
        "AUC"      : round(auc_esm_grid, 4)
    },
    "<----Ensemble (RF+XGB+LGBM)---->": {
        "Accuracy" : round(acc_ens * 100, 2),
        "MCC"      : round(mcc_ens, 4),
        "AUC"      : round(auc_ens, 4)
    },
    "<----Feature Selection---->": {
        "Accuracy" : round(acc_fs * 100, 2),
        "MCC"      : round(mcc_fs, 4),
        "AUC"      : round(auc_fs, 4)
    },
    "<----Neural Network---->": {
    "Accuracy": round(acc_nn * 100, 2),
    "MCC"     : round(mcc_nn, 4),
    "AUC"     : round(auc_nn, 4)
    },
    "<----Stacking (RF+XGB+LGBM)---->": {
    "Accuracy": round(acc_stack * 100, 2),
    "MCC"     : round(mcc_stack, 4),
    "AUC"     : round(auc_stack, 4)
    },
    "<----Ensemble + Feature Selection---->": {
    "Accuracy": round(acc_ens_fs * 100, 2),
    "MCC"     : round(mcc_ens_fs, 4),
    "AUC"     : round(auc_ens_fs, 4)
    },
    "<----Logistic Regression---->": {
    "Accuracy": round(acc_lr * 100, 2),
    "MCC"     : round(mcc_lr, 4),
    "AUC"     : round(auc_lr, 4)
    },
    "<----Stacking + ESM---->": {
    "Accuracy": round(acc_stack_esm * 100, 2),
    "MCC"     : round(mcc_stack_esm, 4),
    "AUC"     : round(auc_stack_esm, 4)
    },
    "<----Stacking(RF+XGB+LGBM+CAT+SVM)": {
    "Acurracy": round(acc_stack_v2 * 100, 2),
    "MCC"     : round(mcc_stack_v2, 4),
    "AUC"     : round(auc_stack_v2, 4)
    },
    "<----Stacking (RF+XGB+LGBM+CatBoost+SVM)---->": {
    "Accuracy": round(acc_stack_v2 * 100, 2),
    "MCC"     : round(mcc_stack_v2, 4),
    "AUC"     : round(auc_stack_v2, 4)
    },
    "<----Two-Level Stacking (RF+XGB+LGBM+CatBoost -> LogReg -> LogReg)---->": {
    "Accuracy": round(acc_deep * 100, 2),
    "MCC"     : round(mcc_deep, 4),
    "AUC"     : round(auc_deep, 4)
    },
    "<----Weighted Voting (RF+XGB+LGBM)---->": {
    "Accuracy": round(acc_weighted * 100, 2),
    "MCC"     : round(mcc_weighted, 4),
    "AUC"     : round(auc_weighted, 4)
    }
}

if os.path.exists("results.json"):
    with open("results.json", "r") as f:
        try:
            all_results = json.load(f)
            if not isinstance(all_results, list):
                all_results = []
        except json.JSONDecodeError:
            all_results = []
else:
    all_results = []

all_results.append(new_result)

with open("results.json", "w") as f:
    json.dump(all_results, f, indent=4)









models = [
    'Ensemble', 'Random\nForest', 'Tuned\nRF', 'Feat\nSelect',
    'LightGBM', 'ESM+RF', 'XGBoost', 'CatBoost',
    'Neural\nNet', 'ESM+SVM', 'KNN+ESM', 'KNN', 'SVM', 'Naive\nBayes'
]

accuracies = [85.94, 85.62, 84.99, 85.15, 84.83, 84.52, 84.12, 82.54, 78.75, 75.28, 74.96, 70.77, 69.83, 69.35]
mccs       = [0.6361, 0.6273, 0.6088, 0.6177, 0.6068, 0.6045, 0.5917, 0.5536, 0.5317, 0.4734, 0.4875, 0.4120, 0.3304, 0.3905]
aucs       = [0.9075, 0.8999, 0.9043, 0.9012, 0.9020, 0.8922, 0.8958, 0.8791, 0.8487, 0.8315, 0.8380, 0.7938, 0.7285, 0.7364]

x = np.arange(len(models))
fig, axes = plt.subplots(1, 3, figsize=(24, 6))
fig.suptitle("Model Comparison — Methylation Prediction", fontsize=14, fontweight='bold')

# Accuracy
axes[0].bar(x, accuracies, width=0.5, color='steelblue', edgecolor='black')
axes[0].set_title("Accuracy (%)")
axes[0].set_xticks(x)
axes[0].set_xticklabels(models, fontsize=6)
axes[0].set_ylim(60, 92)
axes[0].axhline(y=85.94, color='red', linestyle='--', label='Ensemble baseline')
for i, v in enumerate(accuracies):
    axes[0].text(i, v + 0.2, str(v), ha='center', fontsize=5)

# MCC
axes[1].bar(x, mccs, width=0.5, color='seagreen', edgecolor='black')
axes[1].set_title("MCC")
axes[1].set_xticks(x)
axes[1].set_xticklabels(models, fontsize=6)
axes[1].set_ylim(0, 0.8)
axes[1].axhline(y=0.6361, color='red', linestyle='--', label='Ensemble baseline')
for i, v in enumerate(mccs):
    axes[1].text(i, v + 0.01, str(v), ha='center', fontsize=5)

# AUC
axes[2].bar(x, aucs, width=0.5, color='darkorange', edgecolor='black')
axes[2].set_title("AUC")
axes[2].set_xticks(x)
axes[2].set_xticklabels(models, fontsize=6)
axes[2].set_ylim(0.6, 1.0)
axes[2].axhline(y=0.9075, color='red', linestyle='--', label='Ensemble baseline')
for i, v in enumerate(aucs):
    axes[2].text(i, v + 0.005, str(v), ha='center', fontsize=5)

plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150, bbox_inches='tight')
plt.close()










# PDF SAVE

pdf = canvas.Canvas("results.pdf", pagesize=A4)
width, height = A4

# Title
pdf.setFont("Helvetica-Bold", 16)
pdf.drawString(150, height - 50, "Methylation Prediction Results")

# Date
pdf.setFont("Helvetica", 11)
pdf.drawString(50, height - 80, f"Date & Time: {date_time}")
pdf.line(50, height - 90, 550, height - 90)

# ─── RANDOM FOREST ───
pdf.setFont("Helvetica-Bold", 12)
pdf.drawString(50, height - 115, "Classification Report Of Random Forest:")
pdf.setFont("Courier-Bold", 10)
pdf.drawString(50, height - 135, f"{'Metric':<20} {'Non-Methylated':>18} {'Methylated':>15}")
pdf.line(50, height - 145, 550, height - 145)
pdf.setFont("Courier", 10)
pdf.drawString(50, height - 160, f"{'Precision':<20} {precision_nm:>18.4f} {precision_m:>15.4f}")
pdf.drawString(50, height - 175, f"{'Recall':<20} {recall_nm:>18.4f} {recall_m:>15.4f}")
pdf.drawString(50, height - 190, f"{'F1-Score':<20} {f1_nm:>18.4f} {f1_m:>15.4f}")
pdf.drawString(50, height - 205, f"{'Support':<20} {support_nm:>18} {support_m:>15}")
pdf.line(50, height - 215, 550, height - 215)
pdf.drawString(50, height - 230, f"{'Accuracy':<20} {round(acc * 100, 2):>17}%")
pdf.drawString(50, height - 245, f"{'MCC':<20} {round(mcc, 4):>18}")
pdf.drawString(50, height - 260, f"{'AUC':<20} {round(auc, 4):>18}")
pdf.line(50, height - 270, 550, height - 270)

# ─── XGBOOST ───
pdf.setFont("Helvetica-Bold", 12)
pdf.drawString(50, height - 295, "Classification Report Of XGBoost:")
pdf.setFont("Courier-Bold", 10)
pdf.drawString(50, height - 315, f"{'Metric':<20} {'Score':>10}")
pdf.line(50, height - 325, 550, height - 325)
pdf.setFont("Courier", 10)
pdf.drawString(50, height - 340, f"{'Accuracy':<20} {round(acc_xgb * 100, 2):>9}%")
pdf.drawString(50, height - 355, f"{'MCC':<20} {round(mcc_xgb, 4):>10}")
pdf.drawString(50, height - 370, f"{'AUC':<20} {round(auc_xgb, 4):>10}")
pdf.line(50, height - 380, 550, height - 380)

# ─── CATBOOST ───
pdf.setFont("Helvetica-Bold", 12)
pdf.drawString(50, height - 405, "Classification Report Of CatBoost:")
pdf.setFont("Courier-Bold", 10)
pdf.drawString(50, height - 425, f"{'Metric':<20} {'Score':>10}")
pdf.line(50, height - 435, 550, height - 435)
pdf.setFont("Courier", 10)
pdf.drawString(50, height - 450, f"{'Accuracy':<20} {round(acc_cat * 100, 2):>9}%")
pdf.drawString(50, height - 465, f"{'MCC':<20} {round(mcc_cat, 4):>10}")
pdf.drawString(50, height - 480, f"{'AUC':<20} {round(auc_cat, 4):>10}")
pdf.line(50, height - 490, 550, height - 490)

# ─── LIGHTGBM ───
pdf.setFont("Helvetica-Bold", 12)
pdf.drawString(50, height - 515, "Classification Report Of LightGBM:")
pdf.setFont("Courier-Bold", 10)
pdf.drawString(50, height - 535, f"{'Metric':<20} {'Score':>10}")
pdf.line(50, height - 545, 550, height - 545)
pdf.setFont("Courier", 10)
pdf.drawString(50, height - 560, f"{'Accuracy':<20} {round(acc_light * 100, 2):>9}%")
pdf.drawString(50, height - 575, f"{'MCC':<20} {round(mcc_light, 4):>10}")
pdf.drawString(50, height - 590, f"{'AUC':<20} {round(auc_light, 4):>10}")
pdf.line(50, height - 600, 550, height - 600)

# ─── SVM ───
pdf.setFont("Helvetica-Bold", 12)
pdf.drawString(50, height - 625, "Classification Report Of SVM:")
pdf.setFont("Courier-Bold", 10)
pdf.drawString(50, height - 645, f"{'Metric':<20} {'Score':>10}")
pdf.line(50, height - 655, 550, height - 655)
pdf.setFont("Courier", 10)
pdf.drawString(50, height - 670, f"{'Accuracy':<20} {round(acc_svm * 100, 2):>9}%")
pdf.drawString(50, height - 685, f"{'MCC':<20} {round(mcc_svm, 4):>10}")
pdf.drawString(50, height - 700, f"{'AUC':<20} {round(auc_svm, 4):>10}")
pdf.line(50, height - 710, 550, height - 710)

# ─── ESM + RF ───
pdf.setFont("Helvetica-Bold", 12)
pdf.drawString(50, height - 740, "Classification Report Of ESM + RF:")
pdf.setFont("Courier-Bold", 10)
pdf.drawString(50, height - 760, f"{'Metric':<20} {'Score':>10}")
pdf.line(50, height - 770, 550, height - 770)
pdf.setFont("Courier", 10)
pdf.drawString(50, height - 785, f"{'Accuracy':<20} {round(acc_esm_c * 100, 2):>9}%")
pdf.drawString(50, height - 800, f"{'MCC':<20} {round(mcc_esm_c, 4):>10}")
pdf.drawString(50, height - 815, f"{'AUC':<20} {round(auc_esm_c, 4):>10}")
pdf.line(50, height - 825, 550, height - 825)

# ─── NEW PAGE ───
pdf.showPage()

# ─── ESM + SVM ───
pdf.setFont("Helvetica-Bold", 12)
pdf.drawString(50, height - 50, "Classification Report Of ESM + SVM:")
pdf.setFont("Courier-Bold", 10)
pdf.drawString(50, height - 70, f"{'Metric':<20} {'Score':>10}")
pdf.line(50, height - 80, 550, height - 80)
pdf.setFont("Courier", 10)
pdf.drawString(50, height - 95, f"{'Accuracy':<20} {round(acc_esm_svm * 100, 2):>9}%")
pdf.drawString(50, height - 110, f"{'MCC':<20} {round(mcc_esm_svm, 4):>10}")
pdf.drawString(50, height - 125, f"{'AUC':<20} {round(auc_esm_svm, 4):>10}")
pdf.line(50, height - 135, 550, height - 135)

# ─── HYPERPARAMETER TUNED RF ───
pdf.setFont("Helvetica-Bold", 12)
pdf.drawString(50, height - 160, "Classification Report Of Hyperparameter Tuned RF:")
pdf.setFont("Courier-Bold", 10)
pdf.drawString(50, height - 180, f"{'Metric':<20} {'Score':>10}")
pdf.line(50, height - 190, 550, height - 190)
pdf.setFont("Courier", 10)
pdf.drawString(50, height - 205, f"{'Accuracy':<20} {round(acc_best * 100, 2):>9}%")
pdf.drawString(50, height - 220, f"{'MCC':<20} {round(mcc_best, 4):>10}")
pdf.drawString(50, height - 235, f"{'AUC':<20} {round(auc_best, 4):>10}")
pdf.line(50, height - 245, 550, height - 245)



# ─── NEURAL NETWORK ───
pdf.setFont("Helvetica-Bold", 12)
pdf.drawString(50, height - 270, "Classification Report Of Neural Network:")
pdf.setFont("Courier-Bold", 10)
pdf.drawString(50, height - 290, f"{'Metric':<20} {'Score':>10}")
pdf.line(50, height - 300, 550, height - 300)
pdf.setFont("Courier", 10)
pdf.drawString(50, height - 315, f"{'Accuracy':<20} {round(acc_nn * 100, 2):>9}%")
pdf.drawString(50, height - 330, f"{'MCC':<20} {round(mcc_nn, 4):>10}")
pdf.drawString(50, height - 345, f"{'AUC':<20} {round(auc_nn, 4):>10}")
pdf.line(50, height - 355, 550, height - 355)

# ─── NAIVE BAYES ───
pdf.setFont("Helvetica-Bold", 12)
pdf.drawString(50, height - 380, "Classification Report Of Naive Bayes:")
pdf.setFont("Courier-Bold", 10)
pdf.drawString(50, height - 400, f"{'Metric':<20} {'Score':>10}")
pdf.line(50, height - 410, 550, height - 410)
pdf.setFont("Courier", 10)
pdf.drawString(50, height - 425, f"{'Accuracy':<20} {round(acc_nb * 100, 2):>9}%")
pdf.drawString(50, height - 440, f"{'MCC':<20} {round(mcc_nb, 4):>10}")
pdf.drawString(50, height - 455, f"{'AUC':<20} {round(auc_nb, 4):>10}")
pdf.line(50, height - 465, 550, height - 465)

# ─── KNN ───
pdf.setFont("Helvetica-Bold", 12)
pdf.drawString(50, height - 490, "Classification Report Of KNN:")
pdf.setFont("Courier-Bold", 10)
pdf.drawString(50, height - 510, f"{'Metric':<20} {'Score':>10}")
pdf.line(50, height - 520, 550, height - 520)
pdf.setFont("Courier", 10)
pdf.drawString(50, height - 535, f"{'Accuracy':<20} {round(acc_knn * 100, 2):>9}%")
pdf.drawString(50, height - 550, f"{'MCC':<20} {round(mcc_knn, 4):>10}")
pdf.drawString(50, height - 565, f"{'AUC':<20} {round(auc_knn, 4):>10}")
pdf.line(50, height - 575, 550, height - 575)

# ─── ESM + KNN ───
pdf.setFont("Helvetica-Bold", 12)
pdf.drawString(50, height - 610, "Classification Report Of ESM + KNN:")
pdf.setFont("Courier-Bold", 10)
pdf.drawString(50, height - 630, f"{'Metric':<20} {'Score':>10}")
pdf.line(50, height - 640, 550, height - 640)
pdf.setFont("Courier", 10)
pdf.drawString(50, height - 655, f"{'Accuracy':<20} {round(acc_esm_knn * 100, 2):>9}%")
pdf.drawString(50, height - 670, f"{'MCC':<20} {round(mcc_esm_knn, 4):>10}")
pdf.drawString(50, height - 685, f"{'AUC':<20} {round(auc_esm_knn, 4):>10}")
pdf.line(50, height - 695, 550, height - 695)


pdf.showPage() 

y_start = height - 50 

# ─── ESM + GridSearchCV (Tuned RF) ───
pdf.setFont("Helvetica-Bold", 12)
pdf.drawString(50, y_start, "Classification Report Of ESM + Tuned RF (GridSearchCV):")
pdf.setFont("Courier-Bold", 10)
pdf.drawString(50, y_start - 20, f"{'Metric':<20} {'Score':>10}")
pdf.line(50, y_start - 30, 550, y_start - 30)
pdf.setFont("Courier", 10)
pdf.drawString(50, y_start - 45, f"{'Accuracy':<20} {round(acc_esm_grid * 100, 2):>9}%")
pdf.drawString(50, y_start - 60, f"{'MCC':<20} {round(mcc_esm_grid, 4):>10}")
pdf.drawString(50, y_start - 75, f"{'AUC':<20} {round(auc_esm_grid, 4):>10}")

pdf.line(50, y_start - 85, 550, y_start - 85)

# ─── ESM + Feature Selection ───

fs_y = y_start - 170 

pdf.setFont("Helvetica-Bold", 12)
pdf.drawString(50, fs_y, "Classification Report Of ESM + Feature Selection:")
pdf.setFont("Courier-Bold", 10)
pdf.drawString(50, fs_y - 40, f"{'Metric':<20} {'Score':>10}")
pdf.line(50, fs_y - 50, 550, fs_y - 50)
pdf.setFont("Courier", 10)
pdf.drawString(50, fs_y - 65, f"{'Accuracy':<20} {round(acc_fs * 100, 2):>9}%")
pdf.drawString(50, fs_y - 80, f"{'MCC':<20} {round(mcc_fs, 4):>10}")
pdf.drawString(50, fs_y - 95, f"{'AUC':<20} {round(auc_fs, 4):>10}")
pdf.line(50, fs_y - 105, 550, fs_y - 105)

# ─── ENSEMBLE ───
ens_y = fs_y - 130

pdf.setFont("Helvetica-Bold", 12)
pdf.drawString(50, ens_y, "Classification Report Of Ensemble (RF+XGB+LGBM):")
pdf.setFont("Courier-Bold", 10)
pdf.drawString(50, ens_y - 20, f"{'Metric':<25} {'Score':>15}")
pdf.line(50, ens_y - 30, 550, ens_y - 30)
pdf.setFont("Courier", 10)
pdf.drawString(50, ens_y - 45, f"{'Accuracy':<25} {round(acc_ens * 100, 2):>14}%")
pdf.drawString(50, ens_y - 60, f"{'MCC':<25} {round(mcc_ens, 4):>15}")
pdf.drawString(50, ens_y - 75, f"{'AUC':<25} {round(auc_ens, 4):>15}")
pdf.line(50, ens_y - 85, 550, ens_y - 85)

pdf.showPage()

# ─── STACKING ───
pdf.setFont("Helvetica-Bold", 12)
pdf.drawString(50, height - 50, "Classification Report Of Stacking (RF+XGB+LGBM):")
pdf.setFont("Courier-Bold", 10)
pdf.drawString(50, height - 70, f"{'Metric':<25} {'Score':>15}")
pdf.line(50, height - 80, 550, height - 80)
pdf.setFont("Courier", 10)
pdf.drawString(50, height - 95, f"{'Accuracy':<25} {round(acc_stack * 100, 2):>14}%")
pdf.drawString(50, height - 110, f"{'MCC':<25} {round(mcc_stack, 4):>15}")
pdf.drawString(50, height - 125, f"{'AUC':<25} {round(auc_stack, 4):>15}")
pdf.line(50, height - 135, 550, height - 135)

# ─── ENSEMBLE + FS ───
pdf.setFont("Helvetica-Bold", 12)
pdf.drawString(50, height - 160, "Classification Report Of Ensemble + Feature Selection:")
pdf.setFont("Courier-Bold", 10)
pdf.drawString(50, height - 180, f"{'Metric':<25} {'Score':>15}")
pdf.line(50, height - 190, 550, height - 190)
pdf.setFont("Courier", 10)
pdf.drawString(50, height - 205, f"{'Accuracy':<25} {round(acc_ens_fs * 100, 2):>14}%")
pdf.drawString(50, height - 220, f"{'MCC':<25} {round(mcc_ens_fs, 4):>15}")
pdf.drawString(50, height - 235, f"{'AUC':<25} {round(auc_ens_fs, 4):>15}")
pdf.line(50, height - 245, 550, height - 245)

# ─── LOGISTIC REGRESSION ───
pdf.setFont("Helvetica-Bold", 12)
pdf.drawString(50, height - 270, "Classification Report Of Logistic Regression:")
pdf.setFont("Courier-Bold", 10)
pdf.drawString(50, height - 290, f"{'Metric':<25} {'Score':>15}")
pdf.line(50, height - 300, 550, height - 300)
pdf.setFont("Courier", 10)
pdf.drawString(50, height - 315, f"{'Accuracy':<25} {round(acc_lr * 100, 2):>14}%")
pdf.drawString(50, height - 330, f"{'MCC':<25} {round(mcc_lr, 4):>15}")
pdf.drawString(50, height - 345, f"{'AUC':<25} {round(auc_lr, 4):>15}")
pdf.line(50, height - 355, 550, height - 355)

# ─── STACKING + ESM ───
pdf.setFont("Helvetica-Bold", 12)
pdf.drawString(50, height - 380, "Classification Report Of Stacking + ESM:")
pdf.setFont("Courier-Bold", 10)
pdf.drawString(50, height - 400, f"{'Metric':<25} {'Score':>15}")
pdf.line(50, height - 410, 550, height - 410)
pdf.setFont("Courier", 10)
pdf.drawString(50, height - 425, f"{'Accuracy':<25} {round(acc_stack_esm * 100, 2):>14}%")
pdf.drawString(50, height - 440, f"{'MCC':<25} {round(mcc_stack_esm, 4):>15}")
pdf.drawString(50, height - 455, f"{'AUC':<25} {round(auc_stack_esm, 4):>15}")
pdf.line(50, height - 465, 550, height - 465)

# New page
pdf.showPage()

# Title
pdf.setFont("Helvetica-Bold", 14)
pdf.drawString(150, height - 50, "Model Comparison Table")
pdf.line(50, height - 60, 550, height - 60)

# Header
pdf.setFont("Courier-Bold", 9)
pdf.drawString(50, height - 85, f"{'Model':<25} {'Accuracy':>10} {'MCC':>10} {'AUC':>10}")
pdf.line(50, height - 93, 550, height - 93)

# Data rows
pdf.setFont("Courier", 9)
comparison_data = [
    ("Ensemble RF+XGB+LGBM", 85.94, 0.6361, 0.9075),
    ("Random Forest",         85.62, 0.6273, 0.8999),
    ("Tuned RF",              84.99, 0.6088, 0.9043),
    ("Feature Selection",     85.15, 0.6177, 0.9012),
    ("LightGBM",              84.83, 0.6068, 0.9020),
    ("ESM + RF",              84.52, 0.6045, 0.8922),
    ("XGBoost",               84.12, 0.5917, 0.8958),
    ("CatBoost",              82.54, 0.5536, 0.8791),
    ("Neural Network",        78.75, 0.5317, 0.8487),
    ("ESM + SVM",             75.28, 0.4734, 0.8315),
    ("KNN + ESM",             74.96, 0.4875, 0.8380),
    ("KNN",                   70.77, 0.4120, 0.7938),
    ("SVM",                   69.83, 0.3304, 0.7285),
    ("Naive Bayes",           69.35, 0.3905, 0.7364),
]

y_pos = height - 110
for name, acc_v, mcc_v, auc_v in comparison_data:
    pdf.drawString(50, y_pos, f"{name:<25} {acc_v:>9}% {mcc_v:>10} {auc_v:>10}")
    y_pos -= 20

pdf.line(50, y_pos, 550, y_pos)

# Histogram image
from reportlab.lib.utils import ImageReader
pdf.drawImage("model_comparison.png", 50, y_pos - 260, width=500, height=240)


# ─── SAVE ───
pdf.save()



