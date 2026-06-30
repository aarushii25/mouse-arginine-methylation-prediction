# Data Augmentation Method
import os
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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
print(f"Original sequences: {len(df)}")

# ═══════════════════════════════════════
#   DATA AUGMENTATION TECHNIQUE
# ═══════════════════════════════════════

amino_acids = list("ACDEFGHIKLMNPQRSTVWY")

# Conservative amino acid substitutions
# (similar properties wale amino acids — biology mein meaningful hai)
similar_aa = {
    'A': ['G', 'S'],
    'C': ['S'],
    'D': ['E', 'N'],
    'E': ['D', 'Q'],
    'F': ['Y', 'W'],
    'G': ['A', 'S'],
    'H': ['N', 'Q'],
    'I': ['L', 'V'],
    'K': ['R'],
    'L': ['I', 'V'],
    'M': ['L', 'I'],
    'N': ['D', 'Q'],
    'P': ['A'],
    'Q': ['N', 'E'],
    'R': ['K'],
    'S': ['T', 'A'],
    'T': ['S'],
    'V': ['I', 'L'],
    'W': ['F', 'Y'],
    'Y': ['F', 'W']
}

def augment_sequence(sequence, num_mutations=1):
    """
    Sequence mein 1-2 amino acids ko similar amino acid se replace karo
    Center position (R - jo methylate hoti hai) ko mat chhuo!
    """
    seq_list = list(str(sequence).upper())
    center = len(seq_list) // 2  # center position skip karo

    positions = [i for i in range(len(seq_list)) if i != center]
    if len(positions) == 0:
        return sequence

    mutation_positions = random.sample(positions, min(num_mutations, len(positions)))

    for pos in mutation_positions:
        original_aa = seq_list[pos]
        if original_aa in similar_aa:
            seq_list[pos] = random.choice(similar_aa[original_aa])

    return ''.join(seq_list)

# Augmentation karo — sirf METHYLATED (minority class) ke liye
print("\nAugmenting minority class (Methylated)...")

methylated_df = df[df["label"] == 1].copy()
augmented_rows = []

random.seed(42)
for idx, row in methylated_df.iterrows():
    original_seq = row["sequence"]
    # Har sequence se 2 augmented versions banao
    for _ in range(2):
        aug_seq = augment_sequence(original_seq, num_mutations=2)
        augmented_rows.append({"sequence": aug_seq, "label": 1})

augmented_df = pd.DataFrame(augmented_rows)
print(f"Original methylated sequences: {len(methylated_df)}")
print(f"Augmented sequences created  : {len(augmented_df)}")

# Original data + Augmented data combine karo
df_combined = pd.concat([df[["sequence", "label"]], augmented_df], ignore_index=True)
print(f"\nTotal sequences after augmentation: {len(df_combined)}")
print(df_combined["label"].value_counts())

# ═══════════════════════════════════════
#        FEATURE EXTRACTION
# ═══════════════════════════════════════

def get_aac(sequence):
    sequence = str(sequence).upper()
    length = len(sequence)
    aac = []
    for aa in amino_acids:
        count = sequence.count(aa)
        aac.append(count / length if length > 0 else 0)
    return aac

aac_features = df_combined["sequence"].apply(get_aac)
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

dpc_features = df_combined["sequence"].apply(get_dpc)
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

entropy_features = df_combined["sequence"].apply(get_entropy_features)
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

phys_features = df_combined["sequence"].apply(get_physicochemical)
X_phys = np.array(phys_features.tolist())

# ═══════════════════════════════════════
#        COMBINE + TRAIN
# ═══════════════════════════════════════

X = np.hstack([X_aac, X_dpc, X_entropy, X_phys])
y = df_combined["label"].values

print(f"\nFinal Feature Shape: {X.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Halki SMOTE bhi karte hain (agar still imbalance ho)
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print(f"Before SMOTE: {y_train.sum()} methylated vs {len(y_train) - y_train.sum()} non-methylated")
print(f"After SMOTE : {y_train_sm.sum()} methylated vs {len(y_train_sm) - y_train_sm.sum()} non-methylated")

# Random Forest train karo
print("\n------Random Forest on Augmented Data-----\n")
model_aug = RandomForestClassifier(n_estimators=100, random_state=42)
model_aug.fit(X_train_sm, y_train_sm)
print("✅ Training Complete!")

y_pred_aug = model_aug.predict(X_test)
y_prob_aug = model_aug.predict_proba(X_test)[:, 1]

acc_aug = accuracy_score(y_test, y_pred_aug)
mcc_aug = matthews_corrcoef(y_test, y_pred_aug)
auc_aug = roc_auc_score(y_test, y_prob_aug)

print(f"Accuracy : {round(acc_aug * 100, 2)}%")
print(f"MCC      : {round(mcc_aug, 4)}")
print(f"AUC      : {round(auc_aug, 4)}")

print("\n" + "="*50)
print("COMPARISON")
print("="*50)
print(f"Data Augmentation + RF : Acc={round(acc_aug*100,2)}%  MCC={round(mcc_aug,4)}  AUC={round(auc_aug,4)}")
print(f"Previous Best (Ens)    : Acc=85.94%  MCC=0.6361  AUC=0.9075")