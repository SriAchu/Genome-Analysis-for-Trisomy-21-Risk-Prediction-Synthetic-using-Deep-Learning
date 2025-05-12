import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import os
import random

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Print TensorFlow version for debugging
print(f"TensorFlow version: {tf.__version__}")

# Load dataset
df = pd.read_csv("synthetic_t21_rnaseq.csv", index_col=0)

# Extract labels
labels = df.loc["Label"].values
y = (labels == "T21").astype(int)

# Remove Label row
df = df.drop("Label", axis=0)

# Convert to float
df = df.astype(float)

# Check for NaN
if df.isna().any().any():
    print("Warning: Dataset contains NaN values. Filling with 0.")
    df = df.fillna(0)

# Feature selection (t-test for DEGs)
from scipy.stats import ttest_ind
t21_samples = df.columns[labels == "T21"]
control_samples = df.columns[labels == "Control"]
p_values = []
for gene in df.index:
    try:
        p_val = ttest_ind(df.loc[gene, t21_samples], df.loc[gene, control_samples])[1]
        p_values.append(p_val if not np.isnan(p_val) else 1.0)
    except Exception as e:
        print(f"Error processing gene {gene}: {e}")
        p_values.append(1.0)
p_values = np.array(p_values)
top_n = 50  # 50 genes as per your choice
deg_indices = np.argsort(p_values)[:top_n]
deg_genes = df.index[deg_indices]
df = df.loc[deg_genes]
print(f"Selected genes: {len(deg_genes)}")

# Validate gene names
if any(gene == "0" or not isinstance(gene, str) for gene in df.index):
    raise ValueError(f"Invalid gene names detected in df.index: {df.index.tolist()}")

# Verify T21 genes
t21_genes = ["DYRK1A", "APP", "MX1", "IFITM1", "STAT1"]
available_t21_genes = [g for g in t21_genes if g in df.index]
print(f"T21 genes available: {available_t21_genes}")

# Save genes for GUI
pd.Series(df.index).to_csv("common_genes.csv", index=False)
# Debug: Confirm the number of genes saved
saved_genes = pd.read_csv("common_genes.csv", header=None)[0].tolist()
print(f"Genes saved to common_genes.csv: {len(saved_genes)}")

# Prepare data
X = df.T.values
scaler = StandardScaler()
X = scaler.fit_transform(X)
print(f"Input shape: {X.shape}")

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
mlp_accuracies, mlp_aucs = [], []
cnn_accuracies, cnn_aucs = [], []
rnn_accuracies, rnn_aucs = [], []
ensemble_accuracies, ensemble_aucs = [], []

for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
    print(f"\nFold {fold + 1}/5")
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Check for NaN
    if np.any(np.isnan(X_train)) or np.any(np.isnan(X_test)):
        print("Warning: NaN in train/test data. Filling with 0.")
        X_train = np.nan_to_num(X_train, nan=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0)

    # MLP
    mlp_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X.shape[1],)),
        tf.keras.layers.Dense(12, activation="relu"),
        tf.keras.layers.Dropout(0.65),
        tf.keras.layers.Dense(6, activation="relu"),
        tf.keras.layers.Dropout(0.65),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    mlp_model.compile(optimizer=tf.keras.optimizers.Adam(clipnorm=1.0), loss="binary_crossentropy", metrics=["accuracy"])
    mlp_model.fit(X_train, y_train, epochs=200, batch_size=16, validation_split=0.2, verbose=0,
                  callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)])
    mlp_probs = mlp_model.predict(X_test, verbose=0)
    mlp_pred = (mlp_probs > 0.5).astype(int)
    mlp_acc = accuracy_score(y_test, mlp_pred)
    mlp_auc = roc_auc_score(y_test, mlp_probs)
    mlp_accuracies.append(mlp_acc)
    mlp_aucs.append(mlp_auc)

    # CNN (Simplified to avoid serialization issues - no Conv1D)
    X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    cnn_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1], 1)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(12, activation="relu"),
        tf.keras.layers.Dropout(0.65),
        tf.keras.layers.Dense(6, activation="relu"),
        tf.keras.layers.Dropout(0.65),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    cnn_model.compile(optimizer=tf.keras.optimizers.Adam(clipnorm=1.0), loss="binary_crossentropy", metrics=["accuracy"])
    cnn_model.fit(X_train_cnn, y_train, epochs=200, batch_size=16, validation_split=0.2, verbose=0,
                  callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)])
    cnn_probs = cnn_model.predict(X_test_cnn, verbose=0)
    if np.any(np.isnan(cnn_probs)):
        print("Warning: CNN predictions contain NaN. Replacing with 0.5.")
        cnn_probs = np.nan_to_num(cnn_probs, nan=0.5)
    cnn_pred = (cnn_probs > 0.5).astype(int)
    cnn_acc = accuracy_score(y_test, cnn_pred)
    cnn_auc = roc_auc_score(y_test, cnn_probs)
    cnn_accuracies.append(cnn_acc)
    cnn_aucs.append(cnn_auc)

    # RNN
    X_train_rnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_rnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    rnn_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1], 1)),
        tf.keras.layers.LSTM(4, return_sequences=False),
        tf.keras.layers.Dense(4, activation="relu"),
        tf.keras.layers.Dropout(0.65),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    rnn_model.compile(optimizer=tf.keras.optimizers.Adam(clipnorm=1.0), loss="binary_crossentropy", metrics=["accuracy"])
    rnn_model.fit(X_train_rnn, y_train, epochs=200, batch_size=16, validation_split=0.2, verbose=0,
                  callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)])
    rnn_probs = rnn_model.predict(X_test_rnn, verbose=0)
    if np.any(np.isnan(rnn_probs)):
        print("Warning: RNN predictions contain NaN. Replacing with 0.5.")
        rnn_probs = np.nan_to_num(rnn_probs, nan=0.5)
    rnn_pred = (rnn_probs > 0.5).astype(int)
    rnn_acc = accuracy_score(y_test, rnn_pred)
    rnn_auc = roc_auc_score(y_test, rnn_probs)
    rnn_accuracies.append(rnn_acc)
    rnn_aucs.append(rnn_auc)

    # Ensemble
    ensemble_probs = (mlp_probs + cnn_probs + rnn_probs) / 3
    ensemble_pred = (ensemble_probs > 0.5).astype(int)
    ensemble_acc = accuracy_score(y_test, ensemble_pred)
    ensemble_auc = roc_auc_score(y_test, ensemble_probs)
    ensemble_accuracies.append(ensemble_acc)
    ensemble_aucs.append(ensemble_auc)

# Compute mean and std of metrics
print("\nCross-Validation Results (Mean ± Std):")
print(f"MLP Accuracy: {np.mean(mlp_accuracies):.3f} ± {np.std(mlp_accuracies):.3f}, AUC: {np.mean(mlp_aucs):.3f} ± {np.std(mlp_aucs):.3f}")
print(f"CNN Accuracy: {np.mean(cnn_accuracies):.3f} ± {np.std(cnn_accuracies):.3f}, AUC: {np.mean(cnn_aucs):.3f} ± {np.std(cnn_aucs):.3f}")
print(f"RNN Accuracy: {np.mean(rnn_accuracies):.3f} ± {np.std(rnn_accuracies):.3f}, AUC: {np.mean(rnn_aucs):.3f} ± {np.std(rnn_aucs):.3f}")
print(f"Ensemble Accuracy: {np.mean(ensemble_accuracies):.3f} ± {np.std(ensemble_accuracies):.3f}, AUC: {np.mean(ensemble_aucs):.3f} ± {np.std(ensemble_aucs):.3f}")

# Train final models on full dataset
X_scaled = scaler.fit_transform(X)

mlp_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X.shape[1],)),
    tf.keras.layers.Dense(12, activation="relu"),
    tf.keras.layers.Dropout(0.65),
    tf.keras.layers.Dense(6, activation="relu"),
    tf.keras.layers.Dropout(0.65),
    tf.keras.layers.Dense(1, activation="sigmoid")
])
mlp_model.compile(optimizer=tf.keras.optimizers.Adam(clipnorm=1.0), loss="binary_crossentropy", metrics=["accuracy"])
mlp_model.fit(X_scaled, y, epochs=200, batch_size=16, validation_split=0.2, verbose=0,
              callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)])

cnn_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X.shape[1], 1)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(12, activation="relu"),
    tf.keras.layers.Dropout(0.65),
    tf.keras.layers.Dense(6, activation="relu"),
    tf.keras.layers.Dropout(0.65),
    tf.keras.layers.Dense(1, activation="sigmoid")
])
cnn_model.compile(optimizer=tf.keras.optimizers.Adam(clipnorm=1.0), loss="binary_crossentropy", metrics=["accuracy"])
cnn_model.fit(X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1), y, epochs=200, batch_size=16, validation_split=0.2, verbose=0,
              callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)])

rnn_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X.shape[1], 1)),
    tf.keras.layers.LSTM(4, return_sequences=False),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dropout(0.65),
    tf.keras.layers.Dense(1, activation="sigmoid")
])
rnn_model.compile(optimizer=tf.keras.optimizers.Adam(clipnorm=1.0), loss="binary_crossentropy", metrics=["accuracy"])
rnn_model.fit(X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1), y, epochs=200, batch_size=16, validation_split=0.2, verbose=0,
              callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)])

# Save models
mlp_model.save("mlp_t21_model.keras")
cnn_model.save("cnn_t21_model.keras")
rnn_model.save("rnn_t21_model.keras")