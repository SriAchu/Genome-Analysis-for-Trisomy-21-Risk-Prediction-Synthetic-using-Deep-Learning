import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

# Set random seed
np.random.seed(42)

# Parameters
n_samples = 56
n_genes = 9999
t21_samples = 28
control_samples = 28
output_file = "synthetic_t21_rnaseq.csv"

# Gene names
t21_specific_genes = ["DYRK1A", "APP", "MX1", "IFITM1", "STAT1"]
other_genes = [f"GENE_{i}" for i in range(n_genes - len(t21_specific_genes))]
genes = t21_specific_genes + other_genes

# Expression matrix
expression = np.zeros((n_genes, n_samples))

# Control samples
for i in range(n_genes):
    expression[i, t21_samples:] = np.random.normal(loc=8, scale=5, size=control_samples)

# T21 samples
for i, gene in enumerate(genes):
    if gene in ["DYRK1A", "APP"]:  # ~1.005-fold
        expression[i, :t21_samples] = np.random.normal(loc=8.02, scale=5, size=t21_samples)
    elif gene in ["MX1", "IFITM1", "STAT1"]:  # ~1.02-fold
        expression[i, :t21_samples] = np.random.normal(loc=8.08, scale=5, size=t21_samples)
    else:
        expression[i, :t21_samples] = np.random.normal(loc=8.002, scale=5.2, size=t21_samples)

# Batch effects
batch_effect = np.random.normal(loc=0, scale=4.5, size=(n_genes, n_samples))
expression += batch_effect

# Random noise
noise = np.random.normal(loc=0, scale=3.5, size=(n_genes, n_samples))
expression += noise

# Sample-specific variability
for j in range(n_samples):
    sample_effect = np.random.normal(loc=0, scale=3, size=n_genes)
    expression[:, j] += sample_effect

# Dropout for low-expressed genes
for i in range(n_genes):
    for j in range(n_samples):
        if expression[i, j] < 2:
            expression[i, j] *= np.random.choice([0, 1], p=[0.6, 0.4])

# Non-negative counts
expression = np.clip(expression, 0, None)

# Log2-transform
expression = np.log2(expression + 1)

# Z-score
scaler = StandardScaler()
expression_scaled = scaler.fit_transform(expression.T).T

# DataFrame
sample_names = [f"T21_{i+1}" for i in range(t21_samples)] + [f"Control_{i+1}" for i in range(control_samples)]
df = pd.DataFrame(expression_scaled, index=genes, columns=sample_names)

# Label row
labels = ["T21"] * t21_samples + ["Control"] * control_samples
label_row = pd.DataFrame([labels], index=["Label"], columns=sample_names)
df = pd.concat([df, label_row], axis=0)

# Save with error handling
try:
    df.to_csv(output_file)
    print(f"Dataset saved to {output_file}")
except PermissionError as e:
    print(f"PermissionError: Unable to write to {output_file}. Try the following:")
    print("- Ensure the file is not open in another program (e.g., Excel).")
    print("- Run the script as Administrator.")
    print("- Save to a different directory, e.g., C:\\Users\\Krishna\\Documents.")
    alternative_file = os.path.join(os.path.expanduser("~"), "Documents", "synthetic_t21_rnaseq.csv")
    print(f"Attempting to save to {alternative_file}...")
    try:
        df.to_csv(alternative_file)
        print(f"Dataset saved to {alternative_file}")
    except Exception as e2:
        print(f"Failed to save to {alternative_file}: {e2}")
except Exception as e:
    print(f"Error saving dataset: {e}")