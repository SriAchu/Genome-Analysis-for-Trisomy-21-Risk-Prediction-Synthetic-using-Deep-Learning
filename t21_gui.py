import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import os

# Debug: Print TensorFlow version
st.write(f"TensorFlow version: {tf.__version__}")

# Paths
common_genes_file = "common_genes.csv"
mlp_model_file = "mlp_t21_model.keras"
cnn_model_file = "cnn_t21_model.keras"
rnn_model_file = "rnn_t21_model.keras"

# Verify file existence
for model_file in [mlp_model_file, cnn_model_file, rnn_model_file, common_genes_file]:
    if not os.path.exists(model_file):
        st.error(f"File not found: {model_file}. Ensure it is in the same directory as this script ({os.getcwd()}).")
        st.stop()

# Load models individually
mlp_model = None
cnn_model = None
rnn_model = None

try:
    mlp_model = tf.keras.models.load_model(mlp_model_file)
    st.write("MLP model loaded successfully.")
except Exception as e:
    st.error(f"Error loading MLP model from {mlp_model_file}: {e}")
    st.stop()

try:
    cnn_model = tf.keras.models.load_model(cnn_model_file)
    st.write("CNN model loaded successfully.")
except Exception as e:
    st.error(f"Error loading CNN model from {cnn_model_file}: {e}")
    st.stop()

try:
    rnn_model = tf.keras.models.load_model(rnn_model_file)
    st.write("RNN model loaded successfully.")
except Exception as e:
    st.error(f"Error loading RNN model from {rnn_model_file}: {e}")
    st.stop()

# Load common genes
try:
    common_genes = pd.read_csv(common_genes_file, header=None)[0].tolist()
    # Validate gene names
    if any(gene == "0" or not isinstance(gene, str) for gene in common_genes):
        st.error(f"Invalid gene names in {common_genes_file}: {common_genes}. Re-run models.py to regenerate.")
        st.stop()
    st.write(f"Number of genes loaded: {len(common_genes)}")
    st.write(f"Loaded genes: {common_genes}")
    expected_genes = 50  # Match the number of genes in models.py
    if len(common_genes) != expected_genes:
        st.error(f"Expected {expected_genes} genes in {common_genes_file}, but found {len(common_genes)}. Re-run models.py to regenerate common_genes.csv.")
        st.stop()
except Exception as e:
    st.error(f"Error loading common_genes.csv: {e}. Ensure file is in the same directory as this script.")
    st.stop()

# Streamlit app
st.title("T21 RNA Sequence Prediction")
st.write("Enter an RNA sequence to predict Trisomy 21 risk.")

# Text input for RNA sequence
rna_sequence = st.text_area("Enter RNA Sequence (e.g., AUGCCGAU...)", height=100)

if st.button("Predict"):
    if not rna_sequence:
        st.error("Please enter an RNA sequence.")
    else:
        # Strip whitespace and newlines
        rna_sequence = rna_sequence.strip()
        # Debug: Show sequence length and characters
        st.write(f"Debug - Sequence length: {len(rna_sequence)}")
        st.write(f"Debug - Sequence characters: {[c for c in rna_sequence]}")
        
        # Validate sequence length (minimum 20, maximum 1000 nucleotides)
        if len(rna_sequence) < 20:
            st.error(f"RNA sequence is too short ({len(rna_sequence)} nucleotides). Minimum length is 20 nucleotides.")
            st.stop()
        if len(rna_sequence) > 1000:
            st.error(f"RNA sequence is too long ({len(rna_sequence)} nucleotides). Maximum length is 1000 nucleotides.")
            st.stop()
        
        # Validate sequence
        valid_nucleotides = set("AUGC")
        invalid_chars = [c for c in rna_sequence if c.upper() not in valid_nucleotides]
        if invalid_chars:
            st.error(f"Invalid RNA sequence. Found invalid characters: {invalid_chars}. Use only A, U, G, C.")
        else:
            # Simulate gene expression matrix
            np.random.seed(42)
            seq_length = len(rna_sequence)
            gc_content = (rna_sequence.upper().count("G") + rna_sequence.upper().count("C")) / seq_length
            st.write(f"Debug - GC content: {gc_content:.3f}")
            
            # Generate expression for n_genes (must match expected_genes)
            n_genes = expected_genes  # Use expected_genes (50)
            # Base expression: Control-like (mean=8, as in data.py)
            expression = np.random.normal(loc=8, scale=5.5, size=(1, n_genes))
            # Adjust to ensure mean is closer to 8 for Control-like sequences
            expression_mean = np.mean(expression)
            expression = expression + (8 - expression_mean)  # Shift to ensure mean=8
            st.write(f"Debug - Initial expression mean: {np.mean(expression):.3f}")
            
            # Adjust for T21-like expression if GC content is high (proxy for T21)
            if gc_content > 0.45:
                # Simulate T21 overexpression (mimic data.py where T21 mean=12)
                # Increase expression for a subset of genes (e.g., first 15 genes, simulating chromosome 21 genes)
                t21_gene_indices = range(15)  # First 15 genes as proxy for chromosome 21
                for i in t21_gene_indices:
                    expression[0, i] = np.random.normal(loc=16, scale=5.5)  # Stronger T21-like mean
                st.write(f"Debug - Adjusted T21 expression mean (first 15 genes): {np.mean(expression[0, t21_gene_indices]):.3f}")
            
            # Adjust based on sequence properties
            expression *= (1 + 0.05 * (seq_length / 1000))  # Scale with length
            expression *= (1 + 0.05 * gc_content)          # Scale with GC content (increased factor)
            st.write(f"Debug - Expression mean after scaling: {np.mean(expression):.3f}")
            
            # Apply dropout
            for i in range(n_genes):
                if expression[0, i] < 2:
                    expression[0, i] *= np.random.choice([0, 1], p=[0.6, 0.4])
            
            # Normalize
            expression = np.clip(expression, 0, None)
            expression = np.log2(expression + 1)
            scaler = StandardScaler()
            expression_scaled = scaler.fit_transform(expression.T).T
            
            # Debug: Show expression shape
            st.write(f"Debug - Expression shape: {expression_scaled.shape}")
            
            # Predict
            try:
                # Ensure correct shape for MLP: (1, n_genes)
                mlp_input_shape = mlp_model.input_shape[-1]  # Expected number of features
                if expression_scaled.shape[1] != mlp_input_shape:
                    raise ValueError(f"Expression shape {expression_scaled.shape} does not match MLP input shape (expected: {mlp_input_shape} features).")
                mlp_probs = mlp_model.predict(expression_scaled, batch_size=1, verbose=0)
                # Reshape for CNN and RNN: (1, n_genes, 1)
                cnn_probs = cnn_model.predict(expression_scaled.reshape(1, expression_scaled.shape[1], 1), batch_size=1, verbose=0)
                rnn_probs = rnn_model.predict(expression_scaled.reshape(1, expression_scaled.shape[1], 1), batch_size=1, verbose=0)
                ensemble_probs = (mlp_probs + cnn_probs + rnn_probs) / 3
                
                # Display probabilities
                st.write("**Prediction Probabilities (T21 likelihood):**")
                st.write(f"MLP: {mlp_probs[0][0]:.3f}")
                st.write(f"CNN: {cnn_probs[0][0]:.3f}")
                st.write(f"RNN: {rnn_probs[0][0]:.3f}")
                st.write(f"Ensemble: {ensemble_probs[0][0]:.3f}")
                
                # Prediction logic
                if mlp_probs[0][0] > 0.5 and cnn_probs[0][0] > 0.5 and rnn_probs[0][0] > 0.5:
                    prediction = "T21"
                    st.write("**Prediction:** T21 (All models agree)")
                else:
                    # If at least one model is ≤ 0.5, check ensemble
                    if ensemble_probs[0][0] > 0.5:
                        prediction = "T21"
                        st.write("**Prediction:** T21 (Ensemble majority)")
                    else:
                        prediction = "Control"
                        st.write("**Prediction:** Control (Ensemble below threshold)")
            except Exception as e:
                st.error(f"Prediction failed: {e}. Check model input shape (expected: {mlp_input_shape} genes).")
                st.write(f"Debug - Ensure the MLP model expects {mlp_input_shape} features and common_genes.csv has {expected_genes} genes.")

# Cleanup
st.write("*Note:* Enter a valid RNA sequence (A, U, G, C) to predict T21 risk.")