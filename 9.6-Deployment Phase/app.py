import streamlit as st
from pathlib import Path
import pickle
import pandas as pd

# Load model
model_path = Path(__file__).parent / "AdaBoost_final_defect_prediction_model.pkl"
with open(model_path, "rb") as f:
    package = pickle.load(f)

model = package['model'] if isinstance(package, dict) else package

# Load dataset to get min/max range
csv_path = Path(__file__).parent / "DillibabuSarva_DefectDataset.csv"
df_csv = pd.read_csv(csv_path)

# Your 10 final features - keep same order as training
features = ['nosi','dit','cbo','rfc','maxNestedBlocks','uniqueWordsQty','assignmentsQty','numbersQty','tryCatchQty','parenthesizedExpsQty']

# Expanded names + full forms
expanded_names = {
    'nosi': 'NOSI (Number of Static Invocations)',
    'dit': 'DIT (Depth of Inheritance Tree) - CK Metric',
    'cbo': 'CBO (Coupling Between Objects) - CK Metric',
    'rfc': 'RFC (Response For a Class) - CK Metric',
    'maxNestedBlocks': 'Max Nested Blocks (Max depth of nested blocks)',
    'uniqueWordsQty': 'Unique Words Quantity (Vocabulary size)',
    'assignmentsQty': 'Assignments Quantity (No. of assignments)',
    'numbersQty': 'Numbers Quantity (No. of numeric literals)',
    'tryCatchQty': 'Try/Catch Quantity (No. of exception blocks)',
    'parenthesizedExpsQty': 'Parenthesized Expressions (No. of ( ) expressions)'
}

st.title("Intelligent Software Defect Prediction Using AI-Based Code Metrics")
st.write("Enter CK metrics to predict defect")

inputs = {}
for col in features:
    min_val = float(df_csv[col].min())
    max_val = float(df_csv[col].max())
    median_val = float(df_csv[col].median())

    # Label with range in brackets
    label = f"{expanded_names[col]} [{min_val:.0f} - {max_val:.0f}]"

    inputs[col] = st.number_input(
        label,
        min_value=min_val,
        max_value=max_val,
        value=median_val,
        help=f"Valid range from dataset: {min_val} to {max_val}"
    )

if st.button("Predict Defect"):
    df_input = pd.DataFrame([inputs])
    pred = model.predict(df_input[features])[0]
    proba = model.predict_proba(df_input[features])[0]

    if pred == 1:
        st.error(f"Defect Predicted! Probability: {proba[1]:.2%}")
    else:
        st.success(f"No Defect! Probability: {proba[1]:.2%}")