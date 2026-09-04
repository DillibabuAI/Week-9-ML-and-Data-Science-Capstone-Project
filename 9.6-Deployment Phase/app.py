import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load package
with open("AdaBoost_final_defect_prediction_model.pkl", "rb") as f:
    data = pickle.load(f)

model = data['model']
features = data['features']
upper = data['upper']
lower = data['lower']
medians = data['medians']

st.title("Software Defect Prediction")
st.write("Enter CK metrics to predict defect")

inputs = {}
for col in features:
    inputs[col] = st.number_input(f"{col}", value=float(medians[col]))

if st.button("Predict Defect"):
    # Create dataframe
    df = pd.DataFrame([inputs])

    # Apply same outlier replacement you did
    for col in features:
        if df[col].iloc[0] > upper[col] or df[col].iloc[0] < lower[col]:
            df[col] = medians[col]

    pred = model.predict(df[features])
    prob = model.predict_proba(df[features])[0]

    if pred[0] == 1:
        st.error(f"Defect Predicted! Probability: {prob[1]:.2%}")
    else:
        st.success(f"No Defect! Probability: {prob[0]:.2%}")