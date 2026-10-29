import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(
    page_title="MediCheck AI",
    page_icon="🩺",
    layout="centered"
)

data = {
    "fever": [1,1,0,0,1,0,1,0,0,1,0,1],
    "cough": [1,1,1,0,1,0,0,1,0,1,1,0],
    "headache": [1,0,1,1,0,1,1,0,1,0,1,0],
    "fatigue": [1,1,0,1,1,1,0,0,1,0,1,1],
    "sore_throat": [1,1,1,0,0,0,0,1,0,1,0,1],
    "category": [
        "Flu-like symptoms", "Flu-like symptoms",
        "Respiratory symptoms", "General discomfort",
        "Flu-like symptoms", "General discomfort",
        "Flu-like symptoms", "Respiratory symptoms",
        "General discomfort", "Respiratory symptoms",
        "General discomfort", "Flu-like symptoms"
    ]
}

df = pd.DataFrame(data)
X = df.drop(columns=["category"])
y = df["category"]

model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X, y)

st.title("🩺 MediCheck AI")
st.subheader("Smart Symptom Checker")
st.write(
    "An educational demonstration of machine learning "
    "in healthcare."
)

st.warning(
    "This prototype uses invented sample data. "
    "It is not a diagnostic tool and cannot determine "
    "whether you have a disease."
)

st.write("### Select your symptoms")

selected = []

for symptom in X.columns:
    if st.checkbox(symptom.replace("_", " ").title()):
        selected.append(symptom)

if st.button("Analyze Symptoms", type="primary"):
    if not selected:
        st.info("Please select at least one symptom.")
    else:
        sample = pd.DataFrame(
            [[int(s in selected) for s in X.columns]],
            columns=X.columns
        )

        result = model.predict(sample)[0]

        st.write("### Illustrative model output")
        st.info(result)

        st.caption(
            "This output is generated from a toy dataset "
            "and is not a medical assessment. Consult a "
            "qualified healthcare professional for concerns."
        )
