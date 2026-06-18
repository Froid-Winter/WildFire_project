import os, sys, io, glob, joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(__file__))
from config import FEATURE_COLS, CLASS_NAMES

BASE = os.path.dirname(__file__)

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(os.path.join(BASE, 'saved_model', 'fire_severity_model.keras'))
    return model

@st.cache_resource
def load_scaler():
    return joblib.load(os.path.join(BASE, 'saved_model', 'scaler.joblib'))

model = load_model()
scaler = load_scaler()

CLASS_DESCRIPTIONS = {
    "A": "Minimal fire severity. Low-intensity surface fire, easy to control with basic resources. Little to no damage to structures or vegetation.",
    "B": "Low fire severity. Moderate surface fire with some spread potential. May require a small crew and basic equipment to contain.",
    "C": "Moderate fire severity. Active surface fire with occasional torching. Firefighters can manage but resources should be scaled up.",
    "D": "High fire severity. Intense surface fire with group torching and short crown runs. A significant threat requiring coordinated suppression.",
    "E": "Very high fire severity. Extreme fire behavior with sustained crown fire runs. Difficult to control; evacuations may be needed.",
    "F": "Severe fire severity. Intense crown fire with long-range spotting. Major threat to life and property; large-scale response required.",
    "G": "Catastrophic fire severity. Uncontrollable firestorm with extreme spotting and fire whirls. Widespread destruction; full emergency mobilization.",
}

st.set_page_config(page_title="LiveStreamLip — Fire Severity Classifier", layout="wide")
st.title("LiveStreamLip")
st.markdown("Wildfire severity prediction (A–G) from weather data")

tab_home, tab_manual, tab_csv = st.tabs([" Home", " Manual Input", " CSV Upload"])

def make_prediction(features_2d):
    scaled = scaler.transform(features_2d)
    probs = model.predict(scaled, verbose=0)
    classes = np.argmax(probs, axis=1)
    return classes, probs

with tab_home:
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", "1,830,945")
    col2.metric("Weather Features", "15")
    col3.metric("Fire Stations Averaged", "6")

    st.subheader(" About the Dataset")
    st.write(
        "This dataset provides supplementary weather data for the "
        "**1.88 Million US Wildfires** dataset on Kaggle. "
        "Each fire record includes temperature (°C), precipitation (mm), "
        "and wind speed (km/h) calculated for the day of the fire and across "
        "four preceding windows: 0–10, 0–30, 0–60, and 0–180 days. "
        "Temperature and wind speed are averaged; precipitation is summed."
    )
    st.write(
        "Data sourced from [meteostat.net](https://meteostat.net/). "
        "For each fire, the 6 nearest weather stations were identified and their "
        "readings averaged (mean station distance ≈ 50 km)."
    )

    with st.expander(" Column naming conventions"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Prefix — Variable**")
            st.markdown("`tmp` — Temperature (°C)")
            st.markdown("`prcp` — Precipitation (mm)")
            st.markdown("`wspd` — Wind speed (km/h)")
            st.markdown("`mean` — Average (tmp, wspd)")
            st.markdown("`sum` — Sum (prcp)")
        with c2:
            st.markdown("**Suffix — Time window**")
            st.markdown("`_0` — Day of fire")
            st.markdown("`_10` — Previous 10 days")
            st.markdown("`_30` — Previous 30 days")
            st.markdown("`_60` — Previous 60 days")
            st.markdown("`_180` — Previous 180 days")

with tab_manual:
    st.subheader("Enter 15 weather features")
    cols_row1 = st.columns(5)
    vals = {}
    for i, col in enumerate(FEATURE_COLS):
        with cols_row1[i % 5]:
            vals[col] = st.number_input(col, value=0.0, format="%.4f", key=f"inp_{col}")

    if st.button("Predict Severity", type="primary"):
        row = np.array([[vals[c] for c in FEATURE_COLS]], dtype=np.float32)
        cls_idx, probs = make_prediction(row)
        cls_name = CLASS_NAMES[int(cls_idx[0])]
        conf = float(probs[0][int(cls_idx[0])])

        st.success(f"Predicted severity: **{cls_name}** (confidence: {conf:.2%})")
        st.info(CLASS_DESCRIPTIONS[cls_name])

        fig, ax = plt.subplots(figsize=(8, 3))
        colors = ['#2ecc71', '#27ae60', '#f1c40f', '#e67e22', '#e74c3c', '#c0392b', '#8e44ad']
        ax.bar(CLASS_NAMES, probs[0], color=colors)
        ax.set_ylabel("Probability")
        ax.set_title("Class Probability Distribution")
        ax.set_ylim(0, 1)
        for i, v in enumerate(probs[0]):
            ax.text(i, v + 0.02, f"{v:.1%}", ha='center', fontsize=9)
        st.pyplot(fig)

with tab_csv:
    st.subheader("Batch Prediction via CSV")
    st.markdown("Upload a CSV with these 15 columns (order matters):")
    st.code(", ".join(FEATURE_COLS))
    st.markdown("The file must include a header row.")

    uploaded = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded is not None:
        df_in = pd.read_csv(uploaded)
        missing = [c for c in FEATURE_COLS if c not in df_in.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            X = df_in[FEATURE_COLS].values.astype(np.float32)
            cls_idx, probs = make_prediction(X)
            df_out = df_in.copy()
            df_out["predicted_class"] = [CLASS_NAMES[i] for i in cls_idx]
            df_out["class_description"] = [CLASS_DESCRIPTIONS[CLASS_NAMES[i]] for i in cls_idx]
            for i, name in enumerate(CLASS_NAMES):
                df_out[f"prob_{name}"] = probs[:, i]

            st.success(f"Predictions done for {len(df_out)} rows")
            st.dataframe(df_out.head(50))

            csv_bytes = df_out.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Results (CSV)",
                data=csv_bytes,
                file_name="predictions.csv",
                mime="text/csv",
            )

st.markdown("---")
st.caption("Model: 3-layer DNN (256→128→64) trained on 1.83M US wildfire weather records")
