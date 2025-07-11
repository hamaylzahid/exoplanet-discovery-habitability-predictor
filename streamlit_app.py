# streamlit_app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("🌌 Exoplanet Habitability Predictor")

st.markdown("This app analyzes exoplanet data and predicts habitability potential using ML models.")

# Load dataset

df = pd.read_csv("all_exoplanets_2021.csv")

st.write("### 📊 Sample Data", df.head())

# Show EDA
st.write("### 🌐 Orbital Period vs Planet Radius")
fig, ax = plt.subplots()
sns.scatterplot(data=df, x="orbital_period", y="planet_radius", hue="habitability", ax=ax)
st.pyplot(fig)

# Placeholder for ML Prediction
st.write("### 🤖 Predict Habitability")
st.info("ML model integration coming soon!")
