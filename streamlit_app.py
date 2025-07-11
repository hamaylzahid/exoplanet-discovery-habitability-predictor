# streamlit_app.py 

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Set page config
st.set_page_config(page_title="Exoplanet Habitability Predictor", layout="wide")
st.title("🪐 Exoplanet Habitability Predictor")

# Load and preprocess data
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    num_cols = data.select_dtypes(include=['float64', 'int64']).columns
    data[num_cols] = data[num_cols].apply(lambda col: col.fillna(col.median()))
    cat_cols = data.select_dtypes(include=['object']).columns
    data[cat_cols] = data[cat_cols].fillna('Unknown')
    scaled_cols = ['Orbital Period Days', 'Orbit Semi-Major Axis', 'Mass', 'Equilibrium Temperature', 'Distance']
    for col in scaled_cols:
        col_min = data[col].min()
        col_max = data[col].max()
        data[col] = (data[col] - col_min) / (col_max - col_min)
    return data

# Custom ML Model Implementation
def custom_ml_model(X, y, learning_rate=0.01, epochs=100):
    weights = np.random.rand(X.shape[1])
    bias = 0.0

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    for epoch in range(epochs):
        linear_model = np.dot(X, weights) + bias
        predictions = sigmoid(linear_model)
        errors = predictions - y
        gradient_weights = np.dot(X.T, errors) / len(y)
        gradient_bias = np.sum(errors) / len(y)
        weights -= learning_rate * gradient_weights
        bias -= learning_rate * gradient_bias
    return weights, bias

def predict(X, weights, bias):
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))
    linear_model = np.dot(X, weights) + bias
    predictions = sigmoid(linear_model)
    return (predictions >= 0.5).astype(int)

# Visualization

def show_feature_bar_chart(data, features, index):
    exoplanet = data.iloc[index]
    values = exoplanet[features].values
    fig, ax = plt.subplots()
    colors = plt.cm.viridis(np.linspace(0, 1, len(features)))
    ax.bar(features, values, color=colors)
    plt.xticks(rotation=45, ha='right')
    ax.set_ylabel('Normalized Value')
    ax.set_title('Discovered Exoplanet Features')
    st.pyplot(fig)

# Main logic
file_path = 'all_exoplanets_2021.csv'
if not os.path.exists(file_path):
    st.error(f"Dataset not found. Please ensure '{file_path}' is in the repo root.")
    st.stop()

# Load data
data = load_and_preprocess_data(file_path)

# Feature & label engineering
features = ['Orbital Period Days', 'Orbit Semi-Major Axis', 'Mass', 'Equilibrium Temperature', 'Distance']
target = 'Habitability'
data[target] = ((data['Equilibrium Temperature'] > 200) & (data['Equilibrium Temperature'] < 300)).astype(int)

X = data[features].values
y = data[target].values

# Train model
weights, bias = custom_ml_model(X, y)
predictions = predict(X, weights, bias)
accuracy = np.mean(predictions == y)
st.success(f"✅ Model Accuracy: {accuracy * 100:.2f}%")

habitable_indices = np.where(predictions == 1)[0]
if len(habitable_indices) == 0:
    st.warning("No habitable exoplanets predicted.")
    st.stop()

# Display first discovered habitable planet
discovered_index = habitable_indices[0]
st.subheader("🌍 Discovered Exoplanet Data")
st.dataframe(data.iloc[[discovered_index]])

# Feature bar chart
show_feature_bar_chart(data, features, discovered_index)
