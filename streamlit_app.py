# streamlit_app.py 

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D

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

# Visualization 1: Exoplanet Position in 3D Space
def visualize_exoplanet_position(data, index):
    exo = data.iloc[index]
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(0, 0, 0, color='yellow', s=500, label='Sun')
    ax.scatter(1, 0, 0, color='blue', s=100, label='Earth')
    ax.plot([0, 1], [0, 0], [0, 0], color='green', linestyle='--', label='Habitable Zone')
    ax.scatter(exo['Orbit Semi-Major Axis'], 0, exo['Distance'], color='red', s=200, label='Discovered Exoplanet')
    ax.set_xlabel('Orbit Semi-Major Axis')
    ax.set_ylabel('Y (Placeholder)')
    ax.set_zlabel('Distance from Earth')
    ax.legend()
    plt.title('🧭 Position of Discovered Exoplanet in Solar System')
    st.pyplot(fig)

# Visualization 2: Distance Bar Chart
def visualize_distance(data, index):
    exo = data.iloc[index]
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh(['Earth', 'Discovered Exoplanet'], [0.5, exo['Distance']], color=['skyblue', 'darkorange'])
    ax.set_xlabel('Distance (normalized)')
    ax.set_title('📏 Distance of Discovered Exoplanet from Earth')
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.set_xlim(0, max(0.5, exo['Distance']) * 1.2)
    st.pyplot(fig)

# Visualization 3: Feature Bar Chart
def show_feature_bar_chart(data, features, index):
    exo = data.iloc[index]
    values = exo[features].values
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.viridis(np.linspace(0, 1, len(features)))
    ax.bar(features, values, color=colors)
    plt.xticks(rotation=45, ha='right')
    ax.set_ylabel('Normalized Value')
    ax.set_title('🧬 Features of Discovered Exoplanet')
    ax.spines[['top', 'right']].set_visible(False)
    st.pyplot(fig)

# Visualization 4: 3D Orbital Motion
def visualize_orbital_motion(data, index, time_steps=500):
    exo = data.iloc[index]
    a = exo['Orbit Semi-Major Axis']
    T_days = exo['Orbital Period Days']
    T_sec = T_days * 86400
    t = np.linspace(0, T_sec, time_steps)
    x = a * np.cos(2 * np.pi * t / T_sec)
    y = a * np.sin(2 * np.pi * t / T_sec)
    z = np.zeros_like(x)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(0, 0, 0, color='yellow', s=500, label='Star')
    ax.plot(x, y, z, linestyle='--', alpha=0.6, label='Orbit')
    ax.scatter(x[0], y[0], z[0], color='red', s=100, label='Exoplanet Start')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('🌀 3D Orbital Motion of Exoplanet')
    ax.legend()
    st.pyplot(fig)

# Main logic
file_path = 'all_exoplanets_2021.csv'
if not os.path.exists(file_path):
    st.error(f"Dataset not found. Please ensure '{file_path}' is in the repo root.")
    st.stop()

data = load_and_preprocess_data(file_path)
features = ['Orbital Period Days', 'Orbit Semi-Major Axis', 'Mass', 'Equilibrium Temperature', 'Distance']
target = 'Habitability'
data[target] = ((data['Equilibrium Temperature'] > 200) & (data['Equilibrium Temperature'] < 300)).astype(int)

X = data[features].values
y = data[target].values
weights, bias = custom_ml_model(X, y)
predictions = predict(X, weights, bias)
accuracy = np.mean(predictions == y)
st.success(f"✅ Model Accuracy: {accuracy * 100:.2f}%")

habitable_indices = np.where(predictions == 1)[0]
if len(habitable_indices) == 0:
    st.warning("No habitable exoplanets predicted.")
    st.stop()

discovered_index = habitable_indices[0]
st.subheader("🧬 Discovered Exoplanet Details")
st.dataframe(data.iloc[[discovered_index]])

# Run all visualizations in perfect order
st.markdown("---")
st.subheader("📍 Exoplanet Position Visualization")
visualize_exoplanet_position(data, discovered_index)

st.markdown("---")
st.subheader("📏 Distance from Earth")
visualize_distance(data, discovered_index)

st.markdown("---")
st.subheader("🧬 Feature Analysis")
show_feature_bar_chart(data, features, discovered_index)

st.markdown("---")
st.subheader("🌀 3D Orbital Motion")
visualize_orbital_motion(data, discovered_index)
