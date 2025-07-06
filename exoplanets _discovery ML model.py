import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Load and preprocess data

def load_and_preprocess_data(file_path):

    file_path = "C:\\Users\\PMYLS\\Downloads\\archive (2)\\all_exoplanets_2021.csv"
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
    weights = np.random.rand(X.shape[ 1 ])
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
        loss = -np.mean(y * np.log(predictions + 1e-15) + (1 - y) * np.log(1 - predictions + 1e-15))
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
    return weights, bias


# Predict function
def predict(X, weights, bias):
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    linear_model = np.dot(X, weights) + bias
    predictions = sigmoid(linear_model)
    return (predictions >= 0.5).astype(int)


# Visualize discovered exoplanet in solar system
def visualize_exoplanet_position(data, discovered_index):
    exoplanet = data.iloc[discovered_index]
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(0, 0, 0, color='yellow', s=500, label='Sun')
    ax.scatter(1, 0, 0, color='blue', s=100, label='Earth')
    ax.plot([ 0, 1 ], [ 0, 0 ], [ 0, 0 ], color='green', linestyle='--', label='Habitable Zone')
    ax.scatter(exoplanet[ 'Orbit Semi-Major Axis' ], 0, exoplanet[ 'Distance' ], color='red', s=200,
               label='Discovered Exoplanet')
    ax.set_xlabel('Orbit Semi-Major Axis (normalized)')
    ax.set_ylabel('Y-Axis (placeholder)')
    ax.set_zlabel('Distance from Earth (normalized)')
    ax.legend()
    plt.title('Position of Discovered Exoplanet in Solar System')
    plt.show()


# Updated Distance Visualization (Horizontal Bar Chart)
def visualize_distance_from_earth_simple(data, discovered_index):
    exoplanet = data.iloc[discovered_index]
    distance = exoplanet['Distance']

    labels = ['Earth', 'Discovered Exoplanet']
    distances = [0.5, distance]

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh(labels, distances, color=['skyblue', 'darkorange'])

    ax.set_xlabel('Distance (normalized)')
    ax.set_title('Distance of Discovered Exoplanet from Earth')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xlim(0, max(distances) * 1.2)

    plt.show()


# Updated Features Visualization (Bar Chart with different colors)
def visualize_exoplanet_features(data, features, discovered_index):
    exoplanet = data.iloc[discovered_index]
    feature_values = exoplanet[features].values

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(features)))
    ax.bar(features, feature_values, color=colors)

    ax.set_ylabel('Normalized Value')
    ax.set_title('Features of Discovered Exoplanet')
    plt.xticks(rotation=450, ha='right', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()


# 3D Orbital Motion Visualization
def visualize_orbital_motion(data, discovered_index, time_steps=1000):
    exoplanet = data.iloc[ discovered_index ]
    semi_major_axis = exoplanet['Orbit Semi-Major Axis']  # in AU
    orbital_period = exoplanet['Orbital Period Days']  # in days

    # Convert orbital period to seconds for simulation
    orbital_period_seconds = orbital_period * 86400 # 1 day = 86400 seconds

    # Create time array from 0 to orbital period, with time_steps points
    time = np.linspace(0, orbital_period_seconds, time_steps)

    # Orbital parameters (assuming a circular orbit for simplicity)
    x_orbit = semi_major_axis * np.cos(2 * np.pi * time / orbital_period_seconds)
    y_orbit = semi_major_axis * np.sin(2 * np.pi * time / orbital_period_seconds)
    z_orbit = np.zeros_like(x_orbit)  # No vertical motion for simplicity

    # Set up 3D plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the star (central object at the origin)
    ax.scatter(0, 0, 0, color='yellow', s=500, label='Star', zorder=5)

    # Plot the orbit of the exoplanet (in a circle)
    ax.plot(x_orbit, y_orbit, z_orbit, color='blue', label='Exoplanet Orbit', linestyle='--', alpha=0.7)

    # Plot the exoplanet's position at different time steps to simulate motion
    ax.scatter(x_orbit[0], y_orbit[0], z_orbit[0], color='red', s=100, label='Initial Position', zorder=4)

    ax.set_xlabel('X-Axis (AU)')
    ax.set_ylabel('Y-Axis (AU)')
    ax.set_zlabel('Z-Axis (AU)')
    ax.set_title(f"3D Orbital Motion of Discovered Exoplanet\nSemi-Major Axis: {semi_major_axis} AU")

    ax.legend()
    ax.set_box_aspect([ 1, 1, 1 ])  # Equal scaling in all axes

    # Animation for motion (Exoplanet moving along the orbit)
    for i in range(1, time_steps, int(time_steps / 10)):  # Update every 10th time step
        ax.scatter(x_orbit[i], y_orbit[i], z_orbit[i], color='red', s=100, zorder=4)
        plt.pause(0.05)  # Pause to show the motion step by step

    plt.show()


# Main workflow
def main():
    file_path = 'all_exoplanets_2021.csv'
    try:
        data = load_and_preprocess_data(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}. Please provide a valid path.")
        return

    features = [ 'Orbital Period Days', 'Orbit Semi-Major Axis', 'Mass', 'Equilibrium Temperature', 'Distance' ]
    target = 'Habitability'
    data[ target ] = ((data[ 'Equilibrium Temperature' ] > 200) & (data[ 'Equilibrium Temperature' ] < 300)).astype(int)

    X = data[ features ].values
    y = data[ target ].values

    weights, bias = custom_ml_model(X, y)
    predictions = predict(X, weights, bias)
    accuracy = np.mean(predictions == y)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    habitable_indices = np.where(predictions == 1)[0]
    if len(habitable_indices) == 0:
        print("No habitable exoplanets were predicted. Cannot visualize.")
        return

    undiscovered_index = habitable_indices[0]
    print("Discovered Exoplanet Features:")
    print(data.iloc[undiscovered_index])

    visualize_exoplanet_position(data, undiscovered_index)
    visualize_distance_from_earth_simple(data, undiscovered_index)
    visualize_exoplanet_features(data, features, undiscovered_index)
    visualize_orbital_motion(data, undiscovered_index)


if __name__ == '__main__':
    main()
