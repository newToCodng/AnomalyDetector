import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore

# Set random seed for reproducibility
np.random.seed(42)

# Generate time series data
time = np.arange(0, 100, 0.1)
normal_signal = np.sin(time) + np.random.normal(0, 0.2, len(time))

# Simulate sensor failure (Inject anomalies)
num_anomalies = 10  # Adjust to add more or fewer anomalies
anomaly_indices = np.random.choice(len(time), size=num_anomalies, replace=False)
anomalies = normal_signal.copy()
anomalies[anomaly_indices] += np.random.normal(3, 1, len(anomaly_indices))

# Create DataFrame
df = pd.DataFrame({"time": time, "sensor_value": anomalies})

# PLOT 1: Sensor Data with Injected Anomalies
plt.figure(figsize=(12, 5))
plt.plot(df["time"], df["sensor_value"], label="Sensor Signal", color="blue")
plt.scatter(df["time"][anomaly_indices], df["sensor_value"][anomaly_indices], color="red", label="Injected Anomalies", marker="x", s=80, zorder=3)
plt.xlabel("Time")
plt.ylabel("Sensor Value")
plt.title("Simulated Sensor Data with Anomalies")
plt.legend()
plt.grid(True)
plt.show()

# Calculate Z-score for anomaly detection
df["zscore"] = zscore(df["sensor_value"])
threshold = 2.5  # Change to adjust anomaly sensitivity
df["is_anomaly"] = df["zscore"].abs() > threshold

# PLOT 2 & 3: Sensor Data with Detected Anomalies & Z-score Over Time
plt.figure(figsize=(12, 8))

# 2.1: Sensor Data with Detected Anomalies
plt.subplot(2, 1, 1)
plt.plot(df["time"], df["sensor_value"], label="Sensor Signal", color="blue")
plt.scatter(df["time"][df["is_anomaly"]], df["sensor_value"][df["is_anomaly"]], color="red", label="Detected Anomalies", marker="x", s=80, zorder=3)
plt.xlabel("Time")
plt.ylabel("Sensor Value")
plt.title("Sensor Data with Detected Anomalies")
plt.legend()
plt.grid(True)

# 2.2: Z-score Values Over Time
plt.subplot(2, 1, 2)
plt.plot(df["time"], df["zscore"], label="Z-score", color="purple", linewidth=1.5)
plt.axhline(threshold, color="red", linestyle="dashed", linewidth=1.2, label=f"Threshold (+{threshold})")
plt.axhline(-threshold, color="red", linestyle="dashed", linewidth=1.2, label=f"Threshold (-{threshold})")
plt.axhline(0, color="black", linestyle="dashed", linewidth=0.8, alpha=0.7)  # Baseline at zero
plt.xlabel("Time")
plt.ylabel("Z-score")
plt.title("Z-score of Sensor Data Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
