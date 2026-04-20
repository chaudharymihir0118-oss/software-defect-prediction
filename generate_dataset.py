import pandas as pd
import numpy as np
import os

# Create dataset directory
os.makedirs('dataset', exist_ok=True)

# Generate synthetic dataset
np.random.seed(42)
num_samples = 500

# Features:
# loc: Lines of Code (positive correlation with defect)
# complexity: Cyclomatic complexity (positive correlation)
# coupling: Number of dependencies (positive correlation)
# cohesion: Module cohesion (negative correlation)
# defects: 0 or 1

# Non-defective logic
loc_0 = np.random.normal(loc=100, scale=30, size=num_samples // 2)
complexity_0 = np.random.normal(loc=10, scale=4, size=num_samples // 2)
coupling_0 = np.random.normal(loc=5, scale=2, size=num_samples // 2)
cohesion_0 = np.random.normal(loc=80, scale=10, size=num_samples // 2)
defects_0 = np.zeros(num_samples // 2)

# Defective logic
loc_1 = np.random.normal(loc=300, scale=80, size=num_samples // 2)
complexity_1 = np.random.normal(loc=25, scale=8, size=num_samples // 2)
coupling_1 = np.random.normal(loc=15, scale=5, size=num_samples // 2)
cohesion_1 = np.random.normal(loc=40, scale=15, size=num_samples // 2)
defects_1 = np.ones(num_samples // 2)

# Combine
loc = np.concatenate([loc_0, loc_1])
complexity = np.concatenate([complexity_0, complexity_1])
coupling = np.concatenate([coupling_0, coupling_1])
cohesion = np.concatenate([cohesion_0, cohesion_1])
defects = np.concatenate([defects_0, defects_1])

# Ensure positive values
loc = np.maximum(loc, 10)
complexity = np.maximum(complexity, 1)
coupling = np.maximum(coupling, 0)
cohesion = np.clip(cohesion, 0, 100)

df = pd.DataFrame({
    'loc': np.round(loc).astype(int),
    'complexity': np.round(complexity).astype(int),
    'coupling': np.round(coupling).astype(int),
    'cohesion': np.round(cohesion).astype(int),
    'defects': defects.astype(int)
})

# Shuffle dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to CSV
df.to_csv('dataset/sample_data.csv', index=False)
print("Generated dataset/sample_data.csv successfully!")
