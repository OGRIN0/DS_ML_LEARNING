import pandas as pd
import numpy as np
import os

directory = "/Users/amarnath/DS_ML_learning"
filename = "breast_cancer.csv"
file_path = os.path.join(directory, filename)

num_records = 100

data = {
    "id": range(1, num_records + 1),
    "diagnosis": np.random.choice(["M", "B"], size=num_records, p=[0.4, 0.6]),
    "radius_mean": np.random.uniform(10, 20, size=num_records),
    "texture_mean": np.random.uniform(10, 30, size=num_records),
    "perimeter_main": np.random.uniform(50, 100, size=num_records),
    "area_main": np.random.uniform(100, 1000, size=num_records),
    "smoothness_mean": np.random.uniform(0.05, 0.15, size=num_records),
    "compactness_mean": np.random.uniform(0.1, 0.3, size=num_records),
    "concavity_mean": np.random.uniform(0.1, 0.4, size=num_records),
    "concave_points_worst": np.random.uniform(0.1, 0.2, size=num_records),
    "symmetry_worst": np.random.uniform(0.1, 0.3, size=num_records),
    "fractal_dimension_worst": np.random.uniform(0.05, 0.1, size=num_records),
}

df = pd.DataFrame(data)

os.makedirs(directory, exist_ok=True)
df.to_csv(file_path, index=False)
