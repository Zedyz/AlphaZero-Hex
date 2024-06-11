import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Base paths for each training run
base_paths = [
    '../training1/',
    '../training2/',
    '../training3/',
    '../training4/',
    '../training5/',
    '../training6/',
    '../training7/'
]

# File type to plot for total loss
file_type = 'total_loss.csv'

fig, ax = plt.subplots(figsize=(10, 8))

# Loop through each configuration
for col, base_path in enumerate(base_paths):
    file_path = base_path + file_type
    df = pd.read_csv(file_path)
    total_loss = df.iloc[:, 0]  # Assume total loss is the first column

    # Normalize the total loss by dividing by the maximum value or mean
    # normalized_loss = total_loss / total_loss.max()  # Option 1: Normalize by max
    normalized_loss = (total_loss - total_loss.mean()) / total_loss.std()  # Option 2: Normalize by Z-score

    # Plot the normalized data
    ax.plot(normalized_loss, label=f'Config {col + 1}')

ax.set_title('Normalized Total Loss Across Configurations')
ax.set_xlabel('Batch Number')
ax.set_ylabel('Normalized Loss')
ax.legend(loc='upper right')
plt.show()
