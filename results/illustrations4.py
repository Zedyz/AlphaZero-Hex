import pandas as pd
import matplotlib.pyplot as plt

# Base paths for each training run
base_paths = [
    '../../training1/',
    '../../training2/',
    '../../training3/',
    '../../training4/',
    '../../training5/',
    '../../training6/',
    '../../training7/'
]

# Types of files to plot
file_types = ['total_loss.csv', 'policy_loss.csv', 'value_loss.csv', 'entropy.csv', 'depth_counts.csv']

# Initialize a figure for the rolling averages of entropy
entropy_fig, entropy_ax = plt.subplots(figsize=(6, 5))

# Loop through each file type for plotting
for file_type in file_types:
    for col, base_path in enumerate(base_paths):
        file_path = base_path + file_type
        df = pd.read_csv(file_path)

        if 'entropy.csv' in file_type:
            series = df.iloc[:, 0]
            rolling_avg = series.rolling(window=500).mean()  # Calculate rolling average
            entropy_ax.plot(rolling_avg, label=f'Config {col + 1}')

# Configure and show the entropy rolling average plot separately
entropy_ax.set_title('Entropy (window = 500)')
entropy_ax.set_xlabel('Batch')
entropy_ax.set_ylabel('Entropy')
#entropy_ax.legend()  # Make the legend smaller
plt.show()
