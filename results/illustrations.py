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
file_types = ['total_loss.csv', 'depth_counts.csv']

# Loop through each file type for plotting
for file_type in file_types:
    fig, ax = plt.subplots(figsize=(6, 5))

    for col, base_path in enumerate(base_paths):
        file_path = base_path + file_type
        df = pd.read_csv(file_path)

        if 'depth_counts.csv' in file_type:
            counts = df.iloc[:, 1].replace(0, 0.1)  # Handle zeros for log scale
            depths = df.iloc[:, 0]
            ax.scatter(depths, counts, label=f'Config {col + 1}', s=10)
            max_depth = depths.max()  # Find the maximum depth for this configuration
            ax.axvline(max_depth, color='grey', linestyle='--', linewidth=0.5)  # Draw a vertical line
            ax.set_yscale('log')
            ax.set_title('Node Visits (Search Tree Depth)')
            ax.set_xlabel('Depth')
            ax.set_ylabel('Visit Counts (log)')
        else:
            series = df.iloc[:, 0]
            ax.plot(series, label=f'Config {col + 1}')
            title = 'Policy Loss' if 'policy_loss.csv' in file_type else 'Total Loss'
            ax.set_title(title)
            ax.set_xlabel('Batch')
            ax.set_ylabel('Value')  # Updated y-axis label to "Value"

        #ax.legend(loc='upper right')  # Position the legend in the upper right corner

    # Show the plot for the current file type
    plt.show()
