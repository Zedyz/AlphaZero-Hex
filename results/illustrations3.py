import pandas as pd
import matplotlib.pyplot as plt

# File names based on different search numbers
file_names = [
    'mcts_vs_01.txt',
    'mcts_vs_10.txt',
    'mcts_vs_25.txt',
    'mcts_vs_50.txt',
    'mcts_vs_100.txt',
    'mcts_vs_200.txt',
    'mcts_vs_300.txt'
]

# Prepare DataFrame for the summary of results
results_summary = pd.DataFrame()

# Read results and calculate the proportions of victories for each file
for file_name in file_names:
    results = pd.read_csv(file_name, header=None, names=['Result'])
    total_games = len(results)
    p1_wins = (results['Result'] == 1).sum()
    p_neg1_wins = (results['Result'] == -1).sum()
    p1_win_ratio = p1_wins / total_games * 100
    p_neg1_win_ratio = p_neg1_wins / total_games * 100
    opponent_search_number = file_name.split('_vs_')[1].replace('.txt', '')
    temp_df = pd.DataFrame({
        'Trained model Search Number': [opponent_search_number],
        'Baseline Win Ratio': [p1_win_ratio],
        'Trained model Win Ratio': [p_neg1_win_ratio]
    })
    results_summary = pd.concat([results_summary, temp_df], ignore_index=True)

# Update labels for each file based on the order in the array
# Update all labels consistently
test_labels = ['Test 19', 'Test 18', 'Test 17', 'Test 16', 'Test 15', 'Test 14', 'Test 13']  # Reversed order
for i, label in enumerate(test_labels):
    results_summary.at[i, 'Trained model Search Number'] = label

# Reverse the DataFrame for plotting from bottom to top
results_summary = results_summary.iloc[::-1].reset_index(drop=True)

# Plotting the distribution of wins as mirrored horizontal bars
fig, ax1 = plt.subplots(figsize=(6, 5))  # Adjust the figure size for clarity

# Player 1 bars starting from the center and extending left
ax1.barh(results_summary.index, -results_summary['Baseline Win Ratio'], color='red', label='Baseline Wins', height=0.4, align='center')
# Opponent bars starting from the center and extending right
ax1.barh(results_summary.index, results_summary['Trained model Win Ratio'], color='blue', label='Player 2 Wins', height=0.4, align='center')

# Set up the y-axis
ax1.set_yticks(results_summary.index)
ax1.set_yticklabels(results_summary['Trained model Search Number'], fontsize=10)
ax1.set_ylabel('Test Cases')

# Set axis limits and labels
ax1.set_xlim([-100, 100])
ax1.set_xlabel('Percentage of Games Won')
ax1.set_title('Baseline vs. Trained model', fontweight='bold')
plt.axvline(x=0, color='black', linestyle='--')
plt.grid(True)
plt.tight_layout()  # Adjust layout to fit all labels
plt.show()
