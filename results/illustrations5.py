import pandas as pd
import matplotlib.pyplot as plt

# File names based on different search numbers
file_names = [
    '01_vs_mcts.txt',
    '10_vs_mcts.txt',
    '25_vs_mcts.txt',
    '50_vs_mcts.txt',
    '100_vs_mcts.txt',
    '200_vs_mcts.txt',
    '300_vs_mcts.txt'
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
    opponent_search_number = file_name.split('_')[0]  # Correctly extract only the numeric part
    temp_df = pd.DataFrame({
        'Baseline Search Number': [opponent_search_number],
        'Trained model Win Ratio': [p1_win_ratio],
        'Baseline Win Ratio': [p_neg1_win_ratio]
    })
    results_summary = pd.concat([results_summary, temp_df], ignore_index=True)

# Update labels for each file based on the order in the array
test_labels = ['Test 26', 'Test 25', 'Test 24', 'Test 23', 'Test 22', 'Test 21', 'Test 20']
for i, label in enumerate(test_labels):
    results_summary.at[i, 'Baseline Search Number'] = label

# Reverse the DataFrame for plotting from bottom to top
results_summary = results_summary.iloc[::-1].reset_index(drop=True)

# Plotting the distribution of wins as mirrored horizontal bars
fig, ax1 = plt.subplots(figsize=(6, 5))  # Adjust the figure size for clarity

# Player 1 bars starting from the center and extending left
ax1.barh(results_summary.index, -results_summary['Trained model Win Ratio'], color='red', label='Trained model Wins', height=0.4, align='center')
# Opponent bars starting from the center and extending right
ax1.barh(results_summary.index, results_summary['Baseline Win Ratio'], color='blue', label='Baseline Wins', height=0.4, align='center')

# Set up the y-axis
ax1.set_yticks(results_summary.index)
ax1.set_yticklabels(results_summary['Baseline Search Number'], fontsize=10)
ax1.set_ylabel('Test Cases')

# Set axis limits and labels
ax1.set_xlim([-100, 100])
ax1.set_xlabel('Percentage of Games Won')
ax1.set_title('Trained model vs. Baseline', fontweight='bold')
plt.axvline(x=0, color='black', linestyle='--')
plt.grid(True)
plt.tight_layout()  # Adjust layout to fit all labels
plt.show()
