import pandas as pd
import matplotlib.pyplot as plt

# File names based on different search numbers
file_names = [
    '300_vs_01.txt',
    '300_vs_10.txt',
    '300_vs_25.txt',
    '300_vs_50.txt',
    '300_vs_100.txt',
    '300_vs_200.txt'
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
        'Player 2 Search Number': [opponent_search_number],
        'Player 1 Win Ratio': [p1_win_ratio],
        'Player 2 Win Ratio': [p_neg1_win_ratio]
    })
    results_summary = pd.concat([results_summary, temp_df], ignore_index=True)

# Convert Opponent Search Number to integer and sort by it for plotting
results_summary['Player 2 Search Number'] = results_summary['Player 2 Search Number'].astype(int)
results_summary.sort_values('Player 2 Search Number', inplace=True, ascending=False)  # Sorted in descending order

# Assigning test labels correctly based on the sorted order
test_labels = ['Test 1', 'Test 2', 'Test 3', 'Test 4', 'Test 5', 'Test 6']  # Labels in descending order
results_summary['Test Cases'] = test_labels

# Plotting the distribution of wins as mirrored horizontal bars
fig, ax1 = plt.subplots(figsize=(6, 5))  # Adjust the figure size for clarity
indices = range(len(results_summary))

# Player 1 bars starting from the center and extending left
ax1.barh(indices, -results_summary['Player 1 Win Ratio'], color='red', label='Player 1 (300) Wins', height=0.4, align='center')
# Opponent bars starting from the center and extending right
ax1.barh(indices, results_summary['Player 2 Win Ratio'], color='blue', label='Player 2 Wins', height=0.4, align='center')

ax1.set_yticks(indices)
ax1.set_yticklabels(results_summary['Test Cases'])  # Use custom labels
ax1.set_ylabel('Test Cases')

# Adjust the x-axis to show the full range from -100 to 100
ax1.set_xlim([-100, 100])
ax1.set_xlabel('Percentage of Games Won')
ax1.set_title('Player 1 vs. Player 2', fontweight='bold')
plt.axvline(x=0, color='black', linestyle='--')
plt.grid(True)
plt.show()
