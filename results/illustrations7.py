import pandas as pd
import matplotlib.pyplot as plt

# File names based on different search numbers
file_names = [
    '1_vs_1_mod_4_vs_6.txt',
    '2_vs_2_mod_4_vs_6.txt',
    '3_vs_3_mod_4_vs_6.txt',
    '4_vs_4_mod_4_vs_6.txt',
    '5_vs_5_mod_4_vs_6.txt',
    '6_vs_6_mod_4_vs_6.txt',
    '7_vs_7_mod_4_vs_6.txt'
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
    opponent_search_number = file_name.split('_vs_')[0]  # Extract the part before '_vs_'
    temp_df = pd.DataFrame({
        'Test Case': [opponent_search_number],
        'Player 1 Win Ratio': [p1_win_ratio],
        'Player 2 Win Ratio': [p_neg1_win_ratio]
    })
    results_summary = pd.concat([results_summary, temp_df], ignore_index=True)

# Manually assigning test labels based on the order from highest to lowest
results_summary['Test Cases'] = ['Test 27', 'Test 28', 'Test 29', 'Test 30', 'Test 31', 'Test 32', 'Test 33']

# Plotting the distribution of wins as mirrored horizontal bars
fig, ax1 = plt.subplots(figsize=(6, 5))  # Adjusted figure size for better label display
indices = range(len(results_summary))

# Player 1 bars starting from the center and extending left
ax1.barh(indices, -results_summary['Player 1 Win Ratio'], color='red', label='Player 1 Wins', height=0.4, align='center')
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
plt.tight_layout()  # Automatically adjust subplot parameters
plt.show()
