import random
import torch
torch.manual_seed(0)
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def draw_hexagon(ax, center, size=1, color='white'):
    # Orientation with an edge pointing directly up
    hexagon = patches.RegularPolygon(center, numVertices=6, radius=size,
                                     orientation=0, facecolor=color, edgecolor='black')
    ax.add_patch(hexagon)

def plot_hex_grid(matrix, fig=None, player=None, draw = False, counter = None):
    plt.close('all')  # Ensure a fresh start by closing any existing figures
    fig, ax = plt.subplots()

    # Size of the hexagon, this is the distance from center to any vertex.
    size = 0.5
    # The vertical distance (dy) between the centers of two consecutive hexagons in a column.
    dy = size * np.sqrt(3)
    # The horizontal distance (dx) between the centers of two consecutive hexagons in a row.
    dx = size * 1.8

    # Colors for the players
    colors = {1: 'red', -1: 'blue', 0: 'white'}

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            center_x = dx * (j + i / 2)
            center_y = dy * (matrix.shape[0] - 1 - i) * 0.75
            color = colors[matrix[i, j]]
            draw_hexagon(ax, (center_x, center_y), size=size, color=color)

    buffer_space = size - 4
    num_rows, num_cols = matrix.shape
    xmin = -size - buffer_space - 3.5
    xmax = dx * num_cols - (dx - size) - buffer_space + 1.5
    ymin = -dy / 2 - buffer_space - 7
    ymax = dy * num_rows - (dy / 2) - 2

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    fig_width, fig_height = ax.figure.get_size_inches()
    ax.figure.set_size_inches(fig_width, fig_height, forward=True)

    if player is not None and draw is False:
        win_msg = f'Player {player} Wins!'
        ax.text(xmax / 2, ymax + size, win_msg, fontsize=12, ha='center', va='bottom')
    elif player is not None and draw is True:
        win_msg = f'DRAW!'
        ax.text(xmax / 2, ymax + size, win_msg, fontsize=12, ha='center', va='bottom')
    rand = random.randint(0, 120123123)
    filename = f'images\\player{player}_{counter}.png'
    # Save the figure to a file
    plt.savefig(filename)
    plt.close()  # Close the figure to free up memory

    return filename  # Return the filename of the saved image

def write_game_outcome_to_file(player, counter, win=True):
    outcome_msg = f'Game {counter}: {"Player " + str(player) + " wins" if win else "Draw"}\n'
    with open('game_outcomes.txt', 'a') as file:
        file.write(outcome_msg)