import time
from hex import Hex
from net import ResNet
from mcts import MCTS
import torch
torch.manual_seed(0)
import matplotlib.pyplot as plt
import numpy as np
global_counter = 0
torch.manual_seed(0)
import pygame
import math
import sys

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 1200, 800
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)

# Screen setup
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Hexagon Grid Game")
clock = pygame.time.Clock()
device = torch.device("cuda")


def interpolate_color(low_color, high_color, val, max_val):
    """ Interpolate between low_color and high_color based on val and max_val. """

    def interpolate(c1, c2, factor):
        return c1 + (c2 - c1) * factor

    factor = val / max_val
    return (
        int(interpolate(low_color[0], high_color[0], factor)),
        int(interpolate(low_color[1], high_color[1], factor)),
        int(interpolate(low_color[2], high_color[2], factor))
    )


def draw_hexagon_with_label(surface, center, size, matrix_value, prob):
    """Draw a flat-top hexagon with a color that indicates both owner and probability."""
    # Define colors
    high_color = (255, 165, 0)  # Orange for high probability
    low_color = (255, 255, 255)  # White for low probability
    player1_color = (255, 0, 0)  # Red for player 1
    player2_color = (0, 0, 255)  # Blue for player 2

    # Select base color based on ownership
    if matrix_value == 1:
        base_color = player1_color
    elif matrix_value == -1:
        base_color = player2_color
    else:
        # For neutral tiles, interpolate between low_color and high_color based on probability
        base_color = blend_colors(low_color, high_color, prob)

    # Draw hexagon
    points = []
    for angle in range(0, 360, 60):
        rad = math.radians(angle + 30)
        x = center[0] + size * math.cos(rad)
        y = center[1] + size * math.sin(rad)
        points.append((x, y))
    pygame.draw.polygon(surface, base_color, points)
    pygame.draw.polygon(surface, BLACK, points, 1)  # Black outline for visibility

    # Display the probability text only for neutral hexagons
    if matrix_value == 0:
        font = pygame.font.Font(None, 24)
        text = font.render(f"{prob:.2f}", True, BLACK)
        text_rect = text.get_rect(center=center)
        surface.blit(text, text_rect)



def toggle_color(matrix, row, col):
    """ Toggle the color of the hexagon at the given matrix position """
    if 0 <= row < 11 and 0 <= col < 11:
        if matrix[row][col] == 0:
            matrix[row][col] = 1
        elif matrix[row][col] == 1:
            matrix[row][col] = -1
        else:
            matrix[row][col] = 0

def get_hex_coordinates(row, col, size, dx, dy):
    """ Calculate the pixel coordinates of the hexagon based on its row and column """
    x_offset = col * dx + row * (dx / 2)  + 100
    y = row * dy + 100
    return (x_offset + size, y + size * np.sqrt(3) / 2)  # Center of the hexagon

def blend_colors(color1, color2, blend_factor):
    """ Blend two colors together based on blend_factor between 0 and 1."""
    return tuple([
        int(color1[i] * (1 - blend_factor) + color2[i] * blend_factor) for i in range(3)
    ])

def get_hex_at_pos(pos, dx, dy):
    """ Find the row, column of the hexagon that contains the point, accounting for consistent right staggering. """
    x, y = pos
    y_adjusted = y - 100
    x_adjusted = x - 100
    row = int(y_adjusted // dy)
    col = int((x_adjusted - row * (dx / 2)) // dx)  # Reverse calculation of the row-based right offset
    return row, col


def calculate_index(row, col, num_cols=11):
    """ Calculate the linear index for the hexagon from its row and column """
    return row * num_cols + col


def draw_border_hexagons(surface, matrix, size, dx, dy):
    """ Draw border hexagons with specific colors to indicate player goals. """
    # Define border colors for clarity
    border_color_player1 = RED  # Top and bottom for Player 1
    border_color_player2 = BLUE  # Left and right for Player 2

    rows, cols = len(matrix), len(matrix[0])

    # Draw top and bottom borders for Player 1
    for col in range(cols):
        top_hex_center = get_hex_coordinates(-1, col, size, dx, dy)  # Adjusted to be outside the grid
        bottom_hex_center = get_hex_coordinates(rows, col, size, dx, dy)
        draw_hexagon(surface, top_hex_center, size, border_color_player1)
        draw_hexagon(surface, bottom_hex_center, size, border_color_player1)

    # Draw left and right borders for Player 2
    for row in range(rows):
        left_hex_center = get_hex_coordinates(row, -1, size, dx, dy)
        right_hex_center = get_hex_coordinates(row, cols, size, dx, dy)
        draw_hexagon(surface, left_hex_center, size, border_color_player2)
        draw_hexagon(surface, right_hex_center, size, border_color_player2)


def draw_hexagon(surface, center, size, color):
    """ Helper function to draw a single hexagon given a center, size, and color. """
    points = []
    for angle in range(30, 390, 60):  # Start at 30 degrees
        rad = math.radians(angle)
        x = center[0] + size * math.cos(rad)
        y = center[1] + size * math.sin(rad)
        points.append((x, y))
    pygame.draw.polygon(surface, color, points)
    pygame.draw.polygon(surface, BLACK, points, 1)  # Outline for visibility


def plot_hex_grid(matrix, mcts_probs, player=None, fig=None, draw = False, counter = None):
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

            if matrix[i, j] == 0 and mcts_probs is not None:
                prob_index = i * matrix.shape[1] + j
                prob_text = f'{mcts_probs[prob_index]:.2f}'
                ax.text(center_x, center_y, prob_text, ha='center', va='center', fontsize=8, color='black')


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

    global global_counter
    global_counter += 1

    filename = f'images\\player{player}_{global_counter}.png'
    # Save the figure to a file
    plt.savefig(filename)
    plt.close()  # Close the figure to free up memory

    return filename  # Return the filename of the saved image

def draw_text(surface, text, position, font_size=24, color=BLACK):
    font = pygame.font.Font(None, font_size)
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect(center=position)
    surface.blit(text_surface, text_rect)



def play_game(game, model1, model2, args1, args2):
    player = 1
    mcts1 = MCTS(game, args1, model1)
    mcts2 = MCTS(game, args2, model2)
    state = game.get_initial_state()
    size = 30
    dy = size * np.sqrt(3)
    dx = size * 1.8
    matrix = [[0 for _ in range(11)] for _ in range(11)]
    prob_matrix = [0.0] * 121
    screen.fill(WHITE)
    draw_border_hexagons(screen, matrix, size, dx, dy)

    while True:
        # Check for events
        draw_border_hexagons(screen, matrix, size, dx, dy)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        if player == 1:
            # Display current state with probabilities for player 1
            mcts_probs1 = mcts1.search(state)
            for index, prob in enumerate(mcts_probs1):
                prob_matrix[index] = prob

            # Visualize probabilities
            for row in range(11):
                for col in range(11):
                    index = calculate_index(row, col, 11)
                    center = get_hex_coordinates(row, col, size, dx, dy)
                    matrix_value = matrix[row][col]
                    prob = prob_matrix[index]
                    draw_hexagon_with_label(screen, center, size, matrix_value, prob)
            pygame.display.flip()

            # Wait for player 1 to make a move via mouse click
            move_made = False
            while not move_made:
                for event in pygame.event.get():
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        x, y = pygame.mouse.get_pos()
                        r, c = get_hex_at_pos((x, y), dx, dy)
                        if r < 0 or r >= 11 or c < 0 or c >= 11:
                            continue
                        action = calculate_index(r, c)
                        if game.get_valid_moves(state)[action]:
                            state = game.get_next_state(state, action, player)
                            matrix[r][c] = 1
                            player = game.get_opponent(player)
                            move_made = True
                            break
                    elif event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()

        else:  # Player 2 (automated)
            # Display current state with probabilities for player 2
            mcts_probs2 = mcts2.search(state)
            for index, prob in enumerate(mcts_probs2):
                prob_matrix[index] = prob

            # Visualize probabilities
            for row in range(11):
                for col in range(11):
                    index = calculate_index(row, col, 11)
                    center = get_hex_coordinates(row, col, size, dx, dy)
                    matrix_value = matrix[row][col]
                    prob = prob_matrix[index]
                    draw_hexagon_with_label(screen, center, size, matrix_value, prob)
            pygame.display.flip()
            time.sleep(1)  # Display probabilities briefly

            # Select and perform the best move
            action = np.argmax(mcts_probs2)
            if game.get_valid_moves(state)[action]:
                state = game.get_next_state(state, action, player)
                row, col = divmod(action, 11)
                matrix[row][col] = -1
                player = game.get_opponent(player)
            print(state)
        # Check for game end
        value, is_terminal = game.get_value_and_terminated(state, action)
        if is_terminal:
            player = game.get_opponent(player)
            print(f"Game over, player {player} wins!")
            time.sleep(50)
            break
        draw_border_hexagons(screen, matrix, size, dx, dy)

        pygame.display.flip()
        clock.tick(30)

    return player


def main():
    game = Hex()

    #player 1
    model1 = ResNet(game, 9, 128, device)
    model_path_1 = "training1/model_6_Hex.pt"
    model1.load_state_dict(torch.load(model_path_1, map_location=device))
    model1.to(device)
    model1.eval()

    #player 2
    model2 = ResNet(game, 9, 128, device)
    model_path_2 = "training1/model_6_Hex.pt"
    model2.load_state_dict(torch.load(model_path_2, map_location=device))
    model2.to(device)
    model2.eval()

    args1 = {
        'C': 1.5,
        'num_searches': 300,
        'num_iterations': 10,
        'num_selfPlay_iterations': 2000,
        'num_parallel_games': 200,
        'num_epochs': 5,
        'batch_size': 512,
        'temperature': 1.25,
        'dirichlet_epsilon': 0.25,
        'dirichlet_alpha': 0.3
    }

    args2 = {
        'C': 1.5,
        'num_searches': 1600,
        'num_iterations': 10,
        'num_selfPlay_iterations': 2000,
        'num_parallel_games': 200,
        'num_epochs': 5,
        'batch_size': 512,
        'temperature': 1.25,
        'dirichlet_epsilon': 0.25,
        'dirichlet_alpha': 0.3
    }

    for _ in range(500):
        winner = play_game(game, model1, model2, args1, args2)
        print(winner)

if __name__ == "__main__":
    main()