from collections import deque
import numpy as np
import torch
torch.manual_seed(0)
from visualize import plot_hex_grid


class Hex:
    def __init__(self):
        self.row_count = 11
        self.column_count = 11
        self.action_size = self.row_count * self.column_count

    def __repr__(self):
        return "Hex"

    def get_initial_state(self):
        return np.zeros((self.row_count, self.column_count))

    def get_next_state(self, state, action, player):
        # copy to avoid mutating the original state
        new_state = state.copy()
        row, column = divmod(action, self.column_count)

        # apply the action if the cell is empty
        if new_state[row, column] == 0:
            new_state[row, column] = player
        else:
            # raise an error or handle the case where the action is invalid
            raise ValueError(f"Cell ({row}, {column}) is already occupied.")

        return new_state

    def get_valid_moves(self, state):
        return (state.flatten() == 0).astype(np.uint8)

    def check_win(self, state, action):
        row, column = action // self.column_count, action % self.column_count
        player = state[row][column]
        if player == 0:
            return False  # empty cell

        visited = set()
        queue = deque()

        def get_neighbors(r, c):
            neighbors = []
            directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.row_count and 0 <= nc < self.column_count:
                    neighbors.append((nr, nc))
            return neighbors

        # initialize the queue with the correct side based on the player
        if player == 1:  # player 1 (red) starts from the top edge
            for c in range(self.column_count):
                if state[0][c] == player:
                    queue.append((0, c))
                    visited.add((0, c))
        else:  # player -1 (blue) starts from the left edge
            for r in range(self.row_count):
                if state[r][0] == player:
                    queue.append((r, 0))
                    visited.add((r, 0))

        # BFS to find a winning path
        while queue:
            r, c = queue.popleft()
            # check if Player 1 (red) reached the bottom edge
            if player == 1 and r == self.row_count - 1:
                return True
            # check if Player -1 (blue) reached the right edge
            if player == -1 and c == self.column_count - 1:
                return True
            for nr, nc in get_neighbors(r, c):
                if (nr, nc) not in visited and state[nr][nc] == player:
                    queue.append((nr, nc))
                    visited.add((nr, nc))

        return False

    def get_value_and_terminated(self, state, action):
        # create the numpy array from the list
        #state = np.array(state)
        row, column = action // self.column_count, action % self.column_count
        player = state[row][column]
        #plot_hex_grid(state, player=player, counter=self.counter)

        # First, check for a win
        win = self.check_win(state, action)

        if win:
            return 1, True  # Game ends with a win

        # then check if the board is full
        if np.all(state != 0):
            # before declaring a draw, do a final check for a win across the entire board
            for r in range(self.row_count):
                for c in range(self.column_count):
                    if state[r][c] != 0:  # Check only non-empty cells
                        if self.check_win(state, r * self.column_count + c):
                            # If a win is found, return the appropriate value
                            #plot_hex_grid(state, player=state[r][c], draw=False)
                            return 1, True
            # if no win was found, then raise an error
            #plot_hex_grid(state, player=0, draw=True)
            print("Full board detected on the last move by player {}, at position {},{}."
                  .format(player, row, column))
            print(state)
            raise ValueError("Detected a full board without a win, which should not be possible in Hex.")

        # If neither win nor draw, the game continues
        return 0, False

    def get_opponent(self, player):
        return -player

    def get_opponent_value(self, value):
        return -value

    def change_perspective(self, state, player):
        return state * player

    def get_encoded_state(self, state):
        encoded_state = np.stack(
            (state == -1, state == 0, state == 1)
        ).astype(np.float32)

        if len(state.shape) == 3:
            encoded_state = np.swapaxes(encoded_state, 0, 1)

        return encoded_state
