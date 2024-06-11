import torch
from AlphaZeroP import AlphaZeroP

torch.manual_seed(0)
from hex import Hex
import numpy as np
import torch
from net import ResNet
from mcts import MCTS


def play_game():
    game = Hex()
    player = 1
    device = torch.device("cuda")

    args = {
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

    model1 = ResNet(game, 9, 128, device=device)
    model1.load_state_dict(torch.load("training1/model_6_Hex.pt", map_location=device))
    model1.eval()

    model2 = ResNet(game, 9, 128, device=device)
    model2.load_state_dict(torch.load("training1/model_0_Hex.pt", map_location=device))
    model2.eval()

    mcts1 = MCTS(game, args, model1)

    mcts2 = MCTS(game, args, model2)

    state = game.get_initial_state()

    while True:
        print(state)

        if player == 1:
            neutral_state1 = game.change_perspective(state, player)
            mcts_probs1 = mcts1.search(neutral_state1)
            action = np.argmax(mcts_probs1)

        else:
            neutral_state2 = game.change_perspective(state, player)
            mcts_probs2 = mcts2.search(neutral_state2)
            action = np.argmax(mcts_probs2)

        state = game.get_next_state(state, action, player)

        value, is_terminal = game.get_value_and_terminated(state, action)

        if is_terminal:
            print(state)
            if value == 1:
                print(player, "won")
            else:
                print("draw")  # safety check for draw
            break

        player = game.get_opponent(player)


def main():
    game = Hex()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet(game, 9, 128, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    args_config1 = {
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

    alpaZero = AlphaZeroP(model, optimizer, game, args_config1)
    alpaZero.learn()


if __name__ == "__main__":
    main()
