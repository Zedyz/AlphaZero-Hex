import csv
import numpy as np
import torch
from node import Node


class MCTSP:
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model
        self.node_depth_counts = {}

    @torch.no_grad()
    def search(self, states, spGames):
        #define root node

        policy, _ = self.model(
            torch.tensor(self.game.get_encoded_state(states), device=self.model.device)
        )

        policy = torch.softmax(policy, axis=1).cpu().numpy()

        policy = (1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] * np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_size, size=policy.shape[0])

        for i, spg in enumerate(spGames):
            spg_policy = policy[i]
            valid_moves = self.game.get_valid_moves(states[i])
            spg_policy *= valid_moves
            spg_policy /= np.sum(spg_policy)
            spg.root = Node(self.game, self.args, states[i], visit_count=1, depth=0)
            self._increment_node_depth_count(0)
            spg.root.expand(spg_policy)


        for search in range(self.args['num_searches']):
            for spg in spGames:
                spg.node = None
                node = spg.root

                while node.is_fully_expanded():
                    node = node.select()


                value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken)
                value = self.game.get_opponent_value(value)

                if is_terminal:
                    node.backpropagate(value)
                else:
                    spg.node = node

            expandable_spGames = [mappingIdx for mappingIdx in range(len(spGames)) if spGames[mappingIdx].node is not None]

            if len(expandable_spGames) > 0:
                states = np.stack([spGames[mappingIdx].node.state for mappingIdx in expandable_spGames])
                policy, value = self.model(
                    torch.tensor(self.game.get_encoded_state(states), device=self.model.device)

                )
                policy = torch.softmax(policy, axis=1).cpu().numpy()
                value = value.cpu().numpy()

            for i, mappingIdx in enumerate(expandable_spGames):
                node = spGames[mappingIdx].node
                spg_policy, spg_value = policy[i], value[i]


                valid_moves = self.game.get_valid_moves(node.state)
                spg_policy *= valid_moves
                spg_policy /= np.sum(spg_policy)
                node.expand(spg_policy)
                node.backpropagate(spg_value)

                #backpropagation

    def _increment_node_depth_count(self, depth):
        if depth in self.node_depth_counts:
            self.node_depth_counts[depth] += 1
        else:
            self.node_depth_counts[depth] = 1

    def save_depth_counts_to_csv(self, filename="depth_counts.csv"):
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            for depth, count in sorted(self.node_depth_counts.items()):
                writer.writerow([depth, count])