from __future__ import division

import math
import random
from typing import Callable, Any

import dill
from boardgame2 import ReversiEnv
from stable_baselines3.common.base_class import BaseAlgorithm

from reversi_state import Action, CustomReversiState
from multiprocessing import Pool, cpu_count
import torch as th

th.multiprocessing.set_start_method('spawn', force=True)


def run_dill_encoded(encoded_payload):
    _f, _args = dill.loads(encoded_payload)
    return _f(*_args)


def run_async(pool: Pool, f: Callable, args: Any):
    payload = dill.dumps((f, args))
    return pool.apply_async(run_dill_encoded, (payload,))


def randomPolicy(state):
    while not state.isTerminal():
        try:
            action = random.choice(state.getPossibleActions())
        except IndexError:
            raise Exception("Non-terminal state has no possible actions: " + str(state))
        state = state.takeAction(action)
    return state.getReward()


def model_policy(model: BaseAlgorithm = None):
    def sampleModel(state):
        while not state.isTerminal():
            try:
                obs = [state.board * state.getCurrentPlayer(), state.get_actions_mask()]
                # Sample actions using the model predict's probabilities
                action = model.predict([obs], deterministic=True)[0][0]
                coded_action = [action // state.board.shape[0], action % state.board.shape[0]]
                action = Action(coded_action, state.getCurrentPlayer())
            except IndexError:
                raise Exception("Non-terminal state has no possible actions: " + str(state))
            state = state.takeAction(action)
        return state.getReward()

    return sampleModel


class TreeNode:
    def __init__(self, state: CustomReversiState, parent):
        self.state = state
        self.isTerminal = state.isTerminal()
        self.isFullyExpanded = self.isTerminal
        self.parent = parent
        self.numVisits = 0
        self.totalReward = 0
        self.children = {}

    def expand(self):
        actions = self.state.getPossibleActions()
        for action in actions:
            new_node = TreeNode(self.state.takeAction(action), self)
            self.children[action] = new_node
        return self.children.values()

    def backPropagate(self, reward):
        node = self
        while node is not None:
            node.numVisits += 1
            node.totalReward += reward
            node = node.parent

    def findNodeByState(self, state):
        if self.state == state:
            return self
        elif len(self.children) == 0:
            return None
        else:
            for n in self.children.values():
                node_found = n.findNodeByState(state)
                if node_found is not None:
                    return node_found

    def getBestChild(self, explorationConstant=0.0):
        best_value = float("-inf")
        best_nodes = []
        for key, child in self.children.items():
            node_value = self.getNodeValue(child, explorationConstant)

            if node_value > best_value:
                best_value = node_value
                best_nodes = [child]
            elif node_value == best_value:
                best_nodes.append(child)

        return random.choice(best_nodes)

    def getNodeValue(self, child, explorationConstant):
        return self.state.getCurrentPlayer() * child.totalReward / child.numVisits + \
               explorationConstant * math.sqrt(2 * math.log(self.numVisits) / child.numVisits)

    def __eq__(self, other):
        return self.state == other.state

    def __str__(self):
        s = ["totalReward: %s" % self.totalReward,
             "numVisits: %d" % self.numVisits,
             "isTerminal: %s" % self.isTerminal,
             "possibleActions: %s" % (self.children.keys())]
        return "%s: {%s}" % (self.__class__.__name__, ', '.join(s))


class MultiProcessMonteCarlo:
    def __init__(self,
                 levelLimit: int = 1,
                 n_processes: int = None,
                 explorationConstant: float = 1 / math.sqrt(2),
                 rolloutPolicy: Callable = randomPolicy):
        self.searchLimit = levelLimit
        self.currentLevel = 1
        self.n_processes = n_processes if (n_processes is not None) else cpu_count()
        self.explorationConstant = explorationConstant
        self.rollout = rolloutPolicy
        self.root = None

    def search(self, initialState: CustomReversiState):
        self.root = TreeNode(initialState, None)

        with Pool(self.n_processes) as p:
            jobs = []
            leaves = self.getLeaves(self.root)
            for leaf in leaves:
                jobs.append(run_async(p, self.calculateReward, [leaf]))

            for job in jobs:
                state, reward = job.get()
                node = self.root.findNodeByState(state)
                node.backPropagate(reward)

        best_child = self.root.getBestChild(self.explorationConstant)
        action = (action for action, node in self.root.children.items() if node is best_child).__next__()
        return action

    def getLeaves(self, node: TreeNode, lvl: int = 1):
        if lvl > self.searchLimit:
            return [node]

        leaves = []
        for c in node.expand():
            leaves += self.getLeaves(c, lvl + 1)
        return leaves

    def calculateReward(self, node: TreeNode):
        reward = self.rollout(node.state)
        return node.state, reward


def test_monte_carlo(board_shape=4, levelLimit=1, n_processes=1, exploration_constant=0.0, rolloutPolicy=randomPolicy):
    env = ReversiEnv(board_shape=board_shape)
    state = CustomReversiState(env, env.reset())
    searcher = MultiProcessMonteCarlo(levelLimit=levelLimit, n_processes=n_processes,
                                      explorationConstant=exploration_constant, rolloutPolicy=rolloutPolicy)
    action = searcher.search(initialState=state)
    print("Action:", action)


if __name__ == "__main__":
    test_monte_carlo(levelLimit=2, n_processes=4)
