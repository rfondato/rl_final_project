from typing import Type, Tuple

import numpy as np
from boardgame2 import BoardGameEnv


class Action():
    def __init__(self, action, player):
        self.action = action
        self.player = player

    def __str__(self):
        return str((self.action[0], self.action[1], self.player))

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.action[0] == other.action[0] and self.action[1] == other.action[1] and self.player == other.player

    def __hash__(self):
        return hash((self.action[0], self.action[1]))
    

class CustomReversiState:
    def __init__(self,
                 env: BoardGameEnv,
                 currentState: Tuple[np.ndarray, int],
                 reward: float = None,
                 done: bool = False
                 ):
        self.board, self.currentPlayer = currentState
        self.env = env
        self.reward = reward
        self.done = done

    def __eq__(self, other):
        return (self.board == other.board).all()

    def get_actions_mask(self):
        return self.env.get_valid((self.board, self.currentPlayer))

    def getCurrentPlayer(self):
        return self.currentPlayer

    def getPossibleActions(self):
        possible_actions = []
        valid_actions = np.argwhere(self.env.get_valid((self.board, self.currentPlayer)))
        for action in valid_actions:
            possible_actions.append(Action(action, self.currentPlayer))
        return possible_actions

    def takeAction(self, action):
        state, reward, done, _ = self.env.next_step(
            (self.board, self.currentPlayer), (action.action[0], action.action[1]))
        return CustomReversiState(self.env, state, reward, done)

    def isTerminal(self):
        return self.done

    def getReward(self):
        return self.reward
