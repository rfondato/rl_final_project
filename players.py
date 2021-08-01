from typing import Union, Optional, Dict, Any

import numpy as np
from boardgame2 import BoardGameEnv
from abc import ABC, abstractmethod

from stable_baselines3 import PPO


class BasePlayer(ABC):
    def __init__(self,
                 player: int = 1,
                 env: BoardGameEnv = None,
                 flatten_action: bool = False
                 ):
        if env is None:
            print("Environment cannot be None")

        self.env = env
        self.player = player  # player number. 1 o -1
        self.flatten_action = flatten_action
        self.board_shape = self.env.board.shape[0]

    @abstractmethod
    def predict(self, board: np.ndarray) -> Union[int, np.ndarray]:
        """
        Returns the action to play given a board.
        :param board: Numpy array of board_shape x board_shape with current board
        :return: Numpy array of dimension 2 with row and column to play if flatten_action is False.
                If flatten_action is True, it returns an int with the slot number.
        """


class GreedyPlayer(BasePlayer):
    def __init__(self,
                 player: int = 1,
                 env: BoardGameEnv = None,
                 flatten_action: bool = False,
                 **custom_kwargs: Optional[Dict[str, Any]]  # Make subclass constructor generic
                 ):
        super().__init__(player, env, flatten_action)

    def predict(self, board: np.ndarray) -> Union[int, np.ndarray]:
        valid_actions = np.argwhere(self.env.get_valid((board, self.player)) == 1)
        if len(valid_actions) == 0:
            action = self.env.PASS
        else:
            moves_score = []
            for a in valid_actions:
                next_state, _, _, _ = self.env.next_step((board, self.player), a)
                moves_score.append(next_state[0].sum() * self.player)
            best_score = max(moves_score)
            best_actions = valid_actions[np.array(moves_score) == best_score]
            action = best_actions[np.random.randint(len(best_actions))]
        if self.flatten_action:
            return action[0] * self.board_shape + action[1]
        else:
            return action


class RandomPlayer(BasePlayer):
    def __init__(self,
                 player: int = 1,
                 env: BoardGameEnv = None,
                 flatten_action: bool = False,
                 **custom_kwargs: Optional[Dict[str, Any]]  # Make subclass constructor generic
                 ):
        super().__init__(player, env, flatten_action)

    def predict(self, board: np.ndarray) -> Union[int, np.ndarray]:
        valid_actions = np.argwhere(self.env.get_valid((board, self.player)) == 1)
        if len(valid_actions) == 0:
            action = self.env.PASS
        else:
            action = valid_actions[np.random.randint(len(valid_actions))]
        if self.flatten_action:
            return action[0] * self.board_shape + action[1]
        else:
            return action


class DictPolicyPlayer(BasePlayer):
    def __init__(self,
                 player: int = 1,
                 env: BoardGameEnv = None,
                 flatten_action: bool = False,
                 dict_folder: str = 'mdp/pi_func_only_winner.npy',
                 **custom_kwargs: Optional[Dict[str, Any]]  # Make subclass constructor generic
                 ):
        super().__init__(player, env, flatten_action)
        self.pi_dict = np.load(dict_folder, allow_pickle=True).item()

    def predict(self, board: np.ndarray) -> Union[int, np.ndarray]:
        board_tuple = tuple((board * self.player).reshape(-1))
        action = self.pi_dict[board_tuple]
        if self.flatten_action:
            return action
        else:
            return np.array([action // self.board_shape, action % self.board_shape])


class TorchPlayer(BasePlayer):
    def __init__(self,
                 player: int = 1,
                 env: BoardGameEnv = None,
                 flatten_action: bool = False,
                 model_path: str = None,
                 deterministic: bool = True,
                 only_valid: bool = True,
                 mcts: bool = False,
                 iterationLimit: int = None,
                 timeLimit: int = None,
                 **custom_kwargs: Optional[Dict[str, Any]]  # Make subclass constructor generic
                 ):
        super().__init__(player, env, flatten_action)

        if model_path is None:
            raise Exception("model_path cannot be None")

        self.model = PPO.load(model_path)
        self.deterministic = deterministic
        self.only_valid = only_valid
        self.mcts = mcts
        self.iterationLimit = iterationLimit
        self.timeLimit = timeLimit

    def predict(self, board: np.ndarray) -> Union[int, np.ndarray]:
        obs = board if (self.player == 1) else -board
        if self.only_valid:
            obs = [obs, self.env.get_valid((obs, 1))]
        # The model expects a batch of observations.
        # Make a batch of 1 obs
        obs = [obs]
        action = self.model.predict(obs, deterministic=self.deterministic)[0]
        if self.flatten_action:
            return action
        else:
            return np.array([action // self.board_shape, action % self.board_shape])
