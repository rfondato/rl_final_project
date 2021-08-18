import random
from abc import ABC, abstractmethod
from typing import Union, Optional, Dict, Any

import numpy as np
from boardgame2 import BoardGameEnv, ReversiEnv
from stable_baselines3 import PPO

from multi_process_mcts import MultiProcessMonteCarlo, model_policy
from reversi_state import CustomReversiState
import os


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

    def setPlayer(self, new_player):
        self.player = new_player

    def __str__(self):
        return self.__class__.__name__


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
                 levelLimit: int = None,
                 device: str = 'auto',
                 mtcs_n_processes: int = None,
                 **custom_kwargs: Optional[Dict[str, Any]]  # Make subclass constructor generic
                 ):
        super().__init__(player, env, flatten_action)

        if model_path is None:
            raise Exception("model_path cannot be None")

        self.model = PPO.load(model_path, device=device)
        self.model_path = model_path
        self.deterministic = deterministic
        self.only_valid = only_valid
        self.mcts = mcts
        self.levelLimit = levelLimit
        self.mtcs_n_processes = mtcs_n_processes

    def predict(self, board: np.ndarray) -> Union[int, np.ndarray]:
        if self.mcts:
            action = self._get_action_with_mcts(board)
            action = action.action
            if self.flatten_action:
                return action[0] * self.board_shape + action[1]
            else:
                return action
        else:
            obs = self.player * board
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

    def get_model_path(self):
        return self.model_path

    def _get_action_with_mcts(self, board: np.ndarray) -> Union[int]:
        searcher = MultiProcessMonteCarlo(levelLimit=self.levelLimit,
                                          n_processes=self.mtcs_n_processes,
                                          explorationConstant=0.0,
                                          rolloutPolicy=model_policy(self.model))

        state = CustomReversiState(self.env, (board, self.player))
        return searcher.search(initialState=state)

    def __str__(self):
        monte_carlo = f"(With Monte-Carlo, Prediction Depth:{self.levelLimit})" if self.mcts else "(No Monte-Carlo)"
        return self.__class__.__name__ + monte_carlo


class RandomStrategyPlayer(BasePlayer):
    def __init__(self,
                 player: int = 1,
                 env: BoardGameEnv = None,
                 flatten_action: bool = False,
                 only_valid: bool = True,
                 model_path: str = None,
                 deterministic: bool = True,
                 device: str = 'auto',
                 verbose: bool = False,
                 **custom_kwargs: Optional[Dict[str, Any]]  # Make subclass constructor generic
                 ):
        super().__init__(player, env, flatten_action)

        if model_path is None and verbose:
            print("No model_path set. Strategies will be Random or Greedy.")

        if model_path is not None and not os.path.isdir(model_path):
            raise Exception("model_path should be a directory")

        self.model_path = model_path
        self.only_valid = only_valid
        self.deterministic = deterministic
        self.device = device
        self.strategy = self._create_strategy()
        if verbose:
            print(self)

    def predict(self, board: np.ndarray) -> Union[int, np.ndarray]:
        return self.strategy.predict(board)

    def setPlayer(self, new_player):
        super().setPlayer(new_player)
        self.strategy.setPlayer(new_player)

    def __str__(self):
        model_path = f" (Path: {self.strategy.get_model_path()})" if isinstance(self.strategy, TorchPlayer) else ""
        return self.__class__.__name__ + " - Selected Strategy: " + self.strategy.__class__.__name__ + model_path

    def _create_strategy(self):
        strategies = ["random", "greedy"]
        if self.model_path is not None:
            strategies += [f for f in os.listdir(self.model_path) if os.path.isfile(os.path.join(self.model_path, f))]

        strategy = random.choice(strategies)

        if strategy == "random":
            return RandomPlayer(player=self.player, env=self.env, flatten_action=self.flatten_action)
        elif strategy == "greedy":
            return GreedyPlayer(player=self.player, env=self.env, flatten_action=self.flatten_action)
        else:
            return TorchPlayer(player=self.player,
                               env=self.env,
                               flatten_action=self.flatten_action,
                               model_path=os.path.join(self.model_path, strategy),
                               deterministic=self.deterministic,
                               only_valid=self.only_valid,
                               device=self.device
                               )


if __name__ == "__main__":
    player = RandomStrategyPlayer(player=1, env=ReversiEnv(board_shape=8), model_path="./training_opps/")
    print(player)
