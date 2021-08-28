import os
import random
from datetime import datetime
from typing import Type, Union, Optional, List, Dict

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv

from custom_features_extractor import CustomBoardExtractor
from custom_policies import CustomActorCriticPolicy
from multi_env import make_reversi_vec_env, SelfPlayEnv
from players import RandomPlayer, BasePlayer, GreedyPlayer, TorchPlayer


class CustomReversiModel:

    def __init__(self,
                 board_shape: int = 8,
                 n_envs: int = 8,
                 local_player: Union[Type[BasePlayer], str] = RandomPlayer,
                 learning_rate=2e-4,
                 n_steps: int = 2048,
                 gamma: float = 0.99,
                 ent_coef: float = 0.0,
                 gae_lambda: float = 0.95,
                 batch_size: int = 64,
                 n_epochs: int = 10,
                 net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
                 load_from_path: str = None,
                 use_previous_saved_params: bool = False,
                 features_extractor: BaseFeaturesExtractor = CustomBoardExtractor,
                 path_local_player: str = None,
                 device_local_player: str = "auto",
                 verbose: bool = False
                 ):
        if isinstance(local_player, str) and local_player != "multiple":
            raise Exception("local_player should be a BasePlayer class or 'multiple'")

        self.board_shape = board_shape
        self.verbose = verbose

        # Train Params
        self.learning_rate = learning_rate
        self.n_envs = n_envs
        self.n_steps = n_steps
        self.gamma = gamma
        self.ent_coef = ent_coef
        self.gae_lambda = gae_lambda
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.features_extractor = features_extractor
        self.net_arch = net_arch

        # Model to load
        self.path = load_from_path
        self.prev_params = use_previous_saved_params

        # Opponent
        self.local_player = local_player
        self.local_path = path_local_player
        self.device_local = device_local_player

        self.new_model_save_path = self._get_model_save_path()

        self.chosen_players = None
        self._create_train_env()
        self._create_eval_env()
        self._create_callbacks()

        if verbose and (self.chosen_players is not None):
            print("Chosen opponents: ")
            for player in self.chosen_players:
                print(player.__name__ if not isinstance(player, str) else f"TorchPlayer:{player}")

        if load_from_path is not None:
            self._load_model()
        else:
            self._create_new_model()

    def learn(self, total_timesteps=int(1e6)):
        self.model.learn(total_timesteps=total_timesteps, callback=self.callbacks)

    def get_new_model_save_path(self):
        return self.new_model_save_path

    def _create_new_model(self):
        self.model = PPO(
            CustomActorCriticPolicy,
            self.env,
            verbose=1 if self.verbose else 0,
            tensorboard_log='tensorboard_log',
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            learning_rate=self.learning_rate,
            ent_coef=self.ent_coef,
            n_epochs=self.n_epochs,
            n_steps=self.n_steps,
            batch_size=self.batch_size,
            policy_kwargs={
                'features_extractor_class': self.features_extractor,
                'net_arch': self.net_arch
            }
        )

    def _load_model(self):
        custom_objects = dict()
        if not self.prev_params:
            custom_objects = {
                "n_steps": self.n_steps,
                "gamma": self.gamma,
                "ent_coef": self.ent_coef,
                "gae_lambda": self.gae_lambda,
                "n_epochs": self.n_epochs,
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size
            }
        self.model = PPO.load(path=self.path, env=self.env, custom_objects=custom_objects)

    def _create_callbacks(self):
        self.callbacks = [EvalCallback(
            eval_env=self.eval_env,
            eval_freq=1_000,
            n_eval_episodes=100,
            deterministic=True,
            verbose=1 if self.verbose else 0,
            best_model_save_path=self.new_model_save_path
        )]

    def _get_env_args(self, eval=False):
        if self.local_player == "multiple":
            return self._get_multiple_players_eval_env_args() if eval else self._get_multiple_players_train_env_args()
        else:
            return self._create_env_args(self.local_player, self.local_path)

    def _get_multiple_players_train_env_args(self):
        self.chosen_players = []
        args = []
        local_players = self._get_local_players()
        for i in range(self.n_envs):
            local_player = random.choice(local_players)
            self.chosen_players.append(local_player)
            local_players.remove(local_player)
            args.append(
                self._create_env_args(
                    self._get_local_player_class(local_player),
                    os.path.join(self.local_path, local_player) if isinstance(local_player, str) else None
                )
            )
            if len(local_players) == 0:
                local_players = self._get_local_players()
        return args

    def _get_multiple_players_eval_env_args(self):
        args = []
        for i in range(self.n_envs):
            local_player = self.chosen_players[i]
            args.append(
                self._create_env_args(
                    self._get_local_player_class(local_player),
                    os.path.join(self.local_path, local_player) if isinstance(local_player, str) else None
                )
            )
        return args

    def _create_env_args(self, local_player_cls, local_player_path):
        return {
            'board_shape': self.board_shape,
            'local_player_cls': local_player_cls,
            'mask_channel': True,
            'local_player_kwargs': {
                'model_path': local_player_path,
                'device': self.device_local,
                'verbose': self.verbose,
                'deterministic': False
            }
        }

    def _get_local_players(self):
        local_players = [RandomPlayer, GreedyPlayer]
        if self.local_path is not None:
            local_players += [f for f in os.listdir(self.local_path) if os.path.isfile(os.path.join(self.local_path, f))]

        return local_players

    def _get_local_player_class(self, local_player):
        if isinstance(local_player, str):
            return TorchPlayer
        else:
            return local_player

    def _create_eval_env(self):
        self.eval_env = make_reversi_vec_env(
            SelfPlayEnv,
            n_envs=self.n_envs,
            vec_env_cls=SubprocVecEnv,
            env_kwargs=self._get_env_args(eval=True)
        )

    def _create_train_env(self):
        self.env = make_reversi_vec_env(
            SelfPlayEnv,
            n_envs=self.n_envs,
            vec_env_cls=SubprocVecEnv,
            env_kwargs=self._get_env_args(eval=False)
        )

    def _get_model_save_path(self):
        prefix = 'Reversi_PPO'
        suffix = str(datetime.now()).replace(" ", "_")

        model_name = f'{prefix}_{self.board_shape}by{self.board_shape}_{suffix}'

        return f'./models/{model_name}'
