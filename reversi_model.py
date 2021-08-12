from datetime import datetime
from typing import Type

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

from custom_features_extractor import CustomBoardExtractor
from custom_policies import CustomActorCriticPolicy
from multi_env import make_reversi_vec_env, SelfPlayEnv
from players import RandomPlayer, BasePlayer


class CustomReversiModel:

    def __init__(self,
                 board_shape: int = 8,
                 n_envs: int = 8,
                 local_player: Type[BasePlayer] = RandomPlayer,
                 n_steps: int = 2048,
                 gamma: float = 0.99,
                 ent_coef: float = 0.0,
                 gae_lambda: float = 0.95,
                 batch_size: int = 64,
                 n_epochs: int = 10,
                 load_from_path: str = None,
                 use_previous_saved_params: bool = False,
                 path_local_player: str = None,
                 device_local_player: str = "auto"
                 ):
        self.board_shape = board_shape

        # Train Params
        self.n_envs = n_envs
        self.n_steps = n_steps
        self.gamma = gamma
        self.ent_coef = ent_coef
        self.gae_lambda = gae_lambda
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        # Model to load
        self.path = load_from_path
        self.prev_params = use_previous_saved_params

        # Opponent
        self.local_player = local_player
        self.local_path = path_local_player
        self.device_local = device_local_player

        self.new_model_save_path = self._get_model_save_path()

        self._create_train_env()
        self._create_eval_env()
        self._create_callbacks()

        if load_from_path is not None:
            self._load_model()
        else:
            self._create_new_model()

    def learn(self, total_timesteps=int(10e5)):
        self.model.learn(total_timesteps=total_timesteps, callback=self.callbacks)

    def get_new_model_save_path(self):
        return self.new_model_save_path

    def _create_new_model(self):
        self.model = PPO(
            CustomActorCriticPolicy,
            self.env,
            verbose=1,
            tensorboard_log='tensorboard_log',
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            ent_coef=self.ent_coef,
            n_epochs=self.n_epochs,
            n_steps=self.n_steps,
            batch_size=self.batch_size,
            policy_kwargs={'features_extractor_class': CustomBoardExtractor}
        )

    def _load_model(self):
        custom_objects = dict()
        if not self.prev_params:
            custom_objects = {
                "n_steps": self.n_steps,
                "gamma": self.gamma,
                "ent_coef": self.ent_coef,
                "gae_lambda": self.gae_lambda,
                "n_epochs": self.n_epochs
            }
        self.model = PPO.load(path=self.path, env=self.env, custom_objects=custom_objects)

    def _create_callbacks(self):
        self.callbacks = [EvalCallback(
            eval_env=self.eval_env,
            eval_freq=1_000,
            n_eval_episodes=500,
            deterministic=True,
            verbose=1,
            best_model_save_path=self.new_model_save_path
        )]

    def _create_eval_env(self):
        self.eval_env = make_reversi_vec_env(
            SelfPlayEnv,
            n_envs=1,
            vec_env_cls=SubprocVecEnv,
            env_kwargs={
                'board_shape': self.board_shape,
                'local_player_cls': self.local_player,
                'mask_channel': True,
                'local_player_kwargs': {
                    'model_path': self.local_path
                }
            }
        )

    def _create_train_env(self):
        self.env = make_reversi_vec_env(
            SelfPlayEnv,
            n_envs=self.n_envs,
            vec_env_cls=SubprocVecEnv,
            env_kwargs={
                'board_shape': self.board_shape,
                'local_player_cls': self.local_player,
                'mask_channel': True,
                'local_player_kwargs': {
                    'model_path': self.local_path,
                    'device': self.device_local
                }
            }
        )

    def _get_model_save_path(self):
        prefix = 'Reversi_PPO'
        suffix = str(datetime.now()).replace(" ", "_")

        model_name = f'{prefix}_{self.board_shape}by{self.board_shape}_{suffix}'

        return f'./models/{model_name}'
