from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

from custom_features_extractor import CustomBoardExtractor
from custom_policies import CustomActorCriticPolicy
from multi_env import make_reversi_vec_env, SelfPlayEnv
from players import RandomPlayer


class CustomReversiModel:

    def __init__(self, board_shape=8, n_envs=8, local_player=RandomPlayer, n_steps=2048, gamma=0.99, ent_coef=0.0, gae_lambda=0.95,
                 n_epochs=10):
        self._create_train_env(board_shape, local_player, n_envs)
        self._create_eval_env(board_shape)
        self._create_callbacks(board_shape, ent_coef, gae_lambda, gamma, n_envs, n_epochs)
        self._create_model(n_steps, ent_coef, gae_lambda, gamma, n_epochs)

    def learn(self, total_timesteps=int(10e5)):
        self.model.learn(total_timesteps=total_timesteps, callback=self.callbacks)

    def _create_model(self, n_steps, ent_coef, gae_lambda, gamma, n_epochs):
        self.model = PPO(
            CustomActorCriticPolicy,
            self.env,
            verbose=1,
            tensorboard_log='tensorboard_log',
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            n_epochs=n_epochs,
            n_steps=n_steps,
            policy_kwargs={'features_extractor_class': CustomBoardExtractor}
        )

    def _create_callbacks(self, board_shape, ent_coef, gae_lambda, gamma, n_envs, n_epochs):
        prefix = 'Reversi_PPO'
        suffix = 'masked_actions'
        model_name = f'{prefix}_{board_shape}by{board_shape}_{gamma}_{gae_lambda}_{ent_coef}_{n_epochs}_{n_envs}_{suffix}'
        best_model_save_path = f'./models/{model_name}'
        self.callbacks = [EvalCallback(
            eval_env=self.eval_env,
            eval_freq=1_000,
            n_eval_episodes=500,
            deterministic=True,
            verbose=1,
            best_model_save_path=best_model_save_path,
        )]

    def _create_eval_env(self, board_shape):
        self.eval_env = make_reversi_vec_env(
            SelfPlayEnv, n_envs=1, vec_env_cls=SubprocVecEnv,
            env_kwargs={
                'board_shape': board_shape,
                'LocalPlayer': RandomPlayer,
                'mask_channel': True
            }
        )

    def _create_train_env(self, board_shape, local_player, n_envs):
        self.env = make_reversi_vec_env(
            SelfPlayEnv, n_envs=n_envs, vec_env_cls=SubprocVecEnv,
            env_kwargs={
                'board_shape': board_shape,
                'LocalPlayer': local_player,
                'mask_channel': True
            }
        )
