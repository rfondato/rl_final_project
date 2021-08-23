"""
    Este modulo contiene todo lo necesario para poder cargar los modelos PPO desde los zip files, si los mismos
    quieren ser utilizados en otro proyecto.

    Instrucciones:
    - Solo importar la clase RFondatoPlayer y utilizarla como jugador en la arena.
    - Ej: player = RFondatoPlayer(player=1, env=the_reversi_env, model_path="/models/rFondato.zip")
"""
import os
import uuid
from abc import ABC, abstractmethod
from typing import Tuple, Union

import gym
from boardgame2 import BoardGameEnv
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch as th
import numpy as np


class CustomBoardExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.Space):
        super().__init__(observation_space, int(get_flattened_obs_dim(observation_space) / 2))
        self.flatten = nn.Flatten()

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.flatten(observations[:, 0, :, :])


def sample_valid_actions(obs, distribution, deterministic=False, return_distribution=False):
    inf = 10 ** 8
    masks = (obs[:, 1, :, :].double() - 1) * inf
    masked_logits = distribution.logits + masks.reshape(distribution.logits.shape)
    if return_distribution:
        return th.distributions.Categorical(logits=masked_logits)
    if deterministic:
        return th.argmax(masked_logits, axis=1)
    return th.distributions.Categorical(logits=masked_logits).sample()


def sample_actions(observation, distribution, deterministic=False, return_distribution=False):
    # If there's a Mask channel, use it to sample valid actions
    if has_mask_channel(observation):
        return sample_valid_actions(observation, distribution.distribution, deterministic=deterministic,
                                    return_distribution=return_distribution)
    else:  # Otherwise just sample all actions according to distribution or return plain distribution
        return distribution.get_actions(deterministic=deterministic) if not return_distribution else distribution


def has_mask_channel(observation):
    return len(observation.shape) > 3 and observation.shape[1] > 1


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super(CustomActorCriticPolicy, self).__init__(
            *args,
            **kwargs
        )

    def _predict(self, observation, deterministic=False):
        """
        Get the action according to the policy for a given observation.
        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        latent_pi, _, latent_sde = self._get_latent(observation)
        distribution = self._get_action_dist_from_latent(latent_pi, latent_sde)

        return sample_actions(observation, distribution, deterministic)

    def forward(self, obs: th.Tensor, deterministic: bool = False):
        """
        Forward pass in all the networks (actor and critic)
        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        latent_pi, latent_vf, latent_sde = self._get_latent(obs)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi, latent_sde=latent_sde)
        distribution = sample_actions(obs, distribution, return_distribution=True)

        actions = distribution.sample()
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.
        :param obs:
        :param actions:
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        latent_pi, latent_vf, latent_sde = self._get_latent(obs)
        distribution = self._get_action_dist_from_latent(latent_pi, latent_sde)

        distribution = sample_actions(obs, distribution, return_distribution=True)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return values, log_prob, distribution.entropy()


class BasePlayer(ABC):
    def __init__(self,
                 player: int = 1,
                 env: BoardGameEnv = None,
                 flatten_action: bool = False,
                 name: str = None
                 ):
        self.id = uuid.uuid4()
        self.name = name if name is not None else self.__class__.__name__
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

    def __str__(self):
        return self.name

    def __eq__(self, other):
        return self.id == other.id


class RFondatoPlayer(BasePlayer):
    def __init__(self,
                 player: int = 1,
                 env: BoardGameEnv = None,
                 flatten_action: bool = False,
                 model_path: str = None,
                 deterministic: bool = True,
                 only_valid: bool = True,
                 device: str = 'auto'
                 ):

        if model_path is None:
            raise Exception("model_path cannot be None")

        super().__init__(player, env, flatten_action, os.path.splitext(os.path.basename(model_path))[0])

        self.model = PPO.load(model_path, device=device)
        self.model_path = model_path
        self.deterministic = deterministic
        self.only_valid = only_valid

    def predict(self, board: np.ndarray) -> Union[int, np.ndarray]:
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

    def __str__(self):
        return f"{self.__class__.__name__}({self.name})"
