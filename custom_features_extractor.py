from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
import gym
import torch.nn
import torch as th


class CustomBoardExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.Space):
        super().__init__(observation_space, int(get_flattened_obs_dim(observation_space) / 2))
        self.flatten = torch.nn.Flatten()

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.flatten(observations[:, 0, :, :])
