from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
import gym
import torch.nn as nn
import torch as th


class CustomBoardExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.Space):
        super().__init__(observation_space, int(get_flattened_obs_dim(observation_space) / 2))
        self.flatten = nn.Flatten()

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.flatten(observations[:, 0, :, :])


class CNNFeaturesExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.spaces.Box):
        features_dim = int(get_flattened_obs_dim(observation_space) / 2)
        super().__init__(observation_space, features_dim)
        n_input_channels = 1

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=4, stride=2, padding='valid'),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding='valid'),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=4, stride=1, padding='valid'),
            nn.ReLU(),
            nn.Flatten()
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations[:, 0, :, :]))
