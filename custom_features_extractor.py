import gym
import torch as th
import torch.nn as nn
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


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
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor([[observation_space.sample()[0]]]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        board = observations[:, 0, :, :] \
            .reshape((observations.shape[0], 1, observations.shape[2], observations.shape[3]))
        return self.linear(self.cnn(board))


if __name__ == "__main__":
    board_shape = 8
    obs_space = gym.spaces.Box(-1, 1, (2, board_shape, board_shape))
    extractor = CNNFeaturesExtractor(observation_space=obs_space)
    tensor = extractor.forward(th.from_numpy(obs_space.sample().reshape(1, 2, board_shape, board_shape)))
    print(tensor)
