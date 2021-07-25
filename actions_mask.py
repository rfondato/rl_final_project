import numpy as np


class ActionsMask:

    def __init__(self, env):
        self.env = env

    def get_actions_mask(self, state):
        player = 1
        valid_actions = self.env.get_valid((state, player))
        return valid_actions.reshape(-1)

    def get_batch_actions_mask(self, state):
        batch_size = state.shape[0]
        masks = np.array([self.get_actions_mask(s[0]) for s in state])
        return masks.reshape(batch_size, 1, -1)
