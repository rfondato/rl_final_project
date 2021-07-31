import torch as th
from stable_baselines3.common.policies import ActorCriticPolicy
from typing import Tuple


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

        actions = sample_actions(obs, distribution, deterministic)
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
