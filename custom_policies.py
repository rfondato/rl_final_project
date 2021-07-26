import torch as th
from stable_baselines3.common.policies import ActorCriticPolicy
from typing import Tuple


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
            self,
            *args,  # Todos los argumentos posicionales de ActorCriticPolicy
            actions_mask_func=None,  # El nuevo argumento
            **kwargs  # Todos los argumentos opcionales de ActorCriticPolicy
    ):
        super(CustomActorCriticPolicy, self).__init__(
            *args,
            **kwargs
        )
        if actions_mask_func:
            self.get_actions_mask = actions_mask_func

    def sample_masked_actions(self, obs, distribution, deterministic=False, return_distribution=False):
        # Dada las obs y distribuciones luego de evaluar la red neuronal, samplear solo las acciones válidas
        # Las obs se usan para que con self.get_actions_mask se obtengan las acciones válidas
        # las distribuciones son el resultado de evaluar la red neuronal y van a dar acciones no validas
        # Generar una nueva distribución (del lado de los logits preferentemente) donde las acciones no válidas
        # tengan probabildad nula de ser muestreadas
        # Luego se modifican abajo los métodos
        # _predict, forward y evaluate_actions
        # Si tiene el flag de return_distribution en true devuelve la distribución nueva
        # Caso contrario devuelve las acciones
        # Para tener en cuenta, obs tiene dimensión [batch_size, channels, H, W]
        # Recomendamos poner un print(obs.shape)
        # y correr:
        # obs = env.reset()
        # actions, _ = model.predict(obs)
        # Para sacarse las dudas

        masks = 1 - self.get_actions_mask(obs)
        masks = -masks * (10 ** 8)  # Numero negativo grande en los moviemientos invalidos

        if th.is_tensor(distribution.logits):
            masks = th.from_numpy(masks).to(self.device)

        masked_logits = distribution.logits + masks.reshape(distribution.logits.shape[0], distribution.logits.shape[1])
        if return_distribution:
            return th.distributions.Categorical(logits=masked_logits)
        if deterministic:
            return th.argmax(masked_logits, axis=1)
        return th.distributions.Categorical(logits=masked_logits).sample()

    def _predict(self, observation, deterministic=False):
        """
        Get the action according to the policy for a given observation.
        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        latent_pi, _, latent_sde = self._get_latent(observation)
        distribution = self._get_action_dist_from_latent(latent_pi, latent_sde)

        if self.get_actions_mask:
            actions = self.sample_masked_actions(observation, distribution.distribution, deterministic=deterministic)
        else:
            actions = distribution.get_actions(deterministic=deterministic)

        return actions

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

        if self.get_actions_mask:
            actions = self.sample_masked_actions(obs, distribution.distribution, deterministic=deterministic)
        else:
            actions = distribution.get_actions(deterministic=deterministic)

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
        distrib = self.sample_masked_actions(obs, distribution.distribution, return_distribution=True)

        log_prob = distrib.log_prob(actions)
        values = self.value_net(latent_vf)
        return values, log_prob, distrib.entropy()
