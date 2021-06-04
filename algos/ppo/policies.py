
from algos.pop_art import PopArt

from functools import partial

import numpy as np
from torch import nn

from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
)
from stable_baselines3.common.type_aliases import Schedule


from stable_baselines3.common.policies import (
    BasePolicy,
    ActorCriticPolicy
)

class PopArtActorCriticPolicy(ActorCriticPolicy):
  """
  Policy class for actor-critic algorithms (has both policy and value prediction).
  Used by A2C, PPO and the likes.

  :param observation_space: Observation space
  :param action_space: Action space
  :param lr_schedule: Learning rate schedule (could be constant)
  :param net_arch: The specification of the policy and value networks.
  :param activation_fn: Activation function
  :param ortho_init: Whether to use or not orthogonal initialization
  :param use_sde: Whether to use State Dependent Exploration or not
  :param log_std_init: Initial value for the log standard deviation
  :param full_std: Whether to use (n_features x n_actions) parameters
      for the std instead of only (n_features,) when using gSDE
  :param sde_net_arch: Network architecture for extracting features
      when using gSDE. If None, the latent features from the policy will be used.
      Pass an empty list to use the states as features.
  :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
      a positive standard deviation (cf paper). It allows to keep variance
      above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
  :param squash_output: Whether to squash the output using a tanh function,
      this allows to ensure boundaries when using gSDE.
  :param features_extractor_class: Features extractor to use.
  :param features_extractor_kwargs: Keyword arguments
      to pass to the features extractor.
  :param normalize_images: Whether to normalize images or not,
       dividing by 255.0 (True by default)
  :param optimizer_class: The optimizer to use,
      ``th.optim.Adam`` by default
  :param optimizer_kwargs: Additional keyword arguments,
      excluding the learning rate, to pass to the optimizer
  """

  def _build(self, lr_schedule: Schedule) -> None:
    """
    Create the networks and the optimizer.

    :param lr_schedule: Learning rate schedule
        lr_schedule(1) is the initial learning rate
    """
    self._build_mlp_extractor()

    latent_dim_pi = self.mlp_extractor.latent_dim_pi

    # Separate features extractor for gSDE
    if self.sde_net_arch is not None:
      self.sde_features_extractor, latent_sde_dim = create_sde_features_extractor(
        self.features_dim, self.sde_net_arch, self.activation_fn
      )

    if isinstance(self.action_dist, DiagGaussianDistribution):
      self.action_net, self.log_std = self.action_dist.proba_distribution_net(
        latent_dim=latent_dim_pi, log_std_init=self.log_std_init
      )
    elif isinstance(self.action_dist, StateDependentNoiseDistribution):
      latent_sde_dim = latent_dim_pi if self.sde_net_arch is None else latent_sde_dim
      self.action_net, self.log_std = self.action_dist.proba_distribution_net(
        latent_dim=latent_dim_pi, latent_sde_dim=latent_sde_dim, log_std_init=self.log_std_init
      )
    elif isinstance(self.action_dist, CategoricalDistribution):
      self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
    elif isinstance(self.action_dist, MultiCategoricalDistribution):
      self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
    elif isinstance(self.action_dist, BernoulliDistribution):
      self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
    else:
      raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")

    self.value_net = PopArt(self.mlp_extractor.latent_dim_vf, 1)
    # Init weights: use orthogonal initialization
    # with small initial weight for the output
    if self.ortho_init:
      # TODO: check for features_extractor
      # Values from stable-baselines.
      # features_extractor/mlp values are
      # originally from openai/baselines (default gains/init_scales).
      module_gains = {
        self.features_extractor: np.sqrt(2),
        self.mlp_extractor: np.sqrt(2),
        self.action_net: 0.01,
        self.value_net: 1,
      }
      for module, gain in module_gains.items():
        module.apply(partial(self.init_weights, gain=gain))

    # Setup optimizer with initial learning rate
    self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)