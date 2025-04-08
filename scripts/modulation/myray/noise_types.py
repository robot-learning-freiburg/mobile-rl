import random
from typing import Union, Optional

import numpy as np
import torch
from ray.rllib.models import ActionDistribution
from ray.rllib.utils.exploration import GaussianNoise
from ray.rllib.utils.framework import TensorType
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise


class SimplifiedGausseNoise(GaussianNoise):
    def _get_torch_exploration_action(self,
                                      action_dist: ActionDistribution,
                                      explore: bool,
                                      timestep: Union[int, TensorType]):
        # Set last timestep or (if not given) increase by one.
        self.last_timestep = timestep if timestep is not None else self.last_timestep + 1

        # Apply exploration.
        if explore:
            # Random exploration phase.
            if self.last_timestep < self.random_timesteps:
                action, logp = self.random_exploration.get_torch_exploration_action(action_dist, explore=True)
            # Take a Gaussian sample with our stddev (mean=0.0) and scale it.
            else:
                action, logp = self._do_exploration(action_dist)

        # No exploration -> Return deterministic actions.
        else:
            action = action_dist.deterministic_sample()
            # Logp=always zero.
            logp = torch.zeros((action.size()[0],), dtype=torch.float32, device=self.device)

        return action, logp

    def _do_exploration(self, action_dist):
        det_actions = action_dist.deterministic_sample()
        scale = self.scale_schedule(self.last_timestep)
        gaussian_sample = scale * torch.normal(mean=torch.zeros(det_actions.size()), std=self.stddev).to(self.device)
        action = torch.min(torch.max(det_actions + gaussian_sample,
                                     torch.tensor(self.action_space.low, dtype=torch.float32, device=self.device)),
                           torch.tensor( self.action_space.high, dtype=torch.float32, device=self.device))
        logp = torch.zeros((action.size()[0], ), dtype=torch.float32, device=self.device)
        return action, logp


class AddedGaussianNoise(SimplifiedGausseNoise):
    """Add gaussian noise not to the determinist actions but rather to the sampled actions"""

    def _do_exploration(self, action_dist):
        sampled_actions = action_dist.sample()
        # TODO: this should maybe take into account the gaussian noise
        logp = action_dist.sampled_action_logp()
        if self.stddev > 0.0:
            scale = self.scale_schedule(self.last_timestep)
            gaussian_sample = scale * torch.normal(mean=torch.zeros(sampled_actions.size()), std=self.stddev).to(self.device)
            action = torch.min(torch.max(sampled_actions + gaussian_sample,
                                         torch.tensor(self.action_space.low, dtype=torch.float32, device=self.device)),
                               torch.tensor(self.action_space.high, dtype=torch.float32, device=self.device))
        else:
            action = sampled_actions
        return action, logp


class EpsilonGreedyNoise(SimplifiedGausseNoise):
    def _do_exploration(self, action_dist):
        # treat self.stddev as the epsilon in epsilon-greedy
        scale = self.scale_schedule(self.last_timestep)
        if random.random() < scale * self.stddev:
            action = torch.Tensor(self.action_space.sample()[np.newaxis]).to(self.device)
            logp = torch.zeros((action.size()[0],), dtype=torch.float32, device=self.device)
        else:
            action = action_dist.sample()
            logp = action_dist.sampled_action_logp()
        return action, logp


class AddedOUNoise(SimplifiedGausseNoise):
    def __init__(self,
                 action_space,
                 *,
                 framework: str,
                 model,
                 random_timesteps: int = 1000,
                 stddev: float = 0.1,
                 initial_scale: float = 1.0,
                 final_scale: float = 0.02,
                 scale_timesteps: int = 10000,
                 scale_schedule: Optional = None,
                 **kwargs):
        super(AddedOUNoise, self).__init__(action_space,
                                      framework=framework,
                                      model=model,
                                      random_timesteps=random_timesteps,
                                      stddev=stddev,
                                      initial_scale=initial_scale,
                                      final_scale=final_scale,
                                      scale_timesteps=scale_timesteps,
                                      scale_schedule=scale_schedule,
                                      **kwargs)
        self.ou_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(action_space.shape),
                                                     sigma=np.full(action_space.shape, stddev))

    def _do_exploration(self, action_dist):
        sampled_actions = action_dist.sample()
        # TODO: this should maybe take into account the noise
        logp = action_dist.sampled_action_logp()
        if self.stddev > 0.0:
            scale = self.scale_schedule(self.last_timestep)
            ou_sample = scale * torch.Tensor(self.ou_noise()).to(self.device)
            action = torch.min(torch.max(sampled_actions + ou_sample,
                                         torch.tensor(self.action_space.low, dtype=torch.float32, device=self.device)),
                               torch.tensor(self.action_space.high, dtype=torch.float32, device=self.device))
        else:
            action = sampled_actions
        return action, logp

    def on_episode_start(self,
                         policy: "Policy",
                         *,
                         environment = None,
                         episode: int = None,
                         tf_sess: Optional["tf.Session"] = None):
        """Handles necessary exploration logic at the beginning of an episode.

        Args:
            policy (Policy): The Policy object that holds this Exploration.
            environment (BaseEnv): The environment object we are acting in.
            episode (int): The number of the episode that is starting.
            tf_sess (Optional[tf.Session]): In case of tf, the session object.
        """
        self.ou_noise.reset()
