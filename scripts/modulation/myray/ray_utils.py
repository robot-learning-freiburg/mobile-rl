import os
import random
from typing import Union, List, Optional

import torch
import numpy as np
from ray import tune
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.torch.torch_action_dist import TorchDiagGaussian, TorchDistributionWrapper
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils import override
from ray.rllib.utils.exploration.gaussian_noise import GaussianNoise
from ray.rllib.utils.framework import TensorType
from ray.rllib.utils.numpy import MIN_LOG_NN_OUTPUT, MAX_LOG_NN_OUTPUT

from modulation.myray.ray_wandb import WandbSACTrainer
from modulation.myray.custom_sac import SACTrainerCustom


def get_local_ray_dir(wandb_config):
    return f"{wandb_config['logpath']}/ray/"


def get_local_wandb_dir(wandb_config):
    p = f"{wandb_config['logpath']}/wandb/"
    os.makedirs(p, exist_ok=True)
    return p


def get_sac_conf(wandb_config, model_config: dict) -> dict:
    if wandb_config.ent_coef == 'auto':
        initial_alpha = wandb_config.initial_alpha
        ent_lr = wandb_config.ent_lr or wandb_config.lr_start
    else:
        initial_alpha = float(wandb_config.ent_coef)
        ent_lr = 0.0
        assert wandb_config.ent_lr is None, wandb_config.ent_lr

    conf = {
        # === Model ===
        # Use two Q-networks (instead of one) for action-value estimation.
        # Note: Each Q-network will have its own target network.
        # "twin_q": True,
        # Model options for the Q network(s). These will override MODEL_DEFAULTS.
        # The `Q_model` dict is treated just as the top-level `model` dict in
        # setting up the Q-network(s) (2 if twin_q=True).
        # That means, you can do for different observation spaces:
        # obs=Box(1D) -> Tuple(Box(1D) + Action) -> concat -> post_fcnet
        # obs=Box(3D) -> Tuple(Box(3D) + Action) -> vision-net -> concat w/ action
        #   -> post_fcnet
        # obs=Tuple(Box(1D), Box(3D)) -> Tuple(Box(1D), Box(3D), Action)
        #   -> vision-net -> concat w/ Box(1D) and action -> post_fcnet
        # You can also have SAC use your custom_model as Q-model(s), by simply
        # specifying the `custom_model` sub-key in below dict (just like you would
        # do in the top-level `model` dict.
        # PART OF THE WORKAROUND FOR SAC
        'model': {**model_config},
        # "Q_model": {
        #     "fcnet_hiddens": [256, 256],
        #     "fcnet_activation": "relu",
        #     "post_fcnet_hiddens": [],
        #     "post_fcnet_activation": None,
        #     "custom_model": None,  # Use this to define custom Q-model(s).
        #     "custom_model_config": {},
        # },
        # Model options for the policy function (see `Q_model` above for details).
        # The difference to `Q_model` above is that no action concat'ing is
        # performed before the post_fcnet stack.
        # "policy_model": {
        #     "fcnet_hiddens": [256, 256],
        #     "fcnet_activation": "relu",
        #     "post_fcnet_hiddens": [],
        #     "post_fcnet_activation": None,
        #     "custom_model": None,  # Use this to define a custom policy model.
        #     "custom_model_config": {},
        # },
        # Unsquash actions to the upper and lower bounds of env's action space.
        # Ignored for discrete action spaces.
        # NOTE: ALREADY SET IN THE MAIN CONFIG
        # "normalize_actions": True,

        # === Learning ===
        # Update the target by \tau * policy + (1-\tau) * target_policy.
        # 0.001 <= x <= 0.005
        "tau": wandb_config.tau,  # tune.choice([0.001, 0.002]) if wandb_config.param_search_samples else wandb_config.tau,
        # Initial value to use for the entropy weight alpha.
        "initial_alpha": initial_alpha,  # tune.choice([0.1, 0.3, 0.5, 1.0]),
        # Target entropy lower bound. If "auto", will be set to -|A| (e.g. -2.0 for
        # Discrete(2), -3.0 for Box(shape=(3,))).
        # This is the inverse of reward scale, and will be optimized automatically.
        # NOTE: wandb_config.ent_coef as in stableBL is prob. the same as "initial_alpha" = wandb_config.ent_coef & entropy_learning_rate = 0
        # >= -2
        "target_entropy": wandb_config.target_ent if wandb_config.target_ent == 'auto' else float(wandb_config.target_ent),  # tune.uniform(-4.0, -2.0) if wandb_config.param_search_samples else wandb_config.ent_coef,
        # N-step target updates. If >1, sars' tuples in trajectories will be
        # postprocessed to become sa[discounted sum of R][s t+n] tuples.
        # < 10
        # NOTE: ik_fail_thresh > 1 MIGHT LEAD TO BAD VALUE ESTIMATES IF COMBINED WITH n_step > 1
        "n_step": wandb_config.nstep,  # tune.choice([1, 5, 10]) if wandb_config.param_search_samples else wandb_config.nstep,

        # === Replay buffer ===
        # NOTE: this won't get picked up in multiagent settings, move to main ray_config for that!
        # Size of the replay buffer (in time steps).
        "buffer_size": wandb_config.buffer_size,
        # If True prioritized replay buffer will be used.
        "prioritized_replay": wandb_config.prioritizedrp,  # tune.choice([False, True]) if wandb_config.param_search_samples else wandb_config.prioritizedrp,
        "prioritized_replay_alpha": 0.6,
        "prioritized_replay_beta": 0.4,
        "prioritized_replay_eps": 1e-6,
        "prioritized_replay_beta_annealing_timesteps": 200000,
        "final_prioritized_replay_beta": 0.4,
        # Whether to LZ4 compress observations
        # "compress_observations": False,
        # === Optimization ===
        # range 1e-3 to 1e-4
        "optimization": {
            "actor_learning_rate": wandb_config.lr_start,
            "critic_learning_rate": wandb_config.lr_start,  # tune.sample_from(lambda spec: spec.config.optimization.actor_learning_rate)
            "entropy_learning_rate": ent_lr,  # tune.sample_from(lambda spec: spec.config.optimization.actor_learning_rate)
        },
        # requires to have the same learning rate for all optimizers above
        "lr_schedule": ([[0, wandb_config.lr_start], [wandb_config.total_steps, wandb_config.lr_end]],
                        # there are 2 critic optimisers for twin_q...
                        [[0, wandb_config.lr_start], [wandb_config.total_steps, wandb_config.lr_end]],
                        [[0, wandb_config.lr_start], [wandb_config.total_steps, wandb_config.lr_end]],
                        # entropy optimizer
                        [[0, ent_lr], [wandb_config.total_steps, wandb_config.lr_end]]),
        # If not None, clip gradients during optimization at this value.
        "grad_clip": 10,
        # How many steps of the model to sample before learning starts.
        "learning_starts": wandb_config.rnd_steps,
        # Update the replay buffer with this many samples at once. Note that this
        # setting applies per-worker if num_workers > 1.
        # "rollout_fragment_length": 1,
        # Size of a batched sampled from replay buffer for training.
        "train_batch_size": wandb_config.batch_size,
        # Update the target network every `target_network_update_freq` steps.
        # "target_network_update_freq": 0,
        # Use a Beta-distribution instead of a SquashedGaussian for bounded,
        # continuous action spaces (not recommended, for debugging only).
        "_use_beta_distribution": False,
    }
    if not wandb_config.use_map_obs:
        conf["Q_model"] ={"fcnet_hiddens": wandb_config.fcnet_hiddens,
                          "fcnet_activation": "relu",}
        conf["policy_model"] ={"fcnet_hiddens": wandb_config.fcnet_hiddens,
                          "fcnet_activation": "relu",}
    return conf


def get_explore_config(wandb_config) -> dict:
    if wandb_config.explore_noise_type == 'normal':
        noise_type = "modulation.myray.noise_types.AddedGaussianNoise"
    elif wandb_config.explore_noise_type == 'ou':
        noise_type = "modulation.myray.noise_types.AddedOUNoise"
        assert wandb_config.explore_noise, wandb_config.explore_noise
    elif wandb_config.explore_noise_type == 'egreedy':
        noise_type = "modulation.myray.noise_types.EpsilonGreedyNoise"
    else:
        raise NotImplementedError()
    return {
        # The Exploration class to use. In the simplest case, this is the name
        # (str) of any class present in the `rllib.utils.exploration` package.
        # You can also provide the python class directly or the full location
        # of your class (e.g. "ray.rllib.utils.exploration.epsilon_greedy.
        # EpsilonGreedy").
        # NOTE: this will then replace sampling from the SAC distribution. If I want to have both (as in stableBL) I need to write my own class (just small changes to GaussianNoise)
        # "type": "GaussianNoise",
        "type": noise_type,
        # Add constructor kwargs here (if any).
        "random_timesteps": 0,
        # SETTING TO 0 RESULTS IN 'stochastic_sampling' strategy
        # 0.75 definitely bad. Not sure about low values
        "stddev": wandb_config.explore_noise,  # tune.choice([0., 0.1, 0.25, 0.5])
        "initial_scale": 1.0,
        "final_scale": wandb_config.explore_noise_final_scale,  # tune.choice([0., 0.5, 1.0])
        "scale_timesteps": wandb_config.total_steps
    }


def get_policy_config(wandb_config, model_config, algo: str) -> dict:
    common_policy_config = {
        # --------------------------------
        # ALGORITHM
        "gamma": wandb_config.gamma,  # tune.choice([0.98, 0.99, 0.999]) if wandb_config.param_search_samples else wandb_config.gamma,
        # --------------------------------
        # EXPLORATION
        "explore": True,
        # Provide a dict specifying the Exploration object's config.
        "exploration_config": get_explore_config(wandb_config),
    }

    if algo == 'sac':
        policy_config = get_sac_conf(wandb_config, model_config)
    else:
        raise NotImplementedError(algo)
    policy_config.update(common_policy_config)
    return policy_config


def get_normal_agent_model(wandb_config) -> dict:
    """Normal model to process local map & robotstate"""
    conf = {}

    if wandb_config.use_map_obs:
        conf.update({"custom_model": "ray_map_encoder",
                     "post_fcnet_hiddens": wandb_config.fcnet_hiddens,
                     "post_fcnet_activation": "relu",
                     "custom_model_config": {"map_encoder": wandb_config.mapencoder}})
        if wandb_config.algo == 'sac':
            # WORKAROUND FOR SAC: cannot pass a custom_model_config with custom keys into policy_model and Q_model,
            # so instead pass the whole thing as custom_model_config here.
            # NOTE: SBL shares the encoder between actor and critic by default, this won't
            return {"custom_model": "map_encoder_sac_torch_network",
                    "custom_model_config": conf}
        else:
            return conf
    else:
        assert not wandb_config.use_gmm
        return {}


def get_normal_task_config(wandb_config):
    model = get_normal_agent_model(wandb_config)
    config_updates = get_policy_config(wandb_config=wandb_config, model_config=model, algo=wandb_config.algo)
    return config_updates


def get_trainer_fn(algo, use_wandb_mixin: bool = True):
    if algo == 'sac':
        trainer_fn = WandbSACTrainer if use_wandb_mixin else SACTrainerCustom
    else:
        raise NotImplementedError()
    return trainer_fn
