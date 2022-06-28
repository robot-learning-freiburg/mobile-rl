from typing import Dict, List, Type, Union, Optional, Tuple, Iterable
import gym
import numpy as np
import torch
from gym.spaces import Discrete, MultiDiscrete, Box
from ray.rllib.agents.dqn.dqn_tf_policy import PRIO_WEIGHTS
from ray.rllib.agents.sac import DEFAULT_CONFIG
from ray.rllib.agents.sac.sac import SACTrainer
from ray.rllib.agents.sac.sac_torch_policy import postprocess_trajectory, setup_late_mixins, \
    build_sac_model_and_action_dist, TargetNetworkMixin, \
    ComputeTDErrorMixin, _get_dist_class, action_distribution_fn, optimizer_fn, validate_spaces, apply_grad_clipping, build_sac_model
from ray.rllib.models import ActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_action_dist import TorchCategorical, TorchSquashedGaussian, TorchDiagGaussian, \
    TorchBeta, TorchDirichlet
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.policy_template import build_policy_class
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils import override, SMALL_NUMBER
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.schedules import ConstantSchedule, PiecewiseSchedule
from ray.rllib.utils.spaces.simplex import Simplex
from ray.rllib.utils.torch_ops import concat_multi_gpu_td_errors, huber_loss
from ray.rllib.utils.typing import ModelInputDict, TensorType, TrainerConfigDict, LocalOptimizer
from ray.rllib.policy.torch_policy import LearningRateSchedule
torch, nn = try_import_torch()
F = nn.functional


def actor_critic_loss_custom(
        policy: Policy, model: ModelV2,
        dist_class: Type[TorchDistributionWrapper],
        train_batch: SampleBatch) -> Union[TensorType, List[TensorType]]:
    """Constructs the loss for the Soft Actor Critic.

    Args:
        policy (Policy): The Policy to calculate the loss for.
        model (ModelV2): The Model to calculate the loss for.
        dist_class (Type[TorchDistributionWrapper]: The action distr. class.
        train_batch (SampleBatch): The training data.

    Returns:
        Union[TensorType, List[TensorType]]: A single loss tensor or a list
            of loss tensors.
    """
    # Look up the target model (tower) using the model tower.
    target_model = policy.target_models[model]

    # Should be True only for debugging purposes (e.g. test cases)!
    deterministic = policy.config["_deterministic_loss"]

    model_out_t, _ = model({"obs": train_batch[SampleBatch.CUR_OBS], "is_training": True,}, [], None)
    model_out_tp1, _ = model({"obs": train_batch[SampleBatch.NEXT_OBS], "is_training": True,}, [], None)
    target_model_out_tp1, _ = target_model({"obs": train_batch[SampleBatch.NEXT_OBS], "is_training": True,}, [], None)

    alpha = torch.exp(model.log_alpha)

    # Discrete case.
    if model.discrete:
        # Get all action probs directly from pi and form their logp.
        log_pis_t = F.log_softmax(model.get_policy_output(model_out_t), dim=-1)
        log_pis_tp1 = F.log_softmax(model.get_policy_output(model_out_tp1), -1)

        q_t = model.get_q_values(model_out_t)
        q_tp1 = target_model.get_q_values(target_model_out_tp1)
        if policy.config["twin_q"]:
            twin_q_t = model.get_twin_q_values(model_out_t)
            twin_q_tp1 = target_model.get_twin_q_values(target_model_out_tp1)

        policy_t = torch.exp(log_pis_t)
        policy_tp1 = torch.exp(log_pis_tp1)
        # Q-values.
        # Target Q-values.
        # q_tp1 = target_model.get_q_values(target_model_out_tp1)
        if policy.config["twin_q"]:
        #     twin_q_t = model.get_twin_q_values(model_out_t)
        #     twin_q_tp1 = target_model.get_twin_q_values(target_model_out_tp1)
            q_tp1 = torch.min(q_tp1, twin_q_tp1)
        q_tp1 -= alpha * log_pis_tp1

        # Actually selected Q-values (from the actions batch).
        one_hot = F.one_hot(train_batch[SampleBatch.ACTIONS].long(), num_classes=q_t.size()[-1])
        q_t = torch.sum(q_t * one_hot, dim=-1)
        if policy.config["twin_q"]:
            twin_q_t = torch.sum(twin_q_t * one_hot, dim=-1)
        # Discrete case: "Best" means weighted by the policy (prob) outputs.
        q_tp1_best = torch.sum(torch.mul(policy_tp1, q_tp1), dim=-1)
        q_tp1_masked = torch.unsqueeze(1.0 - train_batch[SampleBatch.DONES].float(), 1) * q_tp1_best
    # Continuous actions case.
    else:
        # Sample single actions from distribution.
        action_dist_class = _get_dist_class_fixed(policy.config, policy.action_space)
        action_dist_t = action_dist_class(model.get_policy_output(model_out_t), policy.model)
        policy_t = action_dist_t.sample() if not deterministic else action_dist_t.deterministic_sample()
        log_pis_t = torch.unsqueeze(action_dist_t.logp(policy_t if isinstance(action_dist_t, TorchBeta) else None), -1)
        action_dist_tp1 = action_dist_class(model.get_policy_output(model_out_tp1), policy.model)
        policy_tp1 = action_dist_tp1.sample() if not deterministic else action_dist_tp1.deterministic_sample()
        log_pis_tp1 = torch.unsqueeze(action_dist_tp1.logp(policy_t if isinstance(action_dist_tp1, TorchBeta) else None), -1)

        # Q-values for the actually selected actions.
        q_t = model.get_q_values(model_out_t, train_batch[SampleBatch.ACTIONS])
        q_t = torch.squeeze(q_t, dim=-1)
        if policy.config["twin_q"]:
            twin_q_t = model.get_twin_q_values(model_out_t, train_batch[SampleBatch.ACTIONS])
            twin_q_t = torch.squeeze(twin_q_t, dim=-1)

        # Q-values for current policy in given current state.
        q_t_det_policy = model.get_q_values(model_out_t, policy_t)
        if policy.config["twin_q"]:
            twin_q_t_det_policy = model.get_twin_q_values(model_out_t, policy_t)
            q_t_det_policy = torch.min(q_t_det_policy, twin_q_t_det_policy)

        # Target q network evaluation.
        with torch.no_grad():
            q_tp1 = target_model.get_q_values(target_model_out_tp1, policy_tp1)
            if policy.config["twin_q"]:
                twin_q_tp1 = target_model.get_twin_q_values(target_model_out_tp1, policy_tp1)
                # Take min over both twin-NNs.
                q_tp1 = torch.min(q_tp1, twin_q_tp1)
            q_tp1 -= alpha * log_pis_tp1
            q_tp1 = torch.squeeze(q_tp1, dim=-1)
            q_tp1_masked = (1.0 - train_batch[SampleBatch.DONES].float()) * q_tp1

    # compute RHS of bellman equation
    gamma = policy.config["gamma"]
    fs = policy.config['env_config']['frame_skip'][-1] if isinstance(policy.config['env_config']['frame_skip'], Iterable) else policy.config['env_config']['frame_skip']
    if fs > 1:
        # rllib's DummyBatch for testing is a vector of 0s
        if isinstance(train_batch[SampleBatch.INFOS][0], dict):
            frame_skips = torch.Tensor([i['frame_skip'] for i in train_batch[SampleBatch.INFOS]]).to(q_tp1_masked.device)
            gamma = (gamma ** frame_skips).to(q_tp1_masked.device)

    q_t_target = (train_batch[SampleBatch.REWARDS] + (gamma ** policy.config["n_step"]) * q_tp1_masked).detach()

    # Compute the TD-error (potentially clipped).
    base_td_error = torch.abs(q_t - q_t_target)
    if policy.config["twin_q"]:
        twin_td_error = torch.abs(twin_q_t - q_t_target)
        td_error = 0.5 * (base_td_error + twin_td_error)
    else:
        td_error = base_td_error

    critic_loss = [torch.mean(train_batch[PRIO_WEIGHTS] * huber_loss(base_td_error))]
    if policy.config["twin_q"]:
        critic_loss.append(torch.mean(train_batch[PRIO_WEIGHTS] * huber_loss(twin_td_error)))

    # Alpha- and actor losses.
    # Note: In the papers, alpha is used directly, here we take the log.
    # Discrete case: Multiply the action probs as weights with the original
    # loss terms (no expectations needed).
    if model.discrete:
        weighted_log_alpha_loss = policy_t.detach() * (-model.log_alpha * (log_pis_t + model.target_entropy).detach())
        # Sum up weighted terms and mean over all batch items.
        alpha_loss = torch.mean(torch.sum(weighted_log_alpha_loss, dim=-1))
        # Actor loss.
        # NOTE: No stop_grad around policy output here
        # (compare with q_t_det_policy for continuous case).
        actor_loss = torch.mean(torch.sum(torch.mul(policy_t, alpha.detach() * log_pis_t - q_t.detach()), dim=-1))
    else:
        alpha_loss = -torch.mean(model.log_alpha * (log_pis_t + model.target_entropy).detach())
        # Note: Do not detach q_t_det_policy here b/c is depends partly
        # on the policy vars (policy sample pushed through Q-net).
        # However, we must make sure `actor_loss` is not used to update
        # the Q-net(s)' variables.
        actor_loss = torch.mean(alpha.detach() * log_pis_t - q_t_det_policy)

    # Save for stats function.
    policy.q_t = q_t
    policy.policy_t = policy_t
    policy.log_pis_t = log_pis_t

    # Store td-error in model, such that for multi-GPU, we do not override
    # them during the parallel loss phase. TD-error tensor in final stats
    # can then be concatenated and retrieved for each individual batch item.
    model.td_error = td_error
    policy.td_error = td_error
    policy.actor_loss = actor_loss
    policy.critic_loss = critic_loss
    policy.alpha_loss = alpha_loss
    policy.log_alpha_value = model.log_alpha
    policy.alpha_value = alpha
    policy.target_entropy = model.target_entropy

    # MY STATS
    with torch.no_grad():
        policy.q_t_selected_target = q_t_target
        policy.alpha_q_component = -alpha * log_pis_tp1
        policy.ent = -(policy_tp1 * log_pis_tp1).sum(dim=1)
        policy.dist_scale = action_dist_t.dist.scale if (hasattr(action_dist_t, "dist") and hasattr(action_dist_t.dist, "scale")) else torch.zeros(1)
        policy.dist_loc = action_dist_t.dist.loc if (hasattr(action_dist_t, "dist") and hasattr(action_dist_t.dist, "loc")) else torch.zeros(1)

        # q_tp1_no_alpha = q_tp1 + alpha * log_pis_tp1
        # q_tp1_best_no_alpha = torch.squeeze(input=q_tp1_no_alpha, dim=-1)
        # q_tp1_best_masked_no_alpha = (1.0 - train_batch[SampleBatch.DONES].float()) * \
        #                     q_tp1_best_no_alpha
        # # compute RHS of bellman equation
        # q_t_selected_target_no_alpha = (train_batch[SampleBatch.REWARDS] +
        #                                 (policy.config["gamma"]**policy.config["n_step"]) * q_tp1_best_masked_no_alpha).detach()
        # policy.q_t_selected_target_no_alpha = q_t_selected_target_no_alpha

    # Return all loss terms corresponding to our optimizers.
    return tuple([policy.actor_loss] + policy.critic_loss +
                 [policy.alpha_loss])


def get_custom_sac_policy_class(config):
    """Policy class picker function. Class is chosen based on DL-framework.

    Args:
        config (TrainerConfigDict): The trainer's configuration dict.

    Returns:
        Optional[Type[Policy]]: The Policy class to use with PPOTrainer.
            If None, use `default_policy` provided in build_trainer().
    """
    return SACTorchPolicyCustom


def stats(policy: Policy, train_batch: SampleBatch) -> Dict[str, TensorType]:
    """Stats function for SAC. Returns a dict with important loss stats.

    Args:
        policy (Policy): The Policy to generate stats for.
        train_batch (SampleBatch): The SampleBatch (already) used for training.

    Returns:
        Dict[str, TensorType]: The stats dict.
    """
    return {
        "td_error": policy.td_error,
        "mean_td_error": torch.mean(policy.td_error),
        "actor_loss": torch.mean(policy.actor_loss),
        "critic_loss": torch.mean(torch.stack(policy.critic_loss)),
        "alpha_loss": torch.mean(policy.alpha_loss),
        "alpha_value": torch.mean(policy.alpha_value),
        "log_alpha_value": torch.mean(policy.log_alpha_value),
        "target_entropy": policy.target_entropy,
        "policy_t": torch.mean(policy.policy_t),
        "mean_q": torch.mean(policy.q_t),
        "max_q": torch.max(policy.q_t),
        "min_q": torch.min(policy.q_t),
        # mine
        "log_pis_t_mean": torch.mean(policy.log_pis_t),
        "log_pis_t_max": torch.max(policy.log_pis_t),
        "log_pis_t_min": torch.min(policy.log_pis_t),
        "q_t_selected_target": torch.mean(policy.q_t_selected_target),
        "alpha_q_component": torch.mean(policy.alpha_q_component),
        "ent": torch.mean(policy.ent),
        # "q_t_selected_target_no_alpha": torch.mean(policy.q_t_selected_target_no_alpha),
        "dist_scale_mean": torch.mean(policy.dist_scale),
        "dist_scale_max": torch.max(policy.dist_scale),
        "dist_scale_min": torch.min(policy.dist_scale),
        "dist_loc_mean": torch.mean(policy.dist_loc),
        "dist_loc_max": torch.max(policy.dist_loc),
        "dist_loc_min": torch.min(policy.dist_loc),
        # "policy0_t": policy.policy_t[:, 0].detach().cpu().numpy().tolist(),
        # "policy1_t": policy.policy_t[:, 1].detach().cpu().numpy().tolist(),
        # "policy2_t": policy.policy_t[:, 2].detach().cpu().numpy().tolist(),
    }


def apply_grad_clipping_custom(policy, optimizer, loss):
    """Applies gradient clipping to already computed grads inside `optimizer`.

    Args:
        policy (TorchPolicy): The TorchPolicy, which calculated `loss`.
        optimizer (torch.optim.Optimizer): A local torch optimizer object.
        loss (torch.Tensor): The torch loss tensor.
    """
    info = {}
    if policy.config["grad_clip"]:
        for param_group in optimizer.param_groups:
            # Make sure we only pass params with grad != None into torch
            # clip_grad_norm_. Would fail otherwise.
            params = list(
                filter(lambda p: p.grad is not None, param_group["params"]))
            if params:
                grad_gnorm = nn.utils.clip_grad_norm_(
                    params, policy.config["grad_clip"])
                if isinstance(grad_gnorm, torch.Tensor):
                    grad_gnorm = grad_gnorm.cpu().numpy()
                info["grad_gnorm"] = grad_gnorm
    return info


def get_default_config():
    return DEFAULT_CONFIG


class FixedTorchSquashedGaussian(TorchSquashedGaussian):
    @override(ActionDistribution)
    def deterministic_sample(self) -> TensorType:
        self.last_normal_sample = self.dist.mean
        self.last_sample = self._squash(self.dist.mean)
        return self.last_sample

    @override(TorchDistributionWrapper)
    def sample(self) -> TensorType:
        # Use the reparameterization version of `dist.sample` to allow for
        # the results to be backprop'able e.g. in a loss term.
        normal_sample = self.dist.rsample()
        self.last_normal_sample = normal_sample
        self.last_sample = self._squash(normal_sample)
        return self.last_sample

    @override(ActionDistribution)
    def logp(self, x: TensorType) -> TensorType:
        if x is None:
            unsquashed_values = self.last_normal_sample
        else:
            # Unsquash values (from [low,high] to ]-inf,inf[)
            unsquashed_values = self._unsquash(x)
        # Get log prob of unsquashed values from our Normal.
        log_prob_gaussian = self.dist.log_prob(unsquashed_values)
        # For safety reasons, clamp somehow, only then sum up.
        log_prob_gaussian = torch.clamp(log_prob_gaussian, -100, 100)
        log_prob_gaussian = torch.sum(log_prob_gaussian, dim=-1)
        # Get log-prob for squashed Gaussian.
        unsquashed_values_tanhd = torch.tanh(unsquashed_values)
        log_prob = log_prob_gaussian - torch.sum(torch.log(1 - unsquashed_values_tanhd ** 2 + SMALL_NUMBER), dim=-1)
        return log_prob


def _get_dist_class_fixed(config: TrainerConfigDict, action_space: gym.spaces.Space
                          ) -> Type[TorchDistributionWrapper]:
    """Helper function to return a dist class based on config and action space.

    Args:
        config (TrainerConfigDict): The Trainer's config dict.
        action_space (gym.spaces.Space): The action space used.

    Returns:
        Type[TFActionDistribution]: A TF distribution class.
    """
    if isinstance(action_space, Discrete):
        return TorchCategorical
    elif isinstance(action_space, Simplex):
        return TorchDirichlet
    else:
        if config["normalize_actions"]:
            return FixedTorchSquashedGaussian if \
                not config["_use_beta_distribution"] else TorchBeta
        else:
            return TorchDiagGaussian


def action_distribution_fn_fixed(
        policy: Policy,
        model: ModelV2,
        input_dict: ModelInputDict,
        *,
        state_batches: Optional[List[TensorType]] = None,
        seq_lens: Optional[TensorType] = None,
        prev_action_batch: Optional[TensorType] = None,
        prev_reward_batch=None,
        explore: Optional[bool] = None,
        timestep: Optional[int] = None,
        is_training: Optional[bool] = None) -> \
        Tuple[TensorType, Type[TorchDistributionWrapper], List[TensorType]]:
    """The action distribution function to be used the algorithm.

    An action distribution function is used to customize the choice of action
    distribution class and the resulting action distribution inputs (to
    parameterize the distribution object).
    After parameterizing the distribution, a `sample()` call
    will be made on it to generate actions.

    Args:
        policy (Policy): The Policy being queried for actions and calling this
            function.
        model (TorchModelV2): The SAC specific Model to use to generate the
            distribution inputs (see sac_tf|torch_model.py). Must support the
            `get_policy_output` method.
        input_dict (ModelInputDict): The input-dict to be used for the model
            call.
        state_batches (Optional[List[TensorType]]): The list of internal state
            tensor batches.
        seq_lens (Optional[TensorType]): The tensor of sequence lengths used
            in RNNs.
        prev_action_batch (Optional[TensorType]): Optional batch of prev
            actions used by the model.
        prev_reward_batch (Optional[TensorType]): Optional batch of prev
            rewards used by the model.
        explore (Optional[bool]): Whether to activate exploration or not. If
            None, use value of `config.explore`.
        timestep (Optional[int]): An optional timestep.
        is_training (Optional[bool]): An optional is-training flag.

    Returns:
        Tuple[TensorType, Type[TorchDistributionWrapper], List[TensorType]]:
            The dist inputs, dist class, and a list of internal state outputs
            (in the RNN case).
    """
    # Get base-model output (w/o the SAC specific parts of the network).
    model_out, _ = model(input_dict, [], None)
    # Use the base output to get the policy outputs from the SAC model's
    # policy components.
    distribution_inputs = model.get_policy_output(model_out)
    # Get a distribution class to be used with the just calculated dist-inputs.
    action_dist_class = _get_dist_class_fixed(policy.config, policy.action_space)

    return distribution_inputs, action_dist_class, []


class MyLearningRateSchedule():
    """
    NOTE: do not inherit from LearningRateSchedule, as otherwise super().on_global_var_update() will always call its
    lr update as well
    """
    def __init__(self, lr_schedules: Optional[List]):
        if lr_schedules is None:
            # self.lr_schedule = ConstantSchedule(lr, framework=None)
            self.lr_schedules = None
        else:
            # backwards compatibility where we had only one schedule
            if isinstance(lr_schedules[0][0], (float, int)):
                self.lr_schedules = [PiecewiseSchedule(lr_schedules, outside_value=lr_schedules[-1][-1], framework=None) for _ in range(len(self._optimizers))]
            else:
                assert len(lr_schedules) == len(self._optimizers), lr_schedules
                self.lr_schedules = [PiecewiseSchedule(lr_sched, outside_value=lr_sched[-1][-1], framework=None) for lr_sched in lr_schedules]

    @override(Policy)
    def on_global_var_update(self, global_vars):
        super().on_global_var_update(global_vars)
        if self.lr_schedules is not None:
            for sched, opt in zip(self.lr_schedules, self._optimizers):
                cur_lr = sched.value(global_vars["timestep"])
                for p in opt.param_groups:
                    p["lr"] = cur_lr


def setup_late_mixins_myextension(policy: Policy, obs_space: gym.spaces.Space,
                      action_space: gym.spaces.Space,
                      config: TrainerConfigDict) -> None:
    """Call mixin classes' constructors after Policy initialization.

    - Moves the target model(s) to the GPU, if necessary.
    - Adds the `compute_td_error` method to the given policy.
    Calling `compute_td_error` with batch data will re-calculate the loss
    on that batch AND return the per-batch-item TD-error for prioritized
    replay buffer record weight updating (in case a prioritized replay buffer
    is used).
    - Also adds the `update_target` method to the given policy.
    Calling `update_target` updates all target Q-networks' weights from their
    respective "main" Q-metworks, based on tau (smooth, partial updating).

    Args:
        policy (Policy): The Policy object.
        obs_space (gym.spaces.Space): The Policy's observation space.
        action_space (gym.spaces.Space): The Policy's action space.
        config (TrainerConfigDict): The Policy's config.
    """
    setup_late_mixins(policy=policy, obs_space=obs_space, action_space=action_space,  config=config)
    MyLearningRateSchedule.__init__(policy, config["lr_schedule"])


def get_default_config_myextension():
    default_config = get_default_config()
    default_config['lr_schedule'] = None
    default_config['_use_gmm_distribution'] = False
    default_config['_gmm_k'] = 3
    return default_config


def optimizer_fn_fixed(policy: Policy, config: TrainerConfigDict) -> \
        Tuple[LocalOptimizer]:
    """Creates all necessary optimizers for SAC learning.

    The 3 or 4 (twin_q=True) optimizers returned here correspond to the
    number of loss terms returned by the loss function.

    Args:
        policy (Policy): The policy object to be trained.
        config (TrainerConfigDict): The Trainer's config dict.

    Returns:
        Tuple[LocalOptimizer]: The local optimizers to use for policy training.
    """
    # NOTE: DEFAULT RLLIB IS 1e-7
    def get_adam(params, lr, eps=1e-7):
        return torch.optim.Adam(params=params, lr=lr, eps=eps)
    policy.actor_optim = get_adam(params=policy.model.policy_variables(), lr=config["optimization"]["actor_learning_rate"])

    critic_split = len(policy.model.q_variables())
    if config["twin_q"]:
        critic_split //= 2

    policy.critic_optims = [get_adam(params=policy.model.q_variables()[:critic_split], lr=config["optimization"]["critic_learning_rate"])]
    if config["twin_q"]:
        policy.critic_optims.append(get_adam(params=policy.model.q_variables()[critic_split:], lr=config["optimization"]["critic_learning_rate"]))
    policy.alpha_optim = get_adam(params=[policy.model.log_alpha], lr=config["optimization"]["entropy_learning_rate"])

    return tuple([policy.actor_optim] + policy.critic_optims + [policy.alpha_optim])


SACTorchPolicyCustom = build_policy_class(
    name="SACTorchPolicyCustom",
    framework="torch",
    loss_fn=actor_critic_loss_custom,
    get_default_config=get_default_config_myextension,
    stats_fn=stats,
    postprocess_fn=postprocess_trajectory,
    extra_grad_process_fn=apply_grad_clipping_custom,
    optimizer_fn=optimizer_fn_fixed,
    validate_spaces=validate_spaces,
    before_loss_init=setup_late_mixins_myextension,
    make_model_and_action_dist=build_sac_model_and_action_dist,
    extra_learn_fetches_fn=concat_multi_gpu_td_errors,
    # mixins=[TargetNetworkMixin, ComputeTDErrorMixin, MyLearningRateSchedule],
    mixins=[TargetNetworkMixin, MyLearningRateSchedule],
    action_distribution_fn=action_distribution_fn_fixed,
)

SACTrainerCustom = SACTrainer.with_updates(name="SACTrainerCustom",
                                           get_policy_class=get_custom_sac_policy_class,
                                           default_policy=SACTorchPolicyCustom,
                                           default_config=get_default_config_myextension())
