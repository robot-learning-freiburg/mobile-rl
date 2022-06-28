import copy
import os
from pathlib import Path
from typing import List

from matplotlib import pyplot as plt

plt.style.use('seaborn')
import rospy

import ray
from ray import tune
# from ray.rllib.models.torch.complex_input_net import ComplexInputNetwork
from ray.tune.schedulers.async_hyperband import ASHAScheduler
from ray.tune.callback import Callback
import numpy as np

from modulation.utils import parse_args, frame_skip_curriculum_fn, launch_ros
from modulation.dotdict import DotDict
from modulation.myray.ray_utils import get_normal_task_config, get_local_ray_dir, get_local_wandb_dir, get_trainer_fn
from modulation.myray.ray_callbacks import TrainCallback, EvalCallback
from modulation.myray.ray_wandb import restore_ckpt_files
from evaluation_ray import register_envs_models


def main():
    main_path = Path(__file__).parent.absolute()
    run_name, group, args, cl_args = parse_args((main_path), framework='ray')
    wandb_config = DotDict(args)
    assert not wandb_config.restore_model and not wandb_config.resume_id, "Not implemented yet"

    launch_ros(main_path=main_path, config=wandb_config, task=wandb_config.task, pure_analytical=None)

    if wandb_config.vis_env and wandb_config.num_workers > 0:
        print("WON'T BE ABLE TO SEE THE VISUALISATIONS FROM REMOTE WORKERS. SET num_workers == 0")

    # need a node to listen to some stuff for the task envs
    rospy.init_node('kinematic_feasibility_py', anonymous=False)
    ray.init(logging_level='DEBUG' if wandb_config.debug else 'INFO', local_mode=wandb_config.debug)
    register_envs_models()

    ray_config = {
        # Whether to rollout "complete_episodes" or "truncate_episodes".
        "seed": wandb_config.seed,
        "callbacks": TrainCallback,
        # --------------------------------
        # Training
        # --------------------------------
        # If set, this will fix the ratio of replayed from a buffer and learned on
        # timesteps to sampled from an environment and stored in the replay buffer
        # timesteps. Otherwise, the replay will proceed at the native ratio
        # determined by (train_batch_size / rollout_fragment_length).
        # TODO: new ray version divides also by num_workers -> multiply by num_workers to have same
        "training_intensity": wandb_config.training_intensity * wandb_config.batch_size,
        # Number of env steps to optimize for before returning.
        # This should not be impacting training at all, but only how often we log, checkpoint, etc.
        "timesteps_per_iteration": 1000,
        # --------------------------------
        # Env
        # --------------------------------
        "env": 'modulation_rl_env',
        # ignored atm
        "env_config": {
            # not sure how I could make this unique across workers
            "node_handle": "train_env",
            "env": wandb_config.env,
            "task": wandb_config.task,
            "penalty_scaling": wandb_config.penalty_scaling,  # tune.choice([0.0, 0.1])
            "acceleration_penalty": wandb_config.acceleration_penalty,
            "time_step": wandb_config.time_step,
            "seed": wandb_config.seed,
            "world_type": wandb_config.world_type,
            "init_controllers": wandb_config.init_controllers,
            "learn_vel_norm": wandb_config.learn_vel_norm,  # tune.choice([0.1, 0.3, 0.5])
            "collision_penalty": wandb_config.collision_penalty,  # tune.choice([5, 10, 25])
            "vis_env": wandb_config.vis_env,
            "transition_noise_base": wandb_config.transition_noise_base,  # tune.choice([0.005, 0.01, 0.015]),
            "ikslack_dist": wandb_config.ikslack_dist,
            "ikslack_rot_dist": wandb_config.ikslack_rot_dist,
            "ikslack_sol_dist_reward": wandb_config.ikslack_sol_dist_reward,
            "ikslack_penalty_multiplier": wandb_config.ikslack_penalty_multiplier,
            "ik_fail_thresh": wandb_config.ik_fail_thresh,
            "use_map_obs": wandb_config.use_map_obs,
            "global_map_resolution": wandb_config.global_map_resolution,
            "local_map_resolution": wandb_config.local_map_resolution,
            "overlay_plan": wandb_config.overlay_plan,  # tune.choice([True, False])
            "concat_plan": wandb_config.concat_plan,
            "concat_prev_action": wandb_config.concat_prev_action,
            "gamma": wandb_config.gamma,
            "obstacle_config": wandb_config.obstacle_config,
            "simpleobstacle_spacing": wandb_config.simpleobstacle_spacing,  # tune.choice([1.5, 1.75])
            "simpleobstacle_offsetstd": wandb_config.simpleobstacle_offsetstd,  # tune.choice([1.5, 1.75])
            "eval": False,
            "frame_skip": wandb_config.frame_skip,
            "frame_skip_observe": wandb_config.frame_skip_observe,
            "frame_skip_curriculum": wandb_config.frame_skip_curriculum,
            "start_level": wandb_config.frame_skip,
            "algo": wandb_config.algo,
            "use_fwd_orientation": wandb_config.use_fwd_orientation,
            "iksolver": wandb_config.iksolver,
            "selfcollision_as_failure": wandb_config.selfcollision_as_failure,
            "bioik_center_joints_weight": wandb_config.bioik_center_joints_weight,
            "bioik_avoid_joint_limits_weight": wandb_config.bioik_avoid_joint_limits_weight,
            "bioik_regularization_weight": wandb_config.bioik_regularization_weight,
            "bioik_regularization_type": wandb_config.bioik_regularization_type,
            "learn_torso": wandb_config.learn_torso,
            "learn_joint_values": wandb_config.learn_joint_values,
        },
        "env_task_fn": frame_skip_curriculum_fn if wandb_config.frame_skip_curriculum else None,
        # If True, RLlib will learn entirely inside a normalized action space
        # (0.0 centered with small stddev; only affecting Box components) and
        # only unsquash actions (and clip just in case) to the bounds of
        # env's action space before sending actions back to the env.
        "normalize_actions": True,
        # Whether to clip rewards during Policy's postprocessing.
        # None (default): Clip for Atari only (r=sign(r)).
        # True: r=sign(r): Fixed rewards -1.0, 1.0, or 0.0.
        # False: Never clip.
        # [float value]: Clip at -value and + value.
        # Tuple[value1, value2]: Clip at value1 and value2.
        "clip_rewards": None,
        # If True, RLlib will clip actions according to the env's bounds
        # before sending them back to the env.
        # NOTE: (sven) This option should be obsoleted and always be False.
        "clip_actions": False,
        # Which observation filter to apply to the observation.
        "observation_filter": 'NoFilter',  # tune.choice(["NoFilter", "MeanStdFilter"])
        # Disable setting done=True at end of episode. This should be set to True
        # for infinite-horizon MDPs (e.g., many continuous control problems).
        "no_done_at_end": wandb_config.no_done_at_end,
        # --------------------------------
        # Evaluation
        # --------------------------------
        # Evaluate with every `evaluation_interval` training iterations.
        # The evaluation stats will be reported under the "evaluation" metric key.
        # Note that evaluation is currently not parallelized, and that for Ape-X
        # metrics are already only reported for the lowest epsilon workers.
        # NOTE: in 'training_iterations' == 'training_batch_size' for ppo and 'timesteps_per_iteration' for SAC
        "evaluation_interval": int(np.ceil(wandb_config.evaluation_frequency / 1000)),
        # Number of episodes to run per evaluation period. If using multiple
        # evaluation workers, we will run at least this many episodes total.
        "evaluation_num_episodes": wandb_config.nr_evaluations,
        # Internal flag that is set to True for evaluation workers.
        "in_evaluation": False,
        # Typical usage is to pass extra args to evaluation env creator
        # and to disable exploration by computing deterministic actions.
        # IMPORTANT NOTE: Policy gradient algorithms are able to find the optimal
        # policy, even if this is a stochastic one. Setting "explore=False" here
        # will result in the evaluation workers not using this optimal policy!
        # Switch on evaluation in parallel with training.
        # NOTE: not sure if using >1 is a good idea with the fixed eval episode seed I set in the ray callback!
        "evaluation_num_workers": 0 if wandb_config.debug else 1,
        "evaluation_parallel_to_training": wandb_config.debug == False,
        "evaluation_config": {
            # "num_envs_per_worker": 5,
            # Example: overriding env_config, exploration, etc:
            "env_config": {
                "task": wandb_config.task,
                "node_handle": "eval_env",
                "wandb_config": copy.deepcopy(dict(wandb_config)),
                "transition_noise_base": 0.0,
                "ik_fail_thresh": wandb_config.ik_fail_thresh,
                "eval": True,
                "frame_skip": 1 if wandb_config.frame_skip else 0,
                "simpleobstacle_spacing": 1.75,
                "simpleobstacle_offsetstd": 1.75/4.,
                "eval_seed": wandb_config.eval_seed,
            },
            "explore": False,
            "callbacks": EvalCallback,
        },
        # Example: https://github.com/ray-project/ray/blob/master/rllib/examples/custom_eval.py
        "custom_eval_function": None,
        # --------------------------------
        # Deployment
        # --------------------------------
        "compress_observations": False,
        "num_workers": wandb_config.num_workers,
        "framework": "torch",
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        # "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "num_gpus": wandb_config.num_gpus,
        # Number of CPUs to allocate per worker.
        "num_cpus_per_worker": wandb_config.num_cpus_per_worker,
        # Number of GPUs to allocate per worker. This can be fractional. This is
        # usually needed only if your env itself requires a GPU (i.e., it is a
        # GPU-intensive video game), or model inference is unusually expensive.
        "num_gpus_per_worker": wandb_config.num_gpus_per_worker,
        "num_envs_per_worker": wandb_config.num_envs_per_worker,
        # The strategy for the placement group factory returned by
        # `Trainer.default_resource_request()`. A PlacementGroup defines, which
        # devices (resources) should always be co-located on the same node.
        # For example, a Trainer with 2 rollout workers, running with
        # num_gpus=1 will request a placement group with the bundles:
        # [{"gpu": 1, "cpu": 1}, {"cpu": 1}, {"cpu": 1}], where the first bundle is
        # for the driver and the other 2 bundles are for the two workers.
        # These bundles can now be "placed" on the same or different
        # nodes depending on the value of `placement_strategy`:
        # "PACK": Packs bundles into as few nodes as possible.
        # "SPREAD": Places bundles across distinct nodes as even as possible.
        # "STRICT_PACK": Packs bundles into one node. The group is not allowed
        #   to span multiple nodes.
        # "STRICT_SPREAD": Packs bundles across distinct nodes.
        # "placement_strategy": "PACK",
        # Which metric to use as the "batch size" when building a
        # MultiAgentBatch. The two supported values are:
        # env_steps: Count each time the env is "stepped" (no matter how many
        #   multi-agent actions are passed/how many multi-agent observations
        #   have been returned in the previous step).
        # agent_steps: Count each individual agent step as one step.
        # "count_steps_by": "env_steps",
        "log_level": "DEBUG" if wandb_config.debug else "INFO",
        # "logger_config": {
        "wandb": {
            "project": wandb_config['project_name'],
            "api_key": os.environ.get("WANDB_API_KEY"),
            "group": group,
            # "name": run_name,
            "dir": get_local_wandb_dir(wandb_config),
            'original_config': dict(wandb_config),
        },
        # }
    }

    task_config_updates = get_normal_task_config(wandb_config=wandb_config)
    ray_config.update(task_config_updates)

    trainer_fn = get_trainer_fn(wandb_config.algo)

    # use this to be able to set breakpoints locally
    if wandb_config.debug:
        trainer = trainer_fn(config=ray_config)
        while True:
            trainer.train()
    else:
        callbacks = []
        # callbacks.append(FinalEvaluationCallback())

        # if not wandb_config.dry_run:
        # https://docs.ray.io/en/master/tune/tutorials/tune-wandb.html
        # api_key = os.environ.get("WANDB_API_KEY", None)
        # api_key_file = "~/.wandb_api_key" if api_key is None else None
        # wandb_cb = WandbLoggerCallback(api_key=api_key, api_key_file=api_key_file,
        #                                project=wandb_config['project_name'], group=group, name=run_name)
        # callbacks.append(wandb_cb)

        stop = {  # "training_iteration": args.stop_iters,
            "timesteps_total": wandb_config.total_steps,
            # "episode_reward_mean": args.stop_reward,
        }

        metric = 'evaluation/custom_metrics/success_mean'
        os.environ['TUNE_DISABLE_STRICT_METRIC_CHECKING'] = '1'
        mode = 'max'
        asha_scheduler = ASHAScheduler(
            time_attr='timesteps_total',
            # metric=metric,
            # mode=mode,
            max_t=max(75_000, wandb_config.total_steps),
            grace_period=min(800_000, wandb_config.total_steps),
            reduction_factor=2,
            brackets=1)

        # bohb_hyperband_scheduler = HyperBandForBOHB(
        #     time_attr="timesteps_total",
        #     max_t=3_000_000,
        #     reduction_factor=3,
        #     stop_last_trials=False)
        #
        # search_alg = TuneBOHB(max_concurrent=10, metric="episode_reward_mean", mode="max")

        results = tune.run(trainer_fn,
                           stop=stop,
                           config=ray_config,
                           verbose=wandb_config.ray_verbosity,
                           checkpoint_freq=100,
                           checkpoint_at_end=True,
                           # restore is probably the one I want: takes a path (could first restore from wandb) and allows to continue training
                           # restore=,
                           # bool & does not ally to continue training -> probably not what I want
                           # resume=,
                           local_dir=get_local_ray_dir(wandb_config),
                           # fix potentially too long filename for log dir
                           name=run_name[:180],
                           callbacks=callbacks,
                           scheduler=None,  # asha_scheduler if wandb_config.param_search_samples else None
                           # search_alg=search_alg,
                           metric=metric,
                           mode=mode,
                           num_samples=1,
                           # loggers=DEFAULT_LOGGERS + (WandbLogger, )
                           )
        print(results.best_config)

    ray.shutdown()


if __name__ == "__main__":
    main()
