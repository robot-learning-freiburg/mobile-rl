from pathlib import Path
import json

import numpy as np
import ray
import rospy
from matplotlib import pyplot as plt
from ray.rllib.models.catalog import ModelCatalog
from ray.tune.registry import register_env

import wandb

plt.style.use('seaborn')
from modulation.evaluation import evaluate_on_task, get_next_global_step, get_metric_file_path, download_wandb
from modulation.utils import setup_config_wandb, env_creator, launch_ros
from modulation.myray.ray_wandb import restore_ckpt_files
from modulation.myray.ray_utils import get_local_ray_dir, get_trainer_fn, get_normal_agent_model
from modulation.models.ray_mapencoder import RayMapEncoder, MapEncoderSACTorchModel
from modulation.myray.custom_sac import FixedTorchSquashedGaussian

def register_envs_models():
    register_env('modulation_rl_env', env_creator)
    ModelCatalog.register_custom_model("ray_map_encoder", RayMapEncoder)
    ModelCatalog.register_custom_model("map_encoder_sac_torch_network", MapEncoderSACTorchModel)
    ModelCatalog.register_custom_action_dist("FixedTorchSquashedGaussian", FixedTorchSquashedGaussian)


def ray_eval(trainer, wandb_config, ray_config):
    # create the env for evaluation
    eval_env_config = ray_config['env_config'].copy()
    eval_env_config.update(ray_config['evaluation_config']['env_config'])
    # overwrite world/execution settings with command line args
    eval_env_config['time_step'] = wandb_config.time_step
    eval_env_config['world_type'] = wandb_config.world_type
    eval_env_config['init_controllers'] = wandb_config.init_controllers
    eval_env_config['ik_fail_thresh'] = wandb_config.ik_fail_thresh_eval

    # evaluate
    policy = trainer.get_policy()

    all_metrics, all_episodes = [], []
    world_types = ["world"] if (wandb_config.world_type == "world") else wandb_config.eval_execs
    for world_type in world_types:
        # remove_metric_files_if_already_complete(world_type)
        world_metrics = []
        for task in wandb_config.eval_tasks:
            # can't pickle planners other than a-star yet
            eval_env_config['use_snapshot'] = False

            fwd_orientations = [False, True] if (task in ['picknplace', 'bookstorepnp']) else [False]
            for o in fwd_orientations:
                eval_env_config['use_fwd_orientation'] = o
                m, episodes = evaluate_on_task(wandb_config, policy, eval_env_config, task, world_type,
                                               global_step=None, debug=wandb_config.debug, eval_seed=wandb_config.eval_seed)
                world_metrics.append(m)
                all_metrics.append(m)
                all_episodes.append(episodes)

    return all_metrics, all_episodes


def main(raw_args=None):
    main_path = Path(__file__).parent.absolute()
    run, wandb_config = setup_config_wandb(main_path, sync_tensorboard=False, allow_init=False, no_ckpt_endig=True, framework='ray', raw_args=raw_args)
    launch_ros(main_path=main_path, config=wandb_config, task=wandb_config.eval_tasks[0])

    # need a node to listen to some stuff for the task envs
    rospy.init_node('kinematic_feasibility_py', anonymous=False)
    register_envs_models()

    local_dir = get_local_ray_dir(wandb_config)
    model_file_path, ray_config = restore_ckpt_files(resume_model_name=wandb_config.resume_model_name,
                                                     model_file=wandb_config.model_file,
                                                     target_dir=f"{local_dir}restored/")
    # we're not going to use the wandb_mixin
    ray_config.pop('wandb', None)
    # override certain values
    wandb_config['num_workers'] = 0
    for k in ['num_workers', 'num_cpus_per_worker', 'num_envs_per_worker', 'num_gpus', 'num_gpus_per_worker']:
        ray_config[k] = wandb_config[k]
    for k in ["task", "time_step", "world_type", "init_controllers", "vis_env", "ikslack_dist", "ikslack_rot_dist",
              "ik_fail_thresh", "obstacle_config", "simpleobstacle_spacing", "use_fwd_orientation", "debug",
              "simpleobstacle_offsetstd", "eval_seed", "exec_action_clip", "exec_action_scaling",
              "exec_acceleration_clip", "execute_style", "perception_style",]:
        ray_config['env_config'][k] = wandb_config[k]
        ray_config['evaluation_config']['env_config'][k] = wandb_config[k]

    # pickling error on daim as o/w I have to set num_workers > 0
    ray_config['training_intensity'] = None
    ray_config['evaluation_num_workers'] = 0
    ray_config['evaluation_parallel_to_training'] = False

    # backwards compatibility (key, default_value)
    for k, v in [("acceleration_penalty", 0.0), ("learn_joint_values", False), ('selfcollision_as_failure', wandb_config.selfcollision_as_failure)]:
        if ray_config['env_config'].get(k, None) is None:
            ray_config['env_config'][k] = v

    # restore trainer from checkpoint
    trainer_fn = get_trainer_fn(wandb_config.algo, use_wandb_mixin=False)
    trainer = trainer_fn(config=dict(ray_config))
    # NOTE: this will initialise a ray node, even when not using any remote workers
    ray.init(logging_level='DEBUG' if wandb_config.debug else 'INFO', num_gpus=0, object_store_memory=int(0.5*1e+9))
    trainer.restore(model_file_path)
    # ray puts a copy of the worker into remote storage. When deallocating this, it massively slows down the local one.
    # So shut down ray to prevent this bug
    ray.shutdown()

    return ray_eval(trainer, wandb_config, ray_config)


if __name__ == '__main__':
    main()
