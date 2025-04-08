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
import numpy as np

from modulation.utils import parse_args, launch_ros
from modulation.dotdict import DotDict
from modulation.myray.ray_utils import get_local_ray_dir, get_trainer_fn
from evaluation_ray import register_envs_models
from design_finding.ray_design_util import get_ray_config

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

    ray_config = get_ray_config(wandb_config, group)
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
