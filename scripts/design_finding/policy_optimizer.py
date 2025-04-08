
import os
import rospy
import ray
import logging
from datetime import datetime
from ray import tune
from typing import Any, List
from pathlib import Path
from rosgraph_msgs.msg import Log
from subprocess import Popen


import sys
sys.path.append('scripts/')
from design_finding.ray_design_util import get_ray_config
from design_finding.design_arguments import DesignArguments
from modulation.myray.ray_utils import get_local_ray_dir, get_trainer_fn
from modulation.utils import parse_args, launch_ros
from modulation.dotdict import DotDict
from evaluation_ray import register_envs_models
from typing import Tuple


KEYS_TO_INCLUDE = ['episode_reward_max', 
                   'episode_reward_min', 
                   'episode_reward_mean', 
                   'episode_len_mean', 
                   'custom_metrics', 
                   ] 

class PolicyOptimization():
    '''
    Inner loop
    '''
    def __init__(self, args: DesignArguments, logger: logging.Logger) -> None:
        self.main_path = Path(__file__).parent.parent.absolute()
        self.args = args
        self.wandb_config = None
        self.trainer_fn = None
        self.logger = logger
        self.logger.debug("init Optimizer")
        self.run_folder_name = datetime.now().strftime('%Y%m%d_%H%M')
        self.run_name = ""
        self.ray_config = None
        

    def __call__(self, design_idx: int) -> Tuple[Any, str]:
        '''
        start a normal trainig for x steps and save the results to y so we can evaluate it later
        
        '''
        return self.run_training( design_idx)
    
    def run_training(self, design_idx: int) -> Tuple[Any, str]:
        '''
        start a normal trainig for x steps and save the results to y so we can evaluate it later
        '''
        self.logger.debug(f"train design number {design_idx}")
        result = {}
        if self.args.debug:
            trainer = self.trainer_fn(config=self.ray_config)
            while True:
                  trainer.train()
        else:
            callbacks = []
            stop = {  
                "timesteps_total": self.args.training_steps,
            }
            metric = 'evaluation/custom_metrics/success_mean'
            os.environ['TUNE_DISABLE_STRICT_METRIC_CHECKING'] = '1'
            mode = 'max'
            results = tune.run(
                self.trainer_fn,
                stop=stop,
                config=self.ray_config,
                verbose=self.wandb_config.ray_verbosity,
                checkpoint_at_end=True,
                local_dir=get_local_ray_dir(self.wandb_config),
                name= f"{self.run_name}/{design_idx}",
                callbacks=callbacks,
                scheduler=None,  # asha_scheduler if wandb_config.param_search_samples else None
                metric=metric,
                mode=mode,
                num_samples=1,
                )
            _, first_value = next(iter(results.results.items()))
            result = {key: first_value[key] for key in KEYS_TO_INCLUDE if key in first_value}
            latest_checkpoint = results.trials[-1].checkpoint.value
        return result, latest_checkpoint


    def set_up(self, wbargs: List[dict], group: str) -> Popen:
        '''
        set up the ray trainer for optimize the RL agent for the given task
        '''
        # run_name, group, wbargs, cl_args = parse_args(config_path=(self.main_path), framework='ray', raw_args=ray_args, use_config=True)
        self.wandb_config = DotDict(wbargs)
        # self.wandb_config.eval_tasks = self.wandb_config.eval_tasks[1:]
        assert not self.wandb_config.restore_model and not self.wandb_config.resume_id, "Not implemented yet"

        process = launch_ros(self.main_path, config=self.wandb_config, 
                             always_relaunch = False, task=self.wandb_config.task) 
        rospy.init_node('kinematic_feasibility_py',log_level=Log.DEBUG, anonymous=False)
        ray.init(logging_level='DEBUG' if self.wandb_config.debug else 'INFO', local_mode=self.wandb_config.debug)
        register_envs_models()
        self.ray_config = get_ray_config(self.wandb_config, group)

        self.trainer_fn = get_trainer_fn(self.wandb_config.algo)
        self.run_name = f"design_evaluation_{self.run_folder_name}"
        return process

    def set_max_training_steps(self, steps: int):
        '''
        set the max training steps for the training
        '''
        self.args.training_steps = steps

    def get_main_path(self) -> Path:
        '''
        return the main path of the optimizer
        '''
        return self.main_path
