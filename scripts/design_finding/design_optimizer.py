from typing import Any, Tuple
import numpy as np
from pathlib import Path
import logging
import time
import ray

import sys
sys.path.append("./scripts/")
from modulation.utils import launch_ros
from evaluation_ray import register_envs_models
from design_finding.xacro_handler import XacroHandler 
from design_finding.policy_optimizer import PolicyOptimization
from design_finding.design_eval_task import RobotDesignEvaluator
from design_finding.design_arguments import DesignArguments
from design_finding.optimizer_base import OptimizerInterface

class DesignOptimization(OptimizerInterface):
    '''
    Optimize Design based on the given policy_optimizer and evaluation
    
    Outer loop
    '''
    def __init__(self,  
                 policy_optimizer: PolicyOptimization, 
                 evaluation: RobotDesignEvaluator, 
                 logger : logging.Logger,
                 process : Any,
                 args : DesignArguments,
                 xacro_handler: XacroHandler) -> None:
        super().__init__(logger)
        self.xacro_handler = xacro_handler
        self.training = policy_optimizer
        self.evaluation = evaluation
        self.process = process
        self.args = args
        self.result = {}
        self.evaluation_res = []
        self.design_idx = 0

    def evaluate_design(self, config: dict, budget: float) -> Tuple[float, float]:
        '''
        evaluate the given design
        train the design for a certain budget and test the trained agent afterwards on several evaluation tasks for multiple times
        '''
        self.design_idx += 1
        # 1. load new design
        self.logger.debug("loading new design")
        self.xacro_handler.write_config_to_file(config)
        self.relaunch_nodes(training=True)
        # design_value = self.get_design_value("panda_joint_ewellix_lift_top_link")
        self.logger.info(f"written new design-configuration : {config}")
        # self.logger.info(f"actual design-configuration of simulation: {design_value}")
        # 2. train new design
        self.logger.debug(f"start training of configuration for {budget} timesteps")
        training_res, latest_checkpoint_path = self.training(design_idx=self.design_idx)
        self.logger.debug("finished training")
        self.result[self.design_idx] = training_res
        # 3. evaluate new desing
        self.relaunch_nodes(training=False)
        self.logger.debug("---- evaluation of configuration -----")
        all_metrics, all_episodes  = self.evaluation(Path(latest_checkpoint_path))
        self.evaluation_res.append([metric["success"] for metric in all_metrics]) 
        # 4. analyse evaluation results -> loss
        self.logger.debug(f"evaluation tasks (picknplace has fwd_orientation so 2 times): {self.evaluation.wandb_config.eval_tasks}")
        self.logger.debug(f"result from evaluation tasks: {self.evaluation_res[-1]}")
        # mean_reward = self.result[self.design_idx][self.args.metric]
        mean_success = self.result[self.design_idx]["custom_metrics"]["success_nojumps_mean"]
        # self.logger.debug(f"mean reward          : {mean_reward}")
        self.logger.debug(f"mean success          : {mean_success}")
        task_cnt = len(self.evaluation_res[-1])
        loss = task_cnt - np.sum(self.evaluation_res[-1])
        self.logger.debug(f"loss for desing {config}: {loss}")
        return loss, None



    def relaunch_nodes(self, training: bool):
        '''
        launch ros nodes idependant: starts the handle_launchfiles.py as a extern process
        :training: bool: if true the training nodes are started so we have always the same task we relaunch
        '''
        # shut down running ros processes
        if self.process is not None:
            self.process.terminate()
        
        ray.shutdown()
        time.sleep(5)
        
        if training:
            self.process = launch_ros(main_path=self.training.main_path, config=self.training.wandb_config, 
                                      always_relaunch = True, task=self.training.wandb_config.task) 
                                    #   pure_analytical="no" if self.training.wandb_config.finetune_from_run else None)
            ray.init(logging_level='DEBUG' if self.training.wandb_config.debug else 'INFO', local_mode=self.training.wandb_config.debug)
        else:
            #evaluation
            self.process = launch_ros(main_path=self.evaluation.main_path, config=self.evaluation.wandb_config,
                                      always_relaunch = True, 
                                      task=self.evaluation.wandb_config.eval_tasks[0])

        register_envs_models()
