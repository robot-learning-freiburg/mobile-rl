from hpbandster.core.worker import Worker
import yaml
import ConfigSpace as CS
import numpy as np
from design_finding.optimizer_base import OptimizerInterface
from design_finding.design_arguments import DesignArguments

standard_config = {
    "arm_pitch": 0.0,
    "arm_yaw": 0.0,
    "tower_yaw": 0.0,
    "tower_yValue": 0.0,
    "tower_xValue": 0.0,
    "end_effector_mount": 0.0
}

def load_manipulability_configs(drive_type: str, yaml_file: str):
    with open(yaml_file, 'r') as file:
        data = yaml.safe_load(file)
    if drive_type.lower() == "omni":
        drive_key = "Omni_Drive"  
    elif drive_type.lower() == "diff":
        drive_key = "Diff_Drive"
    elif drive_type.lower() == "ur5":
        drive_key = "UR_5"

    
    if drive_key in data:
        return [entry['info'] for entry in data[drive_key]]
    else:
        return []


class DesignWorker(Worker):

    def __init__(self, *args, design_arguments: DesignArguments,  optimizer: OptimizerInterface, **kwargs):
        super().__init__(*args, **kwargs)
        self.iterator = 0   
        self.optimizer = optimizer
        self.yaml_file = design_arguments.manipulability_design_configs_path
        self.use_manipulability_config = design_arguments.use_manipulability_config

    def compute(self, config: dict, budget: float, **kwargs) -> dict:
        """
        Simple example for a compute function
        The loss is just a the config + some noise (that decreases with the budget)

        For dramatization, the function can sleep for a given interval to emphasizes
        the speed ups achievable with parallel workers.

        Args:
            config: dictionary containing the sampled configurations by the optimizer
            budget: (float) amount of time/epochs/etc. the model can use to train

        Returns:
            dictionary with mandatory fields:
                'loss' (scalar)
                'info' (dict)
        """
        budget = int(budget)
        if self.use_manipulability_config is not None:
            manipulability_configs = load_manipulability_configs(self.use_manipulability_config, self.yaml_file)
            config = manipulability_configs[self.iterator%3] # assume we have top 3 manipulability configs
        # config = standard_config
        self.iterator += 1
        config_rounded = {key: round(value, 4) for key, value in config.items()}
        # check if value is available (not for each optimizer)
        if self.optimizer.training is not None:
            self.optimizer.training.set_max_training_steps(budget)
        
        res, avg_manipulability = self.optimizer.evaluate_design(config_rounded, budget)
        evaluation = self.optimizer.evaluation_res[-1] # get eval per task for visualization

        return({
                    'loss': float(res),  # this is the a mandatory field to run hyperband
                    'info':  {"arm_pitch": config_rounded["arm_pitch"],
                              "arm_yaw": config_rounded["arm_yaw"],
                              "tower_yaw": config_rounded["tower_yaw"],
                              "tower_yValue": config_rounded["tower_yValue"],
                              "tower_xValue": config_rounded["tower_xValue"],
                              "end_effector_mount": config_rounded["end_effector_mount"],
                              "budget": budget,
                              "evaluation":evaluation,
                              "avg_manipulability": avg_manipulability } 
                })
    
    @staticmethod
    def get_configspace() -> CS.ConfigurationSpace:
        '''
        Define the configuration space for the design optimization with all parameters that are optimized
        '''
        config_space = CS.ConfigurationSpace()   
        upper_bound = np.round((np.pi/2),4)-0.00003
        quantization_step = (upper_bound/90) 
        tower_ylimit = 0.2
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('arm_pitch', lower=0.0, upper=upper_bound, q=quantization_step))
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('arm_yaw', lower=-upper_bound, upper=upper_bound, q=quantization_step))
        # tower rotation between - pi/2 and + pi/2
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('tower_yaw', lower=-upper_bound, upper=upper_bound, q=quantization_step))
        # y value between -0.2 and +0.2 (else it isnt anylonger on the base)
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('tower_yValue', lower=-tower_ylimit, upper=tower_ylimit, q=0.01))
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('tower_xValue', lower=-0.05, upper=0.15, q=0.01))
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('end_effector_mount', lower=0, upper=upper_bound, q=quantization_step))
        return(config_space)