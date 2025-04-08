from pathlib import Path
import yaml
import copy
import wandb
import logging
from typing import List, Tuple
import sys


sys.path.append("./scripts/")
from design_finding.ray_design_util import get_trainer_for_eval
from modulation.dotdict import DotDict
from modulation.utils import set_seed
from evaluation_ray import ray_eval

class RobotDesignEvaluator():
    """
    Evaluate the design of a robot with given checkpoint
    """
    def __init__(self, logger: logging.Logger) -> None:
        self.main_path = Path(__file__).parent.parent.absolute()
        self.wandb_config = None
        self.logger = logger

    def __call__(self, latest_checkpoint_path: Path) -> Tuple[list, list]:
        """
        Evaluate the model with the given checkpoint
        used in the design_evaluation RL-Agent context
        """
        self.wandb_config["model_file"] = str(latest_checkpoint_path)
        trainer, ray_config = get_trainer_for_eval(self.wandb_config)

        return ray_eval(trainer, self.wandb_config, ray_config)

    def set_up(
        self, ray_args: List[dict], args: DotDict, sync_tensorboard: bool = False
    ) -> None:
        """
        Set up the evaluator with the given arguments
        """
        # Fix assertion error because we restore from directory
        args["resume_model_name"] = None

        model_file = args["model_file"]
        common_args = {
            "project": args.pop("project_name"),
            "dir": args["logpath"],
            "sync_tensorboard": sync_tensorboard,
        }
        print(f"RESTORING MODEL from {model_file}")
        yaml_file = self.main_path / "model_checkpoints" / args["env"]
        with open(yaml_file / "config.yaml", "rb") as f:
            raw_params = yaml.safe_load(f)
        params = {
            k: v["value"]
            for k, v in raw_params.items()
            if k not in ["_wandb", "wandb_version"]
        }
        params["model_file"] = False
        params["resume_id"] = None
        params["resume_model_name"] = None
        _ = wandb.init(config=params, **common_args)
        if args["evaluation_only"]:
            wandb.config.update({"evaluation_only": True}, allow_val_change=True)

        # update an alternative dict placeholder so we don't change the logged values which it was trained with
        config = DotDict(copy.deepcopy(dict(wandb.config)))

        for k, v in args.items():
            # allow to override loaded config with command line args
            if k in ray_args:
                config[k] = v
            # backwards compatibility if a config value didn't exist before
            if k not in wandb.config.keys():
                print(f"Key {k} not found in config. Setting to {v}")
                config[k] = args[k]
        # always update these values
        for k in [
            "init_controllers",
            "device",
            "num_workers",
            "num_cpus_per_worker",
            "num_envs_per_worker",
            "num_gpus",
            "num_gpus_per_worker",
            "nr_evaluations",
            "logpath",
            "simpleobstacle_spacing",
            "obstacle_config",
            "debug",
        ]:
            config[k] = args[k]

        set_seed(config.seed)
        self.wandb_config = config

