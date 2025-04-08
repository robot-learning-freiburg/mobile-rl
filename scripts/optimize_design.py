from typing import List
from pathlib import Path
import torch
import logging
from datetime import datetime
import copy
import wandb
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.optimizers import BOHB

from modulation.dotdict import DotDict
from modulation.utils import env_creator, parse_args
from design_finding.policy_optimizer import PolicyOptimization
from design_finding.design_eval_task import RobotDesignEvaluator
from design_finding.design_arguments import DesignArguments
from design_finding.design_worker import DesignWorker
from design_finding.design_optimizer import DesignOptimization
from design_finding.xacro_handler import XacroHandler
from design_finding.manipulability.manipulability_optimizer import (
    ManipulabilityAnalysis,
)
from design_finding.design_util import make_output_dir, load_yaml_file, merge_configs
from design_finding.optimization_info.ur5_optimization_info import (
    optimization_joints_ur5,
)
from design_finding.optimization_info.franka_optimization_info import (
    optimization_joints_franka,
)


def optimize_bohp(args: DesignArguments, optimizer, logger: logging.Logger):
    """
    use hpbandster framework for optimize the hyperparameters via BO and the hyperBand algorithm with a bandit based approach
    https://automl.github.io/HpBandSter/build/html/index.html
    """
    run_id = "design_optimization"
    server_ip = "127.0.0.1"
    result_logger = hpres.json_result_logger(
        directory=args.log_path / "bohb", overwrite=True
    )
    NS = hpns.NameServer(run_id=run_id, host=server_ip, port=None)
    NS.start()
    # Step 2: Start a worker
    w = DesignWorker(design_arguments= args, optimizer=optimizer, nameserver=server_ip, run_id=run_id)
    w.run(background=True)
    logger.debug(f"loading previous run: {args.previous_run_dir}")
    previous_run = (
        hpres.logged_results_to_HBS_result(args.previous_run_dir)
        if args.previous_run_dir
        else None
    )

    # Step 3: Run an optimizer
    bohb = BOHB(
        configspace=w.get_configspace(),
        run_id=run_id,
        nameserver=server_ip,
        eta=args.eta,
        min_budget=args.min_budget,
        max_budget=args.max_budget,
        min_points_in_model=args.min_points_in_model,
        bandwidth_factor=args.bandwidth_factor,
        result_logger=result_logger,
        # random_fraction = 0.5,
        num_samples=args.num_samples,
        previous_result=previous_run,  # this is how you tell any optimizer about previous runs
    )
    res = bohb.run(n_iterations=args.n_iterations)  #
    # Step 4: Shutdown
    # After the optimizer run, we must shutdown the master and the nameserver.
    bohb.shutdown(shutdown_workers=True)
    NS.shutdown()
    # Step 5: Analysis
    id2config = res.get_id2config_mapping()
    incumbent = res.get_incumbent_id()
    best_config = id2config[incumbent]["config"]
    logger.debug(f"Best found configuration: {best_config}")
    logger.debug(
        "A total of %i unique configurations where sampled." % len(id2config.keys())
    )
    logger.debug("A total of %i runs where executed." % len(res.get_all_runs()))
    logger.debug(
        "Total budget corresponds to %.1f full function evaluations."
        % (sum([r.budget for r in res.get_all_runs()]) / args.max_budget)
    )


def create_logger(
    name: str, level: int, filehandler: logging.FileHandler
) -> logging.Logger:
    """
    set up individual logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level=level)
    logger.addHandler(filehandler)
    return logger


def setup_logger(
    log_path: Path,
) -> "tuple[logging.Logger, logging.Logger, logging.Logger]":
    log_file = log_path / "design_info.log"
    filehandler = logging.FileHandler(log_file)
    filehandler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logging_level = logging.DEBUG
    designer_logger = create_logger("DesignerLogger", logging_level, filehandler)
    optimizer_logger = create_logger("OptimizerLogger", logging_level, filehandler)
    eval_logger = create_logger("EvaluationLogger", logging_level, filehandler)

    return optimizer_logger, designer_logger, eval_logger



def get_environment(wandb_config, arm_selection):
    """
    set up env for manipulability analysis via ik-solver in c++
    """
    env_config = DotDict(copy.deepcopy(dict(wandb.config)))
    env_config["arm_selection"] = arm_selection
    env_config["task"] = wandb_config.task
    env_config["world_type"] = wandb_config.world_type
    env_config["node_handle"] = "eval_env"
    env_config["eval"] = True
    env_config["transition_noise_base"] = 0.0
    # so we don't have to create the local maps
    env_config["use_map_obs"] = False
    # don't need the visualisations from the env
    env_config["vis_env"] = False
    env_config["gamma"] = wandb_config.gamma
    env_config["fake_gazebo"] = False
    env_config["init_controllers"] = False
    env_config["eval"] = True
    env_config["node_handle"] = "eval_env"
    env = env_creator(env_config)
    return env


def main():
    """
    main function for the design optimization project
    """
    main_path = Path(__file__).parent.absolute()
    modified_config_path = "run_config.yaml"
    _, group, args, _ = parse_args((main_path), framework='ray', add_design_config=True)
    # override default values with config from yaml file
    override_config = load_yaml_file(modified_config_path)

    # Merge the configs
    final_config = merge_configs(args, override_config)
    ray_config, design_config = final_config["ray_config"], final_config["design_config"]
    args = DesignArguments(**design_config)

    print("Arguments:\n")
    for k, v in vars(args).items():
        if v == "None":
            setattr(args, k, None)
        print(f"  {k} = {v}")
    print()

    use_cuda = not args.cpu and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(
        f"-- running on device: {device} (cuda available: {torch.cuda.is_available()})"
    )

    # set up the logger and logging folder
    args.log_path = Path(args.log_path)
    formatted_time = datetime.now().strftime("%Y%m%d_%H%M")
    args.log_path = args.log_path / formatted_time
    make_output_dir(args.log_path)
    make_output_dir(args.log_path / "xacros")

    optimizer_log, designer_log, evaluation_log = setup_logger(args.log_path)
    policy_optimization = PolicyOptimization(args=args, logger=optimizer_log)
    optimization_joints = (
        optimization_joints_franka
        if args.arm_selection == "franka_arm"
        else optimization_joints_ur5
    )
    evaluator = RobotDesignEvaluator(logger=evaluation_log)
    process = policy_optimization.set_up(ray_config, group)
    evaluator.set_up(ray_config, policy_optimization.wandb_config, False)
    xacro_handler = XacroHandler(optimization_joints=optimization_joints, log_path=args.log_path, arm_selection=args.arm_selection)
    env = get_environment(policy_optimization.wandb_config, args.arm_selection)
    if args.optimization_type == "design":
        optimizer = DesignOptimization(
            policy_optimizer=policy_optimization,
            evaluation=evaluator,
            logger=designer_log,
            process=process,
            args=args,
            xacro_handler=xacro_handler,
        )
    elif args.optimization_type == "manipulability":
        optimizer = ManipulabilityAnalysis(
            args=args,
            logger=designer_log,
            wandb_config=policy_optimization.wandb_config,
            main_path=policy_optimization.main_path,
            env=env,
            xacro_handler=xacro_handler,
        )
    optimize_bohp(args, optimizer, logger=optimizer_log)


if __name__ == "__main__":
    main()
