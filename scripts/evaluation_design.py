from pathlib import Path
from typing import List
import rospy
import json
import numpy as np
import csv
import matplotlib.pyplot as plt

from modulation.utils import setup_config_wandb, launch_ros, parse_args, get_design_parser
from evaluation_ray import ray_eval, register_envs_models
from optimize_design import setup_logger, get_environment


from design_finding.design_util import make_output_dir, get_subdirectories, load_yaml_file, merge_configs
from design_finding.ray_design_util import get_trainer_for_eval
from design_finding.design_arguments import DesignArguments
from design_finding.policy_optimizer import PolicyOptimization
from design_finding.manipulability.manipulability_optimizer import (
    MinimalManipulabilityAnalysis,
)
from design_finding.evaluation_logging_paths import BEST_DESIGN_RUNS


evaluation_tasks = [
    "rndstartrndgoal",
    "picknplace",
    "roomDoor",  # open door task
    "drawer",
    "simpleobstacle",  # also named randomobstacle
    "door",  # cabinet task
]

names = [
    "ur5_default",
    "ur5_manipulability",
    "ur5_SO_optimized",
    "ur5_all_tasks_optimized",
]

def set_trained_robot_config(xacro_paths: List[str], target_paths: List[str]):
    """
    The function `set_trained_robot_config` reads the content of source files and writes it to target
    files specified by the input paths.

    :param xacro_paths: The `xacro_paths` parameter is a list of file paths to Xacro files that contain
    the configuration for a trained robot
    :param target_paths: The `target_paths` parameter in the `set_trained_robot_config` function is a
    list of file paths where the content of the source xacro files will be written to. Each element in
    the `target_paths` list corresponds to a target file path where the content from the respective
    xacro file
    """
    # Read the content of the source file
    for xacro_path, target_path in zip(xacro_paths, target_paths):
        with open(xacro_path, "r") as f:
            source_content = f.read()

        # Write the content of the source file to the target file
        with open(target_path, "w") as f:
            f.write(source_content)


def log_res(result: List[dict], logging_path: Path, run: int):
    """
    Log the evaluation results to a json file
    """
    with open(logging_path / f"evaluation_results_{run}.json", "w") as f:
        json.dump(result, f)


def get_ray_dir(run_name: str, run: int) -> Path:
    """
    Get the path to the ray directory

    """
    ray_path = f"scripts/logs/ray/design_evaluation_{run_name}/{run}"
    sub_dirs = get_subdirectories(ray_path)
    if len(sub_dirs) > 1:
        raise ValueError(f"Multiple subdirectories in {ray_path}: {sub_dirs}")
    ray_path = Path(ray_path) / sub_dirs[0] / "checkpoint_001000" / "checkpoint-1000"
    return ray_path


def output_singularity_info(all_episodes, output_file: str):
    with open(output_file+".csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        header = ["Episdoe", "Index", "Rank", "Condition Number Norm 2", "Jacobian"]
        writer.writerow(header)
        # Store singular values for histogram
        singular_values = []

        # Process each Jacobian and write to file
        for episodes in all_episodes:
            for i, episode in enumerate(episodes):
                for j, info in enumerate(episode.infos):
                    jacobian = info['jacobian']
                    rank_J = np.linalg.matrix_rank(jacobian)
                    cond2_J = np.linalg.cond(jacobian, p=2)
                    singular_values.append(cond2_J)
                    jacobian_flat = jacobian.flatten().tolist()
                    writer.writerow([i, j] + [rank_J, cond2_J]+ jacobian_flat)

def get_singularity_info():
    folders = [
        "20241004_0704_18",
        "20241017_0659_3",
        "20241025_0842_11",
        "20241121_0943_2",
        ]
    for folder in folders:
        path = f"scripts/logs/design/evaluation/{folder}/"
        print(path)
        for file_path in Path(path).rglob('*.csv'):
            data = np.loadtxt(file_path, delimiter=',', skiprows=1)  # Adjust skiprows if there's a header

            condition_number = data[:, 3]
            jacobians = data[:, 4:].reshape(-1, 6, 6)

            manipulability = np.sqrt(np.linalg.det(jacobians @ jacobians.transpose(0, 2, 1)))
            plt.hist(manipulability, bins=100)
            plt.title(file_path.name)
            plt.savefig(f'{path}/{file_path.stem}.png')
            plt.close()

            singularities_001 = np.where(manipulability < 0.001)[0]
            singularities_01 = np.where(manipulability < 0.01)[0]
            singularity_ratio_001 = len(singularities_001) / len(manipulability)
            singularity_ratio_01 = len(singularities_01) / len(manipulability)

            condition_number_ratio = len(np.where(condition_number > 1000)[0]) / len(condition_number)
            print(f'{file_path.name:<40} singularity_ratio_001: {singularity_ratio_001:.4f}, singularity_ratio_01: {singularity_ratio_01:.4f}, condition_number_ratio: {condition_number_ratio:.4f}')


def evaluate_name(name):
    run_name = BEST_DESIGN_RUNS[name]["run_name"]
    run = BEST_DESIGN_RUNS[name]["run"]
    xacro_paths = [
        f"scripts/logs/design/{run_name}/xacros/franka_arm.xacro_{run}.urdf.xacro",
        f"scripts/logs/design/{run_name}/xacros/fmm.urdf.xacro_{run}.urdf.xacro",
    ]
    target_paths = [
        "gazebo_world/fmm/franka_arm/franka_arm.xacro",
        "/root/catkin_ws_fmm/src/fmm_description/urdf/fmm.urdf.xacro",
    ]
    if name.startswith("ur5"):
        # ur5 does not have a franka arm
        xacro_paths = [xacro_paths[1]]
        target_paths = [target_paths[1]]
    main_path = Path(__file__).parent.absolute()
    _, wandb_config = setup_config_wandb(
        main_path,
        sync_tensorboard=False,
        allow_init=True,
        no_ckpt_endig=True,
        framework="ray",
        raw_args=None,
    )
    ray_path = get_ray_dir(run_name, run)
    wandb_config.model_file = str(ray_path)
    logging_path = Path(wandb_config.logpath) / "design" / "evaluation" / f"{run_name}_{run}" 
    make_output_dir(logging_path)

    results_summary = []
    set_trained_robot_config(xacro_paths, target_paths)
    for task in evaluation_tasks:
        # set task to config so we load the specific environment for the task
        wandb_config.eval_tasks = [task]
        wandb_config.task = task
        wandb_config
        launch_ros(
            main_path=main_path,
            config=wandb_config,
            task=wandb_config.eval_tasks[0],
            always_relaunch=True,
        )
        rospy.init_node("kinematic_feasibility_py", anonymous=False)
        register_envs_models()
        trainer, ray_config = get_trainer_for_eval(wandb_config)
        all_metrics, all_episodes = ray_eval(trainer, wandb_config, ray_config)
        output_singularity_info(all_episodes,str(logging_path)+f"/singularity_{task}" )
        # reduce size of output file by removing unnecessary informationinfo
        all_metrics[0]["all_dists"] = None
        all_metrics[0]["all_rot_dists"] = None
        all_metrics[0]["task_name"] = task
        results_summary.append(all_metrics[0])

    log_res(results_summary, logging_path=logging_path, run=run)


def config_manipulabilities():
    """
    calculate the manipulability of the robot for the best configurations stored in best_design_runs
    """
    ur5 = False
    main_path = Path(__file__).parent.absolute()
    _, group, args, _ = parse_args((main_path), framework='ray', add_design_config=False)
    arm_selection = "franka_arm"
    if ur5:
        arm_selection = "ur5_arm"
    _, wandb_config = setup_config_wandb(
        main_path,
        sync_tensorboard=False,
        allow_init=False,
        no_ckpt_endig=True,
        framework="ray",
        raw_args=None,
    )
    # override default values with config from yaml file
    modified_config_path = "run_config.yaml"
    override_config = load_yaml_file(modified_config_path)
    ray_config, design_config = args, override_config["design_config"]
    args = DesignArguments(**design_config)
    loggin_path = Path(wandb_config.logpath) / "design" / "evaluation"
    optimizer_log, designer_log, _ = setup_logger(loggin_path)
    policy_optimization = PolicyOptimization(args=args, logger=optimizer_log)
    _ = policy_optimization.set_up(ray_config, group)
    for configuration in BEST_DESIGN_RUNS:
        if configuration != "omni_SO_optimized_retrained":
            continue
        if configuration.startswith("ur5") and not ur5:
            continue
        if not configuration.startswith("ur5") and ur5:
            continue

        run_name = BEST_DESIGN_RUNS[configuration]["run_name"]
        run = BEST_DESIGN_RUNS[configuration]["run"]
        xacro_paths = [
            f"scripts/logs/design/{run_name}/xacros/franka_arm.xacro_{run}.urdf.xacro",
            f"scripts/logs/design/{run_name}/xacros/fmm.urdf.xacro_{run}.urdf.xacro",
        ]
        target_paths = [
            "gazebo_world/fmm/franka_arm/franka_arm.xacro",
            "/root/catkin_ws_fmm/src/fmm_description/urdf/fmm.urdf.xacro",
        ]
        if configuration.startswith("ur5"):
            # ur5 does not have a franka arm
            xacro_paths = [xacro_paths[1]]
            target_paths = [target_paths[1]]
            ur5 = True
        set_trained_robot_config(xacro_paths, target_paths)
        env = get_environment(wandb_config, arm_selection)

        ray_path = get_ray_dir(run_name, run)
        wandb_config.model_file = str(ray_path)
        manipulability_optimization = MinimalManipulabilityAnalysis(
            designer_log, wandb_config, main_path, env
        )
        avg_manipulability = manipulability_optimization.evaluate_design_simple()
        result_str = f"Average manipulability for configuration: {configuration} is: {avg_manipulability}"
        print(result_str)
        optimizer_log.info(result_str)

if __name__ == "__main__":
    for name in names:
        evaluate_name(name)
    # get_singularity_info()
    # config_manipulabilities()
