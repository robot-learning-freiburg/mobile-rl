from matplotlib import pyplot as plt

plt.style.use('seaborn')
from pathlib import Path
import enum
import copy
import wandb
import rospy
import time
from typing import Union
import numpy as np
from pprint import pprint
from modulation.utils import setup_config_wandb, env_creator, launch_ros
from modulation.dotdict import DotDict
from modulation.envs.tasks import TaskGoal
from modulation.envs.env_utils import yaw_to_quaternion, delete_all_markers, quaternion_to_list, quaternion_to_yaw
from pybindings_moveit import run_birrt_scenario_eepose, run_birrt_scenario_startconfig
from modulation.planning.rrt import PlanningEpisode, prepare_next_subgoal, summarise_episodes


class PLANNER_RETURN_CODE(enum.IntEnum):
    NoGoalConfig = 0
    Success = 1
    Failure = 2


def run_birrt(planning_group: str,
              ee_goal_pose_wrist: list,
              max_position_dev: float,  # meter
              max_rot_dev: float,  # rad
              # NOTE: no longer used
              constraint_vec_goal_pose: Union[tuple, list],
              start_ee_pose: list = None,
              constraint_vec_start_pose: list = Union[tuple, list],
              start_configuration: list = None,
              extra_configuration: dict = None,
              search_space: int = 1,
              max_iterations_time: float = 200.0,
              max_iterations_or_time: int = 1,
              rviz_show_tree: bool = True,
              iteration_sleep_time: float = 0.0,
              # NOTE: just set to something definitely larger than our maps (-x, x), (-y, y)
              env_size_x: tuple = (-20.0, 20.0),
              env_size_y: tuple = (-20.0, 20.0),
              n_loops_visualization: int = 1):

    assert len(ee_goal_pose_wrist) == 7, ee_goal_pose_wrist

    # assert len(ee_goal_pose_wrist) == len(constraint_vec_goal_pose) == 6, "xyzRPY, not quaternion"
    assert (start_configuration is not None) or ((start_ee_pose is not None) and (constraint_vec_start_pose is not None))
    assert (start_configuration is None) or (start_ee_pose is None), "Only pass one of start_configuration, start_ee_pose"

    # Permitted displacement for ee coordinates w.r.t desired target frame
    target_coordinate_dev = []
    target_coordinate_dev.append((-max_position_dev, max_position_dev))   # negative, positive X deviation [m]
    target_coordinate_dev.append((-max_position_dev, max_position_dev))   # negative, positive Y deviation [m]
    target_coordinate_dev.append((-max_position_dev, max_position_dev))   # negative, positive Z deviation [m]
    target_coordinate_dev.append((-max_rot_dev, max_rot_dev))   # negative, positive Xrot deviation [rad]
    target_coordinate_dev.append((-max_rot_dev, max_rot_dev))   # negative, positive Yrot deviation [rad]
    target_coordinate_dev.append((-max_rot_dev, max_rot_dev))   # negative, positive Zrot deviation [rad]

    if (start_ee_pose is not None) and (constraint_vec_start_pose is not None):
        raise NotImplementedError("Not fully updated this consctor. Expects ee tip not wrist as input. Uses convertEulertoQuat() which I'm not sure whether it's correct")
        assert len(start_ee_pose) == len(constraint_vec_start_pose) == 6, "xyzRPY, not quaternion"
        assert extra_configuration is None, extra_configuration
        stats = run_birrt_scenario_eepose(planning_group,
                                          list(env_size_x),
                                          list(env_size_y),
                                          start_ee_pose,
                                          list(constraint_vec_start_pose),
                                          ee_goal_pose_wrist,
                                          list(constraint_vec_goal_pose),
                                          target_coordinate_dev,
                                          search_space,
                                          max_iterations_time,
                                          max_iterations_or_time,
                                          rviz_show_tree,
                                          iteration_sleep_time,
                                          n_loops_visualization)
    else:
        stats = run_birrt_scenario_startconfig(planning_group,
                                               list(env_size_x),
                                               list(env_size_y),
                                               start_configuration,
                                               extra_configuration,
                                               ee_goal_pose_wrist,
                                               list(constraint_vec_goal_pose),
                                               target_coordinate_dev,
                                               search_space,
                                               max_iterations_time,
                                               max_iterations_or_time,
                                               rviz_show_tree,
                                               iteration_sleep_time,
                                               n_loops_visualization)
    return stats


def birrt_rollout(env, num_eval_episodes: int, max_iterations_time: float, planning_group: str, eval_seed: int,
                  name_prefix: str = '', n_loops_visualization: int = 1):
    name_prefix = f"{name_prefix + '_' if name_prefix else ''}{env.loggingname}"

    episodes = []

    for i in range(num_eval_episodes):
        t = time.time()
        if eval_seed is not None:
            env.seed(eval_seed + i)
        obs = env.reset()
        planning_time_remaining = max_iterations_time

        if hasattr(env, "goals"):
            # this is a chained task
            goals = env.goals
            # remove for now, as o/w will fail due to collisions with the object of interest
            env.map.simulator.delete_model('pick_obj')
        else:
            goals = [env.unwrapped._ee_planner.gripper_goal_tip]

        episode = PlanningEpisode()
        episode.delete_markers()

        for j, goal in enumerate(goals):
            delete_all_markers('/birrt_star_markers', False)
            delete_all_markers('/birrt_star_markerarrays', True)

            ee_goal_pose_tip, ee_goal_pose_wrist, ee_plan_ours, _ = prepare_next_subgoal(env,
                                                                                         goal=goal,
                                                                                         planning_group=planning_group,
                                                                                         episode=episode,
                                                                                         subgoal_idx=j)
            if j == 0:
                start_configuration = env.get_joint_values(planning_group)
                base = env.get_robot_obs().base_tf
            else:
                start_configuration = planner_output.joint_trajectory[-1]
                base = episode.base_trajectory[-1]
            start_configuration[0] = base[0]
            start_configuration[1] = base[1]
            start_configuration[2] = quaternion_to_yaw(base[3:])
            # extra_configuration = copy.deepcopy(env.robot_config.get("initial_joint_values", {}))
            extra_configuration = dict(zip(env.get_joint_names("whole_body"), env.get_joint_values("whole_body")))
            del extra_configuration['base_x_joint']
            del extra_configuration['base_y_joint']
            del extra_configuration['base_theta_joint']

            print(f"\n\nCalling run_birrt() at {time.strftime('%H:%M:%S', time.localtime())}")
            planner_output = run_birrt(start_ee_pose=None,
                                       constraint_vec_start_pose=None,
                                       start_configuration=start_configuration,
                                       extra_configuration=extra_configuration,
                                       ee_goal_pose_wrist=ee_goal_pose_wrist,
                                       max_position_dev=goal.success_thres_dist if isinstance(goal, TaskGoal) else env.unwrapped._ee_planner._success_thres_dist,
                                       max_rot_dev=goal.success_thres_rot if isinstance(goal, TaskGoal) else env.unwrapped._ee_planner._success_thres_rot,
                                       constraint_vec_goal_pose=tuple([1, 1, 1, 1, 1, 1]),
                                       max_iterations_time=planning_time_remaining,
                                       planning_group=planning_group,
                                       n_loops_visualization=n_loops_visualization)
            stats = planner_output.stats.copy()
            if planner_output.return_code == PLANNER_RETURN_CODE.NoGoalConfig:
                stats['total_plannig_time'] = planning_time_remaining
                stats['first_solution_time'] = planning_time_remaining

            print(PLANNER_RETURN_CODE(planner_output.return_code).name)
            pprint(stats)


            if stats.get("success", None):
                assert stats['success'] == (planner_output.return_code == PLANNER_RETURN_CODE.Success)
            if planner_output.return_code != PLANNER_RETURN_CODE.NoGoalConfig:
                assert stats['success'] == (planner_output.return_code == PLANNER_RETURN_CODE.Success)

            base_trajectory = []
            for p in planner_output.joint_trajectory:
                base_trajectory.append([p[0], p[1], 0.] + quaternion_to_list(yaw_to_quaternion(p[2])))
            episode.add_subgoal(ee_trajectory=planner_output.ee_trajectory,
                                base_trajectory=base_trajectory,
                                ee_trajectory_ours=ee_plan_ours,
                                joint_trajectory=planner_output.joint_trajectory,
                                success=planner_output.return_code == PLANNER_RETURN_CODE.Success,
                                planning_time=stats['total_plannig_time'],
                                planning_time_first=stats['first_solution_time'],
                                extra_metrics=dict(stats),
                                return_code=PLANNER_RETURN_CODE(planner_output.return_code).name)
            episode.publish_ee_path(env)

            # TODO: for chainedtasks: should I continue after finding the first solution? O/w might use up all time trying to increase the first goal
            # NOTE: episode.ee_trajectory is for the last element in the planning_group
            # if episode.success:
            #     dist_to_goal = calc_euclidean_tf_dist(ee_goal_pose_wrist, episode.ee_trajectory[-1])
            #     rot_dist_to_goal = calc_rot_dist(ee_goal_pose_wrist, episode.ee_trajectory[-1])
            #     assert dist_to_goal <= env.unwrapped._ee_planner._success_thres_dist, dist_to_goal
            #     assert rot_dist_to_goal <= env.unwrapped._ee_planner._success_thres_rot, rot_dist_to_goal

            planning_time_remaining -= stats['total_plannig_time']
            if (planning_time_remaining <= 0.0) and (j < len(goals) - 1):
                # mark as a failure
                episode.add_subgoal(ee_trajectory=[], base_trajectory=[], ee_trajectory_ours=[], joint_trajectory=[], success=False, planning_time=0., return_code="REACHED_TIMEOUT")
            print(f"ep {i} SUBGOAL {j + 1}/{len(goals)}: {PLANNER_RETURN_CODE(planner_output.return_code).name}, time used: {stats['total_plannig_time']:.2f}, remaining: {planning_time_remaining:.2f} sec")

            if not episode.success:
                break

        episodes.append(episode)
        rospy.loginfo(f"{name_prefix}: Eval ep {i}: {PLANNER_RETURN_CODE(planner_output.return_code).name}, {episode.total_planning_time:.1f} sec planning time, "
                      f"{episode.ee_path_length / episode.ee_path_length_ours if episode.success else -1:.2f} rel-ee-path length, {sum([e.success for e in episodes])}/{len(episodes)} successfull.")

    overall_stats = summarise_episodes(episodes)

    log_dict = {f'{name_prefix}/{k}': v for k, v in overall_stats.items()}
    log_dict['global_step'] = i
    wandb.log(log_dict, step=i)
    plt.close('all')


def evaluate_birrt(wandb_config, task: str, world_type: str, debug: bool = False):
    env_config = DotDict(copy.deepcopy(dict(wandb.config)))
    env_config['task'] = task
    env_config['world_type'] = world_type
    env_config['node_handle'] = "eval_env"
    env_config['eval'] = True
    env_config["transition_noise_base"] = 0.0
    # so we don't have to create the local maps
    env_config["use_map_obs"] = False
    # don't need the visualisations from the env
    env_config["vis_env"] = False
    env_config['gamma'] = wandb_config.gamma
    # HACK: helper to spawn the task objects, but not having to use the robot (as modified base pr2 does not spawn correctly)
    env_config['fake_gazebo'] = True
    rospy.set_param("/fake_gazebo", True)

    # only to circumvent a check in the robotenv, does not have an actual impact
    iksolvers = {"pr2": "default",
                 "hsr": "bioik"}
    env_config["iksolver"] = iksolvers[wandb_config.env]
    env = env_creator(env_config)

    rospy.loginfo(f"Evaluating on task {env.loggingname} with {world_type} execution.")
    prefix = ''

    birrt_rollout(env,
                  wandb_config["nr_evaluations"],
                  max_iterations_time=wandb_config.planner_max_iterations_time,
                  planning_group="pr2_base_wrist",
                  name_prefix=prefix,
                  n_loops_visualization=1 if wandb_config.vis_env else 0,
                  eval_seed=wandb_config.eval_seed)
    env.clear()


def main():
    main_path = Path(__file__).parent
    run, wandb_config = setup_config_wandb(main_path, sync_tensorboard=False, allow_init=True, no_ckpt_endig=True, framework='ray')
    assert wandb_config['algo'] == 'bi2rrt', wandb_config['algo']

    launch_ros(main_path=main_path, config=wandb_config, task=wandb_config.eval_tasks[0], algo=wandb_config['algo'], pure_analytical="no")

    # need a node to listen to some stuff for the task envs
    rospy.init_node('kinematic_feasibility_py', anonymous=False)

    rospy.set_param('constraint_extend_step_factor', wandb_config.bi2rrt_extend_step_factor)
    rospy.set_param('unconstraint_extend_step_factor', wandb_config.bi2rrt_extend_step_factor)

    world_types = ["world"] if (wandb_config.world_type == "world") else wandb_config.eval_execs
    for world_type in world_types:
        assert world_type == 'sim', "atm have to always run with gazebo to spawn the objects into planning scene"
        for task in wandb_config.eval_tasks:
            evaluate_birrt(wandb_config, task=task, world_type=world_type, debug=wandb_config.debug)


if __name__ == '__main__':
    main()
