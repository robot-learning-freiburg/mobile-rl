from matplotlib import pyplot as plt

plt.style.use('seaborn')
from pathlib import Path
import copy
import wandb
import rospy
import time
from scipy.spatial.transform import Rotation
import numpy as np
from dataclasses import dataclass

from modulation.utils import setup_config_wandb, env_creator, launch_ros
from modulation.dotdict import DotDict
from modulation.envs.tasks import TaskGoal
from modulation.envs.env_utils import calc_rot_dist, calc_euclidean_tf_dist
from modulation.planning.rrt import PlanningEpisode, summarise_episodes, prepare_next_subgoal
from pybindings_moveit import MoveitPlanner

from moveit_msgs.msg import MotionPlanResponse
import inspect
ERROR_CODES = {}
for i in inspect.getmembers(MotionPlanResponse().error_code):
    # to remove private and protected
    # functions
    if not i[0].startswith('_'):
        # To remove other methods that
        # doesnot start with a underscore
        if not inspect.ismethod(i[1]):
            ERROR_CODES[i[1]] = i[0]


# @dataclass
# class MsgCallback:
#     msg = None
#
#     def cb(self, msg):
#         self.msg = msg
#
#     def wait_for_it(self):
#         i = 0
#         while self.msg is None:
#             i += 1
#             time.sleep(0.5)
#             assert i < 10, "Could not get result msg"
#         return self.msg


def run_moveit_cartesian_path(initial_joint_values: dict,
                              waypoints: list,
                              goal_frame_id: str,
                              base_frame_link: str,
                              planning_group: str,
                              param_handle: str, global_link_transform,
                              success_thresh_dist: float,
                              success_thresh_rot: float,
                              jump_threshold: float,
                              eef_step: float,
                              moveit_planner: MoveitPlanner):

    out = moveit_planner.plan_moveit_cartesian_plan(waypoints,
                                     global_link_transform,
                                     goal_frame_id,
                                     planning_group,
                                     success_thresh_dist,
                                     success_thresh_rot,
                                     initial_joint_values,
                                     param_handle,
                                     base_frame_link,
                                     jump_threshold,
                                     eef_step)
    return out


def run_moveit_planning(initial_joint_values: dict,
                        goal_pose: list,
                        goal_frame_id: str,
                        base_frame_link: str,
                        planning_group: str,
                        param_handle: str, global_link_transform,
                        success_thresh_dist: float,
                        success_thresh_rot: float,
                        allowed_planning_time: float,
                        num_planning_attempts: float,
                        planner_id: str,
                        path_orientation_constraint: list,
                        path_orientation_constraint_tolerance: list,
                        moveit_planner: MoveitPlanner,
                        planner_params: dict):
    assert len(goal_pose) in [6, 7], goal_pose
    if path_orientation_constraint is None:
        path_orientation_constraint, path_orientation_constraint_tolerance = [], []
    else:
        assert len(path_orientation_constraint) == 4, path_orientation_constraint
        assert len(path_orientation_constraint_tolerance) == 3, path_orientation_constraint_tolerance

    planner_plugin_name = "ompl_interface/OMPLPlanner"
    planner_params_str = {k: f'{v:g}' for k, v in planner_params.items()}
    out = moveit_planner.plan_moveit_motion(goal_pose,
                                            global_link_transform,
                                            goal_frame_id,
                                            planning_group,
                                            planner_plugin_name,
                                            success_thresh_dist,
                                            success_thresh_rot,
                                            allowed_planning_time,
                                            num_planning_attempts,
                                            initial_joint_values,
                                            param_handle,
                                            base_frame_link,
                                            planner_id,
                                            path_orientation_constraint,
                                            path_orientation_constraint_tolerance,
                                            planner_params_str)
    return out


def moveit_rollout(env, num_eval_episodes: int, planning_group: str, param_handle: str, allowed_planning_time: float,
                   planning_time_per_subgoal: bool, num_planning_attempts: float, goal_frame_id: str,
                   base_frame_link: str, planner_id: str, orientation_constraint: bool, eval_seed: int,
                   planner_params: dict, name_prefix: str = ''):
    name_prefix = f"{name_prefix + '_' if name_prefix else ''}{env.loggingname}"

    episodes = []

    for i in range(num_eval_episodes):
        # NOTE: robot can fall over in gazebo after some time, can be reset with world reset, but does not seem to matter
        t = time.time()
        if eval_seed is not None:
            env.seed(eval_seed + i)
        obs = env.reset()
        planning_time_remaining = allowed_planning_time

        if hasattr(env, "goals"):
            # this is a chained task
            goals = env.goals
            # remove for now, as o/w will fail due to collisions with the object of interest
            env.map.simulator.delete_model('pick_obj')
        else:
            goals = [env.unwrapped._ee_planner.gripper_goal_tip]

        moveit_planner = MoveitPlanner("moveit_planner")
        # result_cb = MsgCallback()
        # result_sub = rospy.topics.Subscriber("/moveit_planner/motion_response", MotionPlanResponse, result_cb.cb, queue_size=10)

        episode = PlanningEpisode()
        episode.delete_markers()
        for j, goal in enumerate(goals):
            ee_goal_pose_tip, ee_goal_pose_wrist, ee_plan_ours, start_configuration = prepare_next_subgoal(env,
                                                                                                           goal=goal,
                                                                                                           planning_group=planning_group,
                                                                                                           episode=episode,
                                                                                                           subgoal_idx=j)

            # account for potentially different ee-orientation. Usually we do this within the robotEnv
            if env.robot_config['gripper_to_base_rot_offset'] != [0., 0., 0., 1.]:
                oriented = Rotation(ee_goal_pose_wrist[3:]) * Rotation(env.robot_config['gripper_to_base_rot_offset']).inv()
                ee_goal_pose_wrist_oriented = ee_goal_pose_wrist[:3] + oriented.as_quat().tolist()
            else:
                ee_goal_pose_wrist_oriented = ee_goal_pose_wrist
            # env.publish_marker(ee_goal_pose_wrist_oriented, 0, "ee_goal_pose_wrist_oriented", "pink", 1.0, frame_id="map")

            success_thresh_dist = goal.success_thres_dist if isinstance(goal, TaskGoal) else env.unwrapped._ee_planner._success_thres_dist
            success_thresh_rot = goal.success_thres_rot if isinstance(goal, TaskGoal) else env.unwrapped._ee_planner._success_thres_rot

            if orientation_constraint and (env.taskname() in ['picknplace', 'bookstorepnp']) and (j == 2):
                # assert env.env_name != 'hsr', "Somehow failing for hsr. Maybe because adding the constraint changes planning to be in task space"
                path_orientation_constraint = ee_goal_pose_wrist_oriented[3:]
                path_orientation_constraint_tolerance = [success_thresh_rot, success_thresh_rot, 2 * np.pi]
            else:
                path_orientation_constraint, path_orientation_constraint_tolerance = None, None

            # TODO: delete pick_obj and other task objects so planning can achieve the goal pose? (Alternative: set to allowed in collision matrix)
            print(f"\n\nCalling run_moveit_planning() at {time.strftime('%H:%M:%S', time.localtime())}")
            moveit_result = run_moveit_planning(initial_joint_values=start_configuration,
                                                goal_pose=ee_goal_pose_wrist_oriented,
                                                planning_group=planning_group,
                                                param_handle=param_handle,
                                                global_link_transform=env.robot_config['global_link_transform'],
                                                success_thresh_dist=success_thresh_dist,
                                                success_thresh_rot=success_thresh_rot,
                                                allowed_planning_time=planning_time_remaining,
                                                num_planning_attempts=num_planning_attempts,
                                                goal_frame_id=goal_frame_id,
                                                base_frame_link=base_frame_link,
                                                planner_id=planner_id,
                                                path_orientation_constraint=path_orientation_constraint,
                                                path_orientation_constraint_tolerance=path_orientation_constraint_tolerance,
                                                moveit_planner=moveit_planner,
                                                planner_params=planner_params)

            # NOTE: by default uses the cartesianInterpolator (see source code: https://github.com/ros-planning/moveit/blob/master/moveit_ros/move_group/src/default_capabilities/cartesian_path_service_capability.cpp)
            # Does not work very well. There are some potentially better approaches: https://picknik.ai/cartesian%20planners/moveit/motion%20planning/2021/01/07/guide-to-cartesian-planners-in-moveit.html
            # run_moveit_cartesian_path(initial_joint_values=start_configuration,
            #                           waypoints=ee_plan_ours.tolist(),
            #                           planning_group=planning_group,
            #                           param_handle=param_handle,
            #                           global_link_transform=env.robot_config['global_link_transform'],
            #                           success_thresh_dist=success_thresh_dist,
            #                           success_thresh_rot=success_thresh_rot,
            #                           goal_frame_id=goal_frame_id,
            #                           base_frame_link=base_frame_link,
            #                           jump_threshold=5.0,
            #                           eef_step=TIME_STEP_TRAIN)

            ee_traj_oriented = []
            for p in moveit_result.ee_trajectory:
                oriented = p[:3] + (Rotation(p[3:]) * Rotation(env.robot_config['gripper_to_base_rot_offset'])).as_quat().tolist()
                ee_traj_oriented.append(oriented)

            episode.add_subgoal(ee_trajectory=ee_traj_oriented,
                                base_trajectory=moveit_result.base_trajectory,
                                ee_trajectory_ours=ee_plan_ours,
                                joint_trajectory=moveit_result.joint_trajectory_point_positions,
                                success=moveit_result.success,
                                planning_time=moveit_result.planning_time,
                                return_code=ERROR_CODES[moveit_result.error_code])
            episode.publish_ee_path(env)

            if episode.success:
                dist_to_goal = calc_euclidean_tf_dist(ee_goal_pose_wrist, episode.ee_trajectory[-1])
                rot_dist_to_goal = calc_rot_dist(ee_goal_pose_wrist, episode.ee_trajectory[-1])
                assert dist_to_goal <= success_thresh_dist, dist_to_goal
                assert rot_dist_to_goal <= success_thresh_rot, rot_dist_to_goal

            if not planning_time_per_subgoal:
                planning_time_remaining -= moveit_result.planning_time
                if (planning_time_remaining <= 0.0) and (j < len(goals) - 1):
                    # mark as a failure
                    episode.add_subgoal(ee_trajectory=[], base_trajectory=[], ee_trajectory_ours=[], joint_trajectory=[], success=False, planning_time=0., return_code="REACHED_TIMEOUT")
                print(f"ep {i}, SUBGOAL {j + 1}/{len(goals)}: {ERROR_CODES[moveit_result.error_code]}, time used: {moveit_result.planning_time:.2f}, remaining: {planning_time_remaining:.2f} sec")

            if not episode.success:
                break

        episodes.append(episode)
        rospy.loginfo(f"{name_prefix}: Eval ep {i}: {episode.total_planning_time:.1f} sec planning time, {ERROR_CODES[moveit_result.error_code]}, "
                      f"{episode.ee_path_length / episode.ee_path_length_ours if episode.success else -1:.2f} rel-ee-path length, {sum([e.success for e in episodes])}/{len(episodes)} successfull.")

    overall_stats = summarise_episodes(episodes)
    log_dict = {f'{name_prefix}/{k}': v for k, v in overall_stats.items()}
    log_dict['global_step'] = i
    wandb.log(log_dict, step=i)
    plt.close('all')


def evaluate_moveit_planning(wandb_config, task: str, world_type: str, param_handle: str, goal_frame_id: str,
                             base_frame_link: str, planning_group: str, planner_id: str, planning_time_per_subgoal: bool,
                             planner_params: dict):
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
    env_config['init_controllers'] = True
    rospy.set_param("/fake_gazebo", True)

    # TODO: maybe the change the ik solver timeout as well? atm 15sec in the Bi2rrt config
    # only to circumvent a check in the robotenv, does not have an actual impact
    iksolvers = {"pr2": "default",
                 "hsr": "bioik"}
    env_config["iksolver"] = iksolvers[wandb_config.env]

    env = env_creator(env_config)
    if env.taskname() == 'picknplace':
        # Check to ensure that things spawn correctly in docker
        assert env.map.simulator.get_model("wall_rightTop_model", "map").success, "Did the modulation_tasks.world map not spawn correctly??"

    rospy.loginfo(f"Evaluating on task {env.loggingname} with {world_type} execution.")
    prefix = ''

    if env.env_name == "hsr":
        env.set_world("gazebo")
        env.set_base_tf([30.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        env.set_world("sim")

    moveit_rollout(env,
                   num_eval_episodes=wandb_config["nr_evaluations"],
                   planning_group=planning_group,
                   name_prefix=prefix,
                   param_handle=param_handle,
                   allowed_planning_time=wandb_config.planner_max_iterations_time,
                   planning_time_per_subgoal=planning_time_per_subgoal,
                   num_planning_attempts=wandb_config.moveit_num_planning_attempts,
                   goal_frame_id=goal_frame_id,
                   base_frame_link=base_frame_link,
                   planner_id=planner_id,
                   orientation_constraint=wandb_config.moveit_orientation_constraint,
                   eval_seed=wandb_config.eval_seed,
                   planner_params=planner_params)


def set_or_delete_param(name, value):
    if (value == -1) and (rospy.get_param(name, None) is not None):
        rospy.delete_param(name)
    else:
        rospy.set_param(name, value)


def set_planner_params(wandb_config, planning_group, param_handle):
    if wandb_config.moveit_planner == 'rrtconnect':
        planner_name = "RRTConnectkConfigDefault"
        planning_time_per_subgoal = False
    elif wandb_config.moveit_planner == 'rrtstar':
        planner_name = "RRTStarkConfigDefault" if wandb_config.env == "pr2" else "RRTstarkConfigDefault"
        # rrt* always usses up full planning time, so interpret as a max time per subgoal
        planning_time_per_subgoal = True
    else:
        raise NotImplementedError(wandb_config.moveit_planner)
    planner_id = f"{planning_group}[{planner_name}]"

    # see http://docs.ros.org/en/kinetic/api/moveit_tutorials/html/doc/ompl_interface/ompl_interface_tutorial.html
    planner_params = {}
    # fraction of the state space
    planner_params["maximum_waypoint_distance"] = wandb_config.moveit_max_waypoint_distance
    planner_params["longest_valid_segment_fraction"] = 0.05
    planner_params["range"] = wandb_config.moveit_range
    planner_params["default_workspace_bounds"] = 20

    for k, v in planner_params.copy().items():
        # set value to -1 to remove from parameter server
        set_or_delete_param(f"{param_handle}/ompl/{k}", v)
        set_or_delete_param(f"{param_handle}/{planner_id}/{k}", v)
        set_or_delete_param(f"{param_handle}/planner_configs/{planner_name}/{k}", v)
        if v == -1:
            del planner_params[k]

    # rospy.set_param(f"{param_handle}/default_workspace_bounds", 20)
    # rospy.set_param(f"{param_handle}/{planner_id}/planner_configs/max_sampling_attempts", 100)
    return planner_params, planner_id, planning_time_per_subgoal


def main():
    main_path = Path(__file__).parent.absolute()
    run, wandb_config = setup_config_wandb(main_path, sync_tensorboard=False, allow_init=True, no_ckpt_endig=True, framework='ray')
    assert wandb_config['algo'] == 'moveit', wandb_config['algo']

    launch_ros(main_path=main_path, config=wandb_config, task=wandb_config.eval_tasks[0], algo=wandb_config['algo'], pure_analytical="no")

    # TODO: could we plan over all subgoals at once or set the planning time for the all subgoals in total (e.g. just subtract what's been used up already)?
    #   -> kind of doing it now. Though the ideal would be if the planner could itself decide how to split the time across the subgoals

    goal_frame_ids = {'hsr': 'odom',
                      'pr2': 'odom_combined'}
    base_frame_links = {'pr2': 'base_box_link',
                        'hsr': 'base_footprint'}
    planning_groups = {'pr2': "pr2_base_wrist",
                       'hsr': 'whole_body'}

    goal_frame_id = goal_frame_ids[wandb_config.env]
    base_frame_link = base_frame_links[wandb_config.env]
    planning_group = planning_groups[wandb_config.env]

    assert wandb_config.env in ['pr2', 'hsr'], "have to change the values above and change the robot joints in the urdf"
    # assert planning_time_per_subgoal or wandb_config.moveit_num_planning_attempts == 1, "Not sure if multiple attempts will work well with defining the planning time as total for all subgoals? I.e. will it just use up all time on the first subgoal?"

    # need a node to listen to some stuff for the task envs
    rospy.init_node('kinematic_feasibility_py', anonymous=False)

    # nodehandle from which to read the motion planning parameters
    param_handle = "/move_group"
    # first initialised node sets the namespace for all others, so initialise once here before the robot_env node
    moveit_planner = MoveitPlanner("/moveit_planner")

    planner_params, planner_id, planning_time_per_subgoal = set_planner_params(wandb_config, planning_group, param_handle)

    world_types = ["world"] if (wandb_config.world_type == "world") else wandb_config.eval_execs
    for world_type in world_types:
        assert world_type == 'sim', f"{world_type} not supported. Always run with gazebo to spawn the objects into planning scene"
        for task in wandb_config.eval_tasks:
            evaluate_moveit_planning(wandb_config, task=task, world_type=world_type,
                                     param_handle=param_handle,
                                     goal_frame_id=goal_frame_id,
                                     base_frame_link=base_frame_link,
                                     planning_group=planning_group,
                                     planner_id=planner_id,
                                     planning_time_per_subgoal=planning_time_per_subgoal,
                                     planner_params=planner_params)

    print("All done")


if __name__ == '__main__':
    main()
