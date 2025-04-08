from typing import Optional, List
import copy

import numpy as np
import time
from dataclasses import dataclass, field

from modulation.envs.env_utils import delete_all_markers, SMALL_NUMBER, quaternion_to_yaw
from modulation.envs.tasks import TaskGoal


@dataclass
class PlanningEpisode:
    ee_trajectory_ours: list = field(default_factory=list)
    ee_trajectory: list = field(default_factory=list)
    joint_trajectory: list = field(default_factory=list)
    base_trajectory: list = field(default_factory=list)
    planning_times: list = field(default_factory=list)
    planning_times_first: list = field(default_factory=list)
    successes: list = field(default_factory=list)
    return_codes: list = field(default_factory=list)
    extra_metrics: list = field(default_factory=list)

    def add_subgoal(self,
                    ee_trajectory_ours: list,
                    ee_trajectory: list,
                    joint_trajectory: list,
                    base_trajectory: list,
                    planning_time: float,
                    success: bool,
                    extra_metrics: Optional[dict] = None,
                    planning_time_first: float = 0.0,
                    return_code: str = 'unknown'):
        self.planning_times.append(planning_time)
        self.planning_times_first.append(planning_time_first)
        self.successes.append(success)
        self.extra_metrics.append(extra_metrics)
        self.return_codes.append(return_code)
        if success:
            self.ee_trajectory_ours.extend(ee_trajectory_ours)
            self.ee_trajectory.extend(ee_trajectory)
            self.joint_trajectory.extend(joint_trajectory)
            self.base_trajectory.extend(base_trajectory)


    def publish_ee_path(self, env, max_points: int = 100):
        marker_id = 0
        for ee_traj, color in zip([self.ee_trajectory_ours, self.ee_trajectory, self.base_trajectory], ['green', 'cyan', 'orange']):
            if len(ee_traj):
                for i, pose in enumerate(ee_traj[0:-1:int(np.ceil(len(ee_traj) / max_points))]):
                    env.publish_marker(pose, marker_id + i, "ee_traj", color, 0.5)
                    time.sleep(0.005)
                marker_id += 10000

    @staticmethod
    def delete_markers(topics: list = None, are_marker_array: list = None, frame_id: str = "map"):
        if topics is None:
            topics = []
        if are_marker_array is None:
            are_marker_array = []
        for _ in range(2):
            for topic, is_array in zip(topics + ["/eval_env/gripper_goal_visualizer"], are_marker_array + [False]):
                delete_all_markers(topic, is_array, frame_id=frame_id)

    @property
    def success(self):
        return all(self.successes)

    @property
    def total_planning_time(self):
        return sum(self.planning_times)

    @property
    def total_planning_time_first(self):
        return sum(self.planning_times_first)

    @property
    def ee_path_length_ours(self):
        return calc_path_length(self.ee_trajectory_ours)

    @property
    def ee_path_length(self):
        return calc_path_length(self.ee_trajectory)

    @property
    def base_path_length(self):
        return calc_path_length(self.base_trajectory)

    @property
    def return_code(self):
        # all earlier return codes must have been 'success', otherwise wouldn't have continued to next subgoal
        return self.return_codes[-1]


def summarise_episodes(episodes: List[PlanningEpisode]):
    stats = {
        "success": np.mean([e.success for e in episodes]),
        "total_planning_time": np.nanmean([e.total_planning_time if e.success else np.nan for e in episodes]),
        "total_planning_time_first": np.nanmean([e.total_planning_time_first if e.success else np.nan for e in episodes]),
        "ee_path_length_ours": np.nanmean([e.ee_path_length_ours if e.success else np.nan for e in episodes]),
        "ee_path_length": np.nanmean([e.ee_path_length if e.success else np.nan for e in episodes]),
        "base_path_length": np.nanmean([e.base_path_length if e.success else np.nan for e in episodes]),
    }
    stats["ee_path_length_relative"] = stats["ee_path_length"] / stats["ee_path_length_ours"]

    return_codes = [e.return_code for e in episodes]
    for code in np.unique(return_codes):
        stats[f'share_{code}'] = return_codes.count(code) / len(return_codes)

    return stats


def calc_path_length(trajectory):
    "input is a list of poses [[xyzXYZW, xyzXYZW, ...]] in meters"
    traj = np.array(trajectory)
    assert len(traj.shape) == 2 and traj.shape[1] >= 3, traj.shape
    traj_diff = np.diff(traj[:, :3], axis=0)
    return np.sum(np.linalg.norm(traj_diff, axis=1))


def get_joint_configuration(env, planning_group: str) -> dict:
    default_values = copy.deepcopy(env.robot_config.get("initial_joint_values", {}))

    # start pose
    if env.env_name == "hsr":
        # segfaults trying to get the joint values for the base
        planning_group = "arm"

    jn = env.get_joint_names(planning_group)
    jv = env.get_joint_values(planning_group)
    assert len(jn) == len(jv)
    jv = np.clip(jv, env.get_joint_minima(planning_group) + SMALL_NUMBER, env.get_joint_maxima(planning_group) - SMALL_NUMBER)
    default_values.update(dict(zip(jn, jv)))

    # base tf
    base = env.get_robot_obs().base_tf
    default_values['world_joint/x'] = base[0]
    default_values['world_joint/y'] = base[1]
    default_values['world_joint/theta'] = quaternion_to_yaw(base[3:])

    return default_values


def prepare_next_subgoal(env, goal, planning_group, episode, subgoal_idx: int):
    ee_goal_pose_tip = goal.gripper_goal_tip if isinstance(goal, TaskGoal) else goal
    ee_goal_pose_wrist = env.tip_to_gripper_tf(ee_goal_pose_tip)

    if subgoal_idx == 0:
        # TODO: check if this returns correct values (e.g. for the base links)
        start_configuration = get_joint_configuration(env, planning_group)
        robot_obs = env.get_robot_obs()
    else:
        # start from configuration achieved at last subgoal
        # start_configuration.update(zip(env.get_joint_names(planning_group), episode.joint_trajectory[-1]))
        # base_tf = episode.base_trajectory[-1]
        # # our env does not use the base_x joints to set the position, but rather the world_joint. So ignore passing in the base joints
        # start_configuration_no_base = start_configuration.copy()
        # for k in ['base_x_joint', 'base_y_joint', 'base_theta_joint', 'odom_x', 'odom_y', 'odom_r']:
        #     if k in start_configuration_no_base:
        #         del start_configuration_no_base[k]
        # start_configuration.update({"world_joint/x": base_tf[0],
        #                             "world_joint/y": base_tf[1],
        #                             "world_joint/theta": quaternion_to_yaw(base_tf[3:]),})
        # set_in_world = True
        # env.set_joint_values(start_configuration_no_base, set_in_world)

        robot_obs = env.get_robot_obs()
        robot_obs.base_tf = episode.base_trajectory[-1]
        robot_obs.gripper_tf = env.unwrapped._ee_planner.gripper_goal_wrist
        start_configuration = {}

        ee_planner = goal.ee_fn(gripper_goal_tip=goal.gripper_goal_tip,
                                head_start=goal.head_start,
                                map=env.map,
                                robot_config=env.robot_config,
                                success_thres_dist=goal.success_thres_dist,
                                success_thres_rot=goal.success_thres_rot)
        obs = env.set_ee_planner(ee_planner=ee_planner, robot_obs=robot_obs)

    # NOTE: we can only rely on get_robot_obs() for the first subgoal or after updating the internal robot state!
    # robot_obs = env.get_robot_obs()
    # env.publish_robot_state(robot_obs.base_tf)
    env.publish_marker(robot_obs.gripper_tf, 9999, "start_pose_wrist", "orange", 1.0)
    env.publish_marker(ee_goal_pose_wrist, 10000, "ee_goal_pose_wrist", "cyan", 1.0)

    # reference plan of our approach to compare lengths
    # NOTE: requires to first have set a new ee-planner for all but the first subgoal
    _, ee_plan_ours = env.unwrapped._ee_planner.generate_obs_step(robot_obs, plan_horizon_meter=50)

    return ee_goal_pose_tip, ee_goal_pose_wrist, ee_plan_ours, start_configuration
