import copy
from typing import Tuple, Optional
import time
import numpy as np
import rospy
from gym import Env
from gym.utils import seeding
import yaml
from dataclasses import dataclass, field
from pybindings import RobotObs, RobotConfig, multiply_tfs  # , gripper_to_tip_goal, tip_to_gripper_goal
from pybindings_robots import RobotPR2, RobotTiago, RobotHSR

from modulation.envs.eeplanner import TIME_STEP_TRAIN
from modulation.envs.env_utils import PROJECT_ROOT, IDENTITY_TF, calc_euclidean_tf_dist, calc_rot_dist, \
    quaternion_to_yaw, SMALL_NUMBER


@dataclass
class RobotTrajectory:
    base_trajectory: list = field(default_factory=list)
    gripper_trajectory: list = field(default_factory=list)
    joint_trajectory: list = field(default_factory=list)

    def __len__(self):
        return len(self.base_trajectory)

    def add(self, robot_obs: RobotObs):
        self.base_trajectory.append(robot_obs.base_tf)
        self.gripper_trajectory.append(robot_obs.gripper_tf)
        self.joint_trajectory.append(robot_obs.joint_values)

    @property
    def joint_base_trajectory(self):
        points = []
        for b, jv in zip(self.base_trajectory, self.joint_trajectory):
            point = [b[0], b[1], quaternion_to_yaw(b[3:])] + list(jv)
            points.append(point)
        return points

    def publish_paths(self, env, max_points: int = 100):
        marker_id = 0
        for ee_traj, color in zip([self.gripper_trajectory, self.base_trajectory], ['green', 'orange']):
            for i, pose in enumerate(ee_traj[0:-1:int(np.ceil(len(ee_traj) / max_points))]):
                env.publish_marker(pose, marker_id, "ee_traj", color, 0.5)
                time.sleep(0.005)
                marker_id += 1


def get_ranges(drive: str, learn_vel_norm: bool, learn_torso: bool, num_learned_joints):
    ks = []
    if learn_vel_norm:
        ks.append('vel_norm')

    if drive == 'diff':
        ks += ['tiago_base_vel', 'tiago_base_angle']
    elif drive == "omni":
        ks += ['base_rot', 'base_x', 'base_y']
    else:
        raise NotImplementedError()

    if learn_torso:
        ks.append('torso')

    if num_learned_joints:
        ks += [f'joint{i}' for i in range(num_learned_joints)]

    n = len(ks)
    min_actions = n * [-1]
    max_actions = n * [1]

    return ks, np.array(min_actions), np.array(max_actions)


def rospy_maybe_initialise(node_name: str, anonymous: bool):
    """Check if there is already an initialised rospy node and initialise it if not"""
    try:
        rospy.get_time()
    except:
        rospy.init_node(node_name, anonymous=anonymous)


def unscale_action(scaled_action, low: float, high: float):
    """
    Rescale the action from [-1, 1] to [low, high]
    (no need for symmetric action space)
    :param scaled_action: Action to un-scale
    """
    return low + (0.5 * (scaled_action + 1.0) * (high - low))


def scale_action(action, low: float, high: float):
    """
    Rescale the action from [low, high] to [-1, 1]
    (no need for symmetric action space)
    :param action: (np.ndarray) Action to scale
    :return: (np.ndarray) Scaled action
    """
    return 2.0 * ((action - low) / (high - low)) - 1.0


class RobotEnv(Env):
    def __init__(self,
                 env: str,
                 node_handle_name: str,
                 penalty_scaling: float,
                 acceleration_penalty: float,
                 time_step_world: float,
                 seed: int,
                 world_type: str,
                 init_controllers: bool,
                 vis_env: bool,
                 transition_noise_base: float,
                 iksolver: str,
                 bioik_center_joints_weight: float,
                 bioik_avoid_joint_limits_weight: float,
                 bioik_regularization_weight: float,
                 bioik_regularization_type: str,
                 ikslack_dist,
                 ikslack_rot_dist: float,
                 ikslack_sol_dist_reward: str,
                 ikslack_penalty_multiplier: float,
                 selfcollision_as_failure: bool,
                 learn_torso: bool,
                 exec_action_clip: Optional[float],
                 exec_action_scaling: Optional[float],
                 exec_acceleration_clip: Optional[float],
                 execute_style: str,
                 perception_style: str,
                 fake_gazebo: bool = False):
        conf_path = PROJECT_ROOT / "gazebo_world" / env / "robot_config.yaml"
        assert conf_path.exists(), conf_path
        with open(conf_path.absolute()) as f:
            self.robot_config = yaml.safe_load(f)

        if not fake_gazebo and (env != 'hsr'):
            solver_param = "/robot_description_kinematics/" + self.robot_config["joint_model_group_name"] + "/kinematics_solver"
            assert (rospy.get_param(solver_param) == "bio_ik/BioIKKinematicsPlugin") == (iksolver == "bioik"), (iksolver, rospy.get_param(solver_param))

        if env == 'pr2':
            self.robot_config['ik_joint_model_group_name'] = 'right_arm' if learn_torso else self.robot_config['joint_model_group_name']
            robot_fn = RobotPR2
            self.drive = "omni"
        elif env == 'tiago':
            self.robot_config['ik_joint_model_group_name'] = 'arm' if learn_torso else self.robot_config['joint_model_group_name']
            robot_fn = RobotTiago
            self.drive = "diff"
        elif env == 'hsr':
            self.robot_config['ik_joint_model_group_name'] = 'arm_no_torso' if learn_torso else self.robot_config['joint_model_group_name']
            robot_fn = RobotHSR
            self.drive = "omni"
            rospy.set_param("/robot_description_planning/joint_limits/wrist_ft_sensor_frame_joint/has_velocity_limits", False)
            rospy.set_param("/robot_description_planning/joint_limits/wrist_ft_sensor_frame_joint/max_velocity", 1000)
        else:
            raise ValueError('Unknown env')
        self._robot_cpp = robot_fn(seed,
                                   node_handle_name,
                                   init_controllers,
                                   world_type,
                                   self.get_cpp_robot_config(),
                                   bioik_center_joints_weight,
                                   bioik_avoid_joint_limits_weight,
                                   bioik_regularization_weight,
                                   bioik_regularization_type)
        if env == "pr2":
            no_lim_joints = ['r_wrist_roll_joint', 'r_forearm_roll_joint']
        else:
            no_lim_joints = [None]
        self.no_limit_joints = np.any(np.stack([np.array(self.get_joint_names()) == jn for jn in no_lim_joints], 0), axis=0)
        self.np_random, _ = seeding.np_random(None)

        rospy_maybe_initialise(node_handle_name + '_py', anonymous=True)

        self.env_name = env
        self.learn_torso = learn_torso
        self._transition_noise_base = transition_noise_base
        self.penalty_scaling = penalty_scaling
        self.acceleration_penalty = acceleration_penalty
        self.vis_env = vis_env
        self.rate = rospy.Rate(1. / time_step_world)
        self.max_joint_velocities = np.array([rospy.get_param(f"/robot_description_planning/joint_limits/{joint}/max_velocity") for joint in self.get_joint_names()])
        # helper for the planning baselines to pretend we use the robot in gazebo while not actually moving the robot
        self.fake_gazebo = fake_gazebo
        if self.drive == "omni":
            self._velocity_ranges = np.array([self.robot_config['base_rot_rng'], self.robot_config['base_vel_rng'], self.robot_config['base_vel_rng']])
        elif self.drive == "diff":
            self._velocity_ranges = np.array([self.robot_config['base_rot_rng'], self.robot_config['base_vel_rng']])
        else:
            raise NotImplementedError(env)
        if self.learn_torso:
            self._velocity_ranges = np.append(self._velocity_ranges, self.robot_config['torso_vel_rng'])

        # TODO: IF NOT RECTANGULAR: CHECK IF (H, W) IS THE CORRECT WAY AROUND
        self.robot_base_size = self.robot_config["robot_base_size_meters"]

        self.ikslack_sol_dist_reward = ikslack_sol_dist_reward
        self.ikslack_dist = ikslack_dist
        self.ikslack_rot_dist = ikslack_rot_dist
        self.ikslack_penalty_multiplier = ikslack_penalty_multiplier
        self.selfcollision_as_failure = selfcollision_as_failure

        self.prev_robot_obs = None

        self.exec_action_clip = exec_action_clip
        self.exec_action_scaling = exec_action_scaling
        self.exec_acceleration_clip = exec_acceleration_clip
        self.execute_style = execute_style
        self.perception_style = perception_style

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError("attempted to get missing private attribute '{}'".format(name))
        return getattr(self._robot_cpp, name)

    def reset(self,
              initial_base_pose: Optional[list],
              initial_joint_values: Optional[list],
              close_gripper: bool = False) -> RobotObs:
        if initial_base_pose is None:
            initial_base_pose = IDENTITY_TF

        if initial_joint_values is None:
            initial_joint_values = self.draw_joint_values("rnd")

        # TODO: maybe always pass around dicts for joint values?
        jn = self.get_joint_names()
        assert len(jn) == len(initial_joint_values)
        initial_joint_values = dict(zip(jn, initial_joint_values))

        initial_joint_values['world_joint/x'] = initial_base_pose[0]
        initial_joint_values['world_joint/y'] = initial_base_pose[1]
        yaw = initial_base_pose[-1] if len(initial_base_pose) == 6 else quaternion_to_yaw(initial_base_pose[3:])
        initial_joint_values['world_joint/theta'] = yaw

        robot_obs = self._robot_cpp.reset(initial_joint_values, close_gripper)
        self.prev_robot_obs = robot_obs
        self._prev_base_actions_scaled = 0.0

        self.trajectory = RobotTrajectory()
        self.trajectory.add(robot_obs)

        if self.vis_env:
            self._robot_cpp.publish_robot_state()
        return robot_obs

    def apply_action_exec(self, base_actions_scaled, prev_base_actions_scaled):
        if self.exec_action_clip:
            clip_scaled = self._velocity_ranges[:len(base_actions_scaled)] * self.exec_action_clip
            base_actions_scaled = np.clip(base_actions_scaled, -clip_scaled, clip_scaled)
        if self.exec_action_scaling:
            base_actions_scaled = self.exec_action_scaling * base_actions_scaled
        if self.exec_acceleration_clip:
            clip_scaled = self._velocity_ranges[:len(base_actions_scaled)] * self.exec_acceleration_clip
            base_actions_scaled = np.clip(base_actions_scaled, prev_base_actions_scaled - clip_scaled, prev_base_actions_scaled + clip_scaled)
        return base_actions_scaled

    def step(self, base_actions, joint_value_deltas, next_desired_gripper_tf, execute_cmds: bool = True) -> Tuple[RobotObs, float, dict]:
        # rate.sleep has a massive overhead in python
        # if (not self.is_analytical_world()) and execute_cmds:
        #     self.rate.sleep()
        dt_exec = TIME_STEP_TRAIN if self.is_analytical_world() else self.rate.sleep_dur.to_sec()

        base_actions_scaled = self._velocity_ranges * base_actions
        if self.learn_torso:
            delta_torso = base_actions_scaled[-1]
            base_actions_scaled = np.delete(base_actions_scaled, -1)
            velocity_ranges_base = self._velocity_ranges[:-1]
        else:
            delta_torso = 0.
            velocity_ranges_base = self._velocity_ranges

        if len(joint_value_deltas):
            # self.max_joint_velocities, prev_robot_obs.joint_values include the torso
            m = dt_exec * self.max_joint_velocities[1:]
            prev = self.prev_robot_obs.joint_values[1:]
            joint_value_deltas = unscale_action(np.array(joint_value_deltas), low=-m, high=m)
            joint_values_action = np.clip(prev + joint_value_deltas,
                                          self.get_joint_minima(self.robot_config['ik_joint_model_group_name']),
                                          self.get_joint_maxima(self.robot_config['ik_joint_model_group_name']))
        else:
            joint_values_action = []

        if self._transition_noise_base:
            base_actions_scaled += self.np_random.normal(0.0, self._transition_noise_base * velocity_ranges_base / TIME_STEP_TRAIN, size=len(base_actions_scaled))

        base_actions_scaled = self.apply_action_exec(base_actions_scaled=base_actions_scaled, prev_base_actions_scaled=self._prev_base_actions_scaled)

        base_translation_relative, base_rotation_relative = self._calculate_base_command(base_actions_scaled)
        self._prev_base_actions_scaled = base_actions_scaled

        robot_obs: RobotObs = self._robot_cpp.step(base_translation_relative, base_rotation_relative,
                                                   next_desired_gripper_tf, self.prev_robot_obs, dt_exec, execute_cmds,
                                                   self.learn_torso, delta_torso, self.execute_style,
                                                   self.perception_style, joint_values_action)
        self.trajectory.add(robot_obs)

        npoints = 1 if self.is_analytical_world() else 3
        if self.vis_env and len(self.trajectory) % npoints == 0:
            self._robot_cpp.publish_robot_state()

        robot_info = {}
        robot_info['gripper_tf_desired'] = next_desired_gripper_tf
        robot_info['gripper_tf_achieved'] = robot_obs.gripper_tf_achieved
        robot_info['dist_to_desired'] = calc_euclidean_tf_dist(next_desired_gripper_tf, robot_obs.gripper_tf)
        robot_info['rot_dist_to_desired'] = calc_rot_dist(next_desired_gripper_tf, robot_obs.gripper_tf)
        if not self.is_analytical_world():
            # TODO: should this be distance to current robot_obs or previous robot_obs, as we've just sent the command?
            robot_info['gripper_dist_to_achieved'] = calc_euclidean_tf_dist(robot_obs.gripper_tf, robot_obs.gripper_tf_achieved)
            robot_info['gripper_rot_dist_to_achieved'] = calc_rot_dist(robot_obs.gripper_tf, robot_obs.gripper_tf_achieved)

        if not self.selfcollision_as_failure and robot_obs.in_selfcollision:
            robot_obs.ik_fail = False
        slack_failure = (robot_info['dist_to_desired'] > self.ikslack_dist + SMALL_NUMBER) or (robot_info['rot_dist_to_desired'] > self.ikslack_rot_dist + SMALL_NUMBER)
        robot_obs.ik_fail |= slack_failure
        robot_info['kin_failure'] = robot_obs.ik_fail
        robot_info['selfcollision'] = robot_obs.in_selfcollision

        # NOTE: incosistency, I set this at the end of combined_env.step() instead, so that the metrics there can use it as well
        # self.prev_robot_obs = robot_obs
        return robot_obs, robot_info

    def _calculate_base_command(self, base_actions: list):
        if self.drive == "omni":
            assert len(base_actions) == 3, base_actions
            base_rotation_relative = base_actions[0]
            base_translation_relative = [base_actions[1], base_actions[2], 0.0]
        elif self.drive == "diff":
            assert len(base_actions) == 2, base_actions
            base_rotation_relative = base_actions[0]
            vel_forward = base_actions[1]
            base_translation_relative = [vel_forward, 0.0, 0.0]
        else:
            raise NotImplementedError(self.drive)

        return base_translation_relative, base_rotation_relative

    def close(self):
        pass

    def open_gripper(self, position: float = 0.08, wait_for_result: bool = True):
        if self.is_analytical_world():
            return True
        else:
            return self._robot_cpp.open_gripper(position, wait_for_result)

    def close_gripper(self, position: float = 0.00, wait_for_result: bool = True):
        if self.is_analytical_world():
            return True
        else:
            return self._robot_cpp.close_gripper(position, wait_for_result)

    def seed(self, seed=None):
        self.np_random, strong_seed = seeding.np_random(seed)
        return [self._robot_cpp.set_rng(strong_seed)]

    def get_joint_names(self, joint_model_group: Optional[str] = None):
        if joint_model_group is None:
            joint_model_group = self.robot_config['joint_model_group_name']
        return self._robot_cpp.get_joint_names(joint_model_group)

    def get_joint_minima(self, joint_model_group: Optional[str] = None):
        if joint_model_group is None:
            joint_model_group = self.robot_config['joint_model_group_name']
        return np.array(self._robot_cpp.get_joint_minima(joint_model_group))

    def get_joint_maxima(self, joint_model_group: Optional[str] = None):
        if joint_model_group is None:
            joint_model_group = self.robot_config['joint_model_group_name']
        return np.array(self._robot_cpp.get_joint_maxima(joint_model_group))

    def get_joint_values_world(self, joint_model_group: Optional[str] = None):
        if joint_model_group is None:
            joint_model_group = self.robot_config['joint_model_group_name']
        return np.array(self._robot_cpp.get_joint_values_world(joint_model_group))

    def get_robot_obs(self, ik_fail: bool = False, in_selfcollision: bool = False):
        # NOTE: calculated velocities can be 0 if prev_robot_obs was already updated. In that case
        if self.prev_robot_obs is None:
            return self._robot_cpp.get_robot_obs(ik_fail, in_selfcollision)
        else:
            return self._robot_cpp.get_robot_obs_with_vel(ik_fail, self.prev_robot_obs, in_selfcollision)

    def get_world(self):
        return "gazebo" if self.fake_gazebo else self._robot_cpp.get_world()

    def publish_marker(self, marker_tf, marker_id: int, namespace: str, color: str, alpha: float,
                       geometry: str = "arrow", marker_scale=(0.1, 0.025, 0.025), frame_id=""):
        if isinstance(color, str):
            self._robot_cpp.publish_marker(marker_tf, marker_id, namespace, color, alpha, geometry, marker_scale, frame_id)
        else:
            self._robot_cpp.publish_marker_rgb(marker_tf, marker_id, namespace, color, alpha, geometry, marker_scale, frame_id)

    def publish_markers(self, markers: list, first_marker_id, *args, **kwargs):
        for i, e in enumerate(markers):
            self.publish_marker(e, first_marker_id + i, *args, **kwargs)
            # time.sleep(0.02)

    @staticmethod
    def world_to_relative_tf(base_tf, ee_transform):
        return multiply_tfs(base_tf, ee_transform, True)

    def draw_joint_values(self, joint_distribution: str = "rnd"):
        if joint_distribution == "rnd":
            z_min = z_max = 0.0
        elif joint_distribution == "restricted_ws":
            z_min = self.robot_config["restricted_ws_z_min"]
            z_max = self.robot_config["restricted_ws_z_max"]
        else:
            raise ValueError(joint_distribution)
        return self._robot_cpp.draw_joint_values(z_min, z_max)

    def get_cpp_robot_config(self) -> RobotConfig:
        return RobotConfig(self.robot_config["name"],
                           self.robot_config["joint_model_group_name"],
                           self.robot_config["ik_joint_model_group_name"],
                           self.robot_config["frame_id"],
                           self.robot_config["global_link_transform"],
                           self.robot_config["tip_to_gripper_offset"],
                           self.robot_config["gripper_to_base_rot_offset"],
                           self.robot_config["base_cmd_topic"],
                           self.robot_config["kinematics_solver_timeout"],
                           self.robot_config.get("initial_joint_values", {}),
                           self.robot_config["torso_joint_name"])
