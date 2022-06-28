from typing import Callable, Union, List, Set, Tuple, Optional, Dict

import numpy as np
import rospy
from PIL import Image, ImageDraw
from gazebo_msgs.msg import ContactsState
from gym import spaces, Env
from gym.utils import seeding
from nav_msgs.msg import OccupancyGrid
from pybindings import RobotObs, multiply_tfs, EEObs

from modulation.envs.eeplanner import EEPlanner, TIME_STEP_TRAIN
from modulation.envs.env_utils import quaternion_to_yaw, yaw_to_quaternion, quaternion_to_list, IDENTITY_TF, SMALL_NUMBER, calc_euclidean_tf_dist, calc_rot_dist
from modulation.envs.map import Map, DummyMap
from modulation.envs.robotenv import RobotEnv, get_ranges, unscale_action, scale_action


class CollisionCB:
    def __init__(self, gazebo_object_to_ignore):
        self.collision_names = None
        self.last_collision = 0.0
        self.gazebo_object_to_ignore = gazebo_object_to_ignore

    def get_last_collisions(self, time_period: float):
        return self.collision_names if (rospy.get_time() - self.last_collision <= time_period) else None

    def cb(self, msg):
        # only process the first, as each message is just for one link, looking at all won't add any information?
        for state in msg.states[:1]:
            collision_names = (state.collision1_name, state.collision2_name)
            for obj_names in self.gazebo_object_to_ignore:
                # handle both cases where obj_names is either a set of one or two objects
                if len(obj_names.intersection(collision_names)) == len(obj_names):
                    break
            else:
                # did not encouter a break in the for loop above, i.e. we cannot ignore this collision
                print(f"Collision: {collision_names}")
                self.collision_names = collision_names
                self.last_collision = msg.header.stamp.to_sec()


def calc_jumps(joint_values: list, prev_joint_values: list, max_joint_velocities: np.ndarray, dt: float, no_limit_joints: np.ndarray) -> Tuple[List, List]:
    dt = max(dt, 0.01)
    joint_diffs = np.array(joint_values) - np.array(prev_joint_values)
    # don't count jumps for jumps like the pr2 wrist which can turn from -2pi to 2pi
    joint_diffs[no_limit_joints] = np.sign(joint_diffs[no_limit_joints]) * np.minimum(abs(joint_diffs[no_limit_joints]), abs(abs(joint_diffs[no_limit_joints]) - 2 * np.pi))
    joint_diffs_abs = np.abs(joint_diffs)

    max_vel_dt = dt * max_joint_velocities
    vel_normed_deviation = joint_diffs_abs / (max_vel_dt + SMALL_NUMBER)

    return joint_diffs, vel_normed_deviation


def draw_footprint(map: Map, robot_base_size) -> np.ndarray:
    img = Image.fromarray(np.zeros(map.output_size[:2])).convert('1')
    draw = ImageDraw.Draw(img)
    robot_size_pixel = np.array(robot_base_size) / map.local_map_resolution
    if len(robot_size_pixel) == 1:
        # circle and robot_size_pixel is the diameter. draw.ellipse() expects a bounding box
        radius = np.ceil(robot_size_pixel[0] / 2)
        x = img.size[0] / 2
        y = img.size[1] / 2
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=1)
    else:
        # rectangle
        x = np.array([(img.size[0] - robot_size_pixel[0]) / 2, (img.size[0] + robot_size_pixel[0]) / 2])
        y = np.array([(img.size[1] - robot_size_pixel[1]) / 2, (img.size[1] + robot_size_pixel[1]) / 2])

        draw.rectangle(list(zip(x, y)), fill=1)
    return np.asarray(img)


class PrevAction:
    def __init__(self, dt, action_dim):
        self.action_dim = action_dim
        # average over the same time as during training
        self.max_size = int(TIME_STEP_TRAIN / dt)
        assert self.max_size >= 1, self.max_size
        self.reset()

    def reset(self):
        self._queue = np.zeros((self.max_size, self.action_dim))

    @property
    def previous_action(self):
        return self._queue.mean(axis=0)

    def put(self, action):
        self._queue = np.append(self._queue[1:], np.array(action)[np.newaxis], 0)


class CombinedEnv(Env):
    metadata = {'render.modes': []}

    def __getattr__(self, name):
        return getattr(self._robot, name)

    def __init__(self,
                 robot_env: RobotEnv,
                 ik_fail_thresh: int,
                 learn_vel_norm_penalty: float,
                 use_map_obs: bool,
                 global_map_resolution: float,
                 local_map_resolution: float,
                 overlay_plan: bool,
                 concat_plan: bool,
                 concat_prev_action: bool,
                 collision_penalty: float,
                 learn_joint_values):
        self.np_random, _ = seeding.np_random(None)
        self._robot = robot_env
        self._ee_planner: EEPlanner = None
        self.plan_horizon_meter = 1.5

        self._ik_fail_thresh = ik_fail_thresh
        self._learn_vel_norm_penalty = learn_vel_norm_penalty
        self._collision_penalty = collision_penalty

        self._use_map_obs = use_map_obs
        self.global_map_resolution = global_map_resolution
        self.local_map_resolution = local_map_resolution
        # each task wrapper should set a map at initialisation if needed, placeholder with no computational costs for when not needed
        self.map = None
        self.set_map(DummyMap(**self.get_map_config()))

        self.learn_joint_values = learn_joint_values
        self.num_learned_joints = len(self._robot.get_joint_minima(self.robot_config['ik_joint_model_group_name'])) if learn_joint_values else 0

        self._overlay_plan = overlay_plan
        self._concat_plan = concat_plan
        self._concat_prev_action = concat_prev_action
        self.action_names, self._min_actions, self._max_actions = get_ranges(drive=self._robot.drive, learn_vel_norm=self.learn_vel_norm, learn_torso=self._robot.learn_torso,
                                                                             num_learned_joints=self.num_learned_joints)
        print(f"Actions to learn: {self.action_names}")
        self.action_dim = len(self._min_actions)
        self._prev_action = PrevAction(action_dim=self.action_dim, dt=TIME_STEP_TRAIN if self._robot.is_analytical_world() else 0.5 * self._robot.rate.sleep_dur.to_sec())

        # NOTE: env does all the action scaling according to _min_actions and _max_actions
        # and just expects actions in the range [-1, 1] from the agent
        self.action_space = spaces.Box(low=np.array(self.action_dim * [-1.0]),
                                       high=np.array(self.action_dim * [1.0]),
                                       shape=[self.action_dim])
        self.reward_range = (-10, 0)

        # somehow not showing the first floormap otherwise
        if self.vis_env:
            self.map.publish_floorplan_rviz()
        self.occ_grid_pub = rospy.Publisher('rl_map_local', OccupancyGrid, queue_size=5)

        # robot base outline as additional map input
        self._add_robot_overlay = False
        self._base_overlay = draw_footprint(self.map, self._robot.robot_base_size)

        self.nr_kin_failures = 0
        self.nr_base_collisions = 0
        self._marker_counter = 0

        self._gazebo_object_to_ignore = set([])
        self._collision_cb = None
        self._sub = None

    def set_gazebo_object_to_ignore(self, object_names: List[Set[str]]):
        """
        Collisions to ignore are a list of sets of either a single object or two objects, if the first, all collisions with that object are ignored, if the latter only that exact collision pair is ignored.
        """
        for obj in object_names:
            assert isinstance(obj, set), set
        self._gazebo_object_to_ignore = tuple(object_names)
        if self._collision_cb is not None:
            self._collision_cb._gazebo_object_to_ignore = self._gazebo_object_to_ignore

    @property
    def learn_vel_norm(self):
        return (self._learn_vel_norm_penalty != -1)

    def get_inflation_radius(self) -> float:
        return max(self._robot.robot_config['inflation_radius_ee'], max(self._robot.robot_base_size) / 2)

    def _local_map_in_collision(self, local_map: np.ndarray) -> bool:
        assert len(local_map.shape) == 2
        return (self._base_overlay * local_map).sum() > 0

    def check_collision(self, local_map):
        if self._robot.get_world() == "gazebo":
            return (self._collision_cb.get_last_collisions(time_period=self._robot.rate.sleep_dur.to_sec()) is not None)
        else:
            return self._local_map_in_collision(local_map[:, :, 0])

    @property
    def observation_space(self):
        ee_obs = EEObs(IDENTITY_TF, IDENTITY_TF, IDENTITY_TF, IDENTITY_TF, 0.0, False)
        local_map = self.get_local_map(IDENTITY_TF, use_ground_truth=True)
        o = self._to_agent_obs(self._robot.get_robot_obs(), ee_obs, [IDENTITY_TF], local_map=local_map)
        if isinstance(o, (list, np.ndarray)):
            return spaces.Box(low=-100, high=100, shape=[len(o)])
        elif isinstance(o, tuple):
            assert len(o) == 2
            return spaces.Tuple([spaces.Box(low=-100, high=100, shape=[len(o[0])]),
                                 # wrong value while it's a dummy map, so don't auto detect
                                 spaces.Box(low=-1.1, high=10.1, shape=list(self.map.output_size) + [1 + self._overlay_plan * 5], dtype=np.float)])
        else:
            raise ValueError(o)

    def get_map_config(self) -> Dict:
        return {'global_map_resolution': self.global_map_resolution,
                'local_map_resolution': self.local_map_resolution,
                'inflation_radius': self.get_inflation_radius(),
                'world_type': self.get_world(),
                'robot_frame_id': self.robot_config["frame_id"]}

    def set_map(self, map: Map):
        if self.map is not None:
            self.map.clear()
        self.map = map
        self.map.np_random = self.np_random
        # ensure map is completely initialised
        # self.map.map_reset()

    def get_local_map(self, base_tf, ee_plan_meters: Optional = None, use_ground_truth: bool = False):
        if self._overlay_plan:
            ee_plan_rel = np.array(ee_plan_meters).copy()
            for i in range(len(ee_plan_meters)):
                ee_plan_rel[i] = self._robot.world_to_relative_tf(base_tf, list(ee_plan_meters[i]))
        else:
            ee_plan_rel = None
        return self.map.get_local_map(base_tf, ee_plan_rel, use_ground_truth=use_ground_truth)

    def set_ee_planner(self,
                       ee_planner: EEPlanner,
                       robot_obs=None):
        is_analytical = self._robot.is_analytical_world()
        if robot_obs is None:
            robot_obs = self._robot.get_robot_obs()

        self._ee_planner = ee_planner
        ee_obs_train_freq, ee_plan = self._ee_planner.reset(robot_obs=robot_obs,
                                                            is_analytic_env=is_analytical,
                                                            plan_horizon_meter=self.plan_horizon_meter)

        if self.vis_env:
            self.map.publish_floorplan_rviz()
            self._vis_env_rviz(robot_obs, ee_obs_train_freq, robot_info={}, ee_plan=ee_plan, local_map=None, done=False, map_changed=True)
            self._robot.publish_marker(ee_planner.gripper_goal_wrist, 0, "gripper_goal", "cyan", 1.0)

        # returns a full obs, not just the new ee_obs to avoid having to stick it together correctly in other places
        return self._to_agent_obs(robot_obs, ee_obs_train_freq, ee_plan_meters=ee_plan)

    def reset(self,
              initial_joint_distribution: Union[str, list],
              ee_planner_fn: Callable,
              gripper_goal_tip_fn: Union[Callable, list],
              close_gripper: bool = False):
        if self._robot.get_world() == "gazebo":
            # move robot out of frame first, so it won't be in collision when the map spawns
            self._robot.set_base_tf([30.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

            if self._sub is not None:
                self._sub.unregister()
            self._collision_cb = CollisionCB(self._gazebo_object_to_ignore)
            self._sub = rospy.topics.Subscriber("/modulation_bumpers", ContactsState, self._collision_cb.cb, queue_size=1)
        self.map.map_reset()

        # arm controllers (particularly for the pr2) can sometimes be unable to reach to drawn joint values in gazebo, in that case draw new ones
        max_i = 10
        for i in range(max_i):
            try:
                base_tf, initial_joint_values, gripper_goal_tip = self.map.draw_startpose_and_goal(self._robot, initial_joint_distribution, gripper_goal_tip_fn)
                robot_obs = self._robot.reset(initial_base_pose=base_tf,
                                              initial_joint_values=initial_joint_values,
                                              close_gripper=close_gripper)
                if self._robot.get_world() == "gazebo":
                    assert not self.check_collision(local_map=None)
                break
            except Exception as e:
                print(f"Reset {i} failed, trying until {max_i - 1}, exception: {e}")
                assert i < (max_i - 1), f"Reached {i} reset attempts, exception: {e}"

        self.nr_base_collisions = 0
        self.nr_kin_failures = 0
        self._marker_counter = 0
        self._prev_action.reset()

        ee_planner = ee_planner_fn(gripper_goal_tip=gripper_goal_tip)
        return self.set_ee_planner(ee_planner=ee_planner)

    def _vis_env_rviz(self, robot_obs: RobotObs, ee_obs, robot_info: dict, ee_plan, local_map: np.ndarray, done: bool, map_changed: bool):
        is_analytical = self.is_analytical_world()
        rate = 0.1 if is_analytical else 0.03
        ik_fail = robot_info.get('kin_failure', False)
        base_collision = robot_info.get('base_collision', False)
        if ik_fail or base_collision or done or robot_info.get('jumps', False) or (self._marker_counter % int(1.0 / rate) == 0):
            self._robot.publish_marker(robot_obs.gripper_tf_achieved, self._marker_counter, "gripper_tf_achieved", "pink", 0.4)
            self._robot.publish_marker(ee_obs.next_gripper_tf, self._marker_counter, "gripper_plan",
                                       "red" if ik_fail else "green", 1.0 if robot_info.get('jumps', False) else 0.5)
            self._robot.publish_marker(robot_obs.base_tf, self._marker_counter, "base_actual",
                                       "red" if base_collision else "yellow", 0.5)
            # can take like half a millisecond -> during actual execution rather look at the global costmap
            if map_changed and is_analytical:
                self.map.publish_floorplan_rviz()
            self.publish_markers(ee_plan, 9999, "gripper_goal", "orange", 0.5)

        if self._use_map_obs and (local_map is not None):
            self._publish_local_map(local_map, robot_obs)
        if done:
            self.publish_trajectory()
        self._marker_counter += 1

    def ik_reward(self, info: dict) -> float:
        # l2 norm, scaled to be -1 when ik_slack_dist and ik_slack_rot_dist away from exact solution
        scaled_failed_reward = -0.5 * ((info['dist_to_desired'] / 0.1) ** 2 + (info['rot_dist_to_desired'] / 0.05) ** 2)
        if self._robot.ikslack_sol_dist_reward == "ik_fail":
            ik_reward = -(info['kin_failure'] or info['selfcollision'])
            if self.env_name == "hsr":
                ik_reward += scaled_failed_reward
        elif self._robot.ikslack_sol_dist_reward == "l2":
            # l2-norm as penalty
            ik_reward = scaled_failed_reward
            if self.selfcollision_as_failure:
                ik_reward -= info['selfcollision']
        else:
            raise ValueError(self.ikslack_sol_dist_reward)
        return self.ikslack_penalty_multiplier * ik_reward

    def calculate_reward(self, base_actions, orig_actions, vel_norm: float, ee_reward: float, robot_obs: RobotObs, robot_info: dict, last_dt: float) -> float:
        base_actions_orig, prev_actions_orig = (np.array(orig_actions[1:]), self._prev_action.previous_action[1:]) if self.learn_vel_norm else (np.array(orig_actions), self._prev_action.previous_action)

        joint_diffs, vel_normed_deviation = calc_jumps(joint_values=robot_obs.joint_values,
                                                       prev_joint_values=self.prev_robot_obs.joint_values,
                                                       max_joint_velocities=self._robot.max_joint_velocities,
                                                       dt=last_dt,
                                                       no_limit_joints=self.no_limit_joints)
        robot_info['vel_normed_avg_deviation'] = np.mean(vel_normed_deviation)
        robot_info['above_joint_vel_limit'] = np.any(vel_normed_deviation > 1)
        robot_info['above_3x_joint_vel_limit'] = np.any(vel_normed_deviation > 5)
        if self.vis_env and robot_info['above_joint_vel_limit']:
            above_vel_limit = vel_normed_deviation > 1
            print(dict(zip(np.array(self.get_joint_names())[above_vel_limit], np.round(joint_diffs[above_vel_limit], 3))))

        reward = (ee_reward
                  - self._use_map_obs * self._collision_penalty * robot_info['base_collision']
                  + self.ik_reward(info=robot_info)
                  - self.penalty_scaling * np.square(base_actions_orig).sum()
                  - self.acceleration_penalty * np.square(base_actions_orig - prev_actions_orig).sum())

        if self.learn_vel_norm:
            max_vel = self._robot.robot_config['base_vel_rng']
            # normalise penalties for ik and collisions to not allow the agent to just move faster to reduce the number of collisions that occur
            # equivalently can think of this as penalizing collisions at high speeds more than at slow speeds
            # or as normalising the penalties wrt to the distance over which they apply
            reward_normaliser = vel_norm / max_vel
            reward *= reward_normaliser
            reward -= self._learn_vel_norm_penalty * ((max_vel - vel_norm) / max_vel) ** 2
        return reward

    def step(self, action, execute_cmds: bool = True, use_ground_truth_map: bool = False):
        vel_norm, base_actions, joint_value_deltas = self._convert_policy_to_env_actions(action)

        if self.exec_action_clip:
            clip_scaled = self._robot.robot_config["base_vel_rng"] * self.exec_action_clip
            vel_norm = np.clip(vel_norm, -clip_scaled, clip_scaled)
        if self.exec_action_scaling:
            vel_norm = self.exec_action_scaling * vel_norm

        ee_obs, last_dt = self._ee_planner.step(self.prev_robot_obs, vel_norm)
        robot_obs, robot_info = self._robot.step(base_actions, joint_value_deltas=joint_value_deltas, next_desired_gripper_tf=ee_obs.next_gripper_tf, execute_cmds=execute_cmds)
        ee_obs_train_freq, ee_plan = self._ee_planner.generate_obs_step(robot_obs, plan_horizon_meter=self.plan_horizon_meter)

        local_map = self.get_local_map(robot_obs.base_tf, ee_plan, use_ground_truth=use_ground_truth_map)

        robot_info['base_collision'] = self.check_collision(local_map)
        robot_info['ee_done'] = ee_obs_train_freq.done
        # NOTE: depends on the robot_info metrics from above, so should stay in this order
        reward = self.calculate_reward(vel_norm=vel_norm, robot_obs=robot_obs, base_actions=base_actions, orig_actions=action,
                                       ee_reward=ee_obs_train_freq.reward, robot_info=robot_info, last_dt=last_dt)

        self.nr_base_collisions += robot_info['base_collision']
        self.nr_kin_failures += robot_info['kin_failure']
        robot_info['nr_base_collisions'] = self.nr_base_collisions
        robot_info['nr_kin_failures'] = self.nr_kin_failures

        done = (((not robot_info['kin_failure']) and ee_obs_train_freq.done)
                or (robot_info['nr_kin_failures'] >= self._ik_fail_thresh)
                or (robot_info['nr_base_collisions'] >= self._ik_fail_thresh))

        obs = self._to_agent_obs(robot_obs, ee_obs_train_freq, ee_plan, local_map)
        self._robot.prev_robot_obs = robot_obs
        # don't update this before calculate_reward()!
        self._prev_action.put(action)

        # e.g. for dynamic obstacles
        map_changed = self.map.update(dt=TIME_STEP_TRAIN)
        if map_changed:
            self._ee_planner.update_weights(robot_obs)

        if self.vis_env:
            obs_map = obs[1] if self._use_map_obs else None
            self._vis_env_rviz(robot_obs, ee_obs, robot_info, ee_plan, obs_map, done, map_changed)
        return obs, reward, done, robot_info

    def _to_agent_obs(self, robot_obs: RobotObs, ee_obs: EEObs, ee_plan_meters: list, local_map=None) -> list:
        goal_relative_to_base = self._robot.world_to_relative_tf(robot_obs.base_tf, list(ee_plan_meters[-1]))
        # relative transform from goal to gripper: might be important for HSR which doesn't get the ik slack at the goal anymore
        wrist_to_goal_relative_to_base = (np.array(goal_relative_to_base[:3]) - robot_obs.relative_gripper_tf[:3]).tolist()
        # e.g. hsr has a virtual link with limits (0., 0.) included, so + SMALL_NUMBER to ensure we don't have nan's in there
        joint_values_scaled = scale_action(robot_obs.joint_values, self.get_joint_minima(), self.get_joint_maxima() + SMALL_NUMBER).tolist()
        obs_vector = (robot_obs.relative_gripper_tf
                      + ee_obs.next_gripper_tf_rel
                      + ee_obs.ee_velocities_rel
                      + goal_relative_to_base
                      + joint_values_scaled
                      + wrist_to_goal_relative_to_base
                      + [robot_obs.in_selfcollision]
                      )
        if self._concat_plan:
            for i in [5, 10, 20]:
                idx = min(i, len(ee_plan_meters) - 1)
                obs_vector += self._robot.world_to_relative_tf(robot_obs.base_tf, list(ee_plan_meters[idx]))
        if self._concat_prev_action:
            obs_vector += list(self._prev_action.previous_action)
        if not self._use_map_obs:
            return np.array(obs_vector, dtype=np.float32)

        # add the information over which obstacles the ee can pass?
        # local_map_ee_height = self.map.get_local_map(robot_obs.base_tf, ignore_below_height=self._robot.robot_config['z_max'] - PointToPoint2DPlannerWrapper.HEIGHT_INFLATION)
        if local_map is None:
            local_map = self.get_local_map(robot_obs.base_tf, ee_plan_meters)
        if not self._overlay_plan:
            local_map = local_map[:, :, :1]
        # NOTE: map_encoder transposes map to pytorch's CHW convention later on
        if self._add_robot_overlay:
            local_map = np.concatenate([local_map, self._base_overlay[:, :, np.newaxis]], axis=-1)
        return obs_vector, local_map

    def _convert_policy_to_env_actions(self, actions):
        # NOTE: all actions are in range [-1, 1] at this point
        actions = list(actions)
        if self.learn_vel_norm:
            min_learned_vel_norm = 0.01
            # into range [min_vel_norm, robot_env.base_vel_rng]
            vel_norm = unscale_action(actions.pop(0), min_learned_vel_norm, self._robot.robot_config["base_vel_rng"])
        else:
            vel_norm = -1
        if self.learn_joint_values:
            base_actions, joint_value_deltas = actions[:-self.num_learned_joints], actions[-self.num_learned_joints:]
        else:
            base_actions, joint_value_deltas = actions, []
        return vel_norm, base_actions, joint_value_deltas

    def _publish_local_map(self, local_map, robot_obs, add_base_footprint: bool = True):
        origin = np.array(local_map.shape[:2]) * self.map.local_map_resolution / 2
        # for visualization add the ee_plan onto the occupancy map
        summed_map = local_map[:, :, 0].astype(np.float)
        if self._overlay_plan:
            summed_map += 100 * local_map[:, :, 1]
        if add_base_footprint:
            summed_map += 100.0 * self._base_overlay.astype(np.float)
        summed_map = np.minimum(summed_map, 100)

        occ_grid = self.map._build_occgrid_msg(summed_map,
                                               origin=(origin[1], origin[0]),
                                               # q=Quaternion(0.0, 0.0, 0.0, 1.0),
                                               tf=robot_obs.base_tf,
                                               resolution=self.map.local_map_resolution)
        self.occ_grid_pub.publish(occ_grid)

    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s)."""
        self.np_random, strong_seed = seeding.np_random(seed)
        if hasattr(self, "map"):
            self.map.np_random = self.np_random
        return [self._robot.seed(strong_seed)]

    def set_world(self, world_type: str):
        self._robot.set_world(world_type)
        self.map.set_world(world_type)

    def send_base_command(self, action):
        vel_norm, base_actions, joint_value_deltas = self._convert_policy_to_env_actions(action)

        base_actions_scaled = self._velocity_ranges * base_actions
        base_translation_relative, base_rotation_relative = self._calculate_base_command(base_actions_scaled)
        self._robot.send_base_command(base_translation_relative, base_rotation_relative)

    def close(self):
        if self._sub is not None:
            self._sub.unregister()
        self._robot.close()
        self.map.clear()
