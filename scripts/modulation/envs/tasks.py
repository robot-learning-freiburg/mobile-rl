from dataclasses import dataclass
from enum import IntEnum
from functools import partial
from typing import Callable, Union, Optional

import numpy as np
import rospy
from geometry_msgs.msg import Pose, Point, PoseStamped, Quaternion
from gym import Wrapper
from std_msgs.msg import Bool

from modulation.envs.combined_env import CombinedEnv
from modulation.envs.eeplanner import PointToPointPlanner, SplinePlanner, LinearPlannerFullPython
from modulation.envs.env_utils import quaternion_to_yaw, yaw_to_quaternion, PROJECT_ROOT, IDENTITY_TF, pose_to_list, \
    list_to_pose
from modulation.envs.map import Map, SceneMap, SimpleObstacleMap
from modulation.envs.simulator_api import SpawnObject, WorldObjects, DynObstacle, sample_circle_goal

GOAL_TOPIC = "mobilerl_interactive_marker"
GOALPOINTER_ENABLED_TOPIC = "goalpointer_enabled"
# also defined in move_obstacles.py
MOVE_OBSTACLES_ENABLED_TOPIC = "pause_move_obstacles"


class GripperActions(IntEnum):
    NONE = 0
    OPEN = 1
    GRASP = 2


@dataclass
class TaskGoal:
    gripper_goal_tip: Union[Callable, list]
    end_action: GripperActions
    success_thres_dist: float
    success_thres_rot: float
    ee_fn: Callable
    head_start: float = 0.0
    joint_values: Optional[list] = None
    enable_goal_pointer: bool = True


@dataclass
class UserGoal:
    goal = None
    time = rospy.Time(0)

    def callback(self, msg: PoseStamped):
        self.goal = msg.pose
        self.time = msg.header.stamp


def publish_goal(goal: Pose):
    pub = rospy.Publisher(GOAL_TOPIC, PoseStamped, queue_size=5, latch=True)
    msg = PoseStamped()
    msg.pose = goal
    msg.header.stamp = rospy.Time.now()
    pub.publish(msg)


def publish_goalpointer_enabled(enabled: bool):
    pub = rospy.Publisher(GOALPOINTER_ENABLED_TOPIC, Bool, queue_size=1, latch=True)
    msg = Bool()
    msg.data = enabled
    pub.publish(msg)


def publish_move_obstacles_enabled(enabled: bool):
    pub = rospy.Publisher(MOVE_OBSTACLES_ENABLED_TOPIC, Bool, queue_size=1, latch=True)
    msg = Bool()
    msg.data = enabled
    pub.publish(msg)


def ask_user_goal(next_goal: list):
    goal = UserGoal()
    rospy.topics.Subscriber(GOAL_TOPIC, PoseStamped, goal.callback, queue_size=1)

    print(f"\n### USER INPUT ###")
    print(f"\tnext goal: {np.round(next_goal, 3)}")
    print(f"\tPress a to accept")
    print("\tor enter a new goal with the interactive marker")
    print(f"\tor over the keyboard in world coordinates as [x y z X Y Z W] or [x y z R P Y]")

    i = 0
    while (i == 0) or (goal.goal is None) or (goal.time and (rospy.get_rostime() - goal.time).to_sec() > 30):
        user_input = input("Enter goal or 'a' to accept:")
        i += 1
        if (user_input in ['a', ""]):
            goal.goal = list_to_pose(next_goal)
            # publish_goal(next_goal)
            # time.sleep(0.1)
            goal.time = rospy.get_rostime()
            break
        else:
            try:
                user_goal = [float(g) for g in user_input.split(" ")]
                goal.goal = list_to_pose(user_goal)
                rospy.loginfo(f"Received {user_goal}")
            except Exception as e:
                print(e)
                rospy.loginfo(f"Invalid input: goal needs to have 6 or 7 values separated by white space. Received {user_input}")

    # publish the goal in either case, so the camera_goalpointer node will receive the goal
    print(f"Ask user goal publishes {goal.goal}")
    publish_goal(goal.goal)
    return goal.goal


def armarker_pose_to_listtf(msg: Pose):
    """Correct for different orientation conventions; Assume marker is always oriented with x-axis to the right and y-axis pointing to the ground"""
    gripper_tf = pose_to_list(msg)
    yaw = quaternion_to_yaw(gripper_tf[3:])
    return gripper_tf[:3] + [0, 0, np.pi / 2. + yaw]


class BaseTask(Wrapper):
    @staticmethod
    def taskname() -> str:
        raise NotImplementedError()

    @property
    def loggingname(self):
        name = (self.taskname()
                + (f"{self.map.obstacle_configuration}" if hasattr(self.map, 'obstacle_configuration') and (self.map.obstacle_configuration != 'none') else "")
                + (f"{self.map._obstacle_spacing}" if hasattr(self.map, '_obstacle_spacing') and (self.map._obstacle_spacing is not None) else "")
                + (f"fwdO" if self.use_fwd_orientation else "")
                + (f"_{self.get_world()}" if self.get_world() != "sim" else ""))
        return name

    @staticmethod
    def requires_simulator() -> bool:
        """
        Whether this task cannot be run in the analytical environment alone and needs e.g. the Gazebo simulator
        available (e.g. to spawn objects, deduct goals, ...)
        """
        raise NotImplementedError()

    def __getattr__(self, name):
        return getattr(self.env, name)

    def __init__(self,
                 env: CombinedEnv,
                 initial_joint_distribution: str,
                 map: Map,
                 success_thres_dist: float = 0.025,
                 success_thres_rot: float = 0.05,
                 close_gripper_at_start: bool = True,
                 use_fwd_orientation: bool = False,
                 eval: bool = False,
                 ):
        super(BaseTask, self).__init__(env=env)
        self._initial_joint_distribution = initial_joint_distribution
        self.env.set_map(map)
        self._success_thres_dist = success_thres_dist
        self._success_thres_rot = success_thres_rot
        self._close_gripper_at_start = close_gripper_at_start
        self.use_fwd_orientation = use_fwd_orientation
        self.eval = eval

        self._goal_sub = None
        self._goal_sub2 = None

    def draw_goal(self) -> TaskGoal:
        raise NotImplementedError()

    def goal_callback(self, msg: PoseStamped):
        """callback to be executed when a new goal is passed in during execution. Only in gazebo and real world envs"""
        gripper_tf = armarker_pose_to_listtf(msg.pose)
        self.env.unwrapped._ee_planner.set_goal(gripper_tf)

    def goal_callback2(self, msg: PoseStamped):
        return

    def reset(self, task_goal: TaskGoal = None):
        if task_goal is None:
            task_goal: TaskGoal = self.draw_goal()
        if self.env.get_world() == 'world':
            publish_goalpointer_enabled(task_goal.enable_goal_pointer)
        if self.env.get_world() == 'gazebo':
            publish_move_obstacles_enabled(False)
        if self._goal_sub is not None:
            self._goal_sub.unregister()
            self._goal_sub2.unregister()
        ee_planner_fn = partial(task_goal.ee_fn,
                                head_start=task_goal.head_start,
                                map=self.env.map,
                                robot_config=self.env.robot_config,
                                success_thres_dist=task_goal.success_thres_dist,
                                success_thres_rot=task_goal.success_thres_rot)
        if self.env.get_world() == "world":
            print(f"\n### USER INPUT ###")
            input("Press enter to do next reset")

        obs = self.env.reset(initial_joint_distribution=self._initial_joint_distribution if task_goal.joint_values is None else task_goal.joint_values,
                             ee_planner_fn=ee_planner_fn,
                             gripper_goal_tip_fn=task_goal.gripper_goal_tip,
                             close_gripper=self._close_gripper_at_start)
        if not self.env.is_analytical_world():
            # cb = UpdateGoalCallback(ee_planner=self.env.unwrapped._ee_planner)
            if self.env.get_world() == "world":
                ask_user_goal(next_goal=self.env.unwrapped._ee_planner.gripper_goal_tip)
                # self.env.publish_marker(self.env.unwrapped._ee_planner.gripper_goal_tip, 0, "gripper_goal", "cyan", 1.0)
                # self.env.publish_marker(self.env.unwrapped._ee_planner.gripper_goal_wrist, 1, "gripper_goal", "pink", 1.0)

                # reset eeplanner start time to not have a large first movement after waiting for the user input
                self.env.unwrapped._ee_planner._rostime_at_start = rospy.get_time()

            self._goal_sub = rospy.topics.Subscriber(GOAL_TOPIC, PoseStamped, self.goal_callback, queue_size=1)
            self._goal_sub2 = rospy.topics.Subscriber(f"{GOAL_TOPIC}2", PoseStamped, self.goal_callback2, queue_size=1)

            # elif self.env.get_world() == "gazebo":
            #     publish_goal(list_to_pose(self.env.unwrapped._ee_planner.gripper_goal_tip))

            publish_move_obstacles_enabled(True)
        return obs

    def step(self, action, **kwargs):
        obs, reward, done, info = self.env.step(action=action, **kwargs)
        if done and self._goal_sub is not None:
            self._goal_sub.unregister()
            self._goal_sub2.unregister()
        return obs, reward, done, info

    def clear(self):
        if self._goal_sub is not None:
            self._goal_sub.unregister()
            self._goal_sub2.unregister()
        self.env.close()


class RndStartRndGoalsTask(BaseTask):
    @staticmethod
    def taskname() -> str:
        return "rndStartRndGoal"

    @staticmethod
    def requires_simulator() -> bool:
        return False

    def __init__(self, env: CombinedEnv, use_fwd_orientation: bool, eval: bool, goal_dist_rng=(1, 5), goal_height_rng=None):
        map = Map(**env.get_map_config(),
                  update_from_global_costmap=False)
        super(RndStartRndGoalsTask, self).__init__(env=env,
                                                   initial_joint_distribution='rnd',
                                                   map=map,
                                                   use_fwd_orientation=use_fwd_orientation,
                                                   eval=eval)

        if goal_height_rng is None:
            goal_height_rng = (env.robot_config["z_min"], env.robot_config["z_max"])
        assert len(goal_dist_rng) == len(goal_height_rng) == 2
        self._goal_dist_rng = goal_dist_rng
        self._goal_height_rng = goal_height_rng

    def draw_goal(self):
        def goal_fn(base_tf: list):
            # assumes we are currently at the origin!
            gripper_goal_wrist = sample_circle_goal(base_tf, self._goal_dist_rng, self._goal_height_rng, self.map._map_W_meter, self.map._map_H_meter, self.np_random)
            # first transform to tip here so that we are sampling in the wrist frame: easier to prevent impossible goals
            return self.env.gripper_to_tip_tf(gripper_goal_wrist)

        return TaskGoal(gripper_goal_tip=goal_fn,
                        end_action=GripperActions.NONE,
                        success_thres_dist=self._success_thres_dist,
                        success_thres_rot=self._success_thres_rot,
                        ee_fn=LinearPlannerFullPython)


class RestrictedWsTask(RndStartRndGoalsTask):
    @staticmethod
    def taskname() -> str:
        return "restrictedWs"

    @staticmethod
    def requires_simulator() -> bool:
        return False

    def __init__(self, env: CombinedEnv, use_fwd_orientation: bool, eval: bool):
        super(RestrictedWsTask, self).__init__(env=env,
                                               goal_height_rng=(env.robot_config["restricted_ws_z_min"],
                                                                env.robot_config["restricted_ws_z_max"]),
                                               use_fwd_orientation=use_fwd_orientation,
                                               eval=eval)


class SplineTask(RndStartRndGoalsTask):
    @staticmethod
    def taskname() -> str:
        return "spline"

    @staticmethod
    def requires_simulator() -> bool:
        return False

    def __init__(self, env: CombinedEnv, use_fwd_orientation: bool, eval: bool):
        super(SplineTask, self).__init__(env=env,
                                         goal_height_rng=(env.robot_config["restricted_ws_z_min"],
                                                          env.robot_config["restricted_ws_z_max"]),
                                         goal_dist_rng=(1, 3),
                                         use_fwd_orientation=use_fwd_orientation,
                                         eval=eval)

    def draw_goal(self):
        def goal_fn(base_tf: list):
            # assert base_tf == IDENTITY_TF, "Atm always assuming we start from the origin"
            return waypoints_wrist[-1]

        num_waypoints = 5

        # NOTE: hack to not have to change the API. Better: change so that we can sample them in goal_fn() depending on base_tf
        base_tf = IDENTITY_TF
        waypoints_wrist = []
        last_start = base_tf
        for _ in range(num_waypoints):
            gripper_goal_tip = sample_circle_goal(last_start, self._goal_dist_rng, self._goal_height_rng, self.map._map_W_meter - 2.5, self.map._map_H_meter - 2.5, self.np_random)
            gripper_goal_wrist = self.env.gripper_to_tip_tf(gripper_goal_tip)
            waypoints_wrist.append(gripper_goal_wrist)
            last_start = gripper_goal_tip

        return TaskGoal(gripper_goal_tip=goal_fn,
                        end_action=GripperActions.NONE,
                        success_thres_dist=max(self._success_thres_dist, 0.05),
                        success_thres_rot=self._success_thres_rot,
                        ee_fn=partial(SplinePlanner, waypoints_wrist=waypoints_wrist[:-1]))


class SimpleObstacleTask(BaseTask):
    @staticmethod
    def taskname() -> str:
        return "simpleObstacle"

    @staticmethod
    def requires_simulator() -> bool:
        return False

    def __init__(self, env: CombinedEnv, obstacle_spacing: float, use_fwd_orientation: bool,
                 eval: bool, offset_std: float, goal_dist_rng=(0.5, 8), goal_height_rng=None):
        map = SimpleObstacleMap(**env.get_map_config(),
                                  obstacle_spacing=obstacle_spacing,
                                  offset_std=offset_std)

        super(SimpleObstacleTask, self).__init__(env=env,
                                                 initial_joint_distribution='rnd',
                                                 map=map,
                                                 use_fwd_orientation=use_fwd_orientation,
                                                 eval=eval)

        if goal_height_rng is None:
            goal_height_rng = (env.robot_config["z_min"], env.robot_config["z_max"])
        assert len(goal_dist_rng) == len(goal_height_rng) == 2
        self._goal_dist_rng = goal_dist_rng
        self._goal_height_rng = goal_height_rng

    def draw_goal(self):
        def goal_fn(base_tf: list):
            gripper_goal_wrist = sample_circle_goal(base_tf, self._goal_dist_rng, self._goal_height_rng, self.map._map_W_meter, self.map._map_H_meter, self.np_random,
                                                    distribution="uniform")
            # first transform to tip here so that we are sampling in the wrist frame: easier to prevent impossible goals
            return self.env.gripper_to_tip_tf(gripper_goal_wrist)

        return TaskGoal(gripper_goal_tip=goal_fn,
                        end_action=GripperActions.NONE,
                        success_thres_dist=self._success_thres_dist,
                        success_thres_rot=self._success_thres_rot,
                        ee_fn=partial(PointToPointPlanner, use_fwd_orientation=self.use_fwd_orientation, eval=self.eval))


class DynamicObstacleMap(SceneMap):
    def get_varying_scene_objects(self):
        return self.dyn_obstacles

    def map_reset(self):
        if self._world_type == 'sim':
            obj = WorldObjects.dyn_obstacle_sim
        elif self._world_type == 'gazebo':
            obj = WorldObjects.dyn_obstacle_gazebo
        elif self._world_type == 'world':
            obj = None
        else:
            raise NotImplementedError(self._world_type)

        values = [-3.5, -1.5, 1.5, 3.5]
        poses = [Pose(Point(x, y, 0), Quaternion(0, 0, 0, 1)) for x in values for y in values]

        self.dyn_obstacles = []
        if self._world_type != 'world':
            for i, pose in enumerate(poses):
                self.dyn_obstacles.append(DynObstacle(f"dyn_obstacle{i}",
                                                      obj,
                                                      pose=pose,
                                                      map_W_meter=self._map_W_meter,
                                                      map_H_meter=self._map_H_meter,
                                                      np_random=self.np_random))
        return super().map_reset()


class DynamicObstacleTask(BaseTask):
    @staticmethod
    def taskname() -> str:
        return "dynObstacle"

    @staticmethod
    def requires_simulator() -> bool:
        return False

    def __init__(self, env: CombinedEnv, use_fwd_orientation: bool, eval: bool, goal_dist_rng=(1, 5), goal_height_rng=None):
        map = DynamicObstacleMap(**env.get_map_config(),
                                 update_from_global_costmap=True,
                                 floorplan=(10, 10),
                                 pad_floorplan_img=False)
        super(DynamicObstacleTask, self).__init__(env=env,
                                                  # for hsr: if gripper too low, it'll block the lidar and gets annoying
                                                  initial_joint_distribution='restricted_ws',
                                                  map=map,
                                                  use_fwd_orientation=use_fwd_orientation,
                                                  eval=eval,
                                                  success_thres_dist=0.05)

        if goal_height_rng is None:
            # use restriced min as if the gripper is too low, blocking the lidar, it can constantly cause obstacles in the global map that the ee endlessly tries to avoid
            goal_height_rng = (env.robot_config["restricted_ws_z_min"], env.robot_config["restricted_ws_z_max"])
        assert len(goal_dist_rng) == len(goal_height_rng) == 2
        self._goal_dist_rng = goal_dist_rng
        self._goal_height_rng = goal_height_rng

    def draw_goal(self):
        def goal_fn(base_tf: list):
            # assumes we are currently at the origin!
            gripper_goal_wrist = sample_circle_goal(base_tf, self._goal_dist_rng, self._goal_height_rng, self.map._map_W_meter, self.map._map_H_meter, self.np_random)
            # first transform to tip here so that we are sampling in the wrist frame: easier to prevent impossible goals
            return self.env.gripper_to_tip_tf(gripper_goal_wrist)

        return TaskGoal(gripper_goal_tip=goal_fn,
                        end_action=GripperActions.NONE,
                        success_thres_dist=self._success_thres_dist,
                        success_thres_rot=self._success_thres_rot,
                        ee_fn=partial(PointToPointPlanner, use_fwd_orientation=self.use_fwd_orientation, eval=self.eval))
