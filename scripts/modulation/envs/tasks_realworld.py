import copy
from functools import partial
from pathlib import Path
from typing import List
import rospy

import numpy as np
from geometry_msgs.msg import PoseStamped, Point

from modulation.envs.combined_env import CombinedEnv
from modulation.envs.eeplanner import SplinePlanner, PointToPointPlanner, LinearPlannerFullPython
from modulation.envs.env_utils import rotate_in_place, translate_in_orientation, pose_to_list
from modulation.envs.map import Map
from modulation.envs.simulator_api import sample_circle_goal, SpawnObject
from modulation.envs.tasks import BaseTask, TaskGoal, GripperActions, DynamicObstacleMap, armarker_pose_to_listtf, publish_goalpointer_enabled
from modulation.envs.tasks_chained import BaseChainedTask, PickNPlaceChainedTask, DoorChainedTask, DrawerChainedTask, RoomDoorChainedTask
from pybindings import multiply_tfs


class AisOfficeSplineTask(BaseTask):
    @staticmethod
    def taskname() -> str:
        return "aisspline"

    @staticmethod
    def requires_simulator() -> bool:
        return False

    def __init__(self, env: CombinedEnv, use_fwd_orientation: bool, eval: bool, goal_dist_rng=(1, 5), goal_height_rng=None):
        map = Map(**env.get_map_config(),
                  update_from_global_costmap=False,
                  floorplan=Path(__file__).parent.parent.parent.parent / "gazebo_world" / "worlds" / "aisoffice2_spline.yaml",
                  initial_base_rng_x=(-6, -14), initial_base_rng_y=(-0.5, 3.5))
        super(AisOfficeSplineTask, self).__init__(env=env,
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
            return waypoints_wrist[-1]

        num_waypoints = 5

        base_tf = self.env.get_basetf_world()
        waypoints_wrist = []
        last_start = base_tf
        x_offset = 7
        for w in range(num_waypoints):
            i, max_i = 0, 1000
            while True:
                last_start_centered = last_start.copy()
                last_start_centered[0] += x_offset
                gripper_goal_tip = sample_circle_goal(last_start_centered, self._goal_dist_rng, self._goal_height_rng, map_width=9, map_height=5, np_random=self.np_random)
                rospy.loginfo_throttle(1, f"base_tf: {np.round(base_tf, 3)}, last_start_centered: {np.round(last_start_centered, 3)} gripper_goal_tip centered: {np.round(last_start_centered, 3)}")
                gripper_goal_tip[0] -= x_offset
                if not self.map.in_collision(np.array(gripper_goal_tip[:2]), use_inflated_map=True, inflation_radius_meter=0.0, ignore_below_height=gripper_goal_tip[2] - PointToPointPlanner.HEIGHT_INFLATION):
                    break
                else:
                    rospy.loginfo_throttle(1, f"Sampled spline waypoint {w}: {np.round(gripper_goal_tip, 3)} is in collision")
                    i += 1
                    assert i < max_i, "Could not draw collision-free waypoint"
            gripper_goal_wrist = self.env.gripper_to_tip_tf(gripper_goal_tip)
            waypoints_wrist.append(gripper_goal_wrist)
            last_start = gripper_goal_tip

        return TaskGoal(gripper_goal_tip=goal_fn,
                        end_action=GripperActions.NONE,
                        success_thres_dist=max(self._success_thres_dist, 0.05),
                        success_thres_rot=self._success_thres_rot,
                        ee_fn=partial(SplinePlanner, waypoints_wrist=waypoints_wrist[:-1]),
                        enable_goal_pointer=False)



class SplineTaskHall(BaseTask):
    @staticmethod
    def taskname() -> str:
        return "splinehall"

    @staticmethod
    def requires_simulator() -> bool:
        return False

    def __init__(self, env: CombinedEnv, use_fwd_orientation: bool, eval: bool, goal_dist_rng=(1, 5), goal_height_rng=None):
        map = Map(**env.get_map_config(),
                  update_from_global_costmap=False,
                  floorplan=Path(__file__).parent.parent.parent.parent / "gazebo_world" / "worlds" / "robothall_map_spline.yaml",
                  initial_base_rng_x=(-1, 5), initial_base_rng_y=(-1, 3.5))
        super(SplineTaskHall, self).__init__(env=env,
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
            return waypoints_wrist[-1]

        num_waypoints = 5

        base_tf = self.env.get_basetf_world()
        waypoints_wrist = []
        last_start = base_tf
        for w in range(num_waypoints):
            i, max_i = 0, 1000
            while True:
                last_start_centered = last_start.copy()
                gripper_goal_tip = sample_circle_goal(last_start_centered, self._goal_dist_rng, self._goal_height_rng, map_width=12, map_height=7, np_random=self.np_random)
                if not self.map.in_collision(np.array(gripper_goal_tip[:2]), use_inflated_map=True, inflation_radius_meter=0.9, ignore_below_height=0.0):
                    rospy.loginfo(f"Sampled spline waypoint {w}: {np.round(gripper_goal_tip, 3)}")
                    break
                else:
                    rospy.loginfo(f"In collision waypoint {w}: base_tf: {np.round(base_tf, 3)}, last_start_centered: {np.round(last_start_centered, 3)} gripper_goal_tip centered: {np.round(last_start_centered, 3)}")
                    i += 1
                    assert i < max_i, "Could not draw collision-free waypoint"
            gripper_goal_wrist = self.env.gripper_to_tip_tf(gripper_goal_tip)
            waypoints_wrist.append(gripper_goal_wrist)
            last_start = gripper_goal_tip

        return TaskGoal(gripper_goal_tip=goal_fn,
                        end_action=GripperActions.NONE,
                        success_thres_dist=max(self._success_thres_dist, 0.05),
                        success_thres_rot=self._success_thres_rot,
                        ee_fn=partial(SplinePlanner, waypoints_wrist=waypoints_wrist[:-1]),
                        enable_goal_pointer=False)


class AisOfficeRoomDoorTask(BaseChainedTask):
    @staticmethod
    def taskname() -> str:
        return "aisroomdoor"

    @staticmethod
    def requires_simulator() -> bool:
        return False

    def __init__(self, env: CombinedEnv, use_fwd_orientation: bool, eval: bool):
        map = DynamicObstacleMap(**env.get_map_config(),
                                 update_from_global_costmap=False)

        super(AisOfficeRoomDoorTask, self).__init__(env=env,
                                                    map=map,
                                                    close_gripper_at_start=False,
                                                    use_fwd_orientation=use_fwd_orientation,
                                                    eval=eval,
                                                    initial_joint_distribution="restricted_ws")
        self._success_thres_dist = 0.02

    def draw_goal(self) -> List[TaskGoal]:
        # in map frame
        x_offset = 0.0
        y_offset = -0.0
        z_offset = -0.0

        waypoint_wrist_infront = [[0.0535 + x_offset, -1.30 + y_offset, 1.041 + z_offset, -0.722, 0.031, 0.013, 0.691]]

        waypoints_wrist_move = [[0.0535 + x_offset, -1.247 + y_offset, 1.041 + z_offset, -0.722, 0.031, 0.013, 0.691],
                                [0.074 + x_offset, -1.092 + y_offset, 1.039 + z_offset, -0.717, -0.086, 0.013, 0.691],
                                [0.092 + x_offset, -0.810 + y_offset, 1.049 + z_offset, -0.652, -0.220, 0.120, 0.716],
                                [-0.039 + x_offset, -0.557 + y_offset, 1.049 + z_offset, -0.600, -0.337, 0.250, 0.681],
                                [-0.136 + x_offset, -0.447 + y_offset, 1.049 + z_offset, -0.553, -0.409, 0.333, 0.645],
                                [-0.352 + x_offset, -0.331 + y_offset, 1.049 + z_offset, -0.476, -0.497, 0.437, 0.579],
                                [-0.519 + x_offset, -0.298 + y_offset, 1.049 + z_offset, -0.418, -0.546, 0.498, 0.528]]

        waypoints_wrist_release = [[-0.519 + x_offset, -0.298 + y_offset, 1.049 + z_offset, -0.418, -0.546, 0.498, 0.528],
                                   [-0.339 + x_offset, -0.177 + y_offset, 1.049+ z_offset, -0.413, -0.550, 0.504, 0.522]]

        def _rotate_waypoints(waypoints):
            for i in range(len(waypoints)):
                waypoints[i] = rotate_in_place(waypoints[i], np.pi / 2)

        _rotate_waypoints(waypoint_wrist_infront)
        _rotate_waypoints(waypoints_wrist_move)
        _rotate_waypoints(waypoints_wrist_release)

        astar_ee_fn = partial(PointToPointPlanner, use_fwd_orientation=self.use_fwd_orientation, eval=self.eval)

        return [TaskGoal(gripper_goal_tip=self.env.gripper_to_tip_tf(waypoint_wrist_infront[0]),
                         end_action=GripperActions.NONE,
                         success_thres_dist=max(self._success_thres_dist, 0.05), success_thres_rot=self._success_thres_rot,
                         head_start=0.0, ee_fn=astar_ee_fn, enable_goal_pointer=False),
                TaskGoal(gripper_goal_tip=self.env.gripper_to_tip_tf(waypoints_wrist_move[0]),
                         end_action=GripperActions.GRASP,
                         success_thres_dist=self._success_thres_dist, success_thres_rot=self._success_thres_rot,
                         head_start=0.0, ee_fn=partial(LinearPlannerFullPython, max_vel=0.05), enable_goal_pointer=False),
                TaskGoal(gripper_goal_tip=waypoints_wrist_move[-1],
                         end_action=GripperActions.OPEN,
                         success_thres_dist=self._success_thres_dist,
                         success_thres_rot=self._success_thres_rot,
                         head_start=self.SUBGOAL_PAUSE,
                         ee_fn=partial(SplinePlanner, waypoints_wrist=waypoints_wrist_move[:-1]),
                         enable_goal_pointer=False),
                TaskGoal(gripper_goal_tip=waypoints_wrist_release[-1],
                         end_action=GripperActions.NONE,
                         success_thres_dist=self._success_thres_dist,
                         success_thres_rot=self._success_thres_rot,
                         head_start=self.SUBGOAL_PAUSE,
                         ee_fn=partial(LinearPlannerFullPython, max_vel=0.05),
                         enable_goal_pointer=False)
                ]

    def goal_callback(self, msg: PoseStamped):
        return


class PicknplaceRobotHall(PickNPlaceChainedTask):
    @staticmethod
    def taskname() -> str:
        return "picknplacehall"

    @staticmethod
    def requires_simulator() -> bool:
        return False

    def __init__(self, env: CombinedEnv, use_fwd_orientation: bool, eval: bool, map=None):
        if map is None:
            map = Map(**env.get_map_config(),
                      floorplan=Path(__file__).parent.parent.parent.parent / "gazebo_world" / "worlds" / "robothall_map.yaml",
                      update_from_global_costmap=env.get_world() == "world",
                      max_map_height=0.2)
        super(PicknplaceRobotHall, self).__init__(env=env,
                                                  obstacle_configuration='none',
                                                  use_fwd_orientation=use_fwd_orientation,
                                                  eval=eval,
                                                  map=map)

    def _sample_pose_on_table(self, table):
        # pick goals
        if table == 0:
            # table towards center of hall
            x = 4.45
            y = self.np_random.uniform(0.1, 0.7)
            z = 0.48
            obj_loc = [x, y, z] + [0, 0, 0]
        elif table == 1:
            #     table at upper hall wall
            x = 3.3
            y = self.np_random.uniform(-4.1, -3.6)
            z = 0.48
            obj_loc = [x, y, z] + [0, 0, 0]
        elif table == 2:
            #     in front of poster
            x = self.np_random.uniform(-0.3, 0.)
            y = 2.22
            z = 0.75
            obj_loc = [x, y, z] + [0, 0, np.pi / 2]
        else:
            raise ValueError()

        # obj_loc[-1] += np.pi / 2
        return obj_loc

    def _sample_table_poses(self):
        pick_table, place_table = self.np_random.choice([0, 1, 2], size=2, replace=False)
        table_names = {0: "table towards center of hall",
                       1: "table at upper hall wall",
                       2: "in front of poster"}
        print(f"\nPICK TABLE: {pick_table} {table_names[pick_table]}\nPLACE TABLE: {place_table} {table_names[place_table]}")
        obj_loc = self._sample_pose_on_table(pick_table)
        place_loc = self._sample_pose_on_table(place_table)
        return obj_loc, place_loc

    def draw_goal(self) -> List[TaskGoal]:
        obj_loc, place_loc = self._sample_table_poses()

        # armarker_pose_to_listtf() adds 90 degree yaw to first table poses. This passes through the same callback, so
        # subtract 90 degree here (but not for the place_loc)
        obj_loc[-1] -= np.pi / 2
        obj_loc_high = copy.deepcopy(obj_loc)
        obj_loc_high[2] += 0.05

        in_front_of_obj_loc = translate_in_orientation(obj_loc, [-0.2, 0., 0.])

        # place goals
        place_loc_high = copy.deepcopy(place_loc)
        place_loc_high[2] += 0.1

        return [TaskGoal(gripper_goal_tip=in_front_of_obj_loc, end_action=GripperActions.NONE,
                         success_thres_dist=max(self._success_thres_dist, 0.05), success_thres_rot=self._success_thres_rot,
                         head_start=0.0, ee_fn=self._ee_fn),
                TaskGoal(gripper_goal_tip=obj_loc, end_action=GripperActions.GRASP,
                         success_thres_dist=self._success_thres_dist, success_thres_rot=self._success_thres_rot,
                         head_start=0.0, ee_fn=partial(LinearPlannerFullPython, max_vel=0.05)),
                TaskGoal(gripper_goal_tip=obj_loc_high, end_action=GripperActions.NONE,
                         success_thres_dist=max(self._success_thres_dist, 0.05), success_thres_rot=self._success_thres_rot,
                         head_start=2 * self.SUBGOAL_PAUSE, ee_fn=partial(LinearPlannerFullPython, max_vel=0.05), enable_goal_pointer=False),
                TaskGoal(gripper_goal_tip=place_loc_high, end_action=GripperActions.NONE,
                         success_thres_dist=max(self._success_thres_dist, 0.05), success_thres_rot=self._success_thres_rot,
                         head_start=self.SUBGOAL_PAUSE, ee_fn=self._ee_fn, enable_goal_pointer=False),
                TaskGoal(gripper_goal_tip=place_loc, end_action=GripperActions.OPEN,
                         success_thres_dist=self._success_thres_dist, success_thres_rot=self._success_thres_rot,
                         head_start=0.0, ee_fn=partial(LinearPlannerFullPython, max_vel=0.05), enable_goal_pointer=False)]

    def goal_callback(self, msg: PoseStamped, marker_offset=(0.04, 0.0, -0.045)):
        if self.current_goal in [0, 1]:
            # pitch correction for marker on top of the cereal box
            # p = rotate_in_place(msg.pose, pitch_radians=-np.pi)
            # pp = msg.pose
            # pp.orientation.x = p[3]
            # pp.orientation.y = p[4]
            # pp.orientation.z = p[5]
            # pp.orientation.w = p[6]

            obj_loc = armarker_pose_to_listtf(msg.pose)

            obj_loc = translate_in_orientation(obj_loc, list(marker_offset))
            in_front_of_obj_loc = translate_in_orientation(obj_loc, [-0.2, 0., 0.])
            self.goals[0].gripper_goal_tip = in_front_of_obj_loc
            self.goals[1].gripper_goal_tip = obj_loc
            obj_loc_high = copy.deepcopy(obj_loc)
            obj_loc_high[2] += 0.1
            self.goals[2].gripper_goal_tip = obj_loc_high
            self.env.unwrapped._ee_planner.set_goal(self.goals[self.current_goal].gripper_goal_tip)

            self.env.publish_marker(self.goals[0].gripper_goal_tip, 11, "aaa", "blue", 1.0)
            self.env.publish_marker(self.goals[1].gripper_goal_tip, 11, "bbb", "blue", 1.0)
            self.env.publish_marker(self.goals[2].gripper_goal_tip, 11, "ccc", "blue", 1.0)


class AisOfficePicknplace(PicknplaceRobotHall):
    @staticmethod
    def taskname() -> str:
        return "aispicknplace"

    def __init__(self, env: CombinedEnv, use_fwd_orientation: bool, eval: bool):
        map = Map(**env.get_map_config(),
                  floorplan=Path(__file__).parent.parent.parent.parent / "gazebo_world" / "worlds" / "aisoffice_new.yaml",
                  update_from_global_costmap=env.get_world() == "world",
                  max_map_height=0.2)
        super(PicknplaceRobotHall, self).__init__(env=env,
                                                  obstacle_configuration='none',
                                                  use_fwd_orientation=use_fwd_orientation,
                                                  eval=eval,
                                                  map=map)

    def _sample_pose_on_table(self, table):
        marker_to_grasp_z = 0.079
        if table == 0:
            # left table in kitchen sitting room
            return [4.386, -2.628, 0.870 - marker_to_grasp_z, 0.0, 0.0, - 0.2]
        elif table == 1:
            # right table in kitchen sitting room
            return [5.389, -5.076, 0.867 - marker_to_grasp_z, 0.0, 0.0, - 0.75 * np.pi]
        elif table == 2:
            #  in front of coffee machine
            return [1.829, -2.826, 1.058 - marker_to_grasp_z, 0.0, 0.0, 0.0]
        elif table == 3:
            # in front of nikolai
            return [-3.402, 1.321, 0.859 - marker_to_grasp_z, 0.0, 0.0, np.pi / 2]
        elif table == 4:
            # in front of johannes
            return [-5.837, 0.256, 0.860 - marker_to_grasp_z, 0.0, 0.0, np.pi]
        elif table == 5:
            # Tim
            return [4.750, 6.679, 0.900 - marker_to_grasp_z, 0.0, 0.0, np.pi / 2]
        elif table == 6:
            # hall table
            return [-0.325, 2.565, 0.829 - marker_to_grasp_z, 0.0, 0.0, np.pi]
        else:
            raise NotImplementedError(table)

    def _sample_table_poses(self):
        pick_table, place_table = self.np_random.choice([0, 1, 3, 4, 5, 6], size=2, replace=False)

        table_names = {0: "kitchen 1",
                       1: "kitchen 2",
                       2: "coffee",
                       3: "nikolai",
                       4: "johannes",
                       5: "tim",
                       6: "hall table"}
        print(f"\nPICK TABLE: {pick_table} {table_names[pick_table]}\nPLACE TABLE: {place_table} {table_names[place_table]}")
        obj_loc = self._sample_pose_on_table(pick_table)
        obj_loc[2] += 0.07
        place_loc = self._sample_pose_on_table(place_table)
        place_loc[2] -= 0.025
        return obj_loc, place_loc

    def goal_callback(self, msg: PoseStamped, marker_offset=(0.08, 0.0, -0.1)):
        return super(AisOfficePicknplace, self).goal_callback(msg=msg, marker_offset=marker_offset)


class DoorChainedTaskHall(DoorChainedTask):
    @staticmethod
    def taskname() -> str:
        return "doorhall"

    def __init__(self, env: CombinedEnv, eval: bool):
        super(DoorChainedTaskHall, self).__init__(env=env, obstacle_configuration='none', eval=eval)
        self._success_thres_dist = 0.01


class AisOfficeDoorChainedTask(DoorChainedTask):
    @staticmethod
    def taskname() -> str:
        return "aisdoor"

    def __init__(self, env: CombinedEnv, eval: bool):
        map = Map(**env.get_map_config(),
                  floorplan=Path(__file__).parent.parent.parent.parent / "gazebo_world" / "worlds" / "aisoffice_door.yaml",
                  update_from_global_costmap=False)
        super(AisOfficeDoorChainedTask, self).__init__(env=env, obstacle_configuration='none', eval=eval, map=map)
        self._success_thres_dist = 0.02

    def get_goal_objects(self, shelf_pos=Point(x=2.53, y=0.125, z=0.24)):
        return super(AisOfficeDoorChainedTask, self).get_goal_objects(shelf_pos=shelf_pos)

    def goal_callback(self, msg: PoseStamped, marker_offset=(-0.008, 0.16, 0.0)):
        return super(AisOfficeDoorChainedTask, self).goal_callback(msg=msg, marker_offset=marker_offset)


class DrawerChainedTaskHall(DrawerChainedTask):
    @staticmethod
    def taskname() -> str:
        return "drawerhall"

    def __init__(self, env: CombinedEnv, eval: bool):
        super(DrawerChainedTaskHall, self).__init__(env=env, obstacle_configuration='none', eval=eval)
        self._success_thres_dist = 0.01

    def get_goal_objects(self, drawer_pos=Point(x=-0.5, y=1.0, z=0.24)):
        return super(DrawerChainedTaskHall, self).get_goal_objects(drawer_pos=drawer_pos)


class AisOfficeDrawerChainedTask(DrawerChainedTask):
    @staticmethod
    def taskname() -> str:
        return "aisdrawer"

    def __init__(self, env: CombinedEnv, eval: bool):
        map = Map(**env.get_map_config(),
                  floorplan=Path(__file__).parent.parent.parent.parent / "gazebo_world" / "worlds" / "aisoffice_door.yaml",
                  update_from_global_costmap=False)
        super(AisOfficeDrawerChainedTask, self).__init__(env=env, obstacle_configuration='none', eval=eval, map=map)
        self._success_thres_dist = 0.0115

    def get_goal_objects(self, drawer_pos=Point(x=2.51, y=0.125, z=0.75)):
        return super(AisOfficeDrawerChainedTask, self).get_goal_objects(drawer_pos=drawer_pos)

    def goal_callback(self, msg: PoseStamped, marker_offset=(-0.02165, -0.027, -0.188)):
        return super(AisOfficeDrawerChainedTask, self).goal_callback(msg=msg, marker_offset=marker_offset)


class RoomDoorChainedTaskHall(RoomDoorChainedTask):
    @staticmethod
    def taskname() -> str:
        return "roomDoorhall"

    def __init__(self, env: CombinedEnv, eval: bool):
        super(RoomDoorChainedTaskHall, self).__init__(env=env, eval=eval)
        self._success_thres_dist = 0.01

    def get_goal_objects(self, door_pos=Point(x=1.96, y=-1, z=2.040139)) -> List[SpawnObject]:
        return super(RoomDoorChainedTaskHall, self).get_goal_objects(door_pos=door_pos)

    def goal_callback(self, msg: PoseStamped):
        # no longer update from marker once we start opening the door
        if self.current_goal == 0:
            gripper_goal_tip = armarker_pose_to_listtf(msg.pose)
            gripper_goal_tip = translate_in_orientation(gripper_goal_tip, [-0.06, 0.065, -0.0475])
            grasp_goal, opening_goal, release_goal = self.draw_goal(gripper_goal_tip)
            # grasp_goal, opening_goal = self.draw_goal(gripper_goal_tip)

            self.goals[0] = grasp_goal
            self.goals[1] = opening_goal
            self.goals[2] = release_goal
            self.env.unwrapped._ee_planner.set_goal(self.goals[self.current_goal].gripper_goal_tip)
