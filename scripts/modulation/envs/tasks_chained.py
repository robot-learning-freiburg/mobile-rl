import copy
from functools import partial
from typing import List, Optional

import numpy as np
from geometry_msgs.msg import Point, Pose, Quaternion, PoseStamped
from pybindings import multiply_tfs

from modulation.envs.combined_env import CombinedEnv
from modulation.envs.eeplanner import GMMPlannerWrapper, PointToPointPlanner, LinearPlannerFullPython
from modulation.envs.env_utils import pose_to_list, list_to_pose, PROJECT_ROOT, yaw_to_quaternion, rpy_to_quaternion, \
    rotate_in_place, translate_in_orientation
from modulation.envs.map import SceneMap
from modulation.envs.simulator_api import WorldObjects, SpawnObject, DEFAULT_FRAME, DynObstacle
from modulation.envs.tasks import BaseTask, TaskGoal, GripperActions, armarker_pose_to_listtf, publish_goalpointer_enabled


def gmm_obj_origin_to_tip(gmm_model_path, obj_origin):
    """creating a dummy planner is simpler than to write another model.csv parser in python"""
    return GMMPlannerWrapper.obj_origin_to_tip(obj_origin, str(gmm_model_path))


def get_gmm_release_input(move_input: list, last_mu_move: list):
    # transform the translation part into the move_input orientation and apply
    translation = multiply_tfs([0, 0, 0] + move_input[3:], last_mu_move, False)
    move_input_translated = multiply_tfs(translation[:3] + [0, 0, 0, 1], move_input, False)

    # rotate around origin
    release_rotation = multiply_tfs(last_mu_move, [0, 0, 0] + move_input[3:], True)
    release_input = move_input_translated[:3] + release_rotation[3:]
    return release_input


class BaseChainedTask(BaseTask):
    SUBGOAL_PAUSE = 2
    _motion_model_path = PROJECT_ROOT / "GMM_models"

    def __init__(self, env: CombinedEnv, map: SceneMap, use_fwd_orientation: bool,
                 eval: bool, close_gripper_at_start: bool = True, initial_joint_distribution: str = "rnd"):
        super(BaseChainedTask, self).__init__(env=env,
                                              initial_joint_distribution=initial_joint_distribution,
                                              map=map,
                                              close_gripper_at_start=close_gripper_at_start,
                                              use_fwd_orientation=use_fwd_orientation,
                                              eval=eval)

        assert self._motion_model_path.exists(), self._motion_model_path

        self.current_goal = 0
        # will be set at every reset
        self.goals = []

    def grasp(self, wait_for_result: bool = False):
        # wait_for_result = (self.SUBGOAL_PAUSE == 0)
        self.env.close_gripper(0.0, wait_for_result)

    def draw_goal(self) -> List[TaskGoal]:
        raise NotImplementedError()

    def get_goal_objects(self) -> List[SpawnObject]:
        return []

    def reset(self):
        # NOTE: OVERRIDING ANY PREVIOUS OBJECTS OF THE MAP. COULD PROBABLY BE HANDLED A BIT MORE GENERAL
        self.env.map.fixed_scene_objects = self.get_goal_objects()
        self.current_goal = 0
        self.goals = self.draw_goal()
        first_goal = self.goals[self.current_goal]
        return super().reset(first_goal)

    def _episode_cleanup(self):
        self.env.open_gripper(wait_for_result=False)
        self.map.clear()

    def _task_success_gazebo(self) -> dict:
        """helper to add additional stats to info if running in gazebo"""
        return {}

    def step(self, action, **kwargs):
        obs, reward, done, info = self.env.step(action=action, **kwargs)

        if done and info['ee_done']:
            end_action = self.goals[self.current_goal].end_action
            if end_action == GripperActions.GRASP:
                self.grasp()
            elif end_action == GripperActions.OPEN:
                self.env.open_gripper(wait_for_result=False)

            if self.current_goal < len(self.goals) - 1:
                self.current_goal += 1
                new = self.goals[self.current_goal]
                if self.env.get_world() == 'world':
                    print(F"\nSUBGOAL {self.current_goal - 1} REACHED\n")
                    publish_goalpointer_enabled(new.enable_goal_pointer)
                ee_planner = new.ee_fn(gripper_goal_tip=new.gripper_goal_tip,
                                       head_start=new.head_start,
                                       map=self.env.map,
                                       robot_config=self.env.robot_config,
                                       success_thres_dist=new.success_thres_dist,
                                       success_thres_rot=new.success_thres_rot)
                obs = self.env.set_ee_planner(ee_planner=ee_planner)
                done = False

        # ensure nothing left attached to the robot / the robot could spawn into / ...
        if done:
            if self.env.get_world() == 'gazebo':
                info.update(self._task_success_gazebo())
            self._episode_cleanup()

        return obs, reward, done, info

    def clear(self):
        self._episode_cleanup()
        super(BaseChainedTask, self).clear()

    def get_gmm_task_goal(self, model_file: str, gripper_action: GripperActions, head_start, obj_origin_input, success_thres_dist: float = None, interpolate_z: bool = True, enable_goal_pointer: bool = True):
        if success_thres_dist is None:
            success_thres_dist = self._success_thres_dist
        motion_plan = str(self._motion_model_path / model_file)
        return TaskGoal(gripper_goal_tip=gmm_obj_origin_to_tip(motion_plan, obj_origin_input),
                        end_action=gripper_action,
                        success_thres_dist=success_thres_dist,
                        success_thres_rot=self._success_thres_rot,
                        head_start=head_start,
                        ee_fn=partial(GMMPlannerWrapper, gmm_model_path=motion_plan, interpolate_z=interpolate_z),
                        enable_goal_pointer=enable_goal_pointer)


class ObstacleConfigMap(SceneMap):
    def __init__(self,
                 global_map_resolution: float,
                 local_map_resolution: float,
                 world_type: str,
                 robot_frame_id: str,
                 obstacle_configuration: str,
                 inflation_radius: float,
                 floorplan=None,
                 orig_resolution: Optional[float] = None,
                 update_from_global_costmap=False,
                 initial_base_rng_x=(-1.0, 1.0),
                 initial_base_rng_y=(-1.0, 1.0),
                 initial_base_rng_yaw=(-np.pi, np.pi),
                 max_map_height: float = None):
        if floorplan is None:
            floorplan = (10, 10)
        super(ObstacleConfigMap, self).__init__(global_map_resolution=global_map_resolution,
                                                local_map_resolution=local_map_resolution,
                                                world_type=world_type,
                                                initial_base_rng_x=initial_base_rng_x,
                                                initial_base_rng_y=initial_base_rng_y,
                                                initial_base_rng_yaw=initial_base_rng_yaw,
                                                robot_frame_id=robot_frame_id,
                                                requires_spawn=False,
                                                inflation_radius=inflation_radius,
                                                update_from_global_costmap=update_from_global_costmap,
                                                floorplan=floorplan,
                                                orig_resolution=orig_resolution,
                                                pad_floorplan_img=False,
                                                max_map_height=max_map_height)
        self.obstacle_configuration = obstacle_configuration

    @staticmethod
    def _add_inpath_obstacles() -> List[SpawnObject]:
        positions = []
        # between robot and pick-table
        positions.append((1.5, 0))
        # between robot and door-kallax.
        positions.append((-0.0, 1.5))
        # between robot and drawer-kallax
        positions.append((-1.5, -0.0))

        spawn_objects = []
        for i, pos in enumerate(positions):
            obstacle_pose = Pose(Point(pos[0], pos[1], 0.24), Quaternion(0, 0, 0, 1))
            obj = SpawnObject(f"obstacle{i}", WorldObjects.cylinder_inpath, obstacle_pose)
            spawn_objects.append((obj))
        return spawn_objects

    def _add_rnd_obstacles(self) -> List[SpawnObject]:
        # between robot and pick-table
        obj0 = SpawnObject(f"obstacle0", WorldObjects.cylinder_inpath,
                           lambda: Pose(Point(self.np_random.uniform(1.0, 2.5), self.np_random.uniform(-1.0, 1.0), 0.24), Quaternion(0, 0, 0, 1)))
        # between robot and door-kallax.
        obj1 = SpawnObject(f"obstacle1", WorldObjects.cylinder_inpath,
                           lambda: Pose(Point(self.np_random.uniform(-1.0, 1.0), self.np_random.uniform(1.0, 2.0), 0.24), Quaternion(0, 0, 0, 1)))
        # between robot and drawer-kallax
        obj2 = SpawnObject(f"obstacle2", WorldObjects.cylinder_inpath,
                           lambda: Pose(Point(self.np_random.uniform(-1.0, -2.0), self.np_random.uniform(-1.0, 1.0), 0.24), Quaternion(0, 0, 0, 1)))
        return [obj0, obj1, obj2]

    def _get_dyn_obstacles(self) -> List[SpawnObject]:
        if self._world_type == 'sim':
            obj = WorldObjects.dyn_obstacle_sim
        elif self._world_type == 'gazebo':
            obj = WorldObjects.dyn_obstacle_gazebo
        elif self._world_type == 'world':
            obj = None
        else:
            raise NotImplementedError(self._world_type)

        poses = [Pose(Point(self.np_random.uniform(1.0, 2.5), self.np_random.uniform(-1.0, 1.0), 0.24),
                      Quaternion(0, 0, 0, 1)),
                 Pose(Point(self.np_random.uniform(-1.0, 1.0), self.np_random.uniform(1.0, 2.0), 0.24),
                      Quaternion(0, 0, 0, 1)),
                 Pose(Point(self.np_random.uniform(-1.0, -2.0), self.np_random.uniform(-1.0, 1.0), 0.24),
                      Quaternion(0, 0, 0, 1))]

        dyn_obstacles = []
        for i, pose in enumerate(poses):
            dyn_obs = DynObstacle(f"dyn_obstacle{i}",
                                  obj,
                                  pose,
                                  map_W_meter=min(self._map_W_meter, 6),
                                  map_H_meter=min(self._map_H_meter, 6),
                                  np_random=self.np_random)
            dyn_obstacles.append(dyn_obs)
        return dyn_obstacles

    def map_reset(self):
        if self.obstacle_configuration == 'dyn':
            self.dyn_obstacles = self._get_dyn_obstacles()
        return super().map_reset()

    def get_varying_scene_objects(self) -> List[SpawnObject]:
        if self.obstacle_configuration == 'inpath':
            return self._add_inpath_obstacles()
        elif self.obstacle_configuration == 'rnd':
            return self._add_rnd_obstacles()
        elif self.obstacle_configuration == 'none':
            return []
        elif self.obstacle_configuration == 'dyn':
            return self.dyn_obstacles
        else:
            raise NotImplementedError(self.obstacle_configuration)


class PickNPlaceChainedTask(BaseChainedTask):
    @staticmethod
    def taskname() -> str:
        return "picknplace"

    @staticmethod
    def requires_simulator() -> bool:
        return False

    def __init__(self, env: CombinedEnv, obstacle_configuration: str, use_fwd_orientation: bool, eval: bool, map=None):
        if map is None:
            map = ObstacleConfigMap(**env.get_map_config(),
                                    obstacle_configuration=obstacle_configuration,
                                    update_from_global_costmap=env.get_world() == "world")
        super(PickNPlaceChainedTask, self).__init__(env=env,
                                                    map=map,
                                                    close_gripper_at_start=False,
                                                    use_fwd_orientation=use_fwd_orientation,
                                                    eval=eval)
        self._pick_obj = WorldObjects.muesli2
        self._pick_table = WorldObjects.reemc_table_low
        self._place_table = WorldObjects.reemc_table_low
        self._ee_fn = partial(PointToPointPlanner, use_fwd_orientation=self.use_fwd_orientation, eval=self.eval)

        self._pick_obj_name = "pick_obj"
        self.set_gazebo_object_to_ignore([{f"{self._pick_obj_name}::link::collision"}])

    def get_goal_objects(self, start_table_pos=Point(x=3.3, y=0, z=0), end_table_rng=(-1.5, 2, -3, -1.5)):
        objects = []
        # pick table
        start_table_pose = Pose(start_table_pos, Quaternion(0, 0, 1, 0))
        objects.append(SpawnObject("pick_table", self._pick_table, start_table_pose))
        # place table
        endx = self.np_random.uniform(end_table_rng[0], end_table_rng[1])
        endy = self.np_random.uniform(end_table_rng[2], end_table_rng[3])
        end_table_pose = Pose(Point(x=endx, y=endy, z=0), Quaternion(0, 0, 1, 1))
        objects.append(SpawnObject("place_table", self._place_table, end_table_pose))

        # place object on edge of the table (relative to the table position)
        # NOTE: won't be correct yet if table not in front of robot
        pose_on_table = Pose(Point(x=self._pick_table.x / 2 - 0.1,
                                   y=self.np_random.uniform(-self._pick_table.y + 0.1, self._pick_table.y - 0.1) / 2,
                                   z=self._pick_table.z + self._pick_obj.z + 0.01),
                             Quaternion(0, 0, 1, 1))
        self.pick_obj_pose = list_to_pose(multiply_tfs(pose_to_list(start_table_pose), pose_to_list(pose_on_table), False))
        objects.append(SpawnObject(self._pick_obj_name, self._pick_obj, self.pick_obj_pose))

        # target position to place the object
        self.place_obj_pose = list_to_pose(multiply_tfs(pose_to_list(end_table_pose), pose_to_list(pose_on_table), False))
        return objects

    def goal_callback(self, msg: PoseStamped):
        if self.current_goal in [0, 1]:
            obj_loc = armarker_pose_to_listtf(msg.pose)
            obj_loc = translate_in_orientation(obj_loc, [0.05, 0.0, -0.05])
            in_front_of_obj_loc = translate_in_orientation(obj_loc, [-0.2, 0., 0.])
            self.goals[0].gripper_goal_tip = in_front_of_obj_loc
            self.goals[1].gripper_goal_tip = obj_loc
            self.env.unwrapped._ee_planner.set_goal(self.goals[self.current_goal].gripper_goal_tip)

    def goal_callback2(self, msg: PoseStamped):
        self.goals[2].gripper_goal_tip = pose_to_list(msg.pose)
        if self.current_goal == 2:
            self.env.unwrapped._ee_planner.set_goal(self.goals[self.current_goal].gripper_goal_tip)

    def draw_goal(self) -> List[TaskGoal]:
        # pick goals
        world_target_pos = self.pick_obj_pose.position
        obj_loc = [world_target_pos.x, world_target_pos.y, world_target_pos.z - 0.04] + [0, 0, 0]
        in_front_of_obj_loc = translate_in_orientation(obj_loc, [-0.2, 0., 0.])

        # place goals
        world_end_target_pos = self.place_obj_pose.position
        place_loc = [world_end_target_pos.x, world_end_target_pos.y + 0.05, world_end_target_pos.z + 0.05] + [0, 0, -np.pi / 2]

        return [TaskGoal(gripper_goal_tip=in_front_of_obj_loc, end_action=GripperActions.NONE,
                         success_thres_dist=max(self._success_thres_dist, 0.05), success_thres_rot=self._success_thres_rot,
                         head_start=0.0, ee_fn=self._ee_fn),
                TaskGoal(gripper_goal_tip=obj_loc, end_action=GripperActions.GRASP,
                         success_thres_dist=self._success_thres_dist, success_thres_rot=self._success_thres_rot,
                         head_start=0.0, ee_fn=partial(LinearPlannerFullPython, max_vel=0.05)),
                TaskGoal(gripper_goal_tip=place_loc, end_action=GripperActions.OPEN,
                         success_thres_dist=self._success_thres_dist, success_thres_rot=self._success_thres_rot,
                         head_start=self.SUBGOAL_PAUSE, ee_fn=self._ee_fn, enable_goal_pointer=False)]

    def _task_success_gazebo(self):
        pick_obj_pose = self.map.simulator.get_model(self._pick_obj_name, DEFAULT_FRAME).pose
        dist_to_start = np.linalg.norm(np.array(self.goals[0].gripper_goal_tip[:3]) - [pick_obj_pose.position.x, pick_obj_pose.position.y, pick_obj_pose.position.z])
        dist_to_goal = np.linalg.norm(np.array(self.goals[-1].gripper_goal_tip[:3]) - [pick_obj_pose.position.x, pick_obj_pose.position.y, pick_obj_pose.position.z])
        return {'object_moved': dist_to_start > 0.5,
                'object_placed': dist_to_goal < 0.5}


class PickNPlaceDynChainedTask(PickNPlaceChainedTask):
    @staticmethod
    def taskname() -> str:
        return "picknplacedyn"

    def __init__(self, env, *args, **kwargs):
        map = ObstacleConfigMap(**env.get_map_config(),
                                obstacle_configuration="dyn",
                                update_from_global_costmap=True)
        super(PickNPlaceDynChainedTask, self).__init__(env=env, map=map, obstacle_configuration="dyn", *args, **kwargs)


class DoorChainedTask(BaseChainedTask):
    @staticmethod
    def taskname() -> str:
        return "door"

    @staticmethod
    def requires_simulator() -> bool:
        return False

    def __init__(self, env: CombinedEnv, obstacle_configuration: str, eval: bool, map=None):
        if map is None:
            map = ObstacleConfigMap(**env.get_map_config(),
                                    obstacle_configuration=obstacle_configuration)
        super(DoorChainedTask, self).__init__(env=env,
                                              map=map,
                                              close_gripper_at_start=False,
                                              use_fwd_orientation=False,
                                              eval=eval)
        self._shelf = WorldObjects.kallax2

        self._top_shelf_name = "target_shelf"
        self.set_gazebo_object_to_ignore(self.get_kallaxdoor_gazebo_objects_to_ignore(self._top_shelf_name, self.env.robot_config["gripper_collision_names"]))

    @staticmethod
    def get_kallaxdoor_gazebo_objects_to_ignore(top_shelf_name: str, gripper_collision_names: list) -> list:
        return ([{top_shelf_name + "::Door::DoorHandle"}] +
                [{top_shelf_name + "::Door::DoorBase", gripper} for gripper in gripper_collision_names])

    def get_goal_objects(self, shelf_pos=Point(x=0.0, y=3.3, z=0.24)):
        objects = []
        objects.append(SpawnObject("Kallax2_bottom", self._shelf, Pose(shelf_pos, Quaternion(0, 0, 0, 1))))
        p = copy.deepcopy(shelf_pos)
        p.z = 0.65
        self.target_shelf_pose = Pose(p, Quaternion(0, 0, 0, 1))
        objects.append(SpawnObject(self._top_shelf_name, self._shelf, self.target_shelf_pose))
        return objects

    @staticmethod
    def get_kallax_door_gmm_input(kallax_pose: Pose):
        kallax2_origin_to_door_pose = list_to_pose([0.02 - 0.01, -0.17, -0.017, 0, 0, -1, 1])
        door_pose_closed = list_to_pose(multiply_tfs(pose_to_list(kallax_pose), pose_to_list(kallax2_origin_to_door_pose), False))
        obj_origin_input = copy.deepcopy(door_pose_closed)
        obj_origin_input = pose_to_list(obj_origin_input)

        last_mu_move = [0.270917, 0.3880705, -0.0015275,  0.0005978116, -0.0016146731, 0.7553946748, -0.6552678236]
        release_input = get_gmm_release_input(obj_origin_input, last_mu_move)

        return obj_origin_input, release_input

    def draw_goal(self, gripper_goal_tip=None) -> List[TaskGoal]:
        if gripper_goal_tip is None:
            obj_origin_input, release_input = self.get_kallax_door_gmm_input(self.target_shelf_pose)
        else:
            obj_origin_input = rotate_in_place(gripper_goal_tip, np.pi)
            obj_origin_input = translate_in_orientation(obj_origin_input, [-0.0341, 0.1499, 0.0349])
        grasp_goal = self.get_gmm_task_goal("GMM_grasp_KallaxTuer.csv", GripperActions.GRASP, 0.0, obj_origin_input)
        opening_goal = self.get_gmm_task_goal("GMM_move_KallaxTuer.csv", GripperActions.OPEN, self.SUBGOAL_PAUSE, obj_origin_input,
                                              success_thres_dist=max(self._success_thres_dist, 0.05), enable_goal_pointer=False,
                                              interpolate_z=False)
        # release_goal = self.get_gmm_task_goal("GMM_release_KallaxTuer.csv", GripperActions.OPEN, self.SUBGOAL_PAUSE, release_input, enable_goal_pointer=False, interpolate_z=False)

        # self.env.publish_marker(obj_origin_input, 11, "gripper_goal", "blue", 1.0)
        # self.env.publish_marker(grasp_goal.gripper_goal_tip, 12, "gripper_goal", "blue", 0.7)
        # self.env.publish_marker(release_input, 1, "gripper_goal", "pink", 1.0)
        # self.env.publish_marker(release_goal.gripper_goal_tip, 3, "gripper_goal", "pink", 0.7)

        return [grasp_goal, opening_goal]

    def goal_callback(self, msg: PoseStamped, marker_offset=(-0.04, 0.16, 0.03)):
        # no longer update from marker once we start opening the door?
        if self.current_goal == 0:
            gripper_goal_tip = armarker_pose_to_listtf(msg.pose)
            gripper_goal_tip = translate_in_orientation(gripper_goal_tip, list(marker_offset))
            grasp_goal, opening_goal = self.draw_goal(gripper_goal_tip)

            self.goals[0] = grasp_goal
            self.goals[1] = opening_goal
            self.env.unwrapped._ee_planner.set_goal(self.goals[self.current_goal].gripper_goal_tip)


class DrawerChainedTask(BaseChainedTask):
    @staticmethod
    def taskname() -> str:
        return "drawer"

    @staticmethod
    def requires_simulator() -> bool:
        return False

    def __init__(self, env: CombinedEnv, obstacle_configuration: str, eval: bool, map=None):
        if map is None:
            map = ObstacleConfigMap(**env.get_map_config(),
                                    obstacle_configuration=obstacle_configuration)
        super(DrawerChainedTask, self).__init__(env=env,
                                                map=map,
                                                close_gripper_at_start=False,
                                                use_fwd_orientation=False,
                                                eval=eval)
        self._drawer = WorldObjects.kallax

        self._top_drawer_name = "target_drawer"
        self.set_gazebo_object_to_ignore([{self._top_drawer_name + "::Drawer1::D1handle"}] +
                                         [{self._top_drawer_name + "::Drawer1::Drawer1Front", gripper} for gripper in self.env.robot_config["gripper_collision_names"]])

    def get_goal_objects(self, drawer_pos=Point(x=-3.3, y=0.0, z=0.24)) -> List[SpawnObject]:
        objects = []
        objects.append(SpawnObject("Kallax_bottom", self._drawer, Pose(drawer_pos, Quaternion(0, 0, 1, 1))))

        p = copy.deepcopy(drawer_pos)
        p.z = 0.65
        self.target_drawer_pose = Pose(p, Quaternion(0, 0, 1, 1))
        objects.append(SpawnObject(self._top_drawer_name, self._drawer, self.target_drawer_pose))
        return objects

    def draw_goal(self, gripper_goal_tip=None) -> List[TaskGoal]:
        if gripper_goal_tip is None:
            door_pose_closed = translate_in_orientation(pose_to_list(self.target_drawer_pose), [-0., 0.03, -0.185])
            obj_origin_goal = rotate_in_place(door_pose_closed, -np.pi / 2)
            obj_origin_goal = translate_in_orientation(obj_origin_goal, [0.04, 0.0, 0.05])
        else:
            obj_origin_goal = rotate_in_place(gripper_goal_tip, np.pi)
            obj_origin_goal = translate_in_orientation(obj_origin_goal, [-0.1900519022, -0.0100236585, 0.0102411923])
        grasp_goal = self.get_gmm_task_goal("GMM_grasp_KallaxDrawer.csv", GripperActions.GRASP, 0.0, obj_origin_goal)
        opening_goal = self.get_gmm_task_goal("GMM_move_KallaxDrawer.csv", GripperActions.OPEN, self.SUBGOAL_PAUSE, obj_origin_goal,
                                              success_thres_dist=max(self._success_thres_dist, 0.05), enable_goal_pointer=False, interpolate_z=False)
        return [grasp_goal, opening_goal]

    def goal_callback(self, msg: PoseStamped, marker_offset=(-0.05, -0.027, 0.1875)):
        if self.current_goal in [0, 1]:
            gripper_goal_tip = armarker_pose_to_listtf(msg.pose)
            gripper_goal_tip = translate_in_orientation(gripper_goal_tip, list(marker_offset))
            grasp_goal, opening_goal = self.draw_goal(gripper_goal_tip)

            self.goals[0] = grasp_goal
            self.goals[1] = opening_goal
            self.env.unwrapped._ee_planner.set_goal(self.goals[self.current_goal].gripper_goal_tip)


class RoomDoorChainedTask(BaseChainedTask):
    @staticmethod
    def taskname() -> str:
        return "roomDoor"

    @staticmethod
    def requires_simulator() -> bool:
        return False

    def __init__(self, env: CombinedEnv, eval: bool):
        # assert env.get_world() == 'sim', "might fail atm because door_pos.z < 0.0 in get_goal_objects()"
        # NOTE: ignoring obstacle_configuration atm
        map_file = PROJECT_ROOT / "gazebo_world" / "worlds" /"modulation_tasks.yaml"
        map = ObstacleConfigMap(**env.get_map_config(),
                                obstacle_configuration='none',
                                initial_base_rng_x=(1.0, 2.5),
                                initial_base_rng_y=(1.0, 3.0),
                                floorplan=map_file)
        super(RoomDoorChainedTask, self).__init__(env=env,
                                                  map=map,
                                                  close_gripper_at_start=False,
                                                  use_fwd_orientation=False,
                                                  eval=eval)
        self._door = WorldObjects.hinged_doorHall78

        self._door_name = "target_door"
        self.set_gazebo_object_to_ignore(self.get_roomdoor_gazebo_objects_to_ignore(self._door_name, self.env.robot_config["gripper_collision_names"]))

    @staticmethod
    def get_roomdoor_gazebo_objects_to_ignore(door_name: str, gripper_collision_names: list) -> list:
        return ([{door_name + "::door::collision", gripper} for gripper in gripper_collision_names] +
                [{door_name + "::handles::DoorHandle1", gripper} for gripper in gripper_collision_names] +
                [{door_name + "::handles::handle1_collision", gripper} for gripper in gripper_collision_names])

    def get_goal_objects(self, door_pos=Point(x=1.96, y=5.02, z=2.040139)) -> List[SpawnObject]:
        # NOTE: the door transform has the z-axis pointing down!
        objects = []
        self.target_door_pose = Pose(door_pos, rpy_to_quaternion(-np.pi, 0.0, -np.pi / 2))
        objects.append(SpawnObject(self._door_name, self._door, self.target_door_pose))
        return objects

    @staticmethod
    def get_roomdoor_gmm_input(target_door_pose: list):
        # somehow the mu from the csv file does barely change the rotation. If I assume opening the door 90 degree (change in yaw of -np.pi/2), then it looks correct
        # last_mu_move = [0.7175531963, 0.9708252228, 0.0216328168, -0.5914252823, 0.02701118, 0.0331148093, 0.8052266395]
        last_mu_move_changed_rotation = [0.7175531963, 0.9708252228, 0.0216328168, 0.0, 0.0, 1.0, 1.0]
        release_input = get_gmm_release_input(target_door_pose, last_mu_move_changed_rotation)
        return target_door_pose, release_input

    def draw_goal(self, gripper_goal_tip=None) -> List[TaskGoal]:
        if gripper_goal_tip is None:
            gripper_target = pose_to_list(self.target_door_pose)
            gripper_target = translate_in_orientation(gripper_target, [0.1, 0.70, 1.05])
            gripper_goal_tip = rotate_in_place(gripper_target, yaw_radians=0, roll_radians=np.pi)

        # rotate once more, obj_origin_input should be oriented from the handle side towards the hinges of the door
        gripper_goal_tip = rotate_in_place(gripper_goal_tip, -np.pi / 2)
        obj_origin_input, release_input = self.get_roomdoor_gmm_input(gripper_goal_tip)

        grasp_goal = self.get_gmm_task_goal("GMM_grasp_roomDoor.csv", GripperActions.GRASP, 0.0, obj_origin_input, interpolate_z=False)
        opening_goal = self.get_gmm_task_goal("GMM_move_roomDoor.csv", GripperActions.OPEN, self.SUBGOAL_PAUSE, obj_origin_input, interpolate_z=False, enable_goal_pointer=False)
        release_goal = self.get_gmm_task_goal("GMM_release_roomDoor.csv", GripperActions.NONE, self.SUBGOAL_PAUSE, release_input, success_thres_dist=max(self._success_thres_dist, 0.05), interpolate_z=False, enable_goal_pointer=False)

        return [grasp_goal, opening_goal, release_goal]

    def goal_callback(self, msg: PoseStamped):
        # no longer update from marker once we start opening the door
        if self.current_goal == 0:
            gripper_goal_tip = armarker_pose_to_listtf(msg.pose)
            gripper_goal_tip = translate_in_orientation(gripper_goal_tip, [-0.06, 0.065, -0.0475])
            grasp_goal, opening_goal, release_goal = self.draw_goal(gripper_goal_tip)

            self.goals[0] = grasp_goal
            self.goals[1] = opening_goal
            self.goals[2] = release_goal
            self.env.unwrapped._ee_planner.set_goal(self.goals[self.current_goal].gripper_goal_tip)


class BookstorePickNPlaceChainedTask(BaseChainedTask):
    @staticmethod
    def taskname() -> str:
        return "bookstorePnP"

    @staticmethod
    def requires_simulator() -> bool:
        return False

    def __init__(self, env: CombinedEnv, use_fwd_orientation: bool, eval: bool):
        map_file = PROJECT_ROOT / "gazebo_world" / "worlds" / "aws_bookstore.yaml"

        # alternative: load a map (roughly) aware of obstacle heights
        # img_path, resolution = SceneMap.get_map_yaml(PROJECT_ROOT / "gazebo_world" / "worlds" / "aws_bookstore.yaml")
        # map_ground = SceneMap.load_map(img_path, resolution, SceneMap._resolution)
        # img_path, resolution = SceneMap.get_map_yaml(PROJECT_ROOT / "gazebo_world" / "worlds" / "aws_bookstore1_2m.yaml")
        # map_1_2m = SceneMap.load_map(img_path, resolution, SceneMap._resolution)
        # map_height = np.asarray(map_ground).copy()
        # map_height[(np.asarray(map_1_2m) == 0.0) & (np.asarray(map_ground) > 0.0)] = 1.1
        # map_file = Image.fromarray(map_height)

        map = ObstacleConfigMap(**env.get_map_config(),
                                obstacle_configuration='none',
                                initial_base_rng_x=(-1.0, 0.0),
                                initial_base_rng_y=(-1.5, -0.5),
                                floorplan=map_file)
        super(BookstorePickNPlaceChainedTask, self).__init__(env=env,
                                                             map=map,
                                                             close_gripper_at_start=False,
                                                             use_fwd_orientation=use_fwd_orientation,
                                                             eval=eval)
        self._pick_obj = WorldObjects.muesli2
        self._ee_fn = partial(PointToPointPlanner, use_fwd_orientation=self.use_fwd_orientation, eval=self.eval)
        self.muesli_poses = [Pose(Point(-5.85, -4.37, 1.175), yaw_to_quaternion(np.pi / 2)),
                             Pose(Point(-5.90, -3.25, 1.175), yaw_to_quaternion(np.pi / 2)),
                             Pose(Point(-5.36, -0.26, 1.06), yaw_to_quaternion(0.)),
                             Pose(Point(-3.06, 0.45, 1.16),  yaw_to_quaternion(0.)),
                             Pose(Point(-2.24, 0.45, 1.16),  yaw_to_quaternion(0.)),
                             Pose(Point(-5.72, 0.88, 1.06),  yaw_to_quaternion(-np.pi / 2)),
                             Pose(Point(-1.00, 2.88, 1.16),  yaw_to_quaternion(0.)),
                             Pose(Point(0.43, 4.00, 1.23),   yaw_to_quaternion(-0.68)),
                             Pose(Point(-5.20, 4.43, 1.05),  yaw_to_quaternion(np.pi / 2)),
                             Pose(Point(5.30, -0.95, 1.06),  yaw_to_quaternion(-np.pi / 2)),]

        # self.map.publish_floorplan_rviz()
        # for i, p in enumerate(self.muesli_poses):
        #     p = rotate_in_place(p, np.pi/2)
        #     self.env.publish_marker(p, 10 + i, "gripper_goal", "pink", 1.0)
        #     in_front_of_obj_loc = translate_in_orientation(p, [-0.2, 0., 0.])
        #     self.env.publish_marker(in_front_of_obj_loc, 100 + i, "gripper_goal", "cyan", 1.0)

        self._pick_obj_name = "pick_obj"
        self.set_gazebo_object_to_ignore([{f"{self._pick_obj_name}::link::collision"}])

    def get_goal_objects(self):
        self.pick_obj_pose, self.place_obj_pose = self.np_random.choice(self.muesli_poses, size=2, replace=False)
        return [SpawnObject(self._pick_obj_name, self._pick_obj, self.pick_obj_pose)]

    def draw_goal(self) -> List[TaskGoal]:
        angle = np.pi / 2
        # muesli orientation is 90 degree rotated from the gripper rotation required to pick it up
        start_goal = rotate_in_place(self.pick_obj_pose, angle)
        start_goal[2] -= 0.04
        in_front_of_obj_loc = translate_in_orientation(start_goal, [-0.2, 0., 0.])

        end_goal = rotate_in_place(self.place_obj_pose, angle)
        # make sure object doesn't hit the table if it slightly slips in the gripper
        end_goal[2] += 0.03
        return [TaskGoal(gripper_goal_tip=in_front_of_obj_loc, end_action=GripperActions.NONE,
                         success_thres_dist=self._success_thres_dist, success_thres_rot=max(self._success_thres_rot, 0.05),
                         head_start=0.0, ee_fn=self._ee_fn),
                TaskGoal(gripper_goal_tip=start_goal, end_action=GripperActions.GRASP,
                         success_thres_dist=self._success_thres_dist, success_thres_rot=self._success_thres_rot,
                         head_start=0, ee_fn=LinearPlannerFullPython),
                TaskGoal(gripper_goal_tip=end_goal, end_action=GripperActions.OPEN,
                         success_thres_dist=self._success_thres_dist, success_thres_rot=self._success_thres_rot,
                         head_start=self.SUBGOAL_PAUSE, ee_fn=self._ee_fn, enable_goal_pointer=False)]

    def _task_success_gazebo(self):
        pick_obj_pose = self.map.simulator.get_model(self._pick_obj_name, DEFAULT_FRAME).pose
        dist_to_start = np.linalg.norm(np.array(self.goals[0].gripper_goal_tip[:3]) - [pick_obj_pose.position.x, pick_obj_pose.position.y, pick_obj_pose.position.z])
        dist_to_goal = np.linalg.norm(np.array(self.goals[-1].gripper_goal_tip[:3]) - [pick_obj_pose.position.x, pick_obj_pose.position.y, pick_obj_pose.position.z])
        return {'object_moved': dist_to_start > 0.5,
                'object_placed': dist_to_goal < 0.5}
