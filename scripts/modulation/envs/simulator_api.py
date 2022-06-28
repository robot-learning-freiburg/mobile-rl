import random
import time
from enum import IntEnum
from typing import Iterable, NamedTuple, Tuple, Callable, Union
from dataclasses import dataclass
import math

import numpy as np
import rospy
from gazebo_msgs.msg import LinkStates
from gazebo_msgs.srv import DeleteModel, SpawnModel, GetModelState, SetModelState, SetModelConfiguration, \
    SetModelConfigurationRequest
from geometry_msgs.msg import Pose, Point, Quaternion
from std_srvs.srv import Empty

from modulation.envs.env_utils import publish_marker, clear_all_markers, SMALL_NUMBER, pose_to_list

DEFAULT_FRAME = "map"

class ObjectGeometry(IntEnum):
    """Shapes that we know how to project as obstacles onto the map during training"""
    UNKNOWN = 0
    RECTANGLE = 1
    CYLINDER = 2


def sample_circle_goal(base_tf: list, goal_dist_rng: Tuple[float, float], goal_height_rng: Tuple[float, float], map_width: float, map_height: float, np_random, distribution: str = "uniform") -> list:
    if distribution == "uniform":
        goal_dist = np_random.uniform(goal_dist_rng[0], goal_dist_rng[1])
    elif distribution == "triangular":
        goal_dist = np_random.triangular(goal_dist_rng[0], goal_dist_rng[1], goal_dist_rng[1])
    else:
        raise NotImplementedError(distribution)
    goal_orientation = np_random.uniform(0.0, math.pi)

    # sample around the origin, then add the base offset in case we don't start at the origin
    x = goal_dist * math.cos(goal_orientation)
    y = np_random.choice([-1, 1]) * goal_dist * math.sin(goal_orientation)
    z = np_random.uniform(goal_height_rng[0], goal_height_rng[1])

    RPY_goal = np_random.uniform(0, 2 * np.pi, 3)

    half_w = map_width / 2 - 0.5
    half_h = map_height / 2 - 0.5

    return [min(max(base_tf[0] + x, -half_w), half_w),
            min(max(base_tf[1] + y, -half_h), half_h),
            z] + RPY_goal.tolist()


def sample_square_goal(base_tf: list, max_goal_dist: float, goal_height_rng: Tuple[float, float], map_width: float, map_height: float, np_random) -> list:
    half_w = map_width / 2 - 0.5
    half_h = map_height / 2 - 0.5

    x = np_random.uniform(max(base_tf[0] - max_goal_dist, -half_w), min(base_tf[0] + max_goal_dist, half_w))
    y = np_random.uniform(max(base_tf[1] - max_goal_dist, -half_h), min(base_tf[1] + max_goal_dist, half_h))
    z = np_random.uniform(goal_height_rng[0], goal_height_rng[1])

    RPY_goal = np_random.uniform(0, 2 * np.pi, 3)

    return [x, y, z] + RPY_goal.tolist()


def get_cylinder(x: float, y: float, z: float):
    assert x == y, "No ellipses allowed, x must equal y"
    radius = x / 2
    length = z
    return f"""
    <?xml version="1.0"?>
    <sdf version="1.5">
        <model name='unit_cylinder'>
            <static>1</static>
            <link name='link'>
                <inertial>
                    <mass>1</mass>
                    <inertia>
                        <ixx>0.145833</ixx>
                        <ixy>0</ixy>
                        <ixz>0</ixz>
                        <iyy>0.145833</iyy>
                        <iyz>0</iyz>
                        <izz>0.125</izz>
                    </inertia>
                    <pose frame=''>0 0 0 0 -0 0</pose>
                </inertial>
                <collision name='collision'>
                    <geometry>
                        <cylinder>
                            <radius>{radius}</radius>
                            <length>{length}</length>
                        </cylinder>
                    </geometry>
                </collision>
                <visual name='visual'>
                    <geometry>
                        <cylinder>
                            <radius>{radius}</radius>
                            <length>{length}</length>
                        </cylinder>
                    </geometry>
                    <material>
                        <script>
                            <name>Gazebo/Black</name>
                            <uri>file://media/materials/scripts/gazebo.material</uri>
                        </script>
                    </material>
                </visual>
                <self_collide>0</self_collide>
                <enable_wind>0</enable_wind>
                <kinematic>0</kinematic>
                <gravity>1</gravity>
            </link>
        </model>
    </sdf>
    """


def get_rectangle(x: float, y: float, z: float):
    return f"""
    <?xml version="1.0"?>
    <sdf version="1.5">
        <model name='rectangle'>
            <static>1</static>
            <link name='link'>
                <inertial>
                    <mass>1</mass>
                    <inertia>
                        <ixx>0.145833</ixx>
                        <ixy>0</ixy>
                        <ixz>0</ixz>
                        <iyy>0.145833</iyy>
                        <iyz>0</iyz>
                        <izz>0.125</izz>
                    </inertia>
                    <pose frame=''>0 0 0 0 -0 0</pose>
                </inertial>
                <collision name='collision'>
                    <geometry>
                        <box>
                            <size>{x} {y} {z}</size>
                        </box>
                    </geometry>
                </collision>
                <visual name='visual'>
                    <geometry>
                        <box>
                            <size>{x} {y} {z}</size>
                        </box>
                    </geometry>
                    <material>
                        <script>
                            <name>Gazebo/Black</name>
                            <uri>file://media/materials/scripts/gazebo.material</uri>
                        </script>
                    </material>
                </visual>
                <self_collide>0</self_collide>
                <enable_wind>0</enable_wind>
                <kinematic>0</kinematic>
                <gravity>1</gravity>
            </link>
        </model>
    </sdf>
    """


def get_dyn_obstacle(x: float, y: float, z: float):
    # if isinstance(initial_velocity, (float, int)):
    #     initial_velocity = 3 * [initial_velocity]
    # elif initial_velocity is None:
    #     initial_velocity = np_random.uniform(size=3)
    #     initial_velocity = velocity_factor * initial_velocity / np.linalg.norm(initial_velocity)

    # """
    #   <plugin name="random" filename="libRandomVelocityPlugin.so">
    #     <link>link</link>
    #     <!-- Initial velocity that is applied to the link -->
    #     <initial_velocity>{initial_velocity[0]} {initial_velocity[1]} {initial_velocity[2]}</initial_velocity>
    #     <!-- Scaling factor that is used to compute a new velocity -->
    #     <velocity_factor>{velocity_factor}</velocity_factor>
    #     <!-- Time, in seconds, between new velocities -->
    #     <update_period>{update_period}</update_period>
    #     <min_z>0</min_z>
    #     <max_z>0</max_z>
    #   </plugin>
    # """

    return f"""
    <?xml version="1.0" ?>
    <sdf version="1.6">
        <model name="DynamicObstacle">
          <pose>0 0 0.15 0 0 0</pose>
          <link name="link">
        <gravity>1</gravity>
            <inertial>
          <inertia>
            <ixx>0.17</ixx>
                <ixy>0.00</ixy>
                <ixz>0.00</ixz>
                <iyy>0.17</iyy>
                <iyz>0.00</iyz>
                <izz>0.17</izz>
              </inertia>
              <mass>100</mass>
            </inertial>
            <kinematic>false</kinematic>
            <collision name="collision">
              <max_contacts>0</max_contacts>
              <geometry>
                <box>
                  <size>{x} {y} {z}</size>
                </box>
              </geometry>
          <surface>
                <friction>
                  <ode>
                    <mu>0.0</mu>
                    <mu2>0.0</mu2>
                  </ode>
                </friction>
                <bounce>
                  <restitution_coefficient>0.0</restitution_coefficient>
                  <threshold>0.0</threshold>
              </bounce>
          <contact>
                <ode>
                  <kp>2000000000</kp>
              <kd>1</kd>
                  <max_vel>0.01</max_vel>
                  <min_depth>0.00</min_depth>
                </ode>
              </contact>
              </surface>
            </collision>
            <visual name="visual">
              <geometry>
                <box>
                  <size>0.3 0.3 0.3</size>
                </box>
              </geometry>
            </visual>
          </link>
        </model>
    </sdf>
    """


@dataclass
class GazeboObject:
    database_name: str
    x: float
    y: float
    z: float
    geometry: ObjectGeometry = ObjectGeometry.UNKNOWN
    getter_fn: Callable = None


class WorldObjects:
    """
    name in gazebo database, x, y, z dimensions
    see e.g. sdfs here: https://github.com/osrf/gazebo_models/blob/master/table/model.sdf
    """
    # NOTE: not all sizes correct yet!
    coke_can = GazeboObject('coke_can', 0.05, 0.05, 0.1)
    table = GazeboObject("table", 1.5, 0.8, 1.0, ObjectGeometry.RECTANGLE)
    kitchen_table = GazeboObject("kitchen_table", 0.68, 1.13, 0.68, ObjectGeometry.RECTANGLE)
    # our own, non-database objects
    muesli2 = GazeboObject('muesli2', 0.05, 0.15, 0.23, ObjectGeometry.RECTANGLE)
    kallax2 = GazeboObject('Kallax2', 0.415, 0.39, 0.65, ObjectGeometry.RECTANGLE)
    kallax = GazeboObject('Kallax', 0.415, 0.39, 0.65, ObjectGeometry.RECTANGLE)
    kallaxDrawer1 = GazeboObject('KallaxDrawer1', 0.415, 0.39, 0.65, ObjectGeometry.RECTANGLE)
    # ATM CRASHING GAZEBO
    tim_bowl = GazeboObject('tim_bowl', 0.05, 0.05, 0.1, ObjectGeometry.RECTANGLE)
    reemc_table_low = GazeboObject('reemc_table_low', 0.75, 0.75, 0.41, ObjectGeometry.RECTANGLE)
    hinged_doorHall78 = GazeboObject('hinged_doorHall78', 0.9144, 0.04445, 2.032, ObjectGeometry.UNKNOWN)
    unit_cylinder = GazeboObject('unit_cylinder', 1., 1., 1., ObjectGeometry.CYLINDER, get_cylinder)
    cylinder_inpath = GazeboObject('cylinder_inpath', 0.375, 0.375, 0.5, ObjectGeometry.CYLINDER, get_cylinder)
    dyn_obstacle_sim = GazeboObject("rectangle", x=0.3, y=0.3, z=2.0, geometry=ObjectGeometry.CYLINDER)
    dyn_obstacle_gazebo = GazeboObject('dynamic_obstacle', 0.3, 0.3, 0.35, ObjectGeometry.RECTANGLE, getter_fn=get_dyn_obstacle)

class SpawnObject:
    def __init__(self,
                 name: str,
                 world_object: GazeboObject,
                 pose: Union[Pose, Callable],
                 frame: str = DEFAULT_FRAME):
        self.name = name
        self.world_object = world_object
        self.pose = pose
        self.frame = frame


class DynObstacle(SpawnObject):
    def __init__(self,
                 name: str,
                 world_object: GazeboObject,
                 pose: Union[Pose, Callable],
                 map_W_meter,
                 map_H_meter,
                 np_random,
                 goal_dist: Tuple[float, float] = (0.25, 5.0),
                 velocity: float = None,
                 goal_center: Tuple[float, float] = (0.0, 0.0)):
        assert "dyn_obstacle" in name, f"Name has to contain 'dyn_obstacle', otherwise move_obstacles.py won't move it in gazebo"
        super().__init__(name=name, world_object=world_object, pose=pose)
        self.np_random = np_random
        x = pose.position.x
        y = pose.position.y
        # if (x is None) or (y is None):
        #     x, y = 0., 0.
        #     border_offset = 2
        #     # do not spawn directly in the center where the robot will spawn
        #     while (abs(x) < 0.7) or (abs(y) < 0.7):
        #         x = self.np_random.uniform(-map_W_meter / 2 + border_offset, map_W_meter / 2 - border_offset)
        #         y = self.np_random.uniform(-map_H_meter / 2 + border_offset, map_H_meter / 2 - border_offset)
        self._current_xyz = np.array([x, y, 0.0])

        if velocity is None:
            velocity = self.np_random.uniform(0.1, 0.15)
        self.velocity = velocity
        self.goal_dist = goal_dist
        self.map_W_meter, self.map_H_meter = map_W_meter, map_H_meter
        self._goal_center = np.array(goal_center)
        self._goal = self.sample_goal()

    def sample_goal(self):
        goal = np.array(sample_circle_goal(base_tf=pose_to_list(self.pose), goal_dist_rng=self.goal_dist, goal_height_rng=(0.0, 0.0),
                                           map_width=self.map_W_meter, map_height=self.map_H_meter, np_random=self.np_random))
        goal[:2] += self._goal_center
        return goal

    def move(self, dt: float):
        vec_to_goal = self._goal[:3] - self._current_xyz
        dist_to_goal = np.linalg.norm(vec_to_goal)
        self._current_xyz += dt * self.velocity * (vec_to_goal / (dist_to_goal + SMALL_NUMBER))
        self.pose = Pose(Point(*self._current_xyz), Quaternion(0., 0., 0., 1.))
        if dist_to_goal < 0.05:
            self._goal = self.sample_goal()


class SimulatorAPI:
    def __init__(self):
        pass

    def spawn_scene_objects(self, spawn_objects: Iterable[SpawnObject]):
        for obj in spawn_objects:
            self._spawn_model(obj.name, obj.world_object, obj.pose, obj.frame)

    def _spawn_model(self, name: str, obj: GazeboObject, pose: Pose, frame=DEFAULT_FRAME):
        raise NotImplementedError()

    def get_model(self, name: str, relative_entity_name: str):
        raise NotImplementedError()

    def set_model(self, name: str, pose: Pose):
        raise NotImplementedError()

    def delete_model(self, name: str):
        raise NotImplementedError()

    def delete_all_spawned(self):
        raise NotImplementedError()

    def clear(self):
        raise NotImplementedError()

    @staticmethod
    def get_link_state(link_name: str):
        raise NotImplementedError()

    def set_joint_angle(self, model_name: str, joint_names: list, angles: list):
        raise NotImplementedError()


class DummySimulatorAPI(SimulatorAPI):
    """
    Dummy so we can use the same API for the analytical env.
    Atm not used for any practical purpose as we can already visualise obstacles through the floorplan.
    """

    def __init__(self, frame_id: str):
        super().__init__()
        self._frame_id = frame_id
        # requires to ahve a rospy node initialised! (Not a given for ray remote actors).
        self._verbose = False

    def spawn_scene_objects(self, spawn_objects: Iterable[SpawnObject]):
        if not self._verbose:
            return

        for obj in spawn_objects:
            if obj.world_object.geometry == ObjectGeometry.RECTANGLE:
                publish_marker("obstacles",
                               marker_pose=obj.pose,
                               marker_scale=[obj.world_object.x, obj.world_object.y, obj.world_object.z],
                               marker_id=random.randint(1000, 100000),
                               frame_id=self._frame_id,
                               geometry="cube")
            elif obj.world_object.geometry == ObjectGeometry.UNKNOWN:
                pass
            else:
                raise NotImplementedError(obj.world_object.geometry)

    def delete_all_spawned(self):
        if self._verbose:
            clear_all_markers(self._frame_id)

    def clear(self):
        if self._verbose:
            clear_all_markers(self._frame_id)


class GazeboAPI(SimulatorAPI):
    def get_model_template(self, obj: GazeboObject):
        if obj.getter_fn is not None:
            return obj.getter_fn(obj.x, obj.y, obj.z)
        else:
            return f"""\
            <sdf version="1.6">
                <world name="default">
                    <include>
                        <uri>model://{obj.database_name}</uri>
                    </include>
                </world>
            </sdf>"""

    def __init__(self, time_out=10):
        super().__init__()
        # https://answers.ros.org/question/246419/gazebo-spawn_model-from-py-source-code/
        # https://github.com/ros-simulation/gazebo_ros_pkgs/pull/948/files
        print("Waiting for gazebo services...")
        rospy.wait_for_service("gazebo/delete_model", timeout=time_out)
        rospy.wait_for_service("gazebo/spawn_sdf_model", timeout=time_out)
        rospy.wait_for_service('/gazebo/get_model_state', timeout=time_out)
        rospy.wait_for_service('/gazebo/set_model_state', timeout=time_out)
        rospy.wait_for_service('/gazebo/reset_simulation', timeout=time_out)
        rospy.wait_for_service('/gazebo/reset_world', timeout=time_out)
        rospy.wait_for_service('/gazebo/pause_physics', timeout=time_out)
        rospy.wait_for_service('/gazebo/unpause_physics', timeout=time_out)
        self._delete_model_srv = rospy.ServiceProxy("gazebo/delete_model", DeleteModel)
        self._spawn_model_srv = rospy.ServiceProxy("gazebo/spawn_sdf_model", SpawnModel)
        self._get_model_srv = rospy.ServiceProxy("gazebo/get_model_state", GetModelState)
        self._set_model_srv = rospy.ServiceProxy("gazebo/set_model_state", SetModelState)
        self._reset_simulation_srv = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self._reset_world_srv = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        self._pause_physics_srv = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self._unpause_physics_srv = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        print("All gazebo services reached")

        self._spawned_models = []

    def _pause_physics(self):
        return self._pause_physics_srv()

    def _unpause_physics(self):
        return self._unpause_physics_srv()

    def spawn_scene_objects(self, spawn_objects: Iterable[SpawnObject]):
        self._pause_physics()
        for obj in spawn_objects:
            self._spawn_model(obj.name, obj.world_object, obj.pose, obj.frame)
        self._unpause_physics()

    def _spawn_model(self, name: str, obj: GazeboObject, pose: Pose, frame=DEFAULT_FRAME):
        # with open("$GAZEBO_MODEL_PATH/product_0/model.sdf", "r") as f:
        #     product_xml = f.read()
        product_xml = self.get_model_template(obj)

        # orient = Quaternion(tf.transformations.quaternion_from_euler(0, 0, 0))
        self._spawned_models.append(name)
        info = self._spawn_model_srv(name, product_xml, "", pose, frame)

        while not self.get_model(name, DEFAULT_FRAME).success:
            info = self._spawn_model_srv(name, product_xml, "", pose, frame)
            time.sleep(0.1)
            print(f"Waiting for model {name} to spawn in gazebo")

        return info

    def get_model(self, name: str, relative_entity_name: str):
        return self._get_model_srv(name, relative_entity_name)

    # def set_model(self, name: str, pose: Pose):
    #     state_msg = ModelState()
    #     state_msg.model_name = name
    #     state_msg.pose = pose
    #     # self.pause_physics()
    #     info = self._set_model_srv(state_msg)
    #     time.sleep(1.0)
    #     # self.unpause_physics()
    #     return info

    def delete_model(self, name: str):
        self._delete_model_srv(name)
        # while self.get_model(name, DEFAULT_FRAME).success:
        #     rospy.loginfo(f"Waiting to delete model {name}")
        #     self._delete_model_srv(name)
        #     time.sleep(0.1)
        self._spawned_models.remove(name)

    def delete_all_spawned(self):
        self._pause_physics()
        while self._spawned_models:
            m = self._spawned_models[0]
            self.delete_model(m)
        self._unpause_physics()
        time.sleep(0.2)

    def reset_simulation(self):
        """'reset world' in gui"""
        print("RESET WORLD MIGHT NOT WORK CORRECTLY ATM")
        return self._reset_simulation_srv()

    def reset_world(self):
        """'reset model poses' in gui"""
        return self._reset_world_srv()

    def clear(self):
        self._pause_physics()
        self.delete_all_spawned()
        self._unpause_physics()
        # self.reset_world()
        self._delete_model_srv.close()
        self._spawn_model_srv.close()
        self._get_model_srv.close()
        self._set_model_srv.close()
        self._reset_simulation_srv.close()

    # @staticmethod
    # def set_link_state(link_name: str, pose: Pose):
    #     msg = LinkState()
    #     msg.link_name = link_name
    #     msg.pose = pose
    #     set_link_state_srv = rospy.ServiceProxy("gazebo/set_link_state", SetLinkState)
    #     return set_link_state_srv(msg)

    @staticmethod
    def get_link_state(link_name: str):
        """NOTE: need to initialise a rospy node first, o/w will hang here!"""
        msg = rospy.wait_for_message("/gazebo/link_states", LinkStates, timeout=10)
        return msg.pose[msg.name.index(link_name)]

    def set_joint_angle(self, model_name: str, joint_names: list, angles: list):
        self._pause_physics()
        # assert 0 <= angle <= np.pi / 2, angle
        assert len(joint_names) == len(angles)
        set_model_configuration_srv = rospy.ServiceProxy("gazebo/set_model_configuration", SetModelConfiguration)
        req = SetModelConfigurationRequest()
        req.model_name = model_name
        # req.urdf_param_name = 'robot_description'
        req.joint_names = joint_names  # list
        req.joint_positions = angles  # list
        res = set_model_configuration_srv(req)
        assert res.success, res
        self._unpause_physics()
        return res
