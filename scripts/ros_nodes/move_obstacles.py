#!/usr/bin/env python
import rospy
from gazebo_msgs.msg import ModelState, ModelStates
from std_msgs.msg import Bool
import numpy as np
from std_srvs.srv import Empty
import time
import math

# NOTE: should match modulation.envs.tasks.MOVE_OBSTACLES_ENABLED_TOPIC. Cannot import due to python2/3 incompatibilities
MOVE_OBSTACLES_ENABLED_TOPIC = "pause_move_obstacles"


class CB:
    def __init__(self):
        self.msg = None

    def cb(self, msg):
        self.msg = msg


class EnabledCallback:
    enabled = False

    def callback(self, msg):
        self.enabled = msg.data


class Mover(object):
    def __init__(self):
        self.pub_model = rospy.Publisher('gazebo/set_model_state', ModelState, queue_size=50)
        self._pause_physics_srv = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self._unpause_physics_srv = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)

        self.cb = CB()
        self.sub = rospy.Subscriber('gazebo/model_states', ModelStates, callback=self.cb.cb, queue_size=1)
        self.enabled_cb = EnabledCallback()
        self.enabled_sub = rospy.topics.Subscriber(MOVE_OBSTACLES_ENABLED_TOPIC, Bool, self.enabled_cb.callback, queue_size=10)
        self.prev_enabled = self.enabled_cb.enabled

    @staticmethod
    def draw_velocity():
        norm = 0
        while norm < 1e-4:
            xy = np.random.uniform(-1, 1, 2)
            norm = np.linalg.norm(xy)

        magnitude = np.random.uniform(0.1, 0.15)
        return magnitude * xy / norm

    def move(self):
        # model_states = rospy.wait_for_message('gazebo/model_states', ModelStates)
        continuously_disabled = not self.enabled_cb.enabled and (self.enabled_cb.enabled == self.prev_enabled)
        if (self.cb.msg is None) or continuously_disabled:
            return
        else:
            model_states = self.cb.msg
        dyn_obstacle_idx = [i for i in range(len(model_states.name)) if "dyn_obstacle" in model_states.name[i]]
        obstacle = ModelState()

        if len(dyn_obstacle_idx):
            self._pause_physics_srv()
            time.sleep(0.1)

            for i in dyn_obstacle_idx:
                # print(model_states.name[i])
                if not self.enabled_cb.enabled and (self.enabled_cb.enabled != self.prev_enabled):
                    xy_velocity = [0.0, 0.0]
                else:
                    xy_velocity = self.draw_velocity()

                obstacle.model_name = model_states.name[i]
                obstacle.pose = model_states.pose[i]
                obstacle.twist.linear.x = xy_velocity[0]
                obstacle.twist.linear.y = xy_velocity[1]
                self.pub_model.publish(obstacle)
            time.sleep(0.1)
            self._unpause_physics_srv()


def pose_to_list(pose):
    return [pose.position.x, pose.position.y, pose.position.z,
            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]


def sample_circle_goal(base_tf, goal_dist_rng, goal_height_rng, map_width, map_height):
    goal_dist = np.random.uniform(goal_dist_rng[0], goal_dist_rng[1])
    goal_orientation = np.random.uniform(0.0, math.pi)

    # sample around the origin, then add the base offset in case we don't start at the origin
    x = goal_dist * math.cos(goal_orientation)
    y = np.random.choice([-1, 1]) * goal_dist * math.sin(goal_orientation)
    z = np.random.uniform(goal_height_rng[0], goal_height_rng[1])

    RPY_goal = np.random.uniform(0, 2 * np.pi, 3)

    half_w = map_width / 2 - 0.5
    half_h = map_height / 2 - 0.5

    return [min(max(base_tf[0] + x, -half_w), half_w),
            min(max(base_tf[1] + y, -half_h), half_h),
            z] + RPY_goal.tolist()


class DynObstacle:
    def __init__(self,
                 pose,
                 map_W_meter,
                 map_H_meter,
                 goal_dist=(0.25, 5.0),
                 velocity=None,
                 goal_center=(0.0, 0.0)):
        if velocity is None:
            velocity = np.random.uniform(0.1, 0.15)
        self.velocity = velocity
        self.goal_dist = goal_dist
        self.map_W_meter, self.map_H_meter = map_W_meter, map_H_meter
        self._goal_center = np.array(goal_center)
        self._goal = self.sample_goal(pose)

    def sample_goal(self, pose):
        goal = np.array(sample_circle_goal(base_tf=pose_to_list(pose), goal_dist_rng=self.goal_dist, goal_height_rng=(0.0, 0.0),
                                           map_width=self.map_W_meter, map_height=self.map_H_meter))
        goal[:2] += self._goal_center
        return goal

    def move(self, new_pose):
        vec_to_goal = self._goal[:3] - [new_pose.position.x, new_pose.position.y, new_pose.position.z]
        dist_to_goal = np.linalg.norm(vec_to_goal)
        if dist_to_goal < 0.5:
            self._goal = self.sample_goal(new_pose)
            vec_to_goal = self._goal[:3] - [new_pose.position.x, new_pose.position.y, new_pose.position.z]
            dist_to_goal = np.linalg.norm(vec_to_goal)
        return self.velocity * (vec_to_goal / (dist_to_goal + 0.01))


class GoalDirectedMover(Mover):
    def __init__(self, goal_max_x, goal_max_y):
        super(GoalDirectedMover, self).__init__()
        self.dyn_obstacles = dict()
        self.goal_max_x = goal_max_x
        self.goal_max_y = goal_max_y

    def move(self):
        # model_states = rospy.wait_for_message('gazebo/model_states', ModelStates)
        continuously_disabled = not self.enabled_cb.enabled and (self.enabled_cb.enabled == self.prev_enabled)
        if (self.cb.msg is None) or continuously_disabled:
            return
        else:
            model_states = self.cb.msg
        dyn_obstacle_idx = [i for i in range(len(model_states.name)) if "dyn_obstacle" in model_states.name[i]]
        obstacle = ModelState()
        if len(dyn_obstacle_idx):
            self._pause_physics_srv()
            time.sleep(0.1)

            for i in dyn_obstacle_idx:
                dyn_obs = self.dyn_obstacles.get(model_states.name[i], None)
                if dyn_obs is None:
                    dyn_obs = DynObstacle(model_states.pose[i],
                                          2 * self.goal_max_x,
                                          2 * self.goal_max_y)
                    self.dyn_obstacles[model_states.name[i]] = dyn_obs
                # print(model_states.name[i])
                if not self.enabled_cb.enabled and (self.enabled_cb.enabled != self.prev_enabled):
                    xy_velocity = [0.0, 0.0]
                else:
                    xy_velocity = dyn_obs.move(model_states.pose[i])

                obstacle.model_name = model_states.name[i]
                obstacle.pose = model_states.pose[i]
                obstacle.twist.linear.x = xy_velocity[0]
                obstacle.twist.linear.y = xy_velocity[1]
                self.pub_model.publish(obstacle)
            time.sleep(0.1)
            self._unpause_physics_srv()


def main():
    rospy.init_node('moving_obstacle')
    rng = rospy.get_param('/moving_obstacle/goal_range', None)
    if rng:
        mover = GoalDirectedMover(rng, rng)
    else:
        mover = Mover()
    change_after_seconds = 4
    rate = rospy.Rate(1. / change_after_seconds)

    while not rospy.is_shutdown():
        mover.move()
        rate.sleep()


if __name__ == '__main__':
    main()
