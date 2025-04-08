//
// Created by daniel on 30.09.21.
//

#ifndef MODULATION_RL_VALIDITY_FUN_H
#define MODULATION_RL_VALIDITY_FUN_H

#include <moveit/planning_scene_monitor/planning_scene_monitor.h>
#include <moveit/robot_state/robot_state.h>
#include <tf/tf.h>

namespace validity_fun {
    bool validityCallbackFnSelfCollision(planning_scene::PlanningScenePtr &planning_scene,
                                         bool *in_selfcollision,
                                         // const kinematics_constraint_aware::KinematicsRequest &request,
                                         // kinematics_constraint_aware::KinematicsResponse &response,
                                         const robot_state::RobotStatePtr &kinematic_state,
                                         const robot_state::JointModelGroup *joint_model_group,
                                         const double *joint_group_variable_values
                                         // const std::vector<double> &joint_group_variable_values
    );
    bool validityCallbackFnIkSlack(planning_scene::PlanningScenePtr &planning_scene,
                                   bool *in_selfcollision,
                                   const tf::Transform &goal_pose,
                                   const double ik_slack_dist,
                                   const double ik_slack_rot_dist,
                                   const std::string &global_link_transform,
                                   const robot_state::RobotStatePtr &kinematic_state,
                                   const robot_state::JointModelGroup *joint_model_group,
                                   const double *joint_group_variable_values);
}

#endif  // MODULATION_RL_VALIDITY_FUN_H
