//
// Created by daniel on 30.09.21.
//
#include <tf_conversions/tf_eigen.h>
#include "modulation_rl/myutils.h"
#include "modulation_rl/validity_fun.h"


////Callback for collision checking in ik search//////////////////////
namespace validity_fun {
    bool validityCallbackFnSelfCollision(planning_scene::PlanningScenePtr &planning_scene,
                                         bool *in_selfcollision,
                                         const robot_state::RobotStatePtr &kinematic_state,
                                         const robot_state::JointModelGroup *joint_model_group,
                                         const double *joint_group_variable_values) {
        kinematic_state->setJointGroupPositions(joint_model_group, joint_group_variable_values);
        // Now check for collisions
        collision_detection::CollisionRequest collision_request;
        collision_request.group_name = joint_model_group->getName();
        collision_detection::CollisionResult collision_result;
        // collision_detection::AllowedCollisionMatrix acm = planning_scene->getAllowedCollisionMatrix();
        // planning_scene->getCurrentStateNonConst().update();
        planning_scene->checkCollision(collision_request, collision_result, *kinematic_state);
        // planning_scene->checkSelfCollision(collision_request, collision_result, *kinematic_state);
         *in_selfcollision = collision_result.collision;

        if (collision_result.collision) {
            // ROS_INFO("IK solution is in collision!");
            return false;
        }
        return true;
    }

    bool validityCallbackFnIkSlack(planning_scene::PlanningScenePtr &planning_scene,
                                   bool *in_selfcollision,
                                   const tf::Transform &goal_pose,
                                   const double ik_slack_dist,
                                   const double ik_slack_rot_dist,
                                   const std::string &global_link_transform,
                                   const robot_state::RobotStatePtr &kinematic_state,
                                   const robot_state::JointModelGroup *joint_model_group,
                                   const double *joint_group_variable_values) {
        // self collision check
        bool valid = validityCallbackFnSelfCollision(planning_scene,
                                                     in_selfcollision,
                                                     kinematic_state,
                                                     joint_model_group,
                                                     joint_group_variable_values);

        // check that not too far from desired
        kinematic_state->setJointGroupPositions(joint_model_group, joint_group_variable_values);

        tf::Transform fwd_ik_gripper;
        tf::transformEigenToTF(kinematic_state->getGlobalLinkTransform(global_link_transform), fwd_ik_gripper);
        double dist_solution_desired = (fwd_ik_gripper.getOrigin() - goal_pose.getOrigin()).length();
        double rot_dist_solution_desired = myutils::calcRotDist(fwd_ik_gripper, goal_pose);
        valid &= (dist_solution_desired < ik_slack_dist) && (rot_dist_solution_desired < ik_slack_rot_dist);

        return valid;
    }
}  // namespace validity_fun