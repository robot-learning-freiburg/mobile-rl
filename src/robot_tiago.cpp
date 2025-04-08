//
// Created by honerkam on 10/1/21.
//

#include <geometry_msgs/Twist.h>
#include <modulation_rl/robot_tiago.h>

using namespace std;
namespace py = pybind11;

RobotTiago::RobotTiago(uint32_t seed,
                   const string &node_handle_name,
                   bool init_controllers,
                   const string &world_type,
                   const RoboConf &robot_conf,
                   double bioik_center_joints_weight,
                   double bioik_avoid_joint_limits_weight,
                   double bioik_regularization_weight,
                   const string &bioik_regularization_type) :
        RobotEnv(seed,
                 node_handle_name,
                 init_controllers,
                 world_type,
                 robot_conf,
                 bioik_center_joints_weight,
                 bioik_avoid_joint_limits_weight,
                 bioik_regularization_weight,
                 bioik_regularization_type) {
    setup();
}

void RobotTiago::setup() {
    if (init_controllers_) {
        arm_client_.reset(new TrajClientTiago("/arm_controller/follow_joint_trajectory"));
        while (!arm_client_->waitForServer(ros::Duration(5.0))) {
            py::print("Waiting for the arm_controller/follow_joint_trajectory action server to come up");
        }

        torso_client_.reset(new TrajClientTiago("/torso_controller/follow_joint_trajectory"));
        while (!torso_client_->waitForServer(ros::Duration(5.0))) {
            py::print("Waiting for the torso_controller/follow_joint_trajectory action server to come up");
        }

        vector<string> joint_names = joint_model_group_->getVariableNames();
        arm_goal_.trajectory.joint_names.resize(joint_names.size() - 1);
        arm_goal_.trajectory.points.resize(1);
        arm_goal_.trajectory.points[0].positions.resize(joint_names.size() - 1);

        torso_goal_.trajectory.joint_names.resize(1);
        torso_goal_.trajectory.points.resize(1);
        torso_goal_.trajectory.points[0].positions.resize(1);

        // move_group_arm_torso_ = new moveit::planning_interface::MoveGroupInterface(joint_model_group_name_);
        // move_group_arm_torso_->setPlannerId("SBLkConfigDefault");
        // move_group_arm_torso_->setMaxVelocityScalingFactor(1.0);
        // //move_group_arm_torso_->setMaxAccelerationScalingFactor(0.05);

        gripper_client_.reset(new TrajClientTiago("/gripper_controller/follow_joint_trajectory"));
        while (!gripper_client_->waitForServer(ros::Duration(5.0))) {
            py::print("Waiting for the gripper_controller/follow_joint_trajectory action server to come up");
        }

        // somehow the controller manager expects the hash from a pr2_mechanism_msgs::SwitchController, not controller_manager_msgs::SwitchController
        // switch_controller_client_ = nh_->serviceClient<pr2_mechanism_msgs::SwitchController>("/controller_manager/switch_controller");
    }
}

tf::Transform RobotTiago::calcDesiredBaseTf(const tf::Transform &base_tf,
                                          const tf::Vector3 &base_translation_relative,
                                          const double base_rotation_relative,
                                          const double dt) {
    return myutils::calcDesiredBaseTfDiffDrive(base_tf,
                                             base_translation_relative,
                                             base_rotation_relative,
                                             dt);
}

void RobotTiago::calcDesiredBaseCommand(const tf::Transform &current_base_tf,
                                      const tf::Transform &desired_base_tf,
                                      const double dt,
                                      tf::Vector3 &base_translation_per_second,
                                      double &base_rotation_per_second){
  return myutils::calcDesiredBaseCommandDiffDrive(current_base_tf,
                                                  desired_base_tf,
                                                  dt,
                                                  base_translation_per_second,
                                                  base_rotation_per_second);
}


void RobotTiago::sendArmCommands(const vector<vector<double>> &joint_values, vector<double> &timestamps) {
    vector<string> joint_names = joint_model_group_->getVariableNames();

    // plan gripper and torso
    // joint_names_ for group arm_torso = [torso_lift_joint, arm_1_joint, arm_2_joint, arm_3_joint, arm_4_joint, arm_5_joint, arm_6_joint, arm_7_joint]
    arm_goal_.trajectory.points.resize(joint_values.size());
    for (int k = 0; k < joint_values.size(); k++) {
        arm_goal_.trajectory.points[k].positions.resize(joint_names.size() - 1);
        torso_goal_.trajectory.points[k].positions.resize(1);
        arm_goal_.trajectory.points[k].time_from_start = ros::Duration(timestamps[k]);
        torso_goal_.trajectory.points[k].time_from_start = ros::Duration(timestamps[k]);

        int arm_idx = 0;
        for (int i = 0; i < joint_names.size(); i++) {
            if (joint_names[i] == robot_config_.torso_joint_name) {
                torso_goal_.trajectory.joint_names[0] = joint_names[i];
                torso_goal_.trajectory.points[k].positions[0] = joint_values[k][i];
            } else {
                arm_goal_.trajectory.joint_names[arm_idx] = joint_names[i];
                arm_goal_.trajectory.points[k].positions[arm_idx] = joint_values[k][i];
                arm_idx += 1;
            }
        }
    }
    // When to start the trajectory
    arm_goal_.trajectory.header.stamp = ros::Time::now() + ros::Duration(0.0);
    torso_goal_.trajectory.header.stamp = ros::Time::now() + ros::Duration(0.0);
    // send off commands to run in parallel
    arm_client_->sendGoal(arm_goal_);
    torso_client_->sendGoal(torso_goal_);
}

bool RobotTiago::getArmSuccess() {
    bool success = true;
    torso_client_->waitForResult(ros::Duration(10.0));
    if (torso_client_->getState() != actionlib::SimpleClientGoalState::SUCCEEDED) {
        py::print("The torso_client_ failed.");
        success = false;
    }
    arm_client_->waitForResult(ros::Duration(10.0));
    if (arm_client_->getState() != actionlib::SimpleClientGoalState::SUCCEEDED) {
        py::print("The arm_client_ failed.");
        // throw runtime_error("The arm_client_ failed.");
        success &= false;
    }
    return success;
}

void RobotTiago::openGripper(double position, bool wait_for_result) {
    control_msgs::FollowJointTrajectoryGoal goal;

    // The joint names, which apply to all waypoints
    goal.trajectory.joint_names.push_back("gripper_left_finger_joint");
    goal.trajectory.joint_names.push_back("gripper_right_finger_joint");
    int n = goal.trajectory.joint_names.size();

    // Two waypoints in this goal trajectory
    goal.trajectory.points.resize(1);

    // First trajectory point
    // Positions
    int index = 0;
    goal.trajectory.points[index].positions.resize(n);
    goal.trajectory.points[index].positions[0] = position / 2;
    goal.trajectory.points[index].positions[1] = position / 2;
    // Velocities
    goal.trajectory.points[index].velocities.resize(n);
    for (int j = 0; j < n; ++j) {
        goal.trajectory.points[index].velocities[j] = 0.0;
    }
    goal.trajectory.header.stamp = ros::Time::now() + ros::Duration(0.0);
    goal.trajectory.points[index].time_from_start = ros::Duration(2.0);

    gripper_client_->sendGoal(goal);

    if (wait_for_result) {
        gripper_client_->waitForResult(ros::Duration(5.0));
        if (gripper_client_->getState() != actionlib::SimpleClientGoalState::SUCCEEDED)
            py::print("The gripper client failed.");
        ros::Duration(0.1).sleep();
    }
}

void RobotTiago::closeGripper(double position, bool wait_for_result) {
    openGripper(position, wait_for_result);
}
