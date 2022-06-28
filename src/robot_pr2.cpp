//
// Created by honerkam on 9/30/21.
//

#include <geometry_msgs/Twist.h>
#include <modulation_rl/robot_pr2.h>
#include <pr2_mechanism_msgs/SwitchController.h>
#include <pr2_msgs/SetPeriodicCmd.h>
#include <pybind11/pybind11.h>

using namespace std;
namespace py = pybind11;

RobotPR2::RobotPR2(uint32_t seed,
               const string& node_handle_name,
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
    if (init_controllers_) {
        arm_client_ = new TrajClientPR2("r_arm_controller/joint_trajectory_action", true);
        while (!arm_client_->waitForServer(ros::Duration(5.0))) {
            py::print("Waiting for the r_arm_controller/joint_trajectory_action action server to come up");
        }

        arm_client_left_ = new TrajClientPR2("l_arm_controller/joint_trajectory_action", true);
        while (!arm_client_->waitForServer(ros::Duration(5.0))) {
            py::print("Waiting for the l_arm_controller/joint_trajectory_action action server to come up");
        }

        // switch_controller_client_ = nh_->serviceClient<pr2_mechanism_msgs::SwitchController>("/pr2_controller_manager/switch_controller");
        gripper_client_ = new GripperClientPR2("r_gripper_controller/gripper_action", true);
        while (!gripper_client_->waitForServer(ros::Duration(5.0))) {
            py::print("Waiting for the r_gripper_controller/gripper_action action server to come up");
        }

        torso_client_ = new TorsoClient("torso_controller/position_joint_action", true);
        while (!torso_client_->waitForServer(ros::Duration(5.0))) {
            py::print("Waiting for the torso_controller/position_joint_action action server to come up");
        }

        const robot_state::JointModelGroup* arm_group = kinematic_state_->getJointModelGroup("right_arm");
        arm_goal_.trajectory.joint_names.resize(arm_group->getVariableNames().size());
        arm_goal_.trajectory.joint_names = arm_group->getVariableNames();

        const robot_state::JointModelGroup* arm_group_left = kinematic_state_->getJointModelGroup("left_arm");
        arm_goal_left_.trajectory.joint_names.resize(arm_group_left->getVariableNames().size());
        arm_goal_left_.trajectory.joint_names = arm_group_left->getVariableNames();

        for (int i = 0; i < joint_model_group_->getVariableNames().size(); i++) {
            if (joint_model_group_->getVariableNames()[i] == robot_config_.torso_joint_name){
                torso_joint_index_ = i;
            }
        }

        // start tilt laser movement: http://wiki.ros.org/pr2_simulator/Tutorials/BasicPR2Controls
        pr2_msgs::SetPeriodicCmd periodic_cmd;
        periodic_cmd.request.command.header.stamp = ros::Time::now();
        periodic_cmd.request.command.profile = "linear";
        periodic_cmd.request.command.period = 3;
        periodic_cmd.request.command.amplitude = 1;
        periodic_cmd.request.command.offset = 0;
        if (!ros::service::call("laser_tilt_controller/set_periodic_cmd", periodic_cmd)) {
            throw runtime_error("Could not start tilt laser service");
        }
    }
}

tf::Transform RobotPR2::calcDesiredBaseTf(const tf::Transform &base_tf,
                                        const tf::Vector3 &base_translation_relative,
                                        const double base_rotation_relative,
                                        const double dt) {
    return myutils::calcDesiredBaseTfOmni(base_tf,
                                        base_translation_relative,
                                        base_rotation_relative,
                                        dt);
}

void RobotPR2::calcDesiredBaseCommand(const tf::Transform &current_base_tf,
                                      const tf::Transform &desired_base_tf,
                                      const double dt,
                                      tf::Vector3 &base_translation_per_second,
                                      double &base_rotation_per_second){
  return myutils::calcDesiredBaseCommandOmni(current_base_tf,
                                             desired_base_tf,
                                             dt,
                                             base_translation_per_second,
                                             base_rotation_per_second);
}

void RobotPR2::sendArmCommands(const vector<vector<double>> &joint_values, vector<double> &timestamps){
    arm_goal_.trajectory.points.resize(joint_values.size());
    pr2_controllers_msgs::SingleJointPositionGoal torso_goal;

    for (int i = 0; i < joint_values.size(); i++) {
        arm_goal_.trajectory.points[i].positions.clear();

        for (int j = 0; j < joint_values[i].size(); j++) {
            if (j == torso_joint_index_) {
                if (i == 0) {
                    torso_goal.position = joint_values[i][j];
                    torso_goal.min_duration = ros::Duration(timestamps[i]);
                }
            } else {
                arm_goal_.trajectory.points[i].positions.push_back(joint_values[i][j]);
            }
        }
        arm_goal_.trajectory.points[i].time_from_start = ros::Duration(timestamps[i]);
    }
    // When to start the trajectory
    arm_goal_.trajectory.header.stamp = ros::Time::now() + ros::Duration(0.0);
    // To be reached x seconds after starting along the trajectory
    // send off commands to run in parallel
    arm_client_->sendGoal(arm_goal_);

    if (torso_joint_index_ >= 0){
        torso_client_->sendGoal(torso_goal);
    }
}

bool RobotPR2::getArmSuccess() {
    bool success = true;
    if (torso_joint_index_ >= 0) {
        torso_client_->waitForResult(ros::Duration(10.0));
        actionlib::SimpleClientGoalState torso_result = torso_client_->getState();
        if (torso_client_->getState() != actionlib::SimpleClientGoalState::SUCCEEDED) {
            py::print("The torso_client_ failed with error code", torso_result.toString().c_str());
            success = false;
        }
    }

    arm_client_->waitForResult(ros::Duration(10.0));
    actionlib::SimpleClientGoalState result = arm_client_->getState();
    if (arm_client_->getState() != actionlib::SimpleClientGoalState::SUCCEEDED) {
        py::print("The arm_client_ failed with error code", result.toString().c_str());
        success = false;
    }
    return success;
}

void RobotPR2::setLeftArm(vector<double> &joint_values) {
    if (joint_values.empty()) {
        tf::Transform desired_left_gripper_tf_world = getGlobalLinkTransform("l_wrist_roll_link");
        desired_left_gripper_tf_world.setOrigin(tf::Vector3(desired_left_gripper_tf_world.getOrigin().x(),
                                                            desired_left_gripper_tf_world.getOrigin().y(),
                                                            std::max(desired_left_gripper_tf_world.getOrigin().z() - 0.1, 0.379)));

        Eigen::Isometry3d state;
        tf::poseTFToEigen(desired_left_gripper_tf_world, state);
        const Eigen::Isometry3d &desired_gripper_eigen = state;

        bool success = kinematic_state_->setFromIK(kinematic_state_->getJointModelGroup("left_arm"), desired_gripper_eigen, 0.001);
        if (!success) {
            return;
        }
        kinematic_state_->copyJointGroupPositions("left_arm", joint_values);
    } else {
        if (joint_values.size() != kinematic_state_->getJointModelGroup("left_arm")->getVariableCount()){
            throw std::runtime_error("Wrong number of joint values");
        }
        kinematic_state_->setJointGroupPositions(kinematic_state_->getJointModelGroup("left_arm"), joint_values);
    }
    if (!isAnalyticalWorld()) {
        arm_goal_left_.trajectory.points.resize(1);
        arm_goal_left_.trajectory.points[0].positions = joint_values;
        arm_goal_left_.trajectory.points[0].time_from_start = ros::Duration(0.02);
        // When to start the trajectory
        arm_goal_left_.trajectory.header.stamp = ros::Time::now() + ros::Duration(0.0);
        // To be reached x seconds after starting along the trajectory
        // send off commands to run in parallel
        arm_client_left_->sendGoal(arm_goal_left_);
    }
}

RobotObs RobotPR2::step(const std::vector<double>& base_translation_relative_list,
              double base_rotation_relative,
              const std::vector<double> &desired_gripper_tf_list,
              const RobotObs &prev_robot_obs,
              double dt_exec,
              bool execute_cmds,
              bool learn_torso,
              double delta_torso,
              const std::string &execute_style,
              const std::string &perception_style,
              const std::vector<double> &joint_values_action){
    vector<double> left_arm_joints;
    setLeftArm(left_arm_joints);
    return RobotEnv::step(base_translation_relative_list,
                          base_rotation_relative,
                          desired_gripper_tf_list,
                          prev_robot_obs,
                          dt_exec,
                          execute_cmds,
                          learn_torso,
                          delta_torso,
                          execute_style,
                          perception_style,
                          joint_values_action);
}

RobotObs RobotPR2::reset(const std::map<std::string, double> &initial_joint_values, bool do_close_gripper){
    // reset to initial values so that we don't draw different random values for the right arm due to collisions of left arm and base for different torso heights
    vector<double> left_arm_joints{ 0.06024, 1.248526, 1.789070, -1.50, -1.7343417, -0.0962141, -0.0864407};
    setLeftArm(left_arm_joints);
    RobotObs robot_obs = RobotEnv::reset(initial_joint_values, do_close_gripper);
    vector<double> jv;
    setLeftArm(jv);
    return robot_obs;
}

// http://library.isr.ist.utl.pt/docs/roswiki/pr2_controllers(2f)Tutorials(2f)Moving(20)the(20)gripper.html
void RobotPR2::moveGripper(double position, double effort, bool wait_for_result) {
    pr2_controllers_msgs::Pr2GripperCommandGoal goal;
    goal.command.position = position;
    goal.command.max_effort = effort;
    gripper_client_->sendGoal(goal);

    if (wait_for_result) {
        gripper_client_->waitForResult(ros::Duration(5.0));
        if (gripper_client_->getState() != actionlib::SimpleClientGoalState::SUCCEEDED)
            py::print("The gripper failed.");
    }
}

void RobotPR2::openGripper(double position, bool wait_for_result) {
    moveGripper(position, -1.0, wait_for_result);  // Do not limit effort (negative)
}

void RobotPR2::closeGripper(double position, bool wait_for_result) {
    moveGripper(position, 200.0, wait_for_result);  // Close gently
}
