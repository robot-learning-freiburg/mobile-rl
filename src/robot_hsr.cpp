//
// Created by honerkam on 10/1/21.
//

#include <controller_manager_msgs/ControllerState.h>
#include <controller_manager_msgs/ListControllers.h>
#include <modulation_rl/robot_hsr.h>
#include <boost/bind.hpp>
#include <pybind11/pybind11.h>

using namespace std;
namespace py = pybind11;

RobotHSR::RobotHSR(uint32_t seed,
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


void RobotHSR::setup() {
    if (init_controllers_) {
        std::vector<std::string> controllers_to_await;

        // base_client_ = new TrajClientHSR("/hsrb/omni_base_controller/follow_joint_trajectory", true);
        // controllers_to_await.emplace_back("omni_base_controller");
        // while (!base_client_->waitForServer(ros::Duration(5.0))) {
        //     ROS_INFO("Waiting for the /hsrb/omni_base_controller/follow_joint_trajectory action server to come up");
        // }

        arm_client_ = new TrajClientHSR("/hsrb/arm_trajectory_controller/follow_joint_trajectory", true);
        controllers_to_await.emplace_back("arm_trajectory_controller");
        while (!arm_client_->waitForServer(ros::Duration(5.0))) {
            py::print("Waiting for the /hsrb/arm_trajectory_controller/follow_joint_trajectory action server to come up");
        }

        // do not check if running, see https://qa.hsr.io/en/question/2164/gripper_controller-not-present-in-simulation/
        gripper_client_ = new GripperClient("/hsrb/gripper_controller/grasp", true);
        // controllers_to_await.push_back("gripper_controller");
        // while(!gripper_client_->waitForServer(ros::Duration(5.0))){
        //     ROS_INFO("Waiting for the /hsrb/gripper_controller/follow_joint_trajectory action server to come up");
        // }

        vector<string> joint_names = joint_model_group_->getVariableNames();
        arm_goal_.trajectory.joint_names.resize(joint_names.size() - 1);
        arm_goal_.trajectory.points.resize(1);
        arm_goal_.trajectory.points[0].positions.resize(joint_names.size() - 1);
        arm_goal_.trajectory.points[0].velocities.resize(joint_names.size() - 1);

        // make sure the controller is running
        ros::ServiceClient controller_manager_client = nh_->serviceClient<controller_manager_msgs::ListControllers>("/hsrb/controller_manager/list_controllers");
        controller_manager_msgs::ListControllers list_controllers;

        while (!controller_manager_client.call(list_controllers)) {
            py::print("Waiting for /hsrb/controller_manager/list_controllers");
            ros::Duration(0.5).sleep();
        }

        std::string cname;
        for (int j = 0; j < controllers_to_await.size(); j++) {
            cname = controllers_to_await.back();
            controllers_to_await.pop_back();
            bool running = false;
            while (!running) {
                py::print("Waiting for /hsrb/", cname);
                ros::Duration(0.5).sleep();
                if (controller_manager_client.call(list_controllers)) {
                    for (const auto &c : list_controllers.response.controller) {
                        if (c.name == cname && c.state == "running") {
                            running = true;
                        }
                    }
                }
            }
        }
    }
}

tf::Transform RobotHSR::calcDesiredBaseTf(const tf::Transform &base_tf,
                                        const tf::Vector3 &base_translation_relative,
                                        const double base_rotation_relative,
                                        const double dt) {
    return myutils::calcDesiredBaseTfOmni(base_tf,
                                        base_translation_relative,
                                        base_rotation_relative,
                                        dt);
}

void RobotHSR::calcDesiredBaseCommand(const tf::Transform &current_base_tf,
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

void RobotHSR::sendArmCommands(const vector<vector<double>> &joint_values, vector<double> &timestamps) {
    vector<string> joint_names = joint_model_group_->getVariableNames();
    arm_goal_.trajectory.points.resize(joint_values.size());

    for (int k = 0; k < joint_values.size(); k++) {
        int j = 0;
        arm_goal_.trajectory.points[k].positions.resize(joint_names.size() - 1);
        arm_goal_.trajectory.points[k].time_from_start = ros::Duration(timestamps[k]);
        for (int i = 0; i < joint_names.size(); i++) {
            // part of the moveit controller definition, but not part of /opt/ros/melodic/share/hsrb_common_config/params/hsrb_controller_config.yaml
            if (joint_names[i] != "wrist_ft_sensor_frame_joint") {
                arm_goal_.trajectory.joint_names[j] = joint_names[i];
                arm_goal_.trajectory.points[k].positions[j] = joint_values[k][i];
                j++;
            }
        }
    }

    // When to start the trajectory. Will get dropped with 0
    arm_goal_.trajectory.header.stamp = ros::Time::now();
    // send off commands to run in parallel
    arm_client_->sendGoal(arm_goal_);
}

bool RobotHSR::getArmSuccess() {
    arm_client_->waitForResult(ros::Duration(10.0));
    if (arm_client_->getState() != actionlib::SimpleClientGoalState::SUCCEEDED) {
        py::print("The arm_client_ failed.");
        // throw std::runtime_error("The arm_client_ failed.");
        return false;
    } else {
        return true;
    }
}

// Alternative to just using twist commands. But reacts way too slow
//void RobotHSR::sendBaseCommand(geometry_msgs::Twist base_cmd_rel, const tf::Transform &desired_base_tf, double exec_duration) {
//        control_msgs::FollowJointTrajectoryGoal goal;
//        // goal.trajectory.header = trajectory.joint_trajectory.header;
//        goal.trajectory.header.frame_id = "map";
//        goal.trajectory.header.stamp = ros::Time::now();
//        goal.trajectory.joint_names.push_back("odom_x");
//        goal.trajectory.joint_names.push_back("odom_y");
//        goal.trajectory.joint_names.push_back("odom_t");
//
//        trajectory_msgs::JointTrajectoryPoint trajectory_point;
//        trajectory_point.time_from_start = ros::Duration(exec_duration);
//        trajectory_point.positions.push_back(desired_base_tf.getOrigin().x());
//        trajectory_point.positions.push_back(desired_base_tf.getOrigin().y());
//        double yaw = myutils::qToRpy(desired_base_tf.getRotation()).z();
//        // -pi - piであればそのままいれても問題ない
//        trajectory_point.positions.push_back(yaw);
//
//        goal.trajectory.points.push_back(trajectory_point);
//
//        base_client_->sendGoal(goal);
//}

void RobotHSR::applyGripperEffort(double effort, bool wait_for_result){
    tmc_control_msgs::GripperApplyEffortGoal goal;
    goal.effort = effort;
    gripper_client_->sendGoal(goal);

    if (wait_for_result) {
        gripper_client_->waitForResult(ros::Duration(5.0));
        if (gripper_client_->getState() != actionlib::SimpleClientGoalState::SUCCEEDED)
            py::print("The gripper controller failed.");
    }
}

void RobotHSR::openGripper(double position, bool wait_for_result) {
    // hsr takes 1.0 as completely open -> calculate proportional to an assumed max. opening of 0.1m
//    position = std::min(position / 0.1, 1.0);
//
//    control_msgs::FollowJointTrajectoryGoal goal;
//    goal.trajectory.joint_names.push_back("hand_motor_joint");
//
//    goal.trajectory.points.resize(1);
//    goal.trajectory.points[0].positions.resize(1);
//    goal.trajectory.points[0].effort.resize(1);
//    goal.trajectory.points[0].velocities.resize(1);
//
//    goal.trajectory.points[0].positions[0] = position;
//    goal.trajectory.points[0].velocities[0] = 0.0;
//    goal.trajectory.points[0].effort[0] = 500;
//    goal.trajectory.points[0].time_from_start = ros::Duration(3.0);
    applyGripperEffort(0.5, wait_for_result);
}

void RobotHSR::closeGripper(double position, bool wait_for_result) {
    // 0.0 is not completely closed, but rather both 'forks' pointing straight ahead. -0.02 is roughly fully closed
    //    openGripper(position - 0.02, wait_for_result);
    applyGripperEffort(-0.5, wait_for_result);
}
