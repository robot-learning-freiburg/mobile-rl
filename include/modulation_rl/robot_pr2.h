//
// Created by honerkam on 9/30/21.
//

#ifndef MODULATION_RL_ROBOT_PR2_H
#define MODULATION_RL_ROBOT_PR2_H

#include <actionlib/client/simple_action_client.h>
#include <modulation_rl/robot_env.h>
#include <pr2_controllers_msgs/JointTrajectoryAction.h>
#include <pr2_controllers_msgs/Pr2GripperCommandAction.h>
#include <pr2_controllers_msgs/SingleJointPositionAction.h>
typedef actionlib::SimpleActionClient<pr2_controllers_msgs::Pr2GripperCommandAction> GripperClientPR2;
typedef actionlib::SimpleActionClient<pr2_controllers_msgs::JointTrajectoryAction> TrajClientPR2;
typedef actionlib::SimpleActionClient<pr2_controllers_msgs::SingleJointPositionAction> TorsoClient;


class RobotPR2 : public RobotEnv {
private:
    TrajClientPR2 *arm_client_;
    TrajClientPR2 *arm_client_left_;
    GripperClientPR2 *gripper_client_;
    TorsoClient *torso_client_;
    int torso_joint_index_ = -1;
    // ros::ServiceClient switch_controller_client_;
    pr2_controllers_msgs::JointTrajectoryGoal arm_goal_;
    pr2_controllers_msgs::JointTrajectoryGoal arm_goal_left_;
    // void stop_controllers();
    // void start_controllers();
    void moveGripper(double position, double effort, bool wait_for_result);
    bool getArmSuccess() override;
    tf::Transform calcDesiredBaseTf(const tf::Transform &base_tf,
                                    const tf::Vector3 &base_translation_relative,
                                    double base_rotation_relative,
                                    double dt) override;
    void calcDesiredBaseCommand(const tf::Transform &current_base_tf,
                                const tf::Transform &desired_base_tf,
                                double dt,
                                tf::Vector3 &base_translation_per_second,
                                double &base_rotation_per_second) override;
    void setLeftArm(std::vector<double> &joint_values);

public:
    RobotPR2(uint32_t seed,
           const std::string &node_handle_name,
           bool init_controllers,
           const std::string &world_type,
           const RoboConf &robot_conf,
           double bioik_center_joints_weight,
           double bioik_avoid_joint_limits_weight,
           double bioik_regularization_weight,
           const std::string &bioik_regularization_type);
    ~RobotPR2(){
        if (init_controllers_){
            delete arm_client_;
            delete gripper_client_;
            delete torso_client_;
        }
    }
    void openGripper(double position, bool wait_for_result) override;
    void closeGripper(double position, bool wait_for_result) override;
    void sendArmCommands(const std::vector<std::vector<double>> &joint_values, std::vector<double> &timestamps) override;
    RobotObs step(const std::vector<double>& base_translation_relative_list,
                  double base_rotation_relative,
                  const std::vector<double> &desired_gripper_tf_list,
                  const RobotObs &prev_robot_obs,
                  double dt_exec,
                  bool execute_cmds,
                  bool learn_torso,
                  double delta_torso,
                  const std::string &execute_style,
                  const std::string &perception_style,
                  const std::vector<double> &joint_values_action) override;
    RobotObs reset(const std::map<std::string, double> &initial_joint_values, bool do_close_gripper) override;
};
#endif  // MODULATION_RL_ROBOT_PR2_H
