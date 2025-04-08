//
// Created by honerkam on 10/1/21.
//

#ifndef MODULATION_RL_ROBOT_HSR_H
#define MODULATION_RL_ROBOT_HSR_H

#include <actionlib/client/simple_action_client.h>
#include <control_msgs/FollowJointTrajectoryAction.h>
#include <control_msgs/FollowJointTrajectoryGoal.h>
#include <tmc_control_msgs/GripperApplyEffortAction.h>
#include <modulation_rl/robot_env.h>

typedef actionlib::SimpleActionClient<control_msgs::FollowJointTrajectoryAction> TrajClientHSR;
typedef actionlib::SimpleActionClient<tmc_control_msgs::GripperApplyEffortAction> GripperClient;


class RobotHSR : public RobotEnv {
private:
    // TrajClientHSR *base_client_;
    TrajClientHSR *arm_client_;
    GripperClient *gripper_client_;
    control_msgs::FollowJointTrajectoryGoal arm_goal_;
    void setup();

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
public:
  RobotHSR(uint32_t seed,
           const std::string &node_handle_name,
           bool init_controllers,
           const std::string &world_type,
           const RoboConf &robot_conf,
           double bioik_center_joints_weight,
           double bioik_avoid_joint_limits_weight,
           double bioik_regularization_weight,
           const std::string &bioik_regularization_type
    );
    ~RobotHSR() {
        if (init_controllers_) {
            delete gripper_client_;
            delete arm_client_;
        }
    }

    void openGripper(double position, bool wait_for_result) override;
    void closeGripper(double position, bool wait_for_result) override;
    void applyGripperEffort(double effort, bool wait_for_result);
    void sendArmCommands(const std::vector<std::vector<double>> &joint_values, std::vector<double> &timestamps) override;
};

#endif  // MODULATION_RL_ROBOT_HSR_H
