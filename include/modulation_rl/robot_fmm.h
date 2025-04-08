//
// Created by honerkam on 9/30/21.
//

#ifndef MODULATION_RL_ROBOT_FMM_H
#define MODULATION_RL_ROBOT_FMM_H

#include <actionlib/client/simple_action_client.h>
#include <control_msgs/FollowJointTrajectoryAction.h>
#include <control_msgs/FollowJointTrajectoryGoal.h>
#include <modulation_rl/robot_env.h>


class RobotFMM : public RobotEnv {
private:
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

public:
    RobotFMM(uint32_t seed,
           const std::string &node_handle_name,
           bool init_controllers,
           const std::string &world_type,
           const RoboConf &robot_conf,
           double bioik_center_joints_weight,
           double bioik_avoid_joint_limits_weight,
           double bioik_regularization_weight,
           const std::string &bioik_regularization_type);
    ~RobotFMM(){}
    void openGripper(double position, bool wait_for_result) override;
    void closeGripper(double position, bool wait_for_result) override;
    void sendArmCommands(const std::vector<std::vector<double>> &joint_values, std::vector<double> &timestamps) override;
};
#endif  // MODULATION_RL_ROBOT_FMM_H
