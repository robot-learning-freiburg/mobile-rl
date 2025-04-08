#include <geometry_msgs/Twist.h>
#include <modulation_rl/robot_fmm.h>
#include <pybind11/pybind11.h>

using namespace std;
namespace py = pybind11;

RobotFMM::RobotFMM(uint32_t seed,
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
        throw runtime_error("init_controllers not implemented yet");
    }
}

tf::Transform RobotFMM::calcDesiredBaseTf(const tf::Transform &base_tf,
                                          const tf::Vector3 &base_translation_relative,
                                          const double base_rotation_relative,
                                          const double dt) {
    return myutils::calcDesiredBaseTfOmni(base_tf,
                                          base_translation_relative,
                                          base_rotation_relative,
                                          dt);
}

void RobotFMM::calcDesiredBaseCommand(const tf::Transform &current_base_tf,
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


void RobotFMM::sendArmCommands(const vector<vector<double>> &joint_values, vector<double> &timestamps) {
    throw runtime_error("sendArmCommands not implemented yet");
}

bool RobotFMM::getArmSuccess() {
    throw runtime_error("getArmSuccess not implemented yet");
}

void RobotFMM::moveGripper(double position, double effort, bool wait_for_result) {
    throw runtime_error("moveGripper not implemented yet");
}

void RobotFMM::openGripper(double position, bool wait_for_result) {
    throw runtime_error("openGripper not implemented yet");
}

void RobotFMM::closeGripper(double position, bool wait_for_result) {
    throw runtime_error("closeGripper not implemented yet");
}
