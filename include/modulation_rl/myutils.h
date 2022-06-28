#ifndef MYUTILS_H
#define MYUTILS_H

#include <tf/tf.h>
#include <fstream>
#include "std_msgs/ColorRGBA.h"
#include "visualization_msgs/MarkerArray.h"

namespace myutils {
    constexpr double epsilon = 0.0000001;

    void printVector3(tf::Vector3 v, const std::string& descr);
    void printQ(tf::Quaternion q, const std::string& descr);
    void printT(tf::Transform t, const std::string& descr);
    void printArrayDouble(std::vector<double> array, const std::string& descr);
    void printArrayStr(std::vector<std::string> array, const std::string& descr);
    tf::Vector3 qToRpy(tf::Quaternion q);
    double calcRotDist(const tf::Transform &a, const tf::Transform &b);
    double vec3AbsMax(tf::Vector3 v);
    visualization_msgs::Marker markerFromTransform(tf::Transform t,
                                                   std::string ns,
                                                   std_msgs::ColorRGBA color,
                                                   int marker_id,
                                                   std::string frame_id,
                                                   const std::string &geometry = "arrow",
                                                   tf::Vector3 marker_scale = tf::Vector3(0.1, 0.025, 0.025));
    std_msgs::ColorRGBA getColorMsg(const std::string &color_name, double alpha = 1.0);
    tf::Vector3 minMaxScaleVel(tf::Vector3 vel, double min_vel, double max_vel);
    tf::Vector3 normScaleVel(tf::Vector3 vel, double min_vel_norm, double max_vel_norm);
    std::vector<double> normScaleVelPython(std::vector<double> vel, double min_vel_norm, double max_vel_norm);
    tf::Vector3 maxClipVel(tf::Vector3 vel, double max_vel);
    double clampDouble(double value, double min_value, double max_value);
    std::vector<double> tipToGripperGoalPython(const std::vector<double> &gripper_tip_goal_world,
                                               const std::vector<double> &tip_to_gripper_offset);
    tf::Transform tipToGripperGoal(const tf::Transform &gripper_tip_goal_world,
                                   const tf::Vector3 &tip_to_gripper_offset,
                                   const tf::Quaternion &gripper_to_base_rot_offset);
//    std::vector<double> gripperToTipGoalPython(const std::vector<double> &gripper_wrist_goal_world,
//                                               const std::vector<double> &tip_to_gripper_offset,
//                                               const std::vector<double> &gripper_to_base_rot_offset);
    tf::Transform gripperToTipGoal(const tf::Transform &gripper_wrist_goal_world,
                                   const tf::Vector3 &tip_to_gripper_offset,
                                   const tf::Quaternion &gripper_to_base_rot_offset);
    double rpyAngleDiff(double next, double prev);
    bool startsWith(const std::string &str, const std::string &substr);
    bool endsWith(const std::string &str, const std::string &substr);
    std::string trim(const std::string &s);
    tf::Transform listToTf(const std::vector<double> &input);
    std::vector<double> tfToList(const tf::Transform &input, bool normalize_q = false);
    std::vector<double> vector3ToList(const tf::Vector3 &input);
    tf::Vector3 listToVector3(const std::vector<double> &input);
    std::vector<double> quaternionToList(const tf::Quaternion &input, bool normalize_q = false);
    tf::Quaternion listToQuaternion(const std::vector<double> &input);
    tf::Quaternion calcDq(tf::Quaternion current, tf::Quaternion next);
    bool tfAlmostEqual(tf::Transform a, tf::Transform b);
    std::vector<double> pythonMultiplyTfs(const std::vector<double> &tf1_list, const std::vector<double> &tf2_list, bool invert_tf1);
    double pythonAngleShortestPath(const std::vector<double> &q1, std::vector<double> &q2);
    tf::Transform calcDesiredBaseTfOmni(const tf::Transform &base_tf,
                                        const tf::Vector3 &base_translation_relative,
                                        double base_rotation_relative,
                                        double dt);
    void calcDesiredBaseCommandOmni(const tf::Transform &current_base_tf,
                                    const tf::Transform &desired_base_tf,
                                    double dt,
                                    tf::Vector3 &base_translation_per_second,
                                    double &base_rotation_per_second);
    tf::Transform calcDesiredBaseTfDiffDrive(const tf::Transform &base_tf,
                                             const tf::Vector3 &base_translation_relative,
                                             double angle,
                                             double dt);
    void calcDesiredBaseCommandDiffDrive(const tf::Transform &current_base_tf,
                                         const tf::Transform &desired_base_tf,
                                         double dt,
                                         tf::Vector3 &base_translation_per_second,
                                         double &base_rotation_per_second);
    double trueModulo(double a, double b);
    std::vector<std::vector<double>> interpolateZ(std::vector<double> &cum_dists, std::vector<double> &obstacle_zs, const double &current_z, const double &max_map_height);
    std::vector<double> slerpPython(std::vector<double> &q_list, std::vector<double> &q2_list, const double &slerp_pct);
}  // namespace myutils


struct RobotObs {
    std::vector<double> base_tf;     // in world frame
    std::vector<double> gripper_tf;  // in world frame
    const std::vector<double> relative_gripper_tf;
    const std::vector<double> joint_values;
    bool ik_fail;
    bool in_selfcollision;
    std::vector<double> link_positions;
    std::vector<double> base_velocity_world;
    double base_rotation_velocity_world;
    std::vector<double> gripper_velocities_world;
    std::vector<double> gripper_tf_achieved;

    static std::vector<double> calcVelocityXyz(const std::vector<double> &current, const std::vector<double> &prev){
        return {current[0] - prev[0], current[1] - prev[1], current[2] - prev[2]};
    }

    static tf::Quaternion qFromListTf(const std::vector<double> &transform){
        return {transform[3], transform[4], transform[5], transform[6]};
    }

    void setVelocities(const RobotObs &prevRobotObs){
        base_velocity_world = calcVelocityXyz(base_tf, prevRobotObs.base_tf);
        double yaw = myutils::qToRpy(qFromListTf(base_tf)).z();
        double yaw2 = myutils::qToRpy(qFromListTf(prevRobotObs.base_tf)).z();
        base_rotation_velocity_world = myutils::rpyAngleDiff(yaw, yaw2);

        gripper_velocities_world = calcVelocityXyz(gripper_tf, prevRobotObs.gripper_tf);
        tf::Quaternion gripper_dq = myutils::calcDq(qFromListTf(gripper_tf), qFromListTf(prevRobotObs.gripper_tf));
        gripper_velocities_world.push_back(gripper_dq.x());
        gripper_velocities_world.push_back(gripper_dq.y());
        gripper_velocities_world.push_back(gripper_dq.z());
        gripper_velocities_world.push_back(gripper_dq.w());
    }
};


struct EEObs {
    std::vector<double> next_gripper_tf;
    const std::vector<double> next_gripper_tf_rel;
    const std::vector<double> ee_velocities_world;
    const std::vector<double> ee_velocities_rel;
    const double reward;
    const bool done;
};

namespace eeobs_utils{
    static EEObs eePlantoEEObs(const tf::Transform &next_gripper_tf, const RobotObs &robot_obs, const double &reward, const bool &done) {
        tf::Transform current_gripper_tf(myutils::listToTf(robot_obs.gripper_tf));
        current_gripper_tf.setRotation(current_gripper_tf.getRotation().normalized());

        tf::Vector3 vel_world(next_gripper_tf.getOrigin() - current_gripper_tf.getOrigin());

        tf::Transform base_tf = myutils::listToTf(robot_obs.base_tf);
        tf::Transform next_gripper_tf_rel = base_tf.inverse() * next_gripper_tf;
        tf::Vector3 vel_rel = next_gripper_tf_rel.getOrigin() - myutils::listToTf(robot_obs.relative_gripper_tf).getOrigin();

        tf::Quaternion dq = myutils::calcDq(current_gripper_tf.getRotation(), next_gripper_tf.getRotation());
        return EEObs{.next_gripper_tf = myutils::tfToList(next_gripper_tf, true),
                     .next_gripper_tf_rel = myutils::tfToList(next_gripper_tf_rel, true),
                     .ee_velocities_world = myutils::tfToList(tf::Transform(dq, vel_world), true),
                     .ee_velocities_rel = myutils::tfToList(tf::Transform(dq, vel_rel), true),
                     .reward = reward,
                     .done = done};
    }

    static EEObs eePlantoEEObsPython(const std::vector<double> &next_gripper_tf, const RobotObs &robot_obs, const double &reward, const bool &done) {
        return eePlantoEEObs(myutils::listToTf(next_gripper_tf), robot_obs, reward, done);
    }
}  // namespace eeobs_utils


struct RoboConf {
    const std::string name;
    const std::string joint_model_group_name;
    const std::string ik_joint_model_group_name;
    const std::string frame_id;
    const std::string global_link_transform;
    const tf::Vector3 tip_to_gripper_offset;
    const tf::Quaternion gripper_to_base_rot_offset;
    const std::string base_cmd_topic;
    const double kinematics_solver_timeout;
    std::map<std::string, double> initial_joint_values;
    const std::string torso_joint_name;

    RoboConf(const std::string &name,
             const std::string &joint_model_group_name,
             const std::string &ik_joint_model_group_name,
             const std::string &frame_id,
             const std::string &global_link_transform,
             const std::vector<double> &tip_to_gripper_offset,
             const std::vector<double> &gripper_to_base_rot_offset,
             const std::string &base_cmd_topic,
             const double kinematics_solver_timeout,
             std::map<std::string, double> initial_joint_values,
             const std::string &torso_joint_name) :
        name{name},
        joint_model_group_name{joint_model_group_name},
        ik_joint_model_group_name{ik_joint_model_group_name},
        frame_id{frame_id},
        global_link_transform{global_link_transform},
        tip_to_gripper_offset{myutils::listToVector3(tip_to_gripper_offset)},
        gripper_to_base_rot_offset{myutils::listToQuaternion(gripper_to_base_rot_offset)},
        base_cmd_topic{base_cmd_topic},
        kinematics_solver_timeout{kinematics_solver_timeout},
        initial_joint_values{initial_joint_values},
        torso_joint_name{torso_joint_name} {};
};

#endif