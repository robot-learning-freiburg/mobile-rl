#ifndef MODULATION_RL_ROBOT_ENV_H
#define MODULATION_RL_ROBOT_ENV_H

#include <moveit/planning_scene_monitor/planning_scene_monitor.h>
#include <moveit/robot_state/robot_state.h>
#include <moveit_msgs/DisplayTrajectory.h>
#include <ros/ros.h>
#include <tf/tf.h>
#include <tf_conversions/tf_eigen.h>
#include <bio_ik/bio_ik.h>

#include <utility>
#include "tf/transform_datatypes.h"
#include "std_msgs/ColorRGBA.h"
#include "visualization_msgs/MarkerArray.h"
#include "modulation_rl/myutils.h"
#include "modulation_rl/worlds.h"
#include "modulation_rl/validity_fun.h"


class ROSCommonNode2 {
protected:
    ROSCommonNode2(int argc, char **argv, const char *node_name) {
        ros::init(argc, argv, node_name, ros::init_options::AnonymousName);
    }
};

class RobotEnv : ROSCommonNode2 {
  private:
    random_numbers::RandomNumberGenerator *rng_;
    BaseWorld *world_;

    ros::AsyncSpinner *spinner_;

    ros::Publisher base_cmd_pub_;
    ros::Publisher robstate_visualizer_;
    ros::Publisher gripper_visualizer_;
    ros::Publisher traj_visualizer_;
    robot_trajectory::RobotTrajectoryPtr trajectory_;
    moveit_msgs::DisplayTrajectory display_msg_;

protected:
    ros::NodeHandle *nh_;
    RoboConf robot_config_;
    bool init_controllers_;
    robot_state::JointModelGroup *joint_model_group_;
    robot_state::JointModelGroup *ik_joint_model_group_;
    std::string ik_solver_name_;
    robot_state::RobotStatePtr kinematic_state_;
    std::string base_link_name_;
    planning_scene_monitor::PlanningSceneMonitorPtr planning_scene_monitor_;

    planning_scene::PlanningScenePtr planning_scene_;
    void updatePlanningScene();

    double bioik_center_joints_weight_;
    double bioik_avoid_joint_limits_weight_;
    double bioik_regularization_weight_;
    const std::string bioik_regularization_type_;
    std::vector<double> bioik_max_velocities_;
    void getBioIkOptions(bio_ik::BioIKKinematicsQueryOptions &ik_options, double dt) const;

public:
    RobotEnv(uint32_t seed,
             const std::string& node_handle_name,
             bool init_controllers,
             const std::string& world_type,
             RoboConf robot_conf,
             double bioik_center_joints_weight,
             double bioik_avoid_joint_limits_weight,
             double bioik_regularization_weight,
             std::string bioik_regularization_type
    );

    ~RobotEnv() {
        delete nh_;
        if (init_controllers_) {
            spinner_->stop();
            delete spinner_;
        }
        delete world_;
        delete rng_;
    }

    virtual RobotObs reset(const std::map<std::string, double> &initial_joint_values, bool do_close_gripper);
    virtual RobotObs step(const std::vector<double> &base_translation_relative_list,
                          double base_rotation_relative,
                          const std::vector<double> &desired_gripper_tf_list,
                          const RobotObs &prev_robot_obs,
                          double dt_exec,
                          bool execute_cmds,
                          bool learn_torso,
                          double delta_torso,
                          const std::string &execute_style,
                          const std::string &perception_style,
                          const std::vector<double> &joint_values_action);
    virtual bool findIkDefault(const tf::Transform &desired_gripper_tf_world, const robot_state::GroupStateValidityCallbackFn &callback_fn);
    virtual bool findIkBioIk(const tf::Transform &desired_gripper_tf_world, double dt, const robot_state::GroupStateValidityCallbackFn &callback_fn);
    virtual bool findIk(const tf::Transform &desired_gripper_tf_world, double dt, bool &in_selfcollision);
    virtual tf::Transform calcDesiredBaseTf(const tf::Transform &base_tf,
                                            const tf::Vector3 &base_translation_relative,
                                            double base_rotation_relative,
                                            double dt) = 0;
    std::vector<double> calcDesiredBaseTfPython(const std::vector<double> &base_tf,
                                            const std::vector<double> &base_translation_relative,
                                            double base_rotation_relative,
                                            double dt){
      return myutils::tfToList(calcDesiredBaseTf(myutils::listToTf(base_tf),
                                                 myutils::listToVector3(base_translation_relative),
                                                 base_rotation_relative,
                                                 dt));
    };
    virtual void calcDesiredBaseCommand(const tf::Transform &current_base_tf,
                                        const tf::Transform &desired_base_tf,
                                        double dt,
                                        tf::Vector3 &base_translation_per_second,
                                        double &base_rotation_per_second) = 0;

    std::vector<double> drawJointValues(double z_min, double z_max);
    bool checkSelfCollision();
    void setJointValuesFromDict(const std::map<std::string, double> &joint_values, bool set_in_world);
    bool checkArmValues();
    void setWorldJoint(tf::Transform &transform);
    void setWorldJointPython(std::vector<double> &base_tf_list);
    void setWorld(const std::string &world_type);
    void setRng(uint32_t seed){ rng_ = new random_numbers::RandomNumberGenerator(seed);}
    tf::Transform getGlobalLinkTransform(const std::string &link_name);
    RobotObs getRobotObs(bool ik_fail, bool in_selfcollision);
    RobotObs getRobotObsWithVelocities(bool ik_fail, const RobotObs &prev_robot_obs, bool in_selfcollision);
    // TODO: return references for all of these?
    std::vector<double> getJointValues(const std::string& joint_group);
    std::vector<double> getJointValuesWorld(const std::string& joint_group);
    std::vector<std::string> getJointNames(const std::string& joint_group);
    std::vector<double> getJointMinima(const std::string& joint_group);
    std::vector<double> getJointMaxima(const std::string& joint_group);
    std::string getWorld() { return world_->name_; };
    bool isAnalyticalWorld() { return world_->isAnalytical(); };
    std::vector<double> getBaseTransformWorld() { return myutils::tfToList(world_->getBaseTransformWorld()); };
    std::vector<double> getGripperTransformWorld(const std::string &frame_id = "map") { return myutils::tfToList(world_->getGripperTransformWorld(frame_id)); };
    void setBioikRegularizationWeight(double weight) { bioik_regularization_weight_ = weight; };
    void setBioikAvoidJointLimitsWeight(double weight) { bioik_avoid_joint_limits_weight_ = weight; };
    void setBioikCenterJointsWeight(double weight) { bioik_center_joints_weight_ = weight; };

    void publishRobotState();
    void publishTrajectory();
    void publishMarker(const std::vector<double> &marker_tf,
                       int marker_id,
                       const std::string &name_space,
                       const std::string &color,
                       double alpha,
                       const std::string &geometry,
                       const std::vector<double> &marker_scale,
                       const std::string &frame_id);
    void publishMarkerRGB(const std::vector<double> &marker_tf,
                          int marker_id,
                          const std::string &name_space,
                          const std::vector<double> &color,
                          double alpha,
                          const std::string &geometry,
                          const std::vector<double> &marker_scale,
                          const std::string &frame_id);
    void publishMarkerMsg(const std::vector<double> &marker_tf,
                          int marker_id,
                          const std::string &name_space,
                          std_msgs::ColorRGBA &color,
                          const std::string &geometry,
                          const std::vector<double> &marker_scale,
                          const std::string &frame_id);

    std::vector<double> tipToGripperTf(const std::vector<double> &tip_tf);
    std::vector<double> gripperToTipTf(const std::vector<double> &tip_tf);

    virtual void sendBaseCommand(const std::vector<double> &base_translation_relative_list, double base_rotation_relative);
    void sendArmCommandCurrent(double exec_duration);
    // bool resetArm(double planning_time);
    virtual void sendArmCommands(const std::vector<std::vector<double>> &joint_values, std::vector<double> &timestamps) = 0;
    virtual bool getArmSuccess() = 0;
    virtual void openGripper(double position, bool wait_for_result) { throw std::runtime_error("NOT IMPLEMENTED YET"); };
    virtual void closeGripper(double position, bool wait_for_result) { throw std::runtime_error("NOT IMPLEMENTED YET"); };
};


#include "bio_ik/goal.h"
class MinimalDisplacementGoalCubed : public bio_ik::Goal {
public:
    explicit MinimalDisplacementGoalCubed(double weight = 1.0, bool secondary = true) {
        weight_ = weight;
        secondary_ = secondary;
    }

    double evaluate(const bio_ik::GoalContext &context) const override {
        double sum = 0.0;
        for (size_t i = 0; i < context.getProblemVariableCount(); i++) {
            double d = context.getProblemVariablePosition(i) - context.getProblemVariableInitialGuess(i);
             d *= context.getProblemVariableWeight(i);
             d = std::abs(d);
             sum += std::pow(d, 4);
        }
        return sum;
    }
};

class AboveVelLimitGoal : public bio_ik::Goal {
public:
    std::vector<double> max_velocities_;
    double dt_;

    explicit AboveVelLimitGoal(std::vector<double> max_velocities, double dt, double weight = 1.0, bool secondary = true) {
        weight_ = weight;
        secondary_ = secondary;
        max_velocities_ = std::move(max_velocities);
        dt_ = dt;
    }

    double evaluate(const bio_ik::GoalContext &context) const override {
        if (context.getProblemVariableCount() != max_velocities_.size()){
            throw std::runtime_error("Wrong max_velocities size");
        }

        double sum = 0.0;
        for (size_t i = 0; i < context.getProblemVariableCount(); i++) {
            double max_vel = dt_ * max_velocities_[i];
            // start penalising at half the max?
            max_vel /= 2.0;
            double d = context.getProblemVariablePosition(i) - context.getProblemVariableInitialGuess(i);
            double above_limit = std::max(d - max_vel, 0.0);
            double above_limit_pct = (max_vel > 0.0) ? above_limit / max_vel : 0.0;
            sum += std::pow(above_limit_pct, 2);
            // sum += d > max_vel;
        }
        return sum;
    }
};


#endif  // MODULATION_RL_ROBOT_ENV_H
