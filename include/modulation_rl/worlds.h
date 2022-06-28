#pragma once

#include <ros/ros.h>
#include <std_srvs/Empty.h>
#include <tf/tf.h>
#include <tf/transform_listener.h>


class BaseWorld {
  public:
    BaseWorld(std::string name,
              // for worlds that do not include full simulation of controllers, continuous time, etc.
              bool is_analytical,
              std::string global_link_transform,
              std::string base_footprint_name);
    virtual ~BaseWorld()= default;
    const std::string name_;
    const std::string global_link_transform_;
    const std::string base_footprint_name_;
    const bool is_analytical_;
    tf::TransformListener listener_;
    virtual tf::Transform getBaseTransformWorld();
    virtual tf::Transform getGripperTransformWorld(const std::string &frame_id = "map");
    virtual void setModelState(const std::string &model_name,
                               const tf::Transform &world_transform,
                               ros::Publisher &cmd_base_vel_pub) = 0;

    std::string getName() const { return name_; };
    bool isAnalytical() const { return is_analytical_; };
    virtual bool isWithinWorld(const tf::Transform &base_transform) { return true; };
    void awaitTfs();
};

class SimWorld : public BaseWorld {
public:
    explicit SimWorld(std::string global_link_transform, std::string base_footprint_name);
    void setModelState(const std::string &model_name,
                       const tf::Transform &world_transform,
                       ros::Publisher &cmd_base_vel_pub) override;
};

class GazeboWorld : public BaseWorld {
  private:
    //    ros::ServiceClient set_model_state_client_;
    //    ros::ServiceClient set_model_configuration_client_;
    //    ros::ServiceClient pause_gazebo_client_;
    //    ros::ServiceClient unpause_gazebo_client_;
  public:
    explicit GazeboWorld(std::string global_link_transform, std::string base_footprint_name);
    void setModelState(const std::string &model_name,
                       const tf::Transform &world_transform,
                       ros::Publisher &cmd_base_vel_pub) override;
};

class RealWorld : public BaseWorld {
  private:
    const double base_vel_;
    const double base_rot_vel_;
  public:
    explicit RealWorld(std::string global_link_transform,
                       std::string base_footprint_name,
                       double base_vel,
                       double base_rot_vel);
    void setModelState(const std::string &model_name,
                       const tf::Transform &world_transform,
                       ros::Publisher &cmd_base_vel_pub) override;
    bool isWithinWorld(const tf::Transform &base_transform) override;
};
