
#ifndef GAUSSIAN_MIXTURE_MODEL_H
#define GAUSSIAN_MIXTURE_MODEL_H

#include <tf/tf.h>
#include <tf_conversions/tf_eigen.h>
#include <Eigen/Core>
#include <fstream>

class GaussianMixtureModel {
  public:
    explicit GaussianMixtureModel(double max_speed_base_rot);
    ~GaussianMixtureModel();

    void adaptModel(tf::Transform obj_origin_goal, tf::Vector3 gmm_base_offset);
    void AdaptModelPython(const std::vector<double> &gripper_goal_tip, const std::vector<double> &gmm_base_offset);
    bool loadFromFile(std::string &filename);
    void integrateModel(double current_time,
                        double dt,
                        Eigen::VectorXf *current_pose,
                        Eigen::VectorXf *current_speed,
                        const double &min_velocity,
                        const double &max_velocity,
                        const double &max_rot_velocity,
                        double gmm_time_offset);
    std::vector<double> integrateModelPython(double current_time,
                                             double dt_real,
                                             std::vector<double> &current_gripper_tf_wrist,
                                             std::vector<double> &current_base_tf_list,
                                             std::vector<double> &current_speed,
                                             const double &min_velocity,
                                             const double &max_velocity,
                                             const double &max_rot_velocity,
                                             double gmm_time_offset,
                                             const std::vector<double> &tip_to_gripper_offset);
    int getNrModes() const { return _nr_modes; };
    std::string getType() const { return _type; };
    void setType(std::string type) { _type = type; };
    double getkP() const { return _kP; };
    double getkV() const { return _kV; };
    std::vector<double> getPriors() const { return _Priors; };
    std::vector<Eigen::VectorXf> getMu() const { return _MuEigen; };
    std::vector<Eigen::MatrixXf> getSigma() const { return _Sigma; };
    tf::StampedTransform getGoalState() const { return _goalState; };
    tf::Transform getStartState() const { return _startState; };
    tf::Transform getGraspPose() const { return _related_object_grasp_pose; };
    tf::Transform getRelatedObjPose() const { return _related_object_pose; };
    tf::Transform getLastMuEigenBckGripper() const;
    std::string getObjectName() const { return _related_object_name; };

    tf::Transform tipToObjOrigin(const tf::Transform &tip) const;
    std::vector<double> objOriginToTip(const std::vector<double> &obj_origin) const;
    std::vector<std::vector<double>> getMusPython() const;
    // double gmm_time_offset_;

  protected:
    int _nr_modes;
    std::string _type;
    double _kP;
    double _kV;
    double _motion_duration;
//    double _max_speed_gripper_rot;
    double _max_speed_base_rot;
    std::vector<double> _Priors;
    std::vector<std::vector<double>> _Mu;
    std::vector<Eigen::VectorXf> _MuEigen;
    std::vector<Eigen::VectorXf> _MuEigenBck;
    std::vector<Eigen::MatrixXf> _Sigma;
    std::vector<Eigen::MatrixXf> _SigmaBck;
    tf::StampedTransform _goalState;
    tf::Transform _startState;
    tf::Transform _related_object_pose;
    tf::Transform _related_object_grasp_pose;
    std::string _related_object_name;
    // std::vector<int> _colliding_poses;
    // int _colliding_base_angle;
    // int _ik_error_counter;
    // int _ik_collision_counter;
    // int _total_nr_poses;
    // int _current_nr_poses;
    // int _nr_pose_last_plotted;
    // int _plot_every_xth;
    // geometry_msgs::PoseArray trajectory_pose_array;

    // ros::NodeHandle nh_;
    // ros::Publisher Mu_pub_;
    // ros::Publisher Traj_pub_;
    // ros::Publisher Traj_pub2_;
    // ros::Publisher Ellipses_pub_;
    // ros::Publisher planning_scene_diff_publisher_;
    // ros::ServiceClient client_get_scene_;
    // boost::shared_ptr<modulation::Modulation_manager> manager_;
    // modulation::Modulation modulation_;

    // IK test stuff
    // boost::shared_ptr<robot_state::RobotState> kinematic_state;
    // robot_state::JointModelGroup* joint_model_group;
    // boost::shared_ptr<planning_scene::PlanningScene> planning_scene_;
    // collision_detection::AllowedCollisionMatrix currentACM_;

    template<typename T> bool parseVector(std::ifstream &is, std::vector<T> &pts, const std::string &name);
    double gaussPDF(double &current_time, int mode_nr);
    void plotEllipses(Eigen::VectorXf &curr_pose, Eigen::VectorXf &curr_speed, double dt);
    void clearMarkers(int nrPoints);
    template<typename T1, typename T2> T1 extract(const T2 &full, const T1 &ind);
};

#endif  // GAUSSIAN_MIXTURE_MODEL_H
