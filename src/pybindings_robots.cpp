#include <modulation_rl/robot_hsr.h>
#include <modulation_rl/robot_pr2.h>
#include <modulation_rl/robot_tiago.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(pybindings_robots, m) {
    py::class_<RobotEnv> robotenv(m, "RobotEnv");
    robotenv.def("reset", &RobotEnv::reset, "Reset.")
        .def("step", &RobotEnv::step, "Execute the next time step in environment.")
        .def("open_gripper", &RobotEnv::openGripper, "Open the gripper.")
        .def("close_gripper", &RobotEnv::closeGripper, "Close the gripper.")
        .def("send_arm_commands", &RobotEnv::sendArmCommands, "sendArmCommands.")
        .def("send_base_command", &RobotEnv::sendBaseCommand, "sendBaseCommand.")
        .def("set_rng", &RobotEnv::setRng, "setRng.")
        .def("set_base_tf", &RobotEnv::setWorldJointPython, "setWorldJointPython.")
        .def("draw_joint_values", &RobotEnv::drawJointValues, "drawJointValues.")
        .def("set_joint_values", &RobotEnv::setJointValuesFromDict, "setJointValuesFromDict.")
        .def("get_joint_values", &RobotEnv::getJointValues, "getJointValues.")
        .def("get_joint_values_world", &RobotEnv::getJointValuesWorld, "getJointValuesWorld.")
        .def("get_joint_names", &RobotEnv::getJointNames, "getJointNames.")
        .def("get_joint_minima", &RobotEnv::getJointMinima, "getJointMinima.")
        .def("get_joint_maxima", &RobotEnv::getJointMaxima, "getJointMaxima.")
        .def("get_robot_obs", &RobotEnv::getRobotObs, "getRobotObs.")
        .def("get_robot_obs_with_vel", &RobotEnv::getRobotObsWithVelocities, "getRobotObsWithVelocities.")
        .def("is_analytical_world", &RobotEnv::isAnalyticalWorld, "isAnalyticalWorld.")
        .def("get_basetf_world", &RobotEnv::getBaseTransformWorld, "getBaseTransformWorld.")
        .def("get_grippertf_world", &RobotEnv::getGripperTransformWorld, "getGripperTransformWorld.")
        .def("get_world", &RobotEnv::getWorld, "getWorld.")
        .def("set_world", &RobotEnv::setWorld, "setWorld.")
        .def("set_bioik_regularization_weight", &RobotEnv::setBioikRegularizationWeight, "setBioikRegularizationWeight.")
        .def("set_bioik_avoidjointlimits_weight", &RobotEnv::setBioikAvoidJointLimitsWeight, "setBioikAvoidJointLimitsWeight.")
        .def("set_bioik_centerjoints_weight", &RobotEnv::setBioikCenterJointsWeight, "setBioikCenterJointsWeight.")
        .def("calc_desired_base_tf", &RobotEnv::calcDesiredBaseTfPython, "calcDesiredBaseTfPython.")
        .def("publish_marker", &RobotEnv::publishMarker, "publishMarker.")
        .def("publish_marker_rgb", &RobotEnv::publishMarkerRGB, "publishMarkerRGB.")
        .def("publish_robot_state", &RobotEnv::publishRobotState, "publishRobotStatePython.")
        .def("publish_trajectory", &RobotEnv::publishTrajectory, "publishTrajectory.")
        .def("tip_to_gripper_tf", &RobotEnv::tipToGripperTf, "tipToGripperTf.")
        .def("gripper_to_tip_tf", &RobotEnv::gripperToTipTf, "gripperToTipTf.");
    py::class_<RobotPR2>(m, "RobotPR2", robotenv /* <- specify Python parent type */)
        .def(py::init<uint32_t, std::string, bool, std::string, RoboConf, double, double, double, std::string>());

    py::class_<RobotTiago>(m, "RobotTiago", robotenv)
        .def(py::init<uint32_t, std::string, bool, std::string, RoboConf, double, double, double, std::string>());

    py::class_<RobotHSR>(m, "RobotHSR", robotenv)
        .def(py::init<uint32_t, std::string, bool, std::string, RoboConf, double, double, double, std::string>());

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
