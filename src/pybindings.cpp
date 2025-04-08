#include <modulation_rl/gaussian_mixture_model.h>
#include <modulation_rl/mycostmap.h>
#include <modulation_rl/myutils.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(pybindings, m) {
    py::class_<RobotObs>(m, "RobotObs")
        .def(py::init<std::vector<double>,
                      std::vector<double>,
                      std::vector<double>,
                      std::vector<double>,
                      bool,
                      bool,
                      std::vector<double>>())
        .def_readwrite("base_tf", &RobotObs::base_tf)
        .def_readwrite("gripper_tf", &RobotObs::gripper_tf)
        .def_readonly("relative_gripper_tf", &RobotObs::relative_gripper_tf)
        .def_readonly("joint_values", &RobotObs::joint_values)
        .def_readwrite("ik_fail", &RobotObs::ik_fail)
        .def_readwrite("in_selfcollision", &RobotObs::in_selfcollision)
        .def_readonly("link_positions", &RobotObs::link_positions)
        .def_readonly("base_velocity_world", &RobotObs::base_velocity_world)
        .def_readonly("base_rotation_velocity_world", &RobotObs::base_rotation_velocity_world)
        .def_readonly("gripper_velocities_world", &RobotObs::gripper_velocities_world)
        .def_readonly("gripper_tf_achieved", &RobotObs::gripper_tf_achieved);

    py::class_<RoboConf>(m, "RobotConfig")
            .def(py::init<std::string,
                    std::string,
                    std::string,
                    std::string,
                    std::string,
                    std::vector<double>,
                    std::vector<double>,
                    std::string,
                    double,
                    std::map<std::string, double>,
                    std::string>())
            .def_readonly("name", &RoboConf::name)
            .def_readonly("joint_model_group_name", &RoboConf::joint_model_group_name)
            .def_readonly("ik_joint_model_group_name", &RoboConf::ik_joint_model_group_name)
            .def_readonly("frame_id", &RoboConf::frame_id)
            .def_readonly("global_link_transform", &RoboConf::global_link_transform)
            .def_readonly("tip_to_gripper_offset", &RoboConf::tip_to_gripper_offset)
            .def_readonly("gripper_to_base_rot_offset", &RoboConf::gripper_to_base_rot_offset)
            .def_readonly("base_cmd_topic", &RoboConf::base_cmd_topic)
            .def_readonly("kinematics_solver_timeout", &RoboConf::kinematics_solver_timeout)
            .def_readonly("initial_joint_values", &RoboConf::initial_joint_values)
            .def_readonly("torso_joint_name", &RoboConf::torso_joint_name);

// .def_readonly("reward", &RobotObs::reward)
//        .def(py::pickle(
//            [](RobotObs &obs) {  // __getstate__
//                /* Return a tuple that fully encodes the state of the object */
//                return py::make_tuple(obs.base_tf,
//                                      obs.gripper_tf,
//                                      obs.relative_gripper_tf,
//                                      obs.joint_values,
//                                      obs.base_velocity,
//                                      obs.base_rotation_velocity,
//                                      obs.gripper_velocities,
//                                      obs.ik_fail,
//                                      obs.reward);
//            },
//            [](const py::tuple& t) {  // __setstate__
//                if (t.size() != 9)
//                    throw std::runtime_error("Invalid state!");
//
//                /* Create a new C++ instance */
//                return RobotObs{t[0].cast<std::vector<double>>(),
//                                    t[1].cast<std::vector<double>>(),
//                                    t[2].cast<std::vector<double>>(),
//                                    t[3].cast<std::vector<double>>(),
//                                    t[4].cast<std::vector<double>>(),
//                                    t[5].cast<double>(),
//                                    t[6].cast<std::vector<double>>(),
//                                    t[7].cast<bool>(),
//                                    t[8].cast<double>()};
//            }));

    py::class_<EEObs>(m, "EEObs")
        .def(py::init<std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>, double, bool>())
        .def_readwrite("next_gripper_tf", &EEObs::next_gripper_tf)
        .def_readonly("next_gripper_tf_rel", &EEObs::next_gripper_tf_rel)
        .def_readonly("ee_velocities_world", &EEObs::ee_velocities_world)
        .def_readonly("ee_velocities_rel", &EEObs::ee_velocities_rel)
        .def_readonly("reward", &EEObs::reward)
        .def_readonly("done", &EEObs::done);

    py::class_<GaussianMixtureModel>(m, "GaussianMixtureModel")
        .def(py::init<double>())
        .def("load_from_file", &GaussianMixtureModel::loadFromFile, "loadFromFile")
        .def("adapt_model", &GaussianMixtureModel::AdaptModelPython, "adaptModel")
        .def("integrate_model", &GaussianMixtureModel::integrateModelPython, "integrateModel")
        .def("get_mus", &GaussianMixtureModel::getMusPython, "getMusPython.")
        .def("obj_origin_to_tip", &GaussianMixtureModel::objOriginToTip, "objOriginToTip.");

    m.def("get_inflated_map", &my_costmap::getInflatedMapPython, "getInflatedMap");
    m.def("multiply_tfs", &myutils::pythonMultiplyTfs, "pythonMultiplyTfs");
    m.def("normscale_vel", &myutils::normScaleVelPython, "normScaleVelPython");
    m.def("angle_shortest_path", &myutils::pythonAngleShortestPath, "pythonAngleShortestPath");
    m.def("tip_to_gripper_goal", &myutils::tipToGripperGoalPython, "tipToGripperGoal");
    // m.def("gripper_to_tip_goal", &myutils::gripperToTipGoalPython, "gripperToTipGoal");
    m.def("interpolate_z", &myutils::interpolateZ, "interpolateZ");
    m.def("slerp_single", &myutils::slerpPython, "slerpPython");
    m.def("eeplan_to_eeobs", &eeobs_utils::eePlantoEEObsPython, "eePlantoEEObsPython");

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
