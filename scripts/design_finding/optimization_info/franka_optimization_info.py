optimization_joints_franka = {
    "fmm_file": {
    "file_path": "/root/catkin_ws_fmm/src/fmm_description/urdf/fmm.urdf.xacro",
    "parameters" : {
    "arm_pitch" : 
        [{"tag": "xacro:franka_arm",
         "attribute": "rpy",
         "position": 2,
         "link_name": "panda_joint_ewellix_lift_top_link"}],
    "arm_yaw" : 
        [{"tag": "xacro:franka_arm",
         "attribute": "rpy",
         "position": 3,
         "link_name": "panda_joint_ewellix_lift_top_link"}],
    "tower_yaw" : 
        [{"tag": "xacro:ewellix_lift_500mm",
         "attribute": "rpy",
         "position": 3,
         "link_name": "ewellix_lift_base_joint"}],
    "tower_yValue" : 
        [{"tag": "xacro:ewellix_lift_500mm",
         "attribute": "xyz",
         "position": 2,
         "link_name": "ewellix_lift_base_joint"}],
    "tower_xValue" : 
        [{"tag": "xacro:ewellix_lift_500mm",
         "attribute": "xyz",
         "position": 1,
         "link_name": "ewellix_lift_base_joint"},
         {"tag": "xacro:franka_arm",
         "attribute": "xyz",
         "position": 1,
         "link_name": "panda_joint_ewellix_lift_top_link"}],
    }
    },
    "robot_arm":
    {
    "file_path": "gazebo_world/fmm/franka_arm/franka_arm.xacro",
    "parameters" : {
    "end_effector_mount" :
        [{"tag": "joint",
         "attribute": "rpy",
         "position": 3,
         "link_name": "panda_joint_new",
         "additional": "joint_new"},
         {"tag": "joint",
         "attribute": "xyz",
         "position": 2,
         "link_name": "panda_joint_new",
         "additional": "joint_new"}],
    }
    }
}