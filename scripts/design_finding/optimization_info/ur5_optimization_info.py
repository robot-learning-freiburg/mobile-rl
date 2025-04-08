# definition of the links that are optimized
optimization_joints_ur5 = {
    "fmm_file": {
    "file_path": "/root/catkin_ws_fmm/src/fmm_description/urdf/fmm.urdf.xacro",
    "parameters" : {
    "arm_pitch" : 
        [{"tag": "xacro:ur5_robot",
         "attribute": "rpy",
         "position": 2,
         "link_name": "ur5_joint_ewellix_lift_top_link"}],
    "arm_yaw" : 
        [{"tag": "xacro:ur5_robot",
         "attribute": "rpy",
         "position": 3,
         "link_name": "ur5_joint_ewellix_lift_top_link"}],
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
         {"tag": "xacro:ur5_robot",
         "attribute": "xyz",
         "position": 1,
         "link_name": "ur5_joint_ewellix_lift_top_link"}],
    "end_effector_mount" :
        [{"tag": "xacro:robotiq_85_gripper",
         "attribute": "rpy",
         "position": 2,
         "link_name": "ur5_ee_fixed_joint",
        #  "additional": "joint_new"
         },
         {"tag": "xacro:robotiq_85_gripper",
         "attribute": "xyz",
         "position": 1,
         "link_name": "ur5_ee_fixed_joint",
        #  "additional": "joint_new"
         },
         {"tag": "xacro:robotiq_85_gripper",
         "attribute": "xyz",
         "position": 3,
         "link_name": "ur5_ee_fixed_joint",
        #  "additional": "joint_new"
         }],
    }
    }
}