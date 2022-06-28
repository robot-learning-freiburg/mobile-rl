import os
import argparse
import time
import rospy
from gazebo_msgs.srv import SetPhysicsProperties, GetPhysicsProperties, SetPhysicsPropertiesRequest
from std_srvs.srv import Empty
from subprocess import Popen
import rospkg

"""
IMPORTANT: ENSURE THAT THIS FILE ONLY RELIES ON PYTHON2 COMPATIBLE SYNTAX
"""

gazebo_cmds = {
    'pr2': "roslaunch modulation_rl pr2_empty_world.launch".split(" "),
    'tiago': "roslaunch modulation_rl tiago_empty_world.launch".split(" "),
    'hsr': "roslaunch modulation_rl hsrb_empty_world.launch".split(" "),
}

analytical_cmds = {
    'pr2': "roslaunch modulation_rl pr2_analytical.launch".split(" "),
    'tiago': "roslaunch modulation_rl tiago_analytical.launch".split(" "),
    'hsr': "roslaunch modulation_rl hsr_analytical.launch".split(" ")
}

def get_world_file(task, algo):
    rospack = rospkg.RosPack()
    if task in ["rndstartrndgoal", "restrictedws", "simpleobstacle", "spline"]:
        return "empty.world"
    elif task in ["picknplace", "picknplacedyn", "door", "drawer", "roomdoor"]:
        return "modulation_tasks.world"
    elif task in ["dynobstacle"]:
        return "dynamic_world.world"
    elif task in ["bookstorepnp", "bookstoredoor"]:
        world_file = "bookstore_simple.world" if algo in ["moveit", "bi2rrt"] else "bookstore.world"

        if algo not in ["moveit", "bi2rrt"]:
            # the planning_scene plugin will fail with some of the collision meshes
            plugin_path = rospack.get_path("modulation_rl").replace("/src/modulation_rl", "") + "/" + "devel/lib/libgazebo_ros_moveit_planning_scene.so"
            if os.path.exists(plugin_path):
                os.remove(plugin_path)

        return rospack.get_path("aws_robomaker_bookstore_world") + "/" + "worlds" + "/" + world_file
    elif task in ["apartment"]:
        return rospack.get_path("aws_robomaker_small_house_world") + "/" + "worlds" + "/" + "small_house.world"
    else:
        raise ValueError("No world defined for task " + task)


def gazebo_set_pyhsics_properties(time_step):
    rospy.init_node('hanlde_launchfiles')
    ns = "gazebo"
    rospy.wait_for_service(ns + '/get_physics_properties')
    rospy.wait_for_service(ns + '/set_physics_properties')

    try:
        get_physics_properties = rospy.ServiceProxy(ns + '/get_physics_properties', GetPhysicsProperties)
        props = get_physics_properties()
    except rospy.ServiceException as e:
        rospy.logerr('couldn\'t get physics properties while preparing to set physics properties: ' + str(e))
        assert False

    try:
        pause_physics_srv = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        unpause_physics_srv = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)

        req = SetPhysicsPropertiesRequest(time_step=time_step,
                                          max_update_rate = int(1. / time_step),
                                          gravity=props.gravity,
                                          ode_config=props.ode_config)
        set_physics_properties = rospy.ServiceProxy(ns + '/set_physics_properties', SetPhysicsProperties)

        pause_physics_srv()
        time.sleep(0.01)

        set_physics_properties(req)
        time.sleep(0.01)
        unpause_physics_srv()
    except rospy.ServiceException as e:
        rospy.logerr('couldn\'t set physics properties: ' + str(e))
        assert False
    rospy.signal_shutdown("handle_launchfiles")


def start_analytical(env_name, gui=False, bioik=False):
    cmd = analytical_cmds[env_name]
    gui = "true" if gui else "false"
    cmd += ["gui:=" + gui]
    bioik = "true" if bioik else "false"
    cmd += ["BIOIK:=" + bioik]
    print("Starting command ", cmd)
    p = Popen(cmd)
    time.sleep(6)
    return p


def start_launch_files(env_name, algo, task, gui=False, debug=False, bioik=False):
    if (algo in ["moveit", "bi2rrt"]):
        if env_name == "pr2":
            gazebo_cmd = ("roslaunch pr2_planning_model pr2.launch").split(" ")
        elif (env_name == "hsr") and (algo == "moveit"):
            gazebo_cmd = gazebo_cmds[env_name]
        else:
            raise NotImplementedError()
    elif algo == 'rl':
        gazebo_cmd = gazebo_cmds[env_name]
        # moveit_cmd = moveit_cmds[env_name]
    else:
        raise NotImplementedError(algo)

    gui = "true" if gui else "false"
    gazebo_cmd += ["gui:=" + gui]

    debug = "true" if debug else "false"
    gazebo_cmd += ["debug:=" + debug]

    bioik = "true" if bioik else "false"
    gazebo_cmd += ["BIOIK:=" + bioik]

    if (env_name == "pr2") and (task == "roomdoor"):
        gazebo_cmd += ["local_costmap_frame:=" + "base_footprint"]

    if (task in ["picknplace", "picknplacedyn", "door", "drawer", "roomdoor"]):
        rospy.set_param('/moving_obstacle/goal_range', 3)

    if (task in ["picknplacedyn", "dynobstacle", "bookstorepnp"]):
        local_map_inflation = 0.03
    else:
        local_map_inflation = 0.0
    rospy.set_param('/costmap_node/costmap/inflation_layer/enabled', local_map_inflation > 0.0)
    rospy.set_param('/costmap_node/costmap/inflation_layer/inflation_radius', local_map_inflation)
    rospy.set_param('/costmap_node/costmap/inflation_radius', local_map_inflation)
    # fast_empty for fast physics, but not sure node will be able to fully keep up
    world_name = get_world_file(task, algo)
    gazebo_cmd += ['world_name:=' + world_name]

    print("Starting command ", gazebo_cmd)
    p_gazebo = Popen(gazebo_cmd) if gazebo_cmd else None
    p_moveit = None

    time.sleep(30)

    if (env_name == "pr2") and (algo not in ["moveit", "bi2rrt"]):
        # NOTE: THE GRIPPER WILL FAIL TO OPEN / CLOSE WITH THIS STEP SIZE
        # works in a faster world, otherwise it will take forever
        gazebo_set_pyhsics_properties(time_step=0.002)

    return p_gazebo, p_moveit


def stop_launch_files(p_gazebo, p_moveit):
    time.sleep(10)
    if p_gazebo:
        p_gazebo.terminate()
    if p_moveit:
        p_moveit.terminate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str.lower, choices=['pr2', 'tiago', 'hsr'], help='')
    parser.add_argument('--pure_analytical', type=str.lower, help='')
    parser.add_argument('--algo', type=str.lower, help='')
    parser.add_argument('--gui', action='store_true', default=False, help='')
    parser.add_argument('--debug', action='store_true', default=False, help='')
    parser.add_argument('--task', default='', type=str.lower, help='')
    parser.add_argument('--bioik', action='store_true', default=False, help='')
    args = parser.parse_args()

    assert args.env, "No env supplied for startup. Make sure to start directly through the runfile"
    assert args.pure_analytical in ['yes', 'no']

    print("starting roscore")
    p_roscore = Popen(["roscore"])
    time.sleep(5)

    if args.pure_analytical == "yes":
        assert args.algo == 'rl', args.algo
        start_analytical(args.env, gui=args.gui, bioik=args.bioik)
    else:
        start_launch_files(args.env, algo=args.algo, gui=args.gui, debug=args.debug, task=args.task, bioik=args.bioik)
    print("\nAll launchfiles started\n")
