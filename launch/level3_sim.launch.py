import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch_ros.actions import Node


def generate_launch_description():
    package_share = get_package_share_directory("vehicle_description")
    world = os.path.join(package_share, "level3_world.sdf")
    urdf = os.path.join(package_share, "urdf", "vehicle.urdf")
    pedestrian_script = os.path.join(package_share, "pedestrian_motion.py")
    with open(urdf, "r", encoding="utf-8") as stream:
        robot_description = stream.read()

    server = ExecuteProcess(cmd=["gz", "sim", "-s", "-r", "-v", "3", world],
                            output="screen")
    gui = ExecuteProcess(cmd=["gz", "sim", "-g"], output="screen")
    state_publisher = Node(
        package="robot_state_publisher", executable="robot_state_publisher",
        output="screen", parameters=[{"robot_description": robot_description,
                                       "use_sim_time": True}],
    )
    spawn_robot = Node(
        package="ros_gz_sim", executable="create", output="screen",
        arguments=["-name", "autonomous_vehicle", "-topic", "robot_description",
                   "-x", "-4.0", "-y", "0.0", "-z", "0.0"],
    )
    bridge = Node(
        package="ros_gz_bridge", executable="parameter_bridge", output="screen",
        arguments=[
            "/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist",
            "/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry",
            "/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan",
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
            "/tf@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V",
            "/world/level3_world/set_pose@ros_gz_interfaces/srv/SetEntityPose",
        ],
    )
    pedestrians = ExecuteProcess(cmd=["python3", pedestrian_script], output="screen")
    clients = TimerAction(period=5.0, actions=[gui, bridge, spawn_robot, pedestrians])
    return LaunchDescription([server, state_publisher, clients])
