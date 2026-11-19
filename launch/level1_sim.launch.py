import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch_ros.actions import Node

def generate_launch_description():
    pkg_share = get_package_share_directory('vehicle_description')

    world_path = os.path.join(pkg_share, 'level1_world.sdf')
    urdf_path = os.path.join(pkg_share, 'urdf', 'vehicle.urdf')

    # Read robot description
    with open(urdf_path, 'r') as f:
        robot_description = f.read()

    # Run server and GUI separately so the Gazebo window is explicitly opened.
    gz_server = ExecuteProcess(
        cmd=['gz', 'sim', '-s', '-r', '-v', '3', world_path],
        output='screen'
    )
    gz_gui = ExecuteProcess(
        cmd=['gz', 'sim', '-g'],
        output='screen'
    )

    # 2. Robot State Publisher (publishes robot_description and tf for joints)
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': True
        }]
    )

    # 3. Spawn a robot; the evaluator moves it to each trial start before driving.
    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', 'autonomous_vehicle',
            '-topic', 'robot_description',
            '-x', '0.0',
            '-y', '-4.5',
            '-z', '0.0',
            '-Y', '1.5708'
        ],
        output='screen'
    )

    # 4. ROS-Gazebo Bridge, including the model-pose service used each trial.
    ros_gz_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            # Command velocity (ROS -> Gazebo)
            '/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
            # Odometry (Gazebo -> ROS)
            '/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry',
            # 2D LiDAR Scan (Gazebo -> ROS)
            '/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
            # Simulation Clock (Gazebo -> ROS)
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            # TF (Gazebo -> ROS)
            '/tf@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V',
            '/world/level1_world/set_pose@ros_gz_interfaces/srv/SetEntityPose',
        ],
        output='screen'
    )

    start_sim_clients = TimerAction(
        period=5.0,
        actions=[
            gz_gui,
            ros_gz_bridge,
            spawn_robot,
        ],
    )

    return LaunchDescription([
        gz_server,
        robot_state_publisher,
        start_sim_clients,
    ])
