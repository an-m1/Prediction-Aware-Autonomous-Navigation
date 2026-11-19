import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    level = LaunchConfiguration("level")
    visualizer = os.path.join(
        get_package_share_directory("vehicle_description"), "navigation_visualizer.py"
    )
    return LaunchDescription([
        DeclareLaunchArgument(
            "level", default_value="1",
            description="Navigation level to visualize: 1, 2, or 3",
        ),
        ExecuteProcess(
            cmd=["python3", visualizer, "--level", level],
            output="screen",
        ),
    ])
