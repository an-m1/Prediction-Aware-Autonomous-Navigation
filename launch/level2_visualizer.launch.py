import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess


def generate_launch_description():
    visualizer_path = os.path.join(
        get_package_share_directory("vehicle_description"), "level2_visualizer.py"
    )
    return LaunchDescription([
        ExecuteProcess(cmd=["python3", visualizer_path], output="screen"),
    ])
