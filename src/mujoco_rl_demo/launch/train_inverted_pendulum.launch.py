from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    package_share_dir = get_package_share_directory('mujoco_rl_demo')
    config_file = os.path.join(package_share_dir, 'config', 'ppo_inverted_pendulum.yaml')

    return LaunchDescription([
        Node(
            package='mujoco_rl_demo',
            executable='train_inverted_pendulum',
            name='train_inverted_pendulum',
            output='screen',
            parameters=[config_file],
        )
    ])