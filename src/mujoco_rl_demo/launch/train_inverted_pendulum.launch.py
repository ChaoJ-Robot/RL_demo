from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    package_share_dir = get_package_share_directory('mujoco_rl_demo')
    config_file = os.path.join(package_share_dir, 'config', 'ppo_inverted_pendulum.yaml')

    return LaunchDescription([
        Node(
            package='mujoco_rl_demo', # 包名
            executable='train_inverted_pendulum', # 可执行文件名
            name='train_inverted_pendulum',      # 节点名
            output='screen',                     # 输出模式
            parameters=[config_file],             # 配置文件路径
        )
    ])