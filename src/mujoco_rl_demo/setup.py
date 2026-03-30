from setuptools import setup
from glob import glob
import os

package_name = 'mujoco_rl_demo'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'models'), glob('models/*')),
        (os.path.join('share', package_name, 'logs'), glob('logs/*')),
        (os.path.join('share', package_name, 'videos'), glob('videos/*')),
    ],
    install_requires=['setuptools', 'numpy>=1.21.0', 'gymnasium>=0.29.0', 'stable-baselines3>=2.0.0', 'matplotlib>=3.5.0'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='ROS 2 MuJoCo RL demo package',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'train_inverted_pendulum = mujoco_rl_demo.train_node:main',
            'eval_inverted_pendulum = mujoco_rl_demo.eval_node:main',
        ],
    },
)