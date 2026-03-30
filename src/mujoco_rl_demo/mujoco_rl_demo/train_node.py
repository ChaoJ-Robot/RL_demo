import os

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory

from mujoco_rl_demo.rl_agent import InvertedPendulumTrainer


class TrainInvertedPendulumNode(Node):
    def __init__(self):
        super().__init__('train_inverted_pendulum')

        self.declare_parameter('env_id', 'InvertedPendulum-v5')
        self.declare_parameter('total_timesteps', 100000)
        self.declare_parameter('learning_rate', 3e-4)
        self.declare_parameter('n_steps', 1024)
        self.declare_parameter('batch_size', 256)
        self.declare_parameter('n_epochs', 10)
        self.declare_parameter('gamma', 0.99)
        self.declare_parameter('gae_lambda', 0.95)
        self.declare_parameter('clip_range', 0.2)
        self.declare_parameter('ent_coef', 0.0)
        self.declare_parameter('n_envs', 4)
        self.declare_parameter('model_save_name', 'ppo_inverted_pendulum')
        self.declare_parameter('use_tensorboard', True)

        package_share_dir = get_package_share_directory('mujoco_rl_demo')

        trainer = InvertedPendulumTrainer(
            env_id=self.get_parameter('env_id').value,
            package_share_dir=package_share_dir,
            total_timesteps=self.get_parameter('total_timesteps').value,
            learning_rate=self.get_parameter('learning_rate').value,
            n_steps=self.get_parameter('n_steps').value,
            batch_size=self.get_parameter('batch_size').value,
            n_epochs=self.get_parameter('n_epochs').value,
            gamma=self.get_parameter('gamma').value,
            gae_lambda=self.get_parameter('gae_lambda').value,
            clip_range=self.get_parameter('clip_range').value,
            ent_coef=self.get_parameter('ent_coef').value,
            n_envs=self.get_parameter('n_envs').value,
            model_save_name=self.get_parameter('model_save_name').value,
            use_tensorboard=self.get_parameter('use_tensorboard').value,
        )

        self.get_logger().info('Start training InvertedPendulum PPO...')
        final_model_path = trainer.train()
        self.get_logger().info(f'Training complete. Final model saved at: {final_model_path}')


def main(args=None):
    rclpy.init(args=args)
    node = TrainInvertedPendulumNode()
    node.destroy_node()
    rclpy.shutdown()