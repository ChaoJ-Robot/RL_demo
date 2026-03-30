from pathlib import Path

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory

from mujoco_rl_demo.rl_agent import InvertedPendulumEvaluator


class EvalInvertedPendulumNode(Node):
    def __init__(self):
        super().__init__('eval_inverted_pendulum')

        self.declare_parameter('env_id', 'InvertedPendulum-v5')
        self.declare_parameter('model_path', 'models/best_model/best_model.zip')
        self.declare_parameter('episodes', 3)
        self.declare_parameter('render_mode', 'human')

        package_share_dir = Path(get_package_share_directory('mujoco_rl_demo'))
        model_rel_path = self.get_parameter('model_path').value
        model_abs_path = package_share_dir / model_rel_path

        evaluator = InvertedPendulumEvaluator(
            env_id=self.get_parameter('env_id').value,
            model_path=str(model_abs_path),
            episodes=self.get_parameter('episodes').value,
            render_mode=self.get_parameter('render_mode').value,
        )

        self.get_logger().info(f'Load model from: {model_abs_path}')
        results = evaluator.evaluate()

        for item in results:
            self.get_logger().info(
                f"Episode {item['episode']}: reward={item['reward']:.2f}, steps={item['steps']}"
            )


def main(args=None):
    rclpy.init(args=args)
    node = EvalInvertedPendulumNode()
    node.destroy_node()
    rclpy.shutdown()