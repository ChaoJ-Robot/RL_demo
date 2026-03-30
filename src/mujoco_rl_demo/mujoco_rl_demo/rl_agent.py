import os
from pathlib import Path
from typing import Optional

import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env


class InvertedPendulumTrainer:
    def __init__(
        self,
        env_id: str,
        package_share_dir: str,
        total_timesteps: int = 100000,
        learning_rate: float = 3e-4,
        n_steps: int = 1024,
        batch_size: int = 256,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.0,
        n_envs: int = 4,
        model_save_name: str = "ppo_inverted_pendulum",
        use_tensorboard: bool = True,
    ):
        self.env_id = env_id
        self.package_share_dir = Path(package_share_dir)
        self.total_timesteps = total_timesteps
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.n_envs = n_envs
        self.model_save_name = model_save_name
        self.use_tensorboard = use_tensorboard

        self.models_dir = self.package_share_dir / "models"
        self.logs_dir = self.package_share_dir / "logs"
        self.videos_dir = self.package_share_dir / "videos"
        self.best_model_dir = self.models_dir / "best_model"

        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.videos_dir.mkdir(parents=True, exist_ok=True)
        self.best_model_dir.mkdir(parents=True, exist_ok=True)

    def _make_train_env(self):
        def _init():
            env = gym.make(self.env_id)
            env = RecordEpisodeStatistics(env)
            return env
        return _init

    def _make_eval_env(self):
        env = gym.make(self.env_id, render_mode="rgb_array")
        env = RecordEpisodeStatistics(env)
        env = RecordVideo(
            env,
            video_folder=str(self.videos_dir),
            episode_trigger=lambda episode_id: True,
            name_prefix="eval"
        )
        return env

    def train(self):
        train_env = make_vec_env(self._make_train_env(), n_envs=self.n_envs) # 使用 4 个环境并行采集经验
        eval_env = self._make_eval_env()

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(self.best_model_dir),
            log_path=str(self.logs_dir),
            eval_freq=5000,
            deterministic=True,
            render=False,
        )

        tensorboard_log = str(self.logs_dir / "tensorboard") if self.use_tensorboard else None

        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=self.learning_rate,
            n_steps=self.n_steps,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            clip_range=self.clip_range,
            ent_coef=self.ent_coef,
            verbose=1,
            tensorboard_log=tensorboard_log,
        )

        model.learn(
            total_timesteps=self.total_timesteps,
            callback=eval_callback,
            progress_bar=True,
        )

        final_model_path = self.models_dir / self.model_save_name
        model.save(str(final_model_path))

        train_env.close()
        eval_env.close()

        return str(final_model_path) + ".zip"


class InvertedPendulumEvaluator:
    def __init__(
        self,
        env_id: str,
        model_path: str,
        episodes: int = 3,
        render_mode: str = "human",
    ):
        self.env_id = env_id
        self.model_path = model_path
        self.episodes = episodes
        self.render_mode = render_mode

    def evaluate(self):
        env = gym.make(self.env_id, render_mode=self.render_mode)
        model = PPO.load(self.model_path)

        results = []

        for ep in range(self.episodes):
            obs, info = env.reset()
            done = False
            truncated = False
            total_reward = 0.0
            steps = 0

            while not (done or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                total_reward += float(reward)
                steps += 1

            results.append({
                "episode": ep + 1,
                "reward": total_reward,
                "steps": steps,
            })

        env.close()
        return results