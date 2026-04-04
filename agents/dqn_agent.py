import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from environment.forest_env import DeforestationEnv


def train_dqn(total_timesteps=50000, save_path="models/dqn_forest"):
    env = DeforestationEnv()
    eval_env = DeforestationEnv()

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_path,
        log_path=save_path,
        eval_freq=5000,
        n_eval_episodes=5,
        deterministic=True,
    )

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=64,
        gamma=0.99,
        exploration_fraction=0.3,
        exploration_final_eps=0.05,
        target_update_interval=500,
        verbose=1,
    )

    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    model.save(f"{save_path}/dqn_final")
    env.close()
    eval_env.close()
    return model


if __name__ == "__main__":
    train_dqn()
