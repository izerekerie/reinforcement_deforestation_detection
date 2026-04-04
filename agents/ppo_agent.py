import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from environment.forest_env import DeforestationEnv


def train_ppo(total_timesteps=50000, save_path="models/ppo_forest"):
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

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
    )

    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    model.save(f"{save_path}/ppo_final")
    env.close()
    eval_env.close()
    return model


if __name__ == "__main__":
    train_ppo()
