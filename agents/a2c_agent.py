import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import EvalCallback
from environment.forest_env import DeforestationEnv


def train_a2c(total_timesteps=50000, save_path="models/a2c_forest"):
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

    model = A2C(
        "MlpPolicy",
        env,
        learning_rate=7e-4,
        n_steps=16,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        vf_coef=0.5,
        verbose=1,
    )

    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    model.save(f"{save_path}/a2c_final")
    env.close()
    eval_env.close()
    return model


if __name__ == "__main__":
    train_a2c()
