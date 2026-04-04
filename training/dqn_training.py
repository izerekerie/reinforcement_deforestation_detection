import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from environment.forest_env import DeforestationEnv

HYPERPARAMETER_CONFIGS = [
    {
        "name": "dqn_baseline",
        "learning_rate": 1e-3,
        "buffer_size": 50000,
        "batch_size": 64,
        "gamma": 0.99,
        "exploration_fraction": 0.3,
        "exploration_final_eps": 0.05,
        "target_update_interval": 500,
        "learning_starts": 1000,
    },
    {
        "name": "dqn_low_lr",
        "learning_rate": 1e-4,
        "buffer_size": 50000,
        "batch_size": 64,
        "gamma": 0.99,
        "exploration_fraction": 0.3,
        "exploration_final_eps": 0.05,
        "target_update_interval": 500,
        "learning_starts": 1000,
    },
    {
        "name": "dqn_high_lr",
        "learning_rate": 5e-3,
        "buffer_size": 50000,
        "batch_size": 64,
        "gamma": 0.99,
        "exploration_fraction": 0.3,
        "exploration_final_eps": 0.05,
        "target_update_interval": 500,
        "learning_starts": 1000,
    },
    {
        "name": "dqn_large_buffer",
        "learning_rate": 1e-3,
        "buffer_size": 100000,
        "batch_size": 128,
        "gamma": 0.99,
        "exploration_fraction": 0.3,
        "exploration_final_eps": 0.05,
        "target_update_interval": 500,
        "learning_starts": 2000,
    },
    {
        "name": "dqn_small_batch",
        "learning_rate": 1e-3,
        "buffer_size": 50000,
        "batch_size": 32,
        "gamma": 0.99,
        "exploration_fraction": 0.3,
        "exploration_final_eps": 0.05,
        "target_update_interval": 250,
        "learning_starts": 1000,
    },
    {
        "name": "dqn_high_explore",
        "learning_rate": 1e-3,
        "buffer_size": 50000,
        "batch_size": 64,
        "gamma": 0.99,
        "exploration_fraction": 0.5,
        "exploration_final_eps": 0.1,
        "target_update_interval": 500,
        "learning_starts": 1000,
    },
    {
        "name": "dqn_low_gamma",
        "learning_rate": 1e-3,
        "buffer_size": 50000,
        "batch_size": 64,
        "gamma": 0.95,
        "exploration_fraction": 0.3,
        "exploration_final_eps": 0.05,
        "target_update_interval": 500,
        "learning_starts": 1000,
    },
    {
        "name": "dqn_fast_target",
        "learning_rate": 1e-3,
        "buffer_size": 50000,
        "batch_size": 64,
        "gamma": 0.99,
        "exploration_fraction": 0.3,
        "exploration_final_eps": 0.05,
        "target_update_interval": 200,
        "learning_starts": 1000,
    },
    {
        "name": "dqn_slow_target",
        "learning_rate": 1e-3,
        "buffer_size": 50000,
        "batch_size": 64,
        "gamma": 0.99,
        "exploration_fraction": 0.3,
        "exploration_final_eps": 0.05,
        "target_update_interval": 1000,
        "learning_starts": 1000,
    },
    {
        "name": "dqn_aggressive",
        "learning_rate": 3e-3,
        "buffer_size": 30000,
        "batch_size": 128,
        "gamma": 0.98,
        "exploration_fraction": 0.2,
        "exploration_final_eps": 0.02,
        "target_update_interval": 300,
        "learning_starts": 500,
    },
]


def evaluate_model(model, n_episodes=20):
    env = DeforestationEnv()
    rewards = []
    healths = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                healths.append(info["forest_health"])
                break
        rewards.append(total_reward)
    env.close()
    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_health": float(np.mean(healths)),
        "win_rate": float(np.mean([h >= 0.7 for h in healths])),
    }


def train_single(config, total_timesteps=50000, save_dir="models/dqn"):
    name = config["name"]
    print(f"\n{'='*50}")
    print(f"Training: {name}")
    print(f"{'='*50}")

    params = {k: v for k, v in config.items() if k != "name"}
    print(f"Params: {json.dumps(params, indent=2)}")

    env = DeforestationEnv()
    eval_env = DeforestationEnv()

    run_dir = os.path.join(save_dir, name)
    os.makedirs(run_dir, exist_ok=True)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=run_dir,
        log_path=run_dir,
        eval_freq=5000,
        n_eval_episodes=5,
        deterministic=True,
    )

    model = DQN("MlpPolicy", env, verbose=1, **params)
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    model.save(os.path.join(run_dir, "final_model"))

    results = evaluate_model(model)
    results["config"] = config
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results: reward={results['mean_reward']:.1f}, health={results['mean_health']:.1%}, win={results['win_rate']:.0%}")

    env.close()
    eval_env.close()
    return name, results


def train_all(total_timesteps=50000):
    os.makedirs("models/dqn", exist_ok=True)
    all_results = {}

    for config in HYPERPARAMETER_CONFIGS:
        name, results = train_single(config, total_timesteps)
        all_results[name] = results

    best_name = max(all_results, key=lambda n: all_results[n]["mean_reward"])
    print(f"\nBest DQN config: {best_name}")
    print(f"  Reward: {all_results[best_name]['mean_reward']:.1f}")
    print(f"  Health: {all_results[best_name]['mean_health']:.1%}")
    print(f"  Win Rate: {all_results[best_name]['win_rate']:.0%}")

    with open("models/dqn/all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    return all_results, best_name


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=50000)
    parser.add_argument("--run", type=str, default=None, help="Run single config by name")
    args = parser.parse_args()

    if args.run:
        config = next((c for c in HYPERPARAMETER_CONFIGS if c["name"] == args.run), None)
        if config:
            train_single(config, args.timesteps)
        else:
            print(f"Config '{args.run}' not found. Available: {[c['name'] for c in HYPERPARAMETER_CONFIGS]}")
    else:
        train_all(args.timesteps)
