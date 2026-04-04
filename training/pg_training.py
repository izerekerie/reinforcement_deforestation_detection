import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import torch
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import EvalCallback
from environment.forest_env import DeforestationEnv
from agents.reinforce_agent import ReinforceAgent

PPO_CONFIGS = [
    {
        "name": "ppo_baseline",
        "learning_rate": 3e-4,
        "n_steps": 512,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
    },
    {
        "name": "ppo_low_lr",
        "learning_rate": 1e-4,
        "n_steps": 512,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
    },
    {
        "name": "ppo_high_lr",
        "learning_rate": 1e-3,
        "n_steps": 512,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
    },
    {
        "name": "ppo_wide_clip",
        "learning_rate": 3e-4,
        "n_steps": 512,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.3,
        "ent_coef": 0.01,
    },
    {
        "name": "ppo_narrow_clip",
        "learning_rate": 3e-4,
        "n_steps": 512,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.1,
        "ent_coef": 0.01,
    },
    {
        "name": "ppo_high_entropy",
        "learning_rate": 3e-4,
        "n_steps": 512,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.05,
    },
    {
        "name": "ppo_long_horizon",
        "learning_rate": 3e-4,
        "n_steps": 1024,
        "batch_size": 128,
        "n_epochs": 10,
        "gamma": 0.995,
        "gae_lambda": 0.98,
        "clip_range": 0.2,
        "ent_coef": 0.01,
    },
    {
        "name": "ppo_short_horizon",
        "learning_rate": 3e-4,
        "n_steps": 256,
        "batch_size": 32,
        "n_epochs": 5,
        "gamma": 0.97,
        "gae_lambda": 0.9,
        "clip_range": 0.2,
        "ent_coef": 0.01,
    },
    {
        "name": "ppo_many_epochs",
        "learning_rate": 1e-4,
        "n_steps": 512,
        "batch_size": 64,
        "n_epochs": 20,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.15,
        "ent_coef": 0.01,
    },
    {
        "name": "ppo_aggressive",
        "learning_rate": 5e-4,
        "n_steps": 256,
        "batch_size": 128,
        "n_epochs": 15,
        "gamma": 0.98,
        "gae_lambda": 0.92,
        "clip_range": 0.25,
        "ent_coef": 0.02,
    },
]

A2C_CONFIGS = [
    {
        "name": "a2c_baseline",
        "learning_rate": 7e-4,
        "n_steps": 16,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
    },
    {
        "name": "a2c_low_lr",
        "learning_rate": 1e-4,
        "n_steps": 16,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
    },
    {
        "name": "a2c_high_lr",
        "learning_rate": 3e-3,
        "n_steps": 16,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
    },
    {
        "name": "a2c_long_steps",
        "learning_rate": 7e-4,
        "n_steps": 64,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
    },
    {
        "name": "a2c_short_steps",
        "learning_rate": 7e-4,
        "n_steps": 5,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
    },
    {
        "name": "a2c_high_entropy",
        "learning_rate": 7e-4,
        "n_steps": 16,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "ent_coef": 0.05,
        "vf_coef": 0.5,
    },
    {
        "name": "a2c_low_vf",
        "learning_rate": 7e-4,
        "n_steps": 16,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "ent_coef": 0.01,
        "vf_coef": 0.25,
    },
    {
        "name": "a2c_high_vf",
        "learning_rate": 7e-4,
        "n_steps": 16,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "ent_coef": 0.01,
        "vf_coef": 0.8,
    },
    {
        "name": "a2c_low_gamma",
        "learning_rate": 7e-4,
        "n_steps": 16,
        "gamma": 0.95,
        "gae_lambda": 0.9,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
    },
    {
        "name": "a2c_tuned",
        "learning_rate": 5e-4,
        "n_steps": 32,
        "gamma": 0.99,
        "gae_lambda": 0.98,
        "ent_coef": 0.02,
        "vf_coef": 0.4,
    },
]

REINFORCE_CONFIGS = [
    {"name": "reinforce_baseline", "lr": 1e-3, "gamma": 0.99, "hidden": 128},
    {"name": "reinforce_low_lr", "lr": 1e-4, "gamma": 0.99, "hidden": 128},
    {"name": "reinforce_high_lr", "lr": 5e-3, "gamma": 0.99, "hidden": 128},
    {"name": "reinforce_low_gamma", "lr": 1e-3, "gamma": 0.95, "hidden": 128},
    {"name": "reinforce_high_gamma", "lr": 1e-3, "gamma": 0.999, "hidden": 128},
    {"name": "reinforce_small_net", "lr": 1e-3, "gamma": 0.99, "hidden": 64},
    {"name": "reinforce_large_net", "lr": 1e-3, "gamma": 0.99, "hidden": 256},
    {"name": "reinforce_slow_learn", "lr": 5e-4, "gamma": 0.995, "hidden": 128},
    {"name": "reinforce_fast_learn", "lr": 3e-3, "gamma": 0.97, "hidden": 128},
    {"name": "reinforce_deep", "lr": 1e-3, "gamma": 0.99, "hidden": 192},
]


def evaluate_sb3(model, n_episodes=20):
    env = DeforestationEnv()
    rewards, healths = [], []
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


def evaluate_reinforce(agent, n_episodes=20):
    env = DeforestationEnv()
    rewards, healths = [], []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0
        while True:
            action, _ = agent.predict(obs, deterministic=True)
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


def train_ppo_configs(total_timesteps=50000):
    save_dir = "models/pg/ppo"
    os.makedirs(save_dir, exist_ok=True)
    all_results = {}

    for config in PPO_CONFIGS:
        name = config["name"]
        print(f"\n{'='*50}")
        print(f"Training PPO: {name}")
        print(f"{'='*50}")

        params = {k: v for k, v in config.items() if k != "name"}
        run_dir = os.path.join(save_dir, name)
        os.makedirs(run_dir, exist_ok=True)

        env = DeforestationEnv()
        eval_env = DeforestationEnv()

        eval_cb = EvalCallback(
            eval_env, best_model_save_path=run_dir, log_path=run_dir,
            eval_freq=5000, n_eval_episodes=5, deterministic=True,
        )

        model = PPO("MlpPolicy", env, verbose=1, **params)
        model.learn(total_timesteps=total_timesteps, callback=eval_cb)
        model.save(os.path.join(run_dir, "final_model"))

        results = evaluate_sb3(model)
        results["config"] = config
        all_results[name] = results

        with open(os.path.join(run_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)

        print(f"  reward={results['mean_reward']:.1f} health={results['mean_health']:.1%} win={results['win_rate']:.0%}")
        env.close()
        eval_env.close()

    with open(os.path.join(save_dir, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    best = max(all_results, key=lambda n: all_results[n]["mean_reward"])
    print(f"\nBest PPO: {best} (reward={all_results[best]['mean_reward']:.1f})")
    return all_results


def train_a2c_configs(total_timesteps=50000):
    save_dir = "models/pg/a2c"
    os.makedirs(save_dir, exist_ok=True)
    all_results = {}

    for config in A2C_CONFIGS:
        name = config["name"]
        print(f"\n{'='*50}")
        print(f"Training A2C: {name}")
        print(f"{'='*50}")

        params = {k: v for k, v in config.items() if k != "name"}
        run_dir = os.path.join(save_dir, name)
        os.makedirs(run_dir, exist_ok=True)

        env = DeforestationEnv()
        eval_env = DeforestationEnv()

        eval_cb = EvalCallback(
            eval_env, best_model_save_path=run_dir, log_path=run_dir,
            eval_freq=5000, n_eval_episodes=5, deterministic=True,
        )

        model = A2C("MlpPolicy", env, verbose=1, **params)
        model.learn(total_timesteps=total_timesteps, callback=eval_cb)
        model.save(os.path.join(run_dir, "final_model"))

        results = evaluate_sb3(model)
        results["config"] = config
        all_results[name] = results

        with open(os.path.join(run_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)

        print(f"  reward={results['mean_reward']:.1f} health={results['mean_health']:.1%} win={results['win_rate']:.0%}")
        env.close()
        eval_env.close()

    with open(os.path.join(save_dir, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    best = max(all_results, key=lambda n: all_results[n]["mean_reward"])
    print(f"\nBest A2C: {best} (reward={all_results[best]['mean_reward']:.1f})")
    return all_results


def train_reinforce_configs(n_episodes=500):
    save_dir = "models/pg/reinforce"
    os.makedirs(save_dir, exist_ok=True)
    all_results = {}

    env = DeforestationEnv()
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n
    env.close()

    for config in REINFORCE_CONFIGS:
        name = config["name"]
        print(f"\n{'='*50}")
        print(f"Training REINFORCE: {name}")
        print(f"{'='*50}")

        run_dir = os.path.join(save_dir, name)
        os.makedirs(run_dir, exist_ok=True)

        agent = ReinforceAgent(obs_size, n_actions, lr=config["lr"], gamma=config["gamma"])
        train_env = DeforestationEnv()
        episode_rewards = []
        best_reward = float("-inf")

        for ep in range(n_episodes):
            obs, _ = train_env.reset()
            ep_reward = 0
            while True:
                action = agent.select_action(obs)
                obs, reward, terminated, truncated, info = train_env.step(action)
                agent.rewards.append(reward)
                ep_reward += reward
                if terminated or truncated:
                    break

            agent.update()
            episode_rewards.append(ep_reward)

            if ep_reward > best_reward:
                best_reward = ep_reward
                agent.save(os.path.join(run_dir, "best_model.pt"))

            if (ep + 1) % 100 == 0:
                avg = np.mean(episode_rewards[-50:])
                print(f"  Episode {ep+1}/{n_episodes} | Avg: {avg:.1f} | Best: {best_reward:.1f}")

        agent.save(os.path.join(run_dir, "final_model.pt"))
        np.save(os.path.join(run_dir, "rewards.npy"), np.array(episode_rewards))

        results = evaluate_reinforce(agent)
        results["config"] = config
        all_results[name] = results

        with open(os.path.join(run_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)

        print(f"  reward={results['mean_reward']:.1f} health={results['mean_health']:.1%} win={results['win_rate']:.0%}")
        train_env.close()

    with open(os.path.join(save_dir, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    best = max(all_results, key=lambda n: all_results[n]["mean_reward"])
    print(f"\nBest REINFORCE: {best} (reward={all_results[best]['mean_reward']:.1f})")
    return all_results


def train_all(total_timesteps=50000, reinforce_episodes=500):
    print("="*60)
    print("POLICY GRADIENT HYPERPARAMETER TUNING")
    print("="*60)

    ppo_results = train_ppo_configs(total_timesteps)
    a2c_results = train_a2c_configs(total_timesteps)
    reinforce_results = train_reinforce_configs(reinforce_episodes)

    combined = {
        "ppo": ppo_results,
        "a2c": a2c_results,
        "reinforce": reinforce_results,
    }
    with open("models/pg/all_results.json", "w") as f:
        json.dump(combined, f, indent=2)

    print("\n" + "="*60)
    print("ALL POLICY GRADIENT TUNING COMPLETE")
    print("="*60)
    return combined


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=50000)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--algo", choices=["ppo", "a2c", "reinforce", "all"], default="all")
    args = parser.parse_args()

    if args.algo == "ppo":
        train_ppo_configs(args.timesteps)
    elif args.algo == "a2c":
        train_a2c_configs(args.timesteps)
    elif args.algo == "reinforce":
        train_reinforce_configs(args.episodes)
    else:
        train_all(args.timesteps, args.episodes)
