import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.callbacks import EvalCallback
from environment.forest_env import DeforestationEnv
from agents.reinforce_agent import ReinforceAgent

TIMESTEPS = 200000
REINFORCE_EPISODES = 1000


def evaluate(model, n_episodes=20):
    env = DeforestationEnv()
    rewards, healths, steps = [], [], []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        tr = 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            tr += r
            if term or trunc:
                rewards.append(tr)
                healths.append(info["forest_health"])
                steps.append(info["step"])
                break
    env.close()
    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_health": float(np.mean(healths)),
        "mean_steps": float(np.mean(steps)),
        "win_rate": float(np.mean([h >= 0.7 for h in healths])),
    }


def train_and_eval(name, cls, params, save_dir):
    print(f"\n{'='*50}")
    print(f"Training: {name}")
    print(f"{'='*50}")
    os.makedirs(save_dir, exist_ok=True)

    env = DeforestationEnv()
    eval_env = DeforestationEnv()

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=save_dir,
        eval_freq=10000,
        n_eval_episodes=10,
        deterministic=True,
    )

    model = cls("MlpPolicy", env, verbose=0, **params)
    model.learn(total_timesteps=TIMESTEPS, callback=eval_cb)
    model.save(os.path.join(save_dir, "final_model"))

    results = evaluate(model)
    results["params"] = {k: str(v) for k, v in params.items()}
    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"  reward={results['mean_reward']:.1f} "
          f"health={results['mean_health']:.1%} "
          f"steps={results['mean_steps']:.0f} "
          f"win={results['win_rate']:.0%}")

    env.close()
    eval_env.close()
    return results


def train_reinforce_single(name, config, save_dir):
    print(f"\n{'='*50}")
    print(f"Training: {name}")
    print(f"{'='*50}")
    os.makedirs(save_dir, exist_ok=True)

    env = DeforestationEnv()
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = ReinforceAgent(
        obs_size, n_actions,
        lr=config["lr"], gamma=config["gamma"]
    )

    best_reward = float("-inf")
    all_rewards = []

    for ep in range(REINFORCE_EPISODES):
        obs, _ = env.reset()
        ep_reward = 0
        while True:
            action = agent.select_action(obs)
            obs, reward, term, trunc, info = env.step(action)
            agent.rewards.append(reward)
            ep_reward += reward
            if term or trunc:
                break
        agent.update()
        all_rewards.append(ep_reward)

        if ep_reward > best_reward:
            best_reward = ep_reward
            agent.save(os.path.join(save_dir, "best_model.pt"))

        if (ep + 1) % 200 == 0:
            avg = np.mean(all_rewards[-50:])
            print(f"  Ep {ep+1}/{REINFORCE_EPISODES} avg={avg:.1f}")

    agent.save(os.path.join(save_dir, "final_model.pt"))
    np.save(os.path.join(save_dir, "rewards.npy"), np.array(all_rewards))

    results = evaluate(agent)
    results["params"] = config
    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"  reward={results['mean_reward']:.1f} "
          f"health={results['mean_health']:.1%} "
          f"win={results['win_rate']:.0%}")

    env.close()
    return results


DQN_CONFIGS = [
    {"name": "dqn_baseline", "learning_rate": 1e-3, "buffer_size": 50000, "batch_size": 64, "gamma": 0.99, "exploration_fraction": 0.3, "exploration_final_eps": 0.05, "target_update_interval": 500, "learning_starts": 1000},
    {"name": "dqn_low_lr", "learning_rate": 1e-4, "buffer_size": 50000, "batch_size": 64, "gamma": 0.99, "exploration_fraction": 0.3, "exploration_final_eps": 0.05, "target_update_interval": 500, "learning_starts": 1000},
    {"name": "dqn_high_lr", "learning_rate": 5e-3, "buffer_size": 50000, "batch_size": 64, "gamma": 0.99, "exploration_fraction": 0.3, "exploration_final_eps": 0.05, "target_update_interval": 500, "learning_starts": 1000},
    {"name": "dqn_large_buffer", "learning_rate": 1e-3, "buffer_size": 100000, "batch_size": 128, "gamma": 0.99, "exploration_fraction": 0.3, "exploration_final_eps": 0.05, "target_update_interval": 500, "learning_starts": 2000},
    {"name": "dqn_small_batch", "learning_rate": 1e-3, "buffer_size": 50000, "batch_size": 32, "gamma": 0.99, "exploration_fraction": 0.3, "exploration_final_eps": 0.05, "target_update_interval": 250, "learning_starts": 1000},
    {"name": "dqn_high_explore", "learning_rate": 1e-3, "buffer_size": 50000, "batch_size": 64, "gamma": 0.99, "exploration_fraction": 0.5, "exploration_final_eps": 0.1, "target_update_interval": 500, "learning_starts": 1000},
    {"name": "dqn_low_gamma", "learning_rate": 1e-3, "buffer_size": 50000, "batch_size": 64, "gamma": 0.95, "exploration_fraction": 0.3, "exploration_final_eps": 0.05, "target_update_interval": 500, "learning_starts": 1000},
    {"name": "dqn_fast_target", "learning_rate": 1e-3, "buffer_size": 50000, "batch_size": 64, "gamma": 0.99, "exploration_fraction": 0.3, "exploration_final_eps": 0.05, "target_update_interval": 200, "learning_starts": 1000},
    {"name": "dqn_slow_target", "learning_rate": 1e-3, "buffer_size": 50000, "batch_size": 64, "gamma": 0.99, "exploration_fraction": 0.3, "exploration_final_eps": 0.05, "target_update_interval": 1000, "learning_starts": 1000},
    {"name": "dqn_aggressive", "learning_rate": 3e-3, "buffer_size": 30000, "batch_size": 128, "gamma": 0.98, "exploration_fraction": 0.2, "exploration_final_eps": 0.02, "target_update_interval": 300, "learning_starts": 500},
]

PPO_CONFIGS = [
    {"name": "ppo_baseline", "learning_rate": 3e-4, "n_steps": 1024, "batch_size": 64, "n_epochs": 10, "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.01},
    {"name": "ppo_low_lr", "learning_rate": 1e-4, "n_steps": 1024, "batch_size": 64, "n_epochs": 10, "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.01},
    {"name": "ppo_high_lr", "learning_rate": 1e-3, "n_steps": 1024, "batch_size": 64, "n_epochs": 10, "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.01},
    {"name": "ppo_wide_clip", "learning_rate": 3e-4, "n_steps": 1024, "batch_size": 64, "n_epochs": 10, "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.3, "ent_coef": 0.01},
    {"name": "ppo_narrow_clip", "learning_rate": 3e-4, "n_steps": 1024, "batch_size": 64, "n_epochs": 10, "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.1, "ent_coef": 0.01},
    {"name": "ppo_high_entropy", "learning_rate": 3e-4, "n_steps": 1024, "batch_size": 64, "n_epochs": 10, "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.05},
    {"name": "ppo_long_horizon", "learning_rate": 3e-4, "n_steps": 2048, "batch_size": 128, "n_epochs": 10, "gamma": 0.995, "gae_lambda": 0.98, "clip_range": 0.2, "ent_coef": 0.01},
    {"name": "ppo_short_horizon", "learning_rate": 3e-4, "n_steps": 256, "batch_size": 32, "n_epochs": 5, "gamma": 0.97, "gae_lambda": 0.9, "clip_range": 0.2, "ent_coef": 0.01},
    {"name": "ppo_many_epochs", "learning_rate": 1e-4, "n_steps": 1024, "batch_size": 64, "n_epochs": 20, "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.15, "ent_coef": 0.01},
    {"name": "ppo_tuned", "learning_rate": 3e-4, "n_steps": 1024, "batch_size": 128, "n_epochs": 10, "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.03},
]

A2C_CONFIGS = [
    {"name": "a2c_baseline", "learning_rate": 7e-4, "n_steps": 16, "gamma": 0.99, "gae_lambda": 0.95, "ent_coef": 0.01, "vf_coef": 0.5},
    {"name": "a2c_low_lr", "learning_rate": 1e-4, "n_steps": 16, "gamma": 0.99, "gae_lambda": 0.95, "ent_coef": 0.01, "vf_coef": 0.5},
    {"name": "a2c_high_lr", "learning_rate": 3e-3, "n_steps": 16, "gamma": 0.99, "gae_lambda": 0.95, "ent_coef": 0.01, "vf_coef": 0.5},
    {"name": "a2c_long_steps", "learning_rate": 7e-4, "n_steps": 64, "gamma": 0.99, "gae_lambda": 0.95, "ent_coef": 0.01, "vf_coef": 0.5},
    {"name": "a2c_short_steps", "learning_rate": 7e-4, "n_steps": 5, "gamma": 0.99, "gae_lambda": 0.95, "ent_coef": 0.01, "vf_coef": 0.5},
    {"name": "a2c_high_entropy", "learning_rate": 7e-4, "n_steps": 16, "gamma": 0.99, "gae_lambda": 0.95, "ent_coef": 0.05, "vf_coef": 0.5},
    {"name": "a2c_low_vf", "learning_rate": 7e-4, "n_steps": 16, "gamma": 0.99, "gae_lambda": 0.95, "ent_coef": 0.01, "vf_coef": 0.25},
    {"name": "a2c_high_vf", "learning_rate": 7e-4, "n_steps": 16, "gamma": 0.99, "gae_lambda": 0.95, "ent_coef": 0.01, "vf_coef": 0.8},
    {"name": "a2c_low_gamma", "learning_rate": 7e-4, "n_steps": 16, "gamma": 0.95, "gae_lambda": 0.9, "ent_coef": 0.01, "vf_coef": 0.5},
    {"name": "a2c_tuned", "learning_rate": 5e-4, "n_steps": 32, "gamma": 0.99, "gae_lambda": 0.98, "ent_coef": 0.02, "vf_coef": 0.4},
]

REINFORCE_CONFIGS = [
    {"name": "reinforce_baseline", "lr": 1e-3, "gamma": 0.99},
    {"name": "reinforce_low_lr", "lr": 1e-4, "gamma": 0.99},
    {"name": "reinforce_high_lr", "lr": 5e-3, "gamma": 0.99},
    {"name": "reinforce_low_gamma", "lr": 1e-3, "gamma": 0.95},
    {"name": "reinforce_high_gamma", "lr": 1e-3, "gamma": 0.999},
    {"name": "reinforce_slow", "lr": 5e-4, "gamma": 0.995},
    {"name": "reinforce_fast", "lr": 3e-3, "gamma": 0.97},
    {"name": "reinforce_balanced", "lr": 7e-4, "gamma": 0.99},
    {"name": "reinforce_cautious", "lr": 2e-4, "gamma": 0.998},
    {"name": "reinforce_bold", "lr": 2e-3, "gamma": 0.98},
]


def main():
    all_results = {"DQN": {}, "PPO": {}, "A2C": {}, "REINFORCE": {}}

    for cfg in DQN_CONFIGS:
        name = cfg.pop("name")
        r = train_and_eval(name, DQN, cfg, f"models/dqn/{name}")
        all_results["DQN"][name] = r
        cfg["name"] = name

    for cfg in PPO_CONFIGS:
        name = cfg.pop("name")
        r = train_and_eval(name, PPO, cfg, f"models/pg/ppo/{name}")
        all_results["PPO"][name] = r
        cfg["name"] = name

    for cfg in A2C_CONFIGS:
        name = cfg.pop("name")
        r = train_and_eval(name, A2C, cfg, f"models/pg/a2c/{name}")
        all_results["A2C"][name] = r
        cfg["name"] = name

    for cfg in REINFORCE_CONFIGS:
        name = cfg["name"]
        params = {k: v for k, v in cfg.items() if k != "name"}
        r = train_reinforce_single(name, params, f"models/pg/reinforce/{name}")
        all_results["REINFORCE"][name] = r

    with open("models/all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Save per-algo results too
    for algo in all_results:
        dir_map = {"DQN": "models/dqn", "PPO": "models/pg/ppo", "A2C": "models/pg/a2c", "REINFORCE": "models/pg/reinforce"}
        os.makedirs(dir_map[algo], exist_ok=True)
        with open(os.path.join(dir_map[algo], "all_results.json"), "w") as f:
            json.dump(all_results[algo], f, indent=2)

    print("\n" + "=" * 60)
    print("BEST PER ALGORITHM")
    print("=" * 60)
    for algo, results in all_results.items():
        if results:
            best = max(results, key=lambda n: results[n]["win_rate"])
            r = results[best]
            print(f"{algo:12s} | {best:25s} | win={r['win_rate']:.0%} health={r['mean_health']:.1%} reward={r['mean_reward']:.1f}")

    print("\nDone. Run: python analysis/compare.py")


if __name__ == "__main__":
    main()
