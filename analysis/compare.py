import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3 import DQN, PPO, A2C
from environment.forest_env import DeforestationEnv
from agents.reinforce_agent import ReinforceAgent


def find_best_per_algo():
    best_models = {}

    dqn_results = "models/dqn/all_results.json"
    if os.path.exists(dqn_results):
        with open(dqn_results) as f:
            data = json.load(f)
        best_name = max(data, key=lambda n: data[n]["mean_reward"])
        path = f"models/dqn/{best_name}/final_model"
        if os.path.exists(path + ".zip"):
            best_models["DQN"] = {"path": path, "config": best_name, "class": DQN}

    for algo, cls in [("ppo", PPO), ("a2c", A2C)]:
        results_path = f"models/pg/{algo}/all_results.json"
        if os.path.exists(results_path):
            with open(results_path) as f:
                data = json.load(f)
            best_name = max(data, key=lambda n: data[n]["mean_reward"])
            path = f"models/pg/{algo}/{best_name}/final_model"
            if os.path.exists(path + ".zip"):
                best_models[algo.upper()] = {"path": path, "config": best_name, "class": cls}

    reinforce_results = "models/pg/reinforce/all_results.json"
    if os.path.exists(reinforce_results):
        with open(reinforce_results) as f:
            data = json.load(f)
        best_name = max(data, key=lambda n: data[n]["mean_reward"])
        path = f"models/pg/reinforce/{best_name}/final_model.pt"
        if os.path.exists(path):
            best_models["REINFORCE"] = {"path": path, "config": best_name}

    for fallback_path, name, cls in [
        ("models/dqn_forest/dqn_final", "DQN", DQN),
        ("models/ppo_forest/ppo_final", "PPO", PPO),
        ("models/a2c_forest/a2c_final", "A2C", A2C),
    ]:
        if name not in best_models and os.path.exists(fallback_path + ".zip"):
            best_models[name] = {"path": fallback_path, "config": "default", "class": cls}

    return best_models


def evaluate(model_info, n_episodes=30):
    env = DeforestationEnv()
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    results = {}
    for name, info in model_info.items():
        if name == "REINFORCE":
            model = ReinforceAgent(obs_size, n_actions)
            model.load(info["path"])
        else:
            model = info["class"].load(info["path"])

        rewards, healths, steps, detections = [], [], [], []
        for _ in range(n_episodes):
            obs, _ = env.reset()
            total_reward = 0
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, ep_info = env.step(action)
                total_reward += reward
                if terminated or truncated:
                    rewards.append(total_reward)
                    healths.append(ep_info["forest_health"])
                    steps.append(ep_info["step"])
                    detections.append(ep_info["detected_threats"])
                    break

        results[name] = {
            "rewards": rewards,
            "healths": healths,
            "steps": steps,
            "detections": detections,
            "config": info["config"],
        }

    env.close()
    return results


def plot_comparison(results):
    os.makedirs("analysis", exist_ok=True)
    names = list(results.keys())
    colors = {"DQN": "#2196F3", "PPO": "#4CAF50", "A2C": "#FF9800", "REINFORCE": "#9C27B0"}
    c = [colors.get(n, "#888") for n in names]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Algorithm Comparison - UAV Deforestation Detection", fontsize=14, fontweight="bold")

    rewards_data = [results[n]["rewards"] for n in names]
    bp = axes[0, 0].boxplot(rewards_data, labels=names, patch_artist=True)
    for patch, color in zip(bp["boxes"], c):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[0, 0].set_title("Episode Reward Distribution")
    axes[0, 0].set_ylabel("Total Reward")
    axes[0, 0].grid(True, alpha=0.3)

    mean_healths = [np.mean(results[n]["healths"]) for n in names]
    bars = axes[0, 1].bar(names, mean_healths, color=c, alpha=0.8)
    axes[0, 1].axhline(y=0.7, color="red", linestyle="--", linewidth=1.5, label="Win threshold (70%)")
    axes[0, 1].set_title("Average Forest Health at Episode End")
    axes[0, 1].set_ylabel("Forest Health")
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    for bar, val in zip(bars, mean_healths):
        axes[0, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                        f"{val:.1%}", ha="center", fontsize=10)

    win_rates = [np.mean([h >= 0.7 for h in results[n]["healths"]]) for n in names]
    bars = axes[1, 0].bar(names, win_rates, color=c, alpha=0.8)
    axes[1, 0].set_title("Win Rate (Health >= 70%)")
    axes[1, 0].set_ylabel("Win Rate")
    axes[1, 0].set_ylim(0, 1.1)
    axes[1, 0].grid(True, alpha=0.3)
    for bar, val in zip(bars, win_rates):
        axes[1, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                        f"{val:.0%}", ha="center", fontsize=10)

    mean_det = [np.mean(results[n]["detections"]) for n in names]
    axes[1, 1].bar(names, mean_det, color=c, alpha=0.8)
    axes[1, 1].set_title("Average Threats Detected per Episode")
    axes[1, 1].set_ylabel("Threats Detected")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("analysis/comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved analysis/comparison.png")


def plot_training_curves():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training Curves", fontsize=14, fontweight="bold")

    for algo, color, label in [
        ("dqn", "#2196F3", "DQN"),
        ("ppo", "#4CAF50", "PPO"),
        ("a2c", "#FF9800", "A2C"),
    ]:
        for search_dir in [f"models/{algo}", f"models/pg/{algo}", f"models/{algo}_forest"]:
            eval_files = []
            if os.path.isdir(search_dir):
                for root, dirs, files in os.walk(search_dir):
                    for f in files:
                        if f == "evaluations.npz":
                            eval_files.append(os.path.join(root, f))

            for ef in eval_files[:1]:
                data = np.load(ef)
                timesteps = data["timesteps"]
                mean_rewards = np.mean(data["results"], axis=1)
                axes[0].plot(timesteps, mean_rewards, color=color, label=label, alpha=0.8)
                break

    reinforce_dirs = ["models/pg/reinforce", "models/reinforce_forest"]
    for rd in reinforce_dirs:
        if not os.path.isdir(rd):
            continue
        for root, dirs, files in os.walk(rd):
            if "rewards.npy" in files:
                rewards = np.load(os.path.join(root, "rewards.npy"))
                window = min(20, len(rewards) // 3) if len(rewards) > 3 else 1
                if window > 1:
                    smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
                    axes[0].plot(np.linspace(0, 50000, len(smoothed)), smoothed,
                                 color="#9C27B0", label="REINFORCE", alpha=0.8)
                break
        break

    axes[0].set_xlabel("Training Steps")
    axes[0].set_ylabel("Mean Reward")
    axes[0].set_title("Evaluation Reward During Training")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    hp_data = {}
    for algo in ["dqn", "ppo", "a2c", "reinforce"]:
        for path in [f"models/{algo}/all_results.json", f"models/pg/{algo}/all_results.json"]:
            if os.path.exists(path):
                with open(path) as f:
                    data = json.load(f)
                rewards = [v["mean_reward"] for v in data.values()]
                hp_data[algo.upper()] = rewards
                break

    if hp_data:
        positions = list(range(len(hp_data)))
        bp = axes[1].boxplot(list(hp_data.values()), labels=list(hp_data.keys()), patch_artist=True)
        algo_colors = {"DQN": "#2196F3", "PPO": "#4CAF50", "A2C": "#FF9800", "REINFORCE": "#9C27B0"}
        for patch, name in zip(bp["boxes"], hp_data.keys()):
            patch.set_facecolor(algo_colors.get(name, "#888"))
            patch.set_alpha(0.7)
        axes[1].set_title("Hyperparameter Sensitivity (10 configs each)")
        axes[1].set_ylabel("Mean Reward")
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("analysis/training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved analysis/training_curves.png")


def generate_summary(results):
    rows = []
    for name in results:
        rows.append({
            "Algorithm": name,
            "Config": results[name]["config"],
            "Mean Reward": f"{np.mean(results[name]['rewards']):.2f}",
            "Std Reward": f"{np.std(results[name]['rewards']):.2f}",
            "Mean Health": f"{np.mean(results[name]['healths']):.2%}",
            "Win Rate": f"{np.mean([h >= 0.7 for h in results[name]['healths']]):.2%}",
            "Avg Steps": f"{np.mean(results[name]['steps']):.0f}",
            "Avg Detections": f"{np.mean(results[name]['detections']):.1f}",
        })

    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False))
    df.to_csv("analysis/summary.csv", index=False)
    print("\nSaved analysis/summary.csv")
    return df


def generate_hyperparameter_table():
    rows = []
    for algo, path in [
        ("DQN", "models/dqn/all_results.json"),
        ("PPO", "models/pg/ppo/all_results.json"),
        ("A2C", "models/pg/a2c/all_results.json"),
        ("REINFORCE", "models/pg/reinforce/all_results.json"),
    ]:
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        for name, res in data.items():
            row = {"Algorithm": algo, "Config": name}
            row["Mean Reward"] = f"{res['mean_reward']:.2f}"
            row["Mean Health"] = f"{res['mean_health']:.2%}"
            row["Win Rate"] = f"{res['win_rate']:.0%}"
            if "config" in res:
                for k, v in res["config"].items():
                    if k != "name":
                        row[k] = v
            rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv("analysis/hyperparameter_results.csv", index=False)
        print(f"\nSaved analysis/hyperparameter_results.csv ({len(rows)} runs)")
        return df
    return None


def main():
    print("Loading best model per algorithm...")
    model_info = find_best_per_algo()

    if not model_info:
        print("No trained models found. Run training first.")
        return

    print(f"Found: {list(model_info.keys())}")
    for name, info in model_info.items():
        print(f"  {name}: {info['config']} ({info['path']})")

    print("\nEvaluating (30 episodes each)...")
    results = evaluate(model_info)

    plot_comparison(results)
    plot_training_curves()
    generate_summary(results)
    generate_hyperparameter_table()


if __name__ == "__main__":
    main()
