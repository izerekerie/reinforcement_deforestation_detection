import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.dqn_agent import train_dqn
from agents.ppo_agent import train_ppo
from agents.a2c_agent import train_a2c
from agents.reinforce_agent import train_reinforce


TIMESTEPS = 50000
REINFORCE_EPISODES = 500


def evaluate_sb3_model(model, env, n_episodes=20):
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
    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_health": float(np.mean(healths)),
        "win_rate": float(np.mean([h >= 0.7 for h in healths])),
    }


def evaluate_reinforce(agent, env, n_episodes=20):
    rewards = []
    healths = []
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
    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_health": float(np.mean(healths)),
        "win_rate": float(np.mean([h >= 0.7 for h in healths])),
    }


def main():
    os.makedirs("models", exist_ok=True)
    results = {}

    from environment.forest_env import DeforestationEnv

    print("=" * 60)
    print("Training DQN")
    print("=" * 60)
    dqn_model = train_dqn(total_timesteps=TIMESTEPS)
    results["DQN"] = evaluate_sb3_model(dqn_model, DeforestationEnv())

    print("=" * 60)
    print("Training PPO")
    print("=" * 60)
    ppo_model = train_ppo(total_timesteps=TIMESTEPS)
    results["PPO"] = evaluate_sb3_model(ppo_model, DeforestationEnv())

    print("=" * 60)
    print("Training A2C")
    print("=" * 60)
    a2c_model = train_a2c(total_timesteps=TIMESTEPS)
    results["A2C"] = evaluate_sb3_model(a2c_model, DeforestationEnv())

    print("=" * 60)
    print("Training REINFORCE")
    print("=" * 60)
    reinforce_agent, _ = train_reinforce(n_episodes=REINFORCE_EPISODES)
    results["REINFORCE"] = evaluate_reinforce(reinforce_agent, DeforestationEnv())

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for name, metrics in results.items():
        print(f"\n{name}:")
        print(f"  Mean Reward:   {metrics['mean_reward']:.2f} +/- {metrics['std_reward']:.2f}")
        print(f"  Mean Health:   {metrics['mean_health']:.2%}")
        print(f"  Win Rate:      {metrics['win_rate']:.2%}")

    with open("models/results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to models/results.json")


if __name__ == "__main__":
    main()
