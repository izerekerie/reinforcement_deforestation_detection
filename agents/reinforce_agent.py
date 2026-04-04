import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from environment.forest_env import DeforestationEnv


class PolicyNetwork(nn.Module):
    def __init__(self, obs_size, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)


class ReinforceAgent:
    def __init__(self, obs_size, n_actions, lr=1e-3, gamma=0.99):
        self.gamma = gamma
        self.policy = PolicyNetwork(obs_size, n_actions)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.log_probs = []
        self.rewards = []

    def select_action(self, obs):
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        probs = self.policy(obs_tensor)
        dist = Categorical(probs)
        action = dist.sample()
        self.log_probs.append(dist.log_prob(action))
        return action.item()

    def update(self):
        returns = []
        g = 0
        for r in reversed(self.rewards):
            g = r + self.gamma * g
            returns.insert(0, g)

        returns = torch.tensor(returns, dtype=torch.float32)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = 0
        for log_prob, g in zip(self.log_probs, returns):
            loss -= log_prob * g

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.log_probs = []
        self.rewards = []
        return loss.item()

    def save(self, path):
        torch.save(self.policy.state_dict(), path)

    def load(self, path):
        self.policy.load_state_dict(torch.load(path, weights_only=True))

    def predict(self, obs, deterministic=False):
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            probs = self.policy(obs_tensor)
        if deterministic:
            action = torch.argmax(probs, dim=-1).item()
        else:
            dist = Categorical(probs)
            action = dist.sample().item()
        return action, None


def train_reinforce(n_episodes=500, save_path="models/reinforce_forest"):
    env = DeforestationEnv()
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = ReinforceAgent(obs_size, n_actions)
    os.makedirs(save_path, exist_ok=True)

    all_rewards = []
    best_reward = float("-inf")

    for episode in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0

        while True:
            action = agent.select_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            agent.rewards.append(reward)
            episode_reward += reward

            if terminated or truncated:
                break

        loss = agent.update()
        all_rewards.append(episode_reward)

        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save(f"{save_path}/best_model.pt")

        if (episode + 1) % 50 == 0:
            avg = np.mean(all_rewards[-50:])
            print(f"Episode {episode + 1}/{n_episodes} | Avg Reward: {avg:.1f} | Best: {best_reward:.1f}")

    agent.save(f"{save_path}/reinforce_final.pt")
    np.save(f"{save_path}/rewards.npy", np.array(all_rewards))
    env.close()
    return agent, all_rewards


if __name__ == "__main__":
    train_reinforce()
