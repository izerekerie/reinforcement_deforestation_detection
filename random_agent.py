import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment.forest_env import DeforestationEnv
from environment.rendering import ForestRenderer

ACTION_NAMES = ["Up", "Down", "Left", "Right", "Scan", "Retardant", "Rangers", "Base"]


def run_random_terminal(n_episodes=3):
    env = DeforestationEnv()
    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0
        step = 0

        print(f"\n{'='*50}")
        print(f"Episode {ep + 1}")
        print(f"{'='*50}")

        while True:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1

            if step % 20 == 0:
                print(f"  Step {step:3d} | Action: {ACTION_NAMES[action]:10s} | "
                      f"Reward: {reward:+7.1f} | Health: {info['forest_health']:.1%} | "
                      f"Fuel: {info['fuel']:3d}")

            if terminated or truncated:
                print(f"\n  Episode ended at step {step}")
                print(f"  Total Reward: {total_reward:.1f}")
                print(f"  Final Health: {info['forest_health']:.1%}")
                print(f"  Threats Detected: {info['detected_threats']}")
                result = "WIN" if info['forest_health'] >= 0.7 else "LOSS"
                print(f"  Result: {result}")
                break

    env.close()


def run_random_visual():
    env = DeforestationEnv()
    obs, _ = env.reset()
    renderer = ForestRenderer(env)
    paused = False
    done = False

    print("Random Agent Visualization")
    print("Controls: SPACE=step | A=pause/resume | R=reset | ESC=quit")

    running = True
    while running:
        event = renderer.handle_events()
        if event == "quit":
            running = False
        elif event == "step" and not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            renderer.total_reward += reward
            renderer.log_action(action, reward)
            done = terminated or truncated
            if done:
                result = "WIN" if info["forest_health"] >= 0.7 else "LOSS"
                print(f"Episode ended: {result} | Health: {info['forest_health']:.1%}")
        elif event == "reset":
            obs, _ = env.reset()
            renderer.total_reward = 0
            renderer.action_log = []
            done = False
            paused = False
        elif event == "auto":
            paused = not paused

        if not paused and not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            renderer.total_reward += reward
            renderer.log_action(action, reward)
            done = terminated or truncated

        renderer.render()
        renderer.tick(8)

    renderer.close()
    env.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Random agent demo")
    parser.add_argument("--visual", action="store_true", help="Run with 2D visualization")
    args = parser.parse_args()

    if args.visual:
        run_random_visual()
    else:
        run_random_terminal()
