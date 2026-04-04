import sys
import os
import json
import argparse
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment.forest_env import DeforestationEnv


ACTION_NAMES = ["Up", "Down", "Left", "Right", "Scan", "Retardant", "Rangers", "Base"]


def find_best_model():
    candidates = []

    dqn_results = "models/dqn/all_results.json"
    if os.path.exists(dqn_results):
        with open(dqn_results) as f:
            data = json.load(f)
        for name, res in data.items():
            path = f"models/dqn/{name}/final_model.zip"
            if os.path.exists(path):
                candidates.append(("DQN", name, res["mean_reward"], path))

    for algo in ["ppo", "a2c"]:
        results_path = f"models/pg/{algo}/all_results.json"
        if os.path.exists(results_path):
            with open(results_path) as f:
                data = json.load(f)
            for name, res in data.items():
                path = f"models/pg/{algo}/{name}/final_model.zip"
                if os.path.exists(path):
                    candidates.append((algo.upper(), name, res["mean_reward"], path))

    reinforce_path = "models/pg/reinforce/all_results.json"
    if os.path.exists(reinforce_path):
        with open(reinforce_path) as f:
            data = json.load(f)
        for name, res in data.items():
            path = f"models/pg/reinforce/{name}/final_model.pt"
            if os.path.exists(path):
                candidates.append(("REINFORCE", name, res["mean_reward"], path))

    for algo_dir, cls_name in [
        ("models/ppo_forest/ppo_final.zip", "PPO"),
        ("models/dqn_forest/dqn_final.zip", "DQN"),
        ("models/a2c_forest/a2c_final.zip", "A2C"),
    ]:
        if os.path.exists(algo_dir):
            candidates.append((cls_name, "legacy", -999, algo_dir))

    if not candidates:
        return None, None, None

    best = max(candidates, key=lambda x: x[2])

    if os.path.exists("models/best_model.zip"):
        return "A2C", "best_model", "models/best_model.zip"

    retrained = [
        ("PPO", "best_retrained", "models/pg/ppo/best_retrained.zip"),
        ("PPO", "retrained", "models/pg/ppo/retrained_model.zip"),
        ("A2C", "retrained", "models/pg/a2c/retrained_model.zip"),
    ]
    for algo, name, path in retrained:
        if os.path.exists(path):
            return algo, name, path

    for algo_dir, cls_name in [
        ("models/pg/a2c", "A2C"),
        ("models/pg/ppo", "PPO"),
    ]:
        results_path = f"{algo_dir}/all_results.json"
        if os.path.exists(results_path):
            with open(results_path) as f:
                data = json.load(f)
            best_name = max(
                data,
                key=lambda n: data[n].get("win_rate", 0)
            )
            if data[best_name].get("win_rate", 0) >= 0.8:
                path = f"{algo_dir}/{best_name}/final_model.zip"
                if os.path.exists(path):
                    return cls_name, best_name, path

    return best[0], best[1], best[3]


def load_model(algo, path):
    if algo == "REINFORCE":
        from agents.reinforce_agent import ReinforceAgent
        env = DeforestationEnv()
        agent = ReinforceAgent(env.observation_space.shape[0], env.action_space.n)
        agent.load(path)
        env.close()
        return agent
    else:
        from stable_baselines3 import DQN, PPO, A2C
        cls_map = {"DQN": DQN, "PPO": PPO, "A2C": A2C}
        return cls_map[algo].load(path.replace(".zip", ""))


def run_terminal(model, n_episodes=5):
    env = DeforestationEnv()
    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0
        step = 0

        print(f"\n{'='*60}")
        print(f"Episode {ep + 1}")
        print(f"{'='*60}")

        while True:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1

            if step % 10 == 0 or terminated or truncated:
                print(f"  Step {step:3d} | {ACTION_NAMES[action]:10s} | "
                      f"R={reward:+7.1f} | Health={info['forest_health']:.1%} | "
                      f"Fuel={info['fuel']:3d} | Threats={info['detected_threats']}")

            if terminated or truncated:
                result = "WIN" if info["forest_health"] >= 0.7 else "LOSS"
                print(f"\n  Total Reward: {total_reward:.1f}")
                print(f"  Final Health: {info['forest_health']:.1%}")
                print(f"  Result: {result}")
                break

    env.close()


def run_visual(model):
    from environment.rendering import run_with_model
    run_with_model(model, deterministic=True)


def run_3d(model):
    from visualization.ursina_vis import main as ursina_main
    ursina_main()


def run_dashboard():
    import uvicorn
    import webbrowser
    import threading

    def open_browser():
        import time
        time.sleep(1.5)
        webbrowser.open("http://localhost:8000")

    print("Starting dashboard at http://localhost:8000")
    threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=False)


def main():
    parser = argparse.ArgumentParser(description="UAV Deforestation Detection - Run Best Model")
    parser.add_argument("--mode", choices=["terminal", "visual", "3d", "dashboard"],
                        default="terminal", help="Display mode")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes for terminal mode")
    parser.add_argument("--model", type=str, default=None, help="Path to specific model file")
    parser.add_argument("--algo", type=str, default=None, help="Algorithm name (DQN/PPO/A2C/REINFORCE)")
    args = parser.parse_args()

    if args.mode == "dashboard":
        run_dashboard()
        return

    if args.model and args.algo:
        algo = args.algo.upper()
        model = load_model(algo, args.model)
        print(f"Loaded {algo} from {args.model}")
    else:
        algo, config_name, path = find_best_model()
        if algo is None:
            print("No trained models found.")
            print("Run training first:")
            print("  python training/dqn_training.py")
            print("  python training/pg_training.py")
            return
        model = load_model(algo, path)
        print(f"Best model: {algo} ({config_name}) from {path}")

    if args.mode == "terminal":
        run_terminal(model, args.episodes)
    elif args.mode == "visual":
        run_visual(model)
    elif args.mode == "3d":
        run_3d(model)


if __name__ == "__main__":
    main()
