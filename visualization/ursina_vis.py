import sys
import os
import pickle
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

GRID = 15
ACTION_NAMES = [
    "Up", "Down", "Left", "Right",
    "Scan", "Retardant", "Rangers", "Base"
]


def run_episode_data():
    from stable_baselines3 import A2C, PPO, DQN
    from environment.forest_env import DeforestationEnv

    env = DeforestationEnv()
    obs, _ = env.reset()

    paths = [
        ("models/best_model", A2C),
        ("models/retrained/a2c_retrained", A2C),
        ("models/retrained/dqn_retrained", DQN),
        ("models/retrained/ppo_retrained", PPO),
    ]

    model = None
    for rel_path, cls in paths:
        full_path = os.path.join(PROJECT_ROOT, rel_path)
        if os.path.exists(full_path + ".zip"):
            try:
                model = cls.load(full_path)
                break
            except Exception:
                continue

    frames = []
    done = False
    total_reward = 0

    state = env.get_full_state()
    state["action"] = -1
    state["reward"] = 0
    state["total_reward"] = 0
    frames.append(state)

    while not done:
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

        state = env.get_full_state()
        state["action"] = action
        state["reward"] = float(reward)
        state["total_reward"] = float(total_reward)
        frames.append(state)

    env.close()
    return frames


def main():
    print("Pre-computing episode with trained agent...")
    frames = run_episode_data()
    print(f"Episode: {len(frames)} frames. Starting 3D visualization...")

    from ursina import (
        Ursina, Entity, Text, EditorCamera,
        DirectionalLight, AmbientLight, color, time,
    )
    from environment.forest_env import HEALTHY, AT_RISK, DEFORESTING, DESTROYED

    TILE_COLORS = {
        HEALTHY: color.rgb(45, 120, 35),
        AT_RISK: color.rgb(210, 180, 20),
        DEFORESTING: color.rgb(220, 90, 10),
        DESTROYED: color.rgb(74, 40, 10),
    }

    app = Ursina(
        title="UAV Deforestation Detection - 3D",
        borderless=False,
    )

    camera_entity = None

    tiles = []
    for r in range(GRID):
        row = []
        for c in range(GRID):
            tile = Entity(
                model="cube",
                color=TILE_COLORS[HEALTHY],
                position=(c - GRID / 2, 0, r - GRID / 2),
                scale=(0.9, 0.3, 0.9),
            )
            row.append(tile)
        tiles.append(row)

    trees = []
    for r in range(GRID):
        for c in range(GRID):
            trunk = Entity(
                model="cube",
                color=color.rgb(80, 50, 25),
                position=(c - GRID / 2, 0.4, r - GRID / 2),
                scale=(0.1, 0.5, 0.1),
            )
            canopy = Entity(
                model="sphere",
                color=color.rgb(30, 100, 25),
                position=(c - GRID / 2, 0.8, r - GRID / 2),
                scale=0.4,
            )
            trees.append((r, c, trunk, canopy))

    drone = Entity(
        model="sphere",
        color=color.cyan,
        position=(0 - GRID / 2, 2, 0 - GRID / 2),
        scale=0.6,
    )

    drone_ring = Entity(
        model="cube",
        color=color.rgba(0, 200, 255, 80),
        position=(0 - GRID / 2, 0.35, 0 - GRID / 2),
        scale=(0.9, 0.05, 0.9),
    )

    truck = Entity(
        model="cube",
        color=color.red,
        position=(0, 0.5, 0),
        scale=(0.8, 0.5, 0.5),
    )

    hud_health = Text(
        text="Health: --", position=(-0.85, 0.47),
        scale=1.5, color=color.white,
    )
    hud_fuel = Text(
        text="Fuel: --", position=(-0.85, 0.43),
        scale=1.3, color=color.white,
    )
    hud_step = Text(
        text="Step: --", position=(-0.85, 0.39),
        scale=1.3, color=color.white,
    )
    hud_action = Text(
        text="Action: --", position=(-0.85, 0.35),
        scale=1.3, color=color.yellow,
    )
    hud_reward = Text(
        text="Reward: 0.0", position=(-0.85, 0.31),
        scale=1.3, color=color.orange,
    )
    Text(
        text="R=restart | A=pause/play",
        position=(-0.85, -0.47), scale=1.0, color=color.gray,
    )

    state = {"idx": 0, "paused": False, "timer": 0}

    from direct.showbase.ShowBase import DirectObject
    listener = DirectObject.DirectObject()

    def show_frame(idx):
        if idx >= len(frames):
            return
        f = frames[idx]
        grid = f["grid"]

        for r in range(GRID):
            for c in range(GRID):
                val = grid[r][c]
                retardant_key = f"{r},{c}"
                if retardant_key in f.get("retardant_tiles", {}):
                    tiles[r][c].color = color.azure
                else:
                    tiles[r][c].color = TILE_COLORS[val]
                heights = {0: 0.3, 1: 0.25, 2: 0.15, 3: 0.05}
                tiles[r][c].scale_y = heights.get(val, 0.3)

        for r, c, trunk_e, canopy_e in trees:
            val = grid[r][c]
            if val == HEALTHY:
                trunk_e.visible = True
                canopy_e.visible = True
                canopy_e.color = color.rgb(30, 100, 25)
            elif val == AT_RISK:
                trunk_e.visible = True
                canopy_e.visible = True
                canopy_e.color = color.rgb(180, 160, 20)
            elif val == DEFORESTING:
                trunk_e.visible = True
                canopy_e.visible = True
                canopy_e.color = color.rgb(200, 60, 10)
            else:
                trunk_e.visible = False
                canopy_e.visible = False

        dp = f["drone_pos"]
        drone.position = (
            dp[1] - GRID / 2,
            2 + np.sin(idx * 0.3) * 0.2,
            dp[0] - GRID / 2,
        )
        drone_ring.position = (
            dp[1] - GRID / 2, 0.35, dp[0] - GRID / 2,
        )

        tp = f.get("truck_pos")
        if tp and f.get("truck_active"):
            truck.visible = True
            truck.position = (
                tp[1] - GRID / 2, 0.5, tp[0] - GRID / 2,
            )
        else:
            truck.visible = False

        h = f["forest_health"]
        hc = "00ff00" if h >= 0.7 else "ffaa00" if h >= 0.5 else "ff3333"
        hud_health.text = f"Health: {h:.1%}"
        hud_health.color = color.hex(hc)
        hud_fuel.text = f"Fuel: {f['fuel']}/200"
        hud_step.text = f"Step: {f['step']}/200"
        act = f.get("action", -1)
        if 0 <= act < 8:
            hud_action.text = f"Action: {ACTION_NAMES[act]}"
        hud_reward.text = f"Reward: {f.get('total_reward', 0):.1f}"

    def update():
        if state["paused"]:
            return
        state["timer"] += time.dt
        if state["timer"] >= 0.35:
            state["timer"] = 0
            if state["idx"] < len(frames):
                show_frame(state["idx"])
                state["idx"] += 1
            else:
                hud_action.text = "EPISODE COMPLETE"

    def on_key(key):
        if key == "r":
            state["idx"] = 0
            state["timer"] = 0
            state["paused"] = False
            show_frame(0)
        elif key == "a":
            state["paused"] = not state["paused"]

    app.update = update
    listener.accept("r", on_key, ["r"])
    listener.accept("a", on_key, ["a"])

    from ursina import camera
    camera.position = (0, 25, -20)
    camera.rotation_x = 50
    EditorCamera()

    show_frame(0)
    app.run()


if __name__ == "__main__":
    main()
