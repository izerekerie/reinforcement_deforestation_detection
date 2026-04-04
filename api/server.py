import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
import uvicorn
import numpy as np

from environment.forest_env import DeforestationEnv
from stable_baselines3 import PPO, DQN, A2C


app = FastAPI(title="UAV Deforestation Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

env = None
model = None
current_obs = None
episode_done = False
episode_history = []
mission_log = []


def load_best_model():
    global model
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_paths = [
        ("models/best_model", A2C),
        ("models/retrained/a2c_retrained", A2C),
        ("models/retrained/best_model", A2C),
        ("models/pg/a2c/a2c_baseline/final_model", A2C),
        ("models/retrained/ppo_retrained", PPO),
        ("models/retrained/dqn_retrained", DQN),
    ]
    for rel_path, cls in model_paths:
        full_path = os.path.join(base, rel_path)
        if os.path.exists(full_path + ".zip"):
            model = cls.load(full_path)
            print(f"Loaded model from {rel_path}")
            return
    print("No trained model found. Using random actions.")


@app.on_event("startup")
def startup():
    load_best_model()


@app.get("/")
def root():
    dashboard_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "dashboard", "index.html"
    )
    if os.path.exists(dashboard_path):
        return FileResponse(dashboard_path, media_type="text/html")
    return HTMLResponse("<h1>UAV Deforestation Detection API</h1><p>Dashboard not found.</p>")


@app.post("/reset")
def reset_env():
    global env, current_obs, episode_done
    env = DeforestationEnv()
    current_obs, _ = env.reset()
    episode_done = False
    return env.get_full_state()


@app.post("/step")
def step_env():
    global current_obs, episode_done
    if env is None or episode_done:
        return {"error": "Call /reset first"}

    if model is not None:
        action, _ = model.predict(current_obs, deterministic=True)
        action = int(action)
    else:
        action = env.action_space.sample()

    current_obs, reward, terminated, truncated, info = env.step(action)
    episode_done = terminated or truncated

    state = env.get_full_state()
    state["reward"] = float(reward)
    state["action"] = action
    state["terminated"] = terminated
    state["truncated"] = truncated
    state["episode_done"] = episode_done
    return state


@app.post("/step/{action}")
def step_manual(action: int):
    global current_obs, episode_done
    if env is None or episode_done:
        return {"error": "Call /reset first"}

    action = max(0, min(7, action))
    current_obs, reward, terminated, truncated, info = env.step(action)
    episode_done = terminated or truncated

    state = env.get_full_state()
    state["reward"] = float(reward)
    state["action"] = action
    state["terminated"] = terminated
    state["truncated"] = truncated
    state["episode_done"] = episode_done
    return state


@app.get("/state")
def get_state():
    if env is None:
        return {"error": "Call /reset first"}
    return env.get_full_state()


@app.post("/run_episode")
def run_full_episode():
    global env, current_obs, episode_done
    env = DeforestationEnv()
    current_obs, _ = env.reset()
    episode_done = False

    states = [env.get_full_state()]
    total_reward = 0

    while not episode_done:
        if model is not None:
            action, _ = model.predict(current_obs, deterministic=True)
            action = int(action)
        else:
            action = env.action_space.sample()

        current_obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        episode_done = terminated or truncated

        state = env.get_full_state()
        state["action"] = action
        state["reward"] = float(reward)
        states.append(state)

    return {
        "total_reward": float(total_reward),
        "n_steps": len(states),
        "final_health": states[-1]["forest_health"],
        "states": states,
    }


# ── Production Pipeline Endpoints ──────────────────────────────

@app.get("/analytics")
def get_analytics():
    """Production endpoint: returns mission analytics and alerts."""
    if env is None:
        return {"error": "No active mission"}
    state = env.get_full_state()
    health = state["forest_health"]
    alerts = []
    if health < 0.5:
        alerts.append({"level": "CRITICAL", "msg": "Forest health below 50%"})
    elif health < 0.7:
        alerts.append({"level": "WARNING", "msg": "Forest health below 70%"})
    deforesting = sum(
        1 for r in state["grid"] for v in r if v == 2
    )
    destroyed = sum(
        1 for r in state["grid"] for v in r if v == 3
    )
    if deforesting > 10:
        alerts.append({
            "level": "WARNING",
            "msg": f"{deforesting} tiles actively deforesting"
        })
    return {
        "forest_health": health,
        "fuel_remaining": state["fuel"],
        "step": state["step"],
        "tiles_healthy": 225 - deforesting - destroyed,
        "tiles_deforesting": deforesting,
        "tiles_destroyed": destroyed,
        "threats_detected": state["detected_threats"],
        "drone_position": state["drone_pos"],
        "wind": state["wind_dir"],
        "rain": state["rain_active"],
        "alerts": alerts,
    }


@app.get("/heatmap")
def get_deforestation_heatmap():
    """Production endpoint: returns grid as risk heatmap for GIS integration."""
    if env is None:
        return {"error": "No active mission"}
    state = env.get_full_state()
    risk_map = []
    for r, row in enumerate(state["grid"]):
        for c, val in enumerate(row):
            if val > 0:
                risk_map.append({
                    "row": r, "col": c,
                    "state": ["healthy", "at_risk", "deforesting", "destroyed"][val],
                    "risk_level": val / 3.0,
                    "lat": -2.4 + r * 0.001,
                    "lon": 29.2 + c * 0.001,
                })
    return {
        "timestamp": state["step"],
        "total_risk_tiles": len(risk_map),
        "risk_tiles": risk_map,
    }


@app.post("/batch_episodes")
def run_batch(n: int = 10):
    """Production endpoint: run N episodes and return aggregate stats."""
    global env, current_obs, episode_done
    results = []
    for ep in range(min(n, 50)):
        env = DeforestationEnv()
        current_obs, _ = env.reset()
        total_reward = 0
        steps = 0
        while True:
            if model is not None:
                action, _ = model.predict(current_obs, deterministic=True)
                action = int(action)
            else:
                action = env.action_space.sample()
            current_obs, reward, t, tr, info = env.step(action)
            total_reward += reward
            steps += 1
            if t or tr:
                break
        results.append({
            "episode": ep + 1,
            "total_reward": round(float(total_reward), 1),
            "final_health": round(float(info["forest_health"]), 3),
            "steps": steps,
            "win": info["forest_health"] >= 0.7,
        })
    wins = sum(1 for r in results if r["win"])
    return {
        "n_episodes": len(results),
        "win_rate": round(wins / len(results), 2),
        "avg_health": round(
            sum(r["final_health"] for r in results) / len(results), 3
        ),
        "avg_reward": round(
            sum(r["total_reward"] for r in results) / len(results), 1
        ),
        "episodes": results,
    }


@app.get("/model_info")
def model_info():
    """Production endpoint: returns loaded model metadata."""
    if model is None:
        return {"model": "none", "status": "no model loaded"}
    return {
        "model_type": type(model).__name__,
        "observation_space": 38,
        "action_space": 8,
        "actions": [
            "Move Up", "Move Down", "Move Left", "Move Right",
            "Scan", "Deploy Retardant", "Call Rangers", "Return to Base"
        ],
        "environment": {
            "grid_size": 15,
            "max_steps": 300,
            "max_fuel": 300,
        },
        "status": "ready",
    }


if __name__ == "__main__":
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=True)
