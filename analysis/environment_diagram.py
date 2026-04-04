"""
Generate environment architecture diagrams for the UAV Deforestation Detection report.
Produces two diagrams:
  1. Environment overview — grid, agent, dynamics, reward structure
  2. RL training pipeline — how algorithms connect to the environment
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def draw_environment_overview():
    """Create a comprehensive environment architecture diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(18, 13))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 13)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    # ── Title ─────────────────────────────────────────────────────────
    ax.text(9, 12.5, "UAV Deforestation Detection — Environment Architecture",
            ha="center", va="center", fontsize=18, fontweight="bold",
            color="white", family="monospace")

    # ── 1. Grid Section (left) ────────────────────────────────────────
    grid_x, grid_y = 0.5, 5.5
    grid_box = FancyBboxPatch((grid_x, grid_y), 5.5, 6.2,
                               boxstyle="round,pad=0.2", facecolor="#1a2332",
                               edgecolor="#30a14e", linewidth=2)
    ax.add_patch(grid_box)
    ax.text(grid_x + 2.75, grid_y + 5.8, "15 x 15 Forest Grid",
            ha="center", va="center", fontsize=13, fontweight="bold",
            color="#58a6ff", family="monospace")

    # Mini grid preview
    tile_colors = {0: "#2d822d", 1: "#c8b41e", 2: "#dc6414", 3: "#502d0f"}
    tile_labels = {0: "Healthy", 1: "At-Risk", 2: "Deforesting", 3: "Destroyed"}
    grid_data = np.zeros((6, 6), dtype=int)
    grid_data[1, 2] = 1; grid_data[2, 3] = 2; grid_data[3, 4] = 2
    grid_data[4, 4] = 3; grid_data[0, 5] = 1; grid_data[5, 1] = 3

    gx0, gy0 = grid_x + 0.4, grid_y + 1.8
    ts = 0.65
    for r in range(6):
        for c in range(6):
            color = tile_colors[grid_data[r, c]]
            rect = plt.Rectangle((gx0 + c * ts, gy0 + (5 - r) * ts),
                                  ts - 0.04, ts - 0.04, facecolor=color,
                                  edgecolor="#333", linewidth=0.5)
            ax.add_patch(rect)

    # Drone on grid
    drone_x, drone_y = gx0 + 2 * ts + ts / 2, gy0 + 3 * ts + ts / 2
    ax.plot(drone_x, drone_y, "o", color="#00d4ff", markersize=14, zorder=5)
    ax.text(drone_x, drone_y, "D", ha="center", va="center", fontsize=8,
            fontweight="bold", color="white", zorder=6)

    # Truck on grid
    truck_x, truck_y = gx0 + 4 * ts + ts / 2, gy0 + 1 * ts + ts / 2
    ax.plot(truck_x, truck_y, "s", color="#e61e3c", markersize=12, zorder=5)
    ax.text(truck_x, truck_y, "T", ha="center", va="center", fontsize=7,
            fontweight="bold", color="white", zorder=6)

    # View radius indicator
    view_rect = plt.Rectangle((gx0 + 0.5 * ts, gy0 + 1.5 * ts),
                               5 * ts - 0.04, 5 * ts - 0.04,
                               facecolor="none", edgecolor="#00d4ff",
                               linewidth=1.5, linestyle="--", alpha=0.5)
    ax.add_patch(view_rect)
    ax.text(gx0 + 3 * ts, gy0 + 1.3 * ts, "5x5 view radius",
            ha="center", fontsize=7, color="#00d4ff", alpha=0.7)

    # Legend
    legend_y = grid_y + 0.3
    for i, (state, label) in enumerate(tile_labels.items()):
        lx = grid_x + 0.5 + i * 1.3
        rect = plt.Rectangle((lx, legend_y), 0.3, 0.3,
                               facecolor=tile_colors[state], edgecolor="#555")
        ax.add_patch(rect)
        ax.text(lx + 0.4, legend_y + 0.15, label, fontsize=7,
                va="center", color="#aaa", family="monospace")

    # ── 2. Observation Space (top center) ─────────────────────────────
    obs_box = FancyBboxPatch((6.5, 9.2), 5, 2.5,
                              boxstyle="round,pad=0.2", facecolor="#1a2332",
                              edgecolor="#58a6ff", linewidth=2)
    ax.add_patch(obs_box)
    ax.text(9, 11.3, "Observation Space (38-dim)",
            ha="center", fontsize=12, fontweight="bold",
            color="#58a6ff", family="monospace")

    obs_items = [
        "5x5 local grid view .............. 25 values",
        "Drone position (x, y) ............  2 values",
        "Fuel level .......................  1 value",
        "Forest health % ..................  1 value",
        "Threats detected / Wind dir ......  3 values",
        "Truck position / Threat dir ......  4 values",
        "Rangers left / Coverage ..........  2 values",
    ]
    for i, item in enumerate(obs_items):
        ax.text(6.8, 10.9 - i * 0.25, item, fontsize=7,
                color="#c9d1d9", family="monospace")

    # ── 3. Action Space (center) ──────────────────────────────────────
    act_box = FancyBboxPatch((6.5, 5.5), 5, 3.2,
                              boxstyle="round,pad=0.2", facecolor="#1a2332",
                              edgecolor="#f0883e", linewidth=2)
    ax.add_patch(act_box)
    ax.text(9, 8.3, "Action Space (8 Discrete)",
            ha="center", fontsize=12, fontweight="bold",
            color="#f0883e", family="monospace")

    actions = [
        ("Move Up/Down/Left/Right", "1 fuel", "#2d822d"),
        ("Scan (5x5 area)", "2 fuel", "#58a6ff"),
        ("Deploy Retardant", "5 fuel", "#1e90ff"),
        ("Call Rangers (3x max)", "0 fuel", "#9b59b6"),
        ("Return to Base (1x)", "+60 fuel", "#e6c619"),
    ]
    for i, (name, cost, color) in enumerate(actions):
        y_pos = 7.9 - i * 0.42
        ax.plot(6.9, y_pos, "s", color=color, markersize=8)
        ax.text(7.2, y_pos, name, fontsize=9, va="center",
                color="#c9d1d9", family="monospace")
        ax.text(10.8, y_pos, cost, fontsize=8, va="center",
                color=color, family="monospace", ha="right")

    # ── 4. Reward Structure (right) ───────────────────────────────────
    rew_box = FancyBboxPatch((12, 5.5), 5.5, 6.2,
                              boxstyle="round,pad=0.2", facecolor="#1a2332",
                              edgecolor="#da3633", linewidth=2)
    ax.add_patch(rew_box)
    ax.text(14.75, 11.3, "Reward Structure",
            ha="center", fontsize=13, fontweight="bold",
            color="#da3633", family="monospace")

    rewards_pos = [
        ("Scan At-Risk", "+15", "#30a14e"),
        ("Scan Deforesting", "+30", "#30a14e"),
        ("Deploy Retardant", "+25", "#30a14e"),
        ("Stop Truck (Rangers)", "+30", "#30a14e"),
        ("Move Toward Threat", "+1.5", "#30a14e"),
        ("Win (health >= 70%)", "+100", "#30a14e"),
    ]
    rewards_neg = [
        ("Each Step (efficiency)", "-0.5", "#da3633"),
        ("Re-scan Known Threat", "-3", "#da3633"),
        ("Wasted Retardant", "-10", "#da3633"),
        ("Truck Escapes", "-50", "#da3633"),
        ("Lose (health < 30%)", "-100", "#da3633"),
    ]

    ax.text(12.4, 10.8, "Positive", fontsize=9, fontweight="bold",
            color="#30a14e", family="monospace")
    for i, (name, val, color) in enumerate(rewards_pos):
        y = 10.4 - i * 0.35
        ax.text(12.4, y, name, fontsize=7.5, color="#aaa", family="monospace")
        ax.text(17, y, val, fontsize=8, color=color, ha="right",
                family="monospace", fontweight="bold")

    ax.text(12.4, 8.15, "Negative", fontsize=9, fontweight="bold",
            color="#da3633", family="monospace")
    for i, (name, val, color) in enumerate(rewards_neg):
        y = 7.8 - i * 0.35
        ax.text(12.4, y, name, fontsize=7.5, color="#aaa", family="monospace")
        ax.text(17, y, val, fontsize=8, color=color, ha="right",
                family="monospace", fontweight="bold")

    # ── 5. World Dynamics (bottom) ────────────────────────────────────
    dyn_box = FancyBboxPatch((0.5, 0.3), 17, 4.8,
                              boxstyle="round,pad=0.2", facecolor="#1a2332",
                              edgecolor="#8b949e", linewidth=2)
    ax.add_patch(dyn_box)
    ax.text(9, 4.7, "Environment Dynamics",
            ha="center", fontsize=13, fontweight="bold",
            color="#8b949e", family="monospace")

    dynamics = [
        ("Wind System", "Changes direction every 8 steps.\n"
         "Deforestation spreads 2.5x faster\ndownwind (15% vs 6% base).",
         "#58a6ff", 2.5),
        ("Rain Events", "30% chance every 10 steps.\n"
         "Completely blocks deforestation\nspread while active.",
         "#1e90ff", 6.5),
        ("Logging Truck", "Moves 1 tile every 4 steps.\n"
         "Destroys every tile it crosses.\n"
         "Agent must call rangers to stop.",
         "#e61e3c", 10.5),
        ("Deforestation Spread", "At-Risk -> Deforesting (every 6 steps)\n"
         "Deforesting spreads to neighbors\n"
         "(every 3 steps, blocked by rain).",
         "#dc6414", 14.5),
    ]

    for name, desc, color, cx in dynamics:
        box = FancyBboxPatch((cx - 1.5, 0.7), 3.4, 3.5,
                              boxstyle="round,pad=0.15", facecolor="#161b22",
                              edgecolor=color, linewidth=1.5)
        ax.add_patch(box)
        ax.text(cx + 0.2, 3.8, name, ha="center", fontsize=10,
                fontweight="bold", color=color, family="monospace")
        ax.text(cx + 0.2, 3.2, desc, ha="center", va="top", fontsize=7,
                color="#8b949e", family="monospace", linespacing=1.5)

    # ── Arrows connecting sections ────────────────────────────────────
    # Grid -> Observation
    ax.annotate("", xy=(6.5, 10.2), xytext=(6, 9.5),
                arrowprops=dict(arrowstyle="->", color="#58a6ff",
                                lw=2, connectionstyle="arc3,rad=0.2"))
    # Observation -> Action (agent decides)
    ax.annotate("", xy=(9, 8.7), xytext=(9, 9.2),
                arrowprops=dict(arrowstyle="->", color="#f0883e",
                                lw=2))
    ax.text(9.5, 8.9, "Agent", fontsize=8, color="#f0883e",
            family="monospace", fontstyle="italic")
    # Action -> Grid (action applied)
    ax.annotate("", xy=(6, 8.5), xytext=(6.5, 7),
                arrowprops=dict(arrowstyle="->", color="#30a14e",
                                lw=2, connectionstyle="arc3,rad=-0.3"))
    # Action -> Reward
    ax.annotate("", xy=(12, 8), xytext=(11.5, 7.5),
                arrowprops=dict(arrowstyle="->", color="#da3633",
                                lw=2, connectionstyle="arc3,rad=0.2"))

    # ── Terminal Conditions Box ───────────────────────────────────────
    term_y = 11.8
    ax.text(15, term_y, "Terminal: fuel=0 | health<30% | 200 steps",
            ha="center", fontsize=8, color="#f85149", family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a2e",
                      edgecolor="#f85149", alpha=0.8))

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "environment_diagram.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"Saved: {path}")
    return path


def draw_training_pipeline():
    """Create a diagram showing the RL training pipeline."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    ax.text(8, 7.5, "RL Training & Deployment Pipeline",
            ha="center", fontsize=16, fontweight="bold",
            color="white", family="monospace")

    # ── Environment box ───────────────────────────────────────────────
    env_box = FancyBboxPatch((0.3, 3), 3, 3.5,
                              boxstyle="round,pad=0.2", facecolor="#1a2332",
                              edgecolor="#30a14e", linewidth=2)
    ax.add_patch(env_box)
    ax.text(1.8, 6.1, "Environment", ha="center", fontsize=12,
            fontweight="bold", color="#30a14e", family="monospace")
    env_details = [
        "DeforestationEnv",
        "Gymnasium API",
        "15x15 Grid",
        "8 Actions",
        "38-dim Observation",
        "Complex Dynamics",
    ]
    for i, d in enumerate(env_details):
        ax.text(1.8, 5.6 - i * 0.38, d, ha="center", fontsize=8,
                color="#8b949e", family="monospace")

    # ── Four algorithm boxes ──────────────────────────────────────────
    algos = [
        ("DQN", "Value-Based\nReplay Buffer\nTarget Network",
         "#58a6ff", "Stable Baselines3"),
        ("PPO", "Policy Gradient\nClipped Objective\nGAE Advantage",
         "#f0883e", "Stable Baselines3"),
        ("A2C", "Actor-Critic\nSynchronous\nValue Baseline",
         "#30a14e", "Stable Baselines3"),
        ("REINFORCE", "Policy Gradient\nMonte Carlo\nNo Critic",
         "#da3633", "Custom PyTorch"),
    ]
    algo_x_positions = [4.2, 6.7, 9.2, 11.7]

    for (name, desc, color, lib), ax_x in zip(algos, algo_x_positions):
        box = FancyBboxPatch((ax_x, 3.5), 2.2, 3,
                              boxstyle="round,pad=0.15", facecolor="#161b22",
                              edgecolor=color, linewidth=2)
        ax.add_patch(box)
        ax.text(ax_x + 1.1, 6.1, name, ha="center", fontsize=12,
                fontweight="bold", color=color, family="monospace")
        ax.text(ax_x + 1.1, 5.2, desc, ha="center", fontsize=7.5,
                color="#8b949e", family="monospace", linespacing=1.4)
        ax.text(ax_x + 1.1, 3.8, lib, ha="center", fontsize=7,
                color="#6e7681", family="monospace", fontstyle="italic")

        # Arrow from env to algo
        ax.annotate("", xy=(ax_x, 4.8), xytext=(3.3, 4.8),
                    arrowprops=dict(arrowstyle="->", color="#30a14e",
                                    lw=1.5, alpha=0.5))

    # ── Hyperparameter tuning label ───────────────────────────────────
    ax.text(8.5, 3.1, "10 hyperparameter configs each  |  50,000 timesteps  |  20 eval episodes",
            ha="center", fontsize=8, color="#f0883e", family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a2e",
                      edgecolor="#f0883e", alpha=0.8))

    # ── Output boxes (bottom) ─────────────────────────────────────────
    outputs = [
        ("Trained Models", "40 models\n(.zip / .pt)", "#58a6ff", 2),
        ("Comparison\nAnalysis", "Reward curves\nWin rates", "#f0883e", 5.5),
        ("Pygame 2D\nVisualization", "Interactive\n1200x800", "#30a14e", 9),
        ("FastAPI +\nThree.js Dashboard", "Live 3D\nWeb browser", "#da3633", 12.5),
    ]

    for name, desc, color, ox in outputs:
        box = FancyBboxPatch((ox, 0.3), 2.5, 2.2,
                              boxstyle="round,pad=0.15", facecolor="#161b22",
                              edgecolor=color, linewidth=1.5)
        ax.add_patch(box)
        ax.text(ox + 1.25, 2.1, name, ha="center", fontsize=9,
                fontweight="bold", color=color, family="monospace")
        ax.text(ox + 1.25, 1.2, desc, ha="center", fontsize=7.5,
                color="#8b949e", family="monospace", linespacing=1.3)

    # Arrows from algos to outputs
    for ox in [2, 5.5, 9, 12.5]:
        ax.annotate("", xy=(ox + 1.25, 2.5), xytext=(8.5, 3.1),
                    arrowprops=dict(arrowstyle="->", color="#8b949e",
                                    lw=1, alpha=0.3,
                                    connectionstyle="arc3,rad=0.1"))

    # ── Technology labels ─────────────────────────────────────────────
    ax.text(8, 0.1, "Python  |  Gymnasium  |  Stable Baselines3  |  PyTorch  |  "
            "Pygame  |  FastAPI  |  Three.js  |  Matplotlib",
            ha="center", fontsize=7, color="#6e7681", family="monospace")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "training_pipeline.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"Saved: {path}")
    return path


def draw_grid_snapshot():
    """Create a clean grid state diagram showing tile states and agent."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.set_xlim(-0.5, 15.5)
    ax.set_ylim(-0.5, 15.5)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    ax.text(7.5, 15.8, "Forest Grid State — Example Snapshot",
            ha="center", fontsize=14, fontweight="bold",
            color="white", family="monospace")

    np.random.seed(42)
    grid = np.zeros((15, 15), dtype=int)
    # Create a realistic-looking scenario
    at_risk_positions = [(3, 5), (3, 6), (4, 5), (7, 10), (8, 11), (10, 2), (11, 3)]
    deforesting_positions = [(4, 6), (4, 7), (8, 10), (8, 12), (11, 2)]
    destroyed_positions = [(5, 7), (5, 8), (9, 11), (12, 2)]
    retardant_positions = [(4, 6), (8, 10)]

    for r, c in at_risk_positions:
        grid[r, c] = 1
    for r, c in deforesting_positions:
        grid[r, c] = 2
    for r, c in destroyed_positions:
        grid[r, c] = 3

    tile_colors = {0: "#2d822d", 1: "#c8b41e", 2: "#dc6414", 3: "#502d0f"}
    retardant_color = "#1e8cda"

    for r in range(15):
        for c in range(15):
            if (r, c) in retardant_positions:
                color = retardant_color
            else:
                color = tile_colors[grid[r, c]]
            rect = plt.Rectangle((c, 14 - r), 0.95, 0.95,
                                  facecolor=color, edgecolor="#333",
                                  linewidth=0.5)
            ax.add_patch(rect)

    # Drone at position (6, 8)
    drone_r, drone_c = 6, 8
    ax.plot(drone_c + 0.475, 14 - drone_r + 0.475, "o",
            color="#00d4ff", markersize=20, zorder=5)
    ax.text(drone_c + 0.475, 14 - drone_r + 0.475, "UAV",
            ha="center", va="center", fontsize=6, fontweight="bold",
            color="white", zorder=6)

    # 5x5 view radius
    vr = 2
    view_rect = plt.Rectangle((drone_c - vr, 14 - (drone_r + vr)),
                               5, 5, facecolor="none",
                               edgecolor="#00d4ff", linewidth=2,
                               linestyle="--", alpha=0.6)
    ax.add_patch(view_rect)

    # Truck at position (2, 12)
    truck_r, truck_c = 2, 12
    ax.plot(truck_c + 0.475, 14 - truck_r + 0.475, "s",
            color="#e61e3c", markersize=18, zorder=5)
    ax.text(truck_c + 0.475, 14 - truck_r + 0.475, "T",
            ha="center", va="center", fontsize=8, fontweight="bold",
            color="white", zorder=6)

    # Wind arrow
    ax.annotate("", xy=(14.5, 10), xytext=(14.5, 12),
                arrowprops=dict(arrowstyle="-|>", color="#58a6ff",
                                lw=3, mutation_scale=20))
    ax.text(15.2, 11, "Wind\nSouth", fontsize=8, color="#58a6ff",
            family="monospace", ha="center")

    # Base marker
    ax.plot(0.475, 14.475, "*", color="#e6c619", markersize=18, zorder=5)
    ax.text(0.475, 13.5, "Base\n(0,0)", ha="center", fontsize=7,
            color="#e6c619", family="monospace")

    # Legend
    legend_items = [
        ("Healthy", "#2d822d"), ("At-Risk", "#c8b41e"),
        ("Deforesting", "#dc6414"), ("Destroyed", "#502d0f"),
        ("Retardant", "#1e8cda"), ("UAV Drone", "#00d4ff"),
        ("Logging Truck", "#e61e3c"),
    ]
    for i, (label, color) in enumerate(legend_items):
        lx = 0.5 + i * 2.1
        rect = plt.Rectangle((lx, -1.2), 0.4, 0.4,
                               facecolor=color, edgecolor="#555")
        ax.add_patch(rect)
        ax.text(lx + 0.55, -1.0, label, fontsize=7.5, va="center",
                color="#ccc", family="monospace")

    ax.set_ylim(-1.8, 16.2)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "grid_snapshot.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"Saved: {path}")
    return path


if __name__ == "__main__":
    print("Generating environment diagrams...")
    draw_environment_overview()
    draw_training_pipeline()
    draw_grid_snapshot()
    print("All diagrams generated!")
