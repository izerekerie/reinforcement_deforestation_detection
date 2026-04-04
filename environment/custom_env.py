"""
UAV Deforestation Detection Environment
========================================
A custom Gymnasium environment simulating autonomous UAV surveillance
over a 15x15 forest grid. The drone agent must detect deforestation,
deploy countermeasures, and maintain forest health above 70%.

Environment dynamics include wind-driven deforestation spread, rain
events, and an illegal logging truck that the agent must intercept.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np

# ── Tile States ──────────────────────────────────────────────────────
HEALTHY = 0       # Forest is intact
AT_RISK = 1       # Early warning signs of deforestation
DEFORESTING = 2   # Actively being destroyed, spreads to neighbors
DESTROYED = 3     # Permanently lost

# ── Environment Parameters ───────────────────────────────────────────
GRID_SIZE = 15            # 15x15 forest grid
MAX_STEPS = 300           # Episode length (time budget)
MAX_FUEL = 300            # Fuel budget (enough for full patrol)
VIEW_RADIUS = 2           # Drone sees a 5x5 window around itself
REFUEL_AMOUNT = 80        # Fuel restored on return to base
MAX_RANGER_CALLS = 3      # Limited ranger deployments per episode
RETARDANT_DURATION = 15   # Steps retardant protects a tile

# ── Action Space (8 discrete actions) ────────────────────────────────
ACTION_UP = 0             # Move north       (1 fuel)
ACTION_DOWN = 1           # Move south       (1 fuel)
ACTION_LEFT = 2           # Move west        (1 fuel)
ACTION_RIGHT = 3          # Move east        (1 fuel)
ACTION_SCAN = 4           # Deep scan 5x5    (2 fuel)
ACTION_RETARDANT = 5      # Deploy retardant (3 fuel)
ACTION_CALL_RANGERS = 6   # Stop the truck   (0 fuel, 3 uses max)
ACTION_RETURN_BASE = 7    # Refuel at (0,0)  (0 fuel, once per ep)

# ── Wind Directions (row_delta, col_delta) ───────────────────────────
WIND_DIRECTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # N, S, W, E


class DeforestationEnv(gym.Env):
    """
    UAV Forest Surveillance Environment.

    Observation Space (38-dim vector):
        - 25 values: 5x5 local grid view (tile states normalized 0-1)
        - 13 values: drone position, fuel, health, threats detected,
          wind direction, truck position, nearest threat direction,
          ranger calls remaining, exploration coverage

    Action Space: Discrete(8) — see ACTION_* constants above

    Reward Structure:
        - Each step:           -0.5 (efficiency pressure)
        - Scan finds At-Risk:  +8   | Scan finds Deforesting: +15
        - Auto-detect At-Risk: +2   | Auto-detect Deforesting: +3
        - Retardant on Deforesting: +50 | on At-Risk: +30
        - Call rangers (truck): +60
        - Move toward threat:  +1.5 | Move away: -1.0
        - New tile explored:   +0.5
        - Win (health >= 70%): +100 | Lose (health < 30%): -100
        - Truck escapes grid:  -50

    Episode Termination:
        - Fuel reaches 0
        - Forest health drops below 30%
        - 300 steps completed (truncation)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.grid_size = GRID_SIZE
        self.action_space = spaces.Discrete(8)

        obs_size = (VIEW_RADIUS * 2 + 1) ** 2 + 13  # 25 + 13 = 38
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self._place_initial_deforestation()

        self.drone_pos = np.array([0, 0])
        self.fuel = MAX_FUEL
        self.step_count = 0
        self.ranger_calls_left = MAX_RANGER_CALLS
        self.detected_threats = set()
        self.retardant_tiles = {}
        self.has_refueled = False

        self.wind_dir = WIND_DIRECTIONS[self.np_random.integers(4)]
        self.wind_timer = 0

        self.truck_pos = np.array([0, self.np_random.integers(self.grid_size)])
        self.truck_active = True
        self.truck_timer = 0
        self.truck_stopped = False

        self.rain_active = False
        self.rain_timer = 0
        self.visited = set()
        self.visited.add((0, 0))

        # Terrain: elevation affects spread speed + river blocks movement
        self._generate_terrain()

        return self._get_obs(), self._get_info()

    def _generate_terrain(self):
        """Generate terrain features — elevation and river."""
        # Elevation map: higher ground spreads slower
        self.elevation = np.zeros(
            (self.grid_size, self.grid_size), dtype=np.float32
        )
        # Create hills using simple gradient
        cx = self.np_random.integers(4, 11)
        cy = self.np_random.integers(4, 11)
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                dist = ((r - cx) ** 2 + (c - cy) ** 2) ** 0.5
                self.elevation[r, c] = max(0, 1.0 - dist / 8.0)

        # River: a winding path that blocks truck movement
        self.river = set()
        rc = self.np_random.integers(3, 12)
        for r in range(self.grid_size):
            self.river.add((r, rc))
            if self.np_random.random() < 0.3:
                rc = max(0, min(self.grid_size - 1,
                                rc + self.np_random.choice([-1, 1])))
            # River tiles stay healthy (water)
            self.grid[r, rc] = HEALTHY

    def _place_initial_deforestation(self):
        n_at_risk = self.np_random.integers(5, 9)
        n_deforesting = self.np_random.integers(2, 4)

        for _ in range(n_at_risk):
            r, c = self.np_random.integers(0, self.grid_size, size=2)
            self.grid[r, c] = AT_RISK

        for _ in range(n_deforesting):
            r, c = self.np_random.integers(0, self.grid_size, size=2)
            self.grid[r, c] = DEFORESTING

    def _get_obs(self):
        view_size = VIEW_RADIUS * 2 + 1
        view = np.zeros(view_size * view_size, dtype=np.float32)

        idx = 0
        for dr in range(-VIEW_RADIUS, VIEW_RADIUS + 1):
            for dc in range(-VIEW_RADIUS, VIEW_RADIUS + 1):
                r = self.drone_pos[0] + dr
                c = self.drone_pos[1] + dc
                if 0 <= r < self.grid_size and 0 <= c < self.grid_size:
                    view[idx] = self.grid[r, c] / 3.0
                else:
                    view[idx] = 0.0
                idx += 1

        threat_dir_r, threat_dir_c, threat_dist = self._nearest_threat_direction()

        extras = np.array([
            self.drone_pos[0] / self.grid_size,
            self.drone_pos[1] / self.grid_size,
            self.fuel / MAX_FUEL,
            self._forest_health(),
            len(self.detected_threats) / (self.grid_size ** 2),
            (WIND_DIRECTIONS.index(self.wind_dir)) / 3.0,
            self.truck_pos[0] / self.grid_size if self.truck_active else 0.0,
            self.truck_pos[1] / self.grid_size if self.truck_active else 0.0,
            threat_dir_r,
            threat_dir_c,
            threat_dist,
            self.ranger_calls_left / MAX_RANGER_CALLS,
            len(self.visited) / (self.grid_size ** 2),
        ], dtype=np.float32)

        return np.concatenate([view, extras])

    def _nearest_threat_direction(self):
        dr, dc = self.drone_pos
        best_dist = float("inf")
        best_r, best_c = 0.0, 0.0
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if self.grid[r, c] in (AT_RISK, DEFORESTING):
                    dist = abs(r - dr) + abs(c - dc)
                    if dist < best_dist:
                        best_dist = dist
                        if dist > 0:
                            best_r = (r - dr) / dist
                            best_c = (c - dc) / dist
        if best_dist == float("inf"):
            return 0.0, 0.0, 0.0
        norm_dist = 1.0 - min(best_dist / self.grid_size, 1.0)
        return best_r, best_c, norm_dist

    def _get_info(self):
        return {
            "forest_health": self._forest_health(),
            "fuel": self.fuel,
            "step": self.step_count,
            "detected_threats": len(self.detected_threats),
            "ranger_calls_left": self.ranger_calls_left,
            "drone_pos": self.drone_pos.tolist(),
            "truck_pos": self.truck_pos.tolist() if self.truck_active else None,
            "grid": self.grid.copy(),
        }

    def _forest_health(self):
        total = self.grid_size ** 2
        destroyed = np.sum(self.grid == DESTROYED)
        deforesting = np.sum(self.grid == DEFORESTING)
        return (total - destroyed - 0.5 * deforesting) / total

    def step(self, action):
        reward = -0.5
        self.step_count += 1

        old_pos = self.drone_pos.copy()
        reward += self._execute_action(action)
        reward += self._movement_bonus(old_pos)
        self._update_environment()

        terminated = False
        truncated = False

        health = self._forest_health()

        if self.fuel <= 0:
            terminated = True
        if health < 0.30:
            reward -= 100.0
            terminated = True
        if self.truck_active and self.truck_pos[0] >= self.grid_size:
            reward -= 50.0
            self.truck_active = False

        if self.step_count >= MAX_STEPS:
            truncated = True
            if health >= 0.70:
                reward += 100.0
            else:
                reward -= 100.0

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _execute_action(self, action):
        reward = 0.0
        r, c = self.drone_pos

        if action == ACTION_UP and r > 0:
            self.drone_pos[0] -= 1
            self.fuel -= 1
            reward += self._auto_scan()
        elif action == ACTION_DOWN and r < self.grid_size - 1:
            self.drone_pos[0] += 1
            self.fuel -= 1
            reward += self._auto_scan()
        elif action == ACTION_LEFT and c > 0:
            self.drone_pos[1] -= 1
            self.fuel -= 1
            reward += self._auto_scan()
        elif action == ACTION_RIGHT and c < self.grid_size - 1:
            self.drone_pos[1] += 1
            self.fuel -= 1
            reward += self._auto_scan()
        elif action == ACTION_SCAN:
            self.fuel -= 2
            reward += self._scan()
        elif action == ACTION_RETARDANT:
            ret_reward = self._deploy_retardant()
            if ret_reward > 0:
                self.fuel -= 3
            reward += ret_reward
        elif action == ACTION_CALL_RANGERS:
            reward += self._call_rangers()
        elif action == ACTION_RETURN_BASE:
            reward += self._return_to_base()

        self.fuel = max(0, self.fuel)
        return reward

    def _auto_scan(self):
        r, c = self.drone_pos
        reward = 0.0
        tile = self.grid[r, c]
        pos = (r, c)
        if tile == AT_RISK and pos not in self.detected_threats:
            self.detected_threats.add(pos)
            reward += 2.0
        elif tile == DEFORESTING and pos not in self.detected_threats:
            self.detected_threats.add(pos)
            reward += 3.0
        return reward

    def _scan(self):
        reward = 0.0
        r, c = self.drone_pos
        found_new = 0
        for dr in range(-VIEW_RADIUS, VIEW_RADIUS + 1):
            for dc in range(-VIEW_RADIUS, VIEW_RADIUS + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                    tile = self.grid[nr, nc]
                    pos = (nr, nc)
                    if pos not in self.detected_threats:
                        if tile == AT_RISK:
                            self.detected_threats.add(pos)
                            reward += 8.0
                            found_new += 1
                        elif tile == DEFORESTING:
                            self.detected_threats.add(pos)
                            reward += 15.0
                            found_new += 1
        if found_new == 0:
            reward -= 2.0
        return reward

    def _deploy_retardant(self):
        r, c = self.drone_pos
        if (r, c) in self.retardant_tiles:
            return -3.0
        tile = self.grid[r, c]
        if tile == DEFORESTING:
            # Retardant stops active destruction and downgrades
            self.retardant_tiles[(r, c)] = RETARDANT_DURATION
            self.grid[r, c] = AT_RISK
            return 50.0
        elif tile == AT_RISK:
            # Retardant protects at-risk tile from getting worse
            self.retardant_tiles[(r, c)] = RETARDANT_DURATION
            return 30.0
        return -3.0

    def _call_rangers(self):
        if self.ranger_calls_left <= 0:
            return -2.0
        if not self.truck_active or self.truck_stopped:
            return -2.0
        self.ranger_calls_left -= 1
        self.truck_stopped = True
        self.truck_active = False
        return 60.0  # Huge reward: stopping the truck

    def _return_to_base(self):
        self.drone_pos = np.array([0, 0])
        if not self.has_refueled:
            self.fuel = min(self.fuel + REFUEL_AMOUNT, MAX_FUEL)
            self.has_refueled = True
        return 0.0

    def _update_environment(self):
        self._update_retardant()
        self._update_wind()
        self._update_rain()

        if self.step_count % 5 == 0:
            self._progress_deforestation()

        if self.step_count % 5 == 0:
            self._spread_deforestation()

        if self.step_count % 5 == 0:
            self._move_truck()

        # New threats appear periodically (realistic: logging never stops)
        if self.step_count % 50 == 0:
            self._spawn_new_threat()

        self._apply_nearby_penalty()

    def _spawn_new_threat(self):
        """New at-risk tile appears — simulates ongoing logging pressure."""
        for _ in range(10):
            r = self.np_random.integers(0, self.grid_size)
            c = self.np_random.integers(0, self.grid_size)
            if self.grid[r, c] == HEALTHY:
                self.grid[r, c] = AT_RISK
                return

    def _update_retardant(self):
        expired = []
        for pos, timer in self.retardant_tiles.items():
            r, c = pos
            if timer <= 1:
                expired.append(pos)
                # Protected tile recovers one stage when done
                if self.grid[r, c] == AT_RISK:
                    self.grid[r, c] = HEALTHY
            else:
                self.retardant_tiles[pos] = timer - 1
                # Slow healing: midway through, at-risk -> healthy
                if timer == RETARDANT_DURATION // 2:
                    if self.grid[r, c] == AT_RISK:
                        self.grid[r, c] = HEALTHY
        for pos in expired:
            del self.retardant_tiles[pos]

    def _update_wind(self):
        self.wind_timer += 1
        if self.wind_timer >= 8:
            self.wind_dir = WIND_DIRECTIONS[self.np_random.integers(4)]
            self.wind_timer = 0

    def _update_rain(self):
        self.rain_timer += 1
        if self.rain_timer >= 10:
            self.rain_active = self.np_random.random() < 0.35
            self.rain_timer = 0

    def _progress_deforestation(self):
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if (r, c) in self.retardant_tiles:
                    continue
                if self.grid[r, c] == AT_RISK:
                    # 50% chance to progress (not guaranteed, gives drone time)
                    if self.np_random.random() < 0.5:
                        self.grid[r, c] = DEFORESTING

    def _spread_deforestation(self):
        if self.rain_active:
            # Rain heals: at-risk tiles have a chance to recover
            for r in range(self.grid_size):
                for c in range(self.grid_size):
                    if self.grid[r, c] == AT_RISK:
                        if self.np_random.random() < 0.08:
                            self.grid[r, c] = HEALTHY
            return

        new_grid = self.grid.copy()
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if self.grid[r, c] != DEFORESTING:
                    continue
                if (r, c) in self.retardant_tiles:
                    continue

                neighbors = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
                for nr, nc in neighbors:
                    if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                        if new_grid[nr, nc] == HEALTHY:
                            # River blocks spread
                            if (nr, nc) in self.river:
                                continue
                            spread_prob = 0.06
                            if (nr - r, nc - c) == self.wind_dir:
                                spread_prob = 0.14
                            # Higher elevation = slower spread
                            elev = self.elevation[nr, nc]
                            spread_prob *= (1.0 - 0.5 * elev)
                            if self.np_random.random() < spread_prob:
                                new_grid[nr, nc] = AT_RISK

                if self.np_random.random() < 0.02:
                    new_grid[r, c] = DESTROYED

        self.grid = new_grid

    def _move_truck(self):
        if not self.truck_active or self.truck_stopped:
            return
        self.truck_pos[0] += 1
        tr, tc = self.truck_pos
        # Truck blocked by river — must go around
        if (tr, tc) in self.river:
            self.truck_pos[1] = min(
                self.grid_size - 1, self.truck_pos[1] + 1
            )
            tc = self.truck_pos[1]
        if 0 <= tr < self.grid_size and 0 <= tc < self.grid_size:
            self.grid[tr, tc] = DESTROYED

    def _movement_bonus(self, old_pos):
        new_pos = self.drone_pos
        if np.array_equal(old_pos, new_pos):
            return 0.0

        bonus = 0.0
        pos_tuple = (int(new_pos[0]), int(new_pos[1]))
        if pos_tuple not in self.visited:
            bonus += 0.5

        best_dist_old = float("inf")
        best_dist_new = float("inf")
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if self.grid[r, c] in (AT_RISK, DEFORESTING):
                    d_old = abs(r - old_pos[0]) + abs(c - old_pos[1])
                    d_new = abs(r - new_pos[0]) + abs(c - new_pos[1])
                    best_dist_old = min(best_dist_old, d_old)
                    best_dist_new = min(best_dist_new, d_new)
        if best_dist_old != float("inf"):
            if best_dist_new < best_dist_old:
                bonus += 1.5
            elif best_dist_new > best_dist_old:
                bonus -= 1.0
        return bonus

    def _apply_nearby_penalty(self):
        r, c = self.drone_pos
        pos = (r, c)
        if pos not in self.visited:
            self.visited.add(pos)
        # Penalty: if drone is standing on a threat but not acting
        tile = self.grid[r, c]
        if tile == DEFORESTING and (r, c) not in self.retardant_tiles:
            self.nearby_idle_count = getattr(
                self, 'nearby_idle_count', 0
            ) + 1

    def get_full_state(self):
        return {
            "grid": self.grid.tolist(),
            "drone_pos": self.drone_pos.tolist(),
            "fuel": int(self.fuel),
            "step": self.step_count,
            "forest_health": float(self._forest_health()),
            "detected_threats": len(self.detected_threats),
            "ranger_calls_left": self.ranger_calls_left,
            "truck_pos": self.truck_pos.tolist()
            if self.truck_active else None,
            "truck_active": self.truck_active,
            "wind_dir": list(self.wind_dir),
            "rain_active": self.rain_active,
            "retardant_tiles": {
                f"{k[0]},{k[1]}": v
                for k, v in self.retardant_tiles.items()
            },
            "elevation": self.elevation.tolist(),
            "river": [[int(r), int(c)] for r, c in self.river],
        }
