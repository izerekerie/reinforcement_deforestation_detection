import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygame
import numpy as np
from environment.forest_env import (
    DeforestationEnv, HEALTHY, AT_RISK, DEFORESTING, DESTROYED,
    GRID_SIZE, MAX_FUEL, MAX_STEPS, ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT,
    ACTION_SCAN, ACTION_RETARDANT, ACTION_CALL_RANGERS, ACTION_RETURN_BASE,
)

WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
GRID_PANEL_SIZE = 700
SIDEBAR_WIDTH = WINDOW_WIDTH - GRID_PANEL_SIZE

TILE_SIZE = GRID_PANEL_SIZE // GRID_SIZE
GRID_OFFSET_X = 20
GRID_OFFSET_Y = (WINDOW_HEIGHT - GRID_SIZE * TILE_SIZE) // 2

BG_COLOR = (20, 20, 35)
SIDEBAR_COLOR = (25, 30, 50)
PANEL_COLOR = (15, 48, 96)
TEXT_COLOR = (220, 220, 230)
MUTED_COLOR = (150, 150, 170)

TILE_COLORS = {
    HEALTHY: (45, 130, 45),
    AT_RISK: (200, 180, 30),
    DEFORESTING: (220, 100, 20),
    DESTROYED: (80, 45, 15),
}

RETARDANT_COLOR = (30, 140, 220)
DRONE_COLOR = (0, 212, 255)
DRONE_SHADOW_COLOR = (0, 100, 140)
TRUCK_COLOR = (230, 30, 60)
GRID_LINE_COLOR = (40, 40, 60)

ACTION_NAMES = ["Up", "Down", "Left", "Right", "Scan", "Retardant", "Rangers", "Base"]

WIND_LABELS = {
    (0, -1): "North",
    (0, 1): "South",
    (-1, 0): "West",
    (1, 0): "East",
}


class ForestRenderer:
    def __init__(self, env):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("UAV Deforestation Detection - Forest Surveillance")
        self.clock = pygame.time.Clock()
        self.env = env
        self.font_large = pygame.font.SysFont("consolas", 28, bold=True)
        self.font_medium = pygame.font.SysFont("consolas", 18)
        self.font_small = pygame.font.SysFont("consolas", 14)
        self.action_log = []
        self.total_reward = 0.0
        self.frame = 0

    def draw_tile(self, r, c, state):
        x = GRID_OFFSET_X + c * TILE_SIZE
        y = GRID_OFFSET_Y + r * TILE_SIZE

        if (r, c) in self.env.retardant_tiles:
            base_color = RETARDANT_COLOR
        else:
            base_color = TILE_COLORS[state]

        pygame.draw.rect(self.screen, base_color, (x + 1, y + 1, TILE_SIZE - 2, TILE_SIZE - 2))

        if state == HEALTHY:
            cx, cy = x + TILE_SIZE // 2, y + TILE_SIZE // 2
            trunk_color = (80, 55, 30)
            leaf_color = (30, 110, 30)
            pygame.draw.rect(self.screen, trunk_color, (cx - 2, cy, 4, 12))
            pygame.draw.circle(self.screen, leaf_color, (cx, cy - 2), 8)
            pygame.draw.circle(self.screen, (40, 130, 40), (cx, cy - 4), 5)
        elif state == AT_RISK:
            cx, cy = x + TILE_SIZE // 2, y + TILE_SIZE // 2
            pygame.draw.polygon(self.screen, (255, 220, 50),
                                [(cx, cy - 10), (cx - 8, cy + 6), (cx + 8, cy + 6)])
            pygame.draw.rect(self.screen, (80, 60, 0), (cx - 1, cy - 4, 3, 6))
            pygame.draw.rect(self.screen, (80, 60, 0), (cx - 1, cy + 3, 3, 3))
        elif state == DEFORESTING:
            cx, cy = x + TILE_SIZE // 2, y + TILE_SIZE // 2
            for dx, dy in [(-6, -4), (4, 2), (-2, 6), (6, -6)]:
                flame_color = (255, 80 + (self.frame * 3 + dx * 7) % 100, 0)
                pygame.draw.circle(self.screen, flame_color, (cx + dx, cy + dy), 4)
            pygame.draw.circle(self.screen, (255, 200, 50), (cx, cy), 3)
        elif state == DESTROYED:
            cx, cy = x + TILE_SIZE // 2, y + TILE_SIZE // 2
            stump_color = (60, 35, 10)
            pygame.draw.rect(self.screen, stump_color, (cx - 4, cy - 2, 8, 8))
            pygame.draw.line(self.screen, (50, 30, 10), (cx - 6, cy + 6), (cx + 6, cy + 6), 2)

    def draw_grid(self):
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                self.draw_tile(r, c, self.env.grid[r, c])

        for i in range(GRID_SIZE + 1):
            x = GRID_OFFSET_X + i * TILE_SIZE
            y = GRID_OFFSET_Y + i * TILE_SIZE
            pygame.draw.line(self.screen, GRID_LINE_COLOR,
                             (x, GRID_OFFSET_Y), (x, GRID_OFFSET_Y + GRID_SIZE * TILE_SIZE))
            pygame.draw.line(self.screen, GRID_LINE_COLOR,
                             (GRID_OFFSET_X, y), (GRID_OFFSET_X + GRID_SIZE * TILE_SIZE, y))

    def draw_drone(self):
        dr, dc = self.env.drone_pos
        cx = GRID_OFFSET_X + dc * TILE_SIZE + TILE_SIZE // 2
        cy = GRID_OFFSET_Y + dr * TILE_SIZE + TILE_SIZE // 2

        bob = int(np.sin(self.frame * 0.15) * 3)

        pygame.draw.circle(self.screen, DRONE_SHADOW_COLOR, (cx + 2, cy + 2 + bob), 14)
        pygame.draw.circle(self.screen, DRONE_COLOR, (cx, cy + bob), 12)
        pygame.draw.circle(self.screen, (255, 255, 255), (cx, cy + bob), 5)

        for angle_offset in [0, 90, 180, 270]:
            rad = np.radians(angle_offset + self.frame * 8)
            px = cx + int(np.cos(rad) * 10)
            py = cy + bob + int(np.sin(rad) * 10)
            pygame.draw.circle(self.screen, (200, 200, 200), (px, py), 3)

    def draw_truck(self):
        if not self.env.truck_active:
            return
        tr, tc = self.env.truck_pos
        if tr >= GRID_SIZE:
            return
        x = GRID_OFFSET_X + tc * TILE_SIZE + TILE_SIZE // 2
        y = GRID_OFFSET_Y + tr * TILE_SIZE + TILE_SIZE // 2

        body_rect = pygame.Rect(x - 14, y - 8, 28, 16)
        pygame.draw.rect(self.screen, TRUCK_COLOR, body_rect, border_radius=3)
        pygame.draw.rect(self.screen, (180, 20, 40), pygame.Rect(x - 14, y - 8, 10, 12), border_radius=2)
        pygame.draw.circle(self.screen, (40, 40, 40), (x - 8, y + 10), 4)
        pygame.draw.circle(self.screen, (40, 40, 40), (x + 8, y + 10), 4)

    def draw_sidebar(self):
        sidebar_x = GRID_OFFSET_X + GRID_SIZE * TILE_SIZE + 20
        sidebar_rect = pygame.Rect(sidebar_x, 10, SIDEBAR_WIDTH - 30, WINDOW_HEIGHT - 20)
        pygame.draw.rect(self.screen, SIDEBAR_COLOR, sidebar_rect, border_radius=10)

        x = sidebar_x + 15
        y = 25

        title = self.font_large.render("UAV Monitor", True, DRONE_COLOR)
        self.screen.blit(title, (x, y))
        y += 45

        health = self.env._forest_health()
        health_color = (76, 175, 80) if health >= 0.7 else (255, 152, 0) if health >= 0.5 else (244, 67, 54)
        self._draw_stat_card(x, y, "Forest Health", f"{health:.1%}", health_color)
        y += 70

        bar_x, bar_y, bar_w, bar_h = x, y, SIDEBAR_WIDTH - 60, 14
        pygame.draw.rect(self.screen, (40, 40, 60), (bar_x, bar_y, bar_w, bar_h), border_radius=4)
        fill_w = int(bar_w * health)
        if fill_w > 0:
            pygame.draw.rect(self.screen, health_color, (bar_x, bar_y, fill_w, bar_h), border_radius=4)
        y += 30

        fuel_pct = self.env.fuel / MAX_FUEL
        fuel_color = (76, 175, 80) if fuel_pct > 0.4 else (255, 152, 0) if fuel_pct > 0.15 else (244, 67, 54)
        self._draw_stat_card(x, y, "Fuel", f"{self.env.fuel} / {MAX_FUEL}", fuel_color)
        y += 65

        bar_y = y
        pygame.draw.rect(self.screen, (40, 40, 60), (bar_x, bar_y, bar_w, bar_h), border_radius=4)
        fill_w = int(bar_w * fuel_pct)
        if fill_w > 0:
            pygame.draw.rect(self.screen, fuel_color, (bar_x, bar_y, fill_w, bar_h), border_radius=4)
        y += 30

        self._draw_stat_card(x, y, "Step", f"{self.env.step_count} / {MAX_STEPS}", TEXT_COLOR)
        y += 65

        self._draw_stat_card(x, y, "Threats Detected", str(len(self.env.detected_threats)), (255, 152, 0))
        y += 65

        self._draw_stat_card(x, y, "Ranger Calls", str(self.env.ranger_calls_left), (156, 39, 176))
        y += 65

        wind_label = WIND_LABELS.get(tuple(self.env.wind_dir), "?")
        rain_label = "Rain" if self.env.rain_active else "Clear"
        self._draw_stat_card(x, y, "Weather", f"Wind: {wind_label} | {rain_label}", (100, 180, 255))
        y += 65

        self._draw_stat_card(x, y, "Total Reward", f"{self.total_reward:.1f}", DRONE_COLOR)
        y += 65

        drone_label = f"Drone: ({self.env.drone_pos[0]}, {self.env.drone_pos[1]})"
        pos_text = self.font_small.render(drone_label, True, MUTED_COLOR)
        self.screen.blit(pos_text, (x, y))
        y += 20

        if self.env.truck_active:
            truck_label = f"Truck: ({self.env.truck_pos[0]}, {self.env.truck_pos[1]})"
        else:
            truck_label = "Truck: Stopped"
        truck_text = self.font_small.render(truck_label, True, MUTED_COLOR)
        self.screen.blit(truck_text, (x, y))
        y += 30

        log_title = self.font_small.render("Action Log", True, MUTED_COLOR)
        self.screen.blit(log_title, (x, y))
        y += 18
        for entry in self.action_log[-8:]:
            log_text = self.font_small.render(entry, True, (130, 130, 150))
            self.screen.blit(log_text, (x, y))
            y += 16

    def _draw_stat_card(self, x, y, label, value, color):
        card_w = SIDEBAR_WIDTH - 60
        card_rect = pygame.Rect(x, y, card_w, 55)
        pygame.draw.rect(self.screen, PANEL_COLOR, card_rect, border_radius=6)

        label_surf = self.font_small.render(label, True, MUTED_COLOR)
        self.screen.blit(label_surf, (x + 10, y + 5))

        value_surf = self.font_medium.render(value, True, color)
        self.screen.blit(value_surf, (x + 10, y + 25))

    def draw_legend(self):
        y = WINDOW_HEIGHT - 35
        x = GRID_OFFSET_X
        items = [
            ("Healthy", TILE_COLORS[HEALTHY]),
            ("At-Risk", TILE_COLORS[AT_RISK]),
            ("Deforesting", TILE_COLORS[DEFORESTING]),
            ("Destroyed", TILE_COLORS[DESTROYED]),
            ("Retardant", RETARDANT_COLOR),
            ("Drone", DRONE_COLOR),
            ("Truck", TRUCK_COLOR),
        ]
        for label, color in items:
            pygame.draw.rect(self.screen, color, (x, y, 14, 14), border_radius=2)
            text = self.font_small.render(label, True, MUTED_COLOR)
            self.screen.blit(text, (x + 18, y))
            x += 95

    def log_action(self, action, reward):
        name = ACTION_NAMES[action] if 0 <= action < 8 else "?"
        entry = f"S{self.env.step_count}: {name} R={reward:+.1f}"
        self.action_log.append(entry)
        if len(self.action_log) > 50:
            self.action_log.pop(0)

    def render(self):
        self.screen.fill(BG_COLOR)
        self.draw_grid()
        self.draw_truck()
        self.draw_drone()
        self.draw_sidebar()
        self.draw_legend()
        self.frame += 1
        pygame.display.flip()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return "quit"
                if event.key == pygame.K_SPACE:
                    return "step"
                if event.key == pygame.K_r:
                    return "reset"
                if event.key == pygame.K_a:
                    return "auto"
        return None

    def tick(self, fps=30):
        self.clock.tick(fps)

    def close(self):
        pygame.quit()


def run_random_demo():
    env = DeforestationEnv()
    obs, _ = env.reset()
    renderer = ForestRenderer(env)
    paused = False
    done = False

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
        renderer.tick(3)

    renderer.close()
    env.close()


def run_with_model(model, deterministic=True):
    env = DeforestationEnv()
    obs, _ = env.reset()
    renderer = ForestRenderer(env)
    paused = False
    done = False

    running = True
    while running:
        event = renderer.handle_events()
        if event == "quit":
            running = False
        elif event == "step" and not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            action = int(action)
            obs, reward, terminated, truncated, info = env.step(action)
            renderer.total_reward += reward
            renderer.log_action(action, reward)
            done = terminated or truncated
        elif event == "reset":
            obs, _ = env.reset()
            renderer.total_reward = 0
            renderer.action_log = []
            done = False
            paused = False
        elif event == "auto":
            paused = not paused

        if not paused and not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            action = int(action)
            obs, reward, terminated, truncated, info = env.step(action)
            renderer.total_reward += reward
            renderer.log_action(action, reward)
            done = terminated or truncated

        renderer.render()
        renderer.tick(3)

    renderer.close()
    env.close()


if __name__ == "__main__":
    run_random_demo()
