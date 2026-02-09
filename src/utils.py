import pygame
import numpy as np
import matplotlib.pyplot as plt

from maze import Maze

def draw_color_map(surface: pygame.Surface, maze: Maze, state, table) -> None:
    row, col = state

    if maze.is_wall(row, col):
        return
    
    max_abs_value: float = np.max(np.abs(table))

    if max_abs_value == 0.0:
        max_abs_value = 1.0
    
    rows = maze.rows
    cols = maze.cols
    cell_width = surface.get_width() / cols
    cell_height = surface.get_height() / rows

    directions = ((-1, 0), (1, 0), (0, -1), (0, 1))
    action = [0, 1, 2, 3]

    for it, (dx, dy) in enumerate(directions):
        new_col = col + dy
        new_row = row + dx

        if maze.is_wall(new_row, new_col) or maze.is_target(new_row, new_col):
            continue

        x = new_col * cell_width
        y = new_row * cell_height

        q_value: float = table[row, col, action[it]]

        ratio = abs(q_value) / max_abs_value
        intensity = int(ratio * 255)

        if q_value > 0:
            pygame.draw.rect(surface, (0, intensity, 0), (x, y, cell_width, cell_height))
        elif q_value < 0:
            pygame.draw.rect(surface, (intensity, 0, 0), (x, y, cell_width, cell_height))

        pygame.draw.rect(surface, "grey", (x, y, cell_width, cell_height), width=1)

def show_rewards_graph(history: list, window:int=100) -> None:
    rewards = np.array(history)

    if len(rewards) < window:
        window = len(rewards)

    rolling_mean = np.convolve(
        rewards, np.ones(window) / window, mode="valid"
    )

    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.3)
    plt.plot(
        range(window - 1, len(rewards)),
        rolling_mean,
        linewidth=2,
    )

    plt.title("Evolução do Agente")
    plt.xlabel("Episódios")
    plt.ylabel("Recompensa Total")
    plt.legend()
    plt.grid(True)
    plt.show()