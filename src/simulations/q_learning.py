import argparse

import pygame

import src.utils as utils

from src.maze import Maze
from src.agents.q_learning_agent import QLearningAgent

def run(max_episodes=0, save_progress=False, path_saved=None, show_results=False, render=True):
    pygame.init()
    pygame.display.set_caption("Q-learning Simulation")

    WIDTH, HEIGHT = 1000, 800
    SCREEN = None
    CLOCK = None
    FPS = 20

    if render:
        SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
        CLOCK = pygame.time.Clock()

    maze = Maze('playground.txt')
    agent = QLearningAgent(rows = maze.rows, cols = maze.cols)

    if path_saved is not None:
        agent.load_model(path_saved)

        if render:
            agent._epsilon = 0.0

    state = maze.reset()
    episode_number = 0
    total_reward = 0
    history = []

    running = True
    while running:

        if render:
            assert SCREEN is not None
            assert CLOCK is not None

            SCREEN.fill('black') 
            CLOCK.tick(FPS)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

        if episode_number != 0 and episode_number == max_episodes:
            running = False

        action = agent.choose_action(state)

        next_state, reward, done = agent.update(state, action, maze)

        total_reward += reward

        agent.learn(state, action, reward, next_state, done, maze)
        # agent.log_step(state, action, reward)

        state = next_state
        maze._agent_position = next_state

        if done:
            print(f"Episode {episode_number} | Total Reward: {total_reward}")

            state = maze.reset()

            if show_results:
                history.append(total_reward)

            total_reward = 0
            episode_number += 1     

        if render:
            assert SCREEN is not None

            maze.draw(SCREEN)
            utils.draw_color_map(SCREEN, maze, next_state, agent._q_table)
            pygame.display.flip()  

    if save_progress:
        agent.save_model(f'treino_{episode_number}.npy')

    if show_results:
        utils.show_rewards_graph(history)

    pygame.quit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Approximate Q-learning Simulataion")

    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--results', action='store_true')
    parser.add_argument('--no-render', action='store_false', dest='render')
    parser.set_defaults(render=True)

    args = parser.parse_args()
    
    run(
        max_episodes=args.episodes,
        save_progress=args.save,
        path_saved=args.load,
        show_results=args.results,
        render=args.render
    )
