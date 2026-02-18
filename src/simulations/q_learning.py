import pygame

import src.utils as utils

from src.maze import Maze
from src.agents.q_learning_agent import QLearningAgent

def run(max_episodes=0, save_progress=False, path_saved=None, show_results=False, render=True):
    WIDTH, HEIGHT = 1000, 800
    SCREEN = None
    CLOCK = None
    FPS = 20

    if render:
        SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
        CLOCK = pygame.time.Clock()

    maze = Maze('data/playground.txt')
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

    return 

if __name__ == '__main__':
    pygame.init()
    pygame.display.set_caption('Q-Learning Simulation')
    run(path_saved="treino_10000.npy")
    pygame.quit()
