import random as rd
import numpy as np

from src.agents.base_agent import BaseAgent

class QLearningAgent(BaseAgent):
    def __init__(self, rows: int, cols: int, number_of_actions: int = 4, alpha: float = 0.1, gamma: float = 0.99, epsilon: float = 0.1) -> None:
        super().__init__(number_of_actions, alpha, gamma, epsilon)
        self._q_table = np.zeros((rows, cols, number_of_actions))

    def update(self, state, action, maze) -> tuple[tuple[int,int], float, bool]:
        next_state = maze.predict_next_state(state, action)
        reward = maze.get_reward(next_state)
        done = maze.is_target(*next_state)

        self.learn(state, action, reward, next_state, done, maze)

        return next_state, reward, done

    def learn(self, state, action, reward, next_state, done, maze) -> None:
        row, col = state
        nrow, ncol = next_state

        old_value = self._q_table[row, col, action]

        if done:
            target = reward
        else:
            next_max = np.max(self._q_table[nrow, ncol])
            target = reward + self._gamma * next_max

        self._q_table[row, col, action] += self._alpha * (target - old_value)

    def choose_action(self, state: tuple[int, int], maze=None) -> int:
        row, col = state

        if rd.uniform(0, 1) <  self._epsilon:
            return rd.randrange(0, self._number_of_actions)

        return int(np.argmax(self._q_table[row, col]))

    def save_model(self, file_name: str) -> None:
        np.save(file_name, self._q_table)

    def load_model(self, file_name: str) -> None:
        try:
            self._q_table = np.load(file_name)
        except FileNotFoundError:
            print("Arquivo não encontrado. Iniciando com a tabela de zeros")
