import random as rd
import numpy as np

from src.agents.base_agent import BaseAgent

class ApproximateQLearningAgent(BaseAgent):
    def __init__(self, number_of_features:int, number_of_actions:int=4, alpha:float=0.1, gamma:float=0.99, epsilon:float=0.1) -> None:
        super().__init__(number_of_actions, alpha, gamma, epsilon)
        self._weights = np.zeros(number_of_features)

    def update(self, state, action, maze) -> tuple[tuple[int, int], float, bool]:
        next_state = maze.predict_next_state(state, action)
        reward = maze.get_reward(next_state)
        done = maze.is_target(*next_state)

        self.learn(state, action, reward, next_state, done, maze)

        return next_state, reward, done

    def get_features(self, state: tuple[int, int], action: int, maze) -> np.ndarray:
        row, col = state
        dr, dc = [(-1, 0), (1, 0), (0, -1), (0, 1)][action]
        intended_row, intended_col = row + dr, col + dc

        next_state = maze.predict_next_state(state, action)
        all_targets = maze._find_all_targets()
        
        distances = [np.linalg.norm(np.array(next_state) - np.array(t)) for t in all_targets]
        min_dist = min(distances) if distances else 0
        
        max_dist = (maze.rows**2 + maze.cols**2)**0.5
        
        f0 = 1.0
        f1 = min_dist / max_dist 
        f2 = 1.0 if not maze.is_on_limits(intended_row, intended_col) or maze.is_wall(intended_row, intended_col) else 0.0

        return np.array([f0, f1, f2])
    
    def get_q_value(self, state:tuple[int, int], action:int, maze) -> float:
        features = self.get_features(state, action, maze)
        return np.dot(self._weights, features)

    def choose_action(self, state: tuple[int, int], maze=None) -> int:
        if rd.uniform(0, 1) < self._epsilon:
            return rd.randrange(0, self._number_of_actions)

        q_values = [self.get_q_value(state, a, maze) for a in range(self._number_of_actions)]
        return int(np.argmax(q_values))

    def learn(self, state, action, reward, next_state, done, maze) -> None:
        features = self.get_features(state, action, maze)
        old_q = self.get_q_value(state, action, maze)

        if done:
            target = reward
        else:
            next_qs = [self.get_q_value(next_state, a, maze) for a in range(self._number_of_actions)]
            target = reward + self._gamma * max(next_qs)

        self._weights += self._alpha * (target - old_q) * features

    def save_model(self, file_name: str) -> None:
        np.save(file_name, self._weights)

    def load_model(self, file_name: str) -> None:
        try:
            self._weights = np.load(file_name)
        except FileNotFoundError:
            print("Arquivo não encontrado.")
