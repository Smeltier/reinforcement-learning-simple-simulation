import numpy as np

class QLearningAgent:

    _alpha: float # G: taxa de aprendizado
    _gamma: float # G: fator de desconto
    _epsilon: float # G: chance de exploração
    _number_of_actions: int

    def __init__(self, rows: int, cols: int, number_of_actions: int = 4, alpha: float = 0.1, gamma: float = 0.99, epsilon: float = 0.1) -> None:
        self._q_table = np.zeros((rows, cols, number_of_actions))
        self._alpha = alpha
        self._gamma = gamma
        self._epsilon = epsilon
        self._number_of_actions = number_of_actions

    def choose_action(self, state: tuple[int, int]) -> int:
        """ Estratégia Epsilon-Greedy """
        row, col = state        

        if np.random.uniform(0, 1) < self._epsilon:
            return np.random.randint(0, self._number_of_actions)
        
        return int(np.argmax(self._q_table[row, col]))
    
    def learn(self, state: tuple[int, int], action: int, reward: float, next_state: tuple[int, int]) -> None:
        """ Atualiza a Q-Table usando a Equação de Bellman """
        row, col = state
        new_row, new_col = next_state

        old_value = self._q_table[row, col, action]
        next_max_value = np.max(self._q_table[new_row, new_col])

        # G: Equação de Bellman:
        # Q(s,a) = Q(s,a) + alpha * (R + gamma * max(Q(s')) - Q(s,a))
        new_value = old_value + self._alpha * (reward + self._gamma * next_max_value - old_value)

        self._q_table[row, col, action] = new_value

    def save_model(self, file_name: str) -> None:
        np.save(file_name, self._q_table)

    def load_model(self, file_name: str) -> None:
        try:
            self._q_table = np.load(file_name)
        except FileNotFoundError:
            print("Arquivo de modelo não encontrado. Começando do zero.")

    def log_step(self, state: tuple[int, int], action: int, reward: float) -> None:
        """ Imprime no terminal o estado atual do aprendizado para este passo. """
        row, col = state
        
        action_names = ['CIMA ', 'BAIXO', 'ESQ. ', 'DIR  ']
        action_str = action_names[action] if 0 <= action < 4 else str(action)

        current_q_values = self._q_table[row, col]
        q_vals_str = " | ".join([f"{v:6.2f}" for v in current_q_values])

        print(f"Pos: {str(state):<8} | "
              f"Ação: {action_str} | "
              f"R: {reward:3} | "
              f"Q-Values [C, B, E, D]: [{q_vals_str}]")
        
    def show_q_table(self) -> None:
        print("Q-table:")
        print(self._q_table)

    @property
    def q_table(self) -> np.ndarray:
        return self._q_table
