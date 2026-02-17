from abc import ABC, abstractmethod

class BaseAgent(ABC):
    _number_of_actions: int
    _alpha: float
    _gamma: float
    _epsilon: float

    def __init__(self, number_of_actions, alpha, gamma, epsilon) -> None:
        self._number_of_actions = number_of_actions
        self._alpha = alpha
        self._gamma = gamma
        self._epsilon = epsilon

    @abstractmethod
    def update(self, state, action, maze) -> tuple[tuple[int,int], float, bool]:
        pass

    @abstractmethod
    def choose_action(self, state:tuple[int,int], maze=None) -> int:
        pass

    @abstractmethod
    def learn(self, state, action, reward, next_state, done, maze) -> None:
        pass

    @abstractmethod
    def save_model(self, file_name:str) -> None:
        pass

    @abstractmethod
    def load_model(self, file_name:str) -> None:
        pass
