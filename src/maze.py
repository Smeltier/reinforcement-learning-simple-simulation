import random

import pygame

class Maze:

    _matrix: list[list[int]]
    _start_position: tuple[int, int]
    _agent_position: tuple[int, int]

    EMPTY, WALL, START, TARGET = 0, 1, 2, 3

    def __init__(self, maze_path: str) -> None:
        self._matrix = self._load_matrix(maze_path)
        self._rows = len(self._matrix)
        self._cols = len(self._matrix[0])
        self._start_position = self._find_item(self.START)
        self._agent_position = self._start_position

    def step(self, action: int) -> tuple[tuple[int, int], int, bool]:
        row, col = self._agent_position
        delta_row, delta_col = [(-1, 0), (1, 0), (0, -1), (0, 1)][action]

        new_row, new_col = row + delta_row, col + delta_col

        if not (0 <= new_row < self._rows and 0 <= new_col < self._cols) or self.is_wall(new_row, new_col):
            return (row, col), -10, False
        
        self._agent_position = (new_row, new_col)

        if self._matrix[new_row][new_col] == self.TARGET:
            return (new_row, new_col), 100, True
        
        return (new_row, new_col), -1, False # !

    def reset(self) -> tuple[int, int]:
        valid_slots: list[tuple[int, int]] = []
        
        for row in range(self._rows):
            for col in range(self._cols):
                cell = self._matrix[row][col]
                if cell == self.EMPTY or cell == self.START:
                    valid_slots.append((row, col))
        
        if valid_slots:
            self._agent_position = random.choice(valid_slots)
        else:
            self._agent_position = self._start_position
            
        return self._agent_position
    
    def is_on_limits(self, row: int, col: int) -> bool:
        return 0 <= row < self._rows and 0 <= col < self._cols

    def is_target(self, row: int, col: int) -> bool:
        return self._matrix[row][col] == self.TARGET

    def is_wall(self, row: int, col: int) -> bool: 
        if not self.is_on_limits(row, col):
            return True
        return self._matrix[row][col] == self.WALL

    def draw(self, surface: pygame.Surface) -> None:
        matrix_rows = len(self._matrix)
        matrix_cols = len(self._matrix[0])

        cell_width = surface.get_width() / self._cols  
        cell_height = surface.get_height() / self._rows

        wall_color = pygame.Color('white')
        entity_color = pygame.Color('blue')
        target_color = pygame.Color('yellow')

        for row in range(matrix_rows):
            for col in range(matrix_cols):
                
                x = col * cell_width
                y = row * cell_height

                cell = self._matrix[row][col]

                if cell == self.WALL: 
                    pygame.draw.rect(surface, wall_color, (x, y, cell_width, cell_height))
                if cell == self.START:
                    pygame.draw.rect(surface, entity_color, (x, y, cell_width, cell_height))
                if cell == self.TARGET:
                    pygame.draw.rect(surface, target_color, (x, y, cell_width, cell_height))

        agent_row, agent_col = self._agent_position
        agent_x = agent_col * cell_width
        agent_y = agent_row * cell_height

        pygame.draw.rect(surface, entity_color, (agent_x, agent_y, cell_width, cell_height))

    @property
    def rows(self) -> int:
        return self._rows
    
    @property
    def cols(self) -> int:
        return self._cols
    
    def _load_matrix(self, maze_path: str) -> list[list[int]]:
        matrix: list[list[int]] = []

        with open(maze_path, 'r') as f:
            for line in f:
                matrix.append([int(x) for x in line.split()])

        return matrix
    
    def _find_item(self, item: int) -> tuple[int, int]:
        for row in range(self._rows):
            for col in range(self._cols):
                if self._matrix[row][col] == item: 
                    return (row, col)
                
        return (0,0)
