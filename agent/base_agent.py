from abc import abstractmethod

import numpy as np

from utils import Direction


class SnakeBaseAgent:
    def __init__(self):
        self.score = 0

    @abstractmethod
    def get_action(self, state):
        """Subclasses Must Implement. Returns an action based on the environment state. Return -1 to exit"""
        pass

    def _infer_direction(self, state):
        """Infers the snake's direction based on its head and body inferred from env state"""
        snake_head = np.argwhere(state == 1)
        snake_body = np.argwhere(state == 2)

        head_y, head_x = snake_head[0]
        # second segment is the closest segment to head
        second_y, second_x = min(snake_body, key=lambda seg: abs(seg[0] - head_y) + abs(seg[1] - head_x))

        if head_x == second_x and head_y < second_y:
            return Direction.UP
        elif head_x == second_x and head_y > second_y:
            return Direction.DOWN
        elif head_y == second_y and head_x > second_x:
            return Direction.RIGHT
        elif head_y == second_y and head_x < second_x:
            return Direction.LEFT
