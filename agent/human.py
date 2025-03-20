import pygame

from agent.base_agent import SnakeBaseAgent
from utils.structs import Direction


class HumanAgent(SnakeBaseAgent):
    """
    Human controller agent: Uses arrow keys to control snake
    """

    def __init__(self):
        super().__init__()
        self.action_mapping = {
            pygame.K_UP: Direction.UP,
            pygame.K_RIGHT: Direction.RIGHT,
            pygame.K_DOWN: Direction.DOWN,
            pygame.K_LEFT: Direction.LEFT,
            pygame.K_ESCAPE: -1
        }

    def get_action(self, state):
        current_direction = self._infer_direction(state)

        action = 0  # default: No-op
        # Handle keyboard input
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key in self.action_mapping:
                    target_direction = self.action_mapping[event.key]
                    action = self._get_relative_action(current_direction, target_direction)

        return action

    def _get_relative_action(self, current_direction, target_direction):
        if target_direction == -1:
            return -1

        direction_cycle_order = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]

        if current_direction == target_direction:  # ignore same direction. options: speed up snake??
            return 0

        snake_direction_index = direction_cycle_order.index(current_direction)
        target_direction_index = direction_cycle_order.index(target_direction)

        if (snake_direction_index - 1) % 4 == target_direction_index:  # left turn if cycling anti-clockwise
            return 1
        elif (snake_direction_index + 1) % 4 == target_direction_index:  # right turn if cycling clockwise
            return 2

        return 0
