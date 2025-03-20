from env.snake_env import SnakeEnv
from ui.snake_renderer import SnakeRenderer
from config.params import GRID_SIZE, CELL_SIZE, SPEED

from agent.human import HumanAgent


if __name__ == "__main__":
    renderer = SnakeRenderer(grid_size=GRID_SIZE, cell_size=CELL_SIZE, render_rate=SPEED)
    agent = HumanAgent()
    env = SnakeEnv(grid_size=GRID_SIZE, renderer=renderer, agent=agent)

    env.run()
