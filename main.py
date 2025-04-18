from agent import DQN_Stable_Baselines_Agent
from agent import PPO_Stable_Baselines_Agent
from agent import PPO_LSTM_Stable_Baselines_Agent
from env.snake_env import SnakeEnv

from config import DQN_TRAIN_TIMESTEPS, DQN_TRAIN_EPOCHS, PPO_TRAIN_TIMESTEPS, PPO_TRAIN_EPOCHS, PPO_LSTM_TRAIN_EPOCHS, \
    PPO_LSTM_TRAIN_TIMESTEPS

from ui.snake_renderer import SnakeRenderer
from config.generic import GRID_SIZE, CELL_SIZE, SPEED

from agent.human import HumanAgent


if __name__ == "__main__":
    renderer = SnakeRenderer(grid_size=GRID_SIZE, cell_size=CELL_SIZE, render_rate=SPEED)

    # Human play
    # agent = HumanAgent()
    # env = SnakeEnv(grid_size=GRID_SIZE, renderer=renderer, agent=agent)
    # env.run()

    # DQN Stable Baselines 3
    # agent = DQN_Stable_Baselines_Agent()
    # agent = PPO_Stable_Baselines_Agent()
    agent = PPO_LSTM_Stable_Baselines_Agent()

    env = SnakeEnv(grid_size=GRID_SIZE, renderer=renderer, agent=agent)

    # for epoch in range(PPO_LSTM_TRAIN_EPOCHS):
    #     agent.score = 0
    #     print(f"Training epoch {epoch}...")
    #     env.train_agent(timesteps=PPO_LSTM_TRAIN_TIMESTEPS)
    #
    #     print(f"Testing epoch {epoch}...")
    #     env.run(close_renderer_after_run=False)

    # Test
    env.run(close_renderer_after_run=True)
