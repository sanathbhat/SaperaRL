import os
import time
import torch

from stable_baselines3 import DQN
from stable_baselines3.common.utils import constant_fn

from agent import SnakeBaseAgent
from config import DQN_SB_TRAIN_HYPER_PARAMS


class DQN_Stable_Baselines_Agent(SnakeBaseAgent):
    def __init__(self):
        super().__init__()
        self.env = None
        self.model = None
        self.model_path = "model/dqn_sb"

    def init_model(self, env):
        if self.model:
            return

        self.env = env

        # load previously trained model if exists
        if os.path.exists(self.model_path + ".zip"):
            self.model = DQN.load(self.model_path, env)
        else:
            self.model = DQN(env=env, verbose=1, **DQN_SB_TRAIN_HYPER_PARAMS)

        self.model.lr_schedule = constant_fn(1e-3)  # sets a new learning rate
        self.model.policy.optimizer.param_groups[0]['lr'] = 1e-3  # update optimize

    def train(self, timesteps):
        if not self.model:
            raise AttributeError("Model not initialized. Call init_model() first")

        start = time.time()
        self.model.learn(total_timesteps=timesteps)
        print(f"Training time: {time.time() - start} sec")

        self.model.save(self.model_path)

    def get_action(self, state):
        if not self.model:
            raise AttributeError("Model not initialized. Call init_model() first")

        obs_tensor = torch.tensor(state).unsqueeze(0).float().to(self.model.device)
        q_values = self.model.q_net(obs_tensor)
        print(state)
        print("Q-values:", q_values.detach().cpu().numpy())

        action, _ = self.model.predict(state, deterministic=True)
        return action
