import os
import time
import torch

from stable_baselines3 import PPO

from agent import SnakeBaseAgent
from config import PPO_SB_TRAIN_HYPER_PARAMS


class PPO_Stable_Baselines_Agent(SnakeBaseAgent):
    def __init__(self):
        super().__init__()
        self.env = None
        self.model = None
        self.model_path = "model/ppo_sb"

    def init_model(self, env):
        if self.model:
            return

        self.env = env

        # load previously trained model if exists
        if os.path.exists(self.model_path + ".zip"):
            self.model = PPO.load(self.model_path, env)
        else:
            self.model = PPO(env=env, verbose=1, **PPO_SB_TRAIN_HYPER_PARAMS)

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
        with torch.no_grad():
            state_dist = self.model.policy.get_distribution(obs_tensor)

            action_probs = state_dist.distribution.probs.squeeze().cpu().numpy()

        print("Action distribution:", action_probs)

        action, _ = self.model.predict(state, deterministic=True)
        return action
