import os
import time
import torch

from sb3_contrib import RecurrentPPO

from agent import SnakeBaseAgent
from config import PPO_LSTM_SB_TRAIN_HYPER_PARAMS


class PPO_LSTM_Stable_Baselines_Agent(SnakeBaseAgent):
    def __init__(self):
        super().__init__()
        self.env = None
        self.model = None
        self.model_path = "model/ppo_lstm_sb"

    def init_model(self, env):
        if self.model:
            return

        self.env = env

        # load previously trained model if exists
        if os.path.exists(self.model_path + ".zip"):
            self.model = RecurrentPPO.load(self.model_path, env)
            self.model.max_grad_norm = 0.1
        else:
            self.model = RecurrentPPO(env=env, verbose=1, **PPO_LSTM_SB_TRAIN_HYPER_PARAMS)

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

        n_lstm_layers, _, n_lstm_units = self.model.policy.lstm_hidden_state_shape

        lstm_states = (
            torch.zeros(n_lstm_layers, 1, n_lstm_units, device=self.model.device),
            torch.zeros(n_lstm_layers, 1, n_lstm_units, device=self.model.device)
        )
        episode_starts = torch.tensor([1], dtype=torch.float16)

        with torch.no_grad():
            state_dist, lstm_states = self.model.policy.get_distribution(obs_tensor, lstm_states, episode_starts)

            action_probs = state_dist.distribution.probs.squeeze().cpu().numpy()

        print("Action distribution:", action_probs)

        action, _ = self.model.predict(state, deterministic=True)
        return action
