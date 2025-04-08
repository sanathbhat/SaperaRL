DQN_SB_TRAIN_HYPER_PARAMS = {
    "policy": "MlpPolicy",
    "learning_rate": 1e-3,
    "buffer_size": 50_000,
    "device": "cuda",
    "exploration_fraction": 0.9,
    "exploration_final_eps": 0.1,
    "tensorboard_log": "tensorboard_logs"
}
DQN_TRAIN_TIMESTEPS = 50000
DQN_TRAIN_EPOCHS = 400

PPO_SB_TRAIN_HYPER_PARAMS = {
    "policy": "MlpPolicy",
    "learning_rate": 3e-4,
    "ent_coef": 0.2,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "tensorboard_log": "tensorboard_logs"
}

PPO_TRAIN_TIMESTEPS = 25000
PPO_TRAIN_EPOCHS = 200

PPO_LSTM_SB_TRAIN_HYPER_PARAMS = {
    "policy": "MlpLstmPolicy",
    "learning_rate": 1e-3,
    "ent_coef": 0.01,
    "max_grad_norm": 0.1,
    "n_steps": 8192,
    "batch_size": 64,
    "n_epochs": 10,
    "policy_kwargs": dict(lstm_hidden_size=32),
    "tensorboard_log": "tensorboard_logs"
}

PPO_LSTM_TRAIN_TIMESTEPS = 24576
PPO_LSTM_TRAIN_EPOCHS = 100
