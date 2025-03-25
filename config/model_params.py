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
