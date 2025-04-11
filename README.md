A framework and some AI agents to master the classic snake game.

The following type of agents are available in this repo:

1. Human Agent (use arrow keys to control snake)
2. Deep Q-network (DQN) (stable baselines 3) - Deep Learning based agent that optimizes the Q values of states to generate an optimal policy for the agent.
3. Proximal Policy Optimization (PPO) (stable baselines 3) - Deep Learning based agent that directly optimizes the policy using the PPO algorithm.
4. Recurrent PPO (stable baselines 3 contrib) - Adds LSTM layers to generate recurrent policies using PPO that can 'plan ahead'