import numpy as np
import pandas as pd
import tensorflow as tf
from HoneypotEnv import HoneypotEnv  # Replace with your environment import
from models.ddqn_model import DDQNAgent  # Replace with your model import
from models.sarsa_model import SARSAAgent  # Assuming SARSAAgent is imported correctly
from models.a2c_model import A2CAgent  # Assuming A2CAgent is imported correctly
import yaml
import os
import matplotlib.pyplot as plt


# Check if weight files exist
print(os.path.exists("DDQN_training_weight.weights.h5"))
print(os.path.exists("SARSA_training_weight.weights.h5"))
print(os.path.exists("A2C_actor_training_weight.weights.h5"))
print(os.path.exists("A2C_critic_training_weight.weights.h5"))

# Load configuration
env = HoneypotEnv(num_nodes = 10, num_honeypots = 4, honeypot_cost = 0.5)
state_shape = env.observation_space.shape
action_shape = env.action_space.n

with open("model_config.yaml", "r") as file:
    model_config = yaml.safe_load(file)

# Initialize the models
ddqn_model = DDQNAgent(state_shape, action_shape, model_config["ddqn"])
sarsa_model = SARSAAgent(state_shape, action_shape, model_config["sarsa"])
a2c_model = A2CAgent(state_shape, action_shape, model_config["a2c"])

# Load weights
ddqn_model.model.load_weights("DDQN_training_weight.weights.h5")
sarsa_model.model.load_weights("SARSA_training_weight.weights.h5")
a2c_model.actor_model.load_weights("A2C_actor_training_weight.weights.h5")
a2c_model.critic_model.load_weights("A2C_critic_training_weight.weights.h5")

# Initialize rewards lists for each model
ddqn_rewards, sarsa_rewards, a2c_rewards = [], [], []

# Number of evaluation episodes
num_episodes = 50

for episode in range(num_episodes):
    state = env.reset()
    done = False

    total_reward_ddqn = 0
    total_reward_sarsa = 0
    total_reward_a2c = 0

    while not done:
        # Get actions from each model
        action_ddqn = ddqn_model.choose_action(state)
        action_sarsa = sarsa_model.choose_action(state)
        action_a2c = a2c_model.choose_action(state)

        # Step through the environment for each model
        # Using DDQN to step through the environment
        next_state, reward, done, _ = env.step(action_ddqn)
        total_reward_ddqn += reward
        state = next_state

    # Store total reward for the DDQN model
    ddqn_rewards.append(total_reward_ddqn)

    # Reset the environment for SARSA evaluation
    state = env.reset()
    done = False

    while not done:
        # Get action from SARSA model
        action_sarsa = sarsa_model.choose_action(state)

        # Step through the environment for SARSA
        next_state, reward, done, _ = env.step(action_sarsa)
        total_reward_sarsa += reward
        state = next_state

    # Store total reward for the SARSA model
    sarsa_rewards.append(total_reward_sarsa)

    # Reset the environment for A2C evaluation
    state = env.reset()
    done = False

    while not done:
        # Get action from A2C model
        action_a2c = a2c_model.choose_action(state)

        # Step through the environment for A2C
        next_state, reward, done, _ = env.step(action_a2c)
        total_reward_a2c += reward
        state = next_state

    # Store total reward for the A2C model
    a2c_rewards.append(total_reward_a2c)

# Optionally, compute average rewards
average_ddqn_reward = np.mean(ddqn_rewards)
average_sarsa_reward = np.mean(sarsa_rewards)
average_a2c_reward = np.mean(a2c_rewards)

print(f"Average DDQN Reward: {average_ddqn_reward}")
print(f"Average SARSA Reward: {average_sarsa_reward}")
print(f"Average A2C Reward: {average_a2c_reward}")

# Plotting results
plt.plot(ddqn_rewards, label='DDQN')
plt.plot(sarsa_rewards, label='SARSA')
plt.plot(a2c_rewards, label='A2C')
plt.xlabel('Episodes')
plt.ylabel('Total Reward')
plt.title('Model Evaluation')
plt.legend()
plt.show()


results_df = pd.DataFrame({
    'Episode': range(num_episodes),
    'DDQN Reward': ddqn_rewards,
    'SARSA Reward': sarsa_rewards,
    'A2C Reward': a2c_rewards
})

results_df.to_csv('evaluation_results.csv', index=False)
