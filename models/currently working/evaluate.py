import yaml
import numpy as np
import matplotlib.pyplot as plt
from environment import HoneypotEnv  # Adjust based on your file structure
from models.ddqn_model import DDQNAgent  # Adjust based on your file structure
from models.sarsa_model import SARSAAgent  # Adjust based on your file structure
from models.a2c_model import A2CAgent  # Adjust based on your file structure

# Load configuration
with open("model_config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Initialize environment
env = HoneypotEnv(num_nodes=5)
num_episodes = 20

# Define models
models = {
    "ddqn": DDQNAgent(env.observation_space.shape, env.action_space.n, config["ddqn"]),
    "sarsa": SARSAAgent(env.observation_space.shape, env.action_space.n, config["sarsa"]),
    "a2c": A2CAgent(env.observation_space.shape, env.action_space.n, config["a2c"])
}

# Load pre-trained weights
weights_paths = {
    "ddqn": "path_to_your_ddqn_model_weights.h5",
    "sarsa": "path_to_your_sarsa_model_weights.h5",
    "a2c": "path_to_your_a2c_model_weights.h5"
}

for agent_name, agent in models.items():
    agent.model.load_weights(weights_paths[agent_name])  # Load model weights

# Evaluate models
results = {}

for agent_name, agent in models.items():
    episode_rewards = []
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.choose_action(state)  # Assuming a method to choose action
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            state = next_state
        episode_rewards.append(total_reward)
        print(f"{agent_name.upper()} - Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

    # Store rewards for this agent
    results[agent_name] = episode_rewards

# Plot results for comparison
for agent_name, rewards in results.items():
    plt.plot(rewards, label=agent_name.upper())

plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Evaluation Comparison of DDQN, SARSA, and A2C')
plt.legend()
plt.show()
