import pandas as pd
import yaml
import matplotlib.pyplot as plt
from testEnv import testEnv
# from HoneypotEnv import HoneypotEnv
from ddqn_model import DDQNAgent
from sarsa_model import SARSAAgent
from a2c_model import A2CAgent
import tensorflow as tf
from tensorflow.keras import Sequential, layers

# Load the malicious IPs from the CSV file
def load_malicious_ips(csv_file):
    df = pd.read_csv(csv_file)
    # Extract the 'Src IP' and 'Anomaly' columns where Anomaly == 1
    malicious_ips = df[df['Anomaly'] == 1]['Src IP'].tolist()
    print(f"Loaded malicious IPs: {malicious_ips}")  # Debug: Print loaded malicious IPs
    return malicious_ips

# Load model configurations
with open("temp/model_config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Define a function to load evaluation results and select the best model
def select_best_model(evaluation_csv):
    # Load evaluation CSV
    evaluation_results = pd.read_csv(evaluation_csv)

    # Calculate average and max rewards for each model
    ddqn_avg_reward = evaluation_results["DDQN Reward"].mean()
    sarsa_avg_reward = evaluation_results["SARSA Reward"].mean()
    a2c_avg_reward = evaluation_results["A2C Reward"].mean()

    ddqn_max_reward = evaluation_results["DDQN Reward"].max()
    sarsa_max_reward = evaluation_results["SARSA Reward"].max()
    a2c_max_reward = evaluation_results["A2C Reward"].max()

    # Select model based on the highest max reward
    if ddqn_max_reward >= sarsa_max_reward and ddqn_max_reward >= a2c_max_reward:
        best_model_name = "ddqn"
    elif sarsa_max_reward >= ddqn_max_reward and sarsa_max_reward >= a2c_max_reward:
        best_model_name = "sarsa"
    else:
        best_model_name = "a2c"

    print(f"Best model selected based on evaluation: {best_model_name}")
    return best_model_name

# Initialize and train the best model
def train_best_model(best_model_name, config):
    # Initialize the environment
    # Example usage:
    malicious_ips = load_malicious_ips('ip_prediction_output.csv')
    env = testEnv(num_nodes=10, num_honeypots=4, honeypot_cost=0.5, malicious_ips=malicious_ips)

    # env = HoneypotEnv(num_nodes=10, num_honeypots=4, honeypot_cost=0.5, malicious_ips=malicious_ips)

    # Select agent based on best model name
    if best_model_name == "ddqn":
        agent = DDQNAgent((env.observation_space.shape[0],), env.action_space.n, config[best_model_name])
    elif best_model_name == "sarsa":
        agent = SARSAAgent((env.observation_space.shape[0],), env.action_space.n, config[best_model_name])
    else:
        agent = A2CAgent((env.observation_space.shape[0],), env.action_space.n, config[best_model_name])

    # Continue training with the best model
    max_episodes = 10
    max_steps = 200
    episode_rewards = []

    for episode in range(max_episodes):
        state = env.reset().reshape(1, -1)
        total_reward = 0
        done = False
        step_count = 0

        while not done and step_count < max_steps:
            action = agent.choose_action(state)
            # env.visualize()  # Visualize after each action
            next_state, reward, done, _ = env.step(action)
            # env.render()  # Check honeypot deployment and attacker logs
            next_state = next_state.reshape(1, -1)

            if best_model_name == "sarsa":
                next_action = agent.choose_action(next_state)
                agent.train_step(state, action, reward, next_state, next_action, done)
            else:
                agent.train_step(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            step_count += 1

        episode_rewards.append(total_reward)
        print(f"{best_model_name} - Episode {episode + 1}/{max_episodes}, Total Reward: {total_reward}")

    # Plot the results
    plt.plot(episode_rewards, label=best_model_name)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title(f'Training Progress of {best_model_name}')
    plt.legend()
    plt.show()

# Main script
if __name__ == "__main__":
    best_model_name = select_best_model("temp/evaluation_results.csv")
    train_best_model(best_model_name, config)
