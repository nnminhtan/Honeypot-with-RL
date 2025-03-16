import pandas as pd
import yaml
import time
import matplotlib.pyplot as plt
from testEnv import testEnv
from models.ddqn_model import DDQNAgent
from models.sarsa_model import SARSAAgent
from models.a2c_model import A2CAgent
import tensorflow as tf

# Load malicious IPs from the CSV file
def load_malicious_ips(csv_file):
    df = pd.read_csv(csv_file)
    malicious_ips = df[df['Anomaly'] == 1]['Src IP'].tolist()
    print(f"Loaded {len(malicious_ips)} malicious IPs.")
    return malicious_ips

# Load model configurations
with open("model_config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Select the best model based on previous evaluation results
def select_best_model(evaluation_csv):
    evaluation_results = pd.read_csv(evaluation_csv)

    # Compute average rewards
    avg_rewards = {
        "ddqn": evaluation_results["DDQN Reward"].mean(),
        "sarsa": evaluation_results["SARSA Reward"].mean(),
        "a2c": evaluation_results["A2C Reward"].mean()
    }
    
    best_model = max(avg_rewards, key=avg_rewards.get)  # Choose model with the highest average reward
    print(f"Best model selected based on evaluation: {best_model.upper()} (Avg Reward: {avg_rewards[best_model]:.2f})")
    
    return best_model


def train_best_model(best_model_name, config):
    malicious_ips = load_malicious_ips('ip_prediction_output.csv')
    env = testEnv(num_nodes=10, num_honeypots=4, honeypot_cost=0.5, malicious_ips=malicious_ips)

    # Wait until at least one honeypot is deployed
    while not env.active_honeypots:
        print("Waiting for intruder activity...")
        time.sleep(5)

    print(f"Intruder detected! Honeypots deployed on nodes: {env.active_honeypots}. Training begins.")

    agent_classes = {"ddqn": DDQNAgent, "sarsa": SARSAAgent, "a2c": A2CAgent}
    agent = agent_classes[best_model_name]((env.observation_space.shape[0],), env.action_space.n, config[best_model_name])

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
            next_state, reward, done, _ = env.step(action)
            next_state = next_state.reshape(1, -1)

            if best_model_name == "sarsa":
                next_action = agent.choose_action(next_state)
                agent.train_step(state, action, reward, next_state, next_action, done)
            else:
                agent.train_step(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            step_count += 1

            if step_count % 5 == 0:
                env.visualize()

        episode_rewards.append(total_reward)
        print(f"{best_model_name.upper()} - Episode {episode + 1}/{max_episodes}, Total Reward: {total_reward:.2f}")

    plt.plot(episode_rewards, label=f"{best_model_name.upper()} Training")
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title(f'Training Progress of {best_model_name.upper()}')
    plt.legend()
    plt.show()

# Main execution
if __name__ == "__main__":
    best_model_name = select_best_model("evaluation_results.csv")
    train_best_model(best_model_name, config)
