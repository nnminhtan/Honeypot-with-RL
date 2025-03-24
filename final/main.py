import pandas as pd
import yaml
import time
import matplotlib.pyplot as plt
from testEnv import testEnv
from models.ddqn_model import DDQNAgent
from models.sarsa_model import SARSAAgent
from models.a2c_model import A2CAgent
import subprocess
import logging
import json

logging.basicConfig(level=logging.INFO)

def get_container_name():
    try:
        # Get the container ID (this is the hostname inside the container)
        result = subprocess.run(["cat", "/etc/hostname"], capture_output=True, text=True)
        container_id = result.stdout.strip()

        # Use Docker API to get the actual container name
        result = subprocess.run(["docker", "ps", "--format", "{{json .}}"], capture_output=True, text=True)
        containers = [json.loads(line) for line in result.stdout.splitlines()]

        for container in containers:
            if container["ID"].startswith(container_id):  # Match with hostname ID
                return container["Names"]  # Get the actual name like 'node_1'

    except Exception as e:
        print(f"Error getting container name: {e}")
        return None

# Load malicious IPs from the CSV file
def load_malicious_ips(csv_file):
    df = pd.read_csv(csv_file)
    malicious_ips = df[df['Anomaly'] == 1]['Src IP'].tolist()
    print(f"Loaded {len(malicious_ips)} malicious IPs.")
    return malicious_ips

# Create Docker network and deploy nodes
def create_docker_nodes(num_nodes):
    node_ips = {}

    for i in range(num_nodes):
        container_name = f"node_{i+1}"
        
        # Start container
        subprocess.run(["docker", "run", "-d", "--network", "honeypot_network",
                        "--name", container_name, "honeypot-ssh-image"])

        # Get container IP address
        result = subprocess.run(["docker", "inspect", "-f",
                                 "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}",
                                 container_name], capture_output=True, text=True)
        ip_address = result.stdout.strip()
        node_ips[i] = ip_address

    logging.info(f"Nodes initialized with IPs: {node_ips}")
    return node_ips

# def create_docker_network(self):
#     node_ips = {}
#     for i in range(self.num_nodes):
#         container_name = f"node_{i+1}"
#         subprocess.run(["docker", "run", "-d", "--network", "honeypot_network",
#                         "--name", container_name, "honeypot-ssh-image"])

#         result = subprocess.run(["docker", "inspect", "-f",
#                                     "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}",
#                                     container_name], capture_output=True, text=True)
#         ip_address = result.stdout.strip()
#         node_ips[i] = ip_address

#     logging.info(f"Nodes initialized with IPs: {node_ips}")
#     return node_ips

# Execute honeypot script inside the corresponding container
def execute_honeypot_script(node_id):
    container_name = f"node_{node_id+1}"

    try:
        logging.info(f"Executing LLMhoneypot.py inside {container_name}...")
        subprocess.Popen(
            ["docker", "exec", "-it", container_name, "python", "LLMhoneypot.py"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        logging.info(f"Successfully started LLMHoneypot.py in {container_name}.")
    except Exception as e:
        logging.error(f"Failed to execute LLMHoneypot.py in {container_name}: {e}")

# Select the best model based on evaluation results
def select_best_model(evaluation_csv):
    evaluation_results = pd.read_csv(evaluation_csv)
    avg_rewards = {
        "ddqn": evaluation_results["DDQN Reward"].mean(),
        "sarsa": evaluation_results["SARSA Reward"].mean(),
        "a2c": evaluation_results["A2C Reward"].mean()
    }
    best_model = max(avg_rewards, key=avg_rewards.get)
    print(f"Best model selected: {best_model.upper()} (Avg Reward: {avg_rewards[best_model]:.2f})")
    return best_model

# Train the best model
def train_best_model(best_model_name, config):
    malicious_ips = load_malicious_ips('ip_prediction_output.csv')
    
    container_name = get_container_name()
    
    # Stop execution if running inside a child container
    if container_name and container_name.startswith("node_"):
        print(f"Detected child container {container_name}. Exiting main.py.")
        exit(0)

    print("Running in the parent honeypot-container.")

    # Create the nodes and get their IP addresses
    num_nodes = 10
    node_ips = create_docker_nodes(num_nodes)

    # Pass node IPs to the test environment
    env = testEnv(num_nodes=num_nodes, num_honeypots=4, honeypot_cost=0.5, malicious_ips=malicious_ips, node_ips=node_ips)

    print("Waiting for honeypot placement...")
    while env.active_honeypots == 0:
        env.step(0)
        time.sleep(1)

    print(f"Honeypots placed: {env.active_honeypots}")

    # Run LLMHoneypot.py in placed honeypot nodes
    for node_id in env.active_honeypots:
        execute_honeypot_script(node_id)

    print("Monitoring for intruder activity...")

    # Wait until an attacker interacts with a honeypot
    while env.attacker_position not in env.active_honeypots:
        time.sleep(5)

    print(f"Intruder detected at Node {env.attacker_position}! Training begins.")

    agent_classes = {"ddqn": DDQNAgent, "sarsa": SARSAAgent, "a2c": A2CAgent}
    agent = agent_classes[best_model_name]((env.observation_space.shape[0],), env.action_space.n, config[best_model_name])

    max_episodes = 1
    max_steps = 100
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
    with open("model_config.yaml", "r") as file:
        config = yaml.safe_load(file)

    best_model_name = select_best_model("evaluation_results.csv")
    train_best_model(best_model_name, config)
