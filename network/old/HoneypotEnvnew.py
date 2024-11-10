import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class HoneypotEnv(gym.Env):
    def __init__(self, num_nodes, num_honeypots, honeypot_cost, malicious_ips, placement_interval=5):
        super(HoneypotEnv, self).__init__()
        self.num_nodes = num_nodes
        self.num_honeypots = num_honeypots
        self.honeypot_cost = honeypot_cost
        self.malicious_ips = malicious_ips  # List of malicious IPs to match during attacks
        self.placement_interval = placement_interval
        
        # Initialize action and observation spaces
        self.action_space = spaces.Discrete(num_nodes)
        self.observation_space = spaces.Box(low=0, high=1, shape=(num_nodes,), dtype=np.float32)

        self.state = None
        self.honeypot_positions = np.zeros(num_nodes)
        self.vulnerabilities = np.random.rand(num_nodes)
        self.attacker_position = np.random.choice(num_nodes)
        self.attacker_ip = None
        self.placed_honeypots = 0
        self.step_count = 0
        self.activated_traps = np.zeros(num_nodes)  # Keep track of activated honeypots
        self.malicious_index = 0  # Keep track of which malicious IP to use

    def reset(self):
        """Resets the environment to its initial state."""
        self.state = np.random.rand(self.num_nodes)
        self.honeypot_positions = np.zeros(self.num_nodes)
        self.activated_traps = np.zeros(self.num_nodes)  # Reset activated traps
        self.vulnerabilities = np.random.rand(self.num_nodes)
        self.attacker_position = np.random.choice(self.num_nodes)
        
        # Get the attacker IP from the malicious IP list based on the current index
        self.attacker_ip = self.malicious_ips[self.malicious_index]
        
        self.placed_honeypots = 0
        self.step_count = 0
        return self.state

    def place_honeypot(self, action):
        """Places a honeypot at the specified node if space is available."""
        if self.placed_honeypots < self.num_honeypots and self.honeypot_positions[action] == 0:
            self.honeypot_positions[action] = 1
            self.placed_honeypots += 1
            return 0.5  # Small reward for placing a honeypot
        else:
            return 0  # No reward if placement exceeds limit or is redundant

    def simulate_attack(self):
        """Simulates an attack and checks if the attacker hits a honeypot."""
        # Randomly choose an attacker position and generate their IP
        self.attacker_position = np.random.choice(self.num_nodes)
        print(f"Attacker IP at node {self.attacker_position}: {self.attacker_ip}")  # Debug: Print attacker IP

        # Check if the attacker IP is in the list of malicious IPs
        is_malicious = self.attacker_ip in self.malicious_ips  # Check if the IP is malicious
        print(f"Is the attacker IP malicious? {is_malicious}")  # Debug: Check malicious status

        # Check for interception and assign rewards
        if self.honeypot_positions[self.attacker_position] == 1:
            # Activate the honeypot trap if it intercepts the attacker
            if is_malicious:
                self.activated_traps[self.attacker_position] = 1  # Mark the honeypot as activated
                print(f"Malicious IP detected and trapped at node {self.attacker_position}")
                return 5.0  # High reward for intercepting a malicious attacker
            else:
                print(f"Non-malicious IP at node {self.attacker_position}")
                return 0.5  # Lower reward for intercepting non-malicious IP
        else:
            return 0  # No reward if the attacker is not intercepted

    def step(self, action):
        """Performs an action in the environment (placing honeypots and handling attacks)."""
        assert self.action_space.contains(action), f"Invalid action: {action}"

        # Place honeypot at intervals and calculate placement reward
        placement_reward = 0
        if self.step_count % self.placement_interval == 0:
            placement_reward = self.place_honeypot(action)

        # Simulate an attack and calculate attack reward
        attack_reward = self.simulate_attack()
        reward = placement_reward + attack_reward

        # Update state and increment step count
        self.state = np.random.rand(self.num_nodes) * (1 - self.honeypot_positions)
        self.step_count += 1

        # Move to the next malicious IP after each step
        self.malicious_index = (self.malicious_index + 1) % len(self.malicious_ips)

        done = self.step_count >= self.num_honeypots * self.placement_interval
        return self.state, reward, done, {}

    def render(self, mode="human"):
        """Renders the current state of the environment."""
        print(f"Honeypot positions: {self.honeypot_positions}")
        print(f"Activated traps: {self.activated_traps}")  # Show which traps have been activated
        print(f"Attacker position: {self.attacker_position}")
        print(f"Node vulnerabilities: {self.vulnerabilities}")

    def visualize(self):
        """Visualizes the honeypot deployment and attack positions."""
        plt.figure(figsize=(10, 2))
        for i in range(self.num_nodes):
            color = 'blue' if self.honeypot_positions[i] == 1 else 'red'
            size = 100 * self.vulnerabilities[i]
            # If the trap is activated, color it differently (e.g., green for activated)
            if self.activated_traps[i] == 1:
                color = 'green'
            plt.scatter(i, 1, s=size, color=color)
            plt.text(i, 1.1, f"{self.vulnerabilities[i]:.2f}", ha='center', fontsize=9)

        plt.scatter(self.attacker_position, 1, color='black', marker='x', s=200, label="Attacker")

        # Show the attacker's IP next to the node they are at
        plt.text(self.attacker_position, 1.15, f"IP: {self.attacker_ip}", ha='center', fontsize=9, color='black')

        plt.plot(range(self.num_nodes), [1] * self.num_nodes, 'k--', linewidth=0.5)
        plt.ylim(0.8, 1.2)
        plt.xticks(range(self.num_nodes), [f'Node {i+1}' for i in range(self.num_nodes)])
        plt.yticks([])
        plt.title("Honeypot Deployment Visualization")
        plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=5)
        plt.show()

# Load the malicious IPs from the CSV file
def load_malicious_ips(csv_file):
    df = pd.read_csv(csv_file)
    # Extract the 'Src IP' and 'Anomaly' columns where Anomaly == 1
    malicious_ips = df[df['Anomaly'] == 1]['Src IP'].tolist()
    print(f"Loaded malicious IPs: {malicious_ips}")  # Debug: Print loaded malicious IPs
    return malicious_ips

# Example usage:
malicious_ips = load_malicious_ips('ip_prediction_output.csv')
env = HoneypotEnv(num_nodes=10, num_honeypots=4, honeypot_cost=0.5, malicious_ips=malicious_ips)
