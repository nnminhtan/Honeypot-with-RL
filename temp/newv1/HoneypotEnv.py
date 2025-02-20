import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt

class HoneypotEnv(gym.Env):
    def __init__(self, num_nodes, num_honeypots, honeypot_cost, malicious_ips):
        super(HoneypotEnv, self).__init__()
        self.num_nodes = num_nodes
        self.num_honeypots = num_honeypots
        self.honeypot_cost = honeypot_cost  # Cost per honeypot placement
        self.malicious_ips = malicious_ips  # List of malicious IP addresses (from the CSV)
        self.action_space = spaces.Discrete(num_nodes)
        self.observation_space = spaces.Box(low=0, high=1, shape=(num_nodes,), dtype=np.float32)

        self.state = None
        self.honeypot_positions = np.zeros(num_nodes)
        self.vulnerabilities = np.random.rand(num_nodes)
        self.attacker_position = np.random.choice(num_nodes)  # Initial attacker position

    def reset(self):
        self.state = np.random.rand(self.num_nodes)
        self.honeypot_positions = np.zeros(self.num_nodes)
        self.vulnerabilities = np.random.rand(self.num_nodes)
        self.attacker_position = np.random.choice(self.num_nodes)
        return self.state

    def place_honeypot(self, action):
        """Place a honeypot with a cost, if we haven't reached the honeypot limit."""
        if np.sum(self.honeypot_positions) < self.num_honeypots and self.honeypot_positions[action] == 0:
            self.honeypot_positions[action] = 1
            placement_reward = 1.0 + self.vulnerabilities[action]  # Reward for strategic placement
            return placement_reward - self.honeypot_cost  # Apply placement cost
        else:
            return self.honeypot_cost  # Penalty if exceeding honeypot limits or redundant placement

    def simulate_attack(self):
        """Move the attacker to a new node each step and check for interception."""
        self.attacker_position = np.random.choice(self.num_nodes)
        
        # Check if the attacker's IP is malicious (based on provided malicious_ips)
        # Simulate mapping each node position to an IP address, here assuming nodes are mapped to IPs like "SrcIP1", "SrcIP2", etc.
        attacker_ip = f"Src IP{self.attacker_position + 1}"  # Example: mapping node to an IP (adjust logic as needed)
        is_malicious = attacker_ip in self.malicious_ips
        
        if self.honeypot_positions[self.attacker_position] == 1:  # Honeypot intercepts attacker
            if is_malicious:
                return 5.0  # High reward for intercepting malicious traffic
            else:
                return -1.0  # Penalty for intercepting normal traffic
        else:
            return -1.0  # Penalty if the attacker reaches an unprotected node

    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action: {action}"

        # Place a honeypot with associated cost and get reward
        placement_reward = self.place_honeypot(action)

        # Simulate an attack and get the reward
        attack_reward = self.simulate_attack()

        # Total reward combines placement and attack outcomes
        reward = placement_reward + attack_reward
        done = np.sum(self.honeypot_positions) >= self.num_honeypots

        # Update the state (e.g., vulnerabilities or honeypot positions)
        self.state = np.random.rand(self.num_nodes) * (1 - self.honeypot_positions)

        return self.state, reward, done, {}

    def render(self, mode="human"):
        print(f"Honeypot positions: {self.honeypot_positions}")
        print(f"Attacker position: {self.attacker_position}")
        print(f"Node vulnerabilities: {self.vulnerabilities}")

    def visualize(self):
        plt.figure(figsize=(10, 2))
        for i in range(self.num_nodes):
            color = 'blue' if self.honeypot_positions[i] == 1 else 'red'
            plt.scatter(i, 1, s=100 * self.vulnerabilities[i], color=color)
            plt.text(i, 1.1, f"{self.vulnerabilities[i]:.2f}", ha='center', fontsize=9)

        # Mark attacker position
        plt.scatter(self.attacker_position, 1, color='black', marker='x', s=200, label="Attacker")

        plt.plot(range(self.num_nodes), [1] * self.num_nodes, 'k--', linewidth=0.5)
        plt.ylim(0.8, 1.2)
        plt.xticks(range(self.num_nodes), [f'Node {i+1}' for i in range(self.num_nodes)])
        plt.yticks([])
        plt.title("Honeypot Deployment Visualization")
        plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=5)
        plt.show()
