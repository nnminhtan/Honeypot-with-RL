import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt

class HoneypotEnv(gym.Env):
    def __init__(self, num_nodes, num_honeypots, honeypot_cost, malicious_ips, placement_interval=5):
        super(HoneypotEnv, self).__init__()
        self.num_nodes = num_nodes
        self.num_honeypots = num_honeypots
        self.honeypot_cost = honeypot_cost
        self.malicious_ips = malicious_ips
        self.placement_interval = placement_interval  # Steps between allowed honeypot placements
        
        self.action_space = spaces.Discrete(num_nodes)
        self.observation_space = spaces.Box(low=0, high=1, shape=(num_nodes,), dtype=np.float32)

        self.state = None
        self.honeypot_positions = np.zeros(num_nodes)
        self.vulnerabilities = np.random.rand(num_nodes)
        self.attacker_position = np.random.choice(num_nodes)
        self.placed_honeypots = 0
        self.step_count = 0

    def reset(self):
        self.state = np.random.rand(self.num_nodes)
        self.honeypot_positions = np.zeros(self.num_nodes)
        self.vulnerabilities = np.random.rand(self.num_nodes)
        self.attacker_position = np.random.choice(self.num_nodes)
        self.placed_honeypots = 0
        self.step_count = 0
        return self.state

    def place_honeypot(self, action):
        if self.placed_honeypots < self.num_honeypots and self.honeypot_positions[action] == 0:
            self.honeypot_positions[action] = 1
            self.placed_honeypots += 1
            return 0.5  # Small positive reward for successful placement
        else:
            return 0  # No penalty for redundant placement during early exploration

    def simulate_attack(self):
        # Randomly move attacker
        self.attacker_position = np.random.choice(self.num_nodes)
        attacker_ip = f"SrcIP{self.attacker_position + 1}"
        is_malicious = attacker_ip in self.malicious_ips

        # Reward/penalty based on interception or evasion
        if self.honeypot_positions[self.attacker_position] == 1:
            return 5.0 if is_malicious else 0.5  # Reward interception of any attacker initially
        else:
            return 0  # No penalty for early exploration of evasion

    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action: {action}"

        # Place honeypot only at intervals
        placement_reward = 0
        if self.step_count % self.placement_interval == 0:
            placement_reward = self.place_honeypot(action)

        # Attack simulation and reward
        attack_reward = self.simulate_attack()
        reward = placement_reward + attack_reward

        # Update state and step count
        self.state = np.random.rand(self.num_nodes) * (1 - self.honeypot_positions)
        self.step_count += 1
        done = self.step_count >= self.num_honeypots * self.placement_interval

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

        plt.scatter(self.attacker_position, 1, color='black', marker='x', s=200, label="Attacker")
        plt.plot(range(self.num_nodes), [1] * self.num_nodes, 'k--', linewidth=0.5)
        plt.ylim(0.8, 1.2)
        plt.xticks(range(self.num_nodes), [f'Node {i+1}' for i in range(self.num_nodes)])
        plt.yticks([])
        plt.title("Honeypot Deployment Visualization")
        plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=5)
        plt.show()
