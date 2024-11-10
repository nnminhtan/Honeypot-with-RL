import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt

class HoneypotEnv(gym.Env):
    def __init__(self, num_nodes, num_honeypots):
        super(HoneypotEnv, self).__init__()
        self.num_nodes = num_nodes
        self.num_honeypots = num_honeypots
        self.action_space = spaces.Discrete(num_nodes)
        self.observation_space = spaces.Box(low=0, high=1, shape=(num_nodes,), dtype=np.float32)

        self.state = None
        self.honeypot_positions = np.zeros(num_nodes)
        self.vulnerabilities = np.random.rand(num_nodes)
        self.attacker_position = np.random.choice(num_nodes)  # Random initial attacker position

    def reset(self):
        self.state = np.random.rand(self.num_nodes)
        self.honeypot_positions = np.zeros(self.num_nodes)
        self.attacker_position = np.random.choice(self.num_nodes)

        self.cumulative_reward = 0  # Reset cumulative reward for the episode if tracked
        return self.state

    def place_honeypot(self, action):
        """Places a honeypot only if we haven’t exceeded the number limit."""
        if np.sum(self.honeypot_positions) < self.num_honeypots and self.honeypot_positions[action] == 0:
            self.honeypot_positions[action] = 1
            return 1.0 + self.vulnerabilities[action]  # Positive reward for strategic placement
        else:
            return -1.0  # Penalty for exceeding honeypot limits or redundant placement

    def simulate_attack(self):
        # self.attacker_position = np.random.choice(self.num_nodes)  # Attacker moves randomly each step
        vulnerable_nodes = np.where(self.honeypot_positions == 0)[0]
        if len(vulnerable_nodes) > 0:
            self.attacker_position = vulnerable_nodes[np.argmax(self.vulnerabilities[vulnerable_nodes])]
        else:
            self.attacker_position = np.random.choice(self.num_nodes)  # Random fallback


        if self.honeypot_positions[self.attacker_position] == 1:
            return 2.0  # High reward for catching attacker
        else:
            return -1.0  # Penalty if attacker reaches an unprotected node

    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action: {action}"

        # Place a honeypot and get reward
        placement_reward = self.place_honeypot(action)

        # Simulate an attack and get the reward
        attack_reward = self.simulate_attack()

        reward = placement_reward + attack_reward
        done = np.sum(self.honeypot_positions) >= self.num_honeypots
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
