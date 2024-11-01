import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt

class HoneypotEnv(gym.Env):
    def __init__(self, num_nodes=5):
        super(HoneypotEnv, self).__init__()
        self.num_nodes = num_nodes
        self.action_space = spaces.Discrete(num_nodes)
        self.observation_space = spaces.Box(low=0, high=1, shape=(num_nodes,), dtype=np.float32)

        self.state = None
        self.honeypot_positions = np.zeros(num_nodes)
        self.vulnerabilities = np.random.rand(self.num_nodes)  # Random vulnerability levels

    def reset(self):
        self.state = np.random.rand(self.num_nodes)
        self.honeypot_positions = np.zeros(self.num_nodes)
        return self.state

    def simulate_attack(self):
        attack_node = np.random.choice(self.num_nodes)
        if self.honeypot_positions[attack_node] == 1:
            return 2.0
        else:
            return -0.5

    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action: {action}"

        if self.honeypot_positions[action] == 0:
            self.honeypot_positions[action] = 1
            reward = 1.0 + self.vulnerabilities[action]
        else:
            reward = -1.0

        reward += self.simulate_attack()
        done = np.all(self.honeypot_positions)
        self.state = np.random.rand(self.num_nodes) * (1 - self.honeypot_positions)
        
        return self.state, reward, done, {}

    def render(self, mode="human"):
        print(f"Honeypot positions: {self.honeypot_positions}")
        print(f"Current state: {self.state}")
        print(f"Node vulnerabilities: {self.vulnerabilities}")

    def visualize(self):
        plt.figure(figsize=(10, 2))
        for i in range(self.num_nodes):
            color = 'blue' if self.honeypot_positions[i] == 1 else 'red'
            plt.scatter(i, 1, s=100 * self.vulnerabilities[i], color=color, label=f"Node {i+1}")
            plt.text(i, 1.1, f"{self.vulnerabilities[i]:.2f}", ha='center', fontsize=9)

        plt.plot(range(self.num_nodes), [1] * self.num_nodes, 'k--', linewidth=0.5)
        plt.ylim(0.8, 1.2)
        plt.xticks(range(self.num_nodes), [f'Node {i+1}' for i in range(self.num_nodes)])
        plt.yticks([])
        plt.title("Honeypot Deployment Visualization")
        plt.legend(["Honeypot" if pos == 1 else "Node" for pos in self.honeypot_positions], loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=5)
        plt.show()
