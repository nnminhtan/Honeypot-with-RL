import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt

class HoneypotEnv(gym.Env):
    def __init__(self, num_nodes):
        super(HoneypotEnv, self).__init__()
        self.num_nodes = num_nodes
        self.action_space = spaces.Discrete(num_nodes)  # Actions: place a honeypot on any node
        self.observation_space = spaces.Box(low=0, high=1, shape=(num_nodes,), dtype=np.float32)

        # State components
        self.state = None
        self.honeypot_positions = np.zeros(num_nodes)
        self.vulnerabilities = np.random.rand(self.num_nodes)  # Random vulnerability levels
        self.last_attack_node = None  # Track the last attack target for visualization

    def _attack_probabilities(self):
        """Generate probabilities for each node being attacked based on vulnerabilities."""
        return self.vulnerabilities / self.vulnerabilities.sum()

    def reset(self):
        self.state = np.random.rand(self.num_nodes)  # Initial random vulnerabilities
        self.honeypot_positions = np.zeros(self.num_nodes)  # Reset honeypot positions
        self.last_attack_node = None
        return self.state

    def simulate_attack(self):
        """Simulate an attack on a node based on attack probabilities."""
        self.last_attack_node = np.random.choice(self.num_nodes, p=self._attack_probabilities())
        return self.last_attack_node, 2.0 if self.honeypot_positions[self.last_attack_node] == 1 else -0.5

    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action: {action}"

        # Place honeypot if none exists in the selected action node
        if self.honeypot_positions[action] == 0:
            self.honeypot_positions[action] = 1
            reward = 1.0 + self.vulnerabilities[action]  # Higher reward for placing on more vulnerable nodes
        else:
            reward = -1.0  # Penalty for redundant honeypot placement

        # Simulate attack and calculate reward based on whether honeypot intercepts it
        attack_node, attack_reward = self.simulate_attack()
        reward += attack_reward  # Add reward if honeypot intercepts attack

        # Check if all honeypots are placed
        done = np.all(self.honeypot_positions)
        
        # Update state, representing remaining vulnerabilities (ignoring nodes with honeypots)
        self.state = np.random.rand(self.num_nodes) * (1 - self.honeypot_positions)
        
        return self.state, reward, done, {"attack_node": attack_node, "intercepted": self.honeypot_positions[attack_node] == 1}

    def render(self, mode="human"):
        print(f"Honeypot positions: {self.honeypot_positions}")
        print(f"Current state: {self.state}")
        print(f"Node vulnerabilities: {self.vulnerabilities}")
        if self.last_attack_node is not None:
            print(f"Last attack targeted node: {self.last_attack_node}")

    def visualize(self):
        plt.figure(figsize=(8, 8))
        angles = np.linspace(0, 2 * np.pi, self.num_nodes, endpoint=False).tolist()
        angles += angles[:1]  # Repeat the first angle to close the circle

        # Create node positions
        node_positions = [(np.cos(angle), np.sin(angle)) for angle in angles[:-1]]

        for i, (x, y) in enumerate(node_positions):
            if i == self.last_attack_node:
                color = 'purple'  # Color for the attacked node
                marker = 'X'       # Different marker for attacked node
                label = f"Attacked Node {i+1}"
            elif self.honeypot_positions[i] == 1:
                color = 'blue'  # Color for honeypot nodes
                marker = 'o'
                label = f"Honeypot {i+1}"
            else:
                color = 'red'   # Color for regular nodes
                marker = 'o'
                label = f"Node {i+1}"
            
            plt.scatter(x, y, s=300 * self.vulnerabilities[i], color=color, marker=marker, label=label)
            plt.text(x, y + 0.1, f"{self.vulnerabilities[i]:.2f}", ha='center', fontsize=9)

        # Draw edges (assuming a fully connected network for simplicity)
        for i, (x_i, y_i) in enumerate(node_positions):
            for j, (x_j, y_j) in enumerate(node_positions):
                if i < j:  # Only one direction to avoid duplicate edges
                    plt.plot([x_i, x_j], [y_i, y_j], 'k--', linewidth=0.5)

        plt.title("Honeypot Deployment Visualization")
        plt.axis('off')
        plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=5)
        plt.show()
