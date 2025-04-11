import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt

class HoneypotEnv(gym.Env):
    def __init__(self, num_nodes, num_honeypots, honeypot_cost, malicious_ips, max_steps):
        super(HoneypotEnv, self).__init__()
        self.num_nodes = num_nodes
        self.num_honeypots = num_honeypots
        self.honeypot_cost = honeypot_cost  # Cost per honeypot placement
        self.malicious_ips = malicious_ips  # List of malicious IP addresses (from the CSV)
        self.action_space = spaces.Discrete(num_nodes)
        self.observation_space = spaces.Box(low=0, high=1, shape=(num_nodes,), dtype=np.float32)
        self.max_steps = max_steps  # <-- Add this
        self.current_step = 0
        self.state = None
        self.honeypot_positions = np.zeros(num_nodes)
        self.vulnerabilities = np.random.rand(num_nodes)
        self.attacker_position = np.random.choice(num_nodes)  # Initial attacker position

    def reset(self):
        self.state = np.random.rand(self.num_nodes)
        self.honeypot_positions = np.zeros(self.num_nodes)
        self.vulnerabilities = np.random.rand(self.num_nodes)
        self.attacker_position = np.random.choice(self.num_nodes)
        self.current_step = 0  # <-- Reset step count here
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
        """Simulates an attacker choosing a target node based on attack strategy (broad or targeted)."""

        # Randomly choose attack type (70% broad, 30% targeted for example)
        # attack_type = np.random.choice(["broad", "targeted"], p=[0.7, 0.3])
        attack_type = "targeted"
        if attack_type == "broad":
            # Broad attack: randomly choose a node
            self.attacker_position = np.random.choice(self.num_nodes)

        elif attack_type == "targeted":
            # Targeted attack: choose from high vulnerability nodes (e.g., top 30%)
            threshold = np.percentile(self.vulnerabilities, 70)
            vulnerable_nodes = [i for i, v in enumerate(self.vulnerabilities) if v >= threshold]
            
            if vulnerable_nodes:
                self.attacker_position = np.random.choice(vulnerable_nodes)
            else:
                # Fallback to random if no vulnerable nodes found
                self.attacker_position = np.random.choice(self.num_nodes)

        # Map attacker position to an IP for malicious IP check
        attacker_ip = f"Src IP{self.attacker_position + 1}"
        is_malicious = attacker_ip in self.malicious_ips

        # Honeypot interception logic
        if self.honeypot_positions[self.attacker_position] == 1:
            if is_malicious:
                return 5.0  # reward for catching malicious
            else:
                return -1.0  # penalty for intercepting normal
        else:
            return -1.0  # attacker succeeded, nothing caught

    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action: {action}"

        reward = 0.0

        # --- Step 1: Try placing a honeypot ---
        placement_reward = self.place_honeypot(action)
        reward += placement_reward  # usually ~0.5 for good placement

        # --- Step 2: Simulate attacker behavior ---
        attack_reward = self.simulate_attack()

        # Check if attacker is on a honeypot
        is_honeypot = self.honeypot_positions[self.attacker_position] == 1
        is_malicious = f"Src IP{self.attacker_position + 1}" in self.malicious_ips

        # --- Step 3: Reward for interception or penalty ---
        if is_honeypot and is_malicious:
            reward += 10.0  # big reward for catching malicious
        elif is_honeypot and not is_malicious:
            reward -= 0.5   # small penalty for trapping normal traffic
        elif not is_honeypot and np.sum(self.honeypot_positions) >= self.num_honeypots:
            reward += 0.2   # small survival bonus for avoiding trap

        # --- Step 4: Prevent over-penalizing missed attacks ---
        if np.sum(self.honeypot_positions) >= self.num_honeypots and attack_reward == -1.0:
            attack_reward = 0.0  # no penalty if agent is done placing
        reward += attack_reward

        # --- Step 5: Done conditions ---
        self.current_step += 1
        done = False

        # Early success: end if malicious attacker is caught
        if is_honeypot and is_malicious:
            done = True

        # End if max steps reached
        if self.current_step >= self.max_steps:
            done = True

        # --- Step 6: Update state (ignore nodes with honeypots) ---
        self.state = np.random.rand(self.num_nodes) * (1 - self.honeypot_positions)
        return self.state.reshape(1, -1), reward, done, {}


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
