import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import logging
import subprocess  # To run bash scripts
import sys
import argparse
import yaml
import os
# sys.path.append('/content/aipot')  # Add the directory containing 'llmhoneypot.py' to the system path
# import LLMhoneypot as LLMHoneypot
import LLMhoneypot  # Import the LLMhoneypot.py script (without the .py extension)
# from LLMhoneypot import read_env

from dotenv import load_dotenv
# load_dotenv('/content/.env')

class testEnv(gym.Env):
    def __init__(self, num_nodes, num_honeypots, honeypot_cost, malicious_ips, placement_interval=5):
        super(testEnv, self).__init__()
        self.num_nodes = num_nodes
        self.num_honeypots = num_honeypots
        self.honeypot_cost = honeypot_cost
        self.malicious_ips = malicious_ips
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
        self.activated_traps = np.zeros(num_nodes)  # Track activated honeypots
        self.malicious_index = 0

        # Create the logging directory if it doesn't exist
        log_dir = 'logs'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        log_file = os.path.join(log_dir, 'honeypot_log.txt')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        logging.info("HoneypotEnv initialized.")


    def reset(self):
        self.state = np.random.rand(self.num_nodes)
        self.honeypot_positions = np.zeros(self.num_nodes)
        self.activated_traps = np.zeros(self.num_nodes)
        self.vulnerabilities = np.random.rand(self.num_nodes)
        self.attacker_position = np.random.choice(self.num_nodes)
        self.attacker_ip = self.malicious_ips[self.malicious_index]
        self.placed_honeypots = 0
        self.step_count = 0
        return self.state

    def place_honeypot(self, action):
        if self.placed_honeypots < self.num_honeypots and self.honeypot_positions[action] == 0:
            self.honeypot_positions[action] = 1
            self.placed_honeypots += 1
            return 0.5  # Return honeypot placement cost as reward
        return 0

    def simulate_attack(self):
        self.attacker_position = np.random.choice(self.num_nodes)
        is_malicious = self.attacker_ip in self.malicious_ips

        if self.honeypot_positions[self.attacker_position] == 1:
            self.activated_traps[self.attacker_position] = 1
            try:
                print("Current working directory:", os.getcwd())
                # Call the LLMhoneypot.main() function when a honeypot is placed
                if self.placed_honeypots > 0:
                    LLMhoneypot.main()  # Call LLMhoneypot only after a honeypot is placed
                else:
                    print("No honeypot placed yet, skipping honeypot script.")

            except Exception as e:
                print(f"Error while executing honeypot script: {e}")
                return 0  # Return a default penalty for error handling

            return 5.0 if is_malicious else 0.5
        return 0

    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action: {action}"

        placement_reward = 0
        if self.step_count % self.placement_interval == 0:
            placement_reward = self.place_honeypot(action)

        attack_reward = self.simulate_attack()
        reward = placement_reward + attack_reward

        self.state = np.random.rand(self.num_nodes) * (1 - self.honeypot_positions)
        self.step_count += 1

        self.malicious_index = (self.malicious_index + 1) % len(self.malicious_ips)
        done = self.step_count >= self.num_honeypots * self.placement_interval
        return self.state, reward, done, {}

    def render(self, mode="human"):
        print(f"Honeypot positions: {self.honeypot_positions}")
        print(f"Activated traps: {self.activated_traps}")
        print(f"Attacker position: {self.attacker_position}")
        print(f"Node vulnerabilities: {self.vulnerabilities}")

    def visualize(self):
        plt.figure(figsize=(10, 2))
        for i in range(self.num_nodes):
            # Use green for nodes with honeypots, red for others
            color = 'green' if self.honeypot_positions[i] == 1 else 'red'
            # Draw the nodes with varying sizes based on vulnerability
            plt.scatter(i, 1, s=100 * self.vulnerabilities[i], color=color)
            # Display the vulnerability value above each node
            plt.text(i, 1.1, f"{self.vulnerabilities[i]:.2f}", ha='center', fontsize=9)

        # Mark attacker position with a black X
        plt.scatter(self.attacker_position, 1, color='black', marker='x', s=200, label="Attacker")
        # Display the attacker's IP above the black X marker
        plt.text(self.attacker_position, 1.2, f"{self.attacker_ip}", ha='center', fontsize=9, color='black')

        # Draw a dashed line to represent the nodes' position
        plt.plot(range(self.num_nodes), [1] * self.num_nodes, 'k--', linewidth=0.5)
        plt.ylim(0.8, 1.2)
        plt.xticks(range(self.num_nodes), [f'Node {i+1}' for i in range(self.num_nodes)])
        plt.yticks([])
        plt.title("Honeypot Deployment Visualization")
        plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=5)
        plt.show()
