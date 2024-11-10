import gym
from gym import spaces
import numpy as np

class HoneypotEnv(gym.Env):
    def __init__(self, num_nodes=5):
        super(HoneypotEnv, self).__init__()
        self.num_nodes = num_nodes
        self.action_space = spaces.Discrete(num_nodes)  # Actions: selecting a node to deploy a honeypot
        self.observation_space = spaces.Box(low=0, high=1, shape=(num_nodes,), dtype=np.float32)

        # Initialize environment-specific parameters
        self.state = None
        self.honeypot_positions = np.zeros(num_nodes)  # Track honeypot locations, initially all empty

    def reset(self):
        # Reset environment to the initial state
        self.state = np.random.rand(self.num_nodes)  # Randomized state to simulate node status
        self.honeypot_positions = np.zeros(self.num_nodes)  # Clear previous honeypot placements
        return self.state

    def step(self, action):
        # Apply an action (placing a honeypot at a node)
        assert self.action_space.contains(action), f"Invalid action: {action}"

        # Simulate reward: higher reward for unique/unoccupied honeypot placement
        if self.honeypot_positions[action] == 0:  # If no honeypot is placed here yet
            self.honeypot_positions[action] = 1  # Place honeypot
            reward = 1.0  # Reward for a successful deployment
        else:
            reward = -1.0  # Penalty for redeployment at the same node

        # Check if all honeypots have been placed as a done condition
        done = np.all(self.honeypot_positions)

        # Update state to reflect new status after the action
        self.state = np.random.rand(self.num_nodes) * (1 - self.honeypot_positions)

        return self.state, reward, done, {}

    def render(self, mode="human"):
        # Optional: Print environment status or other visualization
        print(f"Current honeypot placements: {self.honeypot_positions}")
        print(f"Current state: {self.state}")
