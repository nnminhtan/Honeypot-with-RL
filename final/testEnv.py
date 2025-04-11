import gym
from gym import spaces
import numpy as np
import logging
import subprocess
import os
import paramiko
import time
import threading

class testEnv(gym.Env):
    def __init__(self, num_nodes, num_honeypots, honeypot_cost, malicious_ips, node_ips, placement_interval=5):
        super(testEnv, self).__init__()
        self.num_nodes = num_nodes
        self.num_honeypots = num_honeypots
        self.honeypot_cost = honeypot_cost
        self.malicious_ips = malicious_ips
        self.placement_interval = placement_interval
        self.node_ips = node_ips 

        self.action_space = spaces.Discrete(num_nodes)
        self.observation_space = spaces.Box(low=0, high=1, shape=(num_nodes,), dtype=np.float32)

        self.state = None
        self.honeypot_positions = np.zeros(num_nodes)
        self.attacker_position = np.random.choice(num_nodes)
        self.placed_honeypots = 0
        self.step_count = 0
        self.active_honeypots = 0
        self.ssh_clients = {}
        self.attack_detected = np.zeros(self.num_nodes, dtype=bool)
        
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, 'honeypot_log.txt')),
                logging.StreamHandler()
            ]
        )
        logging.info("HoneypotEnv initialized.")

        self.setup_ssh_connections()

        # **Start SSH Attack Simulation in a Separate Thread**
        attack_thread = threading.Thread(target=self.simulate_attack, daemon=True)
        attack_thread.start()

        # **Start SSH Login Monitoring in a Separate Thread**
        ssh_monitor_thread = threading.Thread(target=self.monitor_ssh_logins, daemon=True)
        ssh_monitor_thread.start()

    def setup_ssh_connections(self):
        for node_id, ip in self.node_ips.items():
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            try:
                ssh.connect(ip, username="honeypotuser", password="honeypotpass")
                self.ssh_clients[node_id] = ssh
                logging.info(f"Connected to Node {node_id} at {ip} via SSH.")
            except Exception as e:
                logging.error(f"Failed to SSH into Node {node_id} ({ip}): {e}")

    def monitor_ssh_logins(self):
        """Continuously checks for active SSH logins on each node."""
        while True:
            for node_id, ip in self.node_ips.items():
                try:
                    result = subprocess.run(
                        ["ssh", "-o", "StrictHostKeyChecking=no", ip, "who"],
                        capture_output=True, text=True, timeout=5
                    )
                    if result.stdout.strip():
                        logging.info(f"SSH detected on {ip} (Node {node_id}):\n{result.stdout.strip()}")
                        print(f"SSH detected on {ip} (Node {node_id}):\n{result.stdout.strip()}")
                except Exception as e:
                    # logging.error(f"Error checking SSH logins on {ip}: {str(e)}")
                    logging.error("")

            time.sleep(10)

    def simulate_attack(self):
        """ Simulates an SSH attack on a random node every 5 seconds. """
        logging.info("[*] Starting simulated SSH attack in the background...")
    
        while True:
            node_id = np.random.choice(list(self.node_ips.keys()))  # ✅ Pick a random attacker target
            self.attacker_position = node_id  # ✅ Update the attacker's position
            ip = self.node_ips[node_id]
    
            fake_user = "honeypotuser"
            fake_pass = np.random.choice(["honeypotpass"])
    
            logging.info(f"[*] Attacker moves to Node {node_id} ({ip}), attempting SSH attack...")
    
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
            try:
                ssh.connect(ip, username=fake_user, password=fake_pass, timeout=5)
                logging.info(f"[!] SSH Attack Success on Node {node_id} ({ip}) using {fake_pass}")
                ssh.close()
            except paramiko.AuthenticationException:
                logging.info(f"[*] Failed SSH login on Node {node_id} ({ip}) with {fake_pass}")
            except Exception as e:
                logging.error(f"[!] SSH attack error on {ip}: {e}")
    
            time.sleep(5)  # ✅ Ensure the attacker moves every 5 seconds

    def place_honeypot(self, action):
        """Places a honeypot at the given node action and runs LLMhoneypot.py inside the corresponding container."""
        if self.honeypot_positions[action] == 0 and self.placed_honeypots < self.num_honeypots:
            self.honeypot_positions[action] = 1
            self.placed_honeypots += 1
            logging.info(f"Honeypot placed at Node {action}")
    
            # Match the container name format from main.py
            container_name = f"node_{action+1}"  # Adjusted to match node index
    
            # Run LLMhoneypot.py inside the corresponding container in the background
            try:
                subprocess.Popen(
                    ["docker", "exec", "-d", container_name, "python3", "LLMhoneypot.py"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                logging.info(f"Started LLMhoneypot.py in container {container_name}")
            except Exception as e:
                logging.error(f"Failed to start LLMhoneypot.py in {container_name}: {e}")
    
            return 0.5
        else:
            logging.warning(f"[!] Honeypot already placed at Node {action}. Selecting a new node...")
            available_nodes = [i for i in range(self.num_nodes) if self.honeypot_positions[i] == 0]
            if available_nodes:
                new_node = np.random.choice(available_nodes)
                return self.place_honeypot(new_node)
        
        return 0

    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action: {action}"
        
        reward = 0
        self.step_count += 1  # Increment step count
        
        # **Introduce Placement Interval**
        if self.step_count % self.placement_interval == 0 and self.placed_honeypots < self.num_honeypots:
            reward += self.place_honeypot(action)  # Place a honeypot only at set intervals
        
        # **Check if Attacker Steps on Honeypot**
        if self.honeypot_positions[self.attacker_position] == 1:
            self.attack_detected[self.attacker_position] = True
            reward += 5.0  # High reward for trapping an attacker
            logging.info(f"[!] Attacker trapped at Node {self.attacker_position}!")
        
        # **Update Observation State**
        self.state = 1 - self.honeypot_positions  # Observation: 1 for empty, 0 for honeypot
        
        # **End the Episode if All Honeypots Are Placed**
        done = self.placed_honeypots >= self.num_honeypots
        self.visualize()
        return self.state, reward, done, {}

    
    def visualize(self):
        """
        Prints the environment state with:
        - 'O' for normal nodes
        - 'H' for honeypots
        - 'XH' for activated honeypots (trapped attack)
        - 'A' for attacker position
        """
        env_state = []
    
        for i in range(self.num_nodes):
            if self.honeypot_positions[i] == 1:  
                if self.attack_detected[i]:  # ✅ Attack trapped at honeypot
                    env_state.append("XH")  
                else:
                    env_state.append("H")
            elif i == self.attacker_position:  # ✅ Show attacker position
                env_state.append("A")
            else:
                env_state.append("O")
    
        print("\nEnvironment State: " + " | ".join(env_state))

    def reset(self):
        self.state = np.random.rand(self.num_nodes)
        self.honeypot_positions = np.zeros(self.num_nodes)
        self.attacker_position = np.random.choice(self.num_nodes)
        self.placed_honeypots = 0
        self.step_count = 0
        return self.state

    def close(self):
        """ Closes all SSH connections. """
        for ssh in self.ssh_clients.values():
            ssh.close()
        logging.info("All SSH connections closed.")

