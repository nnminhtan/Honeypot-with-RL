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
    def __init__(self, num_nodes, num_honeypots, honeypot_cost, malicious_ips, placement_interval=5):
        super(testEnv, self).__init__()
        self.num_nodes = num_nodes
        self.num_honeypots = num_honeypots
        self.honeypot_cost = honeypot_cost
        self.malicious_ips = malicious_ips
        self.placement_interval = placement_interval

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

        self.node_ips = self.create_docker_network()
        self.setup_ssh_connections()

        # **Start SSH Attack Simulation in a Separate Thread**
        attack_thread = threading.Thread(target=self.simulate_attack, daemon=True)
        attack_thread.start()

    def create_docker_network(self):
        node_ips = {}
        for i in range(self.num_nodes):
            container_name = f"node_{i+1}"
            subprocess.run(["docker", "run", "-d", "--network", "honeypot_network",
                            "--name", container_name, "honeypot-ssh-image"])

            result = subprocess.run(["docker", "inspect", "-f",
                                     "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}",
                                     container_name], capture_output=True, text=True)
            ip_address = result.stdout.strip()
            node_ips[i] = ip_address

        logging.info(f"Nodes initialized with IPs: {node_ips}")
        return node_ips

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
        if self.placed_honeypots < self.num_honeypots and self.honeypot_positions[action] == 0:
            self.honeypot_positions[action] = 1
            self.placed_honeypots += 1
            logging.info(f"Honeypot placed at Node {action}")
            return 0.5
        return 0  
    # def place_honeypot(self, node):
    #     if self.honeypot_positions[node] == 1:
    #         logging.warning(f"[!] Honeypot already placed at Node {node}.")
    #         return -1  # Penalty for redundant placement
    
    #     self.honeypot_positions[node] = 1
    #     logging.info(f"[*] Honeypot deployed at Node {node}.")
    
    #     # ✅ Run LLMHoneypot.py on the node's terminal via SSH
    #     if node in self.ssh_clients:
    #         ssh = self.ssh_clients[node]
    #         command = f"python3 /app/LLMhoneypot.py {node}"  # Adjust path if needed
    #         stdin, stdout, stderr = ssh.exec_command(command)
    
    #         logging.info(f"[*] Running LLMHoneypot.py on Node {node}...")
    #         logging.info(stdout.read().decode())  # Print output
    #         logging.error(stderr.read().decode())  # Print errors if any
    #     else:
    #         logging.error(f"[!] SSH connection to Node {node} not available!")
    
    #     return 1  # Small reward for placing honeypot


    # def step(self, action):
    #     assert self.action_space.contains(action), f"Invalid action: {action}"
    
    #     placement_reward = 0
    #     if self.step_count % self.placement_interval == 0:
    #         placement_reward = self.place_honeypot(action)
    
    #     reward = placement_reward
    #     self.state = np.random.rand(self.num_nodes) * (1 - self.honeypot_positions)
    #     self.step_count += 1
    
    #     # ✅ Fix: Check if the attacker is actually at the honeypot
    #     if self.honeypot_positions[self.attacker_position] == 1:
    #         self.attack_detected[self.attacker_position] = True  # ✅ Attack trapped
    #         reward += 5.0  # ✅ Give a high reward for catching an attacker
    #         logging.info(f"[!] Attacker trapped at Node {self.attacker_position}!")
    
    #     done = self.step_count >= self.num_honeypots * self.placement_interval
    
    #     self.visualize()  # ✅ Update visualization
    
    #     return self.state, reward, done, {}
    # def step(self, action):
    #     assert self.action_space.contains(action), f"Invalid action: {action}"
    
    #     placement_reward = 0
    #     if self.step_count % self.placement_interval == 0:
    #         placement_reward = self.place_honeypot(action)
    
    #     reward = placement_reward
    #     self.state = np.random.rand(self.num_nodes) * (1 - self.honeypot_positions)
    #     self.step_count += 1
    
    #     if self.honeypot_positions[self.attacker_position] == 1:
    #         self.attack_detected[self.attacker_position] = True
    #         reward += 5.0  # ✅ Reward for catching an attacker
    #         logging.info(f"[!] Attacker trapped at Node {self.attacker_position}!")
            
    #         # ✅ Pause training until user presses Enter
    #         input("[*] Attacker trapped! Press Enter to resume training...")
    
    #     done = self.step_count >= self.num_honeypots * self.placement_interval
    
    #     self.visualize()
    #     return self.state, reward, done, {}
    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action: {action}"
        
        logging.info(f"Agent chose action: {action}")  # ✅ Debug: Print chosen action
    
        placement_reward = 0
        if self.step_count % self.placement_interval == 0:
            placement_reward = self.place_honeypot(action)
    
        reward = placement_reward
        self.state = np.random.rand(self.num_nodes) * (1 - self.honeypot_positions)
        self.step_count += 1
    
        if self.honeypot_positions[action] == 1 and action == self.attacker_position:
            self.attack_detected[action] = True  
    
        done = self.step_count >= self.num_honeypots * self.placement_interval
    
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
