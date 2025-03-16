import gym
from gym import spaces
import numpy as np
import logging
import subprocess
import os
import paramiko
import time
import shutil  # For copying files
import LLMhoneypot

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
        self.vulnerabilities = np.random.rand(num_nodes)
        self.attacker_position = np.random.choice(num_nodes)
        self.attacker_ip = None
        self.placed_honeypots = 0
        self.step_count = 0
        self.activated_traps = np.zeros(num_nodes)
        self.malicious_index = 0
        self.ssh_clients = {}

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

        # **NEW: Enable SSH logging on each node**
        self.setup_ssh_logging()

        # **NEW: Wait for an attacker before deploying honeypots**
        self.wait_for_intruder()

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

    def setup_ssh_logging(self):
        """ Enables SSH logging on all nodes. """
        for node_id, ssh in self.ssh_clients.items():
            try:
                logging.info(f"[*] Setting up SSH logging on Node {node_id}...")
    
                ssh.exec_command("sudo apt-get update && sudo apt-get install -y rsyslog")
                ssh.exec_command("sudo systemctl enable rsyslog && sudo systemctl start rsyslog")
                ssh.exec_command("echo 'auth,authpriv.* /var/log/auth.log' | sudo tee -a /etc/rsyslog.conf")
                ssh.exec_command("sudo systemctl restart rsyslog")
                ssh.exec_command("sudo systemctl enable ssh && sudo systemctl restart ssh")
    
                # **New Fix: Create auth.log manually if missing**
                ssh.exec_command("sudo touch /var/log/auth.log && sudo chmod 644 /var/log/auth.log")
    
                logging.info(f"[+] SSH logging enabled on Node {node_id}.")
            except Exception as e:
                logging.error(f"[!] Failed to set up logging on Node {node_id}: {e}")

    def wait_for_intruder(self):
        """ Waits for an attacker to SSH into any node before deploying honeypots. """
        logging.info("Waiting for an intruder to SSH into one of the nodes...")
        while True:
            for node_id, ssh in self.ssh_clients.items():
                try:
                    stdin, stdout, stderr = ssh.exec_command("cat /var/log/auth.log | grep 'Accepted password'")
                    login_events = stdout.read().decode().strip()
                    if login_events:
                        logging.info(f"[!] Intruder detected on Node {node_id}: {login_events}")
                        self.attacker_position = node_id
                        return  # Exit loop and proceed
                except Exception as e:
                    logging.error(f"Error reading SSH logs on Node {node_id}: {e}")
            time.sleep(5)

    def place_honeypot(self, action):
        if self.placed_honeypots < self.num_honeypots and self.honeypot_positions[action] == 0:
            self.honeypot_positions[action] = 1
            self.placed_honeypots += 1
            logging.info(f"Honeypot placed at Node {action}")
            return 0.5
        return 0

    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action: {action}"

        placement_reward = 0
        if self.step_count % self.placement_interval == 0:
            placement_reward = self.place_honeypot(action)

        reward = placement_reward
        self.state = np.random.rand(self.num_nodes) * (1 - self.honeypot_positions)
        self.step_count += 1

        self.malicious_index = (self.malicious_index + 1) % len(self.malicious_ips)
        done = self.step_count >= self.num_honeypots * self.placement_interval
        return self.state, reward, done, {}

    def visualize(self):
        """ Visualizes honeypot and attacker positions using ASCII representation. """
        grid = ["0"] * self.num_nodes
        for i in range(self.num_nodes):
            if self.honeypot_positions[i] == 1:
                grid[i] = "H"
        grid[self.attacker_position] = "X"
        print(" ".join(grid))
        print(f"Attacker at Node {self.attacker_position}")

    def reset(self):
        self.state = np.random.rand(self.num_nodes)
        self.honeypot_positions = np.zeros(self.num_nodes)
        self.attacker_position = np.random.choice(self.num_nodes)
        self.attacker_ip = self.malicious_ips[self.malicious_index]
        self.placed_honeypots = 0
        self.step_count = 0
        return self.state

    def close(self):
        """ Closes all SSH connections. """
        for ssh in self.ssh_clients.values():
            ssh.close()
        logging.info("All SSH connections closed.")
