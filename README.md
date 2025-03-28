# DACN

Run these commands
docker network create --internal honeypot_network
docker network create external_network
docker build -t honeypot-ssh-image .
cd finals
docker build -t llmhp .

docker run -dit --network honeypot_network --name honeypot-container -v /var/run/docker.sock:/var/run/docker.sock honeypot-ssh-image