# Use an official Python image as the base
FROM python:3.11

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    openssh-server \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip==24.1.2 \
    && pip install --no-cache-dir google-cloud google-generativeai PyYAML python-dotenv paramiko

# Copy required files into the container
COPY LLMhoneypot.py personalitySSH.yml history.txt .env /app/

# Create the SSH user with a password
RUN useradd -m -s /bin/bash honeypotuser && \
    echo 'honeypotuser:honeypotpass' | chpasswd

# Configure SSH to allow password authentication
RUN sed -i 's/^#PasswordAuthentication no/PasswordAuthentication yes/' /etc/ssh/sshd_config && \
    sed -i 's/^#PermitRootLogin prohibit-password/PermitRootLogin no/' /etc/ssh/sshd_config

# Start SSH service
RUN mkdir /var/run/sshd

# Expose SSH port
# EXPOSE 22

# Start SSH service and run LLMHoneypot.py
CMD ["/bin/bash", "-c", "service ssh start && exec python3 LLMhoneypot.py"]
