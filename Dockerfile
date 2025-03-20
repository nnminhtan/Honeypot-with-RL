# Use Ubuntu as the base image
FROM ubuntu:22.04

# Set environment variables to ensure non-interactive installs
ENV DEBIAN_FRONTEND=noninteractive

# Unminimize Ubuntu (makes sure basic utilities like `sudo` are available)
RUN yes | unminimize

# Install SSH, sudo, and dependencies
RUN apt update && apt install -y \
    openssh-server \
    sudo \
    wget \
    curl \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    llvm \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libffi-dev \
    liblzma-dev \
    python3-dev \
    python3-venv \
    git \
    ca-certificates \
    gnupg && \
    apt clean

# Create a new user with a password and add to sudo group
RUN useradd -m -s /bin/bash honeypotuser && \
    echo "honeypotuser:honeypotpass" | chpasswd && \
    usermod -aG sudo honeypotuser

# Ensure SSH login is allowed for the new user
RUN mkdir -p /var/run/sshd

# Install Docker
RUN install -m 0755 -d /etc/apt/keyrings && \
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | tee /etc/apt/keyrings/docker.asc > /dev/null && \
    chmod a+r /etc/apt/keyrings/docker.asc && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null && \
    apt update && \
    apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin && \
    apt clean

# Install Python 3.11
RUN wget https://www.python.org/ftp/python/3.11.0/Python-3.11.0.tgz && \
    tar -xvzf Python-3.11.0.tgz && \
    cd Python-3.11.0 && \
    ./configure --enable-optimizations && \
    make -j$(nproc) && \
    make altinstall && \
    cd .. && \
    rm -rf Python-3.11.0*  # Clean up to reduce image size

# Ensure python3.11 is installed correctly
RUN python3.11 --version

# Install pip 24.1.2 (the version you want)
RUN curl https://bootstrap.pypa.io/get-pip.py | python3.11 && \
    python3.11 -m pip install --upgrade pip==24.1.2

# Ensure pip 24.1.2 is installed correctly
RUN python3.11 -m pip --version

# Set default python and pip to 3.11
RUN update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.11 1
RUN update-alternatives --install /usr/bin/pip pip /usr/local/bin/pip3.11 1

# Create a virtual environment
RUN python3 -m venv /env
ENV PATH="/env/bin:$PATH"

# Add project files
ADD final /app/
WORKDIR /app

# Create history.txt and set permissions
RUN touch /app/history.txt && chmod 666 /app/history.txt
RUN mkdir -p /app && chmod 777 /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Expose port 22 for SSH
# EXPOSE 22

# Start SSH when the container starts
# CMD ["/bin/bash", "-c", "/usr/sbin/sshd -D"]
# CMD ["/bin/bash", "-c", "/usr/sbin/sshd -D" , "echo 'Container is running' && tail -f /dev/null"]
CMD ["/bin/bash", "-c", "/usr/sbin/sshd -D & echo 'Container is running' | tee -a /app/container.log && tail -f /app/container.log"]
# CMD [ "/bin/bash", "-c", "python main.py" ]
