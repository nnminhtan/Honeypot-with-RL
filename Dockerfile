# Use Ubuntu as the base image
FROM ubuntu:22.04

# Set environment variables to ensure non-interactive installs (prevents prompts)
ENV DEBIAN_FRONTEND=noninteractive
# Install SSH
RUN apt update && apt install -y openssh-server

# # Set up SSH
# RUN mkdir /var/run/sshd

# Create a new user with a password
RUN useradd -m -s /bin/bash honeypotuser && echo "honeypotuser:honeypotpass" | chpasswd

# Allow SSH login for the user
RUN mkdir -p /var/run/sshd

# Update and install dependencies for Python and pip
RUN apt-get update && \
    apt-get install -y \
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
    git && \
    apt-get clean

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

# Create a virtual environment (optional)
RUN python3.11 -m venv /env
ENV PATH="/env/bin:$PATH"

# Set the working directory (you can change this based on your project structure)
WORKDIR /app
# Install dependencies (if you have requirements.txt)
ADD final /app/
# Set working directory
WORKDIR /app

# Create history.txt and set permissions
RUN touch /app/history.txt && chmod 666 /app/history.txt

# RUN source bin/activate

COPY requirements.txt . 
RUN python3.11 -m pip install -r requirements.txt

# Expose a port (for SSH or your web app, adjust as necessary)
# EXPOSE 22

# # Keep the container running
# CMD ["/usr/sbin/sshd", "-D"]

# Start the container with bash (you can replace this with other commands if needed)
CMD ["/bin/bash"]
