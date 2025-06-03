# Use the official Python image as base
ARG VARIANT="3.11-bullseye"
FROM mcr.microsoft.com/vscode/devcontainers/python:0-${VARIANT}

# Avoid warnings by switching to noninteractive
ENV DEBIAN_FRONTEND=noninteractive

# Set up environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="${PATH}:/home/vscode/.local/bin"
ENV PYTHONPATH=/workspaces/IndexAgent
ENV DOCKER_BUILDKIT=1
ENV COMPOSE_DOCKER_CLI_BUILD=1

# Configure apt and install packages
RUN apt-get update \
    && apt-get -y install --no-install-recommends \
        # Essential development tools
        build-essential \
        curl \
        wget \
        git \
        make \
        jq \
        tree \
        htop \
        vim \
        nano \
        # Network tools
        net-tools \
        iputils-ping \
        telnet \
        # Node.js and npm (for Claude CLI and other tools)
        nodejs \
        npm \
        # Additional utilities
        ssh \
        sudo \
        ca-certificates \
        gnupg \
        lsb-release \
        # Python development dependencies
        python3-dev \
        python3-pip \
        python3-venv \
        # Database tools (for potential database work)
        postgresql-client \
        sqlite3 \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

# Install Docker CLI (latest stable version)
RUN curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/debian $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null \
    && apt-get update \
    && apt-get install -y docker-ce-cli docker-compose-plugin \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

# Install Docker Compose (standalone)
RUN curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose \
    && chmod +x /usr/local/bin/docker-compose

# Install Claude CLI and other Node.js tools
RUN npm install -g \
    @anthropic-ai/claude-code@0.2.64 \
    markdownlint-cli \
    npm-check-updates \
    && npm cache clean --force

# Create directories for development
RUN mkdir -p /workspaces/IndexAgent \
    && mkdir -p /repos \
    && mkdir -p /home/vscode/.local/bin

# Set up user permissions and groups
RUN usermod -aG docker vscode \
    && echo "vscode ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/vscode \
    && chmod 0440 /etc/sudoers.d/vscode

# Switch to vscode user for remaining operations
USER vscode

# Upgrade pip and install Python development tools
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --user \
        # Core development tools
        black \
        ruff \
        mypy \
        pytest \
        pytest-cov \
        pytest-asyncio \
        pytest-mock \
        coverage \
        invoke \
        # Additional useful tools
        ipython \
        jupyter \
        pre-commit \
        bandit \
        safety \
        # Documentation tools
        sphinx \
        mkdocs \
        mkdocs-material

# Create useful aliases and shell configuration
RUN echo 'alias ll="ls -alF"' >> ~/.bashrc \
    && echo 'alias la="ls -A"' >> ~/.bashrc \
    && echo 'alias l="ls -CF"' >> ~/.bashrc \
    && echo 'alias ..="cd .."' >> ~/.bashrc \
    && echo 'alias ...="cd ../.."' >> ~/.bashrc \
    && echo 'alias grep="grep --color=auto"' >> ~/.bashrc \
    && echo 'alias fgrep="fgrep --color=auto"' >> ~/.bashrc \
    && echo 'alias egrep="egrep --color=auto"' >> ~/.bashrc \
    && echo 'alias dc="docker-compose"' >> ~/.bashrc \
    && echo 'alias dcu="docker-compose up"' >> ~/.bashrc \
    && echo 'alias dcd="docker-compose down"' >> ~/.bashrc \
    && echo 'alias dcb="docker-compose build"' >> ~/.bashrc \
    && echo 'alias dps="docker ps"' >> ~/.bashrc \
    && echo 'alias di="docker images"' >> ~/.bashrc \
    && echo 'export PYTHONPATH=/workspaces/IndexAgent:$PYTHONPATH' >> ~/.bashrc \
    && echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc

# Set up git configuration template
RUN git config --global init.defaultBranch main \
    && git config --global pull.rebase false \
    && git config --global core.autocrlf input \
    && git config --global core.eol lf

# Create workspace directory and set permissions
RUN sudo mkdir -p /workspaces/IndexAgent \
    && sudo chown -R vscode:vscode /workspaces/IndexAgent

# Switch back to dialog for any ad-hoc use of apt-get
ENV DEBIAN_FRONTEND=dialog

# Set the working directory
WORKDIR /workspaces/IndexAgent

# Default command
CMD ["sleep", "infinity"]