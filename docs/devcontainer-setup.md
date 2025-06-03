# VSCode Dev Containers Setup for IndexAgent

This document provides comprehensive instructions for using VSCode Dev Containers with the IndexAgent project, enabling consistent development environments across macOS and Windows.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
- [Architecture](#architecture)
- [Features](#features)
- [Usage](#usage)
- [Troubleshooting](#troubleshooting)
- [Advanced Configuration](#advanced-configuration)
- [Cross-Platform Considerations](#cross-platform-considerations)

## Overview

VSCode Dev Containers provide a consistent, isolated development environment that works identically across different operating systems. This setup allows you to:

- Work on the same codebase from both macOS and Windows
- Ensure all team members have identical development environments
- Isolate project dependencies from your host system
- Run Docker containers within the development environment
- Access all necessary development tools pre-configured

## Prerequisites

### Required Software

1. **VSCode** with the Dev Containers extension
   ```bash
   # Install the Remote Development extension pack
   code --install-extension ms-vscode-remote.vscode-remote-extensionpack
   ```

2. **Docker Desktop**
   - **macOS**: [Docker Desktop for Mac](https://docs.docker.com/desktop/mac/install/)
   - **Windows**: [Docker Desktop for Windows](https://docs.docker.com/desktop/windows/install/)

3. **Git** (usually pre-installed or available through system package managers)

### System Requirements

- **macOS**: macOS 10.15 or later
- **Windows**: Windows 10/11 with WSL2 enabled
- **RAM**: Minimum 8GB, recommended 16GB
- **Storage**: At least 10GB free space for containers and images

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/IndexAgent.git
cd IndexAgent
```

### 2. Open in VSCode

```bash
code .
```

### 3. Reopen in Container

When you open the project, VSCode will detect the Dev Container configuration and prompt you to reopen in a container. Click "Reopen in Container" or:

1. Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on macOS)
2. Type "Dev Containers: Reopen in Container"
3. Select the command

### 4. Wait for Setup

The first time you open the container, it will:
- Build the development image (5-10 minutes)
- Install all dependencies
- Configure the development environment
- Set up helpful scripts and tools

## Architecture

### Container Structure

```
Dev Container
├── Python 3.11 Environment
├── Docker-in-Docker Support
├── Development Tools
│   ├── black (code formatting)
│   ├── ruff (linting)
│   ├── mypy (type checking)
│   ├── pytest (testing)
│   └── coverage (test coverage)
├── Node.js Tools
│   ├── Claude CLI
│   └── markdownlint
└── System Tools
    ├── git
    ├── make
    └── various utilities
```

### Volume Mounts

- **Project Code**: `/workspaces/IndexAgent` (your project directory)
- **Host Repos**: `~/repos` → `/repos` (for accessing other repositories)
- **Docker Socket**: Host Docker socket mounted for Docker-in-Docker

### Port Forwarding

- **6070**: Zoekt UI
- **3000**: Sourcebot UI  
- **8080**: IndexAgent API

## Features

### Pre-installed Development Tools

#### Python Tools
- **black**: Code formatting with 100-character line length
- **ruff**: Fast Python linter with comprehensive rules
- **mypy**: Static type checking
- **pytest**: Testing framework with coverage support
- **invoke**: Task runner for automation

#### Node.js Tools
- **Claude CLI**: AI-powered code assistance
- **markdownlint**: Markdown linting and formatting

#### System Tools
- **Docker CLI**: Full Docker functionality
- **docker-compose**: Container orchestration
- **git**: Version control with sensible defaults
- **make**: Build automation

### VSCode Extensions

The Dev Container automatically installs and configures:

- **Python Development**
  - Python extension with Pylance
  - Black formatter
  - MyPy type checker
  - Ruff linter
  - Pytest test runner

- **Docker & Infrastructure**
  - Docker extension
  - Makefile tools

- **Code Quality**
  - Coverage gutters
  - Test adapters

- **Documentation**
  - Markdown support
  - Markdownlint

### Helper Scripts

The setup includes several convenience scripts in `~/.local/bin`:

#### `run-tests`
Runs the complete test suite with coverage reporting:
```bash
run-tests
```

#### `lint-code`
Formats code and runs all linting checks:
```bash
lint-code
```

#### `start-stack`
Starts the full IndexAgent Docker stack:
```bash
start-stack
```

## Usage

### Daily Development Workflow

1. **Start Development Session**
   ```bash
   # Open VSCode and reopen in container
   code .
   # VSCode will prompt to reopen in container
   ```

2. **Run Tests**
   ```bash
   run-tests
   # Or use VSCode's integrated test runner
   ```

3. **Format and Lint Code**
   ```bash
   lint-code
   # Or save files to auto-format with black
   ```

4. **Work with Docker**
   ```bash
   # Start the full stack
   start-stack
   
   # Or run individual containers
   docker run -d --name test-container nginx
   
   # Use docker-compose
   cd config
   docker-compose up -d
   ```

### Testing

#### Unit Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=indexagent --cov-report=term-missing

# Run specific test file
pytest tests/test_example.py

# Run with verbose output
pytest -v
```

#### Integration Tests
```bash
# Run integration tests
pytest tests/integration/

# Run specific integration test
pytest tests/integration/test_high_load_20.py
```

### Code Quality

#### Formatting
```bash
# Format all Python files
black .

# Check what would be formatted
black --check .
```

#### Linting
```bash
# Lint all files
ruff check .

# Fix auto-fixable issues
ruff check --fix .
```

#### Type Checking
```bash
# Type check the indexagent package
mypy indexagent/

# Type check with configuration
mypy --config-file=pyproject.toml indexagent/
```

### Docker Operations

#### Running the Full Stack
```bash
# Start all services
cd config
docker-compose up -d

# View logs
docker-compose logs -f

# Stop all services
docker-compose down
```

#### Individual Container Management
```bash
# List running containers
docker ps

# View container logs
docker logs <container-name>

# Execute commands in containers
docker exec -it <container-name> bash
```

## Troubleshooting

### Common Issues

#### 1. Container Won't Start

**Symptoms**: VSCode shows "Failed to start container"

**Solutions**:
```bash
# Check Docker is running
docker version

# Rebuild the container
# In VSCode: Ctrl+Shift+P → "Dev Containers: Rebuild Container"

# Or manually rebuild
docker build -t indexagent-devcontainer .devcontainer/
```

#### 2. Docker Socket Permission Issues

**Symptoms**: "Permission denied" when running Docker commands

**Solutions**:
```bash
# On Linux/WSL2, ensure user is in docker group
sudo usermod -aG docker $USER
# Then restart your session

# On macOS/Windows, restart Docker Desktop
```

#### 3. Port Conflicts

**Symptoms**: "Port already in use" errors

**Solutions**:
```bash
# Check what's using the port
lsof -i :6070  # or :3000, :8080

# Kill the process or change port in devcontainer.json
```

#### 4. Slow Performance

**Symptoms**: Container operations are slow

**Solutions**:
- Increase Docker Desktop memory allocation (8GB+)
- Use volume mounts instead of bind mounts for large directories
- Enable Docker Desktop's "Use the WSL 2 based engine" on Windows

#### 5. Python Dependencies Issues

**Symptoms**: Import errors or missing packages

**Solutions**:
```bash
# Reinstall dependencies
pip install -r requirements-dev.txt

# Clear pip cache
pip cache purge

# Rebuild container with clean slate
# VSCode: "Dev Containers: Rebuild Container Without Cache"
```

### Platform-Specific Issues

#### macOS

**File Permission Issues**:
```bash
# If you see permission errors with mounted volumes
sudo chown -R $(whoami) ~/repos
```

**Docker Desktop Not Starting**:
- Check System Preferences → Security & Privacy for blocked software
- Restart Docker Desktop
- Reset Docker Desktop to factory defaults if needed

#### Windows

**WSL2 Integration Issues**:
```bash
# Ensure WSL2 is enabled
wsl --set-default-version 2

# Update WSL2 kernel
wsl --update
```

**Line Ending Issues**:
```bash
# Configure git to handle line endings
git config --global core.autocrlf input
git config --global core.eol lf
```

**Path Issues**:
- Ensure Docker Desktop has access to your drives
- Use WSL2 paths when possible: `/mnt/c/Users/...`

## Advanced Configuration

### Customizing the Dev Container

#### Adding New Extensions

Edit [`.devcontainer/devcontainer.json`](.devcontainer/devcontainer.json):
```json
{
  "customizations": {
    "vscode": {
      "extensions": [
        "existing-extensions...",
        "new.extension.id"
      ]
    }
  }
}
```

#### Adding System Packages

Edit [`.devcontainer/Dockerfile`](.devcontainer/Dockerfile):
```dockerfile
RUN apt-get update \
    && apt-get -y install --no-install-recommends \
        your-new-package \
    && apt-get clean -y
```

#### Environment Variables

Add to [`devcontainer.json`](.devcontainer/devcontainer.json):
```json
{
  "remoteEnv": {
    "YOUR_VAR": "value"
  }
}
```

### Performance Optimization

#### Volume Mounts
For better performance, consider using named volumes for large directories:
```json
{
  "mounts": [
    "source=indexagent-node-modules,target=/workspaces/IndexAgent/node_modules,type=volume"
  ]
}
```

#### Resource Limits
Configure Docker Desktop resource limits:
- **Memory**: 8GB minimum, 16GB recommended
- **CPU**: Use all available cores
- **Disk**: At least 64GB for container images

## Cross-Platform Considerations

### File Paths

The Dev Container handles path differences automatically:
- **Windows**: `C:\Users\username\repos` → `/repos`
- **macOS**: `/Users/username/repos` → `/repos`
- **Linux**: `/home/username/repos` → `/repos`

### Line Endings

Git is configured to handle line endings consistently:
```bash
# These are set automatically in the container
git config core.autocrlf input
git config core.eol lf
```

### Environment Variables

Use the `localEnv` feature for platform-specific paths:
```json
{
  "mounts": [
    "source=${localEnv:HOME}${localEnv:USERPROFILE}/repos,target=/repos,type=bind"
  ]
}
```

### Docker Socket

The container automatically mounts the correct Docker socket:
- **Unix systems**: `/var/run/docker.sock`
- **Windows**: Named pipe (handled by Docker Desktop)

## Security Considerations

### Docker Socket Access

The Dev Container has access to the host Docker daemon. This means:
- Containers started inside can access the host network
- Be cautious with untrusted code or containers
- Consider using Docker-in-Docker for complete isolation if needed

### Secrets Management

Never commit secrets to the repository. Use:
- Environment variables
- Docker secrets
- External secret management systems

Example `.env` file (not committed):
```bash
CLAUDE_API_KEY=your-secret-key
DATABASE_PASSWORD=your-db-password
```

## Contributing

When contributing to the Dev Container configuration:

1. Test changes on both macOS and Windows
2. Update this documentation for any new features
3. Ensure backward compatibility when possible
4. Add appropriate comments to configuration files

## Support

For issues with the Dev Container setup:

1. Check this documentation first
2. Search existing GitHub issues
3. Create a new issue with:
   - Your operating system
   - Docker Desktop version
   - VSCode version
   - Error messages and logs

## References

- [VSCode Dev Containers Documentation](https://code.visualstudio.com/docs/remote/containers)
- [Docker Desktop Documentation](https://docs.docker.com/desktop/)
- [IndexAgent Project Documentation](../README.md)