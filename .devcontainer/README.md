# IndexAgent Dev Container Configuration

This directory contains the VSCode Dev Container configuration for the IndexAgent project, enabling consistent development environments across macOS and Windows.

## Quick Start

1. **Open in VSCode**: `code .` from the project root
2. **Reopen in Container**: VSCode will prompt, or use `Ctrl+Shift+P` → "Dev Containers: Reopen in Container"
3. **Wait for Setup**: First-time setup takes 5-10 minutes
4. **Start Developing**: All tools and dependencies are pre-configured

## Files Overview

| File | Purpose |
|------|---------|
| [`devcontainer.json`](devcontainer.json) | Main configuration file defining container settings, extensions, and environment |
| [`Dockerfile`](Dockerfile) | Custom container image with Python 3.11, Docker, and development tools |
| [`post-create.sh`](post-create.sh) | Runs once after container creation to set up the environment |
| [`post-start.sh`](post-start.sh) | Runs every time the container starts |
| [`test-devcontainer.sh`](test-devcontainer.sh) | Comprehensive test suite to verify container functionality |

## Key Features

### 🐍 Python Development
- Python 3.11 with all development dependencies
- Pre-configured linting (ruff), formatting (black), and type checking (mypy)
- Pytest with coverage reporting
- Automatic code formatting on save

### 🐳 Docker Integration
- Docker-in-Docker support for running containers
- Access to host Docker daemon
- Pre-installed docker-compose
- Ability to run the full IndexAgent stack

### 🔧 Development Tools
- Claude CLI for AI-assisted development
- Git with sensible cross-platform defaults
- Make for build automation
- Node.js and npm for additional tooling

### 📝 VSCode Integration
- 15+ pre-installed extensions for Python, Docker, and productivity
- Configured settings for optimal development experience
- Port forwarding for web services (6070, 3000, 8080)
- Integrated terminal with helpful aliases

## Helper Scripts

The container includes several convenience scripts in `~/.local/bin`:

```bash
# Run the complete test suite with coverage
run-tests

# Format code and run all quality checks
lint-code

# Start the full Docker stack
start-stack

# Test the Dev Container setup
.devcontainer/test-devcontainer.sh
```

## Quick Commands

### Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=indexagent --cov-report=term-missing

# Run integration tests only
pytest tests/integration/
```

### Code Quality
```bash
# Format all Python files
black .

# Lint all files
ruff check .

# Type check
mypy indexagent/
```

### Docker Operations
```bash
# Start the full stack
cd config && docker-compose up -d

# View running containers
docker ps

# View logs
docker-compose logs -f
```

## Troubleshooting

### Common Issues

**Container won't start**: Ensure Docker Desktop is running and restart VSCode

**Permission errors**: On Linux/WSL2, ensure your user is in the docker group:
```bash
sudo usermod -aG docker $USER
```

**Slow performance**: Increase Docker Desktop memory allocation to 8GB+

**Port conflicts**: Check if ports 6070, 3000, or 8080 are already in use

### Testing the Setup

Run the comprehensive test suite:
```bash
.devcontainer/test-devcontainer.sh
```

This will verify:
- Python environment and dependencies
- Docker functionality
- Development tools
- Project structure
- VSCode integration

## Cross-Platform Notes

### macOS
- Uses native Docker Desktop integration
- Automatic file permission handling
- Optimal performance with volume mounts

### Windows
- Requires WSL2 and Docker Desktop with WSL2 backend
- Automatic line ending conversion (LF)
- Path translation handled automatically

## Customization

### Adding Extensions
Edit [`devcontainer.json`](devcontainer.json) and add to the `extensions` array:
```json
"extensions": [
  "existing.extension",
  "new.extension.id"
]
```

### Adding System Packages
Edit [`Dockerfile`](Dockerfile) and add to the `apt-get install` command:
```dockerfile
RUN apt-get update \
    && apt-get -y install --no-install-recommends \
        your-new-package \
    && apt-get clean -y
```

### Environment Variables
Add to [`devcontainer.json`](devcontainer.json):
```json
"remoteEnv": {
  "YOUR_VARIABLE": "value"
}
```

## Performance Tips

1. **Allocate sufficient resources**: 8GB+ RAM, 4+ CPU cores
2. **Use volume mounts**: For better performance with large directories
3. **Enable BuildKit**: Already configured for faster Docker builds
4. **Close unused applications**: To free up system resources

## Security Considerations

- The container has access to the host Docker daemon
- Be cautious with untrusted code or containers
- Use environment variables for secrets, never commit them
- The container runs as the `vscode` user, not root

## Support

For detailed documentation, see [`docs/devcontainer-setup.md`](../docs/devcontainer-setup.md)

For issues:
1. Check the troubleshooting section above
2. Run the test script to identify problems
3. Check Docker Desktop status and logs
4. Create an issue with your OS, Docker version, and error details

---

**Happy coding!** 🚀 This Dev Container provides everything you need for IndexAgent development across any platform.