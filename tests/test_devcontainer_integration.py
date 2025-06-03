"""
Integration tests for Dev Container functionality.

These tests verify that the Dev Container environment is properly configured
and can run the IndexAgent project components.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest


class TestDevContainerIntegration:
    """Test suite for Dev Container integration."""

    def test_python_environment(self):
        """Test that Python environment is properly configured."""
        # Check Python version
        assert sys.version_info >= (3, 11), "Python 3.11+ required"
        
        # Check PYTHONPATH includes project directory
        python_path = os.environ.get('PYTHONPATH', '')
        assert '/workspaces/IndexAgent' in python_path, "PYTHONPATH should include project directory"

    def test_development_tools_available(self):
        """Test that all development tools are available."""
        tools = [
            'black',
            'ruff', 
            'mypy',
            'pytest',
            'coverage',
            'invoke'
        ]
        
        for tool in tools:
            result = subprocess.run(['which', tool], capture_output=True, text=True)
            assert result.returncode == 0, f"{tool} should be available in PATH"

    def test_docker_integration(self):
        """Test Docker integration if Docker daemon is accessible."""
        try:
            result = subprocess.run(['docker', 'version'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                # Docker is available, test basic functionality
                hello_result = subprocess.run(
                    ['docker', 'run', '--rm', 'hello-world'], 
                    capture_output=True, 
                    text=True,
                    timeout=30
                )
                assert hello_result.returncode == 0, "Should be able to run Docker containers"
            else:
                pytest.skip("Docker daemon not accessible")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("Docker not available or timeout")

    def test_project_structure(self):
        """Test that project structure is accessible."""
        project_root = Path('/workspaces/IndexAgent')
        
        # Check key directories exist
        assert project_root.exists(), "Project root should exist"
        assert (project_root / 'indexagent').exists(), "indexagent package should exist"
        assert (project_root / 'tests').exists(), "tests directory should exist"
        assert (project_root / 'docs').exists(), "docs directory should exist"
        
        # Check key files exist
        assert (project_root / 'pyproject.toml').exists(), "pyproject.toml should exist"
        assert (project_root / 'requirements-dev.txt').exists(), "requirements-dev.txt should exist"

    def test_helper_scripts(self):
        """Test that helper scripts are available and executable."""
        scripts = [
            'run-tests',
            'lint-code', 
            'start-stack'
        ]
        
        home_bin = Path.home() / '.local' / 'bin'
        
        for script in scripts:
            script_path = home_bin / script
            assert script_path.exists(), f"{script} should exist in ~/.local/bin"
            assert script_path.is_file(), f"{script} should be a file"
            assert os.access(script_path, os.X_OK), f"{script} should be executable"

    def test_environment_variables(self):
        """Test that required environment variables are set."""
        # Check Dev Container specific variables
        assert os.environ.get('INDEXAGENT_DEV_CONTAINER') == 'true', \
            "INDEXAGENT_DEV_CONTAINER should be set to 'true'"
        
        # Check Python environment variables
        assert os.environ.get('PYTHONDONTWRITEBYTECODE') == '1', \
            "PYTHONDONTWRITEBYTECODE should be set"
        assert os.environ.get('PYTHONUNBUFFERED') == '1', \
            "PYTHONUNBUFFERED should be set"

    def test_git_configuration(self):
        """Test that Git is properly configured for cross-platform development."""
        # Test core.eol setting
        result = subprocess.run(['git', 'config', 'core.eol'], capture_output=True, text=True)
        assert result.returncode == 0 and result.stdout.strip() == 'lf', \
            "Git should be configured with LF line endings"
        
        # Test core.autocrlf setting
        result = subprocess.run(['git', 'config', 'core.autocrlf'], capture_output=True, text=True)
        assert result.returncode == 0 and result.stdout.strip() == 'input', \
            "Git should be configured with autocrlf=input"

    def test_node_tools(self):
        """Test that Node.js tools are available."""
        tools = [
            'node',
            'npm',
            'markdownlint'
        ]
        
        for tool in tools:
            result = subprocess.run(['which', tool], capture_output=True, text=True)
            assert result.returncode == 0, f"{tool} should be available in PATH"

    def test_claude_cli_installation(self):
        """Test that Claude CLI is installed."""
        # Check if claude command exists
        result = subprocess.run(['which', 'claude'], capture_output=True, text=True)
        if result.returncode != 0:
            # Check if it's installed in the expected location
            claude_path = Path('/usr/local/bin/claude')
            assert claude_path.exists(), "Claude CLI should be installed"

    @pytest.mark.skipif(
        not os.path.exists('/repos'),
        reason="Repos directory not mounted"
    )
    def test_repos_mount(self):
        """Test that repos directory is properly mounted."""
        repos_dir = Path('/repos')
        assert repos_dir.exists(), "Repos directory should be mounted"
        assert repos_dir.is_dir(), "Repos should be a directory"

    def test_working_directory(self):
        """Test that we're in the correct working directory."""
        current_dir = Path.cwd()
        expected_dir = Path('/workspaces/IndexAgent')
        assert current_dir == expected_dir, \
            f"Should be in {expected_dir}, but in {current_dir}"

    def test_python_package_imports(self):
        """Test that key Python packages can be imported."""
        packages = [
            'pytest',
            'coverage',
            'black',
            'invoke'
        ]
        
        for package in packages:
            try:
                __import__(package)
            except ImportError:
                pytest.fail(f"Should be able to import {package}")

    def test_file_permissions(self):
        """Test that file permissions are correctly set."""
        # Test that we can create files in the workspace
        test_file = Path('/workspaces/IndexAgent/test_permissions.tmp')
        try:
            test_file.write_text('test')
            assert test_file.exists(), "Should be able to create files in workspace"
        finally:
            if test_file.exists():
                test_file.unlink()

    def test_port_forwarding_configuration(self):
        """Test that port forwarding is configured (indirectly)."""
        # We can't directly test port forwarding from inside the container,
        # but we can check that the expected ports are documented
        devcontainer_config = Path('/workspaces/IndexAgent/.devcontainer/devcontainer.json')
        assert devcontainer_config.exists(), "devcontainer.json should exist"
        
        config_content = devcontainer_config.read_text()
        expected_ports = ['6070', '3000', '8080']
        
        for port in expected_ports:
            assert port in config_content, f"Port {port} should be configured for forwarding"


if __name__ == '__main__':
    # Run tests when executed directly
    pytest.main([__file__, '-v'])