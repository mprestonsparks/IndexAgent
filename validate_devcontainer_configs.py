#!/usr/bin/env python3
"""
Dev Container Configuration Validation Script

This script validates the multi-repository Dev Container setup configurations
without requiring to be inside a Dev Container environment.
"""

import json
import os
import subprocess
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple

class ConfigValidator:
    """Validates Dev Container configurations."""
    
    def __init__(self):
        self.issues = []
        self.warnings = []
        self.successes = []
        
    def log_issue(self, message: str):
        """Log a critical issue."""
        self.issues.append(message)
        print(f"❌ ISSUE: {message}")
    
    def log_warning(self, message: str):
        """Log a warning."""
        self.warnings.append(message)
        print(f"⚠️  WARNING: {message}")
    
    def log_success(self, message: str):
        """Log a success."""
        self.successes.append(message)
        print(f"✅ SUCCESS: {message}")
    
    def validate_json_file(self, file_path: Path) -> Optional[dict]:
        """Validate and load a JSON file."""
        if not file_path.exists():
            self.log_issue(f"File not found: {file_path}")
            return None
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            self.log_success(f"Valid JSON: {file_path}")
            return data
        except json.JSONDecodeError as e:
            self.log_issue(f"Invalid JSON in {file_path}: {e}")
            return None
        except Exception as e:
            self.log_issue(f"Error reading {file_path}: {e}")
            return None
    
    def validate_yaml_file(self, file_path: Path) -> Optional[dict]:
        """Validate and load a YAML file."""
        if not file_path.exists():
            self.log_issue(f"File not found: {file_path}")
            return None
        
        try:
            with open(file_path, 'r') as f:
                data = yaml.safe_load(f)
            self.log_success(f"Valid YAML: {file_path}")
            return data
        except yaml.YAMLError as e:
            self.log_issue(f"Invalid YAML in {file_path}: {e}")
            return None
        except Exception as e:
            self.log_issue(f"Error reading {file_path}: {e}")
            return None
    
    def validate_port_consistency(self) -> bool:
        """Validate port configuration consistency across all configs."""
        print("\n🔍 Validating Port Configuration Consistency...")
        print("=" * 50)
        
        configs = {
            'IndexAgent': Path('.devcontainer/devcontainer.json'),
            'Workspace': Path('../.devcontainer-workspace/devcontainer.json'),
            'Airflow': Path('../airflow-hub/.devcontainer/devcontainer.json'),
            'Market': Path('../market-analysis/.devcontainer/devcontainer.json'),
            'Infra': Path('../infra/.devcontainer/devcontainer.json'),
            'Docker-Compose': Path('../.devcontainer-workspace/docker-compose.yml')
        }
        
        port_mappings = {}
        all_valid = True
        
        # Load all configurations
        for name, config_path in configs.items():
            if name == 'Docker-Compose':
                config = self.validate_yaml_file(config_path)
            else:
                config = self.validate_json_file(config_path)
            
            if config:
                if name == 'Docker-Compose':
                    # Extract ports from docker-compose
                    services = config.get('services', {})
                    workspace_ports = []
                    for service_name, service_config in services.items():
                        ports = service_config.get('ports', [])
                        for port in ports:
                            if isinstance(port, str) and ':' in port:
                                host_port = port.split(':')[0]
                                workspace_ports.append(int(host_port))
                    port_mappings[name] = {'ports': workspace_ports}
                else:
                    ports = config.get('forwardPorts', [])
                    port_attrs = config.get('portsAttributes', {})
                    port_mappings[name] = {
                        'ports': ports,
                        'attributes': port_attrs
                    }
                
                print(f"  {name}: Ports {port_mappings[name]['ports']}")
            else:
                all_valid = False
        
        # Check critical port assignments
        expected_ports = {
            'IndexAgent': [8081],  # Should be 8081 (conflict resolution)
            'Airflow': [8080],     # Should be 8080
            'Market': [8000],      # Should be 8000
            'Vault': [8200],       # Should be 8200
            'PostgreSQL': [5432]   # Should be 5432
        }
        
        # Validate IndexAgent port conflict resolution
        indexagent_ports = port_mappings.get('IndexAgent', {}).get('ports', [])
        if 8081 not in indexagent_ports:
            self.log_issue("IndexAgent should have port 8081 configured (conflict resolution)")
            all_valid = False
        else:
            self.log_success("IndexAgent correctly configured with port 8081")
        
        if 8080 in indexagent_ports:
            self.log_issue("IndexAgent still has port 8080 configured (should be removed)")
            all_valid = False
        
        # Validate workspace includes all necessary ports
        workspace_ports = port_mappings.get('Workspace', {}).get('ports', [])
        required_workspace_ports = [8080, 8081, 8000, 8200, 5432]
        
        for port in required_workspace_ports:
            if port not in workspace_ports:
                self.log_issue(f"Workspace missing required port {port}")
                all_valid = False
        
        if all_valid:
            self.log_success("Port configuration consistency validation passed")
        
        return all_valid
    
    def validate_docker_compose(self) -> bool:
        """Validate docker-compose configuration."""
        print("\n🐳 Validating Docker Compose Configuration...")
        print("=" * 50)
        
        compose_file = Path('../.devcontainer-workspace/docker-compose.yml')
        compose_config = self.validate_yaml_file(compose_file)
        
        if not compose_config:
            return False
        
        all_valid = True
        
        # Check required services
        services = compose_config.get('services', {})
        required_services = ['workspace', 'postgres', 'vault', 'redis']
        
        for service in required_services:
            if service not in services:
                self.log_issue(f"Missing required service: {service}")
                all_valid = False
            else:
                self.log_success(f"Service '{service}' is configured")
        
        # Check workspace dependencies
        workspace_config = services.get('workspace', {})
        depends_on = workspace_config.get('depends_on', [])
        
        required_deps = ['postgres', 'vault']
        for dep in required_deps:
            if dep not in depends_on:
                self.log_issue(f"Workspace should depend on '{dep}'")
                all_valid = False
            else:
                self.log_success(f"Workspace correctly depends on '{dep}'")
        
        # Check network configuration
        networks = compose_config.get('networks', {})
        if 'multi-repo-network' not in networks:
            self.log_issue("Missing 'multi-repo-network' network configuration")
            all_valid = False
        else:
            self.log_success("Multi-repo network is configured")
        
        # Check volume configuration
        volumes = compose_config.get('volumes', {})
        required_volumes = ['postgres_data', 'vault_data', 'redis_data']
        
        for volume in required_volumes:
            if volume not in volumes:
                self.log_issue(f"Missing volume: {volume}")
                all_valid = False
            else:
                self.log_success(f"Volume '{volume}' is configured")
        
        return all_valid
    
    def validate_individual_configs(self) -> bool:
        """Validate individual repository configurations."""
        print("\n📦 Validating Individual Repository Configurations...")
        print("=" * 50)
        
        repos = {
            'IndexAgent': Path('.devcontainer/devcontainer.json'),
            'Airflow': Path('../airflow-hub/.devcontainer/devcontainer.json'),
            'Market': Path('../market-analysis/.devcontainer/devcontainer.json'),
            'Infra': Path('../infra/.devcontainer/devcontainer.json')
        }
        
        all_valid = True
        
        for repo_name, config_path in repos.items():
            print(f"\n  Validating {repo_name}...")
            config = self.validate_json_file(config_path)
            
            if not config:
                all_valid = False
                continue
            
            # Check required fields
            required_fields = ['name', 'workspaceFolder']
            for field in required_fields:
                if field not in config:
                    self.log_issue(f"{repo_name}: Missing required field '{field}'")
                    all_valid = False
                else:
                    self.log_success(f"{repo_name}: Has required field '{field}'")
            
            # Check build or image configuration
            if 'build' not in config and 'image' not in config:
                self.log_issue(f"{repo_name}: Must have either 'build' or 'image' configuration")
                all_valid = False
            else:
                self.log_success(f"{repo_name}: Has build/image configuration")
            
            # Check features
            features = config.get('features', {})
            if 'ghcr.io/devcontainers/features/docker-in-docker:2' not in features:
                self.log_warning(f"{repo_name}: Missing Docker-in-Docker feature")
            else:
                self.log_success(f"{repo_name}: Has Docker-in-Docker feature")
        
        return all_valid
    
    def validate_environment_consistency(self) -> bool:
        """Validate environment variable consistency."""
        print("\n🌍 Validating Environment Variable Consistency...")
        print("=" * 50)
        
        configs = {
            'IndexAgent': Path('.devcontainer/devcontainer.json'),
            'Workspace': Path('../.devcontainer-workspace/devcontainer.json'),
            'Airflow': Path('../airflow-hub/.devcontainer/devcontainer.json'),
            'Market': Path('../market-analysis/.devcontainer/devcontainer.json')
        }
        
        all_valid = True
        
        for repo_name, config_path in configs.items():
            config = self.validate_json_file(config_path)
            if not config:
                continue
            
            remote_env = config.get('remoteEnv', {})
            container_env = config.get('containerEnv', {})
            
            # Check IndexAgent specific configurations
            if repo_name == 'IndexAgent':
                if remote_env.get('INDEXAGENT_PORT') != '8081':
                    self.log_issue(f"{repo_name}: INDEXAGENT_PORT should be '8081'")
                    all_valid = False
                else:
                    self.log_success(f"{repo_name}: INDEXAGENT_PORT correctly set to 8081")
                
                if container_env.get('MULTI_REPO_AWARE') != 'true':
                    self.log_issue(f"{repo_name}: MULTI_REPO_AWARE should be 'true'")
                    all_valid = False
                else:
                    self.log_success(f"{repo_name}: MULTI_REPO_AWARE correctly set")
            
            # Check database URL configurations
            db_url = remote_env.get('DATABASE_URL', '')
            if repo_name == 'IndexAgent' and 'indexagent' not in db_url:
                self.log_issue(f"{repo_name}: DATABASE_URL should reference 'indexagent' database")
                all_valid = False
            elif repo_name == 'Market' and 'market_analysis' not in db_url:
                self.log_issue(f"{repo_name}: DATABASE_URL should reference 'market_analysis' database")
                all_valid = False
            elif db_url and ('indexagent' in db_url or 'market_analysis' in db_url or 'airflow' in db_url):
                self.log_success(f"{repo_name}: DATABASE_URL correctly configured")
        
        return all_valid
    
    def validate_file_structure(self) -> bool:
        """Validate required file structure."""
        print("\n📁 Validating File Structure...")
        print("=" * 50)
        
        required_files = [
            Path('.devcontainer/devcontainer.json'),
            Path('.devcontainer/Dockerfile'),
            Path('.devcontainer/post-create.sh'),
            Path('.devcontainer/post-start.sh'),
            Path('../.devcontainer-workspace/devcontainer.json'),
            Path('../.devcontainer-workspace/docker-compose.yml'),
            Path('../.devcontainer-workspace/Dockerfile.workspace'),
            Path('../.devcontainer-workspace/init-multiple-databases.sh'),
            Path('../airflow-hub/.devcontainer/devcontainer.json'),
            Path('../market-analysis/.devcontainer/devcontainer.json'),
            Path('../infra/.devcontainer/devcontainer.json')
        ]
        
        all_valid = True
        
        for file_path in required_files:
            if file_path.exists():
                self.log_success(f"Required file exists: {file_path}")
            else:
                self.log_issue(f"Missing required file: {file_path}")
                all_valid = False
        
        # Check script permissions
        script_files = [
            Path('.devcontainer/post-create.sh'),
            Path('.devcontainer/post-start.sh'),
            Path('../.devcontainer-workspace/post-create.sh'),
            Path('../.devcontainer-workspace/post-start.sh'),
            Path('../.devcontainer-workspace/init-multiple-databases.sh')
        ]
        
        for script_path in script_files:
            if script_path.exists():
                if os.access(script_path, os.X_OK):
                    self.log_success(f"Script is executable: {script_path}")
                else:
                    self.log_warning(f"Script is not executable: {script_path}")
        
        return all_valid
    
    def run_validation(self) -> bool:
        """Run all validation checks."""
        print("🚀 Starting Dev Container Configuration Validation...")
        print("=" * 60)
        
        validations = [
            self.validate_file_structure(),
            self.validate_individual_configs(),
            self.validate_docker_compose(),
            self.validate_port_consistency(),
            self.validate_environment_consistency()
        ]
        
        all_passed = all(validations)
        
        # Summary
        print("\n📊 Validation Summary")
        print("=" * 30)
        print(f"✅ Successes: {len(self.successes)}")
        print(f"⚠️  Warnings: {len(self.warnings)}")
        print(f"❌ Issues: {len(self.issues)}")
        
        if self.issues:
            print("\n🔥 Critical Issues Found:")
            for issue in self.issues:
                print(f"  • {issue}")
        
        if self.warnings:
            print("\n⚠️  Warnings:")
            for warning in self.warnings:
                print(f"  • {warning}")
        
        if all_passed and not self.issues:
            print("\n🎉 All validations passed! Configuration looks good.")
        else:
            print("\n❌ Some validations failed. Please address the issues above.")
        
        return all_passed and not self.issues


def main():
    """Main entry point."""
    validator = ConfigValidator()
    success = validator.run_validation()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()