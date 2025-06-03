#!/usr/bin/env python3
"""
Simple Dev Container Configuration Validator

This script validates the multi-repository Dev Container setup configurations
using only standard library modules.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

class SimpleConfigValidator:
    """Simple validator for Dev Container configurations."""
    
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
    
    def parse_docker_compose_ports(self, compose_file: Path) -> List[int]:
        """Parse ports from docker-compose.yml file (simple text parsing)."""
        if not compose_file.exists():
            return []
        
        ports = []
        try:
            with open(compose_file, 'r') as f:
                content = f.read()
            
            # Simple regex-like parsing for ports
            lines = content.split('\n')
            in_ports_section = False
            
            for line in lines:
                stripped = line.strip()
                if 'ports:' in stripped:
                    in_ports_section = True
                    continue
                
                if in_ports_section:
                    if stripped.startswith('-'):
                        # Extract port mapping like "8080:8080"
                        port_line = stripped.replace('-', '').replace('"', '').strip()
                        if ':' in port_line:
                            host_port = port_line.split(':')[0].strip()
                            try:
                                ports.append(int(host_port))
                            except ValueError:
                                pass
                    elif not stripped.startswith(' ') and stripped and not stripped.startswith('#'):
                        # End of ports section
                        in_ports_section = False
            
            return ports
        except Exception as e:
            self.log_warning(f"Could not parse docker-compose ports: {e}")
            return []
    
    def validate_port_consistency(self) -> bool:
        """Validate port configuration consistency across all configs."""
        print("\n🔍 Validating Port Configuration Consistency...")
        print("=" * 50)
        
        configs = {
            'IndexAgent': Path('.devcontainer/devcontainer.json'),
            'Workspace': Path('../.devcontainer-workspace/devcontainer.json'),
            'Airflow': Path('../airflow-hub/.devcontainer/devcontainer.json'),
            'Market': Path('../market-analysis/.devcontainer/devcontainer.json'),
            'Infra': Path('../infra/.devcontainer/devcontainer.json')
        }
        
        port_mappings = {}
        all_valid = True
        
        # Load all configurations
        for name, config_path in configs.items():
            config = self.validate_json_file(config_path)
            
            if config:
                ports = config.get('forwardPorts', [])
                port_attrs = config.get('portsAttributes', {})
                port_mappings[name] = {
                    'ports': ports,
                    'attributes': port_attrs
                }
                print(f"  {name}: Ports {ports}")
            else:
                all_valid = False
        
        # Parse docker-compose ports
        compose_ports = self.parse_docker_compose_ports(Path('../.devcontainer-workspace/docker-compose.yml'))
        if compose_ports:
            port_mappings['Docker-Compose'] = {'ports': compose_ports}
            print(f"  Docker-Compose: Ports {compose_ports}")
        
        # Check critical port assignments
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
        else:
            self.log_success("IndexAgent correctly removed port 8080 (conflict resolved)")
        
        # Validate workspace includes all necessary ports
        workspace_ports = port_mappings.get('Workspace', {}).get('ports', [])
        required_workspace_ports = [8080, 8081, 8000, 8200, 5432]
        
        for port in required_workspace_ports:
            if port not in workspace_ports:
                self.log_issue(f"Workspace missing required port {port}")
                all_valid = False
            else:
                self.log_success(f"Workspace has required port {port}")
        
        # Check port attributes for proper labeling
        indexagent_attrs = port_mappings.get('IndexAgent', {}).get('attributes', {})
        if '8081' in indexagent_attrs:
            label = indexagent_attrs['8081'].get('label', '')
            if 'IndexAgent' in label:
                self.log_success("IndexAgent port 8081 properly labeled")
            else:
                self.log_warning("IndexAgent port 8081 should be labeled as 'IndexAgent API'")
        
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
            
            # Check workspace folder path
            workspace_folder = config.get('workspaceFolder', '')
            expected_path = f'/workspaces/{repo_name.lower()}'
            if repo_name == 'IndexAgent':
                expected_path = '/workspaces/IndexAgent'
            elif repo_name == 'Airflow':
                expected_path = '/workspaces/airflow-hub'
            elif repo_name == 'Market':
                expected_path = '/workspaces/market-analysis'
            elif repo_name == 'Infra':
                expected_path = '/workspaces/infra'
            
            if workspace_folder == expected_path:
                self.log_success(f"{repo_name}: Workspace folder correctly set to {expected_path}")
            else:
                self.log_warning(f"{repo_name}: Workspace folder is '{workspace_folder}', expected '{expected_path}'")
        
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
                    self.log_issue(f"{repo_name}: INDEXAGENT_PORT should be '8081', got '{remote_env.get('INDEXAGENT_PORT')}'")
                    all_valid = False
                else:
                    self.log_success(f"{repo_name}: INDEXAGENT_PORT correctly set to 8081")
                
                if container_env.get('MULTI_REPO_AWARE') != 'true':
                    self.log_issue(f"{repo_name}: MULTI_REPO_AWARE should be 'true', got '{container_env.get('MULTI_REPO_AWARE')}'")
                    all_valid = False
                else:
                    self.log_success(f"{repo_name}: MULTI_REPO_AWARE correctly set")
            
            # Check database URL configurations
            db_url = remote_env.get('DATABASE_URL', '')
            if repo_name == 'IndexAgent' and db_url:
                if 'indexagent' not in db_url:
                    self.log_issue(f"{repo_name}: DATABASE_URL should reference 'indexagent' database")
                    all_valid = False
                else:
                    self.log_success(f"{repo_name}: DATABASE_URL correctly references 'indexagent' database")
            elif repo_name == 'Market' and db_url:
                if 'market_analysis' not in db_url:
                    self.log_issue(f"{repo_name}: DATABASE_URL should reference 'market_analysis' database")
                    all_valid = False
                else:
                    self.log_success(f"{repo_name}: DATABASE_URL correctly references 'market_analysis' database")
            elif repo_name == 'Airflow' and db_url:
                if 'airflow' not in db_url:
                    self.log_issue(f"{repo_name}: DATABASE_URL should reference 'airflow' database")
                    all_valid = False
                else:
                    self.log_success(f"{repo_name}: DATABASE_URL correctly references 'airflow' database")
        
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
            print("\n📋 Next Steps for Testing:")
            print("  1. Test individual repository Dev Containers")
            print("  2. Test workspace-level Dev Container")
            print("  3. Test service integration and networking")
            print("  4. Test port conflict resolution")
            print("  5. Test database connectivity")
        else:
            print("\n❌ Some validations failed. Please address the issues above.")
        
        return all_passed and not self.issues


def main():
    """Main entry point."""
    validator = SimpleConfigValidator()
    success = validator.run_validation()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()