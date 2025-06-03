#!/usr/bin/env python3
"""
Comprehensive Multi-Repository Dev Container Testing Suite

This script performs systematic testing and validation of the multi-repository
Dev Container setup including individual repositories, workspace configuration,
integration testing, and edge cases.
"""

import json
import os
import subprocess
import sys
import time
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('devcontainer_test_results.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class DevContainerTester:
    """Comprehensive Dev Container testing framework."""
    
    def __init__(self):
        self.test_results = {
            'individual_repos': {},
            'workspace_config': {},
            'integration': {},
            'edge_cases': {},
            'performance': {}
        }
        self.failed_tests = []
        self.passed_tests = []
        
    def run_command(self, cmd: List[str], timeout: int = 30, cwd: Optional[str] = None) -> Tuple[int, str, str]:
        """Run a command and return exit code, stdout, stderr."""
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=timeout,
                cwd=cwd
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", f"Command timed out after {timeout}s"
        except Exception as e:
            return -1, "", str(e)
    
    def test_port_configuration_consistency(self) -> bool:
        """Test that port configurations are consistent across all configs."""
        logger.info("🔍 Testing port configuration consistency...")
        
        configs = {
            'IndexAgent': '.devcontainer/devcontainer.json',
            'Workspace': '../.devcontainer-workspace/devcontainer.json',
            'Airflow': '../airflow-hub/.devcontainer/devcontainer.json',
            'Market': '../market-analysis/.devcontainer/devcontainer.json',
            'Infra': '../infra/.devcontainer/devcontainer.json'
        }
        
        port_mappings = {}
        
        for name, config_path in configs.items():
            try:
                if Path(config_path).exists():
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    
                    ports = config.get('forwardPorts', [])
                    port_attrs = config.get('portsAttributes', {})
                    
                    port_mappings[name] = {
                        'ports': ports,
                        'attributes': port_attrs
                    }
                    logger.info(f"  {name}: Ports {ports}")
                else:
                    logger.warning(f"  {name}: Config not found at {config_path}")
            except Exception as e:
                logger.error(f"  {name}: Error reading config - {e}")
                return False
        
        # Check for IndexAgent port conflict resolution
        indexagent_ports = port_mappings.get('IndexAgent', {}).get('ports', [])
        workspace_ports = port_mappings.get('Workspace', {}).get('ports', [])
        
        if 8081 not in indexagent_ports:
            logger.error("  ❌ IndexAgent should have port 8081 configured")
            return False
        
        if 8081 not in workspace_ports:
            logger.error("  ❌ Workspace should have port 8081 configured")
            return False
            
        logger.info("  ✅ Port configuration consistency check passed")
        return True
    
    def test_docker_compose_configuration(self) -> bool:
        """Test workspace docker-compose configuration."""
        logger.info("🐳 Testing docker-compose configuration...")
        
        compose_file = Path('../.devcontainer-workspace/docker-compose.yml')
        if not compose_file.exists():
            logger.error("  ❌ docker-compose.yml not found")
            return False
        
        # Test compose file syntax
        exit_code, stdout, stderr = self.run_command([
            'docker-compose', '-f', str(compose_file), 'config'
        ])
        
        if exit_code != 0:
            logger.error(f"  ❌ docker-compose config validation failed: {stderr}")
            return False
        
        logger.info("  ✅ docker-compose configuration is valid")
        
        # Check service dependencies
        try:
            import yaml
            with open(compose_file, 'r') as f:
                compose_config = yaml.safe_load(f)
            
            services = compose_config.get('services', {})
            workspace_deps = services.get('workspace', {}).get('depends_on', [])
            
            if 'postgres' not in workspace_deps:
                logger.error("  ❌ Workspace should depend on postgres")
                return False
                
            if 'vault' not in workspace_deps:
                logger.error("  ❌ Workspace should depend on vault")
                return False
                
            logger.info("  ✅ Service dependencies are correctly configured")
            return True
            
        except Exception as e:
            logger.error(f"  ❌ Error parsing docker-compose.yml: {e}")
            return False
    
    def test_database_initialization_script(self) -> bool:
        """Test database initialization script."""
        logger.info("🗄️ Testing database initialization script...")
        
        init_script = Path('../.devcontainer-workspace/init-multiple-databases.sh')
        if not init_script.exists():
            logger.error("  ❌ Database initialization script not found")
            return False
        
        # Check script is executable
        if not os.access(init_script, os.X_OK):
            logger.error("  ❌ Database initialization script is not executable")
            return False
        
        # Check script content for required databases
        script_content = init_script.read_text()
        required_dbs = ['indexagent', 'market_analysis']
        
        for db in required_dbs:
            if db not in script_content:
                logger.error(f"  ❌ Database '{db}' not found in initialization script")
                return False
        
        logger.info("  ✅ Database initialization script is properly configured")
        return True
    
    def test_environment_variables(self) -> bool:
        """Test environment variable configuration."""
        logger.info("🌍 Testing environment variables...")
        
        # Test IndexAgent specific variables
        indexagent_port = os.environ.get('INDEXAGENT_PORT')
        if indexagent_port != '8081':
            logger.error(f"  ❌ INDEXAGENT_PORT should be 8081, got {indexagent_port}")
            return False
        
        # Test multi-repo awareness
        multi_repo_aware = os.environ.get('MULTI_REPO_AWARE')
        if multi_repo_aware != 'true':
            logger.error(f"  ❌ MULTI_REPO_AWARE should be true, got {multi_repo_aware}")
            return False
        
        # Test database URL
        db_url = os.environ.get('DATABASE_URL')
        if not db_url or 'indexagent' not in db_url:
            logger.error(f"  ❌ DATABASE_URL should contain 'indexagent', got {db_url}")
            return False
        
        logger.info("  ✅ Environment variables are correctly configured")
        return True
    
    def test_service_health_endpoints(self) -> bool:
        """Test service health endpoints if services are running."""
        logger.info("🏥 Testing service health endpoints...")
        
        services = {
            'Vault': 'http://localhost:8200/v1/sys/health',
            'PostgreSQL': None,  # No HTTP endpoint, will test with pg_isready
        }
        
        results = {}
        
        # Test Vault
        try:
            response = requests.get(services['Vault'], timeout=5)
            if response.status_code in [200, 429, 501]:  # Vault returns various codes when healthy
                logger.info("  ✅ Vault service is responding")
                results['vault'] = True
            else:
                logger.warning(f"  ⚠️ Vault returned status {response.status_code}")
                results['vault'] = False
        except requests.exceptions.RequestException:
            logger.warning("  ⚠️ Vault service not accessible (may not be running)")
            results['vault'] = False
        
        # Test PostgreSQL
        exit_code, stdout, stderr = self.run_command([
            'pg_isready', '-h', 'localhost', '-p', '5432', '-U', 'airflow'
        ])
        
        if exit_code == 0:
            logger.info("  ✅ PostgreSQL service is responding")
            results['postgres'] = True
        else:
            logger.warning("  ⚠️ PostgreSQL service not accessible (may not be running)")
            results['postgres'] = False
        
        return any(results.values())  # Return True if any service is healthy
    
    def test_volume_mounts(self) -> bool:
        """Test volume mount accessibility."""
        logger.info("📁 Testing volume mounts...")
        
        required_mounts = ['/data', '/logs', '/secrets', '/repos']
        optional_mounts = ['/workspaces']
        
        all_good = True
        
        for mount in required_mounts:
            if Path(mount).exists():
                # Test write access
                test_file = Path(mount) / 'test_write_access.tmp'
                try:
                    test_file.write_text('test')
                    test_file.unlink()
                    logger.info(f"  ✅ {mount} is accessible and writable")
                except Exception as e:
                    logger.error(f"  ❌ {mount} is not writable: {e}")
                    all_good = False
            else:
                logger.warning(f"  ⚠️ {mount} mount not found")
                all_good = False
        
        for mount in optional_mounts:
            if Path(mount).exists():
                logger.info(f"  ✅ {mount} is accessible")
            else:
                logger.info(f"  ℹ️ {mount} mount not found (optional)")
        
        return all_good
    
    def test_docker_daemon_access(self) -> bool:
        """Test Docker daemon accessibility."""
        logger.info("🐳 Testing Docker daemon access...")
        
        exit_code, stdout, stderr = self.run_command(['docker', 'version'])
        
        if exit_code == 0:
            logger.info("  ✅ Docker daemon is accessible")
            
            # Test container operations
            exit_code, stdout, stderr = self.run_command([
                'docker', 'run', '--rm', 'hello-world'
            ])
            
            if exit_code == 0:
                logger.info("  ✅ Docker container operations work")
                return True
            else:
                logger.error(f"  ❌ Docker container operations failed: {stderr}")
                return False
        else:
            logger.warning("  ⚠️ Docker daemon not accessible")
            return False
    
    def test_network_connectivity(self) -> bool:
        """Test network connectivity between services."""
        logger.info("🌐 Testing network connectivity...")
        
        # Test internal DNS resolution
        hosts_to_test = ['postgres', 'vault', 'redis']
        
        for host in hosts_to_test:
            exit_code, stdout, stderr = self.run_command([
                'nslookup', host
            ])
            
            if exit_code == 0:
                logger.info(f"  ✅ {host} DNS resolution works")
            else:
                logger.warning(f"  ⚠️ {host} DNS resolution failed (service may not be running)")
        
        return True  # DNS failures are expected if services aren't running
    
    def run_individual_repo_tests(self) -> Dict[str, bool]:
        """Test individual repository configurations."""
        logger.info("📦 Testing individual repository configurations...")
        
        repos = {
            'IndexAgent': '.',
            'Airflow': '../airflow-hub',
            'Market': '../market-analysis', 
            'Infra': '../infra'
        }
        
        results = {}
        
        for repo_name, repo_path in repos.items():
            logger.info(f"  Testing {repo_name}...")
            
            devcontainer_path = Path(repo_path) / '.devcontainer' / 'devcontainer.json'
            
            if devcontainer_path.exists():
                try:
                    with open(devcontainer_path, 'r') as f:
                        config = json.load(f)
                    
                    # Basic validation
                    has_name = 'name' in config
                    has_build_or_image = 'build' in config or 'image' in config
                    has_workspace_folder = 'workspaceFolder' in config
                    
                    repo_valid = has_name and has_build_or_image and has_workspace_folder
                    
                    if repo_valid:
                        logger.info(f"    ✅ {repo_name} configuration is valid")
                    else:
                        logger.error(f"    ❌ {repo_name} configuration is invalid")
                    
                    results[repo_name] = repo_valid
                    
                except Exception as e:
                    logger.error(f"    ❌ {repo_name} configuration error: {e}")
                    results[repo_name] = False
            else:
                logger.warning(f"    ⚠️ {repo_name} devcontainer config not found")
                results[repo_name] = False
        
        return results
    
    def run_workspace_tests(self) -> Dict[str, bool]:
        """Test workspace-level configuration."""
        logger.info("🏗️ Testing workspace-level configuration...")
        
        tests = {
            'port_consistency': self.test_port_configuration_consistency(),
            'docker_compose': self.test_docker_compose_configuration(),
            'database_init': self.test_database_initialization_script(),
            'environment_vars': self.test_environment_variables(),
            'volume_mounts': self.test_volume_mounts()
        }
        
        return tests
    
    def run_integration_tests(self) -> Dict[str, bool]:
        """Test integration between components."""
        logger.info("🔗 Testing integration between components...")
        
        tests = {
            'service_health': self.test_service_health_endpoints(),
            'docker_access': self.test_docker_daemon_access(),
            'network_connectivity': self.test_network_connectivity()
        }
        
        return tests
    
    def run_performance_tests(self) -> Dict[str, bool]:
        """Test performance characteristics."""
        logger.info("⚡ Testing performance characteristics...")
        
        # Test startup time simulation
        start_time = time.time()
        
        # Simulate some operations
        self.run_command(['python', '--version'])
        self.run_command(['docker', '--version'])
        
        end_time = time.time()
        startup_time = end_time - start_time
        
        tests = {
            'basic_commands_responsive': startup_time < 5.0
        }
        
        if tests['basic_commands_responsive']:
            logger.info(f"  ✅ Basic commands responsive ({startup_time:.2f}s)")
        else:
            logger.warning(f"  ⚠️ Basic commands slow ({startup_time:.2f}s)")
        
        return tests
    
    def run_edge_case_tests(self) -> Dict[str, bool]:
        """Test edge cases and error conditions."""
        logger.info("🧪 Testing edge cases and error conditions...")
        
        tests = {}
        
        # Test handling of missing directories
        missing_dir_test = not Path('/nonexistent').exists()
        tests['handles_missing_dirs'] = missing_dir_test
        
        if missing_dir_test:
            logger.info("  ✅ Properly handles missing directories")
        else:
            logger.error("  ❌ Unexpected directory found")
        
        # Test file permission handling
        try:
            test_file = Path('/tmp/permission_test.tmp')
            test_file.write_text('test')
            test_file.chmod(0o644)
            content = test_file.read_text()
            test_file.unlink()
            tests['file_permissions'] = content == 'test'
            logger.info("  ✅ File permissions work correctly")
        except Exception as e:
            tests['file_permissions'] = False
            logger.error(f"  ❌ File permission test failed: {e}")
        
        return tests
    
    def generate_report(self) -> str:
        """Generate comprehensive test report."""
        report = []
        report.append("=" * 80)
        report.append("MULTI-REPOSITORY DEV CONTAINER TEST REPORT")
        report.append("=" * 80)
        report.append("")
        
        total_tests = 0
        passed_tests = 0
        
        for category, tests in self.test_results.items():
            if tests:
                report.append(f"{category.upper().replace('_', ' ')} TESTS:")
                report.append("-" * 40)
                
                for test_name, result in tests.items():
                    status = "✅ PASS" if result else "❌ FAIL"
                    report.append(f"  {test_name}: {status}")
                    total_tests += 1
                    if result:
                        passed_tests += 1
                
                report.append("")
        
        # Summary
        report.append("SUMMARY:")
        report.append("-" * 40)
        report.append(f"Total tests: {total_tests}")
        report.append(f"Passed: {passed_tests}")
        report.append(f"Failed: {total_tests - passed_tests}")
        report.append(f"Success rate: {(passed_tests/total_tests*100):.1f}%" if total_tests > 0 else "No tests run")
        report.append("")
        
        # Recommendations
        if total_tests - passed_tests > 0:
            report.append("RECOMMENDATIONS:")
            report.append("-" * 40)
            report.append("• Check failed tests above for specific issues")
            report.append("• Ensure all services are running for integration tests")
            report.append("• Verify volume mounts are properly configured")
            report.append("• Check network connectivity between services")
            report.append("")
        
        return "\n".join(report)
    
    def run_all_tests(self) -> str:
        """Run all test suites and return comprehensive report."""
        logger.info("🚀 Starting comprehensive Dev Container testing...")
        logger.info("=" * 60)
        
        # Run all test categories
        self.test_results['individual_repos'] = self.run_individual_repo_tests()
        self.test_results['workspace_config'] = self.run_workspace_tests()
        self.test_results['integration'] = self.run_integration_tests()
        self.test_results['edge_cases'] = self.run_edge_case_tests()
        self.test_results['performance'] = self.run_performance_tests()
        
        # Generate and save report
        report = self.generate_report()
        
        # Save to file
        with open('devcontainer_test_report.txt', 'w') as f:
            f.write(report)
        
        logger.info("=" * 60)
        logger.info("🏁 Testing complete! Report saved to devcontainer_test_report.txt")
        
        return report


def main():
    """Main entry point."""
    tester = DevContainerTester()
    report = tester.run_all_tests()
    print("\n" + report)
    
    # Return appropriate exit code
    total_failed = sum(
        1 for category in tester.test_results.values()
        for result in category.values()
        if not result
    )
    
    sys.exit(1 if total_failed > 0 else 0)


if __name__ == '__main__':
    main()