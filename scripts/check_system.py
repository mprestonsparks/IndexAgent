#!/usr/bin/env python3
"""
System readiness check for the Parallel Processing System.
Verifies all prerequisites are met before running production workloads.
"""

import sys
import subprocess
import os
import sqlite3
import importlib.util
from pathlib import Path
from typing import Tuple, List, Dict, Any

# ANSI color codes
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

class SystemChecker:
    def __init__(self):
        self.checks_passed = 0
        self.checks_failed = 0
        self.warnings = 0
        self.results: List[Dict[str, Any]] = []
        
    def print_header(self, text: str) -> None:
        """Print a section header."""
        print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.ENDC}")
        print(f"{Colors.BOLD}{Colors.BLUE}{text:^60}{Colors.ENDC}")
        print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.ENDC}\n")
        
    def print_result(self, check: str, status: str, message: str = "", details: str = "") -> None:
        """Print a check result with color coding."""
        if status == "PASS":
            color = Colors.GREEN
            symbol = "✓"
            self.checks_passed += 1
        elif status == "FAIL":
            color = Colors.RED
            symbol = "✗"
            self.checks_failed += 1
        elif status == "WARN":
            color = Colors.YELLOW
            symbol = "!"
            self.warnings += 1
        else:
            color = ""
            symbol = "•"
            
        print(f"{color}{symbol} {check:<40} [{status}]{Colors.ENDC}")
        if message:
            print(f"  {message}")
        if details:
            print(f"  {Colors.BLUE}Details: {details}{Colors.ENDC}")
            
        self.results.append({
            "check": check,
            "status": status,
            "message": message,
            "details": details
        })
        
    def check_command_version(self, command: str, version_flag: str = "--version", 
                            min_version: str = None, extract_version_fn=None) -> Tuple[bool, str]:
        """Check if a command exists and optionally verify its version."""
        try:
            result = subprocess.run([command, version_flag], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                return False, f"Command failed: {result.stderr}"
                
            version_output = result.stdout.strip()
            
            if min_version and extract_version_fn:
                try:
                    current_version = extract_version_fn(version_output)
                    if self.compare_versions(current_version, min_version) < 0:
                        return False, f"Version {current_version} is below minimum {min_version}"
                    return True, f"Version {current_version}"
                except Exception as e:
                    return True, f"Version check failed: {e}, but command exists"
                    
            return True, version_output.split('\n')[0]
            
        except FileNotFoundError:
            return False, "Command not found"
        except subprocess.TimeoutExpired:
            return False, "Command timed out"
        except Exception as e:
            return False, f"Error: {str(e)}"
            
    def compare_versions(self, v1: str, v2: str) -> int:
        """Compare two version strings. Returns -1 if v1 < v2, 0 if equal, 1 if v1 > v2."""
        def normalize(v):
            parts = v.split('.')
            return [int(x) for x in parts[:3]]  # Compare major.minor.patch only
            
        v1_parts = normalize(v1)
        v2_parts = normalize(v2)
        
        for i in range(max(len(v1_parts), len(v2_parts))):
            v1_part = v1_parts[i] if i < len(v1_parts) else 0
            v2_part = v2_parts[i] if i < len(v2_parts) else 0
            
            if v1_part < v2_part:
                return -1
            elif v1_part > v2_part:
                return 1
                
        return 0
        
    def check_git_version(self) -> None:
        """Check Git installation and version."""
        def extract_git_version(output: str) -> str:
            # Extract version from "git version 2.34.1"
            parts = output.split()
            if len(parts) >= 3:
                return parts[2]
            return "0.0.0"
            
        success, message = self.check_command_version(
            "git", "--version", "2.20.0", extract_git_version
        )
        
        if success:
            self.print_result("Git version", "PASS", message)
        else:
            self.print_result("Git version", "FAIL", message, 
                            "Git >= 2.20 is required for worktree support")
                            
    def check_python_version(self) -> None:
        """Check Python version."""
        version = sys.version.split()[0]
        major, minor = sys.version_info[:2]
        
        if major >= 3 and minor >= 10:
            self.print_result("Python version", "PASS", f"Python {version}")
        else:
            self.print_result("Python version", "FAIL", 
                            f"Python {version} (requires >= 3.10)")
                            
    def check_airflow(self) -> None:
        """Check Airflow installation and configuration."""
        # Check if Airflow is installed
        success, message = self.check_command_version("airflow", "version")
        
        if not success:
            self.print_result("Airflow installation", "FAIL", message)
            return
            
        self.print_result("Airflow installation", "PASS", message)
        
        # Check AIRFLOW_HOME
        airflow_home = os.environ.get("AIRFLOW_HOME", os.path.expanduser("~/airflow"))
        if os.path.exists(airflow_home):
            self.print_result("AIRFLOW_HOME", "PASS", airflow_home)
        else:
            self.print_result("AIRFLOW_HOME", "WARN", 
                            f"Directory not found: {airflow_home}",
                            "Run 'airflow db init' to initialize")
                            
        # Check if database is initialized
        try:
            result = subprocess.run(["airflow", "db", "check"], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                self.print_result("Airflow database", "PASS", "Database is initialized")
            else:
                self.print_result("Airflow database", "FAIL", 
                                "Database not initialized",
                                "Run 'airflow db init'")
        except Exception as e:
            self.print_result("Airflow database", "FAIL", f"Check failed: {e}")
            
    def check_python_packages(self) -> None:
        """Check required Python packages."""
        required_packages = [
            ("airflow", "apache-airflow"),
            ("sqlite3", None),  # Built-in
            ("git", "GitPython"),
            ("invoke", "invoke"),
            ("pytest", "pytest"),
        ]
        
        for import_name, pip_name in required_packages:
            spec = importlib.util.find_spec(import_name)
            if spec is not None:
                self.print_result(f"Python package: {import_name}", "PASS")
            else:
                install_cmd = f"pip install {pip_name}" if pip_name else "Built-in module"
                self.print_result(f"Python package: {import_name}", "FAIL", 
                                f"Not installed", install_cmd)
                                
    def check_worktree_support(self) -> None:
        """Verify Git worktree functionality."""
        try:
            # Check if we're in a git repository
            result = subprocess.run(["git", "rev-parse", "--git-dir"], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                self.print_result("Git repository", "FAIL", 
                                "Not in a Git repository")
                return
                                
            self.print_result("Git repository", "PASS", "Valid Git repository found")
            
            # Test worktree functionality
            test_worktree = "/tmp/system_check_worktree_test"
            try:
                # Remove if exists
                subprocess.run(["git", "worktree", "remove", test_worktree, "--force"], 
                             capture_output=True)
                             
                # Try to create a worktree
                result = subprocess.run(["git", "worktree", "add", test_worktree, "HEAD"], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    self.print_result("Git worktree support", "PASS", 
                                    "Successfully created test worktree")
                    # Clean up
                    subprocess.run(["git", "worktree", "remove", test_worktree], 
                                 capture_output=True)
                else:
                    self.print_result("Git worktree support", "FAIL", 
                                    f"Failed to create worktree: {result.stderr}")
            except Exception as e:
                self.print_result("Git worktree support", "FAIL", f"Error: {e}")
                
        except Exception as e:
            self.print_result("Git repository check", "FAIL", f"Error: {e}")
            
    def check_database_connectivity(self) -> None:
        """Test database connectivity and operations."""
        db_path = Path("shard_status.db")
        test_db_path = Path("test_shard_status.db")
        
        try:
            # Test creating a database
            conn = sqlite3.connect(str(test_db_path))
            cursor = conn.cursor()
            
            # Create test table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS test_table (
                    id INTEGER PRIMARY KEY,
                    test_value TEXT
                )
            """)
            
            # Test insert
            cursor.execute("INSERT INTO test_table (test_value) VALUES (?)", ("test",))
            
            # Test select
            cursor.execute("SELECT COUNT(*) FROM test_table")
            count = cursor.fetchone()[0]
            
            conn.close()
            test_db_path.unlink()  # Clean up
            
            self.print_result("SQLite functionality", "PASS", 
                            "Database operations successful")
                            
            # Check if production database exists
            if db_path.exists():
                self.print_result("Production database", "PASS", 
                                f"Found at {db_path}")
                                
                # Check database integrity
                conn = sqlite3.connect(str(db_path))
                cursor = conn.cursor()
                cursor.execute("PRAGMA integrity_check")
                result = cursor.fetchone()[0]
                conn.close()
                
                if result == "ok":
                    self.print_result("Database integrity", "PASS", 
                                    "Database is healthy")
                else:
                    self.print_result("Database integrity", "WARN", 
                                    f"Issues found: {result}")
            else:
                self.print_result("Production database", "WARN", 
                                "Not found (will be created on first run)")
                                
        except Exception as e:
            self.print_result("Database connectivity", "FAIL", f"Error: {e}")
            
    def check_system_resources(self) -> None:
        """Check system resources."""
        try:
            # Check disk space in /tmp
            result = subprocess.run(["df", "-h", "/tmp"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) >= 2:
                    # Parse the df output
                    parts = lines[1].split()
                    if len(parts) >= 4:
                        available = parts[3]
                        use_percent = parts[4].rstrip('%')
                        
                        try:
                            if int(use_percent) > 90:
                                self.print_result("/tmp disk space", "FAIL", 
                                                f"Only {available} available ({use_percent}% used)")
                            elif int(use_percent) > 80:
                                self.print_result("/tmp disk space", "WARN", 
                                                f"{available} available ({use_percent}% used)")
                            else:
                                self.print_result("/tmp disk space", "PASS", 
                                                f"{available} available ({use_percent}% used)")
                        except ValueError:
                            self.print_result("/tmp disk space", "WARN", 
                                            "Could not parse disk usage")
                                            
            # Check CPU count
            cpu_count = os.cpu_count()
            if cpu_count:
                if cpu_count < 4:
                    self.print_result("CPU cores", "WARN", 
                                    f"{cpu_count} cores (recommended: 4+)")
                else:
                    self.print_result("CPU cores", "PASS", f"{cpu_count} cores")
                    
            # Check memory (Linux/Mac)
            try:
                if sys.platform == "darwin":  # macOS
                    result = subprocess.run(["sysctl", "-n", "hw.memsize"], 
                                          capture_output=True, text=True)
                    if result.returncode == 0:
                        mem_bytes = int(result.stdout.strip())
                        mem_gb = mem_bytes / (1024**3)
                        if mem_gb < 8:
                            self.print_result("System memory", "WARN", 
                                            f"{mem_gb:.1f} GB (recommended: 8+ GB)")
                        else:
                            self.print_result("System memory", "PASS", f"{mem_gb:.1f} GB")
                else:  # Linux
                    with open("/proc/meminfo", "r") as f:
                        for line in f:
                            if line.startswith("MemTotal:"):
                                mem_kb = int(line.split()[1])
                                mem_gb = mem_kb / (1024**2)
                                if mem_gb < 8:
                                    self.print_result("System memory", "WARN", 
                                                    f"{mem_gb:.1f} GB (recommended: 8+ GB)")
                                else:
                                    self.print_result("System memory", "PASS", 
                                                    f"{mem_gb:.1f} GB")
                                break
            except Exception:
                self.print_result("System memory", "WARN", "Could not determine memory size")
                
        except Exception as e:
            self.print_result("System resources", "FAIL", f"Error checking resources: {e}")
            
    def check_project_structure(self) -> None:
        """Verify project structure and required files."""
        required_files = [
            ("dags/parallel_maintenance_poc.py", "Main DAG file"),
            ("indexagent/utils/worktree_manager.py", "Worktree manager module"),
            ("scripts/smoke_run.sh", "Smoke test script"),
            ("pyproject.toml", "Project configuration"),
        ]
        
        project_root = Path.cwd()
        
        for file_path, description in required_files:
            full_path = project_root / file_path
            if full_path.exists():
                self.print_result(f"File: {file_path}", "PASS", description)
            else:
                self.print_result(f"File: {file_path}", "FAIL", 
                                f"{description} not found")
                                
    def print_summary(self) -> None:
        """Print a summary of all checks."""
        self.print_header("SYSTEM CHECK SUMMARY")
        
        total_checks = self.checks_passed + self.checks_failed + self.warnings
        
        print(f"{Colors.GREEN}Passed: {self.checks_passed}{Colors.ENDC}")
        print(f"{Colors.YELLOW}Warnings: {self.warnings}{Colors.ENDC}")
        print(f"{Colors.RED}Failed: {self.checks_failed}{Colors.ENDC}")
        print(f"Total checks: {total_checks}")
        
        if self.checks_failed == 0:
            print(f"\n{Colors.GREEN}{Colors.BOLD}✓ System is ready for production!{Colors.ENDC}")
            return 0
        else:
            print(f"\n{Colors.RED}{Colors.BOLD}✗ System is not ready. Please fix the failed checks.{Colors.ENDC}")
            return 1
            
    def run_all_checks(self) -> int:
        """Run all system checks and return exit code."""
        self.print_header("PARALLEL PROCESSING SYSTEM CHECK")
        
        print("Checking system prerequisites...\n")
        
        # Run all checks
        self.check_python_version()
        self.check_git_version()
        self.check_airflow()
        self.check_python_packages()
        self.check_worktree_support()
        self.check_database_connectivity()
        self.check_system_resources()
        self.check_project_structure()
        
        # Print summary and return exit code
        return self.print_summary()


def main():
    """Main entry point."""
    checker = SystemChecker()
    sys.exit(checker.run_all_checks())


if __name__ == "__main__":
    main()