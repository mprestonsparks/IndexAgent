# Scripts Directory

This directory contains utility scripts for operating and maintaining the Parallel Processing System.

## Overview

The scripts in this directory are organized into categories:
- **System Management**: Scripts for system checks and smoke testing
- **Documentation**: Scripts for generating and updating documentation
- **Maintenance**: Scripts for maintenance tasks and cleanup
- **Testing**: Scripts for test execution and coverage analysis

## Script Inventory

### System Management Scripts

#### `smoke_run.sh`
**Purpose**: Performs a smoke test of the parallel maintenance DAG to verify system functionality.

**Usage**:
```bash
./scripts/smoke_run.sh
```

**Required Permissions**: Execute permission, access to Airflow CLI

**Expected Output**:
- Pre-flight checks with colored status indicators
- DAG execution logs
- Post-run verification results
- Summary of test execution

**Environment Variables**:
- `PARALLEL_LIMIT`: Set to 10 by default, controls parallel execution limit
- `AIRFLOW_HOME`: Airflow home directory (uses default if not set)

---

#### `check_system.py`
**Purpose**: Comprehensive system readiness check that verifies all prerequisites for production deployment.

**Usage**:
```bash
./scripts/check_system.py
```

**Required Permissions**: Execute permission, read access to system information

**Expected Output**:
- Detailed check results for each component
- Color-coded status (PASS/FAIL/WARN)
- Summary with total passed/failed/warning counts
- Exit code 0 if ready, 1 if not ready

**Checks Performed**:
- Python version (≥3.10)
- Git version (≥2.20)
- Airflow installation and configuration
- Required Python packages
- Git worktree support
- Database connectivity
- System resources (disk, CPU, memory)
- Project file structure

---

### Documentation Scripts

#### `documentation/agent_write_docs.sh`
**Purpose**: Automated documentation generation using AI assistance.

**Usage**:
```bash
./scripts/documentation/agent_write_docs.sh [module_name]
```

**Required Permissions**: Execute permission, write access to docs directory

**Expected Output**:
- Generated documentation files
- Update status messages

---

#### `documentation/find_undocumented.py`
**Purpose**: Identifies Python modules that lack documentation.

**Usage**:
```bash
python scripts/documentation/find_undocumented.py
```

**Required Permissions**: Read access to source files

**Expected Output**:
- List of undocumented modules
- JSON report in `reports/undoc.json`

---

#### `documentation/update_docs_coverage.sh`
**Purpose**: Updates documentation coverage metrics.

**Usage**:
```bash
./scripts/documentation/update_docs_coverage.sh
```

**Required Permissions**: Execute permission, write access to metrics files

**Expected Output**:
- Updated coverage metrics
- Coverage report

---

### Maintenance Scripts

#### `maintenance/agent_fix_todos.sh`
**Purpose**: Automated TODO comment resolution using AI assistance.

**Usage**:
```bash
./scripts/maintenance/agent_fix_todos.sh
```

**Required Permissions**: Execute permission, write access to source files

**Expected Output**:
- List of TODOs found
- Resolution status for each TODO
- Updated source files

---

#### `maintenance/baseline_audit.sh`
**Purpose**: Performs baseline audit of the codebase.

**Usage**:
```bash
./scripts/maintenance/baseline_audit.sh
```

**Required Permissions**: Execute permission, read access to all source files

**Expected Output**:
- Audit report with findings
- Recommendations for improvements

---

### Testing Scripts

#### `testing/ai_test_loop.sh`
**Purpose**: Runs AI-assisted test generation and execution loop.

**Usage**:
```bash
./scripts/testing/ai_test_loop.sh [test_target]
```

**Required Permissions**: Execute permission, write access for test generation

**Expected Output**:
- Generated test files
- Test execution results
- Coverage improvements

---

#### `testing/run_cov.py`
**Purpose**: Executes tests with coverage analysis.

**Usage**:
```bash
python scripts/testing/run_cov.py [options]
```

**Required Permissions**: Execute permission, pytest installed

**Expected Output**:
- Test execution results
- Coverage report (HTML and terminal)
- Coverage metrics

**Options**:
- `--html`: Generate HTML coverage report
- `--threshold`: Set minimum coverage threshold
- `--module`: Target specific module

---

#### `test_sanity_checks.sh`
**Purpose**: Quick sanity checks for the codebase.

**Usage**:
```bash
./scripts/test_sanity_checks.sh
```

**Required Permissions**: Execute permission

**Expected Output**:
- Import check results
- Syntax validation
- Basic functionality verification

---

## Best Practices

1. **Always run `check_system.py` before production deployment**
   - Ensures all prerequisites are met
   - Identifies potential issues early

2. **Use `smoke_run.sh` after any DAG changes**
   - Validates DAG functionality
   - Catches configuration errors

3. **Script Execution Order for New Deployments**:
   ```bash
   # 1. Check system readiness
   ./scripts/check_system.py
   
   # 2. Run smoke test
   ./scripts/smoke_run.sh
   
   # 3. Verify coverage
   python scripts/testing/run_cov.py
   ```

4. **Maintenance Schedule**:
   - Daily: Run `smoke_run.sh` in development
   - Weekly: Run `maintenance/baseline_audit.sh`
   - Before releases: Full system check and documentation update

## Troubleshooting

### Common Issues

1. **Permission Denied**
   ```bash
   # Make script executable
   chmod +x scripts/script_name.sh
   ```

2. **Module Not Found**
   ```bash
   # Ensure you're in project root
   cd /path/to/IndexAgent
   
   # Install dependencies
   pip install -r requirements-dev.txt
   ```

3. **Airflow Not Found**
   ```bash
   # Check Airflow installation
   which airflow
   
   # Install if needed
   pip install apache-airflow
   ```

## Adding New Scripts

When adding new scripts to this directory:

1. **Follow naming conventions**:
   - Use descriptive names
   - Use `.sh` for bash scripts
   - Use `.py` for Python scripts

2. **Add documentation**:
   - Update this README
   - Include docstring/comments in script
   - Add usage examples

3. **Set permissions**:
   ```bash
   chmod +x scripts/new_script.sh
   ```

4. **Test thoroughly**:
   - Test with various inputs
   - Handle errors gracefully
   - Provide helpful error messages

## Environment Variables

Scripts may use these environment variables:

- `AIRFLOW_HOME`: Airflow installation directory
- `PARALLEL_LIMIT`: Maximum parallel tasks
- `PROJECT_ROOT`: Override project root detection
- `DEBUG`: Enable debug output (set to any value)

## Security Considerations

1. **Never commit sensitive data** in scripts
2. **Use environment variables** for credentials
3. **Validate all inputs** to prevent injection
4. **Set appropriate file permissions**
5. **Review scripts before execution** in production

## Support

For issues or questions about these scripts:
1. Check the troubleshooting section above
2. Review script comments and docstrings
3. Consult the main project documentation
4. Contact the development team