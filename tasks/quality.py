"""Quality assurance tasks using invoke."""

import sys
from pathlib import Path
from invoke import task, Context
from typing import Optional


# ANSI color codes for output
class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


def print_header(message: str) -> None:
    """Print a formatted header."""
    print(f"\n{Colors.BLUE}{Colors.BOLD}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.BLUE}{Colors.BOLD}{message:^60}{Colors.RESET}")
    print(f"{Colors.BLUE}{Colors.BOLD}{'=' * 60}{Colors.RESET}\n")


def print_success(message: str) -> None:
    """Print a success message."""
    print(f"{Colors.GREEN}✓ {message}{Colors.RESET}")


def print_error(message: str) -> None:
    """Print an error message."""
    print(f"{Colors.RED}✗ {message}{Colors.RESET}")


def print_warning(message: str) -> None:
    """Print a warning message."""
    print(f"{Colors.YELLOW}⚠ {message}{Colors.RESET}")


@task
def black_check(c: Context) -> None:
    """Run black in check mode."""
    print_header("Running Black Code Formatter Check")
    
    result = c.run("black --check --diff .", warn=True)
    
    if result.exited == 0:
        print_success("Black check passed - all files are properly formatted")
    else:
        print_error("Black check failed - files need formatting")
        print_warning("Run 'black .' to automatically format files")
        sys.exit(1)


@task
def ruff_check(c: Context) -> None:
    """Run ruff linter."""
    print_header("Running Ruff Linter")
    
    result = c.run("ruff check .", warn=True)
    
    if result.exited == 0:
        print_success("Ruff check passed - no linting issues found")
    else:
        print_error(f"Ruff check failed - found {result.exited} issue(s)")
        print_warning("Fix the issues above or add appropriate ignore comments")
        sys.exit(1)


@task
def mypy(c: Context) -> None:
    """Run mypy type checker."""
    print_header("Running MyPy Type Checker")
    
    # Check if indexagent directory exists
    if not Path("indexagent").exists():
        print_warning("indexagent directory not found, skipping mypy")
        return
    
    result = c.run("mypy indexagent", warn=True)
    
    if result.exited == 0:
        print_success("MyPy check passed - no type errors found")
    else:
        print_error("MyPy check failed - type errors found")
        sys.exit(1)


@task
def pytest_unit(c: Context, verbose: bool = False) -> None:
    """Run unit tests with coverage."""
    print_header("Running Unit Tests with Coverage")
    
    cmd = "pytest tests/ -v --cov=indexagent --cov-report=term-missing"
    if not verbose:
        cmd += " -q"
    
    # Exclude integration tests
    cmd += " --ignore=tests/integration"
    
    result = c.run(cmd, warn=True)
    
    if result.exited == 0:
        print_success("Unit tests passed")
    else:
        print_error("Unit tests failed")
        sys.exit(1)


@task
def pytest_integration(c: Context, verbose: bool = False) -> None:
    """Run integration tests."""
    print_header("Running Integration Tests")
    
    # Check if integration tests directory exists
    if not Path("tests/integration").exists():
        print_warning("No integration tests found")
        return
    
    cmd = "pytest tests/integration/ -v"
    if not verbose:
        cmd += " -q"
    
    result = c.run(cmd, warn=True)
    
    if result.exited == 0:
        print_success("Integration tests passed")
    else:
        print_error("Integration tests failed")
        sys.exit(1)


@task
def pytest_all(c: Context, verbose: bool = False) -> None:
    """Run all tests and enforce ≥90% coverage."""
    print_header("Running All Tests with Coverage Requirements")
    
    cmd = "pytest tests/ -v --cov=indexagent --cov-report=term-missing --cov-fail-under=90"
    if not verbose:
        cmd += " -q"
    
    result = c.run(cmd, warn=True)
    
    if result.exited == 0:
        print_success("All tests passed with ≥90% coverage")
    else:
        if "Required test coverage of 90% not reached" in result.stderr:
            print_error("Coverage requirement not met (< 90%)")
        else:
            print_error("Tests failed")
        sys.exit(1)


@task
def all(c: Context, verbose: bool = False) -> None:
    """Run all quality checks."""
    print_header("Running All Quality Checks")
    
    # Track failures
    failures = []
    
    # Run each check
    checks = [
        ("Black", black_check),
        ("Ruff", ruff_check),
        ("MyPy", mypy),
        ("Tests", pytest_all),
    ]
    
    for name, check_func in checks:
        try:
            if name == "Tests":
                check_func(c, verbose=verbose)
            else:
                check_func(c)
        except SystemExit:
            failures.append(name)
    
    # Summary
    print_header("Quality Check Summary")
    
    if not failures:
        print_success("All quality checks passed! 🎉")
    else:
        print_error(f"Failed checks: {', '.join(failures)}")
        print_warning("Fix the issues above before committing")
        sys.exit(1)


@task
def coverage_report(c: Context) -> None:
    """Generate HTML coverage report."""
    print_header("Generating HTML Coverage Report")
    
    c.run("coverage html")
    print_success("Coverage report generated in htmlcov/")
    print_warning("Open htmlcov/index.html in a browser to view the report")


# Aliases for convenience
@task
def black(c: Context) -> None:
    """Format code with black."""
    print_header("Formatting Code with Black")
    c.run("black .")
    print_success("Code formatted successfully")


@task
def ruff(c: Context) -> None:
    """Fix auto-fixable ruff issues."""
    print_header("Fixing Ruff Issues")
    c.run("ruff check --fix .")
    print_success("Auto-fixable issues resolved")