import subprocess
from pathlib import Path

def test_claude_help():
    """Tests that 'claude --help' returns usage info."""
    result = subprocess.run(
        ["claude", "--help"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert result.returncode == 0
    assert "Usage: claude" in result.stdout

def test_wrapper_contains_claude_invocation():
    """Tests that the maintenance script contains a claude invocation."""
    content = Path("scripts/maintenance/agent_fix_todos.sh").read_text()
    assert "claude -m" in content