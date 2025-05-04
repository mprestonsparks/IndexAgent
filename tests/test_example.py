# tests/test_example.py

from src.example import add, subtract

def test_add():
    """Tests the add function."""
    assert add(1, 2) == 3
    assert add(-1, 1) == 0
    assert add(0, 0) == 0

def test_subtract():
    """Tests the subtract function."""
    assert subtract(5, 3) == 2
    assert subtract(10, 10) == 0
    assert subtract(0, 5) == -5