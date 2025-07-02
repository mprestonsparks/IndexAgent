"""
IndexAgent Action Framework
Core actions that agents can execute using Claude Code CLI
"""

from .base import Action, ActionResult, ActionMetrics
from .implement_todos import ImplementTodosAction
from .improve_test_coverage import ImproveTestCoverageAction
from .refactor_complexity import RefactorComplexityAction

__all__ = [
    'Action',
    'ActionResult',
    'ActionMetrics',
    'ImplementTodosAction',
    'ImproveTestCoverageAction',
    'RefactorComplexityAction'
]