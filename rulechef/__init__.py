"""RuleChef - Learn rule-based models from examples and LLM interactions"""

from rulechef.engine import RuleChef
from rulechef.core import (
    Task,
    Dataset,
    Example,
    Correction,
    Span,
    Rule,
)

__version__ = "0.1.0"
__all__ = [
    "RuleChef",
    "Task",
    "Dataset",
    "Example",
    "Correction",
    "Span",
    "Rule",
]
