"""Prompt Evaluation Framework (PEF).

A small, dependency-light toolkit for comparing prompt versions against
a shared set of test cases using explicit, explainable scoring criteria.
"""

from .evaluator import PromptEvaluator
from .backends import MockBackend

__all__ = ["PromptEvaluator", "MockBackend"]

__version__ = "0.1.0"
