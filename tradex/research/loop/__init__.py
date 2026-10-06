"""The weekly research engine loop: propose -> implement -> screen -> walk-forward -> holdout -> paper queue -> health -> report.

Entry points: ``tradex research loop`` (see ``tradex.research.loop.cli``) and ``run_loop`` for tests and other callers.
"""
from tradex.research.loop.config import LoopConfig
from tradex.research.loop.engine import run_loop
from tradex.research.loop.ports import Ports
from tradex.research.loop.stages import STAGES
from tradex.research.loop.state import LoopState

__all__ = ["LoopConfig", "LoopState", "Ports", "STAGES", "run_loop"]
