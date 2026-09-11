"""Coordination layer between the scheduler and the per-robot NMPC.

See `coordinator.Coordinator`. Enabled per run with `general_funct(..., coordinator=True)`.
"""

from .config import CoordinatorConfig
from .coordinator import Coordinator

__all__ = ["Coordinator", "CoordinatorConfig"]
