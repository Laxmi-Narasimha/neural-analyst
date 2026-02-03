"""AI Data Adequacy Agent - Backend Application Package."""

__version__ = "1.0.0"
__author__ = "AI Data Adequacy Agent Team"
__description__ = "Comprehensive data validation system for AI assistants"

from .config import config
from .orchestrator import OrchestratorAgent

__all__ = ["config", "OrchestratorAgent"]
