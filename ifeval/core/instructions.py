"""Base instruction classes and interfaces."""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Sequence, Union, Any


class BaseInstruction(ABC):
    """Base class for all instruction checking classes."""
    @abstractmethod
    def check_following(self, value: str) -> bool:
        """Check if a response follows this instruction.
        
        Args:
            value: A string representing the response to check.
            
        Returns:
            True if the instruction is followed, False otherwise.
        """
        pass