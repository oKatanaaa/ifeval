"""Base instruction classes and interfaces."""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Sequence, Union, Any


class BaseInstruction(ABC):
    """Base class for all instruction checking classes."""

    @abstractmethod
    def get_instruction_args(self) -> Optional[Dict[str, Any]]:
        """Return the keyword arguments needed for this instruction.
        
        Returns:
            A dictionary of arguments or None if no arguments are needed.
        """
        pass

    @abstractmethod
    def get_instruction_args_keys(self) -> Sequence[str]:
        """Return the keys for arguments needed for this instruction.
        
        Returns:
            A list of argument keys.
        """
        pass

    @abstractmethod
    def check_following(self, value: str) -> bool:
        """Check if a response follows this instruction.
        
        Args:
            value: A string representing the response to check.
            
        Returns:
            True if the instruction is followed, False otherwise.
        """
        pass