"""Base instruction classes and interfaces."""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Set, Sequence, Union, Any


class BaseInstruction(ABC):
    """Base class for all instruction checking classes."""

    def __init__(self, instruction_id: str):
        """Initialize a base instruction.
        
        Args:
            instruction_id: A string identifier for this instruction.
        """
        self.id = instruction_id

    @abstractmethod
    def build_description(self, **kwargs) -> str:
        """Build a human-readable description of this instruction.
        
        Args:
            **kwargs: Keyword arguments specific to the instruction type.
            
        Returns:
            A string description of the instruction.
        """
        pass

    @abstractmethod
    def get_instruction_args(self) -> Optional[Dict[str, Any]]:
        """Return the keyword arguments used in build_description.
        
        Returns:
            A dictionary of arguments or None if no arguments are needed.
        """
        pass

    @abstractmethod
    def get_instruction_args_keys(self) -> Sequence[str]:
        """Return the keys for arguments used in build_description.
        
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