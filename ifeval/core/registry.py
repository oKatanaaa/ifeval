"""Registry for instruction classes."""

import collections
from typing import Dict, Set, Type, Callable, TypeVar, Optional

from ifeval.core.instructions import BaseInstruction

T = TypeVar('T', bound=BaseInstruction)


class InstructionRegistry:
    """Registry for instruction classes."""
    
    def __init__(self):
        """Initialize an empty registry."""
        self._instructions: Dict[str, Type[BaseInstruction]] = {}
        self._conflicts: Dict[str, Set[str]] = collections.defaultdict(set)
    
    def register(self, instruction_id: str) -> Callable[[Type[T]], Type[T]]:
        """Register an instruction class with the registry.
        
        Args:
            instruction_id: The ID to register the instruction under.
            
        Returns:
            A decorator function that registers the class.
        """
        def decorator(cls: Type[T]) -> Type[T]:
            self._instructions[instruction_id] = cls
            return cls
        return decorator
    
    def get_instruction(self, instruction_id: str) -> Type[BaseInstruction]:
        """Get an instruction class by ID.
        
        Args:
            instruction_id: The ID of the instruction to get.
            
        Returns:
            The instruction class.
            
        Raises:
            ValueError: If the instruction ID is not registered.
        """
        if instruction_id not in self._instructions:
            raise ValueError(f"Unknown instruction ID: {instruction_id}")
        return self._instructions[instruction_id]
        
    def create_instruction(self, instruction_id: str, **kwargs) -> BaseInstruction:
        """Create an instruction instance by ID with given parameters.
        
        Args:
            instruction_id: The ID of the instruction to create.
            **kwargs: Parameters to pass to the instruction constructor.
            
        Returns:
            An instance of the instruction class.
            
        Raises:
            ValueError: If the instruction ID is not registered.
        """
        if instruction_id not in self._instructions:
            raise ValueError(f"Unknown instruction ID: {instruction_id}")
        
        instruction_cls = self._instructions[instruction_id]
        return instruction_cls(**kwargs)