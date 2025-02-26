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
    
    def register_conflict(self, instruction_id: str, conflicts_with: str) -> None:
        """Register a conflict between instructions.
        
        Args:
            instruction_id: The ID of the first instruction.
            conflicts_with: The ID of the instruction it conflicts with.
        """
        self._conflicts[instruction_id].add(conflicts_with)
        self._conflicts[conflicts_with].add(instruction_id)
    
    def get_conflicts(self, instruction_id: str) -> Set[str]:
        """Get all instructions that conflict with the given instruction.
        
        Args:
            instruction_id: The ID of the instruction to check.
            
        Returns:
            A set of instruction IDs that conflict with the given one.
        """
        return self._conflicts.get(instruction_id, set())
    
    def conflict_make(self) -> None:
        """Ensure conflicts are bidirectional and instructions conflict with themselves.
        
        For each instruction A that conflicts with B, ensure B conflicts with A.
        Also ensure each instruction conflicts with itself.
        """
        for key in self._instructions:
            for conflict_key in self._conflicts[key]:
                self._conflicts[conflict_key].add(key)
            self._conflicts[key].add(key)