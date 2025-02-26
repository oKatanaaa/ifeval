"""Configuration utilities for instruction following evaluation."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class Config:
    """Configuration for instruction following evaluation."""
    
    # Evaluation settings
    strict_mode: bool = True
    
    # Input/output settings
    input_data_path: Optional[str] = None
    input_response_path: Optional[str] = None
    output_dir: Optional[str] = None
    
    # Language settings
    language: str = "en"
    
    # Logging settings
    verbose: bool = False
    
    # Extra settings
    extra: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create a config from a dictionary.
        
        Args:
            config_dict: Dictionary containing configuration values.
            
        Returns:
            A Config object.
        """
        extra = {}
        recognized_fields = {field.name for field in cls.__dataclass_fields__.values()}
        
        for key, value in config_dict.items():
            if key not in recognized_fields:
                extra[key] = value
        
        config_dict_clean = {k: v for k, v in config_dict.items() if k in recognized_fields}
        config = cls(**config_dict_clean)
        config.extra = extra
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the config to a dictionary.
        
        Returns:
            Dictionary representation of the config.
        """
        result = {}
        for field_name in self.__dataclass_fields__:
            if field_name != 'extra':
                result[field_name] = getattr(self, field_name)
        
        # Add extra fields
        result.update(self.extra)
        
        return result
    
    def __getitem__(self, key: str) -> Any:
        """Get a config value.
        
        Args:
            key: The config key.
            
        Returns:
            The config value.
            
        Raises:
            KeyError: If the key is not in the config.
        """
        if hasattr(self, key):
            return getattr(self, key)
        elif key in self.extra:
            return self.extra[key]
        else:
            raise KeyError(f"Config has no key '{key}'")
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Set a config value.
        
        Args:
            key: The config key.
            value: The config value.
        """
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            self.extra[key] = value