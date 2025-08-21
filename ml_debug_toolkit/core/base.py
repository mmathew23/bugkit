"""
Base classes and utilities for ML debugging tools
"""

import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union
import logging


class BaseDebugTool(ABC):
    """Base class for all debugging tools"""
    
    def __init__(self, output_dir: Optional[Union[str, Path]] = None, verbose: bool = True):
        self.output_dir = Path(output_dir) if output_dir else Path("debug_output")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        self.enabled = False
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"{self.__class__.__name__}")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '[%(name)s] %(asctime)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO if self.verbose else logging.WARNING)
        return logger
    
    @abstractmethod
    def enable(self) -> None:
        """Enable the debugging tool"""
        pass
    
    @abstractmethod
    def disable(self) -> None:
        """Disable the debugging tool"""
        pass
    
    def save_json(self, data: Any, filename: str) -> Path:
        """Save data as JSON to output directory"""
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        return filepath
    
    def load_json(self, filename: str) -> Any:
        """Load JSON data from output directory"""
        filepath = self.output_dir / filename
        with open(filepath, 'r') as f:
            return json.load(f)


class ContextManager:
    """Base context manager for debugging operations"""
    
    def __init__(self, tool: BaseDebugTool, operation_name: str):
        self.tool = tool
        self.operation_name = operation_name
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        if self.tool.verbose:
            self.tool.logger.info(f"{self.operation_name} completed in {duration:.4f}s")


def format_tensor_info(tensor) -> Dict[str, Any]:
    """Format tensor information for logging"""
    try:
        import torch
        if isinstance(tensor, torch.Tensor):
            return {
                "type": "torch.Tensor",
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "device": str(tensor.device),
                "requires_grad": tensor.requires_grad,
                "is_leaf": tensor.is_leaf,
                "grad_fn": str(tensor.grad_fn) if tensor.grad_fn else None,
                "memory_format": str(tensor.memory_format()) if hasattr(tensor, 'memory_format') else None,
                "numel": tensor.numel(),
                "element_size": tensor.element_size(),
                "storage_size": tensor.storage().size() if tensor.storage() else 0,
            }
    except ImportError:
        pass
    
    try:
        import numpy as np
        if isinstance(tensor, np.ndarray):
            return {
                "type": "numpy.ndarray",
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "size": tensor.size,
                "itemsize": tensor.itemsize,
                "nbytes": tensor.nbytes,
            }
    except ImportError:
        pass
    
    return {"type": str(type(tensor)), "repr": repr(tensor)[:100]}


def get_memory_info() -> Dict[str, Any]:
    """Get current memory information"""
    import psutil
    
    memory_info = {
        "cpu_memory": {
            "total": psutil.virtual_memory().total,
            "available": psutil.virtual_memory().available,
            "percent": psutil.virtual_memory().percent,
        }
    }
    
    try:
        import torch
        if torch.cuda.is_available():
            memory_info["gpu_memory"] = {}
            for i in range(torch.cuda.device_count()):
                memory_info["gpu_memory"][f"device_{i}"] = {
                    "allocated": torch.cuda.memory_allocated(i),
                    "reserved": torch.cuda.memory_reserved(i),
                    "max_allocated": torch.cuda.max_memory_allocated(i),
                    "max_reserved": torch.cuda.max_memory_reserved(i),
                }
    except ImportError:
        pass
    
    return memory_info