"""
Comprehensive input/output logger for ML debugging
"""

import functools
import inspect
import json
import time
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Union

from .base import BaseDebugTool, ContextManager, format_tensor_info, get_memory_info


class IOLogger(BaseDebugTool):
    """Comprehensive input/output logger with filtering and analysis capabilities"""
    
    def __init__(
        self, 
        output_dir: Optional[Union[str, Path]] = None,
        verbose: bool = True,
        track_memory: bool = True,
        track_gradients: bool = True,
        max_tensor_elements: int = 10,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        capture_stack_trace: bool = False,
    ):
        super().__init__(output_dir, verbose)
        self.track_memory = track_memory
        self.track_gradients = track_gradients
        self.max_tensor_elements = max_tensor_elements
        self.include_patterns = include_patterns or []
        self.exclude_patterns = exclude_patterns or ['__', 'torch.nn.functional']
        self.capture_stack_trace = capture_stack_trace
        
        self.call_log: List[Dict[str, Any]] = []
        self.stats = defaultdict(int)
        self.timing_stats = defaultdict(list)
        self.wrapped_functions: Set[Callable] = set()
        self._lock = threading.Lock()
        
    def enable(self) -> None:
        """Enable I/O logging"""
        self.enabled = True
        if self.verbose:
            self.logger.info("I/O logging enabled")
            
    def disable(self) -> None:
        """Disable I/O logging and save results"""
        self.enabled = False
        self._save_logs()
        if self.verbose:
            self.logger.info("I/O logging disabled and logs saved")
    
    def should_log_function(self, func_name: str) -> bool:
        """Determine if function should be logged based on patterns"""
        if self.include_patterns:
            if not any(pattern in func_name for pattern in self.include_patterns):
                return False
        
        if self.exclude_patterns:
            if any(pattern in func_name for pattern in self.exclude_patterns):
                return False
                
        return True
    
    def wrap_function(self, func: Callable, module_name: str = "") -> Callable:
        """Wrap a function to log its I/O"""
        if func in self.wrapped_functions:
            return func
            
        func_name = f"{module_name}.{func.__name__}" if module_name else func.__name__
        
        if not self.should_log_function(func_name):
            return func
            
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not self.enabled:
                return func(*args, **kwargs)
                
            call_id = f"{func_name}_{int(time.time() * 1000000)}"
            start_time = time.time()
            
            # Capture input information
            input_info = self._capture_inputs(args, kwargs)
            
            # Capture memory before
            memory_before = get_memory_info() if self.track_memory else None
            
            # Capture stack trace if requested
            stack_trace = None
            if self.capture_stack_trace:
                stack_trace = inspect.stack()[1:6]  # Skip wrapper frame
            
            try:
                # Execute function
                result = func(*args, **kwargs)
                
                # Capture output information
                output_info = self._capture_outputs(result)
                
                # Capture memory after
                memory_after = get_memory_info() if self.track_memory else None
                
                # Calculate timing
                execution_time = time.time() - start_time
                
                # Log the call
                with self._lock:
                    call_entry = {
                        "call_id": call_id,
                        "function": func_name,
                        "timestamp": start_time,
                        "execution_time": execution_time,
                        "inputs": input_info,
                        "outputs": output_info,
                        "memory_before": memory_before,
                        "memory_after": memory_after,
                        "stack_trace": [
                            {
                                "filename": frame.filename,
                                "lineno": frame.lineno,
                                "function": frame.function,
                                "code_context": frame.code_context[0].strip() if frame.code_context else None
                            }
                            for frame in stack_trace
                        ] if stack_trace else None,
                        "success": True
                    }
                    
                    self.call_log.append(call_entry)
                    self.stats[func_name] += 1
                    self.timing_stats[func_name].append(execution_time)
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                
                with self._lock:
                    call_entry = {
                        "call_id": call_id,
                        "function": func_name,
                        "timestamp": start_time,
                        "execution_time": execution_time,
                        "inputs": input_info,
                        "outputs": None,
                        "error": {
                            "type": type(e).__name__,
                            "message": str(e),
                            "args": e.args
                        },
                        "memory_before": memory_before,
                        "memory_after": get_memory_info() if self.track_memory else None,
                        "stack_trace": [
                            {
                                "filename": frame.filename,
                                "lineno": frame.lineno,
                                "function": frame.function,
                                "code_context": frame.code_context[0].strip() if frame.code_context else None
                            }
                            for frame in stack_trace
                        ] if stack_trace else None,
                        "success": False
                    }
                    
                    self.call_log.append(call_entry)
                    self.stats[f"{func_name}_errors"] += 1
                
                raise
        
        self.wrapped_functions.add(func)
        return wrapper
    
    def _capture_inputs(self, args: tuple, kwargs: dict) -> Dict[str, Any]:
        """Capture and format input arguments"""
        input_info = {
            "args": [],
            "kwargs": {},
            "summary": {
                "num_args": len(args),
                "num_kwargs": len(kwargs),
                "total_tensors": 0,
                "total_parameters": 0
            }
        }
        
        # Process positional arguments
        for i, arg in enumerate(args):
            arg_info = self._format_argument(arg, f"arg_{i}")
            input_info["args"].append(arg_info)
            if arg_info.get("is_tensor"):
                input_info["summary"]["total_tensors"] += 1
            if arg_info.get("parameter_count"):
                input_info["summary"]["total_parameters"] += arg_info["parameter_count"]
        
        # Process keyword arguments
        for key, value in kwargs.items():
            arg_info = self._format_argument(value, key)
            input_info["kwargs"][key] = arg_info
            if arg_info.get("is_tensor"):
                input_info["summary"]["total_tensors"] += 1
            if arg_info.get("parameter_count"):
                input_info["summary"]["total_parameters"] += arg_info["parameter_count"]
        
        return input_info
    
    def _capture_outputs(self, result: Any) -> Dict[str, Any]:
        """Capture and format output results"""
        if isinstance(result, (list, tuple)):
            return {
                "type": "sequence",
                "length": len(result),
                "elements": [self._format_argument(item, f"item_{i}") for i, item in enumerate(result[:5])]  # Limit to first 5
            }
        else:
            return self._format_argument(result, "result")
    
    def _format_argument(self, arg: Any, name: str) -> Dict[str, Any]:
        """Format a single argument for logging"""
        try:
            # Handle None
            if arg is None:
                return {"name": name, "type": "NoneType", "value": None}
            
            # Handle basic types
            if isinstance(arg, (int, float, str, bool)):
                return {
                    "name": name,
                    "type": type(arg).__name__,
                    "value": arg if len(str(arg)) <= 100 else f"{str(arg)[:100]}..."
                }
            
            # Handle tensors
            tensor_info = format_tensor_info(arg)
            if tensor_info.get("type") in ["torch.Tensor", "numpy.ndarray"]:
                result = {
                    "name": name,
                    "is_tensor": True,
                    **tensor_info
                }
                
                # Add sample values if tensor is small enough
                try:
                    if hasattr(arg, 'numel') and arg.numel() <= self.max_tensor_elements:
                        result["values"] = arg.detach().cpu().numpy().tolist() if hasattr(arg, 'detach') else arg.tolist()
                    elif hasattr(arg, 'size') and arg.size <= self.max_tensor_elements:
                        result["values"] = arg.tolist()
                except:
                    pass
                
                # Add gradient information if tracking
                if self.track_gradients and hasattr(arg, 'grad') and arg.grad is not None:
                    result["grad_info"] = format_tensor_info(arg.grad)
                
                return result
            
            # Handle PyTorch modules
            try:
                import torch.nn as nn
                if isinstance(arg, nn.Module):
                    param_count = sum(p.numel() for p in arg.parameters())
                    return {
                        "name": name,
                        "type": "torch.nn.Module",
                        "module_type": type(arg).__name__,
                        "parameter_count": param_count,
                        "trainable_params": sum(p.numel() for p in arg.parameters() if p.requires_grad),
                        "training_mode": arg.training
                    }
            except ImportError:
                pass
            
            # Handle collections
            if isinstance(arg, (list, tuple, set)):
                return {
                    "name": name,
                    "type": type(arg).__name__,
                    "length": len(arg),
                    "sample_elements": [self._format_argument(item, f"{name}[{i}]") for i, item in enumerate(list(arg)[:3])]
                }
            
            if isinstance(arg, dict):
                return {
                    "name": name,
                    "type": "dict",
                    "length": len(arg),
                    "keys": list(arg.keys())[:10],  # First 10 keys
                    "sample_items": {k: self._format_argument(v, f"{name}[{k}]") for k, v in list(arg.items())[:3]}
                }
            
            # Fallback for other types
            return {
                "name": name,
                "type": type(arg).__name__,
                "repr": repr(arg)[:200],
                "str": str(arg)[:200]
            }
            
        except Exception as e:
            return {
                "name": name,
                "type": "ERROR",
                "error": f"Failed to format: {str(e)}",
                "raw_type": str(type(arg))
            }
    
    def _save_logs(self) -> None:
        """Save all collected logs to files"""
        if not self.call_log:
            return
        
        # Save detailed call log
        self.save_json(self.call_log, "detailed_call_log.json")
        
        # Save summary statistics
        summary_stats = {
            "total_calls": len(self.call_log),
            "unique_functions": len(self.stats),
            "function_call_counts": dict(self.stats),
            "timing_statistics": {
                func: {
                    "count": len(times),
                    "total_time": sum(times),
                    "avg_time": sum(times) / len(times),
                    "min_time": min(times),
                    "max_time": max(times)
                }
                for func, times in self.timing_stats.items()
            },
            "error_summary": {
                k: v for k, v in self.stats.items() if k.endswith("_errors")
            }
        }
        self.save_json(summary_stats, "logging_summary.json")
        
        # Save memory analysis if enabled
        if self.track_memory:
            memory_analysis = self._analyze_memory_usage()
            self.save_json(memory_analysis, "memory_analysis.json")
        
        if self.verbose:
            self.logger.info(f"Saved {len(self.call_log)} function calls to {self.output_dir}")
    
    def _analyze_memory_usage(self) -> Dict[str, Any]:
        """Analyze memory usage patterns from logged calls"""
        memory_deltas = []
        gpu_memory_deltas = []
        
        for call in self.call_log:
            if call.get("memory_before") and call.get("memory_after"):
                before = call["memory_before"]["cpu_memory"]["available"]
                after = call["memory_after"]["cpu_memory"]["available"]
                memory_deltas.append({
                    "function": call["function"],
                    "delta": before - after,  # Positive means memory was consumed
                    "execution_time": call["execution_time"]
                })
                
                # GPU memory analysis
                if "gpu_memory" in call["memory_before"]:
                    for device, gpu_before in call["memory_before"]["gpu_memory"].items():
                        gpu_after = call["memory_after"]["gpu_memory"][device]
                        gpu_memory_deltas.append({
                            "function": call["function"],
                            "device": device,
                            "allocated_delta": gpu_after["allocated"] - gpu_before["allocated"],
                            "reserved_delta": gpu_after["reserved"] - gpu_before["reserved"]
                        })
        
        return {
            "cpu_memory_analysis": {
                "total_calls_with_memory_data": len(memory_deltas),
                "high_memory_functions": sorted(
                    [d for d in memory_deltas if d["delta"] > 1024*1024*100],  # >100MB
                    key=lambda x: x["delta"], reverse=True
                )[:10],
                "memory_leaks": [d for d in memory_deltas if d["delta"] > 0]  # Memory not freed
            },
            "gpu_memory_analysis": {
                "total_calls_with_gpu_data": len(gpu_memory_deltas),
                "high_gpu_memory_functions": sorted(
                    [d for d in gpu_memory_deltas if d["allocated_delta"] > 1024*1024*50],  # >50MB
                    key=lambda x: x["allocated_delta"], reverse=True
                )[:10]
            }
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current logging statistics"""
        return {
            "total_calls": len(self.call_log),
            "function_counts": dict(self.stats),
            "recent_calls": self.call_log[-10:] if self.call_log else []
        }
    
    def context(self, operation_name: str) -> ContextManager:
        """Create a context manager for logging an operation"""
        return IOLoggerContext(self, operation_name)


class IOLoggerContext(ContextManager):
    """Context manager for I/O logger operations"""
    
    def __init__(self, logger: IOLogger, operation_name: str):
        super().__init__(logger, operation_name)
        self.logger = logger
        
    def __enter__(self):
        if self.logger.enabled and self.logger.verbose:
            self.logger.logger.info(f"Starting operation: {self.operation_name}")
        return super().__enter__()
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        super().__exit__(exc_type, exc_val, exc_tb)
        if exc_type and self.logger.enabled:
            self.logger.logger.error(f"Operation {self.operation_name} failed: {exc_val}")