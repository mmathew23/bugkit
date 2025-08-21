"""
Chrome trace format tracer for profiling ML workloads
"""

import json
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..core.base import BaseDebugTool


class ChromeTracer(BaseDebugTool):
    """Generate Chrome trace format files for performance analysis"""
    
    def __init__(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        verbose: bool = True,
        auto_save: bool = True,
        buffer_size: int = 10000,
        include_memory: bool = True,
        include_cuda: bool = True,
        sample_rate: float = 0.001,  # Sample every 1ms for counters
    ):
        super().__init__(output_dir, verbose)
        self.auto_save = auto_save
        self.buffer_size = buffer_size
        self.include_memory = include_memory
        self.include_cuda = include_cuda
        self.sample_rate = sample_rate
        
        self.events: List[Dict[str, Any]] = []
        self.event_stack: List[Dict[str, Any]] = []
        self.process_id = os.getpid()
        self.thread_local = threading.local()
        self.counter_thread: Optional[threading.Thread] = None
        self.counter_running = False
        self._lock = threading.Lock()
        
        # Initialize PyTorch hooks if available
        self.torch_hooks = []
        self._setup_torch_hooks()
        
        # Initialize CUDA hooks if available
        self.cuda_hooks = []
        self._setup_cuda_hooks()
    
    def enable(self) -> None:
        """Enable Chrome tracing"""
        self.enabled = True
        self._start_counter_thread()
        if self.verbose:
            self.logger.info("Chrome tracer enabled")
    
    def disable(self) -> None:
        """Disable Chrome tracing and save results"""
        self.enabled = False
        self._stop_counter_thread()
        self._cleanup_hooks()
        if self.auto_save:
            self.save_trace()
        if self.verbose:
            self.logger.info("Chrome tracer disabled")
    
    @contextmanager
    def trace(self, name: str, category: str = "default", args: Optional[Dict[str, Any]] = None):
        """Context manager for tracing a code block"""
        if not self.enabled:
            yield
            return
        
        thread_id = threading.get_ident()
        start_time = time.time_ns() // 1000  # Convert to microseconds
        
        # Start event
        start_event = {
            "name": name,
            "cat": category,
            "ph": "B",  # Begin
            "ts": start_time,
            "pid": self.process_id,
            "tid": thread_id,
            "args": args or {}
        }
        
        with self._lock:
            self.events.append(start_event)
            self.event_stack.append(start_event)
        
        try:
            yield
        finally:
            end_time = time.time_ns() // 1000
            
            # End event
            end_event = {
                "name": name,
                "cat": category,
                "ph": "E",  # End
                "ts": end_time,
                "pid": self.process_id,
                "tid": thread_id,
                "args": args or {}
            }
            
            with self._lock:
                self.events.append(end_event)
                if self.event_stack and self.event_stack[-1]["name"] == name:
                    self.event_stack.pop()
                
                # Auto-save if buffer is full
                if len(self.events) >= self.buffer_size and self.auto_save:
                    self._flush_events()
    
    def add_instant_event(
        self, 
        name: str, 
        category: str = "default", 
        args: Optional[Dict[str, Any]] = None,
        scope: str = "t"  # t=thread, p=process, g=global
    ) -> None:
        """Add an instant event (point in time)"""
        if not self.enabled:
            return
        
        event = {
            "name": name,
            "cat": category,
            "ph": "i",  # Instant
            "ts": time.time_ns() // 1000,
            "pid": self.process_id,
            "tid": threading.get_ident(),
            "s": scope,
            "args": args or {}
        }
        
        with self._lock:
            self.events.append(event)
    
    def add_counter_event(
        self, 
        name: str, 
        value: Union[float, Dict[str, float]], 
        category: str = "counters"
    ) -> None:
        """Add a counter event"""
        if not self.enabled:
            return
        
        if isinstance(value, dict):
            args = value
        else:
            args = {name: value}
        
        event = {
            "name": name,
            "cat": category,
            "ph": "C",  # Counter
            "ts": time.time_ns() // 1000,
            "pid": self.process_id,
            "args": args
        }
        
        with self._lock:
            self.events.append(event)
    
    def add_metadata_event(
        self, 
        name: str, 
        args: Dict[str, Any], 
        category: str = "metadata"
    ) -> None:
        """Add metadata event"""
        if not self.enabled:
            return
        
        event = {
            "name": name,
            "cat": category,
            "ph": "M",  # Metadata
            "ts": time.time_ns() // 1000,
            "pid": self.process_id,
            "tid": threading.get_ident(),
            "args": args
        }
        
        with self._lock:
            self.events.append(event)
    
    def _setup_torch_hooks(self) -> None:
        """Setup PyTorch profiling hooks"""
        try:
            import torch
            
            # Hook for CUDA kernel launches
            def cuda_kernel_hook(name, *args, **kwargs):
                if self.enabled and self.include_cuda:
                    self.add_instant_event(f"CUDA: {name}", "cuda_kernels")
            
            # Hook for memory allocations
            def memory_hook(device, alloc_size, stream_ptr):
                if self.enabled and self.include_memory:
                    self.add_counter_event(
                        "CUDA Memory Allocated",
                        torch.cuda.memory_allocated(device) / 1024 / 1024,  # MB
                        "memory"
                    )
            
            # Register hooks if CUDA is available
            if torch.cuda.is_available():
                # Note: This is a simplified version. Real implementation would use
                # torch.profiler or other official PyTorch profiling APIs
                pass
                
        except ImportError:
            if self.verbose:
                self.logger.info("PyTorch not available, skipping torch hooks")
    
    def _setup_cuda_hooks(self) -> None:
        """Setup CUDA profiling hooks"""
        try:
            # Try to import NVIDIA's profiling tools
            import pynvml
            pynvml.nvmlInit()
            
            def sample_gpu_metrics():
                if not self.enabled or not self.include_cuda:
                    return {}
                
                metrics = {}
                try:
                    device_count = pynvml.nvmlDeviceGetCount()
                    for i in range(device_count):
                        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                        
                        # GPU utilization
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        metrics[f"GPU_{i}_Utilization"] = util.gpu
                        metrics[f"GPU_{i}_Memory_Utilization"] = util.memory
                        
                        # Memory info
                        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        metrics[f"GPU_{i}_Memory_Used_MB"] = mem_info.used / 1024 / 1024
                        metrics[f"GPU_{i}_Memory_Free_MB"] = mem_info.free / 1024 / 1024
                        
                        # Temperature
                        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                        metrics[f"GPU_{i}_Temperature_C"] = temp
                        
                        # Power
                        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                        metrics[f"GPU_{i}_Power_W"] = power
                        
                except Exception:
                    pass
                
                return metrics
            
            self.sample_gpu_metrics = sample_gpu_metrics
            
        except ImportError:
            if self.verbose:
                self.logger.info("pynvml not available, skipping CUDA hooks")
            self.sample_gpu_metrics = lambda: {}
    
    def _start_counter_thread(self) -> None:
        """Start background thread for counter sampling"""
        if self.counter_thread is not None:
            return
        
        self.counter_running = True
        self.counter_thread = threading.Thread(target=self._counter_loop, daemon=True)
        self.counter_thread.start()
    
    def _stop_counter_thread(self) -> None:
        """Stop background counter thread"""
        self.counter_running = False
        if self.counter_thread:
            self.counter_thread.join(timeout=1.0)
            self.counter_thread = None
    
    def _counter_loop(self) -> None:
        """Background loop for sampling counters"""
        import psutil
        
        while self.counter_running:
            try:
                # CPU and memory metrics
                if self.include_memory:
                    cpu_percent = psutil.cpu_percent()
                    memory = psutil.virtual_memory()
                    
                    self.add_counter_event("CPU_Usage", cpu_percent, "system")
                    self.add_counter_event("Memory_Usage_MB", memory.used / 1024 / 1024, "system")
                    self.add_counter_event("Memory_Available_MB", memory.available / 1024 / 1024, "system")
                
                # GPU metrics
                if self.include_cuda:
                    gpu_metrics = self.sample_gpu_metrics()
                    for name, value in gpu_metrics.items():
                        self.add_counter_event(name, value, "gpu")
                
                time.sleep(self.sample_rate)
                
            except Exception as e:
                if self.verbose:
                    self.logger.error(f"Counter sampling error: {e}")
                time.sleep(1.0)  # Back off on error
    
    def _cleanup_hooks(self) -> None:
        """Clean up any registered hooks"""
        for hook in self.torch_hooks:
            try:
                hook.remove()
            except:
                pass
        self.torch_hooks.clear()
        
        for hook in self.cuda_hooks:
            try:
                hook.remove()
            except:
                pass
        self.cuda_hooks.clear()
    
    def _flush_events(self) -> None:
        """Flush events to disk"""
        if not self.events:
            return
        
        timestamp = int(time.time())
        filename = f"trace_buffer_{timestamp}.json"
        self.save_trace(filename)
        
        # Keep only recent events
        with self._lock:
            self.events = self.events[-1000:]  # Keep last 1000 events
    
    def save_trace(self, filename: Optional[str] = None) -> Path:
        """Save trace events to Chrome trace format file"""
        if filename is None:
            timestamp = int(time.time())
            filename = f"chrome_trace_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        # Add process metadata
        metadata_events = [
            {
                "name": "process_name",
                "ph": "M",
                "pid": self.process_id,
                "args": {"name": "ML Debug Session"}
            },
            {
                "name": "process_sort_index",
                "ph": "M", 
                "pid": self.process_id,
                "args": {"sort_index": 0}
            }
        ]
        
        # Create Chrome trace format
        trace_data = {
            "traceEvents": metadata_events + self.events,
            "displayTimeUnit": "ms",
            "metadata": {
                "chrome-trace-format": "https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview",
                "user-agent": "ml-debug-toolkit",
                "total_events": len(self.events)
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(trace_data, f, separators=(',', ':'))
        
        if self.verbose:
            self.logger.info(f"Saved {len(self.events)} trace events to {filepath}")
        
        return filepath
    
    def clear_events(self) -> None:
        """Clear all accumulated events"""
        with self._lock:
            self.events.clear()
            self.event_stack.clear()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current tracing statistics"""
        with self._lock:
            event_categories = {}
            for event in self.events:
                cat = event.get("cat", "unknown")
                event_categories[cat] = event_categories.get(cat, 0) + 1
            
            return {
                "total_events": len(self.events),
                "event_categories": event_categories,
                "stack_depth": len(self.event_stack),
                "buffer_usage": f"{len(self.events)}/{self.buffer_size}",
                "counter_thread_running": self.counter_running,
            }