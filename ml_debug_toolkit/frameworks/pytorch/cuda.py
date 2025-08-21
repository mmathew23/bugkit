"""
CUDA-specific debugging utilities for PyTorch
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch

from ...core.base import BaseDebugTool


class CUDADebugger(BaseDebugTool):
    """CUDA memory and kernel debugging utilities"""
    
    def __init__(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        verbose: bool = True,
        track_allocations: bool = True,
        profile_kernels: bool = True,
    ):
        super().__init__(output_dir, verbose)
        self.track_allocations = track_allocations
        self.profile_kernels = profile_kernels
        
        self.memory_snapshots: List[Dict[str, Any]] = []
        self.kernel_profiles: List[Dict[str, Any]] = []
        self.allocation_history: List[Dict[str, Any]] = []
        
        # Check CUDA availability
        if not torch.cuda.is_available():
            self.logger.warning("CUDA not available - CUDADebugger will have limited functionality")
    
    def enable(self) -> None:
        """Enable CUDA debugging"""
        self.enabled = True
        
        if torch.cuda.is_available() and self.track_allocations:
            try:
                # Enable memory history tracking - this API may not be available in all PyTorch versions
                if hasattr(torch.cuda.memory, '_record_memory_history'):
                    torch.cuda.memory._record_memory_history(
                        enabled=True,
                        alloc_trace_record_context=True,
                        alloc_trace_max_entries=100000,
                    )
                else:
                    self.logger.warning("Memory history tracking not available in this PyTorch version")
            except Exception as e:
                self.logger.warning(f"Failed to enable memory history tracking: {e}")
                self.track_allocations = False
        
        if self.verbose:
            self.logger.info("CUDA debugger enabled")
    
    def disable(self) -> None:
        """Disable CUDA debugging and save results"""
        self.enabled = False
        
        if torch.cuda.is_available() and self.track_allocations:
            try:
                if hasattr(torch.cuda.memory, '_record_memory_history'):
                    torch.cuda.memory._record_memory_history(enabled=False)
            except Exception as e:
                self.logger.warning(f"Failed to disable memory history tracking: {e}")
        
        self._save_debug_data()
        
        if self.verbose:
            self.logger.info("CUDA debugger disabled")
    
    def snapshot_memory(self, name: str = "snapshot") -> Dict[str, Any]:
        """Take a memory snapshot"""
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}
        
        snapshot = {
            "name": name,
            "timestamp": time.time(),
            "device_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device(),
            "memory_stats": {},
        }
        
        for device_id in range(torch.cuda.device_count()):
            device_stats = {
                "device_name": torch.cuda.get_device_name(device_id),
                "memory_allocated": torch.cuda.memory_allocated(device_id),
                "memory_reserved": torch.cuda.memory_reserved(device_id),
                "max_memory_allocated": torch.cuda.max_memory_allocated(device_id),
                "max_memory_reserved": torch.cuda.max_memory_reserved(device_id),
                "memory_stats": torch.cuda.memory_stats(device_id),
            }
            
            # Calculate memory utilization
            total_memory = torch.cuda.get_device_properties(device_id).total_memory
            device_stats["total_memory"] = total_memory
            device_stats["memory_utilization"] = device_stats["memory_allocated"] / total_memory
            device_stats["reserved_utilization"] = device_stats["memory_reserved"] / total_memory
            
            snapshot["memory_stats"][f"device_{device_id}"] = device_stats
        
        self.memory_snapshots.append(snapshot)
        
        if self.verbose:
            current_device = snapshot["current_device"]
            allocated = snapshot["memory_stats"][f"device_{current_device}"]["memory_allocated"]
            reserved = snapshot["memory_stats"][f"device_{current_device}"]["memory_reserved"]
            self.logger.info(f"Memory snapshot '{name}': {allocated / 1e9:.2f}GB allocated, {reserved / 1e9:.2f}GB reserved")
        
        return snapshot
    
    def compare_memory_snapshots(
        self,
        snapshot1_name: str,
        snapshot2_name: str,
    ) -> Dict[str, Any]:
        """Compare two memory snapshots"""
        snap1 = next((s for s in self.memory_snapshots if s["name"] == snapshot1_name), None)
        snap2 = next((s for s in self.memory_snapshots if s["name"] == snapshot2_name), None)
        
        if not snap1 or not snap2:
            return {"error": "One or both snapshots not found"}
        
        comparison = {
            "snapshot1": snapshot1_name,
            "snapshot2": snapshot2_name,
            "time_diff": snap2["timestamp"] - snap1["timestamp"],
            "device_comparisons": {},
        }
        
        for device_key in snap1["memory_stats"]:
            if device_key in snap2["memory_stats"]:
                stats1 = snap1["memory_stats"][device_key]
                stats2 = snap2["memory_stats"][device_key]
                
                comparison["device_comparisons"][device_key] = {
                    "allocated_diff": stats2["memory_allocated"] - stats1["memory_allocated"],
                    "reserved_diff": stats2["memory_reserved"] - stats1["memory_reserved"],
                    "max_allocated_diff": stats2["max_memory_allocated"] - stats1["max_memory_allocated"],
                    "max_reserved_diff": stats2["max_memory_reserved"] - stats1["max_memory_reserved"],
                    "utilization_diff": stats2["memory_utilization"] - stats1["memory_utilization"],
                }
        
        return comparison
    
    def detect_memory_leaks(self, threshold_mb: float = 100.0) -> Dict[str, Any]:
        """Detect potential memory leaks by analyzing snapshots"""
        if len(self.memory_snapshots) < 2:
            return {"error": "Need at least 2 snapshots to detect leaks"}
        
        leak_analysis = {
            "threshold_mb": threshold_mb,
            "potential_leaks": [],
            "memory_growth_trend": {},
        }
        
        # Analyze memory growth over time
        for device_key in self.memory_snapshots[0]["memory_stats"]:
            allocated_over_time = []
            reserved_over_time = []
            
            for snapshot in self.memory_snapshots:
                if device_key in snapshot["memory_stats"]:
                    allocated_over_time.append(snapshot["memory_stats"][device_key]["memory_allocated"])
                    reserved_over_time.append(snapshot["memory_stats"][device_key]["memory_reserved"])
            
            # Calculate growth trend
            if len(allocated_over_time) > 1:
                allocated_growth = allocated_over_time[-1] - allocated_over_time[0]
                reserved_growth = reserved_over_time[-1] - reserved_over_time[0]
                
                leak_analysis["memory_growth_trend"][device_key] = {
                    "allocated_growth_mb": allocated_growth / 1e6,
                    "reserved_growth_mb": reserved_growth / 1e6,
                    "snapshots_analyzed": len(allocated_over_time),
                }
                
                # Check for potential leaks
                if allocated_growth / 1e6 > threshold_mb:
                    leak_analysis["potential_leaks"].append({
                        "device": device_key,
                        "type": "allocated_memory",
                        "growth_mb": allocated_growth / 1e6,
                        "growth_percentage": (allocated_growth / allocated_over_time[0]) * 100 if allocated_over_time[0] > 0 else 0,
                    })
                
                if reserved_growth / 1e6 > threshold_mb:
                    leak_analysis["potential_leaks"].append({
                        "device": device_key,
                        "type": "reserved_memory", 
                        "growth_mb": reserved_growth / 1e6,
                        "growth_percentage": (reserved_growth / reserved_over_time[0]) * 100 if reserved_over_time[0] > 0 else 0,
                    })
        
        return leak_analysis
    
    def profile_kernel_launch(
        self,
        operation_name: str,
        operation_func: callable,
        *args,
        num_runs: int = 10,
        warmup_runs: int = 3,
        **kwargs
    ) -> Dict[str, Any]:
        """Profile CUDA kernel launch timing"""
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}
        
        # Warmup runs
        for _ in range(warmup_runs):
            operation_func(*args, **kwargs)
            torch.cuda.synchronize()
        
        # Profiling runs
        start_events = []
        end_events = []
        
        for _ in range(num_runs):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            result = operation_func(*args, **kwargs)
            end_event.record()
            
            start_events.append(start_event)
            end_events.append(end_event)
        
        torch.cuda.synchronize()
        
        # Calculate timings
        elapsed_times = []
        for start_event, end_event in zip(start_events, end_events):
            elapsed_times.append(start_event.elapsed_time(end_event))
        
        import statistics
        
        profile_result = {
            "operation_name": operation_name,
            "num_runs": num_runs,
            "warmup_runs": warmup_runs,
            "timing_ms": {
                "mean": statistics.mean(elapsed_times),
                "median": statistics.median(elapsed_times),
                "std": statistics.stdev(elapsed_times) if len(elapsed_times) > 1 else 0,
                "min": min(elapsed_times),
                "max": max(elapsed_times),
                "p95": sorted(elapsed_times)[int(0.95 * len(elapsed_times))],
                "p99": sorted(elapsed_times)[int(0.99 * len(elapsed_times))],
            },
            "device_info": {
                "current_device": torch.cuda.current_device(),
                "device_name": torch.cuda.get_device_name(),
                "device_capability": torch.cuda.get_device_capability(),
            },
        }
        
        self.kernel_profiles.append(profile_result)
        
        if self.verbose:
            mean_time = profile_result["timing_ms"]["mean"]
            self.logger.info(f"Kernel '{operation_name}' avg time: {mean_time:.3f}ms")
        
        return profile_result
    
    def analyze_memory_fragmentation(self) -> Dict[str, Any]:
        """Analyze CUDA memory fragmentation"""
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}
        
        fragmentation_analysis = {
            "device_analyses": {},
        }
        
        for device_id in range(torch.cuda.device_count()):
            memory_stats = torch.cuda.memory_stats(device_id)
            
            # Calculate fragmentation metrics
            allocated = memory_stats.get("allocated_bytes.all.current", 0)
            reserved = memory_stats.get("reserved_bytes.all.current", 0)
            
            # External fragmentation (reserved but not allocated)
            external_fragmentation = reserved - allocated
            external_frag_ratio = external_fragmentation / reserved if reserved > 0 else 0
            
            device_analysis = {
                "device_id": device_id,
                "device_name": torch.cuda.get_device_name(device_id),
                "allocated_bytes": allocated,
                "reserved_bytes": reserved,
                "external_fragmentation_bytes": external_fragmentation,
                "external_fragmentation_ratio": external_frag_ratio,
                "allocation_count": memory_stats.get("allocation.all.current", 0),
                "free_count": memory_stats.get("segment.all.current", 0),
                "memory_stats": memory_stats,
            }
            
            # Add recommendations
            if external_frag_ratio > 0.3:
                device_analysis["recommendation"] = "High memory fragmentation detected. Consider torch.cuda.empty_cache()"
            elif external_frag_ratio > 0.1:
                device_analysis["recommendation"] = "Moderate memory fragmentation. Monitor usage patterns"
            else:
                device_analysis["recommendation"] = "Memory fragmentation is within normal range"
            
            fragmentation_analysis["device_analyses"][f"device_{device_id}"] = device_analysis
        
        return fragmentation_analysis
    
    def get_cuda_info(self) -> Dict[str, Any]:
        """Get comprehensive CUDA information"""
        if not torch.cuda.is_available():
            return {"cuda_available": False}
        
        cuda_info = {
            "cuda_available": True,
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "device_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device(),
            "devices": {},
        }
        
        for device_id in range(torch.cuda.device_count()):
            try:
                props = torch.cuda.get_device_properties(device_id)
                
                device_info = {
                    "name": getattr(props, 'name', 'unknown'),
                    "major": getattr(props, 'major', 0),
                    "minor": getattr(props, 'minor', 0),
                    "total_memory": getattr(props, 'total_memory', 0),
                    "total_memory_gb": getattr(props, 'total_memory', 0) / 1e9,
                    "multi_processor_count": getattr(props, 'multi_processor_count', 0),
                }
                
                # These attributes might not exist in all PyTorch versions
                if hasattr(props, 'max_threads_per_multi_processor'):
                    device_info["max_threads_per_multi_processor"] = props.max_threads_per_multi_processor
                
                if hasattr(props, 'max_threads_per_block'):
                    device_info["max_threads_per_block"] = props.max_threads_per_block
                    
                if hasattr(props, 'max_block_dimensions'):
                    device_info["max_block_dimensions"] = list(props.max_block_dimensions)
                    
                if hasattr(props, 'max_grid_dimensions'):
                    device_info["max_grid_dimensions"] = list(props.max_grid_dimensions)
                    
                if hasattr(props, 'warp_size'):
                    device_info["warp_size"] = props.warp_size
                    
            except Exception as e:
                device_info = {"error": f"Failed to get properties for device {device_id}: {e}"}
            
            cuda_info["devices"][f"device_{device_id}"] = device_info
        
        return cuda_info
    
    def _save_debug_data(self) -> None:
        """Save CUDA debug data"""
        debug_data = {
            "cuda_info": self.get_cuda_info(),
            "memory_snapshots": self.memory_snapshots,
            "kernel_profiles": self.kernel_profiles,
            "allocation_history": self.allocation_history,
            "timestamp": time.time(),
        }
        
        self.save_json(debug_data, "cuda_debug_data.json")
        
        # Generate summary
        summary = {
            "snapshots_taken": len(self.memory_snapshots),
            "kernels_profiled": len(self.kernel_profiles),
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
        
        self.save_json(summary, "cuda_debug_summary.json")
        
        if self.verbose:
            self.logger.info(f"CUDA debug data saved to {self.output_dir}")


def profile_cuda_operation(
    operation_func: callable,
    operation_name: str = "cuda_operation",
    *args,
    num_runs: int = 100,
    warmup_runs: int = 10,
    **kwargs
) -> Dict[str, Any]:
    """
    Quick CUDA operation profiling
    
    Args:
        operation_func: Function to profile
        operation_name: Name for the operation
        *args: Arguments for operation_func
        num_runs: Number of profiling runs
        warmup_runs: Number of warmup runs
        **kwargs: Keyword arguments for operation_func
    
    Returns:
        Profiling results
    
    Example:
        >>> def matmul_op():
        >>>     a = torch.randn(1000, 1000, device='cuda')
        >>>     b = torch.randn(1000, 1000, device='cuda')
        >>>     return torch.matmul(a, b)
        >>> results = profile_cuda_operation(matmul_op, "matrix_multiply")
    """
    debugger = CUDADebugger(verbose=False)
    return debugger.profile_kernel_launch(
        operation_name, operation_func, *args,
        num_runs=num_runs, warmup_runs=warmup_runs, **kwargs
    )


def auto_cuda_debug(
    output_dir: Optional[Union[str, Path]] = None,
    **kwargs
) -> CUDADebugger:
    """
    Quick CUDA debugging setup
    
    Args:
        output_dir: Output directory
        **kwargs: Additional arguments for CUDADebugger
    
    Returns:
        Enabled CUDADebugger instance
    
    Example:
        >>> cuda_debugger = auto_cuda_debug()
        >>> cuda_debugger.snapshot_memory("before_training")
        >>> # ... run training ...
        >>> cuda_debugger.snapshot_memory("after_training")
        >>> leak_analysis = cuda_debugger.detect_memory_leaks()
        >>> cuda_debugger.disable()
    """
    debugger = CUDADebugger(output_dir=output_dir, **kwargs)
    debugger.enable()
    return debugger