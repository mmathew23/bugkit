"""
Triton kernel debugging and profiling utilities
"""

import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import torch

from ...core.base import BaseDebugTool


class TritonDebugger(BaseDebugTool):
    """Triton kernel debugging and profiling utilities"""
    
    def __init__(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        verbose: bool = True,
        profile_kernels: bool = True,
        capture_kernel_source: bool = True,
    ):
        super().__init__(output_dir, verbose)
        self.profile_kernels = profile_kernels
        self.capture_kernel_source = capture_kernel_source
        
        self.kernel_profiles: List[Dict[str, Any]] = []
        self.kernel_compilations: List[Dict[str, Any]] = []
        self.kernel_launches: List[Dict[str, Any]] = []
        
        # Check if Triton is available
        try:
            import triton
            import triton.language as tl
            self.triton_available = True
            self.triton = triton
            self.tl = tl
        except ImportError:
            self.triton_available = False
            if verbose:
                self.logger.warning("Triton not available - TritonDebugger will have limited functionality")
    
    def enable(self) -> None:
        """Enable Triton debugging"""
        self.enabled = True
        
        if self.verbose:
            status = "enabled" if self.triton_available else "enabled (limited - no Triton)"
            self.logger.info(f"Triton debugger {status}")
    
    def disable(self) -> None:
        """Disable Triton debugging and save results"""
        self.enabled = False
        self._save_debug_data()
        
        if self.verbose:
            self.logger.info("Triton debugger disabled")
    
    def profile_kernel(
        self,
        kernel_func: Callable,
        kernel_name: str,
        grid: tuple,
        input_tensors: Dict[str, torch.Tensor],
        num_runs: int = 100,
        warmup_runs: int = 10,
        **kernel_kwargs
    ) -> Dict[str, Any]:
        """
        Profile a Triton kernel
        
        Args:
            kernel_func: Triton kernel function
            kernel_name: Name for the kernel
            grid: Grid configuration for kernel launch
            input_tensors: Input tensors for the kernel
            num_runs: Number of profiling runs
            warmup_runs: Number of warmup runs
            **kernel_kwargs: Additional kernel arguments
        
        Returns:
            Kernel profiling results
        """
        if not self.triton_available:
            return {"error": "Triton not available"}
        
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}
        
        profile_result = {
            "kernel_name": kernel_name,
            "grid": grid,
            "input_shapes": {k: list(v.shape) for k, v in input_tensors.items()},
            "input_dtypes": {k: str(v.dtype) for k, v in input_tensors.items()},
            "kernel_kwargs": kernel_kwargs,
            "num_runs": num_runs,
            "warmup_runs": warmup_runs,
            "compilation_info": {},
            "timing_results": {},
            "memory_usage": {},
        }
        
        # Capture kernel source if available
        if self.capture_kernel_source and hasattr(kernel_func, 'src'):
            profile_result["kernel_source"] = kernel_func.src
        
        # Warmup runs
        for _ in range(warmup_runs):
            try:
                kernel_func[grid](**input_tensors, **kernel_kwargs)
                torch.cuda.synchronize()
            except Exception as e:
                profile_result["warmup_error"] = str(e)
                return profile_result
        
        # Profiling runs
        start_events = []
        end_events = []
        
        for _ in range(num_runs):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            kernel_func[grid](**input_tensors, **kernel_kwargs)
            end_event.record()
            
            start_events.append(start_event)
            end_events.append(end_event)
        
        torch.cuda.synchronize()
        
        # Calculate timings
        elapsed_times = []
        for start_event, end_event in zip(start_events, end_events):
            elapsed_times.append(start_event.elapsed_time(end_event))
        
        import statistics
        
        profile_result["timing_results"] = {
            "mean_ms": statistics.mean(elapsed_times),
            "median_ms": statistics.median(elapsed_times),
            "std_ms": statistics.stdev(elapsed_times) if len(elapsed_times) > 1 else 0,
            "min_ms": min(elapsed_times),
            "max_ms": max(elapsed_times),
            "p95_ms": sorted(elapsed_times)[int(0.95 * len(elapsed_times))],
            "p99_ms": sorted(elapsed_times)[int(0.99 * len(elapsed_times))],
        }
        
        # Memory usage analysis
        try:
            mem_before = torch.cuda.memory_allocated()
            kernel_func[grid](**input_tensors, **kernel_kwargs)
            torch.cuda.synchronize()
            mem_after = torch.cuda.memory_allocated()
            
            profile_result["memory_usage"] = {
                "memory_delta_bytes": mem_after - mem_before,
                "memory_delta_mb": (mem_after - mem_before) / 1e6,
            }
        except Exception as e:
            profile_result["memory_analysis_error"] = str(e)
        
        # Kernel compilation info
        try:
            if hasattr(kernel_func, 'cache'):
                cache_info = {}
                for key, compiled_kernel in kernel_func.cache.items():
                    cache_info[str(key)] = {
                        "num_warps": getattr(compiled_kernel, 'num_warps', None),
                        "num_stages": getattr(compiled_kernel, 'num_stages', None),
                        "shared_memory": getattr(compiled_kernel, 'shared', None),
                    }
                profile_result["compilation_info"]["cache"] = cache_info
        except Exception as e:
            profile_result["compilation_analysis_error"] = str(e)
        
        self.kernel_profiles.append(profile_result)
        
        if self.verbose:
            mean_time = profile_result["timing_results"]["mean_ms"]
            self.logger.info(f"Triton kernel '{kernel_name}' avg time: {mean_time:.3f}ms")
        
        return profile_result
    
    def compare_kernels(
        self,
        kernel_configs: List[Dict[str, Any]],
        comparison_name: str = "kernel_comparison",
    ) -> Dict[str, Any]:
        """
        Compare multiple kernel configurations
        
        Args:
            kernel_configs: List of kernel configuration dictionaries
            comparison_name: Name for the comparison
        
        Returns:
            Kernel comparison results
        """
        if not self.triton_available:
            return {"error": "Triton not available"}
        
        comparison_results = {
            "comparison_name": comparison_name,
            "num_kernels": len(kernel_configs),
            "individual_results": {},
            "performance_ranking": [],
            "memory_ranking": [],
            "recommendations": [],
        }
        
        kernel_results = []
        
        for i, config in enumerate(kernel_configs):
            kernel_name = config.get("name", f"kernel_{i}")
            
            # Profile the kernel
            result = self.profile_kernel(
                kernel_func=config["kernel_func"],
                kernel_name=kernel_name,
                grid=config["grid"],
                input_tensors=config["input_tensors"],
                num_runs=config.get("num_runs", 50),
                warmup_runs=config.get("warmup_runs", 5),
                **config.get("kernel_kwargs", {})
            )
            
            if "error" not in result:
                kernel_results.append((kernel_name, result))
                comparison_results["individual_results"][kernel_name] = result
        
        if not kernel_results:
            comparison_results["error"] = "No successful kernel profiles"
            return comparison_results
        
        # Performance ranking
        performance_ranking = sorted(
            kernel_results,
            key=lambda x: x[1]["timing_results"]["mean_ms"]
        )
        
        comparison_results["performance_ranking"] = [
            {
                "rank": i + 1,
                "kernel_name": name,
                "mean_time_ms": result["timing_results"]["mean_ms"],
                "speedup_vs_slowest": performance_ranking[-1][1]["timing_results"]["mean_ms"] / result["timing_results"]["mean_ms"],
            }
            for i, (name, result) in enumerate(performance_ranking)
        ]
        
        # Memory ranking (if available)
        memory_results = [(name, result) for name, result in kernel_results 
                         if "memory_usage" in result and "memory_delta_mb" in result["memory_usage"]]
        
        if memory_results:
            memory_ranking = sorted(
                memory_results,
                key=lambda x: abs(x[1]["memory_usage"]["memory_delta_mb"])
            )
            
            comparison_results["memory_ranking"] = [
                {
                    "rank": i + 1,
                    "kernel_name": name,
                    "memory_delta_mb": result["memory_usage"]["memory_delta_mb"],
                }
                for i, (name, result) in enumerate(memory_ranking)
            ]
        
        # Generate recommendations
        if len(performance_ranking) > 1:
            fastest = performance_ranking[0]
            slowest = performance_ranking[-1]
            speedup = slowest[1]["timing_results"]["mean_ms"] / fastest[1]["timing_results"]["mean_ms"]
            
            comparison_results["recommendations"].append(
                f"'{fastest[0]}' is {speedup:.2f}x faster than '{slowest[0]}'"
            )
            
            if speedup > 2:
                comparison_results["recommendations"].append(
                    f"Consider using '{fastest[0]}' configuration for better performance"
                )
        
        return comparison_results
    
    def analyze_kernel_occupancy(
        self,
        kernel_func: Callable,
        grid: tuple,
        input_tensors: Dict[str, torch.Tensor],
        **kernel_kwargs
    ) -> Dict[str, Any]:
        """
        Analyze kernel occupancy and resource usage
        
        Args:
            kernel_func: Triton kernel function
            grid: Grid configuration
            input_tensors: Input tensors
            **kernel_kwargs: Additional kernel arguments
        
        Returns:
            Occupancy analysis results
        """
        if not self.triton_available:
            return {"error": "Triton not available"}
        
        occupancy_analysis = {
            "grid": grid,
            "resource_usage": {},
            "occupancy_metrics": {},
            "recommendations": [],
        }
        
        try:
            # Get device properties
            device = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device)
            
            occupancy_analysis["device_info"] = {
                "name": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "multiprocessor_count": props.multi_processor_count,
                "max_threads_per_multiprocessor": props.max_threads_per_multi_processor,
                "max_threads_per_block": props.max_threads_per_block,
                "warp_size": props.warp_size,
                "total_memory": props.total_memory,
            }
            
            # Analyze grid configuration
            total_blocks = 1
            for dim in grid:
                total_blocks *= dim
            
            occupancy_analysis["grid_analysis"] = {
                "total_blocks": total_blocks,
                "blocks_per_sm": total_blocks / props.multi_processor_count,
            }
            
            # Get compiled kernel info if available
            if hasattr(kernel_func, 'cache'):
                for key, compiled_kernel in kernel_func.cache.items():
                    kernel_info = {
                        "cache_key": str(key),
                        "num_warps": getattr(compiled_kernel, 'num_warps', None),
                        "num_stages": getattr(compiled_kernel, 'num_stages', None),
                        "shared_memory_bytes": getattr(compiled_kernel, 'shared', None),
                    }
                    
                    if kernel_info["num_warps"]:
                        threads_per_block = kernel_info["num_warps"] * props.warp_size
                        kernel_info["threads_per_block"] = threads_per_block
                        kernel_info["theoretical_occupancy"] = min(
                            props.max_threads_per_block / threads_per_block,
                            props.max_threads_per_multi_processor / threads_per_block
                        )
                    
                    occupancy_analysis["resource_usage"][str(key)] = kernel_info
            
        except Exception as e:
            occupancy_analysis["analysis_error"] = str(e)
        
        return occupancy_analysis
    
    def _save_debug_data(self) -> None:
        """Save Triton debug data"""
        debug_data = {
            "triton_available": self.triton_available,
            "kernel_profiles": self.kernel_profiles,
            "kernel_compilations": self.kernel_compilations,
            "kernel_launches": self.kernel_launches,
            "timestamp": time.time(),
        }
        
        if self.triton_available:
            try:
                debug_data["triton_version"] = self.triton.__version__
            except:
                pass
        
        self.save_json(debug_data, "triton_debug_data.json")
        
        # Generate summary
        summary = {
            "triton_available": self.triton_available,
            "kernels_profiled": len(self.kernel_profiles),
            "compilations_tracked": len(self.kernel_compilations),
            "launches_tracked": len(self.kernel_launches),
        }
        
        self.save_json(summary, "triton_debug_summary.json")
        
        if self.verbose:
            self.logger.info(f"Triton debug data saved to {self.output_dir}")


def profile_triton_kernel(
    kernel_func: Callable,
    grid: tuple,
    input_tensors: Dict[str, torch.Tensor],
    kernel_name: str = "triton_kernel",
    num_runs: int = 100,
    warmup_runs: int = 10,
    **kernel_kwargs
) -> Dict[str, Any]:
    """
    Quick Triton kernel profiling
    
    Args:
        kernel_func: Triton kernel function
        grid: Grid configuration
        input_tensors: Input tensors
        kernel_name: Name for the kernel
        num_runs: Number of profiling runs
        warmup_runs: Number of warmup runs
        **kernel_kwargs: Additional kernel arguments
    
    Returns:
        Profiling results
    
    Example:
        >>> @triton.jit
        >>> def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        >>>     # kernel implementation
        >>>     pass
        >>> 
        >>> x = torch.randn(1024, device='cuda')
        >>> y = torch.randn(1024, device='cuda')
        >>> output = torch.empty_like(x)
        >>> 
        >>> inputs = {'x_ptr': x, 'y_ptr': y, 'output_ptr': output, 'n_elements': 1024}
        >>> results = profile_triton_kernel(add_kernel, (32,), inputs, "vector_add")
    """
    debugger = TritonDebugger(verbose=False)
    return debugger.profile_kernel(
        kernel_func, kernel_name, grid, input_tensors,
        num_runs=num_runs, warmup_runs=warmup_runs, **kernel_kwargs
    )


def auto_triton_debug(
    output_dir: Optional[Union[str, Path]] = None,
    **kwargs
) -> TritonDebugger:
    """
    Quick Triton debugging setup
    
    Args:
        output_dir: Output directory
        **kwargs: Additional arguments for TritonDebugger
    
    Returns:
        Enabled TritonDebugger instance
    
    Example:
        >>> triton_debugger = auto_triton_debug()
        >>> # Profile kernels...
        >>> triton_debugger.disable()
    """
    debugger = TritonDebugger(output_dir=output_dir, **kwargs)
    debugger.enable()
    return debugger