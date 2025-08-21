"""
Multi-GPU and distributed training debugging utilities
"""

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.distributed as dist
import torch.nn as nn

from ...core.base import BaseDebugTool


class DistributedDebugger(BaseDebugTool):
    """Debugging utilities for multi-GPU and distributed training"""
    
    def __init__(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        verbose: bool = True,
        track_communication: bool = True,
        profile_allreduce: bool = True,
        monitor_load_balance: bool = True,
    ):
        super().__init__(output_dir, verbose)
        self.track_communication = track_communication
        self.profile_allreduce = profile_allreduce
        self.monitor_load_balance = monitor_load_balance
        
        self.communication_logs: List[Dict[str, Any]] = []
        self.load_balance_data: List[Dict[str, Any]] = []
        self.gradient_sync_data: List[Dict[str, Any]] = []
        
        # Check distributed training environment
        self.distributed_info = self._analyze_distributed_setup()
        
    def enable(self) -> None:
        """Enable distributed debugging"""
        self.enabled = True
        
        if self.verbose:
            rank = self.distributed_info.get("rank", "unknown")
            world_size = self.distributed_info.get("world_size", "unknown")
            self.logger.info(f"Distributed debugger enabled (rank {rank}/{world_size})")
    
    def disable(self) -> None:
        """Disable distributed debugging and save results"""
        self.enabled = False
        self._save_debug_data()
        
        if self.verbose:
            self.logger.info("Distributed debugger disabled")
    
    def _analyze_distributed_setup(self) -> Dict[str, Any]:
        """Analyze the current distributed training setup"""
        setup_info = {
            "distributed_available": dist.is_available(),
            "distributed_initialized": False,
            "backend": None,
            "rank": None,
            "world_size": None,
            "local_rank": None,
            "node_rank": None,
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "environment_vars": {},
        }
        
        # Check if distributed is initialized
        if dist.is_available() and dist.is_initialized():
            setup_info.update({
                "distributed_initialized": True,
                "backend": dist.get_backend(),
                "rank": dist.get_rank(),
                "world_size": dist.get_world_size(),
            })
        
        # Check environment variables commonly used in distributed training
        env_vars = [
            "RANK", "WORLD_SIZE", "LOCAL_RANK", "NODE_RANK",
            "MASTER_ADDR", "MASTER_PORT", "NCCL_DEBUG",
            "CUDA_VISIBLE_DEVICES", "PYTORCH_CUDA_ALLOC_CONF",
            "NCCL_SOCKET_IFNAME", "NCCL_IB_DISABLE",
        ]
        
        for var in env_vars:
            if var in os.environ:
                setup_info["environment_vars"][var] = os.environ[var]
        
        # Try to get local rank from environment
        if "LOCAL_RANK" in os.environ:
            setup_info["local_rank"] = int(os.environ["LOCAL_RANK"])
        elif "CUDA_VISIBLE_DEVICES" in os.environ:
            # Infer local rank from CUDA_VISIBLE_DEVICES
            visible_devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
            setup_info["local_rank"] = len(visible_devices) - 1 if visible_devices else 0
        
        return setup_info
    
    def profile_gradient_sync(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: callable,
        inputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        sync_name: str = "gradient_sync",
    ) -> Dict[str, Any]:
        """Profile gradient synchronization in distributed training"""
        if not self.enabled:
            raise RuntimeError("DistributedDebugger is not enabled")
        
        sync_profile = {
            "sync_name": sync_name,
            "timestamp": time.time(),
            "distributed_info": self.distributed_info,
            "timing": {},
            "gradient_stats": {},
            "communication_volume": {},
        }
        
        # Profile forward pass
        start_time = time.perf_counter()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        forward_start = time.perf_counter()
        outputs = model(**inputs)
        loss = loss_fn(outputs, targets)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        forward_end = time.perf_counter()
        sync_profile["timing"]["forward_ms"] = (forward_end - forward_start) * 1000
        
        # Profile backward pass and gradient sync
        optimizer.zero_grad()
        
        backward_start = time.perf_counter()
        loss.backward()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        backward_end = time.perf_counter()
        sync_profile["timing"]["backward_ms"] = (backward_end - backward_start) * 1000
        
        # Analyze gradients before sync
        gradient_stats_before = self._analyze_gradients(model, "before_sync")
        
        # Profile optimizer step (includes AllReduce for DDP)
        optimizer_start = time.perf_counter()
        optimizer.step()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        optimizer_end = time.perf_counter()
        sync_profile["timing"]["optimizer_step_ms"] = (optimizer_end - optimizer_start) * 1000
        
        # Analyze gradients after sync
        gradient_stats_after = self._analyze_gradients(model, "after_sync")
        
        sync_profile["gradient_stats"] = {
            "before_sync": gradient_stats_before,
            "after_sync": gradient_stats_after,
        }
        
        # Calculate total timing
        total_time = (optimizer_end - start_time) * 1000
        sync_profile["timing"]["total_ms"] = total_time
        sync_profile["timing"]["sync_overhead_ms"] = (
            total_time - sync_profile["timing"]["forward_ms"] - sync_profile["timing"]["backward_ms"]
        )
        
        # Estimate communication volume
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        param_bytes = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)
        
        sync_profile["communication_volume"] = {
            "total_parameters": total_params,
            "parameter_bytes": param_bytes,
            "parameter_mb": param_bytes / 1e6,
            "estimated_allreduce_bytes": param_bytes * 2 if self.distributed_info.get("world_size", 1) > 1 else 0,
        }
        
        self.gradient_sync_data.append(sync_profile)
        
        if self.verbose:
            sync_time = sync_profile["timing"]["sync_overhead_ms"]
            self.logger.info(f"Gradient sync '{sync_name}' overhead: {sync_time:.2f}ms")
        
        return sync_profile
    
    def monitor_load_balance(
        self,
        batch_sizes: Dict[int, int],  # rank -> batch_size
        processing_times: Dict[int, float],  # rank -> processing_time
        analysis_name: str = "load_balance",
    ) -> Dict[str, Any]:
        """Monitor load balance across ranks"""
        if not self.enabled:
            raise RuntimeError("DistributedDebugger is not enabled")
        
        load_analysis = {
            "analysis_name": analysis_name,
            "timestamp": time.time(),
            "distributed_info": self.distributed_info,
            "batch_analysis": {},
            "timing_analysis": {},
            "load_balance_metrics": {},
            "recommendations": [],
        }
        
        # Analyze batch size distribution
        if batch_sizes:
            batch_values = list(batch_sizes.values())
            load_analysis["batch_analysis"] = {
                "min_batch_size": min(batch_values),
                "max_batch_size": max(batch_values),
                "mean_batch_size": sum(batch_values) / len(batch_values),
                "batch_size_std": self._calculate_std(batch_values),
                "batch_imbalance_ratio": max(batch_values) / min(batch_values) if min(batch_values) > 0 else float('inf'),
                "rank_batch_sizes": batch_sizes,
            }
        
        # Analyze processing time distribution
        if processing_times:
            time_values = list(processing_times.values())
            load_analysis["timing_analysis"] = {
                "min_processing_time": min(time_values),
                "max_processing_time": max(time_values),
                "mean_processing_time": sum(time_values) / len(time_values),
                "processing_time_std": self._calculate_std(time_values),
                "timing_imbalance_ratio": max(time_values) / min(time_values) if min(time_values) > 0 else float('inf'),
                "rank_processing_times": processing_times,
            }
        
        # Calculate load balance metrics
        if batch_sizes and processing_times:
            # Calculate efficiency metrics
            max_time = max(processing_times.values())
            total_useful_time = sum(processing_times.values())
            num_ranks = len(processing_times)
            
            load_analysis["load_balance_metrics"] = {
                "parallel_efficiency": (total_useful_time / (num_ranks * max_time)) * 100,
                "load_balance_score": (1 - (max_time - min(processing_times.values())) / max_time) * 100,
                "wasted_time_ratio": ((num_ranks * max_time - total_useful_time) / (num_ranks * max_time)) * 100,
            }
            
            # Generate recommendations
            batch_imbalance = load_analysis["batch_analysis"]["batch_imbalance_ratio"]
            timing_imbalance = load_analysis["timing_analysis"]["timing_imbalance_ratio"]
            
            if batch_imbalance > 1.2:
                load_analysis["recommendations"].append(f"High batch size imbalance (ratio: {batch_imbalance:.2f}) - consider better data distribution")
            
            if timing_imbalance > 1.3:
                load_analysis["recommendations"].append(f"High processing time imbalance (ratio: {timing_imbalance:.2f}) - check for stragglers")
            
            efficiency = load_analysis["load_balance_metrics"]["parallel_efficiency"]
            if efficiency < 80:
                load_analysis["recommendations"].append(f"Low parallel efficiency ({efficiency:.1f}%) - investigate load balancing issues")
        
        self.load_balance_data.append(load_analysis)
        
        if self.verbose and "load_balance_metrics" in load_analysis:
            efficiency = load_analysis["load_balance_metrics"]["parallel_efficiency"]
            self.logger.info(f"Load balance analysis '{analysis_name}': {efficiency:.1f}% parallel efficiency")
        
        return load_analysis
    
    def profile_communication_collective(
        self,
        operation: str,  # "allreduce", "allgather", "broadcast", etc.
        tensor: torch.Tensor,
        operation_name: str = "collective_op",
        num_runs: int = 10,
        warmup_runs: int = 3,
    ) -> Dict[str, Any]:
        """Profile distributed communication collectives"""
        if not self.enabled:
            raise RuntimeError("DistributedDebugger is not enabled")
        
        if not dist.is_initialized():
            return {"error": "Distributed training not initialized"}
        
        comm_profile = {
            "operation_name": operation_name,
            "operation_type": operation,
            "timestamp": time.time(),
            "distributed_info": self.distributed_info,
            "tensor_info": {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "device": str(tensor.device),
                "size_bytes": tensor.numel() * tensor.element_size(),
                "size_mb": (tensor.numel() * tensor.element_size()) / 1e6,
            },
            "timing_results": {},
            "bandwidth_analysis": {},
        }
        
        # Warmup runs
        for _ in range(warmup_runs):
            test_tensor = tensor.clone()
            try:
                if operation == "allreduce":
                    dist.all_reduce(test_tensor)
                elif operation == "allgather":
                    output_tensors = [torch.zeros_like(test_tensor) for _ in range(dist.get_world_size())]
                    dist.all_gather(output_tensors, test_tensor)
                elif operation == "broadcast":
                    dist.broadcast(test_tensor, src=0)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except Exception as e:
                comm_profile["warmup_error"] = str(e)
                return comm_profile
        
        # Profiling runs
        elapsed_times = []
        
        for _ in range(num_runs):
            test_tensor = tensor.clone()
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            
            try:
                if operation == "allreduce":
                    dist.all_reduce(test_tensor)
                elif operation == "allgather":
                    output_tensors = [torch.zeros_like(test_tensor) for _ in range(dist.get_world_size())]
                    dist.all_gather(output_tensors, test_tensor)
                elif operation == "broadcast":
                    dist.broadcast(test_tensor, src=0)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                elapsed_times.append((end_time - start_time) * 1000)  # Convert to ms
                
            except Exception as e:
                comm_profile["execution_error"] = str(e)
                return comm_profile
        
        # Calculate timing statistics
        import statistics
        
        comm_profile["timing_results"] = {
            "mean_ms": statistics.mean(elapsed_times),
            "median_ms": statistics.median(elapsed_times),
            "std_ms": statistics.stdev(elapsed_times) if len(elapsed_times) > 1 else 0,
            "min_ms": min(elapsed_times),
            "max_ms": max(elapsed_times),
            "p95_ms": sorted(elapsed_times)[int(0.95 * len(elapsed_times))],
            "p99_ms": sorted(elapsed_times)[int(0.99 * len(elapsed_times))],
        }
        
        # Calculate bandwidth
        data_size_bytes = comm_profile["tensor_info"]["size_bytes"]
        mean_time_seconds = comm_profile["timing_results"]["mean_ms"] / 1000
        
        if operation == "allreduce":
            # AllReduce transfers (N-1)/N * data_size in ring algorithm
            world_size = dist.get_world_size()
            effective_data_size = data_size_bytes * 2 * (world_size - 1) / world_size
        elif operation == "allgather":
            # AllGather transfers (N-1) * data_size
            effective_data_size = data_size_bytes * (dist.get_world_size() - 1)
        elif operation == "broadcast":
            # Broadcast transfers data_size from root to all others
            effective_data_size = data_size_bytes
        else:
            effective_data_size = data_size_bytes
        
        comm_profile["bandwidth_analysis"] = {
            "effective_data_size_bytes": effective_data_size,
            "effective_data_size_mb": effective_data_size / 1e6,
            "bandwidth_gbps": (effective_data_size * 8) / (mean_time_seconds * 1e9),  # Gbps
            "bandwidth_mb_per_s": effective_data_size / (mean_time_seconds * 1e6),  # MB/s
        }
        
        self.communication_logs.append(comm_profile)
        
        if self.verbose:
            bandwidth = comm_profile["bandwidth_analysis"]["bandwidth_gbps"]
            self.logger.info(f"Communication '{operation_name}' ({operation}): {bandwidth:.2f} Gbps")
        
        return comm_profile
    
    def _analyze_gradients(self, model: nn.Module, stage: str) -> Dict[str, Any]:
        """Analyze gradient statistics"""
        gradient_stats = {
            "stage": stage,
            "parameter_count": 0,
            "gradient_norms": [],
            "total_gradient_norm": 0,
            "gradient_statistics": {},
        }
        
        gradient_norms = []
        
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                gradient_stats["parameter_count"] += 1
                grad_norm = float(param.grad.norm())
                gradient_norms.append(grad_norm)
                
                gradient_stats["gradient_statistics"][name] = {
                    "norm": grad_norm,
                    "mean": float(param.grad.mean()),
                    "std": float(param.grad.std()),
                    "min": float(param.grad.min()),
                    "max": float(param.grad.max()),
                }
        
        if gradient_norms:
            gradient_stats["gradient_norms"] = gradient_norms
            gradient_stats["total_gradient_norm"] = float(torch.tensor(gradient_norms).norm())
            gradient_stats["mean_gradient_norm"] = sum(gradient_norms) / len(gradient_norms)
            gradient_stats["max_gradient_norm"] = max(gradient_norms)
            gradient_stats["min_gradient_norm"] = min(gradient_norms)
        
        return gradient_stats
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation"""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5
    
    def get_distributed_summary(self) -> Dict[str, Any]:
        """Get a summary of distributed training analysis"""
        return {
            "distributed_info": self.distributed_info,
            "gradient_syncs_profiled": len(self.gradient_sync_data),
            "load_balance_analyses": len(self.load_balance_data),
            "communication_ops_profiled": len(self.communication_logs),
            "enabled": self.enabled,
        }
    
    def _save_debug_data(self) -> None:
        """Save distributed debug data"""
        debug_data = {
            "distributed_info": self.distributed_info,
            "gradient_sync_data": self.gradient_sync_data,
            "load_balance_data": self.load_balance_data,
            "communication_logs": self.communication_logs,
            "timestamp": time.time(),
        }
        
        self.save_json(debug_data, "distributed_debug_data.json")
        
        # Generate summary
        summary = self.get_distributed_summary()
        self.save_json(summary, "distributed_debug_summary.json")
        
        if self.verbose:
            self.logger.info(f"Distributed debug data saved to {self.output_dir}")


def auto_distributed_debug(
    output_dir: Optional[Union[str, Path]] = None,
    **kwargs
) -> DistributedDebugger:
    """
    Quick distributed training debugging setup
    
    Args:
        output_dir: Output directory
        **kwargs: Additional arguments for DistributedDebugger
    
    Returns:
        Enabled DistributedDebugger instance
    
    Example:
        >>> dist_debugger = auto_distributed_debug()
        >>> # Profile gradient synchronization
        >>> sync_profile = dist_debugger.profile_gradient_sync(model, optimizer, loss_fn, inputs, targets)
        >>> dist_debugger.disable()
    """
    debugger = DistributedDebugger(output_dir=output_dir, **kwargs)
    debugger.enable()
    return debugger