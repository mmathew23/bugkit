"""
Accelerate debugging utilities for distributed training and optimization
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

from ...core.base import BaseDebugTool


class AccelerateDebugger(BaseDebugTool):
    """Debugging utilities for Accelerate-based training"""
    
    def __init__(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        verbose: bool = True,
        track_device_placement: bool = True,
        monitor_gradient_sync: bool = True,
        profile_dataloader: bool = True,
    ):
        super().__init__(output_dir, verbose)
        self.track_device_placement = track_device_placement
        self.monitor_gradient_sync = monitor_gradient_sync
        self.profile_dataloader = profile_dataloader
        
        self.device_analyses: List[Dict[str, Any]] = []
        self.gradient_sync_data: List[Dict[str, Any]] = []
        self.dataloader_profiles: List[Dict[str, Any]] = []
        self.training_profiles: List[Dict[str, Any]] = []
        
        # Check if Accelerate is available
        try:
            import accelerate
            self.accelerate_available = True
            self.accelerate = accelerate
            self.accelerator = None  # Will be set if provided
        except ImportError:
            self.accelerate_available = False
            if verbose:
                self.logger.warning("Accelerate not available - AccelerateDebugger will have limited functionality")
    
    def enable(self) -> None:
        """Enable Accelerate debugging"""
        self.enabled = True
        
        if self.verbose:
            status = "enabled" if self.accelerate_available else "enabled (limited - no Accelerate)"
            self.logger.info(f"Accelerate debugger {status}")
    
    def disable(self) -> None:
        """Disable Accelerate debugging and save results"""
        self.enabled = False
        self._save_debug_data()
        
        if self.verbose:
            self.logger.info("Accelerate debugger disabled")
    
    def set_accelerator(self, accelerator) -> None:
        """Set the Accelerator instance to monitor"""
        if not self.accelerate_available:
            self.logger.warning("Accelerate not available - cannot set accelerator")
            return
        
        self.accelerator = accelerator
        if self.verbose:
            self.logger.info("Accelerator instance set for monitoring")
    
    def analyze_device_placement(
        self,
        model: nn.Module,
        analysis_name: str = "device_placement",
    ) -> Dict[str, Any]:
        """Analyze device placement of model components"""
        if not self.enabled:
            raise RuntimeError("AccelerateDebugger is not enabled")
        
        device_analysis = {
            "analysis_name": analysis_name,
            "timestamp": time.time(),
            "accelerator_info": {},
            "model_device_mapping": {},
            "parameter_distribution": {},
            "memory_analysis": {},
            "recommendations": [],
        }
        
        # Get accelerator info if available
        if self.accelerator is not None:
            try:
                device_analysis["accelerator_info"] = {
                    "device": str(self.accelerator.device),
                    "num_processes": self.accelerator.num_processes,
                    "process_index": self.accelerator.process_index,
                    "local_process_index": self.accelerator.local_process_index,
                    "is_main_process": self.accelerator.is_main_process,
                    "is_local_main_process": self.accelerator.is_local_main_process,
                    "mixed_precision": str(self.accelerator.mixed_precision),
                    "distributed_type": str(self.accelerator.distributed_type),
                }
                
                if hasattr(self.accelerator, 'state'):
                    state = self.accelerator.state
                    device_analysis["accelerator_info"].update({
                        "deepspeed_plugin": state.deepspeed_plugin is not None,
                        "fsdp_plugin": state.fsdp_plugin is not None,
                        "use_fp16": state.use_fp16,
                        "use_bf16": state.use_bf16,
                    })
                    
            except Exception as e:
                device_analysis["accelerator_info"] = {"error": str(e)}
        
        # Analyze model device placement
        device_counts = {}
        parameter_devices = {}
        memory_per_device = {}
        
        for name, param in model.named_parameters():
            device = str(param.device)
            param_size = param.numel() * param.element_size()
            
            # Count parameters per device
            if device not in device_counts:
                device_counts[device] = {"count": 0, "memory_bytes": 0, "parameters": []}
            
            device_counts[device]["count"] += param.numel()
            device_counts[device]["memory_bytes"] += param_size
            device_counts[device]["parameters"].append({
                "name": name,
                "shape": list(param.shape),
                "size_mb": param_size / (1024 * 1024),
                "dtype": str(param.dtype),
            })
            
            parameter_devices[name] = device
        
        # Analyze buffers as well
        for name, buffer in model.named_buffers():
            device = str(buffer.device)
            buffer_size = buffer.numel() * buffer.element_size()
            
            if device not in device_counts:
                device_counts[device] = {"count": 0, "memory_bytes": 0, "parameters": []}
            
            device_counts[device]["count"] += buffer.numel()
            device_counts[device]["memory_bytes"] += buffer_size
            device_counts[device]["parameters"].append({
                "name": f"buffer:{name}",
                "shape": list(buffer.shape),
                "size_mb": buffer_size / (1024 * 1024),
                "dtype": str(buffer.dtype),
            })
        
        device_analysis["model_device_mapping"] = device_counts
        device_analysis["parameter_distribution"] = parameter_devices
        
        # Memory analysis per device
        total_memory = sum(info["memory_bytes"] for info in device_counts.values())
        
        for device, info in device_counts.items():
            memory_per_device[device] = {
                "memory_mb": info["memory_bytes"] / (1024 * 1024),
                "memory_percentage": (info["memory_bytes"] / total_memory) * 100 if total_memory > 0 else 0,
                "parameter_count": info["count"],
            }
        
        device_analysis["memory_analysis"] = memory_per_device
        
        # Generate recommendations
        unique_devices = len(device_counts)
        if unique_devices > 1:
            device_analysis["recommendations"].append(f"Model spans {unique_devices} devices - ensure communication is optimized")
        
        # Check for uneven distribution
        if unique_devices > 1:
            memory_values = [info["memory_mb"] for info in memory_per_device.values()]
            max_memory = max(memory_values)
            min_memory = min(memory_values)
            
            if max_memory > min_memory * 2:
                device_analysis["recommendations"].append("Uneven memory distribution across devices - consider load balancing")
        
        # Check for CPU tensors in GPU model
        has_cpu = any("cpu" in device for device in device_counts.keys())
        has_gpu = any("cuda" in device for device in device_counts.keys())
        
        if has_cpu and has_gpu:
            device_analysis["recommendations"].append("Mixed CPU/GPU placement detected - verify this is intentional")
        
        self.device_analyses.append(device_analysis)
        
        if self.verbose:
            devices = list(device_counts.keys())
            self.logger.info(f"Device placement analysis '{analysis_name}': model spans {devices}")
        
        return device_analysis
    
    def profile_training_step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        data_batch: Dict[str, torch.Tensor],
        loss_fn: callable,
        step_name: str = "training_step",
    ) -> Dict[str, Any]:
        """Profile a complete Accelerate training step"""
        if not self.enabled:
            raise RuntimeError("AccelerateDebugger is not enabled")
        
        step_profile = {
            "step_name": step_name,
            "timestamp": time.time(),
            "accelerator_info": {},
            "timing_breakdown": {},
            "memory_tracking": {},
            "gradient_analysis": {},
            "device_transfers": {},
            "recommendations": [],
        }
        
        # Get accelerator info
        if self.accelerator is not None:
            step_profile["accelerator_info"] = {
                "device": str(self.accelerator.device),
                "mixed_precision": str(self.accelerator.mixed_precision),
                "num_processes": self.accelerator.num_processes,
            }
        
        # Track memory before
        memory_before = self._get_memory_snapshot()
        
        start_time = time.perf_counter()
        
        # Forward pass timing
        forward_start = time.perf_counter()
        
        if self.accelerator is not None:
            with self.accelerator.accumulate(model):
                outputs = model(**data_batch)
                loss = loss_fn(outputs, data_batch.get("labels", outputs))
        else:
            outputs = model(**data_batch)
            loss = loss_fn(outputs, data_batch.get("labels", outputs))
        
        forward_end = time.perf_counter()
        
        # Backward pass timing
        backward_start = time.perf_counter()
        
        if self.accelerator is not None:
            self.accelerator.backward(loss)
        else:
            loss.backward()
        
        backward_end = time.perf_counter()
        
        # Optimizer step timing
        optimizer_start = time.perf_counter()
        
        if self.accelerator is not None:
            # Gradient clipping if configured
            if hasattr(self.accelerator, 'clip_grad_norm_'):
                self.accelerator.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        optimizer.zero_grad()
        
        optimizer_end = time.perf_counter()
        
        total_end = time.perf_counter()
        
        # Calculate timing breakdown
        step_profile["timing_breakdown"] = {
            "forward_ms": (forward_end - forward_start) * 1000,
            "backward_ms": (backward_end - backward_start) * 1000,
            "optimizer_ms": (optimizer_end - optimizer_start) * 1000,
            "total_ms": (total_end - start_time) * 1000,
            "overhead_ms": (total_end - start_time - (forward_end - forward_start) - (backward_end - backward_start) - (optimizer_end - optimizer_start)) * 1000,
        }
        
        # Memory tracking
        memory_after = self._get_memory_snapshot()
        step_profile["memory_tracking"] = {
            "memory_before": memory_before,
            "memory_after": memory_after,
            "memory_delta": self._calculate_memory_delta(memory_before, memory_after),
        }
        
        # Gradient analysis
        if self.monitor_gradient_sync:
            step_profile["gradient_analysis"] = self._analyze_gradients(model)
        
        # Generate recommendations
        timing = step_profile["timing_breakdown"]
        if timing["overhead_ms"] > timing["forward_ms"]:
            step_profile["recommendations"].append("High overhead detected - check for synchronization issues")
        
        if timing["backward_ms"] > timing["forward_ms"] * 3:
            step_profile["recommendations"].append("Backward pass significantly slower than forward - check gradient computation")
        
        self.training_profiles.append(step_profile)
        
        if self.verbose:
            total_time = timing["total_ms"]
            self.logger.info(f"Training step '{step_name}' completed in {total_time:.2f}ms")
        
        return step_profile
    
    def analyze_dataloader_efficiency(
        self,
        dataloader,
        analysis_name: str = "dataloader_analysis",
        num_batches: int = 10,
    ) -> Dict[str, Any]:
        """Analyze dataloader efficiency and bottlenecks"""
        if not self.enabled:
            raise RuntimeError("AccelerateDebugger is not enabled")
        
        dataloader_analysis = {
            "analysis_name": analysis_name,
            "timestamp": time.time(),
            "dataloader_info": {},
            "timing_analysis": {},
            "batch_analysis": {},
            "efficiency_metrics": {},
            "recommendations": [],
        }
        
        # Get dataloader info
        try:
            dataloader_info = {
                "batch_size": getattr(dataloader, 'batch_size', 'unknown'),
                "num_workers": getattr(dataloader, 'num_workers', 'unknown'),
                "pin_memory": getattr(dataloader, 'pin_memory', 'unknown'),
                "drop_last": getattr(dataloader, 'drop_last', 'unknown'),
                "dataset_size": len(dataloader.dataset) if hasattr(dataloader, 'dataset') else 'unknown',
            }
            
            if self.accelerator is not None:
                dataloader_info.update({
                    "device": str(self.accelerator.device),
                    "num_processes": self.accelerator.num_processes,
                    "split_batches": getattr(dataloader, 'split_batches', 'unknown'),
                })
            
            dataloader_analysis["dataloader_info"] = dataloader_info
        except Exception as e:
            dataloader_analysis["dataloader_info"] = {"error": str(e)}
        
        # Time batch loading
        batch_times = []
        batch_sizes = []
        transfer_times = []
        
        try:
            iterator = iter(dataloader)
            
            for i in range(min(num_batches, len(dataloader))):
                # Time batch loading
                load_start = time.perf_counter()
                batch = next(iterator)
                load_end = time.perf_counter()
                
                batch_times.append((load_end - load_start) * 1000)  # ms
                
                # Analyze batch
                if isinstance(batch, dict):
                    batch_size = next(iter(batch.values())).shape[0] if batch else 0
                    
                    # Time device transfer if needed
                    transfer_start = time.perf_counter()
                    if self.accelerator is not None:
                        # This would normally be done by accelerator.prepare()
                        pass
                    transfer_end = time.perf_counter()
                    
                    transfer_times.append((transfer_end - transfer_start) * 1000)
                    
                elif isinstance(batch, (list, tuple)):
                    batch_size = batch[0].shape[0] if batch and hasattr(batch[0], 'shape') else 0
                else:
                    batch_size = batch.shape[0] if hasattr(batch, 'shape') else 0
                
                batch_sizes.append(batch_size)
                
        except Exception as e:
            dataloader_analysis["timing_analysis"] = {"error": str(e)}
            return dataloader_analysis
        
        # Calculate timing statistics
        if batch_times:
            import statistics
            
            dataloader_analysis["timing_analysis"] = {
                "mean_batch_load_ms": statistics.mean(batch_times),
                "std_batch_load_ms": statistics.stdev(batch_times) if len(batch_times) > 1 else 0,
                "min_batch_load_ms": min(batch_times),
                "max_batch_load_ms": max(batch_times),
                "mean_transfer_ms": statistics.mean(transfer_times) if transfer_times else 0,
                "total_samples": len(batch_times),
            }
            
            # Batch analysis
            if batch_sizes:
                dataloader_analysis["batch_analysis"] = {
                    "mean_batch_size": statistics.mean(batch_sizes),
                    "batch_size_consistency": len(set(batch_sizes)) == 1,
                    "actual_batch_sizes": batch_sizes if len(set(batch_sizes)) > 1 else None,
                }
            
            # Efficiency metrics
            mean_load_time = dataloader_analysis["timing_analysis"]["mean_batch_load_ms"]
            std_load_time = dataloader_analysis["timing_analysis"]["std_batch_load_ms"]
            
            dataloader_analysis["efficiency_metrics"] = {
                "loading_stability": 1 - (std_load_time / mean_load_time) if mean_load_time > 0 else 0,
                "batches_per_second": 1000 / mean_load_time if mean_load_time > 0 else 0,
            }
            
            # Generate recommendations
            if mean_load_time > 100:  # 100ms threshold
                dataloader_analysis["recommendations"].append(f"Slow batch loading ({mean_load_time:.1f}ms) - consider increasing num_workers")
            
            if std_load_time > mean_load_time * 0.5:
                dataloader_analysis["recommendations"].append("High variability in loading times - check for I/O bottlenecks")
            
            num_workers = dataloader_info.get("num_workers", 0)
            if isinstance(num_workers, int) and num_workers == 0:
                dataloader_analysis["recommendations"].append("Single-threaded data loading - consider using num_workers > 0")
            
            if not dataloader_info.get("pin_memory", False) and "cuda" in str(dataloader_info.get("device", "")):
                dataloader_analysis["recommendations"].append("pin_memory=False with CUDA - consider enabling for faster transfers")
        
        self.dataloader_profiles.append(dataloader_analysis)
        
        if self.verbose and "timing_analysis" in dataloader_analysis:
            mean_time = dataloader_analysis["timing_analysis"]["mean_batch_load_ms"]
            self.logger.info(f"Dataloader analysis '{analysis_name}': {mean_time:.2f}ms avg batch load time")
        
        return dataloader_analysis
    
    def _get_memory_snapshot(self) -> Dict[str, Any]:
        """Get current memory usage snapshot"""
        memory_info = {"timestamp": time.time()}
        
        try:
            import psutil
            memory_info["cpu_memory"] = {
                "available_gb": psutil.virtual_memory().available / (1024**3),
                "percent_used": psutil.virtual_memory().percent,
            }
        except:
            pass
        
        try:
            if torch.cuda.is_available():
                memory_info["gpu_memory"] = {}
                for i in range(torch.cuda.device_count()):
                    memory_info["gpu_memory"][f"device_{i}"] = {
                        "allocated_gb": torch.cuda.memory_allocated(i) / (1024**3),
                        "reserved_gb": torch.cuda.memory_reserved(i) / (1024**3),
                    }
        except:
            pass
        
        return memory_info
    
    def _calculate_memory_delta(self, before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate memory usage delta"""
        delta = {}
        
        if "gpu_memory" in before and "gpu_memory" in after:
            delta["gpu_memory"] = {}
            for device in before["gpu_memory"]:
                if device in after["gpu_memory"]:
                    delta["gpu_memory"][device] = {
                        "allocated_delta_gb": after["gpu_memory"][device]["allocated_gb"] - before["gpu_memory"][device]["allocated_gb"],
                        "reserved_delta_gb": after["gpu_memory"][device]["reserved_gb"] - before["gpu_memory"][device]["reserved_gb"],
                    }
        
        return delta
    
    def _analyze_gradients(self, model: nn.Module) -> Dict[str, Any]:
        """Analyze gradient statistics"""
        gradient_stats = {
            "total_parameters": 0,
            "parameters_with_gradients": 0,
            "gradient_norms": [],
            "zero_gradients": [],
            "large_gradients": [],
        }
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                gradient_stats["total_parameters"] += 1
                
                if param.grad is not None:
                    gradient_stats["parameters_with_gradients"] += 1
                    grad_norm = float(param.grad.norm())
                    gradient_stats["gradient_norms"].append(grad_norm)
                    
                    if grad_norm < 1e-8:
                        gradient_stats["zero_gradients"].append(name)
                    elif grad_norm > 100:
                        gradient_stats["large_gradients"].append(name)
        
        if gradient_stats["gradient_norms"]:
            import statistics
            gradient_stats["mean_gradient_norm"] = statistics.mean(gradient_stats["gradient_norms"])
            gradient_stats["max_gradient_norm"] = max(gradient_stats["gradient_norms"])
        
        return gradient_stats
    
    def get_accelerate_summary(self) -> Dict[str, Any]:
        """Get comprehensive Accelerate debugging summary"""
        return {
            "accelerate_available": self.accelerate_available,
            "accelerator_set": self.accelerator is not None,
            "device_analyses": len(self.device_analyses),
            "training_profiles": len(self.training_profiles),
            "dataloader_profiles": len(self.dataloader_profiles),
            "enabled": self.enabled,
        }
    
    def _save_debug_data(self) -> None:
        """Save Accelerate debug data"""
        debug_data = {
            "accelerate_available": self.accelerate_available,
            "device_analyses": self.device_analyses,
            "gradient_sync_data": self.gradient_sync_data,
            "dataloader_profiles": self.dataloader_profiles,
            "training_profiles": self.training_profiles,
            "timestamp": time.time(),
        }
        
        if self.accelerate_available:
            try:
                debug_data["accelerate_version"] = self.accelerate.__version__
            except:
                pass
        
        self.save_json(debug_data, "accelerate_debug_data.json")
        
        # Generate summary
        summary = self.get_accelerate_summary()
        self.save_json(summary, "accelerate_debug_summary.json")
        
        if self.verbose:
            self.logger.info(f"Accelerate debug data saved to {self.output_dir}")


def auto_accelerate_debug(
    accelerator=None,
    output_dir: Optional[Union[str, Path]] = None,
    **kwargs
) -> AccelerateDebugger:
    """
    Quick Accelerate debugging setup
    
    Args:
        accelerator: Accelerator instance to monitor
        output_dir: Output directory
        **kwargs: Additional arguments for AccelerateDebugger
    
    Returns:
        Enabled AccelerateDebugger instance
    
    Example:
        >>> from accelerate import Accelerator
        >>> accelerator = Accelerator()
        >>> acc_debugger = auto_accelerate_debug(accelerator)
        >>> device_analysis = acc_debugger.analyze_device_placement(model)
        >>> acc_debugger.disable()
    """
    debugger = AccelerateDebugger(output_dir=output_dir, **kwargs)
    debugger.enable()
    
    if accelerator is not None:
        debugger.set_accelerator(accelerator)
    
    return debugger