"""
Comprehensive benchmarking utilities for HuggingFace models
"""

import gc
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer

from ...core.base import BaseDebugTool
from ...tracing.chrome_tracer import ChromeTracer
from ..storage import DiskTensorStorage, MultiDtypeComparer


class ModelBenchmarker(BaseDebugTool):
    """Comprehensive model benchmarking with detailed analysis"""
    
    def __init__(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        verbose: bool = True,
        warmup_runs: int = 3,
        measure_memory: bool = True,
        measure_flops: bool = False,
        profile_detailed: bool = True,
    ):
        super().__init__(output_dir, verbose)
        self.warmup_runs = warmup_runs
        self.measure_memory = measure_memory
        self.measure_flops = measure_flops
        self.profile_detailed = profile_detailed
        
        self.benchmark_results: List[Dict[str, Any]] = []
        
        if profile_detailed:
            self.tracer = ChromeTracer(
                output_dir=self.output_dir / "benchmark_traces",
                verbose=verbose,
                include_memory=measure_memory,
                include_cuda=True,
            )
        else:
            self.tracer = None
    
    def enable(self) -> None:
        """Enable benchmarker"""
        self.enabled = True
        if self.tracer:
            self.tracer.enable()
        if self.verbose:
            self.logger.info("Model benchmarker enabled")
    
    def disable(self) -> None:
        """Disable benchmarker and save results"""
        self.enabled = False
        if self.tracer:
            self.tracer.disable()
        self._save_benchmark_results()
        if self.verbose:
            self.logger.info("Model benchmarker disabled")
    
    def benchmark_model(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        num_runs: int = 100,
        model_name: str = "model",
        batch_sizes: Optional[List[int]] = None,
        dtypes: Optional[List[torch.dtype]] = None,
        devices: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Comprehensive model benchmarking
        
        Args:
            model: Model to benchmark
            inputs: Input tensors
            num_runs: Number of benchmark runs
            model_name: Name for the model
            batch_sizes: Different batch sizes to test
            dtypes: Different dtypes to test
            devices: Different devices to test
        """
        if not self.enabled:
            raise RuntimeError("ModelBenchmarker is not enabled")
        
        benchmark_result = {
            "model_name": model_name,
            "model_class": model.__class__.__name__,
            "timestamp": time.time(),
            "parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "benchmark_configs": [],
            "summary": {},
        }
        
        # Default configurations
        batch_sizes = batch_sizes or [1]
        dtypes = dtypes or [torch.float32]
        devices = devices or ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        
        # Test each configuration
        for batch_size in batch_sizes:
            for dtype in dtypes:
                for device in devices:
                    try:
                        config_result = self._benchmark_single_config(
                            model, inputs, num_runs, batch_size, dtype, device
                        )
                        
                        config_result.update({
                            "batch_size": batch_size,
                            "dtype": str(dtype),
                            "device": device,
                        })
                        
                        benchmark_result["benchmark_configs"].append(config_result)
                        
                    except Exception as e:
                        if self.verbose:
                            self.logger.warning(
                                f"Benchmark failed for config (batch={batch_size}, "
                                f"dtype={dtype}, device={device}): {e}"
                            )
                        
                        benchmark_result["benchmark_configs"].append({
                            "batch_size": batch_size,
                            "dtype": str(dtype),
                            "device": device,
                            "error": str(e),
                            "failed": True,
                        })
        
        # Generate summary
        benchmark_result["summary"] = self._generate_benchmark_summary(benchmark_result)
        
        self.benchmark_results.append(benchmark_result)
        
        if self.verbose:
            self.logger.info(f"Benchmark completed for {model_name}")
        
        return benchmark_result
    
    def _benchmark_single_config(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        num_runs: int,
        batch_size: int,
        dtype: torch.dtype,
        device: str,
    ) -> Dict[str, Any]:
        """Benchmark a single configuration"""
        
        # Prepare model and inputs
        original_device = next(model.parameters()).device
        model = model.to(device).to(dtype)
        
        # Adjust batch size
        adjusted_inputs = {}
        for key, tensor in inputs.items():
            if tensor.dim() > 0:
                # Repeat or slice to match batch size
                current_batch = tensor.shape[0]
                if current_batch != batch_size:
                    if batch_size > current_batch:
                        # Repeat to increase batch size
                        repeat_factor = (batch_size + current_batch - 1) // current_batch
                        adjusted_tensor = tensor.repeat(repeat_factor, *([1] * (tensor.dim() - 1)))[:batch_size]
                    else:
                        # Slice to decrease batch size
                        adjusted_tensor = tensor[:batch_size]
                    adjusted_inputs[key] = adjusted_tensor.to(device).to(dtype)
                else:
                    adjusted_inputs[key] = tensor.to(device).to(dtype)
            else:
                adjusted_inputs[key] = tensor.to(device)
        
        # Memory measurement
        initial_memory = self._get_memory_usage(device)
        
        # Warmup
        model.eval()
        with torch.no_grad():
            for _ in range(self.warmup_runs):
                _ = model(**adjusted_inputs)
                if device == "cuda":
                    torch.cuda.synchronize()
        
        # Clear cache
        if device == "cuda":
            torch.cuda.empty_cache()
        
        # Benchmark runs
        timings = []
        memory_snapshots = []
        
        for run in range(num_runs):
            if self.measure_memory:
                pre_memory = self._get_memory_usage(device)
            
            # Time the forward pass
            if device == "cuda":
                torch.cuda.synchronize()
            
            if self.tracer and run == 0:  # Profile first run
                with self.tracer.trace(f"forward_pass_{batch_size}_{dtype}_{device}", "benchmark"):
                    start_time = time.perf_counter()
                    
                    with torch.no_grad():
                        outputs = model(**adjusted_inputs)
                    
                    if device == "cuda":
                        torch.cuda.synchronize()
                    
                    end_time = time.perf_counter()
            else:
                start_time = time.perf_counter()
                
                with torch.no_grad():
                    outputs = model(**adjusted_inputs)
                
                if device == "cuda":
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
            
            run_time = end_time - start_time
            timings.append(run_time)
            
            if self.measure_memory:
                post_memory = self._get_memory_usage(device)
                memory_snapshots.append({
                    "pre_memory_mb": pre_memory,
                    "post_memory_mb": post_memory,
                    "memory_delta_mb": post_memory - pre_memory,
                })
        
        # Calculate statistics
        import statistics
        
        timing_stats = {
            "mean_time_ms": statistics.mean(timings) * 1000,
            "median_time_ms": statistics.median(timings) * 1000,
            "std_time_ms": statistics.stdev(timings) * 1000 if len(timings) > 1 else 0,
            "min_time_ms": min(timings) * 1000,
            "max_time_ms": max(timings) * 1000,
            "throughput_samples_per_sec": batch_size / statistics.mean(timings),
            "all_timings_ms": [t * 1000 for t in timings],
        }
        
        memory_stats = {}
        if memory_snapshots:
            memory_deltas = [snap["memory_delta_mb"] for snap in memory_snapshots]
            memory_stats = {
                "mean_memory_delta_mb": statistics.mean(memory_deltas),
                "max_memory_delta_mb": max(memory_deltas),
                "min_memory_delta_mb": min(memory_deltas),
                "peak_memory_mb": initial_memory + max(memory_deltas),
            }
        
        # FLOPS estimation (simplified)
        flops_stats = {}
        if self.measure_flops:
            try:
                flops_stats = self._estimate_flops(model, adjusted_inputs, timing_stats["mean_time_ms"] / 1000)
            except Exception as e:
                flops_stats = {"error": str(e)}
        
        # Restore original device
        model = model.to(original_device)
        
        return {
            "timing_stats": timing_stats,
            "memory_stats": memory_stats,
            "flops_stats": flops_stats,
            "num_runs": num_runs,
            "warmup_runs": self.warmup_runs,
        }
    
    def _get_memory_usage(self, device: str) -> float:
        """Get memory usage in MB"""
        if device == "cuda" and torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 ** 2)
        else:
            # Approximate CPU memory (simplified)
            import psutil
            return psutil.Process().memory_info().rss / (1024 ** 2)
    
    def _estimate_flops(self, model: nn.Module, inputs: Dict[str, torch.Tensor], duration: float) -> Dict[str, Any]:
        """Estimate FLOPS (simplified calculation)"""
        try:
            # This is a very simplified FLOPS estimation
            # In practice, you'd use tools like torchprofile or fvcore
            
            total_params = sum(p.numel() for p in model.parameters())
            batch_size = next(iter(inputs.values())).shape[0]
            
            # Rough estimate: 2 * params * batch_size (forward pass)
            estimated_flops = 2 * total_params * batch_size
            
            return {
                "estimated_flops": estimated_flops,
                "flops_per_second": estimated_flops / duration,
                "gflops_per_second": estimated_flops / duration / 1e9,
                "note": "Simplified estimation, use specialized tools for accuracy",
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _generate_benchmark_summary(self, benchmark_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of benchmark results"""
        configs = benchmark_result["benchmark_configs"]
        successful_configs = [c for c in configs if not c.get("failed", False)]
        
        if not successful_configs:
            return {"error": "No successful benchmark configurations"}
        
        # Find best performing configurations
        best_throughput = max(
            successful_configs, 
            key=lambda x: x["timing_stats"]["throughput_samples_per_sec"]
        )
        
        best_latency = min(
            successful_configs,
            key=lambda x: x["timing_stats"]["mean_time_ms"]
        )
        
        # Memory analysis
        memory_efficient = None
        if any("memory_stats" in c and c["memory_stats"] for c in successful_configs):
            memory_efficient = min(
                [c for c in successful_configs if "memory_stats" in c and c["memory_stats"]],
                key=lambda x: x["memory_stats"]["peak_memory_mb"]
            )
        
        summary = {
            "total_configs": len(configs),
            "successful_configs": len(successful_configs),
            "failed_configs": len(configs) - len(successful_configs),
            "best_throughput": {
                "config": f"batch={best_throughput['batch_size']}, dtype={best_throughput['dtype']}, device={best_throughput['device']}",
                "throughput_samples_per_sec": best_throughput["timing_stats"]["throughput_samples_per_sec"],
            },
            "best_latency": {
                "config": f"batch={best_latency['batch_size']}, dtype={best_latency['dtype']}, device={best_latency['device']}",
                "latency_ms": best_latency["timing_stats"]["mean_time_ms"],
            },
        }
        
        if memory_efficient:
            summary["most_memory_efficient"] = {
                "config": f"batch={memory_efficient['batch_size']}, dtype={memory_efficient['dtype']}, device={memory_efficient['device']}",
                "peak_memory_mb": memory_efficient["memory_stats"]["peak_memory_mb"],
            }
        
        return summary
    
    def _save_benchmark_results(self) -> None:
        """Save benchmark results"""
        if self.benchmark_results:
            self.save_json(self.benchmark_results, "benchmark_results.json")
            
            # Generate comparative analysis
            if len(self.benchmark_results) > 1:
                comparison = self._compare_benchmark_results()
                self.save_json(comparison, "benchmark_comparison.json")
            
            if self.verbose:
                self.logger.info(f"Saved {len(self.benchmark_results)} benchmark results")
    
    def _compare_benchmark_results(self) -> Dict[str, Any]:
        """Compare multiple benchmark results"""
        comparison = {
            "models_compared": [r["model_name"] for r in self.benchmark_results],
            "performance_ranking": {},
            "efficiency_analysis": {},
            "recommendations": [],
        }
        
        # Compare best configurations for each model
        model_best_configs = {}
        for result in self.benchmark_results:
            model_name = result["model_name"]
            best_config = max(
                [c for c in result["benchmark_configs"] if not c.get("failed", False)],
                key=lambda x: x["timing_stats"]["throughput_samples_per_sec"],
                default=None
            )
            if best_config:
                model_best_configs[model_name] = best_config
        
        # Rank by throughput
        throughput_ranking = sorted(
            model_best_configs.items(),
            key=lambda x: x[1]["timing_stats"]["throughput_samples_per_sec"],
            reverse=True
        )
        
        comparison["performance_ranking"]["throughput"] = [
            {
                "model": model_name,
                "throughput_samples_per_sec": config["timing_stats"]["throughput_samples_per_sec"],
                "config": f"batch={config['batch_size']}, dtype={config['dtype']}, device={config['device']}",
            }
            for model_name, config in throughput_ranking
        ]
        
        return comparison


def benchmark_model(
    model: nn.Module,
    inputs: Dict[str, torch.Tensor],
    num_runs: int = 100,
    model_name: Optional[str] = None,
    output_dir: Optional[Union[str, Path]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Quick model benchmarking function
    
    Args:
        model: Model to benchmark
        inputs: Input tensors
        num_runs: Number of benchmark runs
        model_name: Name for the model
        output_dir: Output directory
        **kwargs: Additional arguments for ModelBenchmarker
    
    Returns:
        Benchmark results dictionary
    
    Example:
        >>> inputs = {"input_ids": torch.randint(0, 1000, (1, 512))}
        >>> results = benchmark_model(model, inputs, num_runs=50)
        >>> print(f"Throughput: {results['summary']['best_throughput']['throughput_samples_per_sec']:.2f} samples/sec")
    """
    if model_name is None:
        model_name = model.__class__.__name__
    
    benchmarker = ModelBenchmarker(output_dir=output_dir, **kwargs)
    benchmarker.enable()
    
    try:
        results = benchmarker.benchmark_model(model, inputs, num_runs, model_name)
        return results
    finally:
        benchmarker.disable()


def compare_models(
    models: List[nn.Module],
    inputs: Dict[str, torch.Tensor],
    model_names: Optional[List[str]] = None,
    num_runs: int = 100,
    output_dir: Optional[Union[str, Path]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Compare multiple models
    
    Args:
        models: List of models to compare
        inputs: Input tensors
        model_names: Names for the models
        num_runs: Number of benchmark runs per model
        output_dir: Output directory
        **kwargs: Additional arguments for ModelBenchmarker
    
    Returns:
        Comparison results dictionary
    
    Example:
        >>> models = [model_baseline, model_optimized]
        >>> names = ["baseline", "optimized"]
        >>> comparison = compare_models(models, inputs, names, num_runs=50)
        >>> print("Best model:", comparison['performance_ranking']['throughput'][0]['model'])
    """
    if model_names is None:
        model_names = [f"model_{i}" for i in range(len(models))]
    
    if len(models) != len(model_names):
        raise ValueError("Number of models must match number of model names")
    
    benchmarker = ModelBenchmarker(output_dir=output_dir, **kwargs)
    benchmarker.enable()
    
    try:
        # Benchmark each model
        for model, name in zip(models, model_names):
            benchmarker.benchmark_model(model, inputs, num_runs, name)
        
        # Get comparison results
        comparison = benchmarker._compare_benchmark_results()
        return comparison
        
    finally:
        benchmarker.disable()


def benchmark_training_step(
    model: nn.Module,
    optimizer: Optimizer,
    loss_fn: Callable,
    batch: Dict[str, torch.Tensor],
    labels: Optional[torch.Tensor] = None,
    num_runs: int = 10,
    output_dir: Optional[Union[str, Path]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Benchmark a full training step
    
    Args:
        model: Model to train
        optimizer: Optimizer
        loss_fn: Loss function
        batch: Training batch
        labels: Labels (if not in batch)
        num_runs: Number of training steps to benchmark
        output_dir: Output directory
        **kwargs: Additional arguments
    
    Returns:
        Training step benchmark results
    
    Example:
        >>> batch = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        >>> results = benchmark_training_step(model, optimizer, loss_fn, batch)
        >>> print(f"Training step time: {results['timing_stats']['mean_time_ms']:.2f}ms")
    """
    
    benchmarker = ModelBenchmarker(output_dir=output_dir, **kwargs)
    benchmarker.enable()
    
    try:
        device = next(model.parameters()).device
        
        # Prepare batch
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        if labels is not None:
            batch["labels"] = labels.to(device)
        
        # Warmup
        model.train()
        for _ in range(3):
            optimizer.zero_grad()
            outputs = model(**batch)
            
            if hasattr(outputs, 'loss'):
                loss = outputs.loss
            else:
                # Assume outputs is logits and we need to calculate loss
                logits = outputs if isinstance(outputs, torch.Tensor) else outputs[0]
                loss = loss_fn(logits, batch.get("labels"))
            
            loss.backward()
            optimizer.step()
        
        # Benchmark training steps
        timings = []
        memory_snapshots = []
        loss_values = []
        
        for run in range(num_runs):
            pre_memory = benchmarker._get_memory_usage("cuda" if device.type == "cuda" else "cpu")
            
            # Time full training step
            if device.type == "cuda":
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(**batch)
            
            # Loss calculation
            if hasattr(outputs, 'loss'):
                loss = outputs.loss
            else:
                logits = outputs if isinstance(outputs, torch.Tensor) else outputs[0]
                loss = loss_fn(logits, batch.get("labels"))
            
            # Backward pass
            loss.backward()
            
            # Optimizer step
            optimizer.step()
            
            if device.type == "cuda":
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            
            # Record metrics
            run_time = end_time - start_time
            timings.append(run_time)
            loss_values.append(float(loss.item()))
            
            post_memory = benchmarker._get_memory_usage("cuda" if device.type == "cuda" else "cpu")
            memory_snapshots.append({
                "pre_memory_mb": pre_memory,
                "post_memory_mb": post_memory,
                "memory_delta_mb": post_memory - pre_memory,
            })
        
        # Calculate statistics
        import statistics
        
        results = {
            "training_step_benchmark": {
                "num_runs": num_runs,
                "timing_stats": {
                    "mean_time_ms": statistics.mean(timings) * 1000,
                    "median_time_ms": statistics.median(timings) * 1000,
                    "std_time_ms": statistics.stdev(timings) * 1000 if len(timings) > 1 else 0,
                    "min_time_ms": min(timings) * 1000,
                    "max_time_ms": max(timings) * 1000,
                },
                "loss_stats": {
                    "mean_loss": statistics.mean(loss_values),
                    "std_loss": statistics.stdev(loss_values) if len(loss_values) > 1 else 0,
                    "min_loss": min(loss_values),
                    "max_loss": max(loss_values),
                    "loss_trend": "decreasing" if loss_values[-1] < loss_values[0] else "increasing" if loss_values[-1] > loss_values[0] else "stable",
                },
                "memory_stats": {
                    "mean_memory_delta_mb": statistics.mean([s["memory_delta_mb"] for s in memory_snapshots]),
                    "max_memory_delta_mb": max([s["memory_delta_mb"] for s in memory_snapshots]),
                    "peak_memory_mb": max([s["post_memory_mb"] for s in memory_snapshots]),
                },
                "model_info": {
                    "model_class": model.__class__.__name__,
                    "parameters": sum(p.numel() for p in model.parameters()),
                    "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                    "device": str(device),
                },
            }
        }
        
        return results
        
    finally:
        benchmarker.disable()