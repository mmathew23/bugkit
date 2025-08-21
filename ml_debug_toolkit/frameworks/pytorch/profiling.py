"""
PyTorch profiling utilities for performance analysis
"""

import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer

from ...core.base import BaseDebugTool


def profile_forward_pass(
    model: nn.Module,
    inputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
    num_runs: int = 100,
    warmup_runs: int = 10,
    detailed: bool = True,
) -> Dict[str, Any]:
    """
    Profile forward pass performance
    
    Args:
        model: PyTorch model
        inputs: Input tensor(s)
        num_runs: Number of profiling runs
        warmup_runs: Number of warmup runs
        detailed: Whether to include detailed profiling
    
    Returns:
        Profiling results
    
    Example:
        >>> inputs = torch.randn(32, 3, 224, 224)
        >>> results = profile_forward_pass(model, inputs, num_runs=50)
        >>> print(f"Average time: {results['timing']['mean_ms']:.2f}ms")
    """
    
    device = next(model.parameters()).device
    
    # Ensure inputs are on the same device
    if isinstance(inputs, torch.Tensor):
        inputs = inputs.to(device)
    elif isinstance(inputs, dict):
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    
    # Warmup
    model.eval()
    with torch.no_grad():
        for _ in range(warmup_runs):
            if isinstance(inputs, dict):
                _ = model(**inputs)
            else:
                _ = model(inputs)
            if device.type == "cuda":
                torch.cuda.synchronize()
    
    # Profiling runs
    timings = []
    
    with torch.no_grad():
        for _ in range(num_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            
            if isinstance(inputs, dict):
                outputs = model(**inputs)
            else:
                outputs = model(inputs)
            
            if device.type == "cuda":
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            timings.append((end_time - start_time) * 1000)  # Convert to ms
    
    # Calculate statistics
    import statistics
    
    results = {
        "timing": {
            "mean_ms": statistics.mean(timings),
            "median_ms": statistics.median(timings),
            "std_ms": statistics.stdev(timings) if len(timings) > 1 else 0,
            "min_ms": min(timings),
            "max_ms": max(timings),
            "p95_ms": sorted(timings)[int(0.95 * len(timings))],
            "p99_ms": sorted(timings)[int(0.99 * len(timings))],
        },
        "model_info": {
            "model_class": model.__class__.__name__,
            "parameters": sum(p.numel() for p in model.parameters()),
            "device": str(device),
        },
        "input_info": {
            "input_shape": inputs.shape if isinstance(inputs, torch.Tensor) else {k: v.shape for k, v in inputs.items() if isinstance(v, torch.Tensor)},
            "batch_size": inputs.shape[0] if isinstance(inputs, torch.Tensor) else next(iter(inputs.values())).shape[0],
        },
        "throughput": {
            "samples_per_second": (inputs.shape[0] if isinstance(inputs, torch.Tensor) else next(iter(inputs.values())).shape[0]) / (statistics.mean(timings) / 1000),
        },
        "config": {
            "num_runs": num_runs,
            "warmup_runs": warmup_runs,
        }
    }
    
    return results


def profile_backward_pass(
    model: nn.Module,
    loss_fn: Callable,
    inputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
    targets: torch.Tensor,
    num_runs: int = 50,
    warmup_runs: int = 5,
) -> Dict[str, Any]:
    """
    Profile backward pass performance
    
    Args:
        model: PyTorch model
        loss_fn: Loss function
        inputs: Input tensor(s)
        targets: Target tensors
        num_runs: Number of profiling runs
        warmup_runs: Number of warmup runs
    
    Returns:
        Backward pass profiling results
    """
    
    device = next(model.parameters()).device
    
    # Ensure inputs are on the same device
    if isinstance(inputs, torch.Tensor):
        inputs = inputs.to(device)
    elif isinstance(inputs, dict):
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    
    targets = targets.to(device)
    
    # Warmup
    model.train()
    for _ in range(warmup_runs):
        if isinstance(inputs, dict):
            outputs = model(**inputs)
        else:
            outputs = model(inputs)
        
        loss = loss_fn(outputs, targets)
        loss.backward()
        model.zero_grad()
        
        if device.type == "cuda":
            torch.cuda.synchronize()
    
    # Profiling runs
    forward_timings = []
    backward_timings = []
    total_timings = []
    
    for _ in range(num_runs):
        model.zero_grad()
        
        # Forward pass timing
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        forward_start = time.perf_counter()
        
        if isinstance(inputs, dict):
            outputs = model(**inputs)
        else:
            outputs = model(inputs)
        
        loss = loss_fn(outputs, targets)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        forward_end = time.perf_counter()
        forward_timings.append((forward_end - forward_start) * 1000)
        
        # Backward pass timing
        backward_start = time.perf_counter()
        
        loss.backward()
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        backward_end = time.perf_counter()
        backward_timings.append((backward_end - backward_start) * 1000)
        
        total_timings.append((backward_end - forward_start) * 1000)
    
    # Calculate statistics
    import statistics
    
    results = {
        "forward_timing": {
            "mean_ms": statistics.mean(forward_timings),
            "std_ms": statistics.stdev(forward_timings) if len(forward_timings) > 1 else 0,
            "min_ms": min(forward_timings),
            "max_ms": max(forward_timings),
        },
        "backward_timing": {
            "mean_ms": statistics.mean(backward_timings),
            "std_ms": statistics.stdev(backward_timings) if len(backward_timings) > 1 else 0,
            "min_ms": min(backward_timings),
            "max_ms": max(backward_timings),
        },
        "total_timing": {
            "mean_ms": statistics.mean(total_timings),
            "std_ms": statistics.stdev(total_timings) if len(total_timings) > 1 else 0,
            "min_ms": min(total_timings),
            "max_ms": max(total_timings),
        },
        "ratios": {
            "backward_to_forward_ratio": statistics.mean(backward_timings) / statistics.mean(forward_timings),
            "forward_percentage": (statistics.mean(forward_timings) / statistics.mean(total_timings)) * 100,
            "backward_percentage": (statistics.mean(backward_timings) / statistics.mean(total_timings)) * 100,
        },
        "model_info": {
            "model_class": model.__class__.__name__,
            "parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        },
        "config": {
            "num_runs": num_runs,
            "warmup_runs": warmup_runs,
        }
    }
    
    return results


def compare_optimizers(
    model: nn.Module,
    optimizers: List[Optimizer],
    optimizer_names: List[str],
    loss_fn: Callable,
    inputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
    targets: torch.Tensor,
    num_steps: int = 10,
) -> Dict[str, Any]:
    """
    Compare different optimizers
    
    Args:
        model: PyTorch model  
        optimizers: List of optimizers to compare
        optimizer_names: Names for the optimizers
        loss_fn: Loss function
        inputs: Input tensor(s)
        targets: Target tensors
        num_steps: Number of optimization steps
    
    Returns:
        Optimizer comparison results
    
    Example:
        >>> opt1 = torch.optim.Adam(model.parameters(), lr=0.001)
        >>> opt2 = torch.optim.SGD(model.parameters(), lr=0.01)
        >>> results = compare_optimizers(model, [opt1, opt2], ["Adam", "SGD"], loss_fn, inputs, targets)
    """
    
    if len(optimizers) != len(optimizer_names):
        raise ValueError("Number of optimizers must match number of names")
    
    device = next(model.parameters()).device
    
    # Ensure inputs are on the same device
    if isinstance(inputs, torch.Tensor):
        inputs = inputs.to(device)
    elif isinstance(inputs, dict):
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    
    targets = targets.to(device)
    
    # Store initial model state
    initial_state = {name: param.clone() for name, param in model.named_parameters()}
    
    comparison_results = {
        "optimizer_results": {},
        "convergence_comparison": {},
        "timing_comparison": {},
        "final_comparison": {},
        "config": {
            "num_steps": num_steps,
            "model_class": model.__class__.__name__,
        }
    }
    
    for opt_idx, (optimizer, opt_name) in enumerate(zip(optimizers, optimizer_names)):
        # Reset model to initial state
        for name, param in model.named_parameters():
            param.data.copy_(initial_state[name])
        
        optimizer.zero_grad()  # Reset optimizer state
        
        # Training steps
        step_results = {
            "optimizer_name": opt_name,
            "optimizer_class": optimizer.__class__.__name__,
            "hyperparameters": optimizer.defaults.copy(),
            "loss_history": [],
            "step_timings": [],
            "gradient_norms": [],
        }
        
        model.train()
        
        for step in range(num_steps):
            step_start = time.perf_counter()
            
            optimizer.zero_grad()
            
            if isinstance(inputs, dict):
                outputs = model(**inputs)
            else:
                outputs = model(inputs)
            
            loss = loss_fn(outputs, targets)
            loss.backward()
            
            # Calculate gradient norm
            total_norm = 0
            for param in model.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            
            optimizer.step()
            
            step_end = time.perf_counter()
            
            step_results["loss_history"].append(float(loss.item()))
            step_results["step_timings"].append((step_end - step_start) * 1000)
            step_results["gradient_norms"].append(total_norm)
        
        # Calculate final metrics
        step_results["final_loss"] = step_results["loss_history"][-1]
        step_results["loss_reduction"] = step_results["loss_history"][0] - step_results["loss_history"][-1]
        step_results["average_step_time_ms"] = sum(step_results["step_timings"]) / len(step_results["step_timings"])
        step_results["average_gradient_norm"] = sum(step_results["gradient_norms"]) / len(step_results["gradient_norms"])
        
        comparison_results["optimizer_results"][opt_name] = step_results
    
    # Generate comparisons
    optimizer_results = comparison_results["optimizer_results"]
    
    # Convergence comparison
    best_final_loss = min(result["final_loss"] for result in optimizer_results.values())
    best_loss_reduction = max(result["loss_reduction"] for result in optimizer_results.values())
    
    comparison_results["convergence_comparison"] = {
        "best_final_loss": {
            "optimizer": min(optimizer_results.items(), key=lambda x: x[1]["final_loss"])[0],
            "loss": best_final_loss,
        },
        "best_loss_reduction": {
            "optimizer": max(optimizer_results.items(), key=lambda x: x[1]["loss_reduction"])[0],
            "reduction": best_loss_reduction,
        },
        "convergence_ranking": sorted(
            [(name, result["loss_reduction"]) for name, result in optimizer_results.items()],
            key=lambda x: x[1],
            reverse=True
        ),
    }
    
    # Timing comparison
    fastest_optimizer = min(optimizer_results.items(), key=lambda x: x[1]["average_step_time_ms"])
    
    comparison_results["timing_comparison"] = {
        "fastest_optimizer": {
            "optimizer": fastest_optimizer[0],
            "average_time_ms": fastest_optimizer[1]["average_step_time_ms"],
        },
        "speed_ranking": sorted(
            [(name, result["average_step_time_ms"]) for name, result in optimizer_results.items()],
            key=lambda x: x[1]
        ),
    }
    
    # Final comparison summary
    comparison_results["final_comparison"] = {
        "recommended_optimizer": comparison_results["convergence_comparison"]["best_loss_reduction"]["optimizer"],
        "trade_offs": {},
    }
    
    # Analyze trade-offs
    for name, result in optimizer_results.items():
        comparison_results["final_comparison"]["trade_offs"][name] = {
            "convergence_rank": [i for i, (opt_name, _) in enumerate(comparison_results["convergence_comparison"]["convergence_ranking"]) if opt_name == name][0] + 1,
            "speed_rank": [i for i, (opt_name, _) in enumerate(comparison_results["timing_comparison"]["speed_ranking"]) if opt_name == name][0] + 1,
            "overall_score": result["loss_reduction"] / result["average_step_time_ms"],  # Simple score
        }
    
    return comparison_results