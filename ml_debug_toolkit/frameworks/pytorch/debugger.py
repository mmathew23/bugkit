"""
PyTorch-specific debugging utilities
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer

from ...core.base import BaseDebugTool
from ...core.logger import IOLogger
from ...tracing.chrome_tracer import ChromeTracer
from ..storage import DiskTensorStorage, MultiDtypeComparer


class PyTorchDebugger(BaseDebugTool):
    """General PyTorch model and training debugger"""
    
    def __init__(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        verbose: bool = True,
        track_gradients: bool = True,
        track_parameters: bool = True,
        track_buffers: bool = True,
        trace_execution: bool = True,
    ):
        super().__init__(output_dir, verbose)
        self.track_gradients = track_gradients
        self.track_parameters = track_parameters
        self.track_buffers = track_buffers
        self.trace_execution = trace_execution
        
        # Initialize components
        self.io_logger = IOLogger(
            output_dir=self.output_dir / "io_logs",
            verbose=verbose,
            track_gradients=track_gradients,
        )
        
        if trace_execution:
            self.tracer = ChromeTracer(
                output_dir=self.output_dir / "traces",
                verbose=verbose,
                include_memory=True,
                include_cuda=True,
            )
        else:
            self.tracer = None
        
        self.debug_data: Dict[str, Any] = {
            "models": {},
            "optimizers": {},
            "training_steps": {},
            "gradient_analysis": {},
            "parameter_analysis": {},
        }
    
    def enable(self) -> None:
        """Enable PyTorch debugger"""
        self.enabled = True
        self.io_logger.enable()
        if self.tracer:
            self.tracer.enable()
        
        if self.verbose:
            self.logger.info("PyTorch debugger enabled")
    
    def disable(self) -> None:
        """Disable debugger and save results"""
        self.enabled = False
        self.io_logger.disable()
        if self.tracer:
            self.tracer.disable()
        
        self._save_debug_data()
        
        if self.verbose:
            self.logger.info("PyTorch debugger disabled")
    
    def debug_model(
        self,
        model: nn.Module,
        model_name: str = "model",
        analyze_architecture: bool = True,
        check_initialization: bool = True,
    ) -> Dict[str, Any]:
        """Debug a PyTorch model"""
        if not self.enabled:
            raise RuntimeError("PyTorchDebugger is not enabled")
        
        model_debug = {
            "model_name": model_name,
            "model_class": model.__class__.__name__,
            "timestamp": time.time(),
            "basic_info": {},
            "architecture_analysis": {},
            "initialization_analysis": {},
            "parameter_analysis": {},
        }
        
        # Basic model info
        model_debug["basic_info"] = {
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "total_buffers": sum(b.numel() for b in model.buffers()),
            "device": str(next(model.parameters()).device),
            "dtype": str(next(model.parameters()).dtype),
            "training_mode": model.training,
        }
        
        # Architecture analysis
        if analyze_architecture:
            model_debug["architecture_analysis"] = self._analyze_model_architecture(model)
        
        # Initialization analysis
        if check_initialization:
            model_debug["initialization_analysis"] = self._analyze_initialization(model)
        
        # Parameter analysis
        if self.track_parameters:
            model_debug["parameter_analysis"] = self._analyze_parameters(model)
        
        self.debug_data["models"][model_name] = model_debug
        
        if self.verbose:
            self.logger.info(f"Model debug completed for {model_name}")
        
        return model_debug
    
    def debug_optimizer(
        self,
        optimizer: Optimizer,
        optimizer_name: str = "optimizer",
    ) -> Dict[str, Any]:
        """Debug a PyTorch optimizer"""
        if not self.enabled:
            raise RuntimeError("PyTorchDebugger is not enabled")
        
        optimizer_debug = {
            "optimizer_name": optimizer_name,
            "optimizer_class": optimizer.__class__.__name__,
            "timestamp": time.time(),
            "hyperparameters": {},
            "parameter_groups": [],
            "state_analysis": {},
        }
        
        # Extract hyperparameters
        if hasattr(optimizer, 'defaults'):
            optimizer_debug["hyperparameters"] = optimizer.defaults.copy()
        
        # Analyze parameter groups
        for i, group in enumerate(optimizer.param_groups):
            group_info = {
                "group_index": i,
                "num_parameters": len(group['params']),
                "hyperparameters": {k: v for k, v in group.items() if k != 'params'},
                "parameter_shapes": [list(p.shape) for p in group['params']],
                "total_elements": sum(p.numel() for p in group['params']),
            }
            optimizer_debug["parameter_groups"].append(group_info)
        
        # Analyze optimizer state
        optimizer_debug["state_analysis"] = self._analyze_optimizer_state(optimizer)
        
        self.debug_data["optimizers"][optimizer_name] = optimizer_debug
        
        if self.verbose:
            self.logger.info(f"Optimizer debug completed for {optimizer_name}")
        
        return optimizer_debug
    
    def debug_training_step(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        loss_fn: callable,
        inputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        step_name: str = "training_step",
        analyze_gradients: bool = True,
    ) -> Dict[str, Any]:
        """Debug a complete training step"""
        if not self.enabled:
            raise RuntimeError("PyTorchDebugger is not enabled")
        
        step_debug = {
            "step_name": step_name,
            "timestamp": time.time(),
            "model_info": {
                "model_class": model.__class__.__name__,
                "parameters": sum(p.numel() for p in model.parameters()),
                "training_mode": model.training,
            },
            "optimizer_info": {
                "optimizer_class": optimizer.__class__.__name__,
                "parameter_groups": len(optimizer.param_groups),
            },
            "forward_pass": {},
            "loss_info": {},
            "backward_pass": {},
            "optimizer_step": {},
            "gradient_analysis": {},
        }
        
        # Forward pass
        if self.tracer:
            with self.tracer.trace("forward_pass", "training"):
                model.train()
                outputs = model(**inputs)
                loss = loss_fn(outputs, targets)
        else:
            model.train()
            outputs = model(**inputs)
            loss = loss_fn(outputs, targets)
        
        step_debug["forward_pass"] = {
            "input_shapes": {k: list(v.shape) for k, v in inputs.items()},
            "output_shape": list(outputs.shape) if hasattr(outputs, 'shape') else "complex_output",
            "target_shape": list(targets.shape),
        }
        
        step_debug["loss_info"] = {
            "loss_value": float(loss.item()),
            "loss_dtype": str(loss.dtype), 
            "loss_device": str(loss.device),
        }
        
        # Backward pass
        optimizer.zero_grad()
        
        if self.tracer:
            with self.tracer.trace("backward_pass", "training"):
                loss.backward()
        else:
            loss.backward()
        
        # Gradient analysis
        if analyze_gradients and self.track_gradients:
            step_debug["gradient_analysis"] = self._analyze_gradients(model)
        
        # Optimizer step
        if self.tracer:
            with self.tracer.trace("optimizer_step", "training"):
                optimizer.step()
        else:
            optimizer.step()
        
        step_debug["optimizer_step"] = {
            "step_completed": True,
            "post_step_loss": float(loss.item()),  # Same as pre-step for this iteration
        }
        
        self.debug_data["training_steps"][step_name] = step_debug
        
        if self.verbose:
            self.logger.info(f"Training step debug completed for {step_name}")
        
        return step_debug
    
    def _analyze_model_architecture(self, model: nn.Module) -> Dict[str, Any]:
        """Analyze model architecture"""
        architecture = {
            "total_layers": 0,
            "layer_types": {},
            "depth_analysis": {},
            "connection_analysis": {},
        }
        
        # Count layers and types
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules
                architecture["total_layers"] += 1
                module_type = module.__class__.__name__
                architecture["layer_types"][module_type] = architecture["layer_types"].get(module_type, 0) + 1
        
        # Analyze depth
        max_depth = 0
        for name, _ in model.named_modules():
            depth = len(name.split('.')) if name else 0
            max_depth = max(max_depth, depth)
        
        architecture["depth_analysis"] = {
            "max_depth": max_depth,
            "has_skip_connections": self._detect_skip_connections(model),
            "has_residual_blocks": self._detect_residual_blocks(model),
        }
        
        return architecture
    
    def _detect_skip_connections(self, model: nn.Module) -> bool:
        """Simple heuristic to detect skip connections"""
        # Look for common skip connection patterns in module names
        skip_patterns = ["skip", "residual", "shortcut", "identity"]
        
        for name, _ in model.named_modules():
            if any(pattern in name.lower() for pattern in skip_patterns):
                return True
        
        return False
    
    def _detect_residual_blocks(self, model: nn.Module) -> bool:
        """Simple heuristic to detect residual blocks"""
        residual_patterns = ["resnet", "residual", "basicblock", "bottleneck"]
        
        for module in model.modules():
            module_name = module.__class__.__name__.lower()
            if any(pattern in module_name for pattern in residual_patterns):
                return True
        
        return False
    
    def _analyze_initialization(self, model: nn.Module) -> Dict[str, Any]:
        """Analyze parameter initialization"""
        init_analysis = {
            "parameter_statistics": {},
            "potential_issues": [],
            "recommendations": [],
        }
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                param_stats = {
                    "shape": list(param.shape),
                    "mean": float(param.mean()),
                    "std": float(param.std()),
                    "min": float(param.min()),
                    "max": float(param.max()),
                    "zero_fraction": float((param == 0).float().mean()),
                }
                
                init_analysis["parameter_statistics"][name] = param_stats
                
                # Check for common initialization issues
                if param_stats["std"] < 1e-6:
                    init_analysis["potential_issues"].append(f"Very small std in {name}: {param_stats['std']}")
                
                if param_stats["std"] > 10:
                    init_analysis["potential_issues"].append(f"Very large std in {name}: {param_stats['std']}")
                
                if param_stats["zero_fraction"] > 0.9:
                    init_analysis["potential_issues"].append(f"High zero fraction in {name}: {param_stats['zero_fraction']}")
        
        # Generate recommendations
        if init_analysis["potential_issues"]:
            init_analysis["recommendations"].append("Consider reviewing parameter initialization scheme")
        
        return init_analysis
    
    def _analyze_parameters(self, model: nn.Module) -> Dict[str, Any]:
        """Analyze model parameters"""
        param_analysis = {
            "parameter_count_by_type": {},
            "parameter_size_distribution": {},
            "gradient_requirements": {},
        }
        
        # Count parameters by layer type
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules
                module_type = module.__class__.__name__
                param_count = sum(p.numel() for p in module.parameters())
                
                if module_type not in param_analysis["parameter_count_by_type"]:
                    param_analysis["parameter_count_by_type"][module_type] = {
                        "total_params": 0,
                        "layer_count": 0,
                    }
                
                param_analysis["parameter_count_by_type"][module_type]["total_params"] += param_count
                param_analysis["parameter_count_by_type"][module_type]["layer_count"] += 1
        
        # Analyze gradient requirements
        total_params = 0
        trainable_params = 0
        frozen_params = 0
        
        for param in model.parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
            else:
                frozen_params += param.numel()
        
        param_analysis["gradient_requirements"] = {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "frozen_parameters": frozen_params,
            "trainable_ratio": trainable_params / total_params if total_params > 0 else 0,
        }
        
        return param_analysis
    
    def _analyze_optimizer_state(self, optimizer: Optimizer) -> Dict[str, Any]:
        """Analyze optimizer state"""
        state_analysis = {
            "has_state": len(optimizer.state) > 0,
            "state_keys": [],
            "state_statistics": {},
        }
        
        if optimizer.state:
            # Get common state keys
            all_keys = set()
            for param_state in optimizer.state.values():
                all_keys.update(param_state.keys())
            
            state_analysis["state_keys"] = list(all_keys)
            
            # Analyze state statistics
            for key in all_keys:
                values = []
                for param, param_state in optimizer.state.items():
                    if key in param_state and isinstance(param_state[key], torch.Tensor):
                        values.append(param_state[key])
                
                if values:
                    stacked = torch.stack([v.flatten() for v in values])
                    state_analysis["state_statistics"][key] = {
                        "mean": float(stacked.mean()),
                        "std": float(stacked.std()),
                        "min": float(stacked.min()),
                        "max": float(stacked.max()),
                    }
        
        return state_analysis
    
    def _analyze_gradients(self, model: nn.Module) -> Dict[str, Any]:
        """Analyze gradients after backward pass"""
        gradient_analysis = {
            "gradient_statistics": {},
            "gradient_flow": {},
            "potential_issues": [],
        }
        
        gradient_norms = []
        zero_grad_params = []
        large_grad_params = []
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = float(param.grad.norm())
                gradient_norms.append(grad_norm)
                
                grad_stats = {
                    "norm": grad_norm,
                    "mean": float(param.grad.mean()),
                    "std": float(param.grad.std()),
                    "min": float(param.grad.min()),
                    "max": float(param.grad.max()),
                }
                
                gradient_analysis["gradient_statistics"][name] = grad_stats
                
                # Check for issues
                if grad_norm < 1e-7:
                    zero_grad_params.append(name)
                elif grad_norm > 100:
                    large_grad_params.append(name)
        
        # Overall gradient flow analysis
        if gradient_norms:
            gradient_analysis["gradient_flow"] = {
                "mean_gradient_norm": float(sum(gradient_norms) / len(gradient_norms)),
                "max_gradient_norm": float(max(gradient_norms)),
                "min_gradient_norm": float(min(gradient_norms)),
                "gradient_norm_std": float(torch.tensor(gradient_norms).std()),
            }
        
        # Issues
        if zero_grad_params:
            gradient_analysis["potential_issues"].append(f"Zero gradients in: {', '.join(zero_grad_params[:5])}")
        
        if large_grad_params:
            gradient_analysis["potential_issues"].append(f"Large gradients in: {', '.join(large_grad_params[:5])}")
        
        return gradient_analysis
    
    def _save_debug_data(self) -> None:
        """Save all debug data"""
        if any(self.debug_data.values()):
            self.save_json(self.debug_data, "pytorch_debug_data.json")
            
            # Generate summary
            summary = {
                "models_debugged": len(self.debug_data["models"]),
                "optimizers_debugged": len(self.debug_data["optimizers"]),
                "training_steps_debugged": len(self.debug_data["training_steps"]),
                "timestamp": time.time(),
            }
            
            self.save_json(summary, "pytorch_debug_summary.json")
            
            if self.verbose:
                self.logger.info(f"PyTorch debug data saved to {self.output_dir}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get debugger statistics"""
        return {
            "models_debugged": len(self.debug_data["models"]),
            "optimizers_debugged": len(self.debug_data["optimizers"]),
            "training_steps_debugged": len(self.debug_data["training_steps"]),
            "enabled": self.enabled,
        }


def auto_debug_module(
    model: nn.Module,
    model_name: str = "model",
    output_dir: Optional[Union[str, Path]] = None,
    **kwargs
) -> PyTorchDebugger:
    """
    Quick PyTorch model debugging setup
    
    Args:
        model: PyTorch model to debug
        model_name: Name for the model
        output_dir: Output directory
        **kwargs: Additional arguments for PyTorchDebugger
    
    Returns:
        Enabled PyTorchDebugger instance
    
    Example:
        >>> debugger = auto_debug_module(model, "my_model")
        >>> debugger.debug_model(model)
        >>> debugger.disable()
    """
    debugger = PyTorchDebugger(output_dir=output_dir, **kwargs)
    debugger.enable()
    return debugger