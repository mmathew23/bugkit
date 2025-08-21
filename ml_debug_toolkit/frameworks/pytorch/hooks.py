"""
PyTorch hook utilities for debugging
"""

from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn


class GradientHook:
    """Hook for capturing and analyzing gradients"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.gradient_data: Dict[str, List[torch.Tensor]] = {}
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
    
    def register(self, model: nn.Module, layer_names: Optional[List[str]] = None) -> None:
        """Register gradient hooks on model layers"""
        if layer_names is None:
            # Register on all parameters
            for name, param in model.named_parameters():
                if param.requires_grad:
                    hook = param.register_hook(self._make_gradient_hook(name))
                    self.hooks.append(hook)
        else:
            # Register on specific layers
            for name, module in model.named_modules():
                if name in layer_names:
                    for param_name, param in module.named_parameters():
                        if param.requires_grad:
                            full_name = f"{name}.{param_name}"
                            hook = param.register_hook(self._make_gradient_hook(full_name))
                            self.hooks.append(hook)
    
    def _make_gradient_hook(self, name: str) -> Callable:
        """Create a gradient hook for a specific parameter"""
        def hook(grad):
            if name not in self.gradient_data:
                self.gradient_data[name] = []
            
            self.gradient_data[name].append(grad.clone())
            
            if self.verbose:
                print(f"Gradient captured for {name}: norm={grad.norm():.6f}")
        
        return hook
    
    def get_gradient_norms(self) -> Dict[str, List[float]]:
        """Get gradient norms for all captured gradients"""
        norms = {}
        for name, grads in self.gradient_data.items():
            norms[name] = [float(grad.norm()) for grad in grads]
        return norms
    
    def clear(self) -> None:
        """Clear captured gradient data"""
        self.gradient_data.clear()
    
    def remove_hooks(self) -> None:
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


class ActivationHook:
    """Hook for capturing layer activations"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.activations: Dict[str, List[torch.Tensor]] = {}
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
    
    def register(self, model: nn.Module, layer_names: Optional[List[str]] = None) -> None:
        """Register forward hooks on model layers"""
        target_modules = {}
        
        if layer_names is None:
            # Register on all leaf modules
            for name, module in model.named_modules():
                if len(list(module.children())) == 0:  # Leaf modules
                    target_modules[name] = module
        else:
            # Register on specific layers
            for name, module in model.named_modules():
                if name in layer_names:
                    target_modules[name] = module
        
        for name, module in target_modules.items():
            hook = module.register_forward_hook(self._make_activation_hook(name))
            self.hooks.append(hook)
    
    def _make_activation_hook(self, name: str) -> Callable:
        """Create an activation hook for a specific layer"""
        def hook(module, input, output):
            if name not in self.activations:
                self.activations[name] = []
            
            # Store activation (detached to avoid memory issues)
            if isinstance(output, torch.Tensor):
                self.activations[name].append(output.detach().cpu())
            elif isinstance(output, (list, tuple)):
                # Store first tensor output for complex outputs
                for item in output:
                    if isinstance(item, torch.Tensor):
                        self.activations[name].append(item.detach().cpu())
                        break
            
            if self.verbose:
                if isinstance(output, torch.Tensor):
                    print(f"Activation captured for {name}: shape={output.shape}")
                else:
                    print(f"Activation captured for {name}: type={type(output)}")
        
        return hook
    
    def get_activation_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all captured activations"""
        stats = {}
        
        for name, activations in self.activations.items():
            if activations:
                # Use the most recent activation
                latest_activation = activations[-1]
                
                stats[name] = {
                    "shape": list(latest_activation.shape),
                    "mean": float(latest_activation.mean()),
                    "std": float(latest_activation.std()),
                    "min": float(latest_activation.min()),
                    "max": float(latest_activation.max()),
                    "zero_fraction": float((latest_activation == 0).float().mean()),
                    "capture_count": len(activations),
                }
        
        return stats
    
    def clear(self) -> None:
        """Clear captured activation data"""
        self.activations.clear()
    
    def remove_hooks(self) -> None:
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


def hook_model_layers(
    model: nn.Module,
    layer_names: Optional[List[str]] = None,
    capture_gradients: bool = True,
    capture_activations: bool = True,
    verbose: bool = False,
) -> Dict[str, Union[GradientHook, ActivationHook]]:
    """
    Convenience function to hook model layers for debugging
    
    Args:
        model: PyTorch model
        layer_names: Specific layer names to hook (None for all)
        capture_gradients: Whether to capture gradients
        capture_activations: Whether to capture activations
        verbose: Whether to print hook activity
    
    Returns:
        Dictionary containing the hooks
    
    Example:
        >>> hooks = hook_model_layers(model, capture_gradients=True, capture_activations=True)
        >>> # Run forward/backward pass
        >>> gradient_norms = hooks['gradient_hook'].get_gradient_norms()
        >>> activation_stats = hooks['activation_hook'].get_activation_statistics()
        >>> # Clean up
        >>> for hook in hooks.values():
        >>>     hook.remove_hooks()
    """
    
    hooks = {}
    
    if capture_gradients:
        grad_hook = GradientHook(verbose=verbose)
        grad_hook.register(model, layer_names)
        hooks['gradient_hook'] = grad_hook
    
    if capture_activations:
        act_hook = ActivationHook(verbose=verbose)
        act_hook.register(model, layer_names)
        hooks['activation_hook'] = act_hook
    
    return hooks