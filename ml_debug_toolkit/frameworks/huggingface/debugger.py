"""
HuggingFace model debugger with comprehensive analysis capabilities
"""

import functools
import inspect
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Union

import torch
import torch.nn as nn

from ...core.base import BaseDebugTool
from ...core.logger import IOLogger
from ...tracing.chrome_tracer import ChromeTracer
from ..storage import DiskTensorStorage, MultiDtypeComparer


class HuggingFaceDebugger(BaseDebugTool):
    """Comprehensive debugger for HuggingFace models with minimal setup"""
    
    def __init__(
        self,
        model: nn.Module,
        output_dir: Optional[Union[str, Path]] = None,
        verbose: bool = True,
        capture_level: str = "model",  # "model", "layer", "attention", "all"
        storage_mode: str = "memory",  # "memory", "disk", "auto"
        trace_execution: bool = True,
        compare_dtypes: bool = False,
        max_memory_mb: int = 1024,
        capture_gradients: bool = False,
        capture_attention: bool = True,
    ):
        super().__init__(output_dir, verbose)
        
        self.model = model
        self.capture_level = capture_level
        self.storage_mode = storage_mode
        self.trace_execution = trace_execution
        self.compare_dtypes = compare_dtypes
        self.max_memory_mb = max_memory_mb
        self.capture_gradients = capture_gradients
        self.capture_attention = capture_attention
        
        # Initialize debugging components
        self.io_logger = IOLogger(
            output_dir=self.output_dir / "io_logs",
            verbose=verbose,
            track_memory=True,
            track_gradients=capture_gradients,
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
        
        if storage_mode in ["disk", "auto"]:
            self.storage = DiskTensorStorage(
                storage_dir=self.output_dir / "tensors",
                max_memory_mb=max_memory_mb,
                compress=True,
            )
        else:
            self.storage = None
        
        if compare_dtypes:
            self.dtype_comparer = MultiDtypeComparer(
                output_dir=self.output_dir / "dtype_analysis",
                verbose=verbose,
            )
        else:
            self.dtype_comparer = None
        
        # Hook management
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        self.captured_data: Dict[str, Any] = {
            "inputs": {},
            "outputs": {},
            "intermediate": {},
            "attention": {},
            "gradients": {},
            "metadata": {},
        }
        
        # Model analysis
        self.model_info = self._analyze_model_structure()
        
    def enable(self) -> None:
        """Enable debugging with automatic hook registration"""
        self.enabled = True
        
        # Enable all components
        self.io_logger.enable()
        if self.tracer:
            self.tracer.enable()
        if self.storage:
            self.storage.enable()
        if self.dtype_comparer:
            self.dtype_comparer.enable()
        
        # Register hooks based on capture level
        self._register_hooks()
        
        if self.verbose:
            self.logger.info(f"HuggingFace debugger enabled for {self.model.__class__.__name__}")
            self.logger.info(f"Capture level: {self.capture_level}, Storage: {self.storage_mode}")
            self.logger.info(f"Registered {len(self.hooks)} hooks")
    
    def disable(self) -> None:
        """Disable debugging and save all results"""
        self.enabled = False
        
        # Remove all hooks
        self._remove_hooks()
        
        # Disable components and save results
        self.io_logger.disable()
        if self.tracer:
            self.tracer.disable()
        if self.storage:
            self.storage.disable()
        if self.dtype_comparer:
            self.dtype_comparer.disable()
        
        # Save comprehensive analysis
        self._save_debug_session()
        
        if self.verbose:
            self.logger.info("HuggingFace debugger disabled and results saved")
    
    def _analyze_model_structure(self) -> Dict[str, Any]:
        """Analyze model structure for intelligent hook placement"""
        model_info = {
            "model_class": self.model.__class__.__name__,
            "total_parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            "layers": [],
            "attention_layers": [],
            "embedding_layers": [],
            "output_layers": [],
            "special_layers": {},
        }
        
        # Analyze each named module
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                layer_info = {
                    "name": name,
                    "type": module.__class__.__name__,
                    "parameters": sum(p.numel() for p in module.parameters()),
                }
                
                model_info["layers"].append(layer_info)
                
                # Classify layer types
                if "attention" in name.lower() or "attn" in module.__class__.__name__.lower():
                    model_info["attention_layers"].append(layer_info)
                elif "embed" in name.lower() or "Embed" in module.__class__.__name__:
                    model_info["embedding_layers"].append(layer_info)
                elif "output" in name.lower() or "head" in name.lower():
                    model_info["output_layers"].append(layer_info)
        
        # Detect model architecture
        model_info["architecture"] = self._detect_architecture()
        
        return model_info
    
    def _detect_architecture(self) -> str:
        """Detect HuggingFace model architecture"""
        model_name = self.model.__class__.__name__.lower()
        
        if "bert" in model_name:
            return "bert"
        elif "gpt" in model_name:
            return "gpt"
        elif "t5" in model_name:
            return "t5"
        elif "roberta" in model_name:
            return "roberta"
        elif "distilbert" in model_name:
            return "distilbert"
        elif "xlm" in model_name:
            return "xlm"
        elif "electra" in model_name:
            return "electra"
        elif "deberta" in model_name:
            return "deberta"
        elif "longformer" in model_name:
            return "longformer"
        elif "reformer" in model_name:
            return "reformer"
        else:
            return "unknown"
    
    def _register_hooks(self) -> None:
        """Register hooks based on capture level"""
        if self.capture_level == "model":
            self._register_model_level_hooks()
        elif self.capture_level == "layer":
            self._register_layer_level_hooks()
        elif self.capture_level == "attention":
            self._register_attention_level_hooks()
        elif self.capture_level == "all":
            self._register_comprehensive_hooks()
    
    def _register_model_level_hooks(self) -> None:
        """Register hooks at model level only"""
        # Forward hook for main model
        def model_forward_hook(module, input, output):
            self._capture_io("model", "forward", input, output, module)
        
        handle = self.model.register_forward_hook(model_forward_hook)
        self.hooks.append(handle)
        
        # Backward hook if capturing gradients
        if self.capture_gradients:
            def model_backward_hook(module, grad_input, grad_output):
                self._capture_gradients("model", grad_input, grad_output)
            
            handle = self.model.register_backward_hook(model_backward_hook)
            self.hooks.append(handle)
    
    def _register_layer_level_hooks(self) -> None:
        """Register hooks at each layer level"""
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                
                def make_forward_hook(layer_name):
                    def forward_hook(module, input, output):
                        self._capture_io(layer_name, "forward", input, output, module)
                    return forward_hook
                
                handle = module.register_forward_hook(make_forward_hook(name))
                self.hooks.append(handle)
                
                if self.capture_gradients:
                    def make_backward_hook(layer_name):
                        def backward_hook(module, grad_input, grad_output):
                            self._capture_gradients(layer_name, grad_input, grad_output)
                        return backward_hook
                    
                    handle = module.register_backward_hook(make_backward_hook(name))
                    self.hooks.append(handle)
    
    def _register_attention_level_hooks(self) -> None:
        """Register hooks specifically for attention mechanisms"""
        attention_modules = []
        
        # Find attention modules
        for name, module in self.model.named_modules():
            if ("attention" in name.lower() or 
                "attn" in module.__class__.__name__.lower() or
                "MultiHeadAttention" in module.__class__.__name__):
                attention_modules.append((name, module))
        
        if not attention_modules:
            self.logger.warning("No attention modules found, falling back to layer-level hooks")
            self._register_layer_level_hooks()
            return
        
        for name, module in attention_modules:
            def make_attention_hook(layer_name):
                def attention_forward_hook(module, input, output):
                    self._capture_attention(layer_name, input, output, module)
                return attention_forward_hook
            
            handle = module.register_forward_hook(make_attention_hook(name))
            self.hooks.append(handle)
        
        # Also register model-level hook for overall I/O
        self._register_model_level_hooks()
    
    def _register_comprehensive_hooks(self) -> None:
        """Register hooks at all levels"""
        self._register_model_level_hooks()
        self._register_layer_level_hooks()
        if self.capture_attention:
            self._register_attention_level_hooks()
    
    def _capture_io(self, layer_name: str, operation: str, inputs, outputs, module) -> None:
        """Capture input/output data"""
        if not self.enabled:
            return
        
        capture_key = f"{layer_name}_{operation}_{int(time.time()*1000000)}"
        
        # Trace execution if enabled
        if self.tracer:
            with self.tracer.trace(f"{layer_name}_{operation}", "hf_model"):
                pass
        
        # Process inputs
        processed_inputs = self._process_tensor_data(inputs, f"{capture_key}_input")
        
        # Process outputs
        processed_outputs = self._process_tensor_data(outputs, f"{capture_key}_output")
        
        # Store capture data
        self.captured_data["inputs"][capture_key] = processed_inputs
        self.captured_data["outputs"][capture_key] = processed_outputs
        
        # Add metadata
        self.captured_data["metadata"][capture_key] = {
            "layer_name": layer_name,
            "operation": operation,
            "module_type": module.__class__.__name__,
            "timestamp": time.time(),
            "parameters": sum(p.numel() for p in module.parameters()),
        }
        
        # Log with IOLogger
        if hasattr(inputs, '__iter__') and not isinstance(inputs, torch.Tensor):
            input_dict = {f"input_{i}": inp for i, inp in enumerate(inputs)}
        else:
            input_dict = {"input": inputs}
            
        if hasattr(outputs, '__iter__') and not isinstance(outputs, torch.Tensor):
            output_dict = {f"output_{i}": out for i, out in enumerate(outputs)}
        else:
            output_dict = {"output": outputs}
        
        # Use wrapped function for IOLogger
        wrapped_func = self.io_logger.wrap_function(lambda: None, layer_name)
        # This is a simplified integration - in practice we'd need more sophisticated integration
        
        # Multi-dtype comparison if enabled
        if self.dtype_comparer and isinstance(outputs, torch.Tensor):
            try:
                dtype_analysis = self.dtype_comparer.compare_across_dtypes(
                    outputs, f"{layer_name}_output"
                )
                self.captured_data["intermediate"][f"{capture_key}_dtype"] = dtype_analysis
            except Exception as e:
                if self.verbose:
                    self.logger.warning(f"Dtype comparison failed for {layer_name}: {e}")
    
    def _capture_attention(self, layer_name: str, inputs, outputs, module) -> None:
        """Capture attention-specific data"""
        if not self.enabled:
            return
        
        capture_key = f"{layer_name}_attention_{int(time.time()*1000000)}"
        
        # Extract attention weights if available
        attention_data = {}
        
        if hasattr(outputs, 'attentions') and outputs.attentions is not None:
            attention_weights = outputs.attentions
            attention_data["attention_weights"] = self._process_tensor_data(
                attention_weights, f"{capture_key}_weights"
            )
        
        # Try to extract attention from output tuple/dict
        if isinstance(outputs, tuple) and len(outputs) > 1:
            # Convention: (hidden_states, attention_weights, ...)
            if len(outputs) >= 2:
                potential_attention = outputs[1]
                if isinstance(potential_attention, torch.Tensor):
                    attention_data["attention_weights"] = self._process_tensor_data(
                        potential_attention, f"{capture_key}_weights"
                    )
        
        elif isinstance(outputs, dict) and "attentions" in outputs:
            attention_data["attention_weights"] = self._process_tensor_data(
                outputs["attentions"], f"{capture_key}_weights"
            )
        
        # Store attention data
        if attention_data:
            self.captured_data["attention"][capture_key] = attention_data
            self.captured_data["attention"][capture_key]["metadata"] = {
                "layer_name": layer_name,
                "timestamp": time.time(),
                "module_type": module.__class__.__name__,
            }
    
    def _capture_gradients(self, layer_name: str, grad_inputs, grad_outputs) -> None:
        """Capture gradient information"""
        if not self.enabled:
            return
        
        capture_key = f"{layer_name}_gradients_{int(time.time()*1000000)}"
        
        gradient_data = {
            "grad_inputs": self._process_tensor_data(grad_inputs, f"{capture_key}_grad_in"),
            "grad_outputs": self._process_tensor_data(grad_outputs, f"{capture_key}_grad_out"),
            "metadata": {
                "layer_name": layer_name,
                "timestamp": time.time(),
            }
        }
        
        self.captured_data["gradients"][capture_key] = gradient_data
    
    def _process_tensor_data(self, data: Any, storage_key: str) -> Any:
        """Process tensor data for storage and analysis"""
        if data is None:
            return None
        
        # Handle different data types
        if isinstance(data, torch.Tensor):
            return self._process_single_tensor(data, storage_key)
        elif isinstance(data, (list, tuple)):
            return [self._process_tensor_data(item, f"{storage_key}_{i}") for i, item in enumerate(data)]
        elif isinstance(data, dict):
            return {k: self._process_tensor_data(v, f"{storage_key}_{k}") for k, v in data.items()}
        else:
            return data
    
    def _process_single_tensor(self, tensor: torch.Tensor, storage_key: str) -> Dict[str, Any]:
        """Process a single tensor"""
        tensor_info = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "requires_grad": tensor.requires_grad,
            "size_mb": tensor.numel() * tensor.element_size() / (1024 * 1024),
        }
        
        # Decide storage strategy
        should_store = (
            self.storage_mode == "disk" or
            (self.storage_mode == "auto" and tensor_info["size_mb"] > self.max_memory_mb)
        )
        
        if should_store and self.storage:
            # Store to disk
            storage_id = self.storage.store_tensor(tensor, storage_key)
            tensor_info["storage_id"] = storage_id
            tensor_info["stored_on_disk"] = True
        else:
            # Keep in memory (store detached copy)
            tensor_info["data"] = tensor.detach().cpu()
            tensor_info["stored_on_disk"] = False
        
        return tensor_info
    
    def _remove_hooks(self) -> None:
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def get_analysis(self) -> Dict[str, Any]:
        """Get comprehensive analysis of captured data"""
        analysis = {
            "model_info": self.model_info,
            "capture_summary": self._generate_capture_summary(),
            "layer_analysis": self._analyze_layers(),
            "attention_analysis": self._analyze_attention_patterns(),
            "gradient_analysis": self._analyze_gradients(),
            "performance_analysis": self._analyze_performance(),
            "recommendations": self._generate_recommendations(),
        }
        
        return analysis
    
    def _generate_capture_summary(self) -> Dict[str, Any]:
        """Generate summary of captured data"""
        return {
            "total_captures": len(self.captured_data["inputs"]),
            "attention_captures": len(self.captured_data["attention"]), 
            "gradient_captures": len(self.captured_data["gradients"]),
            "storage_stats": self.storage.get_storage_stats() if self.storage else {},
            "capture_level": self.capture_level,
            "storage_mode": self.storage_mode,
        }
    
    def _analyze_layers(self) -> Dict[str, Any]:
        """Analyze layer-level patterns"""
        layer_analysis = {
            "activation_statistics": {},
            "layer_timing": {},
            "memory_usage": {},
        }
        
        # Analyze each captured layer
        for capture_key, capture_data in self.captured_data["inputs"].items():
            try:
                # Check if metadata exists for this capture_key
                if capture_key not in self.captured_data["metadata"]:
                    if self.verbose:
                        self.logger.warning(f"Missing metadata for capture key: {capture_key}")
                    continue
                
                layer_name = self.captured_data["metadata"][capture_key]["layer_name"]
                
                # Get corresponding output
                output_data = self.captured_data["outputs"].get(capture_key)
                
                if output_data and isinstance(output_data, dict) and not output_data.get("stored_on_disk", True):
                    output_tensor = output_data.get("data")
                    
                    # Calculate activation statistics
                    if isinstance(output_tensor, torch.Tensor):
                        try:
                            stats = {
                                "mean": float(output_tensor.mean()),
                                "std": float(output_tensor.std()),
                                "min": float(output_tensor.min()),
                                "max": float(output_tensor.max()),
                                "zero_fraction": float((output_tensor == 0).float().mean()),
                                "activation_pattern": "sparse" if (output_tensor == 0).float().mean() > 0.5 else "dense",
                            }
                            
                            layer_analysis["activation_statistics"][layer_name] = stats
                        except Exception as e:
                            if self.verbose:
                                self.logger.warning(f"Failed to calculate statistics for {layer_name}: {e}")
                    
            except Exception as e:
                if self.verbose:
                    self.logger.warning(f"Error analyzing layer data for {capture_key}: {e}")
                continue
        
        return layer_analysis
    
    def _analyze_attention_patterns(self) -> Dict[str, Any]:
        """Analyze attention patterns"""
        if not self.captured_data["attention"]:
            return {"message": "No attention data captured"}
        
        attention_analysis = {
            "attention_statistics": {},
            "head_analysis": {},
            "pattern_analysis": {},
        }
        
        for capture_key, attention_data in self.captured_data["attention"].items():
            if "attention_weights" in attention_data:
                weights_info = attention_data["attention_weights"]
                layer_name = attention_data["metadata"]["layer_name"]
                
                if not weights_info["stored_on_disk"]:
                    weights = weights_info["data"]
                    
                    if isinstance(weights, torch.Tensor) and weights.dim() >= 3:
                        # Analyze attention patterns
                        # Shape is typically [batch, heads, seq_len, seq_len]
                        
                        stats = {
                            "entropy": float(self._calculate_attention_entropy(weights)),
                            "sparsity": float((weights < 0.01).float().mean()),
                            "max_attention": float(weights.max()),
                            "diagonal_dominance": float(self._calculate_diagonal_dominance(weights)),
                        }
                        
                        attention_analysis["attention_statistics"][layer_name] = stats
        
        return attention_analysis
    
    def _calculate_attention_entropy(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """Calculate entropy of attention distribution"""
        # Add small epsilon to avoid log(0)
        eps = 1e-8
        attention_weights = attention_weights + eps
        
        # Calculate entropy over the last dimension (attended positions)
        entropy = -(attention_weights * torch.log(attention_weights)).sum(-1)
        return entropy.mean()
    
    def _calculate_diagonal_dominance(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """Calculate how much attention focuses on diagonal (self-attention)"""
        if attention_weights.dim() < 3:
            return torch.tensor(0.0)
        
        # Get diagonal elements
        min_dim = min(attention_weights.shape[-2], attention_weights.shape[-1])
        diagonal = torch.diagonal(attention_weights, dim1=-2, dim2=-1)[..., :min_dim]
        
        return diagonal.mean()
    
    def _analyze_gradients(self) -> Dict[str, Any]:
        """Analyze gradient patterns"""
        if not self.captured_data["gradients"]:
            return {"message": "No gradient data captured"}
        
        gradient_analysis = {
            "gradient_norms": {},
            "vanishing_gradients": [],
            "exploding_gradients": [],
        }
        
        for capture_key, grad_data in self.captured_data["gradients"].items():
            layer_name = grad_data["metadata"]["layer_name"]
            
            # Analyze gradient outputs (more relevant)
            if "grad_outputs" in grad_data:
                grad_info = grad_data["grad_outputs"]
                
                if not grad_info["stored_on_disk"]:
                    grads = grad_info["data"]
                    
                    if isinstance(grads, (list, tuple)):
                        for i, grad in enumerate(grads):
                            if isinstance(grad, torch.Tensor):
                                norm = float(grad.norm())
                                gradient_analysis["gradient_norms"][f"{layer_name}_{i}"] = norm
                                
                                if norm < 1e-6:
                                    gradient_analysis["vanishing_gradients"].append(f"{layer_name}_{i}")
                                elif norm > 100:
                                    gradient_analysis["exploding_gradients"].append(f"{layer_name}_{i}")
        
        return gradient_analysis
    
    def _analyze_performance(self) -> Dict[str, Any]:
        """Analyze performance metrics"""
        performance_analysis = {
            "timing_analysis": {},
            "memory_analysis": {},
            "efficiency_metrics": {},
        }
        
        # Get timing data from tracer if available
        if self.tracer:
            tracer_stats = self.tracer.get_statistics()
            performance_analysis["timing_analysis"] = tracer_stats
        
        # Memory analysis
        if self.storage:
            storage_stats = self.storage.get_storage_stats()
            performance_analysis["memory_analysis"] = storage_stats
        
        return performance_analysis
    
    def _generate_recommendations(self) -> List[str]:
        """Generate debugging and optimization recommendations"""
        recommendations = []
        
        # Analyze captured data for recommendations
        layer_analysis = self._analyze_layers()
        attention_analysis = self._analyze_attention_patterns()
        gradient_analysis = self._analyze_gradients()
        
        # Check for common issues
        if gradient_analysis.get("vanishing_gradients"):
            recommendations.append(
                f"Vanishing gradients detected in {len(gradient_analysis['vanishing_gradients'])} layers. "
                "Consider using residual connections, different initialization, or gradient clipping."
            )
        
        if gradient_analysis.get("exploding_gradients"):
            recommendations.append(
                f"Exploding gradients detected in {len(gradient_analysis['exploding_gradients'])} layers. "
                "Consider gradient clipping or reducing learning rate."
            )
        
        # Check attention patterns
        if attention_analysis.get("attention_statistics"):
            high_sparsity_layers = [
                layer for layer, stats in attention_analysis["attention_statistics"].items()
                if stats.get("sparsity", 0) > 0.8
            ]
            
            if high_sparsity_layers:
                recommendations.append(
                    f"High attention sparsity in layers: {', '.join(high_sparsity_layers)}. "
                    "Consider sparse attention mechanisms for efficiency."
                )
        
        # Storage recommendations
        if self.storage:
            storage_stats = self.storage.get_storage_stats()
            if storage_stats.get("disk_usage_mb", 0) > 1000:  # >1GB
                recommendations.append(
                    "Large disk usage detected. Consider increasing memory limits or "
                    "using more selective capture levels."
                )
        
        return recommendations
    
    def _save_debug_session(self) -> None:
        """Save comprehensive debug session data"""
        # Save analysis results
        analysis = self.get_analysis()
        self.save_json(analysis, "debug_analysis.json")
        
        # Save captured data (metadata only, tensors handled by storage)
        captured_metadata = {
            "metadata": self.captured_data["metadata"],
            "summary": {
                "total_captures": len(self.captured_data["inputs"]),
                "attention_captures": len(self.captured_data["attention"]),
                "gradient_captures": len(self.captured_data["gradients"]),
            }
        }
        
        self.save_json(captured_metadata, "captured_data_summary.json")
        
        if self.verbose:
            self.logger.info(f"Debug session saved to {self.output_dir}")
    
    def __enter__(self):
        """Context manager entry"""
        self.enable()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disable()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get debugger statistics"""
        return {
            "model_class": self.model.__class__.__name__,
            "capture_level": self.capture_level,
            "total_hooks": len(self.hooks),
            "total_captures": len(self.captured_data["inputs"]),
            "enabled": self.enabled,
        }


def auto_debug_model(
    model: nn.Module,
    level: str = "layer",
    storage: str = "auto",
    output_dir: Optional[Union[str, Path]] = None,
    **kwargs
) -> HuggingFaceDebugger:
    """
    One-line model debugging setup
    
    Args:
        model: HuggingFace model to debug
        level: Capture level ("model", "layer", "attention", "all")  
        storage: Storage mode ("memory", "disk", "auto")
        output_dir: Output directory for debug data
        **kwargs: Additional arguments for HuggingFaceDebugger
    
    Returns:
        Configured and enabled HuggingFaceDebugger
    
    Example:
        >>> debugger = auto_debug_model(model, level="attention", storage="disk")
        >>> outputs = model(**inputs)
        >>> analysis = debugger.get_analysis()
        >>> debugger.disable()
    """
    debugger = HuggingFaceDebugger(
        model,
        output_dir=output_dir,
        capture_level=level,
        storage_mode=storage,
        **kwargs
    )
    
    debugger.enable()
    return debugger