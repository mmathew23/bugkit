"""
PEFT (Parameter Efficient Fine-Tuning) debugging utilities
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

from ...core.base import BaseDebugTool


class PEFTDebugger(BaseDebugTool):
    """Debugging utilities for PEFT models (LoRA, AdaLoRA, etc.)"""
    
    def __init__(
        self,
        model: nn.Module,
        output_dir: Optional[Union[str, Path]] = None,
        verbose: bool = True,
        track_adapter_gradients: bool = True,
        compare_base_vs_adapter: bool = True,
    ):
        super().__init__(output_dir, verbose)
        self.model = model
        self.track_adapter_gradients = track_adapter_gradients
        self.compare_base_vs_adapter = compare_base_vs_adapter
        
        self.peft_analysis: List[Dict[str, Any]] = []
        self.adapter_comparisons: List[Dict[str, Any]] = []
        self.gradient_analyses: List[Dict[str, Any]] = []
        
        # Check if PEFT is available and detect PEFT model
        self.peft_info = self._analyze_peft_model()
    
    def enable(self) -> None:
        """Enable PEFT debugging"""
        self.enabled = True
        
        if self.verbose:
            peft_type = self.peft_info.get("peft_type", "unknown")
            adapter_count = self.peft_info.get("adapter_count", 0)
            self.logger.info(f"PEFT debugger enabled for {peft_type} model with {adapter_count} adapters")
    
    def disable(self) -> None:
        """Disable PEFT debugging and save results"""
        self.enabled = False
        self._save_debug_data()
        
        if self.verbose:
            self.logger.info("PEFT debugger disabled")
    
    def _analyze_peft_model(self) -> Dict[str, Any]:
        """Analyze PEFT model structure and configuration"""
        peft_info = {
            "is_peft_model": False,
            "peft_type": None,
            "peft_config": {},
            "adapter_count": 0,
            "base_model_parameters": 0,
            "adapter_parameters": 0,
            "trainable_parameters": 0,
            "adapter_modules": [],
        }
        
        try:
            # Check if model has PEFT attributes
            if hasattr(self.model, 'peft_config'):
                peft_info["is_peft_model"] = True
                peft_info["peft_config"] = getattr(self.model, 'peft_config', {})
                
                # Get PEFT type from config
                if isinstance(peft_info["peft_config"], dict):
                    for adapter_name, config in peft_info["peft_config"].items():
                        if hasattr(config, 'peft_type'):
                            peft_info["peft_type"] = str(config.peft_type)
                            break
                        elif hasattr(config, '__class__'):
                            peft_info["peft_type"] = config.__class__.__name__
                            break
            
            # Alternative check for PEFT models
            elif hasattr(self.model, 'base_model') and hasattr(self.model, 'get_peft_config'):
                peft_info["is_peft_model"] = True
                try:
                    config = self.model.get_peft_config()
                    peft_info["peft_config"] = config
                    if hasattr(config, 'peft_type'):
                        peft_info["peft_type"] = str(config.peft_type)
                except:
                    pass
            
            # Count parameters
            total_params = 0
            trainable_params = 0
            adapter_params = 0
            
            for name, param in self.model.named_parameters():
                total_params += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()
                    
                    # Detect adapter parameters (common patterns)
                    if any(pattern in name.lower() for pattern in ['lora', 'adapter', 'prompt']):
                        adapter_params += param.numel()
                        peft_info["adapter_modules"].append({
                            "name": name,
                            "shape": list(param.shape),
                            "parameters": param.numel(),
                            "dtype": str(param.dtype),
                        })
            
            peft_info.update({
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "adapter_parameters": adapter_params,
                "base_model_parameters": total_params - adapter_params,
                "adapter_count": len(peft_info["adapter_modules"]),
                "efficiency_ratio": adapter_params / total_params if total_params > 0 else 0,
            })
            
        except Exception as e:
            if self.verbose:
                self.logger.warning(f"Error analyzing PEFT model: {e}")
        
        return peft_info
    
    def analyze_adapter_efficiency(
        self,
        analysis_name: str = "adapter_efficiency",
    ) -> Dict[str, Any]:
        """Analyze parameter efficiency of adapters"""
        if not self.enabled:
            raise RuntimeError("PEFTDebugger is not enabled")
        
        efficiency_analysis = {
            "analysis_name": analysis_name,
            "timestamp": time.time(),
            "peft_info": self.peft_info,
            "parameter_breakdown": {},
            "efficiency_metrics": {},
            "adapter_analysis": {},
            "recommendations": [],
        }
        
        # Analyze parameter distribution
        module_params = {}
        adapter_params_by_layer = {}
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Extract layer/module info
                layer_parts = name.split('.')
                if len(layer_parts) > 1:
                    layer_name = '.'.join(layer_parts[:-1])
                    param_name = layer_parts[-1]
                    
                    if layer_name not in module_params:
                        module_params[layer_name] = {"total": 0, "adapter": 0, "params": []}
                    
                    module_params[layer_name]["total"] += param.numel()
                    module_params[layer_name]["params"].append({
                        "name": param_name,
                        "shape": list(param.shape),
                        "parameters": param.numel(),
                        "is_adapter": any(pattern in name.lower() for pattern in ['lora', 'adapter', 'prompt']),
                    })
                    
                    if any(pattern in name.lower() for pattern in ['lora', 'adapter', 'prompt']):
                        module_params[layer_name]["adapter"] += param.numel()
                        
                        if layer_name not in adapter_params_by_layer:
                            adapter_params_by_layer[layer_name] = 0
                        adapter_params_by_layer[layer_name] += param.numel()
        
        efficiency_analysis["parameter_breakdown"] = module_params
        
        # Calculate efficiency metrics
        total_params = self.peft_info["total_parameters"]
        adapter_params = self.peft_info["adapter_parameters"]
        
        efficiency_analysis["efficiency_metrics"] = {
            "parameter_efficiency": adapter_params / total_params if total_params > 0 else 0,
            "memory_reduction_ratio": (total_params - adapter_params) / total_params if total_params > 0 else 0,
            "adapter_to_base_ratio": adapter_params / (total_params - adapter_params) if (total_params - adapter_params) > 0 else 0,
            "trainable_ratio": self.peft_info["trainable_parameters"] / total_params if total_params > 0 else 0,
        }
        
        # Analyze adapter distribution
        if adapter_params_by_layer:
            layer_counts = list(adapter_params_by_layer.values())
            efficiency_analysis["adapter_analysis"] = {
                "layers_with_adapters": len(adapter_params_by_layer),
                "total_layers": len(module_params),
                "adapter_coverage": len(adapter_params_by_layer) / len(module_params) if module_params else 0,
                "max_adapter_params_per_layer": max(layer_counts),
                "min_adapter_params_per_layer": min(layer_counts),
                "mean_adapter_params_per_layer": sum(layer_counts) / len(layer_counts),
                "adapter_distribution": adapter_params_by_layer,
            }
        
        # Generate recommendations
        efficiency_ratio = efficiency_analysis["efficiency_metrics"]["parameter_efficiency"]
        
        if efficiency_ratio < 0.01:  # Less than 1%
            efficiency_analysis["recommendations"].append(f"Excellent parameter efficiency: {efficiency_ratio:.3%} of parameters are adapters")
        elif efficiency_ratio < 0.05:  # Less than 5%
            efficiency_analysis["recommendations"].append(f"Good parameter efficiency: {efficiency_ratio:.3%} of parameters are adapters")
        else:
            efficiency_analysis["recommendations"].append(f"High parameter usage: {efficiency_ratio:.3%} of parameters are adapters - consider more efficient adapter methods")
        
        # Check adapter coverage
        if "adapter_analysis" in efficiency_analysis:
            coverage = efficiency_analysis["adapter_analysis"]["adapter_coverage"]
            if coverage < 0.5:
                efficiency_analysis["recommendations"].append(f"Low adapter coverage: only {coverage:.1%} of layers have adapters")
            elif coverage > 0.9:
                efficiency_analysis["recommendations"].append(f"High adapter coverage: {coverage:.1%} of layers have adapters - consider selective adaptation")
        
        self.peft_analysis.append(efficiency_analysis)
        
        if self.verbose:
            self.logger.info(f"Adapter efficiency analysis '{analysis_name}': {efficiency_ratio:.3%} parameter efficiency")
        
        return efficiency_analysis
    
    def compare_adapter_gradients(
        self,
        model_outputs: torch.Tensor,
        comparison_name: str = "adapter_gradients",
    ) -> Dict[str, Any]:
        """Compare gradients between base model and adapters"""
        if not self.enabled:
            raise RuntimeError("PEFTDebugger is not enabled")
        
        if not self.peft_info["is_peft_model"]:
            return {"error": "Model is not a PEFT model"}
        
        gradient_comparison = {
            "comparison_name": comparison_name,
            "timestamp": time.time(),
            "gradient_statistics": {},
            "adapter_vs_base": {},
            "gradient_flow": {},
            "potential_issues": [],
        }
        
        # Perform backward pass to compute gradients
        if model_outputs.requires_grad:
            loss = model_outputs.sum()  # Simple loss for gradient computation
            loss.backward(retain_graph=True)
        
        base_gradients = []
        adapter_gradients = []
        gradient_stats = {}
        
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_norm = float(param.grad.norm())
                is_adapter = any(pattern in name.lower() for pattern in ['lora', 'adapter', 'prompt'])
                
                gradient_stats[name] = {
                    "gradient_norm": grad_norm,
                    "gradient_mean": float(param.grad.mean()),
                    "gradient_std": float(param.grad.std()),
                    "gradient_min": float(param.grad.min()),
                    "gradient_max": float(param.grad.max()),
                    "is_adapter": is_adapter,
                    "parameter_shape": list(param.shape),
                }
                
                if is_adapter:
                    adapter_gradients.append(grad_norm)
                else:
                    base_gradients.append(grad_norm)
                
                # Check for gradient issues
                if grad_norm < 1e-8:
                    gradient_comparison["potential_issues"].append(f"Very small gradient in {name}: {grad_norm}")
                elif grad_norm > 100:
                    gradient_comparison["potential_issues"].append(f"Large gradient in {name}: {grad_norm}")
        
        gradient_comparison["gradient_statistics"] = gradient_stats
        
        # Compare adapter vs base gradients
        if base_gradients and adapter_gradients:
            base_mean = sum(base_gradients) / len(base_gradients)
            adapter_mean = sum(adapter_gradients) / len(adapter_gradients)
            
            gradient_comparison["adapter_vs_base"] = {
                "base_gradient_count": len(base_gradients),
                "adapter_gradient_count": len(adapter_gradients),
                "base_mean_gradient": base_mean,
                "adapter_mean_gradient": adapter_mean,
                "base_max_gradient": max(base_gradients),
                "adapter_max_gradient": max(adapter_gradients),
                "adapter_to_base_ratio": adapter_mean / base_mean if base_mean > 0 else float('inf'),
            }
            
            # Analyze gradient flow
            total_gradient_norm = sum(base_gradients + adapter_gradients)
            gradient_comparison["gradient_flow"] = {
                "total_gradient_norm": total_gradient_norm,
                "base_gradient_contribution": sum(base_gradients) / total_gradient_norm if total_gradient_norm > 0 else 0,
                "adapter_gradient_contribution": sum(adapter_gradients) / total_gradient_norm if total_gradient_norm > 0 else 0,
            }
        
        self.gradient_analyses.append(gradient_comparison)
        
        if self.verbose:
            if "adapter_vs_base" in gradient_comparison:
                ratio = gradient_comparison["adapter_vs_base"]["adapter_to_base_ratio"]
                self.logger.info(f"Gradient comparison '{comparison_name}': adapter/base ratio = {ratio:.3f}")
        
        return gradient_comparison
    
    def analyze_lora_ranks(self) -> Dict[str, Any]:
        """Analyze LoRA rank efficiency if model uses LoRA"""
        if not self.peft_info["is_peft_model"]:
            return {"error": "Model is not a PEFT model"}
        
        rank_analysis = {
            "timestamp": time.time(),
            "lora_modules": {},
            "rank_statistics": {},
            "efficiency_analysis": {},
            "recommendations": [],
        }
        
        lora_ranks = []
        
        # Find LoRA modules and analyze ranks
        for name, module in self.model.named_modules():
            # Check for LoRA-specific attributes
            if hasattr(module, 'r') or 'lora' in name.lower():
                module_info = {
                    "module_name": name,
                    "module_type": module.__class__.__name__,
                }
                
                if hasattr(module, 'r'):
                    rank = getattr(module, 'r')
                    module_info["rank"] = rank
                    lora_ranks.append(rank)
                
                if hasattr(module, 'scaling'):
                    module_info["scaling"] = getattr(module, 'scaling')
                
                if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                    lora_a = getattr(module, 'lora_A')
                    lora_b = getattr(module, 'lora_B')
                    
                    if hasattr(lora_a, 'weight') and hasattr(lora_b, 'weight'):
                        module_info["lora_A_shape"] = list(lora_a.weight.shape)
                        module_info["lora_B_shape"] = list(lora_b.weight.shape)
                        module_info["parameters"] = lora_a.weight.numel() + lora_b.weight.numel()
                
                rank_analysis["lora_modules"][name] = module_info
        
        # Analyze rank statistics
        if lora_ranks:
            rank_analysis["rank_statistics"] = {
                "rank_count": len(lora_ranks),
                "min_rank": min(lora_ranks),
                "max_rank": max(lora_ranks),
                "mean_rank": sum(lora_ranks) / len(lora_ranks),
                "unique_ranks": len(set(lora_ranks)),
                "rank_distribution": {rank: lora_ranks.count(rank) for rank in set(lora_ranks)},
            }
            
            # Efficiency analysis
            mean_rank = rank_analysis["rank_statistics"]["mean_rank"]
            max_rank = rank_analysis["rank_statistics"]["max_rank"]
            
            rank_analysis["efficiency_analysis"] = {
                "rank_efficiency": mean_rank / max_rank if max_rank > 0 else 0,
                "rank_uniformity": len(set(lora_ranks)) / len(lora_ranks) if lora_ranks else 0,
            }
            
            # Generate recommendations
            if mean_rank < 8:
                rank_analysis["recommendations"].append(f"Low LoRA ranks (mean: {mean_rank:.1f}) - good for efficiency")
            elif mean_rank > 64:
                rank_analysis["recommendations"].append(f"High LoRA ranks (mean: {mean_rank:.1f}) - consider reducing for efficiency")
            
            if len(set(lora_ranks)) == 1:
                rank_analysis["recommendations"].append("All LoRA modules use the same rank - consider rank optimization")
        
        return rank_analysis
    
    def get_peft_summary(self) -> Dict[str, Any]:
        """Get comprehensive PEFT model summary"""
        return {
            "peft_info": self.peft_info,
            "analyses_performed": len(self.peft_analysis),
            "gradient_comparisons": len(self.gradient_analyses),
            "adapter_comparisons": len(self.adapter_comparisons),
            "enabled": self.enabled,
        }
    
    def _save_debug_data(self) -> None:
        """Save PEFT debug data"""
        debug_data = {
            "peft_info": self.peft_info,
            "peft_analysis": self.peft_analysis,
            "gradient_analyses": self.gradient_analyses,
            "adapter_comparisons": self.adapter_comparisons,
            "timestamp": time.time(),
        }
        
        self.save_json(debug_data, "peft_debug_data.json")
        
        # Generate summary
        summary = self.get_peft_summary()
        self.save_json(summary, "peft_debug_summary.json")
        
        if self.verbose:
            self.logger.info(f"PEFT debug data saved to {self.output_dir}")


def auto_peft_debug(
    model: nn.Module,
    output_dir: Optional[Union[str, Path]] = None,
    **kwargs
) -> PEFTDebugger:
    """
    Quick PEFT model debugging setup
    
    Args:
        model: PEFT model to debug
        output_dir: Output directory
        **kwargs: Additional arguments for PEFTDebugger
    
    Returns:
        Enabled PEFTDebugger instance
    
    Example:
        >>> peft_debugger = auto_peft_debug(peft_model)
        >>> efficiency = peft_debugger.analyze_adapter_efficiency()
        >>> lora_analysis = peft_debugger.analyze_lora_ranks()
        >>> peft_debugger.disable()
    """
    debugger = PEFTDebugger(model, output_dir=output_dir, **kwargs)
    debugger.enable()
    return debugger