"""
BitsAndBytes quantization debugging utilities
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

from ...core.base import BaseDebugTool


class BitsAndBytesDebugger(BaseDebugTool):
    """Debugging utilities for BitsAndBytes quantization"""
    
    def __init__(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        verbose: bool = True,
        track_quantization_errors: bool = True,
        compare_precisions: bool = True,
    ):
        super().__init__(output_dir, verbose)
        self.track_quantization_errors = track_quantization_errors
        self.compare_precisions = compare_precisions
        
        self.quantization_analyses: List[Dict[str, Any]] = []
        self.precision_comparisons: List[Dict[str, Any]] = []
        self.quantization_configs: List[Dict[str, Any]] = []
        
        # Check if BitsAndBytes is available
        try:
            import bitsandbytes as bnb
            self.bnb_available = True
            self.bnb = bnb
        except ImportError:
            self.bnb_available = False
            if verbose:
                self.logger.warning("BitsAndBytes not available - BnBDebugger will have limited functionality")
    
    def enable(self) -> None:
        """Enable BitsAndBytes debugging"""
        self.enabled = True
        
        if self.verbose:
            status = "enabled" if self.bnb_available else "enabled (limited - no BitsAndBytes)"
            self.logger.info(f"BitsAndBytes debugger {status}")
    
    def disable(self) -> None:
        """Disable BitsAndBytes debugging and save results"""
        self.enabled = False
        self._save_debug_data()
        
        if self.verbose:
            self.logger.info("BitsAndBytes debugger disabled")
    
    def analyze_quantized_model(
        self,
        model: nn.Module,
        model_name: str = "quantized_model",
        analyze_weights: bool = True,
        analyze_gradients: bool = True,
    ) -> Dict[str, Any]:
        """
        Analyze a quantized model for potential issues
        
        Args:
            model: Quantized PyTorch model
            model_name: Name for the model
            analyze_weights: Whether to analyze weight distributions
            analyze_gradients: Whether to analyze gradient behavior
        
        Returns:
            Quantization analysis results
        """
        if not self.bnb_available:
            return {"error": "BitsAndBytes not available"}
        
        analysis = {
            "model_name": model_name,
            "timestamp": time.time(),
            "quantized_parameters": {},
            "quantization_schemes": {},
            "memory_analysis": {},
            "precision_analysis": {},
            "potential_issues": [],
            "recommendations": [],
        }
        
        # Analyze quantized parameters
        total_params = 0
        quantized_params = 0
        quantization_types = {}
        
        for name, param in model.named_parameters():
            total_params += 1
            
            # Check if parameter is quantized (BitsAndBytes specific)
            param_info = {
                "shape": list(param.shape),
                "dtype": str(param.dtype),
                "requires_grad": param.requires_grad,
                "is_quantized": False,
                "quantization_type": None,
            }
            
            # Check for BitsAndBytes quantized parameters
            if hasattr(param, 'CB') or hasattr(param, 'SCB'):
                param_info["is_quantized"] = True
                quantized_params += 1
                
                if hasattr(param, 'CB'):
                    param_info["quantization_type"] = "int8"
                    quantization_types["int8"] = quantization_types.get("int8", 0) + 1
                elif hasattr(param, 'SCB'):
                    param_info["quantization_type"] = "4bit"
                    quantization_types["4bit"] = quantization_types.get("4bit", 0) + 1
            
            # Weight distribution analysis
            if analyze_weights and param.requires_grad:
                with torch.no_grad():
                    param_info["weight_stats"] = {
                        "mean": float(param.mean()),
                        "std": float(param.std()),
                        "min": float(param.min()),
                        "max": float(param.max()),
                        "zero_fraction": float((param.abs() < 1e-6).float().mean()),
                    }
                    
                    # Check for potential quantization issues
                    if param_info["weight_stats"]["std"] < 1e-4:
                        analysis["potential_issues"].append(f"Very small weight variance in {name}")
                    
                    if param_info["weight_stats"]["zero_fraction"] > 0.5:
                        analysis["potential_issues"].append(f"High sparsity in {name} ({param_info['weight_stats']['zero_fraction']:.2%})")
            
            analysis["quantized_parameters"][name] = param_info
        
        # Overall quantization statistics
        analysis["quantization_schemes"] = {
            "total_parameters": total_params,
            "quantized_parameters": quantized_params,
            "quantization_ratio": quantized_params / total_params if total_params > 0 else 0,
            "quantization_types": quantization_types,
        }
        
        # Memory analysis
        try:
            model_memory = sum(p.numel() * p.element_size() for p in model.parameters())
            analysis["memory_analysis"] = {
                "model_memory_bytes": model_memory,
                "model_memory_mb": model_memory / 1e6,
                "estimated_memory_savings": self._estimate_memory_savings(quantization_types, total_params),
            }
        except Exception as e:
            analysis["memory_analysis"] = {"error": str(e)}
        
        # Generate recommendations
        if quantized_params == 0:
            analysis["recommendations"].append("No quantized parameters found - consider applying quantization")
        elif quantized_params / total_params < 0.5:
            analysis["recommendations"].append("Low quantization ratio - consider quantizing more layers")
        
        if len(analysis["potential_issues"]) == 0:
            analysis["recommendations"].append("No obvious quantization issues detected")
        
        self.quantization_analyses.append(analysis)
        
        if self.verbose:
            ratio = analysis["quantization_schemes"]["quantization_ratio"]
            self.logger.info(f"Quantization analysis for '{model_name}': {ratio:.1%} parameters quantized")
        
        return analysis
    
    def compare_quantization_schemes(
        self,
        base_model: nn.Module,
        quantized_models: Dict[str, nn.Module],
        test_inputs: torch.Tensor,
        comparison_name: str = "quantization_comparison",
    ) -> Dict[str, Any]:
        """
        Compare different quantization schemes
        
        Args:
            base_model: Original unquantized model
            quantized_models: Dictionary of quantized models {name: model}
            test_inputs: Test inputs for comparison
            comparison_name: Name for the comparison
        
        Returns:
            Quantization comparison results
        """
        comparison = {
            "comparison_name": comparison_name,
            "timestamp": time.time(),
            "base_model_info": {},
            "quantized_model_results": {},
            "accuracy_comparison": {},
            "performance_comparison": {},
            "memory_comparison": {},
            "recommendations": [],
        }
        
        # Analyze base model
        with torch.no_grad():
            base_outputs = base_model(test_inputs)
            base_memory = sum(p.numel() * p.element_size() for p in base_model.parameters())
            
            comparison["base_model_info"] = {
                "parameters": sum(p.numel() for p in base_model.parameters()),
                "memory_bytes": base_memory,
                "memory_mb": base_memory / 1e6,
                "output_shape": list(base_outputs.shape),
            }
        
        # Analyze quantized models
        best_accuracy = {"model": None, "mse": float('inf')}
        best_compression = {"model": None, "ratio": 0}
        
        for model_name, quant_model in quantized_models.items():
            try:
                with torch.no_grad():
                    quant_outputs = quant_model(test_inputs)
                    quant_memory = sum(p.numel() * p.element_size() for p in quant_model.parameters())
                    
                    # Accuracy comparison
                    mse = torch.nn.functional.mse_loss(quant_outputs, base_outputs).item()
                    mae = torch.nn.functional.l1_loss(quant_outputs, base_outputs).item()
                    max_error = (quant_outputs - base_outputs).abs().max().item()
                    
                    # Memory comparison
                    memory_ratio = base_memory / quant_memory
                    compression_ratio = (base_memory - quant_memory) / base_memory
                    
                    model_results = {
                        "parameters": sum(p.numel() for p in quant_model.parameters()),
                        "memory_bytes": quant_memory,
                        "memory_mb": quant_memory / 1e6,
                        "memory_reduction_ratio": memory_ratio,
                        "compression_ratio": compression_ratio,
                        "accuracy_metrics": {
                            "mse": mse,
                            "mae": mae,
                            "max_error": max_error,
                            "relative_mse": mse / torch.var(base_outputs).item() if torch.var(base_outputs) > 0 else 0,
                        },
                    }
                    
                    # Performance analysis (simple timing)
                    times = []
                    for _ in range(10):
                        start_time = time.perf_counter()
                        with torch.no_grad():
                            _ = quant_model(test_inputs)
                        end_time = time.perf_counter()
                        times.append((end_time - start_time) * 1000)  # ms
                    
                    model_results["performance"] = {
                        "mean_inference_time_ms": sum(times) / len(times),
                        "min_inference_time_ms": min(times),
                        "max_inference_time_ms": max(times),
                    }
                    
                    comparison["quantized_model_results"][model_name] = model_results
                    
                    # Track best models
                    if mse < best_accuracy["mse"]:
                        best_accuracy = {"model": model_name, "mse": mse}
                    
                    if compression_ratio > best_compression["ratio"]:
                        best_compression = {"model": model_name, "ratio": compression_ratio}
                        
            except Exception as e:
                comparison["quantized_model_results"][model_name] = {"error": str(e)}
        
        # Generate comparison summaries
        if comparison["quantized_model_results"]:
            comparison["accuracy_comparison"] = {
                "best_accuracy_model": best_accuracy["model"],
                "best_accuracy_mse": best_accuracy["mse"],
                "accuracy_ranking": sorted(
                    [(name, results["accuracy_metrics"]["mse"]) 
                     for name, results in comparison["quantized_model_results"].items() 
                     if "accuracy_metrics" in results],
                    key=lambda x: x[1]
                ),
            }
            
            comparison["memory_comparison"] = {
                "best_compression_model": best_compression["model"],
                "best_compression_ratio": best_compression["ratio"],
                "compression_ranking": sorted(
                    [(name, results["compression_ratio"]) 
                     for name, results in comparison["quantized_model_results"].items() 
                     if "compression_ratio" in results],
                    key=lambda x: x[1],
                    reverse=True
                ),
            }
            
            # Generate recommendations
            if best_accuracy["model"] == best_compression["model"]:
                comparison["recommendations"].append(f"'{best_accuracy['model']}' offers best balance of accuracy and compression")
            else:
                comparison["recommendations"].append(f"'{best_accuracy['model']}' for accuracy, '{best_compression['model']}' for compression")
        
        self.precision_comparisons.append(comparison)
        
        if self.verbose:
            self.logger.info(f"Quantization comparison '{comparison_name}' completed")
        
        return comparison
    
    def test_quantization_config(
        self,
        model: nn.Module,
        quantization_config: Dict[str, Any],
        test_data: torch.Tensor,
        config_name: str = "quantization_test",
    ) -> Dict[str, Any]:
        """
        Test a specific quantization configuration
        
        Args:
            model: Model to quantize
            quantization_config: Quantization configuration
            test_data: Test data for validation
            config_name: Name for the configuration
        
        Returns:
            Quantization test results
        """
        if not self.bnb_available:
            return {"error": "BitsAndBytes not available"}
        
        test_result = {
            "config_name": config_name,
            "timestamp": time.time(),
            "quantization_config": quantization_config,
            "test_results": {},
            "performance_metrics": {},
            "quality_metrics": {},
            "recommendations": [],
        }
        
        try:
            # Get original outputs for comparison
            model.eval()
            with torch.no_grad():
                original_outputs = model(test_data)
            
            # Apply quantization (this is a simplified example)
            # In practice, you'd use specific BitsAndBytes APIs
            quantized_model = self._apply_quantization_config(model, quantization_config)
            
            if quantized_model is not None:
                with torch.no_grad():
                    quantized_outputs = quantized_model(test_data)
                
                # Quality metrics
                mse = torch.nn.functional.mse_loss(quantized_outputs, original_outputs).item()
                mae = torch.nn.functional.l1_loss(quantized_outputs, original_outputs).item()
                
                test_result["quality_metrics"] = {
                    "mse": mse,
                    "mae": mae,
                    "max_absolute_error": (quantized_outputs - original_outputs).abs().max().item(),
                    "signal_to_noise_ratio": 10 * torch.log10(torch.var(original_outputs) / mse).item() if mse > 0 else float('inf'),
                }
                
                # Memory metrics
                original_memory = sum(p.numel() * p.element_size() for p in model.parameters())
                quantized_memory = sum(p.numel() * p.element_size() for p in quantized_model.parameters())
                
                test_result["performance_metrics"] = {
                    "memory_reduction_bytes": original_memory - quantized_memory,
                    "memory_reduction_ratio": (original_memory - quantized_memory) / original_memory,
                    "compression_ratio": original_memory / quantized_memory,
                }
                
                # Generate recommendations
                if mse < 1e-3:
                    test_result["recommendations"].append("Good quantization quality - minimal accuracy loss")
                elif mse < 1e-2:
                    test_result["recommendations"].append("Acceptable quantization quality - minor accuracy loss")
                else:
                    test_result["recommendations"].append("High quantization error - consider different configuration")
                
                memory_savings = test_result["performance_metrics"]["memory_reduction_ratio"]
                if memory_savings > 0.5:
                    test_result["recommendations"].append(f"Excellent memory savings: {memory_savings:.1%}")
                elif memory_savings > 0.25:
                    test_result["recommendations"].append(f"Good memory savings: {memory_savings:.1%}")
                else:
                    test_result["recommendations"].append(f"Limited memory savings: {memory_savings:.1%}")
            
        except Exception as e:
            test_result["test_results"]["error"] = str(e)
        
        self.quantization_configs.append(test_result) 
        
        if self.verbose:
            self.logger.info(f"Quantization config test '{config_name}' completed")
        
        return test_result
    
    def _apply_quantization_config(self, model: nn.Module, config: Dict[str, Any]) -> Optional[nn.Module]:
        """Apply quantization configuration to model (simplified implementation)"""
        # This is a placeholder - in practice you'd use specific BitsAndBytes APIs
        # depending on the quantization scheme (8bit, 4bit, etc.)
        try:
            # Example: apply 8bit quantization to Linear layers
            if config.get("quantization_type") == "8bit":
                for name, module in model.named_modules():
                    if isinstance(module, nn.Linear) and self.bnb_available:
                        # Replace with BitsAndBytes quantized linear layer
                        quantized_linear = self.bnb.nn.Linear8bitLt(
                            module.in_features,
                            module.out_features,
                            bias=module.bias is not None,
                            has_fp16_weights=config.get("has_fp16_weights", True),
                            threshold=config.get("threshold", 6.0),
                        )
                        
                        # Copy weights
                        quantized_linear.weight.data = module.weight.data
                        if module.bias is not None:
                            quantized_linear.bias.data = module.bias.data
                        
                        # Replace module
                        parent = model
                        path = name.split('.')
                        for component in path[:-1]:
                            parent = getattr(parent, component)
                        setattr(parent, path[-1], quantized_linear)
            
            return model
            
        except Exception as e:
            if self.verbose:
                self.logger.error(f"Failed to apply quantization config: {e}")
            return None
    
    def _estimate_memory_savings(self, quantization_types: Dict[str, int], total_params: int) -> Dict[str, Any]:
        """Estimate memory savings from quantization"""
        savings = {
            "estimated_savings_ratio": 0,
            "breakdown": {},
        }
        
        # Rough estimates for memory savings
        type_savings = {
            "8bit": 0.75,  # 75% savings vs fp32
            "4bit": 0.875,  # 87.5% savings vs fp32
            "int8": 0.75,
            "int4": 0.875,
        }
        
        total_savings = 0
        for quant_type, count in quantization_types.items():
            if quant_type in type_savings:
                param_ratio = count / total_params if total_params > 0 else 0
                type_savings_contribution = param_ratio * type_savings[quant_type]
                total_savings += type_savings_contribution
                
                savings["breakdown"][quant_type] = {
                    "parameters": count,
                    "parameter_ratio": param_ratio,
                    "estimated_savings": type_savings[quant_type],
                    "contribution_to_total": type_savings_contribution,
                }
        
        savings["estimated_savings_ratio"] = total_savings
        return savings
    
    def _save_debug_data(self) -> None:
        """Save BitsAndBytes debug data"""
        debug_data = {
            "bitsandbytes_available": self.bnb_available,
            "quantization_analyses": self.quantization_analyses,
            "precision_comparisons": self.precision_comparisons,
            "quantization_configs": self.quantization_configs,
            "timestamp": time.time(),
        }
        
        if self.bnb_available:
            try:
                debug_data["bitsandbytes_version"] = self.bnb.__version__
            except:
                pass
        
        self.save_json(debug_data, "bitsandbytes_debug_data.json")
        
        # Generate summary
        summary = {
            "bitsandbytes_available": self.bnb_available,
            "models_analyzed": len(self.quantization_analyses),
            "comparisons_performed": len(self.precision_comparisons),
            "configs_tested": len(self.quantization_configs),
        }
        
        self.save_json(summary, "bitsandbytes_debug_summary.json")
        
        if self.verbose:
            self.logger.info(f"BitsAndBytes debug data saved to {self.output_dir}")


def auto_quantization_debug(
    model: nn.Module,
    model_name: str = "model",
    output_dir: Optional[Union[str, Path]] = None,
    **kwargs
) -> BitsAndBytesDebugger:
    """
    Quick quantization debugging setup
    
    Args:
        model: Model to analyze
        model_name: Name for the model
        output_dir: Output directory
        **kwargs: Additional arguments for BitsAndBytesDebugger
    
    Returns:
        Enabled BitsAndBytesDebugger instance with initial analysis
    
    Example:
        >>> bnb_debugger = auto_quantization_debug(quantized_model, "my_quantized_model")
        >>> # Debugger automatically analyzes the model
        >>> bnb_debugger.disable()
    """
    debugger = BitsAndBytesDebugger(output_dir=output_dir, **kwargs)
    debugger.enable()
    
    # Automatically analyze the provided model
    debugger.analyze_quantized_model(model, model_name)
    
    return debugger