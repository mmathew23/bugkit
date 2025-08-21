"""
Advanced storage and comparison utilities for large tensors and multi-dtype analysis
"""

import hashlib
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
import warnings

import numpy as np

from ..core.base import BaseDebugTool, format_tensor_info


class DiskTensorStorage(BaseDebugTool):
    """Efficient disk storage for large tensors with compression and metadata"""
    
    def __init__(
        self,
        storage_dir: Optional[Union[str, Path]] = None,
        compress: bool = True,
        max_memory_mb: int = 1024,  # Max memory before forcing disk storage
        use_memmap: bool = True,
        auto_cleanup: bool = True,
        verbose: bool = True,
    ):
        super().__init__(storage_dir or "tensor_storage", verbose=verbose)
        self.compress = compress
        self.max_memory_mb = max_memory_mb
        self.use_memmap = use_memmap
        self.auto_cleanup = auto_cleanup
        
        # Storage tracking
        self.stored_tensors: Dict[str, Dict[str, Any]] = {}
        self.memory_usage_mb = 0
        self.disk_usage_mb = 0
        
        # Create storage directories
        self.tensor_dir = self.output_dir / "tensors"
        self.metadata_dir = self.output_dir / "metadata"
        self.tensor_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
    
    def enable(self) -> None:
        """Enable tensor storage"""
        self.enabled = True
        if self.verbose:
            self.logger.info(f"Disk tensor storage enabled at {self.output_dir}")
    
    def disable(self) -> None:
        """Disable storage and optionally cleanup"""
        self.enabled = False
        if self.auto_cleanup:
            self._cleanup_storage()
        self._save_storage_manifest()
        if self.verbose:
            self.logger.info("Disk tensor storage disabled")
    
    def store_tensor(
        self,
        tensor: Any,
        key: str,
        metadata: Optional[Dict[str, Any]] = None,
        force_disk: bool = False,
    ) -> str:
        """Store tensor with optional metadata, return storage key"""
        if not self.enabled:
            raise RuntimeError("DiskTensorStorage is not enabled")
        
        # Convert to numpy for standardized storage
        np_tensor = self._to_numpy(tensor)
        tensor_info = format_tensor_info(tensor)
        
        # Calculate memory usage
        tensor_size_mb = np_tensor.nbytes / (1024 * 1024)
        
        # Decide storage method
        use_disk = force_disk or tensor_size_mb > self.max_memory_mb or self.memory_usage_mb + tensor_size_mb > self.max_memory_mb
        
        storage_key = self._generate_storage_key(key, tensor_info)
        
        storage_info = {
            "key": key,
            "storage_key": storage_key,
            "original_type": str(type(tensor)),
            "tensor_info": tensor_info,
            "size_mb": tensor_size_mb,
            "timestamp": time.time(),
            "metadata": metadata or {},
            "on_disk": use_disk,
        }
        
        if use_disk:
            # Store to disk
            tensor_path = self.tensor_dir / f"{storage_key}.npz"
            
            if self.compress:
                np.savez_compressed(tensor_path, tensor=np_tensor)
            else:
                np.savez(tensor_path, tensor=np_tensor)
            
            storage_info.update({
                "storage_path": str(tensor_path),
                "compressed": self.compress,
                "disk_size_mb": tensor_path.stat().st_size / (1024 * 1024),
            })
            
            self.disk_usage_mb += storage_info["disk_size_mb"]
            
            if self.verbose:
                compression_ratio = tensor_size_mb / storage_info["disk_size_mb"] if self.compress else 1.0
                self.logger.info(
                    f"Stored tensor '{key}' to disk ({tensor_size_mb:.2f}MB -> "
                    f"{storage_info['disk_size_mb']:.2f}MB, {compression_ratio:.1f}x compression)"
                )
        else:
            # Store in memory
            storage_info.update({
                "tensor_data": np_tensor,
                "in_memory": True,
            })
            
            self.memory_usage_mb += tensor_size_mb
            
            if self.verbose:
                self.logger.info(f"Stored tensor '{key}' in memory ({tensor_size_mb:.2f}MB)")
        
        # Save metadata to the metadata directory
        with open(self.metadata_dir / f"{storage_key}.json", 'w') as f:
            import json
            json.dump(storage_info, f, indent=2, default=str)
        
        self.stored_tensors[storage_key] = storage_info
        
        return storage_key
    
    def load_tensor(self, storage_key: str) -> Tuple[Any, Dict[str, Any]]:
        """Load tensor and metadata by storage key"""
        if storage_key not in self.stored_tensors:
            # Try to load from metadata file
            self._load_storage_manifest()
            
            if storage_key not in self.stored_tensors:
                raise KeyError(f"Storage key not found: {storage_key}")
        
        storage_info = self.stored_tensors[storage_key]
        
        if storage_info.get("on_disk", False):
            # Load from disk
            tensor_path = Path(storage_info["storage_path"])
            if not tensor_path.exists():
                raise FileNotFoundError(f"Tensor file not found: {tensor_path}")
            
            with np.load(tensor_path) as data:
                np_tensor = data["tensor"]
            
            if self.verbose:
                self.logger.info(f"Loaded tensor from disk: {storage_key}")
        else:
            # Load from memory
            np_tensor = storage_info["tensor_data"]
            
            if self.verbose:
                self.logger.info(f"Loaded tensor from memory: {storage_key}")
        
        # Convert back to original tensor type if possible
        original_type = storage_info.get("original_type", "")
        
        if "torch.Tensor" in original_type:
            try:
                import torch
                tensor = torch.from_numpy(np_tensor)
                return tensor, storage_info
            except Exception:
                pass
        
        return np_tensor, storage_info
    
    def store_tensor_dict(
        self,
        tensor_dict: Dict[str, Any],
        prefix: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """Store multiple tensors, return mapping of keys to storage keys"""
        storage_keys = {}
        
        for key, tensor in tensor_dict.items():
            full_key = f"{prefix}_{key}" if prefix else key
            storage_key = self.store_tensor(tensor, full_key, metadata)
            storage_keys[key] = storage_key
        
        return storage_keys
    
    def load_tensor_dict(self, storage_keys: Dict[str, str]) -> Dict[str, Any]:
        """Load multiple tensors from storage keys"""
        tensors = {}
        
        for key, storage_key in storage_keys.items():
            tensor, _ = self.load_tensor(storage_key)
            tensors[key] = tensor
        
        return tensors
    
    def compare_stored_tensors(
        self,
        storage_key1: str,
        storage_key2: str,
        comparer: Optional['MultiDtypeComparer'] = None,
        **compare_kwargs
    ) -> Dict[str, Any]:
        """Compare two stored tensors"""
        tensor1, info1 = self.load_tensor(storage_key1)
        tensor2, info2 = self.load_tensor(storage_key2)
        
        if comparer is None:
            from ..testing.tensor_compare import TensorComparer
            comparer = TensorComparer()
        
        return comparer.compare(
            tensor1, tensor2,
            name1=info1["key"], name2=info2["key"],
            **compare_kwargs
        )
    
    def _to_numpy(self, tensor: Any) -> np.ndarray:
        """Convert tensor to numpy array"""
        if isinstance(tensor, np.ndarray):
            return tensor
        
        # Handle PyTorch tensors
        try:
            import torch
            if isinstance(tensor, torch.Tensor):
                # Handle complex tensors that can't be easily converted
                if tensor.dtype in [torch.complex64, torch.complex128]:
                    return tensor.detach().cpu().numpy()
                # Handle quantized tensors
                elif tensor.is_quantized:
                    return tensor.dequantize().detach().cpu().numpy()
                else:
                    return tensor.detach().cpu().numpy()
        except ImportError:
            pass
        except Exception as e:
            # Fall through to other conversion methods
            pass
        
        # Handle TensorFlow tensors
        try:
            import tensorflow as tf
            if isinstance(tensor, (tf.Tensor, tf.Variable)):
                return tensor.numpy()
        except ImportError:
            pass
        
        # Handle JAX arrays
        try:
            import jax.numpy as jnp
            if isinstance(tensor, jnp.ndarray):
                return np.array(tensor)
        except ImportError:
            pass
        
        # Fallback
        try:
            return np.array(tensor)
        except Exception as e:
            raise ValueError(f"Cannot convert {type(tensor)} to numpy array: {e}")
    
    def _generate_storage_key(self, key: str, tensor_info: Dict[str, Any]) -> str:
        """Generate unique storage key"""
        # Create hash from key + tensor properties for uniqueness
        key_data = f"{key}_{tensor_info.get('shape', '')}_{tensor_info.get('dtype', '')}_{time.time()}"
        return hashlib.md5(key_data.encode()).hexdigest()[:16]
    
    def _save_storage_manifest(self) -> None:
        """Save storage manifest for persistence"""
        manifest = {
            "stored_tensors": self.stored_tensors,
            "memory_usage_mb": self.memory_usage_mb,
            "disk_usage_mb": self.disk_usage_mb,
            "timestamp": time.time(),
        }
        
        self.save_json(manifest, "storage_manifest.json")
    
    def _load_storage_manifest(self) -> None:
        """Load storage manifest"""
        try:
            manifest_path = self.output_dir / "storage_manifest.json"
            if manifest_path.exists():
                manifest = self.load_json("storage_manifest.json")
                self.stored_tensors = manifest.get("stored_tensors", {})
                self.memory_usage_mb = manifest.get("memory_usage_mb", 0)
                self.disk_usage_mb = manifest.get("disk_usage_mb", 0)
        except Exception as e:
            if self.verbose:
                self.logger.warning(f"Failed to load storage manifest: {e}")
    
    def _cleanup_storage(self) -> None:
        """Clean up storage files"""
        if self.tensor_dir.exists():
            for tensor_file in self.tensor_dir.glob("*.npz"):
                tensor_file.unlink()
        
        if self.metadata_dir.exists():
            for metadata_file in self.metadata_dir.glob("*.json"):
                metadata_file.unlink()
        
        if self.verbose:
            self.logger.info("Storage files cleaned up")
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get current storage statistics"""
        return {
            "total_tensors": len(self.stored_tensors),
            "memory_usage_mb": self.memory_usage_mb,
            "disk_usage_mb": self.disk_usage_mb,
            "tensors_in_memory": sum(1 for info in self.stored_tensors.values() if not info.get("on_disk", False)),
            "tensors_on_disk": sum(1 for info in self.stored_tensors.values() if info.get("on_disk", False)),
        }


class MultiDtypeComparer(BaseDebugTool):
    """Compare tensors across different dtypes and precisions"""
    
    def __init__(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        verbose: bool = True,
        reference_dtype: str = "float32",
        comparison_dtypes: Optional[List[str]] = None,
    ):
        super().__init__(output_dir, verbose)
        self.reference_dtype = reference_dtype
        self.comparison_dtypes = comparison_dtypes or ["float32", "float16", "bfloat16", "int8", "int16"]
        
        # Tolerance settings for different dtype comparisons
        self.dtype_tolerances = {
            ("float32", "float32"): {"rtol": 1e-7, "atol": 1e-9},
            ("float32", "float16"): {"rtol": 1e-3, "atol": 1e-5},
            ("float32", "bfloat16"): {"rtol": 1e-2, "atol": 1e-4},
            ("float16", "float16"): {"rtol": 1e-3, "atol": 1e-5},
            ("bfloat16", "bfloat16"): {"rtol": 1e-2, "atol": 1e-4},
            ("int8", "int8"): {"rtol": 0, "atol": 0},
            ("int16", "int16"): {"rtol": 0, "atol": 0},
        }
        
        self.comparisons: Dict[str, Dict[str, Any]] = {}
        
        # Try to get torch for dtype conversion
        self.torch_available = False
        try:
            import torch
            self.torch = torch
            self.torch_available = True
        except ImportError:
            pass
    
    def enable(self) -> None:
        """Enable multi-dtype comparer"""
        self.enabled = True
        if self.verbose:
            self.logger.info("Multi-dtype comparer enabled")
    
    def disable(self) -> None:
        """Disable comparer and save results"""
        self.enabled = False
        self._save_comparison_results()
        if self.verbose:
            self.logger.info("Multi-dtype comparer disabled")
    
    def compare_across_dtypes(
        self,
        tensor: Any,
        tensor_name: str = "tensor",
        dtypes: Optional[List[str]] = None,
        operations: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Compare tensor across different dtypes"""
        if not self.enabled:
            raise RuntimeError("MultiDtypeComparer is not enabled")
        
        dtypes = dtypes or self.comparison_dtypes
        operations = operations or ["identity", "sum", "mean", "max", "min"]
        
        # Convert to reference dtype first
        reference_tensor = self._convert_to_dtype(tensor, self.reference_dtype)
        
        comparison_result = {
            "tensor_name": tensor_name,
            "reference_dtype": self.reference_dtype,
            "compared_dtypes": dtypes,
            "operations_tested": operations,
            "original_info": format_tensor_info(tensor),
            "dtype_comparisons": {},
            "operation_comparisons": {},
            "summary": {},
            "timestamp": time.time(),
        }
        
        # Convert to each dtype and compare
        converted_tensors = {}
        for dtype in dtypes:
            try:
                converted_tensor = self._convert_to_dtype(tensor, dtype)
                converted_tensors[dtype] = converted_tensor
                
                # Compare against reference
                dtype_comparison = self._compare_dtype_pair(
                    reference_tensor, converted_tensor,
                    self.reference_dtype, dtype,
                    f"{tensor_name}_{self.reference_dtype}_vs_{dtype}"
                )
                
                comparison_result["dtype_comparisons"][dtype] = dtype_comparison
                
            except Exception as e:
                comparison_result["dtype_comparisons"][dtype] = {
                    "error": str(e),
                    "conversion_failed": True
                }
                if self.verbose:
                    self.logger.warning(f"Failed to convert to {dtype}: {e}")
        
        # Test operations across dtypes
        for operation in operations:
            operation_results = {}
            
            try:
                ref_result = self._apply_operation(reference_tensor, operation)
                operation_results[self.reference_dtype] = {
                    "result": ref_result,
                    "shape": getattr(ref_result, 'shape', None),
                    "dtype": str(getattr(ref_result, 'dtype', type(ref_result))),
                }
                
                for dtype in dtypes:
                    if dtype in converted_tensors:
                        try:
                            dtype_result = self._apply_operation(converted_tensors[dtype], operation)
                            operation_results[dtype] = {
                                "result": dtype_result,
                                "shape": getattr(dtype_result, 'shape', None),
                                "dtype": str(getattr(dtype_result, 'dtype', type(dtype_result))),
                            }
                            
                            # Compare operation results
                            if not np.isscalar(ref_result) and not np.isscalar(dtype_result):
                                op_comparison = self._compare_dtype_pair(
                                    ref_result, dtype_result,
                                    self.reference_dtype, dtype,
                                    f"{tensor_name}_{operation}_{self.reference_dtype}_vs_{dtype}"
                                )
                                operation_results[dtype]["comparison"] = op_comparison
                            
                        except Exception as e:
                            operation_results[dtype] = {"error": str(e)}
                
                comparison_result["operation_comparisons"][operation] = operation_results
                
            except Exception as e:
                comparison_result["operation_comparisons"][operation] = {
                    "error": f"Operation {operation} failed on reference: {e}"
                }
        
        # Generate summary
        comparison_result["summary"] = self._generate_dtype_summary(comparison_result)
        
        # Store comparison
        comparison_key = f"{tensor_name}_{int(time.time())}"
        self.comparisons[comparison_key] = comparison_result
        
        if self.verbose:
            self.logger.info(f"Multi-dtype comparison completed for {tensor_name}")
        
        return comparison_result
    
    def _convert_to_dtype(self, tensor: Any, target_dtype: str) -> Any:
        """Convert tensor to target dtype"""
        if not self.torch_available:
            # Fallback to numpy
            np_tensor = self._to_numpy(tensor)
            
            if target_dtype == "float32":
                return np_tensor.astype(np.float32)
            elif target_dtype == "float16":
                return np_tensor.astype(np.float16)
            elif target_dtype == "int8":
                return np_tensor.astype(np.int8)
            elif target_dtype == "int16":
                return np_tensor.astype(np.int16)
            elif target_dtype == "bfloat16":
                warnings.warn("bfloat16 not supported in numpy, using float32")
                return np_tensor.astype(np.float32)
            else:
                return np_tensor
        
        # Use PyTorch for more comprehensive dtype support
        if isinstance(tensor, self.torch.Tensor):
            torch_tensor = tensor
        else:
            # Convert to torch first
            np_tensor = self._to_numpy(tensor)
            torch_tensor = self.torch.from_numpy(np_tensor)
        
        # Convert to target dtype
        if target_dtype == "float32":
            return torch_tensor.float()
        elif target_dtype == "float16":
            return torch_tensor.half()
        elif target_dtype == "bfloat16":
            return torch_tensor.bfloat16()
        elif target_dtype == "int8":
            return torch_tensor.to(self.torch.int8)
        elif target_dtype == "int16":
            return torch_tensor.to(self.torch.int16)
        else:
            return torch_tensor
    
    def _to_numpy(self, tensor: Any) -> np.ndarray:
        """Convert tensor to numpy array with robust error handling"""
        if isinstance(tensor, np.ndarray):
            return tensor
        
        try:
            import torch
            if isinstance(tensor, torch.Tensor):
                # Handle edge cases
                if tensor.dtype in [torch.complex64, torch.complex128]:
                    return tensor.detach().cpu().numpy()
                elif tensor.is_quantized:
                    return tensor.dequantize().detach().cpu().numpy()
                else:
                    return tensor.detach().cpu().numpy()
        except ImportError:
            pass
        except Exception:
            pass
        
        try:
            import tensorflow as tf
            if isinstance(tensor, (tf.Tensor, tf.Variable)):
                return tensor.numpy()
        except ImportError:
            pass
        except Exception:
            pass
        
        try:
            return np.array(tensor)
        except Exception as e:
            raise ValueError(f"Cannot convert {type(tensor)} to numpy array: {e}")
    
    def _compare_dtype_pair(
        self, 
        tensor1: Any, 
        tensor2: Any, 
        dtype1: str, 
        dtype2: str, 
        comparison_name: str
    ) -> Dict[str, Any]:
        """Compare two tensors of different dtypes"""
        from ..testing.tensor_compare import TensorComparer
        
        # Get appropriate tolerances
        tolerance_key = (dtype1, dtype2)
        if tolerance_key not in self.dtype_tolerances:
            tolerance_key = (dtype2, dtype1)  # Try reverse
        
        tolerances = self.dtype_tolerances.get(tolerance_key, {"rtol": 1e-3, "atol": 1e-5})
        
        # Create comparer with appropriate tolerances
        comparer = TensorComparer(
            rtol=tolerances["rtol"],
            atol=tolerances["atol"],
            verbose=False
        )
        
        comparer.enable()
        
        try:
            # Convert both to numpy for comparison
            np_tensor1 = self._to_numpy(tensor1) 
            np_tensor2 = self._to_numpy(tensor2)
            
            comparison = comparer.compare(
                np_tensor1, np_tensor2,
                name1=f"{comparison_name}_{dtype1}",
                name2=f"{comparison_name}_{dtype2}",
                rtol=tolerances["rtol"],
                atol=tolerances["atol"]
            )
            
            # Add dtype-specific analysis
            comparison["dtype_analysis"] = {
                "dtype1": dtype1,
                "dtype2": dtype2,
                "tolerances_used": tolerances,
                "dtype_conversion_error": self._calculate_dtype_conversion_error(np_tensor1, np_tensor2),
            }
            
            return comparison
            
        except Exception as e:
            return {
                "error": str(e),
                "dtype1": dtype1,
                "dtype2": dtype2,
                "comparison_failed": True
            }
        finally:
            comparer.disable()
    
    def _calculate_dtype_conversion_error(self, tensor1: np.ndarray, tensor2: np.ndarray) -> Dict[str, float]:
        """Calculate specific error metrics for dtype conversion"""
        try:
            if tensor1.shape != tensor2.shape:
                return {"error": "Shape mismatch"}
            
            abs_error = np.abs(tensor1.astype(np.float64) - tensor2.astype(np.float64))
            
            return {
                "max_absolute_error": float(np.max(abs_error)),
                "mean_absolute_error": float(np.mean(abs_error)),
                "std_absolute_error": float(np.std(abs_error)),
                "median_absolute_error": float(np.median(abs_error)),
                "percentile_95_error": float(np.percentile(abs_error, 95)),
                "percentile_99_error": float(np.percentile(abs_error, 99)),
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _apply_operation(self, tensor: Any, operation: str) -> Any:
        """Apply operation to tensor"""
        if operation == "identity":
            return tensor
        elif operation == "sum":
            if hasattr(tensor, 'sum'):
                return tensor.sum()
            else:
                return np.sum(self._to_numpy(tensor))
        elif operation == "mean":
            if hasattr(tensor, 'mean'):
                return tensor.mean()
            else:
                return np.mean(self._to_numpy(tensor))
        elif operation == "max":
            if hasattr(tensor, 'max'):
                return tensor.max()
            else:
                return np.max(self._to_numpy(tensor))
        elif operation == "min":
            if hasattr(tensor, 'min'):
                return tensor.min()
            else:
                return np.min(self._to_numpy(tensor))
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    def _generate_dtype_summary(self, comparison_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of dtype comparison"""
        dtype_comparisons = comparison_result["dtype_comparisons"]
        
        summary = {
            "total_dtypes_tested": len(dtype_comparisons),
            "successful_conversions": sum(1 for comp in dtype_comparisons.values() if not comp.get("conversion_failed", False)),
            "failed_conversions": sum(1 for comp in dtype_comparisons.values() if comp.get("conversion_failed", False)),
            "accuracy_summary": {},
            "recommended_dtypes": [],
        }
        
        # Analyze accuracy
        accuracy_scores = {}
        for dtype, comp in dtype_comparisons.items():
            if not comp.get("conversion_failed", False) and "overall_match" in comp:
                match_percentage = comp.get("match_percentage", 0)
                accuracy_scores[dtype] = match_percentage
        
        if accuracy_scores:
            summary["accuracy_summary"] = {
                "best_dtype": max(accuracy_scores, key=accuracy_scores.get),
                "worst_dtype": min(accuracy_scores, key=accuracy_scores.get),
                "average_accuracy": np.mean(list(accuracy_scores.values())),
                "accuracy_scores": accuracy_scores,
            }
            
            # Recommend dtypes with >95% accuracy
            summary["recommended_dtypes"] = [
                dtype for dtype, score in accuracy_scores.items() if score > 95.0
            ]
        
        return summary
    
    def compare_quantization_schemes(
        self,
        tensor: Any,
        schemes: Optional[List[str]] = None,
        tensor_name: str = "tensor"
    ) -> Dict[str, Any]:
        """Compare different quantization schemes"""
        schemes = schemes or ["int8", "int16", "dynamic", "static"]
        
        quantization_result = {
            "tensor_name": tensor_name,
            "schemes_tested": schemes,
            "quantization_comparisons": {},
            "summary": {},
            "timestamp": time.time(),
        }
        
        reference_tensor = self._to_numpy(tensor).astype(np.float32)
        
        for scheme in schemes:
            try:
                if scheme == "int8":
                    quantized = self._quantize_int8(reference_tensor)
                elif scheme == "int16": 
                    quantized = self._quantize_int16(reference_tensor)
                elif scheme == "dynamic":
                    quantized = self._quantize_dynamic(reference_tensor)
                elif scheme == "static":
                    quantized = self._quantize_static(reference_tensor)
                else:
                    continue
                
                # Compare quantized vs original
                comparison = self._compare_dtype_pair(
                    reference_tensor, quantized,
                    "float32", scheme,
                    f"{tensor_name}_quantization_{scheme}"
                )
                
                quantization_result["quantization_comparisons"][scheme] = comparison
                
            except Exception as e:
                quantization_result["quantization_comparisons"][scheme] = {
                    "error": str(e),
                    "quantization_failed": True
                }
        
        quantization_result["summary"] = self._generate_quantization_summary(quantization_result)
        
        return quantization_result
    
    def _quantize_int8(self, tensor: np.ndarray) -> np.ndarray:
        """Simple int8 quantization"""
        tensor_min, tensor_max = tensor.min(), tensor.max()
        scale = (tensor_max - tensor_min) / 255.0
        zero_point = int(-tensor_min / scale)
        
        quantized = np.clip(np.round(tensor / scale + zero_point), 0, 255).astype(np.uint8)
        dequantized = (quantized.astype(np.float32) - zero_point) * scale
        
        return dequantized
    
    def _quantize_int16(self, tensor: np.ndarray) -> np.ndarray:
        """Simple int16 quantization"""
        tensor_min, tensor_max = tensor.min(), tensor.max()
        scale = (tensor_max - tensor_min) / 65535.0
        zero_point = int(-tensor_min / scale)
        
        quantized = np.clip(np.round(tensor / scale + zero_point), 0, 65535).astype(np.uint16)
        dequantized = (quantized.astype(np.float32) - zero_point) * scale
        
        return dequantized
    
    def _quantize_dynamic(self, tensor: np.ndarray) -> np.ndarray:
        """Dynamic range quantization"""
        # Simplified dynamic quantization
        abs_max = np.abs(tensor).max()
        scale = abs_max / 127.0
        
        quantized = np.clip(np.round(tensor / scale), -128, 127).astype(np.int8)
        dequantized = quantized.astype(np.float32) * scale
        
        return dequantized
    
    def _quantize_static(self, tensor: np.ndarray) -> np.ndarray:
        """Static quantization (simplified)"""
        # Use percentile-based clipping for static quantization
        p1, p99 = np.percentile(tensor, [1, 99])
        clipped = np.clip(tensor, p1, p99)
        
        return self._quantize_int8(clipped)
    
    def _generate_quantization_summary(self, quantization_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of quantization comparison"""
        comparisons = quantization_result["quantization_comparisons"]
        
        successful_schemes = {
            scheme: comp for scheme, comp in comparisons.items()
            if not comp.get("quantization_failed", False)
        }
        
        if not successful_schemes:
            return {"error": "No successful quantization schemes"}
        
        # Rank by accuracy
        accuracy_ranking = []
        for scheme, comp in successful_schemes.items():
            if "match_percentage" in comp:
                accuracy_ranking.append((scheme, comp["match_percentage"]))
        
        accuracy_ranking.sort(key=lambda x: x[1], reverse=True)
        
        return {
            "total_schemes_tested": len(comparisons),
            "successful_schemes": len(successful_schemes),
            "failed_schemes": len(comparisons) - len(successful_schemes),
            "accuracy_ranking": accuracy_ranking,
            "best_scheme": accuracy_ranking[0][0] if accuracy_ranking else None,
            "recommended_schemes": [scheme for scheme, acc in accuracy_ranking if acc > 90.0],
        }
    
    def _save_comparison_results(self) -> None:
        """Save all comparison results"""
        if self.comparisons:
            self.save_json(self.comparisons, "multi_dtype_comparisons.json")
            
            # Generate summary report
            summary = {
                "total_comparisons": len(self.comparisons),
                "comparison_keys": list(self.comparisons.keys()),
                "dtype_support": {
                    "reference_dtype": self.reference_dtype,
                    "comparison_dtypes": self.comparison_dtypes,
                    "supported_tolerances": list(self.dtype_tolerances.keys()),
                }
            }
            
            self.save_json(summary, "multi_dtype_summary.json")
            
            if self.verbose:
                self.logger.info(f"Saved {len(self.comparisons)} multi-dtype comparisons")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comparer statistics"""
        return {
            "total_comparisons": len(self.comparisons),
            "reference_dtype": self.reference_dtype,
            "comparison_dtypes": self.comparison_dtypes,
            "torch_available": self.torch_available,
        }