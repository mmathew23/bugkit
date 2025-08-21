"""
Tensor comparison utilities with configurable tolerance settings
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ..core.base import BaseDebugTool, format_tensor_info


class TensorComparer(BaseDebugTool):
    """Compare tensors with detailed analysis and configurable tolerances"""
    
    def __init__(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        verbose: bool = True,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        equal_nan: bool = False,
    ):
        super().__init__(output_dir, verbose)
        self.rtol = rtol
        self.atol = atol
        self.equal_nan = equal_nan
        
        self.comparison_history: List[Dict[str, Any]] = []
        self.tolerance_profiles: Dict[str, Dict[str, float]] = {
            "strict": {"rtol": 1e-7, "atol": 1e-10},
            "default": {"rtol": 1e-5, "atol": 1e-8},
            "relaxed": {"rtol": 1e-3, "atol": 1e-6},
            "loose": {"rtol": 1e-2, "atol": 1e-4},
        }
        
    def enable(self) -> None:
        """Enable tensor comparer"""
        self.enabled = True
        if self.verbose:
            self.logger.info("Tensor comparer enabled")
    
    def disable(self) -> None:
        """Disable tensor comparer and save results"""
        self.enabled = False
        self._save_comparisons()
        if self.verbose:
            self.logger.info("Tensor comparer disabled and results saved")
    
    def compare(
        self,
        tensor1: Any,
        tensor2: Any,
        name1: str = "tensor1",
        name2: str = "tensor2",
        rtol: Optional[float] = None,
        atol: Optional[float] = None,
        equal_nan: Optional[bool] = None,
        tolerance_profile: Optional[str] = None,
        return_details: bool = True,
    ) -> Dict[str, Any]:
        """Compare two tensors with detailed analysis"""
        if not self.enabled:
            raise RuntimeError("TensorComparer is not enabled")
        
        # Set tolerance values
        if tolerance_profile and tolerance_profile in self.tolerance_profiles:
            profile = self.tolerance_profiles[tolerance_profile]
            rtol = rtol or profile["rtol"]
            atol = atol or profile["atol"]
        
        rtol = rtol if rtol is not None else self.rtol
        atol = atol if atol is not None else self.atol
        equal_nan = equal_nan if equal_nan is not None else self.equal_nan
        
        comparison_result = {
            "timestamp": time.time(),
            "name1": name1,
            "name2": name2,
            "tolerance_settings": {
                "rtol": rtol,
                "atol": atol,
                "equal_nan": equal_nan,
                "profile": tolerance_profile
            },
            "tensor1_info": format_tensor_info(tensor1),
            "tensor2_info": format_tensor_info(tensor2),
            "comparison": {},
            "statistics": {},
            "diagnostics": {},
        }
        
        try:
            # Convert to numpy arrays for analysis
            arr1, arr2 = self._to_numpy(tensor1), self._to_numpy(tensor2)
            
            # Basic comparison
            comparison_result["comparison"] = self._basic_comparison(
                arr1, arr2, rtol, atol, equal_nan
            )
            
            # Statistical analysis
            comparison_result["statistics"] = self._statistical_analysis(arr1, arr2)
            
            # Detailed diagnostics
            if return_details:
                comparison_result["diagnostics"] = self._detailed_diagnostics(
                    arr1, arr2, rtol, atol
                )
            
            # Overall assessment
            comparison_result["overall_match"] = comparison_result["comparison"]["all_close"]
            comparison_result["match_percentage"] = comparison_result["comparison"]["match_percentage"]
            
            self.comparison_history.append(comparison_result)
            
            if self.verbose:
                match_status = "✓" if comparison_result["overall_match"] else "✗"
                self.logger.info(
                    f"{match_status} Comparison {name1} vs {name2}: "
                    f"{comparison_result['match_percentage']:.2f}% match "
                    f"(rtol={rtol:.2e}, atol={atol:.2e})"
                )
            
            return comparison_result
            
        except Exception as e:
            comparison_result["error"] = {
                "type": type(e).__name__,
                "message": str(e),
                "traceback": str(e)
            }
            comparison_result["overall_match"] = False
            comparison_result["match_percentage"] = 0.0
            
            self.comparison_history.append(comparison_result)
            return comparison_result
    
    def _to_numpy(self, tensor: Any) -> np.ndarray:
        """Convert tensor to numpy array"""
        if isinstance(tensor, np.ndarray):
            return tensor
        
        # Handle PyTorch tensors
        try:
            import torch
            if isinstance(tensor, torch.Tensor):
                return tensor.detach().cpu().numpy()
        except ImportError:
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
        
        # Fallback to numpy conversion
        return np.array(tensor)
    
    def _basic_comparison(
        self, 
        arr1: np.ndarray, 
        arr2: np.ndarray, 
        rtol: float, 
        atol: float, 
        equal_nan: bool
    ) -> Dict[str, Any]:
        """Perform basic tensor comparison"""
        result = {
            "shapes_match": arr1.shape == arr2.shape,
            "dtypes_match": arr1.dtype == arr2.dtype,
            "all_close": False,
            "all_equal": False,
            "match_percentage": 0.0,
            "total_elements": 0,
            "matching_elements": 0,
            "different_elements": 0,
        }
        
        # Shape compatibility check
        if not result["shapes_match"]:
            result["shape_difference"] = {
                "shape1": arr1.shape,
                "shape2": arr2.shape,
                "can_broadcast": self._can_broadcast(arr1.shape, arr2.shape)
            }
            if not result["shape_difference"]["can_broadcast"]:
                return result
        
        # Handle broadcasting if needed
        try:
            if arr1.shape != arr2.shape:
                arr1_broadcast, arr2_broadcast = np.broadcast_arrays(arr1, arr2)
            else:
                arr1_broadcast, arr2_broadcast = arr1, arr2
            
            # Element-wise comparison
            result["total_elements"] = arr1_broadcast.size
            
            # Check for exact equality
            if equal_nan:
                equal_mask = (arr1_broadcast == arr2_broadcast) | (
                    np.isnan(arr1_broadcast) & np.isnan(arr2_broadcast)
                )
            else:
                equal_mask = arr1_broadcast == arr2_broadcast
            
            result["matching_elements"] = np.sum(equal_mask)
            result["different_elements"] = result["total_elements"] - result["matching_elements"]
            result["all_equal"] = result["matching_elements"] == result["total_elements"]
            
            # Check for approximate equality
            result["all_close"] = np.allclose(
                arr1_broadcast, arr2_broadcast, rtol=rtol, atol=atol, equal_nan=equal_nan
            )
            
            # Calculate match percentage based on tolerance
            close_mask = np.isclose(
                arr1_broadcast, arr2_broadcast, rtol=rtol, atol=atol, equal_nan=equal_nan
            )
            result["match_percentage"] = (np.sum(close_mask) / result["total_elements"]) * 100
            
        except Exception as e:
            result["comparison_error"] = str(e)
        
        return result
    
    def _statistical_analysis(self, arr1: np.ndarray, arr2: np.ndarray) -> Dict[str, Any]:
        """Perform statistical analysis of tensor differences"""
        stats = {}
        
        try:
            # Handle broadcasting
            if arr1.shape != arr2.shape:
                arr1, arr2 = np.broadcast_arrays(arr1, arr2)
            
            # Basic statistics for each tensor
            stats["tensor1_stats"] = {
                "mean": float(np.mean(arr1)),
                "std": float(np.std(arr1)),
                "min": float(np.min(arr1)),
                "max": float(np.max(arr1)),
                "median": float(np.median(arr1)),
                "has_nan": bool(np.isnan(arr1).any()),
                "has_inf": bool(np.isinf(arr1).any()),
                "zero_count": int(np.sum(arr1 == 0)),
            }
            
            stats["tensor2_stats"] = {
                "mean": float(np.mean(arr2)),
                "std": float(np.std(arr2)),
                "min": float(np.min(arr2)),
                "max": float(np.max(arr2)),
                "median": float(np.median(arr2)),
                "has_nan": bool(np.isnan(arr2).any()),
                "has_inf": bool(np.isinf(arr2).any()),
                "zero_count": int(np.sum(arr2 == 0)),
            }
            
            # Difference analysis
            diff = arr1 - arr2
            abs_diff = np.abs(diff)
            
            stats["difference_stats"] = {
                "mean_diff": float(np.mean(diff)),
                "std_diff": float(np.std(diff)),
                "mean_abs_diff": float(np.mean(abs_diff)),
                "max_abs_diff": float(np.max(abs_diff)),
                "min_abs_diff": float(np.min(abs_diff)),
                "median_abs_diff": float(np.median(abs_diff)),
                "l1_norm": float(np.linalg.norm(diff.flatten(), ord=1)),
                "l2_norm": float(np.linalg.norm(diff.flatten(), ord=2)),
                "linf_norm": float(np.linalg.norm(diff.flatten(), ord=np.inf)),
            }
            
            # Relative error analysis (avoid division by zero)
            arr2_safe = np.where(arr2 == 0, np.finfo(arr2.dtype).eps, arr2)
            rel_error = np.abs(diff / arr2_safe)
            
            stats["relative_error_stats"] = {
                "mean_rel_error": float(np.mean(rel_error)),
                "max_rel_error": float(np.max(rel_error)),
                "median_rel_error": float(np.median(rel_error)),
            }
            
            # Correlation analysis
            if arr1.size > 1:
                try:
                    correlation_matrix = np.corrcoef(arr1.flatten(), arr2.flatten())
                    stats["correlation"] = float(correlation_matrix[0, 1])
                except:
                    stats["correlation"] = None
            
        except Exception as e:
            stats["error"] = str(e)
        
        return stats
    
    def _detailed_diagnostics(
        self, 
        arr1: np.ndarray, 
        arr2: np.ndarray, 
        rtol: float, 
        atol: float
    ) -> Dict[str, Any]:
        """Perform detailed diagnostic analysis"""
        diagnostics = {}
        
        try:
            if arr1.shape != arr2.shape:
                arr1, arr2 = np.broadcast_arrays(arr1, arr2)
            
            # Find problematic elements
            close_mask = np.isclose(arr1, arr2, rtol=rtol, atol=atol, equal_nan=self.equal_nan)
            problem_mask = ~close_mask
            
            if np.any(problem_mask):
                problem_indices = np.where(problem_mask)
                num_problems = min(10, len(problem_indices[0]))  # Limit to first 10
                
                diagnostics["problem_elements"] = []
                for i in range(num_problems):
                    idx = tuple(coord[i] for coord in problem_indices)
                    diagnostics["problem_elements"].append({
                        "index": idx,
                        "value1": float(arr1[idx]),
                        "value2": float(arr2[idx]),
                        "difference": float(arr1[idx] - arr2[idx]),
                        "relative_error": float(abs(arr1[idx] - arr2[idx]) / (abs(arr2[idx]) + atol))
                    })
                
                diagnostics["total_problem_elements"] = int(np.sum(problem_mask))
            
            # Tolerance analysis
            diff = np.abs(arr1 - arr2)
            threshold = atol + rtol * np.abs(arr2)
            
            diagnostics["tolerance_analysis"] = {
                "elements_within_atol": int(np.sum(diff <= atol)),
                "elements_within_rtol": int(np.sum(diff <= rtol * np.abs(arr2))),
                "elements_within_combined": int(np.sum(diff <= threshold)),
                "max_violation": float(np.max(diff - threshold)),
                "mean_violation": float(np.mean(np.maximum(0, diff - threshold))),
            }
            
            # Distribution analysis
            diagnostics["distribution_analysis"] = self._analyze_distributions(arr1, arr2)
            
        except Exception as e:
            diagnostics["error"] = str(e)
        
        return diagnostics
    
    def _analyze_distributions(self, arr1: np.ndarray, arr2: np.ndarray) -> Dict[str, Any]:
        """Analyze the distributions of the two tensors"""
        try:
            # Histogram comparison
            bins = min(50, max(10, int(np.sqrt(arr1.size))))
            
            # Determine common range
            combined_min = min(np.min(arr1), np.min(arr2))
            combined_max = max(np.max(arr1), np.max(arr2))
            
            if combined_min == combined_max:
                return {"note": "All values are identical"}
            
            hist1, bin_edges = np.histogram(arr1, bins=bins, range=(combined_min, combined_max))
            hist2, _ = np.histogram(arr2, bins=bins, range=(combined_min, combined_max))
            
            # Calculate histogram similarity metrics
            hist1_norm = hist1 / np.sum(hist1)
            hist2_norm = hist2 / np.sum(hist2)
            
            # Chi-squared test statistic
            chi_squared = np.sum((hist1_norm - hist2_norm) ** 2 / (hist1_norm + hist2_norm + 1e-10))
            
            # Kolmogorov-Smirnov test approximation
            cdf1 = np.cumsum(hist1_norm)
            cdf2 = np.cumsum(hist2_norm)
            ks_statistic = np.max(np.abs(cdf1 - cdf2))
            
            return {
                "histogram_similarity": {
                    "chi_squared": float(chi_squared),
                    "ks_statistic": float(ks_statistic),
                    "bin_count": bins,
                    "value_range": [float(combined_min), float(combined_max)]
                }
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _can_broadcast(self, shape1: Tuple[int, ...], shape2: Tuple[int, ...]) -> bool:
        """Check if two shapes can be broadcast together"""
        try:
            np.broadcast_shapes(shape1, shape2)
            return True
        except ValueError:
            return False
    
    def compare_multiple(
        self,
        tensors: List[Any],
        names: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Compare multiple tensors pairwise"""
        if len(tensors) < 2:
            raise ValueError("Need at least 2 tensors for comparison")
        
        if names is None:
            names = [f"tensor_{i}" for i in range(len(tensors))]
        elif len(names) != len(tensors):
            raise ValueError("Number of names must match number of tensors")
        
        results = {
            "tensors": len(tensors),
            "comparisons": [],
            "summary": {
                "total_comparisons": 0,
                "matching_pairs": 0,
                "average_match_percentage": 0.0,
            }
        }
        
        total_match_percentage = 0.0
        
        for i in range(len(tensors)):
            for j in range(i + 1, len(tensors)):
                comparison = self.compare(
                    tensors[i], tensors[j], names[i], names[j], **kwargs
                )
                results["comparisons"].append(comparison)
                results["summary"]["total_comparisons"] += 1
                
                if comparison["overall_match"]:
                    results["summary"]["matching_pairs"] += 1
                
                total_match_percentage += comparison["match_percentage"]
        
        if results["summary"]["total_comparisons"] > 0:
            results["summary"]["average_match_percentage"] = (
                total_match_percentage / results["summary"]["total_comparisons"]
            )
        
        return results
    
    def set_tolerance_profile(self, profile_name: str, rtol: float, atol: float) -> None:
        """Add or update a tolerance profile"""
        self.tolerance_profiles[profile_name] = {"rtol": rtol, "atol": atol}
        if self.verbose:
            self.logger.info(f"Set tolerance profile '{profile_name}': rtol={rtol:.2e}, atol={atol:.2e}")
    
    def _save_comparisons(self) -> None:
        """Save comparison history"""
        if not self.comparison_history:
            return
        
        # Save detailed history
        self.save_json(self.comparison_history, "tensor_comparisons.json")
        
        # Create summary
        summary = {
            "total_comparisons": len(self.comparison_history),
            "successful_comparisons": sum(1 for c in self.comparison_history if c["overall_match"]),
            "average_match_percentage": np.mean([c["match_percentage"] for c in self.comparison_history]),
            "tolerance_profiles_used": list(set(
                c["tolerance_settings"].get("profile") 
                for c in self.comparison_history 
                if c["tolerance_settings"].get("profile")
            )),
            "common_issues": self._analyze_common_issues(),
        }
        
        self.save_json(summary, "tensor_comparison_summary.json")
        
        if self.verbose:
            self.logger.info(f"Saved {len(self.comparison_history)} tensor comparisons to {self.output_dir}")
    
    def _analyze_common_issues(self) -> Dict[str, Any]:
        """Analyze common issues across comparisons"""
        issues = {
            "shape_mismatches": 0,
            "dtype_mismatches": 0,
            "nan_issues": 0,
            "inf_issues": 0,
            "large_relative_errors": 0,
        }
        
        for comparison in self.comparison_history:
            if not comparison.get("comparison", {}).get("shapes_match", True):
                issues["shape_mismatches"] += 1
            
            if not comparison.get("comparison", {}).get("dtypes_match", True):
                issues["dtype_mismatches"] += 1
            
            stats = comparison.get("statistics", {})
            if (stats.get("tensor1_stats", {}).get("has_nan") or 
                stats.get("tensor2_stats", {}).get("has_nan")):
                issues["nan_issues"] += 1
            
            if (stats.get("tensor1_stats", {}).get("has_inf") or 
                stats.get("tensor2_stats", {}).get("has_inf")):
                issues["inf_issues"] += 1
            
            if stats.get("relative_error_stats", {}).get("max_rel_error", 0) > 0.1:
                issues["large_relative_errors"] += 1
        
        return issues
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current comparison statistics"""
        if not self.comparison_history:
            return {"message": "No comparisons performed yet"}
        
        recent = self.comparison_history[-5:] if self.comparison_history else []
        return {
            "total_comparisons": len(self.comparison_history),
            "recent_match_rates": [c["match_percentage"] for c in recent],
            "available_tolerance_profiles": list(self.tolerance_profiles.keys()),
        }