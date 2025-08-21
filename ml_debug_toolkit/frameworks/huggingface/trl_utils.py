"""
TRL (Transformer Reinforcement Learning) debugging utilities
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable

import torch
import torch.nn as nn

from ...core.base import BaseDebugTool


class TRLDebugger(BaseDebugTool):
    """Debugging utilities for TRL training (PPO, DPO, etc.)"""
    
    def __init__(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        verbose: bool = True,
        track_rewards: bool = True,
        track_kl_divergence: bool = True,
        monitor_value_function: bool = True,
    ):
        super().__init__(output_dir, verbose)
        self.track_rewards = track_rewards
        self.track_kl_divergence = track_kl_divergence
        self.monitor_value_function = monitor_value_function
        
        self.reward_analyses: List[Dict[str, Any]] = []
        self.kl_analyses: List[Dict[str, Any]] = []
        self.value_function_analyses: List[Dict[str, Any]] = []
        self.training_step_data: List[Dict[str, Any]] = []
        
        # Check if TRL is available
        try:
            import trl
            self.trl_available = True
            self.trl = trl
        except ImportError:
            self.trl_available = False
            if verbose:
                self.logger.warning("TRL not available - TRLDebugger will have limited functionality")
    
    def enable(self) -> None:
        """Enable TRL debugging"""
        self.enabled = True
        
        if self.verbose:
            status = "enabled" if self.trl_available else "enabled (limited - no TRL)"
            self.logger.info(f"TRL debugger {status}")
    
    def disable(self) -> None:
        """Disable TRL debugging and save results"""
        self.enabled = False
        self._save_debug_data()
        
        if self.verbose:
            self.logger.info("TRL debugger disabled")
    
    def analyze_reward_distribution(
        self,
        rewards: torch.Tensor,
        step_name: str = "reward_analysis",
        baseline_rewards: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Analyze reward distribution and statistics"""
        if not self.enabled:
            raise RuntimeError("TRLDebugger is not enabled")
        
        reward_analysis = {
            "step_name": step_name,
            "timestamp": time.time(),
            "reward_statistics": {},
            "distribution_analysis": {},
            "outlier_analysis": {},
            "baseline_comparison": {},
            "potential_issues": [],
            "recommendations": [],
        }
        
        # Basic reward statistics
        with torch.no_grad():
            reward_stats = {
                "count": rewards.numel(),
                "mean": float(rewards.mean()),
                "std": float(rewards.std()),
                "min": float(rewards.min()),
                "max": float(rewards.max()),
                "median": float(rewards.median()),
                "q25": float(rewards.quantile(0.25)),
                "q75": float(rewards.quantile(0.75)),
                "zero_fraction": float((rewards == 0).float().mean()),
                "positive_fraction": float((rewards > 0).float().mean()),
                "negative_fraction": float((rewards < 0).float().mean()),
            }
            
            reward_analysis["reward_statistics"] = reward_stats
        
        # Distribution analysis
        try:
            # Check for distribution shape
            skewness = self._calculate_skewness(rewards)
            kurtosis = self._calculate_kurtosis(rewards)
            
            reward_analysis["distribution_analysis"] = {
                "skewness": skewness,
                "kurtosis": kurtosis,
                "is_approximately_normal": abs(skewness) < 2 and abs(kurtosis) < 7,
                "distribution_type": self._classify_distribution(skewness, kurtosis),
            }
        except Exception as e:
            reward_analysis["distribution_analysis"] = {"error": str(e)}
        
        # Outlier analysis
        try:
            iqr = reward_stats["q75"] - reward_stats["q25"]
            lower_bound = reward_stats["q25"] - 1.5 * iqr
            upper_bound = reward_stats["q75"] + 1.5 * iqr
            
            outliers = rewards[(rewards < lower_bound) | (rewards > upper_bound)]
            
            reward_analysis["outlier_analysis"] = {
                "outlier_count": len(outliers),
                "outlier_fraction": len(outliers) / len(rewards),
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "outlier_values": outliers.tolist() if len(outliers) < 20 else f"{len(outliers)} outliers",
            }
        except Exception as e:
            reward_analysis["outlier_analysis"] = {"error": str(e)}
        
        # Baseline comparison if provided
        if baseline_rewards is not None:
            try:
                with torch.no_grad():
                    reward_improvement = rewards.mean() - baseline_rewards.mean()
                    correlation = torch.corrcoef(torch.stack([rewards.flatten(), baseline_rewards.flatten()]))[0, 1]
                    
                    reward_analysis["baseline_comparison"] = {
                        "baseline_mean": float(baseline_rewards.mean()),
                        "improvement": float(reward_improvement),
                        "relative_improvement": float(reward_improvement / baseline_rewards.mean()) if baseline_rewards.mean() != 0 else float('inf'),
                        "correlation": float(correlation),
                    }
            except Exception as e:
                reward_analysis["baseline_comparison"] = {"error": str(e)}
        
        # Generate insights and recommendations
        if reward_stats["std"] < 0.01:
            reward_analysis["potential_issues"].append("Very low reward variance - check reward scaling")
        
        if reward_stats["zero_fraction"] > 0.5:
            reward_analysis["potential_issues"].append(f"High fraction of zero rewards ({reward_stats['zero_fraction']:.1%})")
        
        if reward_analysis["outlier_analysis"].get("outlier_fraction", 0) > 0.1:
            reward_analysis["potential_issues"].append("High fraction of outliers - consider reward clipping")
        
        # Recommendations
        if reward_stats["mean"] < 0:
            reward_analysis["recommendations"].append("Negative mean reward - consider reward shaping")
        
        if reward_stats["std"] > abs(reward_stats["mean"]) * 2:
            reward_analysis["recommendations"].append("High reward variance - consider reward normalization")
        
        self.reward_analyses.append(reward_analysis)
        
        if self.verbose:
            mean_reward = reward_stats["mean"]
            std_reward = reward_stats["std"]
            self.logger.info(f"Reward analysis '{step_name}': mean={mean_reward:.4f}, std={std_reward:.4f}")
        
        return reward_analysis
    
    def analyze_kl_divergence(
        self,
        log_probs_current: torch.Tensor,
        log_probs_reference: torch.Tensor,
        analysis_name: str = "kl_analysis",
        target_kl: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Analyze KL divergence between current and reference policies"""
        if not self.enabled:
            raise RuntimeError("TRLDebugger is not enabled")
        
        kl_analysis = {
            "analysis_name": analysis_name,
            "timestamp": time.time(),
            "kl_statistics": {},
            "constraint_analysis": {},
            "distribution_analysis": {},
            "potential_issues": [],
            "recommendations": [],
        }
        
        try:
            with torch.no_grad():
                # Calculate KL divergence
                kl_div = log_probs_current - log_probs_reference
                kl_values = kl_div.exp() * kl_div  # KL(P||Q) = P * log(P/Q)
                
                # Aggregate KL statistics
                kl_stats = {
                    "mean_kl": float(kl_values.mean()),
                    "std_kl": float(kl_values.std()),
                    "min_kl": float(kl_values.min()),
                    "max_kl": float(kl_values.max()),
                    "median_kl": float(kl_values.median()),
                    "q95_kl": float(kl_values.quantile(0.95)),
                    "q99_kl": float(kl_values.quantile(0.99)),
                }
                
                kl_analysis["kl_statistics"] = kl_stats
                
                # Constraint analysis
                if target_kl is not None:
                    constraint_violations = (kl_values > target_kl).float().mean()
                    kl_analysis["constraint_analysis"] = {
                        "target_kl": target_kl,
                        "violation_rate": float(constraint_violations),
                        "mean_excess": float((kl_values - target_kl).clamp(min=0).mean()),
                        "constraint_satisfied": kl_stats["mean_kl"] <= target_kl,
                    }
                
                # Check for potential issues
                if kl_stats["mean_kl"] > 1.0:
                    kl_analysis["potential_issues"].append(f"High mean KL divergence: {kl_stats['mean_kl']:.4f}")
                
                if kl_stats["std_kl"] > kl_stats["mean_kl"] * 2:
                    kl_analysis["potential_issues"].append("High KL variance - unstable policy updates")
                
                if kl_stats["max_kl"] > 10.0:
                    kl_analysis["potential_issues"].append(f"Very high max KL: {kl_stats['max_kl']:.4f}")
                
                # Generate recommendations
                if target_kl is not None and kl_stats["mean_kl"] > target_kl:
                    kl_analysis["recommendations"].append("Mean KL exceeds target - consider reducing learning rate")
                
                if kl_stats["std_kl"] > 1.0:
                    kl_analysis["recommendations"].append("High KL variance - consider gradient clipping or smaller updates")
                
        except Exception as e:
            kl_analysis["error"] = str(e)
        
        self.kl_analyses.append(kl_analysis)
        
        if self.verbose and "kl_statistics" in kl_analysis:
            mean_kl = kl_analysis["kl_statistics"]["mean_kl"]
            self.logger.info(f"KL analysis '{analysis_name}': mean KL = {mean_kl:.6f}")
        
        return kl_analysis
    
    def _calculate_skewness(self, tensor: torch.Tensor) -> float:
        """Calculate skewness of tensor values"""
        with torch.no_grad():
            mean = tensor.mean()
            std = tensor.std()
            if std == 0:
                return 0.0
            return float(((tensor - mean) / std).pow(3).mean())
    
    def _calculate_kurtosis(self, tensor: torch.Tensor) -> float:
        """Calculate kurtosis of tensor values"""
        with torch.no_grad():
            mean = tensor.mean()
            std = tensor.std()
            if std == 0:
                return 0.0
            return float(((tensor - mean) / std).pow(4).mean()) - 3.0  # Excess kurtosis
    
    def _classify_distribution(self, skewness: float, kurtosis: float) -> str:
        """Classify distribution type based on skewness and kurtosis"""
        if abs(skewness) < 0.5 and abs(kurtosis) < 0.5:
            return "approximately_normal"
        elif skewness > 1:
            return "right_skewed"
        elif skewness < -1:
            return "left_skewed"
        elif kurtosis > 2:
            return "heavy_tailed"
        elif kurtosis < -1:
            return "light_tailed"
        else:
            return "moderately_skewed"
    
    def get_trl_summary(self) -> Dict[str, Any]:
        """Get comprehensive TRL debugging summary"""
        return {
            "trl_available": self.trl_available,
            "reward_analyses": len(self.reward_analyses),
            "kl_analyses": len(self.kl_analyses),
            "value_function_analyses": len(self.value_function_analyses),
            "training_step_analyses": len(self.training_step_data),
            "enabled": self.enabled,
        }
    
    def _save_debug_data(self) -> None:
        """Save TRL debug data"""
        debug_data = {
            "trl_available": self.trl_available,
            "reward_analyses": self.reward_analyses,
            "kl_analyses": self.kl_analyses,
            "value_function_analyses": self.value_function_analyses,
            "training_step_data": self.training_step_data,
            "timestamp": time.time(),
        }
        
        if self.trl_available:
            try:
                debug_data["trl_version"] = self.trl.__version__
            except:
                pass
        
        self.save_json(debug_data, "trl_debug_data.json")
        
        # Generate summary
        summary = self.get_trl_summary()
        self.save_json(summary, "trl_debug_summary.json")
        
        if self.verbose:
            self.logger.info(f"TRL debug data saved to {self.output_dir}")


def auto_trl_debug(
    output_dir: Optional[Union[str, Path]] = None,
    **kwargs
) -> TRLDebugger:
    """
    Quick TRL debugging setup
    
    Args:
        output_dir: Output directory
        **kwargs: Additional arguments for TRLDebugger
    
    Returns:
        Enabled TRLDebugger instance
    
    Example:
        >>> trl_debugger = auto_trl_debug()
        >>> # Analyze rewards, KL divergence, value function
        >>> reward_analysis = trl_debugger.analyze_reward_distribution(rewards)
        >>> kl_analysis = trl_debugger.analyze_kl_divergence(log_probs, ref_log_probs)
        >>> trl_debugger.disable()
    """
    debugger = TRLDebugger(output_dir=output_dir, **kwargs)
    debugger.enable()
    return debugger