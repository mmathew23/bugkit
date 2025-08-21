"""
Loss curve analyzer and comparer for analyzing training results
"""

import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks, savgol_filter

from ..core.base import BaseDebugTool


class LossAnalyzer(BaseDebugTool):
    """Analyze and compare loss curves from different training runs"""
    
    def __init__(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        verbose: bool = True,
        smoothing_window: int = 5,
        significance_threshold: float = 0.05,
    ):
        super().__init__(output_dir, verbose)
        self.smoothing_window = smoothing_window
        self.significance_threshold = significance_threshold
        
        self.loaded_runs: Dict[str, Dict[str, Any]] = {}
        self.comparisons: List[Dict[str, Any]] = []
        
    def enable(self) -> None:
        """Enable loss analyzer"""
        self.enabled = True
        if self.verbose:
            self.logger.info("Loss analyzer enabled")
    
    def disable(self) -> None:
        """Disable loss analyzer and save results"""
        self.enabled = False
        self._save_analysis_results()
        if self.verbose:
            self.logger.info("Loss analyzer disabled and results saved")
    
    def load_run_from_json(self, json_file: Union[str, Path], run_id: Optional[str] = None) -> Dict[str, Any]:
        """Load training run data from JSON file"""
        json_file = Path(json_file)
        if not json_file.exists():
            raise FileNotFoundError(f"JSON file not found: {json_file}")
        
        if run_id is None:
            run_id = json_file.stem
        
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Process and normalize the data
        processed_data = self._process_run_data(data, run_id, str(json_file))
        self.loaded_runs[run_id] = processed_data
        
        if self.verbose:
            self.logger.info(f"Loaded run {run_id} from {json_file}")
        
        return processed_data
    
    def load_run_from_csv(self, csv_file: Union[str, Path], run_id: Optional[str] = None) -> Dict[str, Any]:
        """Load training run data from CSV file"""
        csv_file = Path(csv_file)
        if not csv_file.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_file}")
        
        if run_id is None:
            run_id = csv_file.stem
        
        df = pd.read_csv(csv_file)
        
        # Convert DataFrame to nested dict format
        data = {"metrics_history": {}}
        for phase in df['phase'].unique():
            phase_df = df[df['phase'] == phase]
            data["metrics_history"][phase] = {}
            
            for metric in phase_df['metric'].unique():
                metric_df = phase_df[phase_df['metric'] == metric]
                values = metric_df['value'].tolist()
                steps = metric_df['step'].tolist()
                
                data["metrics_history"][phase][metric] = values
                data["metrics_history"][phase][f"{metric}_steps"] = steps
        
        processed_data = self._process_run_data(data, run_id, str(csv_file))
        self.loaded_runs[run_id] = processed_data
        
        if self.verbose:
            self.logger.info(f"Loaded run {run_id} from {csv_file}")
        
        return processed_data
    
    def _process_run_data(self, raw_data: Dict[str, Any], run_id: str, source_file: str) -> Dict[str, Any]:
        """Process and normalize run data"""
        processed = {
            "run_id": run_id,
            "source_file": source_file,
            "metrics_history": raw_data.get("metrics_history", {}),
            "metadata": raw_data.get("metadata", {}),
            "epoch_times": raw_data.get("epoch_times", []),
            "processed_metrics": {},
            "analysis": {},
        }
        
        # Process each phase
        for phase, phase_metrics in processed["metrics_history"].items():
            processed["processed_metrics"][phase] = {}
            
            for metric_name, values in phase_metrics.items():
                if not metric_name.endswith("_steps") and isinstance(values, list):
                    # Calculate derived metrics
                    processed_metrics = self._calculate_metric_statistics(values, metric_name)
                    processed["processed_metrics"][phase][metric_name] = processed_metrics
        
        # Perform basic analysis
        processed["analysis"] = self._analyze_single_run(processed)
        
        return processed
    
    def _calculate_metric_statistics(self, values: List[float], metric_name: str) -> Dict[str, Any]:
        """Calculate comprehensive statistics for a metric"""
        if not values:
            return {}
        
        values_array = np.array(values)
        
        stats_dict = {
            "final_value": values[-1],
            "best_value": min(values) if "loss" in metric_name.lower() else max(values),
            "worst_value": max(values) if "loss" in metric_name.lower() else min(values),
            "mean": np.mean(values_array),
            "std": np.std(values_array),
            "median": np.median(values_array),
            "min": np.min(values_array),
            "max": np.max(values_array),
            "range": np.max(values_array) - np.min(values_array),
            "coefficient_of_variation": np.std(values_array) / np.mean(values_array) if np.mean(values_array) != 0 else float('inf'),
            "total_points": len(values),
        }
        
        # Calculate additional derived metrics
        if len(values) > 1:
            # Trend analysis
            x = np.arange(len(values))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, values_array)
            
            stats_dict.update({
                "trend_slope": slope,
                "trend_intercept": intercept,
                "trend_r_squared": r_value ** 2,
                "trend_p_value": p_value,
                "trend_direction": "improving" if (slope < 0 and "loss" in metric_name.lower()) or (slope > 0 and "loss" not in metric_name.lower()) else "worsening",
            })
            
            # Smoothed version
            if len(values) >= self.smoothing_window:
                smoothed = savgol_filter(values_array, min(self.smoothing_window, len(values)//2*2+1), 2)
                stats_dict["smoothed_values"] = smoothed.tolist()
                stats_dict["smoothed_final"] = smoothed[-1]
            
            # Improvement calculation
            if "loss" in metric_name.lower():
                improvement = (values[0] - values[-1]) / values[0] if values[0] != 0 else 0
            else:
                improvement = (values[-1] - values[0]) / values[0] if values[0] != 0 else float('inf')
            
            stats_dict["total_improvement_percent"] = improvement * 100
            
            # Convergence analysis
            convergence_info = self._analyze_convergence(values, metric_name)
            stats_dict.update(convergence_info)
        
        return stats_dict
    
    def _analyze_convergence(self, values: List[float], metric_name: str) -> Dict[str, Any]:
        """Analyze convergence patterns"""
        if len(values) < 10:
            return {"convergence_analysis": "insufficient_data"}
        
        values_array = np.array(values)
        convergence_info = {}
        
        # Find epochs to reach different improvement levels
        initial_val = values[0]
        final_val = values[-1]
        
        if "loss" in metric_name.lower():
            total_improvement = initial_val - final_val
            is_loss = True
        else:
            total_improvement = final_val - initial_val
            is_loss = False
        
        if total_improvement > 0:
            # Calculate epochs to reach 50%, 90%, 95% of final improvement
            improvement_thresholds = [0.5, 0.9, 0.95]
            
            for threshold in improvement_thresholds:
                target_improvement = threshold * total_improvement
                
                if is_loss:
                    target_value = initial_val - target_improvement
                    epoch_reached = next((i for i, v in enumerate(values) if v <= target_value), None)
                else:
                    target_value = initial_val + target_improvement
                    epoch_reached = next((i for i, v in enumerate(values) if v >= target_value), None)
                
                convergence_info[f"epochs_to_{int(threshold*100)}percent"] = epoch_reached
        
        # Detect plateaus
        plateaus = self._detect_plateaus_in_series(values, patience=5, min_delta=0.001)
        convergence_info["plateaus"] = plateaus
        
        # Calculate recent stability (last 20% of training)
        recent_portion = max(1, len(values) // 5)
        recent_values = values[-recent_portion:]
        recent_std = np.std(recent_values)
        recent_mean = np.mean(recent_values)
        
        convergence_info.update({
            "recent_stability_cv": recent_std / recent_mean if recent_mean != 0 else float('inf'),
            "recent_std": recent_std,
            "recent_mean": recent_mean,
            "is_stable": recent_std / recent_mean < 0.1 if recent_mean != 0 else False,
        })
        
        return convergence_info
    
    def _detect_plateaus_in_series(self, values: List[float], patience: int = 5, min_delta: float = 0.001) -> List[Dict[str, Any]]:
        """Detect plateaus in a metric series"""
        plateaus = []
        
        if len(values) < patience * 2:
            return plateaus
        
        current_plateau_start = None
        
        for i in range(patience, len(values)):
            recent_values = values[i-patience:i]
            value_range = max(recent_values) - min(recent_values)
            
            if value_range < min_delta:
                if current_plateau_start is None:
                    current_plateau_start = i - patience
            else:
                if current_plateau_start is not None:
                    plateaus.append({
                        "start_epoch": current_plateau_start,
                        "end_epoch": i - 1,
                        "duration": i - 1 - current_plateau_start,
                        "avg_value": np.mean(values[current_plateau_start:i]),
                        "value_range": value_range
                    })
                    current_plateau_start = None
        
        # Handle plateau extending to end
        if current_plateau_start is not None:
            plateaus.append({
                "start_epoch": current_plateau_start,
                "end_epoch": len(values) - 1,
                "duration": len(values) - 1 - current_plateau_start,
                "avg_value": np.mean(values[current_plateau_start:]),
                "value_range": max(values[current_plateau_start:]) - min(values[current_plateau_start:])
            })
        
        return plateaus
    
    def _analyze_single_run(self, run_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive analysis on a single run"""
        analysis = {
            "run_summary": {},
            "convergence_analysis": {},
            "stability_analysis": {},
            "efficiency_analysis": {},
            "anomaly_detection": {},
            "recommendations": [],
        }
        
        processed_metrics = run_data["processed_metrics"]
        
        # Run summary
        total_metrics = sum(len(phase_metrics) for phase_metrics in processed_metrics.values())
        analysis["run_summary"] = {
            "total_phases": len(processed_metrics),
            "total_metrics": total_metrics,
            "phase_names": list(processed_metrics.keys()),
        }
        
        # Convergence analysis
        for phase, phase_metrics in processed_metrics.items():
            for metric_name, metric_stats in phase_metrics.items():
                key = f"{phase}_{metric_name}"
                
                if "trend_slope" in metric_stats:
                    analysis["convergence_analysis"][key] = {
                        "trend_direction": metric_stats["trend_direction"],
                        "r_squared": metric_stats["trend_r_squared"],
                        "total_improvement_percent": metric_stats["total_improvement_percent"],
                        "epochs_to_90_percent": metric_stats.get("epochs_to_90percent"),
                        "plateaus_detected": len(metric_stats.get("plateaus", [])),
                    }
                
                # Stability analysis
                analysis["stability_analysis"][key] = {
                    "coefficient_of_variation": metric_stats["coefficient_of_variation"],
                    "recent_stability_cv": metric_stats.get("recent_stability_cv"),
                    "is_stable": metric_stats.get("is_stable", False),
                }
        
        # Efficiency analysis
        if run_data["epoch_times"]:
            epoch_times = run_data["epoch_times"]
            analysis["efficiency_analysis"] = {
                "avg_epoch_time": np.mean(epoch_times),
                "total_training_time": sum(epoch_times),
                "epoch_time_trend": self._calculate_time_trend(epoch_times),
                "time_efficiency": "improving" if len(epoch_times) > 5 and np.mean(epoch_times[-5:]) < np.mean(epoch_times[:5]) else "stable",
            }
        
        # Anomaly detection
        analysis["anomaly_detection"] = self._detect_anomalies(run_data)
        
        # Generate recommendations
        analysis["recommendations"] = self._generate_run_recommendations(analysis)
        
        return analysis
    
    def _calculate_time_trend(self, epoch_times: List[float]) -> Dict[str, float]:
        """Calculate trend in epoch times"""
        if len(epoch_times) < 2:
            return {}
        
        x = np.arange(len(epoch_times))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, epoch_times)
        
        return {
            "slope": slope,
            "r_squared": r_value ** 2,
            "direction": "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"
        }
    
    def _detect_anomalies(self, run_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect anomalies in training data"""
        anomalies = {
            "spikes": [],
            "sudden_changes": [],
            "outliers": [],
        }
        
        for phase, phase_metrics in run_data["metrics_history"].items():
            for metric_name, values in phase_metrics.items():
                if not metric_name.endswith("_steps") and len(values) > 10:
                    # Detect spikes using z-score
                    z_scores = np.abs(stats.zscore(values))
                    spike_indices = np.where(z_scores > 3)[0]
                    
                    for idx in spike_indices:
                        anomalies["spikes"].append({
                            "phase": phase,
                            "metric": metric_name,
                            "epoch": idx,
                            "value": values[idx],
                            "z_score": z_scores[idx]
                        })
                    
                    # Detect sudden changes
                    if len(values) > 1:
                        diffs = np.diff(values)
                        diff_std = np.std(diffs)
                        large_changes = np.where(np.abs(diffs) > 3 * diff_std)[0]
                        
                        for idx in large_changes:
                            anomalies["sudden_changes"].append({
                                "phase": phase,
                                "metric": metric_name,
                                "epoch": idx + 1,
                                "change": diffs[idx],
                                "from_value": values[idx],
                                "to_value": values[idx + 1]
                            })
        
        return anomalies
    
    def _generate_run_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations for a single run"""
        recommendations = []
        
        # Convergence recommendations
        convergence = analysis.get("convergence_analysis", {})
        non_converging = [k for k, v in convergence.items() if v.get("trend_direction") == "worsening"]
        
        if non_converging:
            recommendations.append(
                f"Non-converging metrics detected: {', '.join(non_converging)}. "
                "Consider adjusting learning rate or model architecture."
            )
        
        # Stability recommendations
        stability = analysis.get("stability_analysis", {})
        unstable_metrics = [k for k, v in stability.items() if not v.get("is_stable", True)]
        
        if unstable_metrics:
            recommendations.append(
                f"Unstable training in: {', '.join(unstable_metrics)}. "
                "Consider reducing learning rate or increasing batch size."
            )
        
        # Efficiency recommendations
        efficiency = analysis.get("efficiency_analysis", {})
        if efficiency.get("time_efficiency") == "worsening":
            recommendations.append(
                "Training time is increasing per epoch. Check for memory leaks or "
                "computational inefficiencies."
            )
        
        # Anomaly recommendations
        anomalies = analysis.get("anomaly_detection", {})
        if anomalies.get("spikes"):
            recommendations.append(
                f"Training spikes detected ({len(anomalies['spikes'])} instances). "
                "Consider gradient clipping or learning rate adjustment."
            )
        
        return recommendations
    
    def compare_runs(
        self,
        run_ids: List[str],
        comparison_name: Optional[str] = None,
        focus_metrics: Optional[List[str]] = None,
        phases: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Compare multiple training runs"""
        if not self.enabled:
            raise RuntimeError("LossAnalyzer is not enabled")
        
        # Validate run IDs
        missing_runs = [run_id for run_id in run_ids if run_id not in self.loaded_runs]
        if missing_runs:
            raise ValueError(f"Runs not found: {missing_runs}")
        
        if comparison_name is None:
            comparison_name = "_vs_".join(run_ids)
        
        if self.verbose:
            self.logger.info(f"Comparing runs: {', '.join(run_ids)}")
        
        comparison = {
            "comparison_name": comparison_name,
            "run_ids": run_ids,
            "timestamp": time.time(),
            "metric_comparisons": {},
            "convergence_comparison": {},
            "efficiency_comparison": {},
            "statistical_tests": {},
            "ranking": {},
            "summary": {},
            "recommendations": [],
        }
        
        # Get common metrics across all runs
        common_metrics = self._find_common_metrics(run_ids, phases)
        focus_metrics = focus_metrics or list(common_metrics.keys())
        
        # Compare each metric
        for metric_key in focus_metrics:
            if metric_key in common_metrics:
                metric_comparison = self._compare_metric_across_runs(
                    run_ids, metric_key, common_metrics[metric_key]
                )
                comparison["metric_comparisons"][metric_key] = metric_comparison
        
        # Convergence comparison
        comparison["convergence_comparison"] = self._compare_convergence(run_ids)
        
        # Efficiency comparison  
        comparison["efficiency_comparison"] = self._compare_efficiency(run_ids)
        
        # Statistical significance tests
        comparison["statistical_tests"] = self._perform_statistical_tests(run_ids, focus_metrics)
        
        # Ranking
        comparison["ranking"] = self._rank_runs(run_ids, focus_metrics)
        
        # Summary
        comparison["summary"] = self._generate_comparison_summary(comparison)
        
        # Recommendations
        comparison["recommendations"] = self._generate_comparison_recommendations(comparison)
        
        self.comparisons.append(comparison)
        
        if self.verbose:
            self.logger.info(f"Comparison completed: {comparison_name}")
        
        return comparison
    
    def _find_common_metrics(self, run_ids: List[str], phases: Optional[List[str]] = None) -> Dict[str, Tuple[str, str]]:
        """Find metrics common to all runs"""
        common_metrics = {}
        
        # Get all possible metric combinations
        all_metrics = set()
        for run_id in run_ids:
            run_data = self.loaded_runs[run_id]
            for phase, phase_metrics in run_data["processed_metrics"].items():
                if phases is None or phase in phases:
                    for metric_name in phase_metrics.keys():
                        all_metrics.add((phase, metric_name))
        
        # Check which metrics exist in all runs
        for phase, metric_name in all_metrics:
            metric_key = f"{phase}_{metric_name}"
            
            exists_in_all = True
            for run_id in run_ids:
                run_data = self.loaded_runs[run_id]
                if (phase not in run_data["processed_metrics"] or 
                    metric_name not in run_data["processed_metrics"][phase]):
                    exists_in_all = False
                    break
            
            if exists_in_all:
                common_metrics[metric_key] = (phase, metric_name)
        
        return common_metrics
    
    def _compare_metric_across_runs(
        self, 
        run_ids: List[str], 
        metric_key: str, 
        metric_info: Tuple[str, str]
    ) -> Dict[str, Any]:
        """Compare a specific metric across runs"""
        phase, metric_name = metric_info
        
        comparison = {
            "metric_name": metric_name,
            "phase": phase,
            "run_statistics": {},
            "best_run": None,
            "worst_run": None,
            "statistical_significance": {},
            "improvement_analysis": {},
        }
        
        # Collect statistics for each run
        run_stats = {}
        final_values = {}
        best_values = {}
        
        for run_id in run_ids:
            run_data = self.loaded_runs[run_id]
            metric_stats = run_data["processed_metrics"][phase][metric_name]
            
            run_stats[run_id] = {
                "final_value": metric_stats["final_value"],
                "best_value": metric_stats["best_value"],
                "total_improvement_percent": metric_stats.get("total_improvement_percent", 0),
                "convergence_speed": metric_stats.get("epochs_to_90percent"),
                "stability": metric_stats.get("is_stable", False),
            }
            
            final_values[run_id] = metric_stats["final_value"]
            best_values[run_id] = metric_stats["best_value"]
        
        comparison["run_statistics"] = run_stats
        
        # Determine best and worst runs
        is_loss_metric = "loss" in metric_name.lower()
        
        if is_loss_metric:
            best_run = min(best_values, key=best_values.get)
            worst_run = max(best_values, key=best_values.get)
        else:
            best_run = max(best_values, key=best_values.get)
            worst_run = min(best_values, key=best_values.get)
        
        comparison["best_run"] = {
            "run_id": best_run,
            "value": best_values[best_run],
            "improvement_over_worst": self._calculate_improvement(
                best_values[best_run], best_values[worst_run], is_loss_metric
            )
        }
        
        comparison["worst_run"] = {
            "run_id": worst_run,
            "value": best_values[worst_run]
        }
        
        return comparison
    
    def _calculate_improvement(self, best_val: float, worst_val: float, is_loss: bool) -> float:
        """Calculate improvement percentage"""
        if worst_val == 0:
            return float('inf') if best_val != 0 else 0
        
        if is_loss:
            return ((worst_val - best_val) / worst_val) * 100
        else:
            return ((best_val - worst_val) / worst_val) * 100
    
    def _compare_convergence(self, run_ids: List[str]) -> Dict[str, Any]:
        """Compare convergence characteristics across runs"""
        convergence_comparison = {
            "convergence_speed": {},
            "stability_comparison": {},
            "plateau_analysis": {},
        }
        
        for run_id in run_ids:
            run_analysis = self.loaded_runs[run_id]["analysis"]
            convergence_analysis = run_analysis.get("convergence_analysis", {})
            
            # Aggregate convergence metrics
            speed_metrics = {}
            stability_metrics = {}
            plateau_counts = {}
            
            for key, metrics in convergence_analysis.items():
                if isinstance(metrics, dict):
                    if "epochs_to_90_percent" in metrics and metrics["epochs_to_90_percent"]:
                        speed_metrics[key] = metrics["epochs_to_90_percent"]
                    
                    if "plateaus_detected" in metrics:
                        plateau_counts[key] = metrics["plateaus_detected"]
            
            convergence_comparison["convergence_speed"][run_id] = speed_metrics
            convergence_comparison["plateau_analysis"][run_id] = plateau_counts
        
        return convergence_comparison
    
    def _compare_efficiency(self, run_ids: List[str]) -> Dict[str, Any]:
        """Compare training efficiency across runs"""
        efficiency_comparison = {
            "training_time": {},
            "convergence_efficiency": {},
            "resource_usage": {},
        }
        
        for run_id in run_ids:
            run_data = self.loaded_runs[run_id]
            efficiency_analysis = run_data["analysis"].get("efficiency_analysis", {})
            
            efficiency_comparison["training_time"][run_id] = {
                "total_time": efficiency_analysis.get("total_training_time"),
                "avg_epoch_time": efficiency_analysis.get("avg_epoch_time"),
                "time_trend": efficiency_analysis.get("epoch_time_trend", {}).get("direction"),
            }
        
        return efficiency_comparison
    
    def _perform_statistical_tests(self, run_ids: List[str], focus_metrics: List[str]) -> Dict[str, Any]:
        """Perform statistical significance tests between runs"""
        statistical_tests = {}
        
        if len(run_ids) < 2:
            return statistical_tests
        
        # For each metric, perform pairwise t-tests
        for metric_key in focus_metrics:
            if metric_key not in self._find_common_metrics(run_ids):
                continue
                
            phase, metric_name = metric_key.split("_", 1)
            
            # Collect raw values for each run
            run_values = {}
            for run_id in run_ids:
                try:
                    values = self.loaded_runs[run_id]["metrics_history"][phase][metric_name]
                    if values:
                        run_values[run_id] = values
                except KeyError:
                    continue
            
            if len(run_values) >= 2:
                # Perform pairwise t-tests
                test_results = {}
                run_list = list(run_values.keys())
                
                for i in range(len(run_list)):
                    for j in range(i + 1, len(run_list)):
                        run1, run2 = run_list[i], run_list[j]
                        
                        # Align lengths for comparison
                        min_len = min(len(run_values[run1]), len(run_values[run2]))
                        values1 = run_values[run1][:min_len]
                        values2 = run_values[run2][:min_len]
                        
                        # Perform t-test
                        if len(values1) > 5 and len(values2) > 5:
                            t_stat, p_value = stats.ttest_ind(values1, values2)
                            
                            test_results[f"{run1}_vs_{run2}"] = {
                                "t_statistic": t_stat,
                                "p_value": p_value,
                                "significant": p_value < self.significance_threshold,
                                "mean_difference": np.mean(values1) - np.mean(values2),
                                "effect_size": (np.mean(values1) - np.mean(values2)) / np.sqrt((np.var(values1) + np.var(values2)) / 2)
                            }
                
                statistical_tests[metric_key] = test_results
        
        return statistical_tests
    
    def _rank_runs(self, run_ids: List[str], focus_metrics: List[str]) -> Dict[str, Any]:
        """Rank runs based on multiple criteria"""
        ranking = {
            "overall_ranking": [],
            "metric_rankings": {},
            "scoring_methodology": "weighted_average",
        }
        
        # Score each run on each metric
        run_scores = {run_id: [] for run_id in run_ids}
        
        common_metrics = self._find_common_metrics(run_ids)
        
        for metric_key in focus_metrics:
            if metric_key not in common_metrics:
                continue
                
            phase, metric_name = common_metrics[metric_key]
            is_loss_metric = "loss" in metric_name.lower()
            
            # Get best values for this metric across all runs
            metric_values = {}
            for run_id in run_ids:
                metric_stats = self.loaded_runs[run_id]["processed_metrics"][phase][metric_name]
                metric_values[run_id] = metric_stats["best_value"]
            
            # Rank runs for this metric
            if is_loss_metric:
                sorted_runs = sorted(metric_values.items(), key=lambda x: x[1])  # Lower is better
            else:
                sorted_runs = sorted(metric_values.items(), key=lambda x: x[1], reverse=True)  # Higher is better
            
            metric_ranking = [{"run_id": run_id, "value": value, "rank": idx + 1} 
                            for idx, (run_id, value) in enumerate(sorted_runs)]
            
            ranking["metric_rankings"][metric_key] = metric_ranking
            
            # Add scores (inverse of rank for aggregation)
            for item in metric_ranking:
                run_scores[item["run_id"]].append(len(run_ids) + 1 - item["rank"])
        
        # Calculate overall ranking
        overall_scores = {}
        for run_id, scores in run_scores.items():
            if scores:
                overall_scores[run_id] = np.mean(scores)
        
        overall_ranking = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)
        ranking["overall_ranking"] = [
            {"run_id": run_id, "score": score, "rank": idx + 1}
            for idx, (run_id, score) in enumerate(overall_ranking)
        ]
        
        return ranking
    
    def _generate_comparison_summary(self, comparison: Dict[str, Any]) -> Dict[str, Any]:
        """Generate high-level comparison summary"""
        ranking = comparison.get("ranking", {})
        overall_ranking = ranking.get("overall_ranking", [])
        
        summary = {
            "total_runs": len(comparison["run_ids"]),
            "metrics_compared": len(comparison["metric_comparisons"]),
            "best_overall_run": overall_ranking[0]["run_id"] if overall_ranking else None,
            "worst_overall_run": overall_ranking[-1]["run_id"] if overall_ranking else None,
            "significant_differences": 0,
            "key_findings": [],
        }
        
        # Count significant differences
        statistical_tests = comparison.get("statistical_tests", {})
        for metric_tests in statistical_tests.values():
            for test_result in metric_tests.values():
                if test_result.get("significant", False):
                    summary["significant_differences"] += 1
        
        # Generate key findings
        if summary["best_overall_run"]:
            best_run = summary["best_overall_run"]
            summary["key_findings"].append(f"Run '{best_run}' performed best overall")
        
        # Find metrics with largest improvements
        largest_improvements = []
        for metric_key, metric_comp in comparison["metric_comparisons"].items():
            best_run_info = metric_comp.get("best_run", {})
            improvement = best_run_info.get("improvement_over_worst", 0)
            if improvement > 5:  # >5% improvement
                largest_improvements.append((metric_key, improvement, best_run_info.get("run_id")))
        
        if largest_improvements:
            largest_improvements.sort(key=lambda x: x[1], reverse=True)
            metric, improvement, run_id = largest_improvements[0]
            summary["key_findings"].append(
                f"Largest improvement in {metric}: {improvement:.1f}% by run '{run_id}'"
            )
        
        return summary
    
    def _generate_comparison_recommendations(self, comparison: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on comparison"""
        recommendations = []
        
        ranking = comparison.get("ranking", {})
        summary = comparison.get("summary", {})
        
        # Best run recommendation
        if summary.get("best_overall_run"):
            best_run = summary["best_overall_run"]
            recommendations.append(
                f"Consider using the configuration and hyperparameters from run '{best_run}' "
                "as it performed best overall."
            )
        
        # Significant differences
        if summary.get("significant_differences", 0) > 0:
            recommendations.append(
                f"Found {summary['significant_differences']} statistically significant differences. "
                "Focus on understanding what caused these performance variations."
            )
        
        # Convergence recommendations
        convergence_comp = comparison.get("convergence_comparison", {})
        if convergence_comp:
            # Find runs with fastest convergence
            speed_data = convergence_comp.get("convergence_speed", {})
            if speed_data:
                recommendations.append(
                    "Analyze convergence patterns to identify optimal stopping points "
                    "and early stopping strategies."
                )
        
        return recommendations
    
    def create_comparison_plots(self, comparison_name: str, save_path: Optional[Path] = None) -> Path:
        """Create comprehensive comparison plots"""
        if comparison_name not in [c["comparison_name"] for c in self.comparisons]:
            raise ValueError(f"Comparison {comparison_name} not found")
        
        comparison = next(c for c in self.comparisons if c["comparison_name"] == comparison_name)
        run_ids = comparison["run_ids"]
        
        if save_path is None:
            save_path = self.output_dir / f"comparison_{comparison_name}.png"
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(f"Training Runs Comparison: {comparison_name}", fontsize=16)
        
        # Plot 1: Loss curves
        ax1 = axes[0, 0]
        for run_id in run_ids:
            run_data = self.loaded_runs[run_id]
            for phase in ["train", "val"]:
                if phase in run_data["metrics_history"] and "loss" in run_data["metrics_history"][phase]:
                    loss_values = run_data["metrics_history"][phase]["loss"]
                    ax1.plot(loss_values, label=f"{run_id}_{phase}", linewidth=2)
        
        ax1.set_title("Loss Curves Comparison")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Final performance comparison
        ax2 = axes[0, 1]
        metric_comparisons = comparison["metric_comparisons"]
        
        if metric_comparisons:
            # Create bar plot of final values
            metric_names = []
            run_performances = {run_id: [] for run_id in run_ids}
            
            for metric_key, metric_comp in list(metric_comparisons.items())[:5]:  # Top 5 metrics
                metric_names.append(metric_key)
                run_stats = metric_comp["run_statistics"]
                
                for run_id in run_ids:
                    if run_id in run_stats:
                        run_performances[run_id].append(run_stats[run_id]["final_value"])
                    else:
                        run_performances[run_id].append(0)
            
            x = np.arange(len(metric_names))
            width = 0.8 / len(run_ids)
            
            for i, run_id in enumerate(run_ids):
                ax2.bar(x + i * width, run_performances[run_id], width, label=run_id)
            
            ax2.set_title("Final Performance Comparison")
            ax2.set_xlabel("Metrics")
            ax2.set_ylabel("Value")
            ax2.set_xticks(x + width * (len(run_ids) - 1) / 2)
            ax2.set_xticklabels(metric_names, rotation=45, ha='right')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Overall ranking
        ax3 = axes[0, 2]
        ranking = comparison.get("ranking", {})
        overall_ranking = ranking.get("overall_ranking", [])
        
        if overall_ranking:
            run_names = [item["run_id"] for item in overall_ranking]
            scores = [item["score"] for item in overall_ranking]
            
            bars = ax3.barh(run_names, scores)
            ax3.set_title("Overall Performance Ranking")
            ax3.set_xlabel("Score")
            
            # Color bars by rank
            colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(bars)))
            for bar, color in zip(bars, colors):
                bar.set_color(color)
        
        # Plot 4: Training time comparison
        ax4 = axes[1, 0]
        efficiency_comp = comparison.get("efficiency_comparison", {})
        training_times = efficiency_comp.get("training_time", {})
        
        if training_times:
            runs = []
            times = []
            
            for run_id, time_info in training_times.items():
                total_time = time_info.get("total_time")
                if total_time:
                    runs.append(run_id)
                    times.append(total_time / 3600)  # Convert to hours
            
            if runs and times:
                bars = ax4.bar(runs, times)
                ax4.set_title("Training Time Comparison")
                ax4.set_ylabel("Time (hours)")
                ax4.tick_params(axis='x', rotation=45)
                
                # Color by efficiency
                max_time = max(times)
                colors = plt.cm.RdYlGn([1 - (t / max_time) for t in times])
                for bar, color in zip(bars, colors):
                    bar.set_color(color)
        
        # Plot 5: Convergence speed comparison
        ax5 = axes[1, 1]
        convergence_comp = comparison.get("convergence_comparison", {})
        speed_data = convergence_comp.get("convergence_speed", {})
        
        if speed_data:
            # Average convergence speed across metrics
            run_avg_speeds = {}
            for run_id, metrics in speed_data.items():
                speeds = [v for v in metrics.values() if v is not None]
                if speeds:
                    run_avg_speeds[run_id] = np.mean(speeds)
            
            if run_avg_speeds:
                runs = list(run_avg_speeds.keys())
                speeds = list(run_avg_speeds.values())
                
                bars = ax5.bar(runs, speeds)
                ax5.set_title("Average Convergence Speed")
                ax5.set_ylabel("Epochs to 90% convergence")
                ax5.tick_params(axis='x', rotation=45)
                
                # Color by speed (faster is better)
                max_speed = max(speeds)
                colors = plt.cm.RdYlGn([1 - (s / max_speed) for s in speeds])
                for bar, color in zip(bars, colors):
                    bar.set_color(color)
        
        # Plot 6: Summary statistics
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        # Create summary text
        summary_text = self._generate_comparison_summary_text(comparison)
        ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        if self.verbose:
            self.logger.info(f"Comparison plots saved to {save_path}")
        
        return save_path
    
    def _generate_comparison_summary_text(self, comparison: Dict[str, Any]) -> str:
        """Generate text summary for comparison plots"""
        summary = comparison.get("summary", {})
        ranking = comparison.get("ranking", {})
        
        lines = ["Comparison Summary", "=" * 18, ""]
        
        lines.append(f"Runs Compared: {summary.get('total_runs', 0)}")
        lines.append(f"Metrics Analyzed: {summary.get('metrics_compared', 0)}")
        lines.append("")
        
        if summary.get("best_overall_run"):
            lines.append(f"🏆 Best Run: {summary['best_overall_run']}")
        
        if summary.get("worst_overall_run"):
            lines.append(f"❌ Worst Run: {summary['worst_overall_run']}")
        
        lines.append("")
        lines.append(f"Significant Differences: {summary.get('significant_differences', 0)}")
        
        # Key findings
        if summary.get("key_findings"):
            lines.append("")
            lines.append("Key Findings:")
            for finding in summary["key_findings"][:3]:  # Top 3
                lines.append(f"• {finding}")
        
        return "\n".join(lines)
    
    def _save_analysis_results(self) -> None:
        """Save all analysis results"""
        if not self.loaded_runs and not self.comparisons:
            return
        
        # Save loaded runs analysis
        if self.loaded_runs:
            self.save_json(self.loaded_runs, "analyzed_runs.json")
        
        # Save comparisons
        if self.comparisons:
            self.save_json(self.comparisons, "loss_curve_comparisons.json")
        
        # Generate overall summary
        summary = {
            "total_runs_analyzed": len(self.loaded_runs),
            "total_comparisons": len(self.comparisons),
            "run_ids": list(self.loaded_runs.keys()),
            "comparison_names": [c["comparison_name"] for c in self.comparisons],
        }
        
        self.save_json(summary, "loss_analyzer_summary.json")
        
        if self.verbose:
            self.logger.info(f"Analysis results saved to {self.output_dir}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get analyzer statistics"""
        return {
            "loaded_runs": len(self.loaded_runs),
            "completed_comparisons": len(self.comparisons),
            "run_ids": list(self.loaded_runs.keys()),
            "recent_comparisons": [c["comparison_name"] for c in self.comparisons[-3:]],
        }