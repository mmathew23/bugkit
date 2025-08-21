"""
Training run and test differ for comparing ML experiments
"""

import json
import os
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

from ..core.base import BaseDebugTool


class TrainingDiffer(BaseDebugTool):
    """Compare training runs and test results to identify differences and regressions"""
    
    def __init__(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        verbose: bool = True,
        metric_tolerance: float = 0.01,  # 1% tolerance for metrics
        significant_change_threshold: float = 0.05,  # 5% for significant changes
    ):
        super().__init__(output_dir, verbose)
        self.metric_tolerance = metric_tolerance
        self.significant_change_threshold = significant_change_threshold
        
        self.comparisons: List[Dict[str, Any]] = []
        self.run_data: Dict[str, Dict[str, Any]] = {}
        
    def enable(self) -> None:
        """Enable training differ"""
        self.enabled = True
        if self.verbose:
            self.logger.info("Training differ enabled")
    
    def disable(self) -> None:
        """Disable training differ and save results"""
        self.enabled = False
        self._save_comparisons()
        if self.verbose:
            self.logger.info("Training differ disabled and results saved")
    
    def load_run_data(
        self,
        run_dir: Union[str, Path],
        run_id: Optional[str] = None,
        config_file: str = "config.json",
        metrics_file: str = "metrics.json",
        logs_file: str = "training.log",
        checkpoints_dir: str = "checkpoints",
    ) -> Dict[str, Any]:
        """Load training run data from directory"""
        run_dir = Path(run_dir)
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        
        if run_id is None:
            run_id = run_dir.name
        
        run_data = {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "config": {},
            "metrics": {},
            "logs": [],
            "checkpoints": [],
            "metadata": {
                "load_time": time.time(),
                "files_found": [],
                "files_missing": [],
            }
        }
        
        # Load configuration
        config_path = run_dir / config_file
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    run_data["config"] = json.load(f)
                run_data["metadata"]["files_found"].append(config_file)
            except Exception as e:
                if self.verbose:
                    self.logger.warning(f"Failed to load config from {config_path}: {e}")
                run_data["metadata"]["files_missing"].append(config_file)
        else:
            run_data["metadata"]["files_missing"].append(config_file)
        
        # Load metrics
        metrics_path = run_dir / metrics_file
        if metrics_path.exists():
            try:
                with open(metrics_path, 'r') as f:
                    metrics_data = json.load(f)
                run_data["metrics"] = self._process_metrics(metrics_data)
                run_data["metadata"]["files_found"].append(metrics_file)
            except Exception as e:
                if self.verbose:
                    self.logger.warning(f"Failed to load metrics from {metrics_path}: {e}")
                run_data["metadata"]["files_missing"].append(metrics_file)
        else:
            run_data["metadata"]["files_missing"].append(metrics_file)
        
        # Load logs
        logs_path = run_dir / logs_file
        if logs_path.exists():
            try:
                run_data["logs"] = self._parse_training_logs(logs_path)
                run_data["metadata"]["files_found"].append(logs_file)
            except Exception as e:
                if self.verbose:
                    self.logger.warning(f"Failed to load logs from {logs_path}: {e}")
                run_data["metadata"]["files_missing"].append(logs_file)
        else:
            run_data["metadata"]["files_missing"].append(logs_file)
        
        # Find checkpoints
        checkpoints_path = run_dir / checkpoints_dir
        if checkpoints_path.exists():
            run_data["checkpoints"] = self._find_checkpoints(checkpoints_path)
            run_data["metadata"]["files_found"].append(checkpoints_dir)
        else:
            run_data["metadata"]["files_missing"].append(checkpoints_dir)
        
        # Extract additional metadata
        run_data["metadata"].update(self._extract_run_metadata(run_dir))
        
        self.run_data[run_id] = run_data
        
        if self.verbose:
            self.logger.info(f"Loaded run data for {run_id}: {len(run_data['metadata']['files_found'])} files found")
        
        return run_data
    
    def compare_runs(
        self,
        run1_id: str,
        run2_id: str,
        comparison_name: Optional[str] = None,
        focus_metrics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Compare two training runs comprehensively"""
        if not self.enabled:
            raise RuntimeError("TrainingDiffer is not enabled")
        
        if run1_id not in self.run_data:
            raise ValueError(f"Run {run1_id} not found. Load it first with load_run_data()")
        if run2_id not in self.run_data:
            raise ValueError(f"Run {run2_id} not found. Load it first with load_run_data()")
        
        run1 = self.run_data[run1_id]
        run2 = self.run_data[run2_id]
        comparison_name = comparison_name or f"{run1_id}_vs_{run2_id}"
        
        if self.verbose:
            self.logger.info(f"Comparing runs: {run1_id} vs {run2_id}")
        
        comparison = {
            "comparison_name": comparison_name,
            "run1_id": run1_id,
            "run2_id": run2_id,
            "timestamp": time.time(),
            
            # Core comparisons
            "config_comparison": self._compare_configs(run1["config"], run2["config"]),
            "metrics_comparison": self._compare_metrics(run1["metrics"], run2["metrics"], focus_metrics),
            "training_comparison": self._compare_training_progress(run1, run2),
            "performance_comparison": self._compare_performance(run1, run2),
            "checkpoint_comparison": self._compare_checkpoints(run1["checkpoints"], run2["checkpoints"]),
            
            # Analysis
            "regression_analysis": {},
            "improvement_analysis": {},
            "summary": {},
            "recommendations": [],
        }
        
        # Perform deeper analysis
        comparison["regression_analysis"] = self._analyze_regressions(comparison)
        comparison["improvement_analysis"] = self._analyze_improvements(comparison)
        comparison["summary"] = self._generate_comparison_summary(comparison)
        comparison["recommendations"] = self._generate_recommendations(comparison)
        
        self.comparisons.append(comparison)
        
        if self.verbose:
            self.logger.info(f"Comparison completed: {comparison_name}")
        
        return comparison
    
    def _process_metrics(self, metrics_data: Any) -> Dict[str, Any]:
        """Process and normalize metrics data"""
        processed = {
            "final_metrics": {},
            "training_history": {},
            "best_metrics": {},
            "metadata": {}
        }
        
        if isinstance(metrics_data, dict):
            # Handle different metrics formats
            if "history" in metrics_data:
                processed["training_history"] = metrics_data["history"]
            elif "train" in metrics_data or "val" in metrics_data:
                processed["training_history"] = metrics_data
            else:
                # Assume it's final metrics
                processed["final_metrics"] = metrics_data
            
            # Extract best metrics
            if processed["training_history"]:
                processed["best_metrics"] = self._extract_best_metrics(processed["training_history"])
            
            # Extract metadata
            processed["metadata"] = {
                "total_epochs": self._extract_total_epochs(processed["training_history"]),
                "metric_names": self._extract_metric_names(processed["training_history"]),
            }
        
        return processed
    
    def _parse_training_logs(self, log_file: Path) -> List[Dict[str, Any]]:
        """Parse training logs for key information"""
        logs = []
        
        try:
            with open(log_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Try to parse as JSON
                    try:
                        log_entry = json.loads(line)
                        log_entry["line_number"] = line_num
                        logs.append(log_entry)
                        continue
                    except json.JSONDecodeError:
                        pass
                    
                    # Parse common log patterns
                    log_entry = self._parse_log_line(line, line_num)
                    if log_entry:
                        logs.append(log_entry)
        
        except Exception as e:
            if self.verbose:
                self.logger.error(f"Error parsing log file {log_file}: {e}")
        
        return logs
    
    def _parse_log_line(self, line: str, line_num: int) -> Optional[Dict[str, Any]]:
        """Parse a single log line for metrics and information"""
        import re
        
        entry = {
            "line_number": line_num,
            "timestamp": None,
            "level": "INFO",
            "message": line,
            "metrics": {},
            "epoch": None,
            "step": None,
        }
        
        # Common patterns
        patterns = [
            # Epoch pattern: "Epoch 5/10"
            r"Epoch\s+(\d+)(?:/(\d+))?",
            # Step pattern: "Step 1000"
            r"Step\s+(\d+)",
            # Loss pattern: "loss: 0.1234"
            r"loss:\s*([\d.]+)",
            # Accuracy pattern: "acc: 0.95"
            r"acc(?:uracy)?:\s*([\d.]+)",
            # Learning rate: "lr: 1e-4"
            r"lr:\s*([\d.e-]+)",
            # Time pattern: "time: 1.23s"
            r"time:\s*([\d.]+)s",
        ]
        
        # Extract epoch
        epoch_match = re.search(r"Epoch\s+(\d+)", line, re.IGNORECASE)
        if epoch_match:
            entry["epoch"] = int(epoch_match.group(1))
        
        # Extract step
        step_match = re.search(r"Step\s+(\d+)", line, re.IGNORECASE)
        if step_match:
            entry["step"] = int(step_match.group(1))
        
        # Extract metrics
        metric_patterns = {
            "loss": r"loss:\s*([\d.e-]+)",
            "accuracy": r"acc(?:uracy)?:\s*([\d.]+)",
            "learning_rate": r"lr:\s*([\d.e-]+)",
            "time": r"time:\s*([\d.]+)s",
        }
        
        for metric_name, pattern in metric_patterns.items():
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                try:
                    entry["metrics"][metric_name] = float(match.group(1))
                except ValueError:
                    pass
        
        # Only return if we found something useful
        if entry["epoch"] is not None or entry["step"] is not None or entry["metrics"]:
            return entry
        
        return None
    
    def _find_checkpoints(self, checkpoints_dir: Path) -> List[Dict[str, Any]]:
        """Find and analyze checkpoint files"""
        checkpoints = []
        
        for file_path in checkpoints_dir.rglob("*"):
            if file_path.is_file() and any(ext in file_path.suffix for ext in ['.pt', '.pth', '.ckpt', '.h5']):
                checkpoint_info = {
                    "filename": file_path.name,
                    "path": str(file_path),
                    "size_bytes": file_path.stat().st_size,
                    "size_mb": file_path.stat().st_size / (1024 * 1024),
                    "modified_time": file_path.stat().st_mtime,
                    "epoch": self._extract_epoch_from_filename(file_path.name),
                }
                checkpoints.append(checkpoint_info)
        
        # Sort by epoch or modification time
        checkpoints.sort(key=lambda x: x["epoch"] if x["epoch"] is not None else x["modified_time"])
        
        return checkpoints
    
    def _extract_epoch_from_filename(self, filename: str) -> Optional[int]:
        """Extract epoch number from checkpoint filename"""
        import re
        patterns = [
            r"epoch[-_]?(\d+)",
            r"ep(\d+)",
            r"(\d+)\.pt",
            r"(\d+)\.pth",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename, re.IGNORECASE)
            if match:
                try:
                    return int(match.group(1))
                except ValueError:
                    pass
        
        return None
    
    def _extract_run_metadata(self, run_dir: Path) -> Dict[str, Any]:
        """Extract additional metadata about the run"""
        metadata = {
            "total_files": 0,
            "total_size_bytes": 0,
            "start_time": None,
            "end_time": None,
            "duration_seconds": None,
        }
        
        # Count files and calculate total size
        for file_path in run_dir.rglob("*"):
            if file_path.is_file():
                metadata["total_files"] += 1
                metadata["total_size_bytes"] += file_path.stat().st_size
        
        metadata["total_size_mb"] = metadata["total_size_bytes"] / (1024 * 1024)
        
        # Find earliest and latest file times
        file_times = []
        for file_path in run_dir.rglob("*"):
            if file_path.is_file():
                file_times.append(file_path.stat().st_mtime)
        
        if file_times:
            metadata["start_time"] = min(file_times)
            metadata["end_time"] = max(file_times)
            metadata["duration_seconds"] = metadata["end_time"] - metadata["start_time"]
        
        return metadata
    
    def _extract_best_metrics(self, history: Dict[str, Any]) -> Dict[str, Any]:
        """Extract best metrics from training history"""
        best_metrics = {}
        
        for phase in ["train", "val", "test"]:
            if phase in history:
                phase_data = history[phase]
                best_metrics[phase] = {}
                
                for metric_name, values in phase_data.items():
                    if isinstance(values, list) and values:
                        if "loss" in metric_name.lower():
                            # For loss, best is minimum
                            best_metrics[phase][f"best_{metric_name}"] = min(values)
                        else:
                            # For other metrics, best is maximum
                            best_metrics[phase][f"best_{metric_name}"] = max(values)
        
        return best_metrics
    
    def _extract_total_epochs(self, history: Dict[str, Any]) -> Optional[int]:
        """Extract total number of epochs from history"""
        max_epochs = 0
        
        for phase_data in history.values():
            if isinstance(phase_data, dict):
                for values in phase_data.values():
                    if isinstance(values, list):
                        max_epochs = max(max_epochs, len(values))
        
        return max_epochs if max_epochs > 0 else None
    
    def _extract_metric_names(self, history: Dict[str, Any]) -> List[str]:
        """Extract all metric names from history"""
        metric_names = set()
        
        for phase_data in history.values():
            if isinstance(phase_data, dict):
                metric_names.update(phase_data.keys())
        
        return sorted(list(metric_names))
    
    def _compare_configs(self, config1: Dict[str, Any], config2: Dict[str, Any]) -> Dict[str, Any]:
        """Compare configuration dictionaries"""
        comparison = {
            "identical": config1 == config2,
            "differences": [],
            "added_keys": [],
            "removed_keys": [],
            "changed_values": [],
            "summary": {}
        }
        
        # Find all keys
        keys1 = set(self._flatten_dict(config1).keys()) if config1 else set()
        keys2 = set(self._flatten_dict(config2).keys()) if config2 else set()
        
        comparison["added_keys"] = list(keys2 - keys1)
        comparison["removed_keys"] = list(keys1 - keys2)
        
        # Compare common keys
        flat1 = self._flatten_dict(config1) if config1 else {}
        flat2 = self._flatten_dict(config2) if config2 else {}
        
        for key in keys1 & keys2:
            if flat1[key] != flat2[key]:
                comparison["changed_values"].append({
                    "key": key,
                    "old_value": flat1[key],
                    "new_value": flat2[key],
                })
        
        comparison["summary"] = {
            "total_changes": len(comparison["added_keys"]) + len(comparison["removed_keys"]) + len(comparison["changed_values"]),
            "config_similarity": 1.0 - (len(comparison["changed_values"]) / max(len(keys1 | keys2), 1)),
        }
        
        return comparison
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
        """Flatten nested dictionary"""
        items = []
        
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep).items())
            else:
                items.append((new_key, v))
        
        return dict(items)
    
    def _compare_metrics(
        self, 
        metrics1: Dict[str, Any], 
        metrics2: Dict[str, Any], 
        focus_metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Compare metrics between runs"""
        comparison = {
            "final_metrics_comparison": {},
            "best_metrics_comparison": {},
            "training_progress_comparison": {},
            "significant_changes": [],
            "improvements": [],
            "regressions": [],
        }
        
        # Compare final metrics
        final1 = metrics1.get("final_metrics", {})
        final2 = metrics2.get("final_metrics", {})
        comparison["final_metrics_comparison"] = self._compare_metric_values(final1, final2, "final")
        
        # Compare best metrics
        best1 = metrics1.get("best_metrics", {})
        best2 = metrics2.get("best_metrics", {})
        comparison["best_metrics_comparison"] = self._compare_metric_values(best1, best2, "best")
        
        # Identify significant changes
        all_comparisons = [comparison["final_metrics_comparison"], comparison["best_metrics_comparison"]]
        
        for comp in all_comparisons:
            for change in comp.get("metric_changes", []):
                if abs(change.get("percentage_change", 0)) > self.significant_change_threshold * 100:
                    comparison["significant_changes"].append(change)
                    
                    if change.get("percentage_change", 0) > 0:
                        if "loss" in change["metric_name"].lower():
                            comparison["regressions"].append(change)
                        else:
                            comparison["improvements"].append(change)
                    else:
                        if "loss" in change["metric_name"].lower():
                            comparison["improvements"].append(change)
                        else:
                            comparison["regressions"].append(change)
        
        return comparison
    
    def _compare_metric_values(self, metrics1: Dict[str, Any], metrics2: Dict[str, Any], category: str) -> Dict[str, Any]:
        """Compare two sets of metric values"""
        flat1 = self._flatten_dict(metrics1) if metrics1 else {}
        flat2 = self._flatten_dict(metrics2) if metrics2 else {}
        
        all_metrics = set(flat1.keys()) | set(flat2.keys())
        
        comparison = {
            "category": category,
            "total_metrics": len(all_metrics),
            "common_metrics": len(set(flat1.keys()) & set(flat2.keys())),
            "metric_changes": [],
            "new_metrics": list(set(flat2.keys()) - set(flat1.keys())),
            "removed_metrics": list(set(flat1.keys()) - set(flat2.keys())),
        }
        
        for metric_name in all_metrics:
            value1 = flat1.get(metric_name)
            value2 = flat2.get(metric_name)
            
            change_info = {
                "metric_name": metric_name,
                "value1": value1,
                "value2": value2,
                "absolute_change": None,
                "percentage_change": None,
                "direction": "unchanged",
            }
            
            if value1 is not None and value2 is not None:
                try:
                    val1 = float(value1)
                    val2 = float(value2)
                    
                    change_info["absolute_change"] = val2 - val1
                    
                    if val1 != 0:
                        change_info["percentage_change"] = (val2 - val1) / val1 * 100
                    
                    if val2 > val1:
                        change_info["direction"] = "increased"
                    elif val2 < val1:
                        change_info["direction"] = "decreased"
                    
                except (ValueError, TypeError):
                    pass
            
            comparison["metric_changes"].append(change_info)
        
        return comparison
    
    def _compare_training_progress(self, run1: Dict[str, Any], run2: Dict[str, Any]) -> Dict[str, Any]:
        """Compare training progress and convergence"""
        history1 = run1["metrics"].get("training_history", {})
        history2 = run2["metrics"].get("training_history", {})
        
        comparison = {
            "epochs_run1": run1["metrics"]["metadata"].get("total_epochs", 0),
            "epochs_run2": run2["metrics"]["metadata"].get("total_epochs", 0),
            "convergence_analysis": {},
            "stability_analysis": {},
            "efficiency_analysis": {},
        }
        
        # Convergence analysis
        comparison["convergence_analysis"] = self._analyze_convergence(history1, history2)
        
        # Stability analysis
        comparison["stability_analysis"] = self._analyze_stability(history1, history2)
        
        # Training efficiency
        comparison["efficiency_analysis"] = self._analyze_training_efficiency(run1, run2)
        
        return comparison
    
    def _analyze_convergence(self, history1: Dict[str, Any], history2: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze convergence patterns"""
        analysis = {
            "convergence_speed": {},
            "final_convergence": {},
            "early_stopping": {},
        }
        
        # Analyze loss convergence for each phase
        for phase in ["train", "val"]:
            if phase in history1 and phase in history2:
                phase_data1 = history1[phase]
                phase_data2 = history2[phase]
                
                for metric in ["loss"]:
                    if metric in phase_data1 and metric in phase_data2:
                        values1 = phase_data1[metric]
                        values2 = phase_data2[metric]
                        
                        if values1 and values2:
                            # Calculate convergence speed (epochs to reach 90% of final improvement)
                            conv_speed1 = self._calculate_convergence_speed(values1)
                            conv_speed2 = self._calculate_convergence_speed(values2)
                            
                            analysis["convergence_speed"][f"{phase}_{metric}"] = {
                                "run1_epochs": conv_speed1,
                                "run2_epochs": conv_speed2,
                                "speed_change": conv_speed2 - conv_speed1 if conv_speed1 and conv_speed2 else None,
                            }
        
        return analysis
    
    def _calculate_convergence_speed(self, values: List[float]) -> Optional[int]:
        """Calculate epochs needed to reach 90% of final improvement"""
        if len(values) < 3:
            return None
        
        initial_val = values[0]
        final_val = values[-1]
        target_val = initial_val + 0.9 * (final_val - initial_val)
        
        for i, val in enumerate(values):
            if (final_val < initial_val and val <= target_val) or (final_val > initial_val and val >= target_val):
                return i + 1
        
        return len(values)
    
    def _analyze_stability(self, history1: Dict[str, Any], history2: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze training stability"""
        analysis = {
            "variance_comparison": {},
            "oscillation_analysis": {},
        }
        
        for phase in ["train", "val"]:
            if phase in history1 and phase in history2:
                phase_data1 = history1[phase]
                phase_data2 = history2[phase]
                
                for metric in phase_data1.keys() & phase_data2.keys():
                    values1 = phase_data1[metric]
                    values2 = phase_data2[metric]
                    
                    if values1 and values2 and len(values1) > 1 and len(values2) > 1:
                        var1 = statistics.variance(values1)
                        var2 = statistics.variance(values2)
                        
                        analysis["variance_comparison"][f"{phase}_{metric}"] = {
                            "variance_run1": var1,
                            "variance_run2": var2,
                            "stability_change": "more_stable" if var2 < var1 else "less_stable" if var2 > var1 else "similar",
                            "variance_ratio": var2 / var1 if var1 > 0 else None,
                        }
        
        return analysis
    
    def _analyze_training_efficiency(self, run1: Dict[str, Any], run2: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze training efficiency metrics"""
        analysis = {
            "time_efficiency": {},
            "epoch_efficiency": {},
        }
        
        # Time efficiency
        duration1 = run1["metadata"].get("duration_seconds")
        duration2 = run2["metadata"].get("duration_seconds")
        
        if duration1 and duration2:
            analysis["time_efficiency"] = {
                "duration_run1_hours": duration1 / 3600,
                "duration_run2_hours": duration2 / 3600,
                "time_change_percent": (duration2 - duration1) / duration1 * 100,
                "efficiency_change": "faster" if duration2 < duration1 else "slower" if duration2 > duration1 else "similar",
            }
        
        # Epochs efficiency
        epochs1 = run1["metrics"]["metadata"].get("total_epochs", 0)
        epochs2 = run2["metrics"]["metadata"].get("total_epochs", 0)
        
        if epochs1 and epochs2:
            analysis["epoch_efficiency"] = {
                "epochs_run1": epochs1,
                "epochs_run2": epochs2,
                "epoch_change": epochs2 - epochs1,
                "convergence_efficiency": "better" if epochs2 < epochs1 else "worse" if epochs2 > epochs1 else "similar",
            }
        
        return analysis
    
    def _compare_performance(self, run1: Dict[str, Any], run2: Dict[str, Any]) -> Dict[str, Any]:
        """Compare overall performance between runs"""
        return {
            "resource_usage": self._compare_resource_usage(run1, run2),
            "model_performance": self._compare_model_performance(run1, run2),
            "efficiency_metrics": self._compare_efficiency_metrics(run1, run2),
        }
    
    def _compare_resource_usage(self, run1: Dict[str, Any], run2: Dict[str, Any]) -> Dict[str, Any]:
        """Compare resource usage between runs"""
        comparison = {
            "storage_usage": {},
            "checkpoint_efficiency": {},
        }
        
        # Storage comparison
        size1 = run1["metadata"].get("total_size_mb", 0)
        size2 = run2["metadata"].get("total_size_mb", 0)
        
        comparison["storage_usage"] = {
            "total_size_run1_mb": size1,
            "total_size_run2_mb": size2,
            "size_change_mb": size2 - size1,
            "size_change_percent": (size2 - size1) / size1 * 100 if size1 > 0 else 0,
        }
        
        # Checkpoint efficiency
        ckpt1 = run1.get("checkpoints", [])
        ckpt2 = run2.get("checkpoints", [])
        
        if ckpt1 and ckpt2:
            avg_size1 = statistics.mean([c["size_mb"] for c in ckpt1])
            avg_size2 = statistics.mean([c["size_mb"] for c in ckpt2])
            
            comparison["checkpoint_efficiency"] = {
                "checkpoint_count_run1": len(ckpt1),
                "checkpoint_count_run2": len(ckpt2),
                "avg_checkpoint_size_run1_mb": avg_size1,
                "avg_checkpoint_size_run2_mb": avg_size2,
                "checkpoint_size_change_percent": (avg_size2 - avg_size1) / avg_size1 * 100 if avg_size1 > 0 else 0,
            }
        
        return comparison
    
    def _compare_model_performance(self, run1: Dict[str, Any], run2: Dict[str, Any]) -> Dict[str, Any]:
        """Compare final model performance"""
        best1 = run1["metrics"].get("best_metrics", {})
        best2 = run2["metrics"].get("best_metrics", {})
        
        return self._compare_metric_values(best1, best2, "model_performance")
    
    def _compare_efficiency_metrics(self, run1: Dict[str, Any], run2: Dict[str, Any]) -> Dict[str, Any]:
        """Compare training efficiency metrics"""
        duration1 = run1["metadata"].get("duration_seconds", 1)
        duration2 = run2["metadata"].get("duration_seconds", 1)
        epochs1 = run1["metrics"]["metadata"].get("total_epochs", 1)
        epochs2 = run2["metrics"]["metadata"].get("total_epochs", 1)
        
        return {
            "time_per_epoch_run1": duration1 / epochs1,
            "time_per_epoch_run2": duration2 / epochs2,
            "time_efficiency_change_percent": ((duration2 / epochs2) - (duration1 / epochs1)) / (duration1 / epochs1) * 100,
        }
    
    def _compare_checkpoints(self, checkpoints1: List[Dict[str, Any]], checkpoints2: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare checkpoint information"""
        return {
            "checkpoint_count_run1": len(checkpoints1),
            "checkpoint_count_run2": len(checkpoints2),
            "checkpoint_count_change": len(checkpoints2) - len(checkpoints1),
            "total_checkpoint_size_run1_mb": sum(c["size_mb"] for c in checkpoints1),
            "total_checkpoint_size_run2_mb": sum(c["size_mb"] for c in checkpoints2),
            "avg_checkpoint_size_run1_mb": statistics.mean([c["size_mb"] for c in checkpoints1]) if checkpoints1 else 0,
            "avg_checkpoint_size_run2_mb": statistics.mean([c["size_mb"] for c in checkpoints2]) if checkpoints2 else 0,
        }
    
    def _analyze_regressions(self, comparison: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze potential regressions"""
        regressions = comparison["metrics_comparison"].get("regressions", [])
        
        analysis = {
            "total_regressions": len(regressions),
            "severe_regressions": [r for r in regressions if abs(r.get("percentage_change", 0)) > 10],
            "regression_categories": defaultdict(int),
            "most_concerning": None,
        }
        
        for regression in regressions:
            metric_name = regression["metric_name"].lower()
            if "accuracy" in metric_name or "f1" in metric_name:
                analysis["regression_categories"]["performance"] += 1
            elif "loss" in metric_name:
                analysis["regression_categories"]["loss"] += 1
            else:
                analysis["regression_categories"]["other"] += 1
        
        if analysis["severe_regressions"]:
            analysis["most_concerning"] = max(
                analysis["severe_regressions"],
                key=lambda x: abs(x.get("percentage_change", 0))
            )
        
        return analysis
    
    def _analyze_improvements(self, comparison: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze improvements"""
        improvements = comparison["metrics_comparison"].get("improvements", [])
        
        analysis = {
            "total_improvements": len(improvements),
            "significant_improvements": [i for i in improvements if abs(i.get("percentage_change", 0)) > 10],
            "improvement_categories": defaultdict(int),
            "best_improvement": None,
        }
        
        for improvement in improvements:
            metric_name = improvement["metric_name"].lower()
            if "accuracy" in metric_name or "f1" in metric_name:
                analysis["improvement_categories"]["performance"] += 1
            elif "loss" in metric_name:
                analysis["improvement_categories"]["loss"] += 1
            else:
                analysis["improvement_categories"]["other"] += 1
        
        if analysis["significant_improvements"]:
            analysis["best_improvement"] = max(
                analysis["significant_improvements"],
                key=lambda x: abs(x.get("percentage_change", 0))
            )
        
        return analysis
    
    def _generate_comparison_summary(self, comparison: Dict[str, Any]) -> Dict[str, Any]:
        """Generate high-level comparison summary"""
        regressions = comparison["regression_analysis"]["total_regressions"]
        improvements = comparison["improvement_analysis"]["total_improvements"]
        
        summary = {
            "overall_assessment": "unknown",
            "net_change": "neutral",
            "confidence": "medium",
            "key_findings": [],
            "metrics_summary": {
                "total_regressions": regressions,
                "total_improvements": improvements,
                "net_score": improvements - regressions,
            }
        }
        
        # Overall assessment
        if improvements > regressions * 2:
            summary["overall_assessment"] = "significant_improvement"
            summary["net_change"] = "positive"
        elif regressions > improvements * 2:
            summary["overall_assessment"] = "concerning_regression"
            summary["net_change"] = "negative"
        elif improvements > regressions:
            summary["overall_assessment"] = "mild_improvement"
            summary["net_change"] = "slightly_positive"
        elif regressions > improvements:
            summary["overall_assessment"] = "mild_regression"
            summary["net_change"] = "slightly_negative"
        else:
            summary["overall_assessment"] = "mixed_results"
            summary["net_change"] = "neutral"
        
        # Key findings
        if comparison["regression_analysis"]["severe_regressions"]:
            summary["key_findings"].append("Severe performance regressions detected")
        
        if comparison["improvement_analysis"]["significant_improvements"]:
            summary["key_findings"].append("Significant performance improvements found")
        
        config_changes = comparison["config_comparison"]["summary"]["total_changes"]
        if config_changes > 0:
            summary["key_findings"].append(f"{config_changes} configuration changes detected")
        
        return summary
    
    def _generate_recommendations(self, comparison: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        summary = comparison["summary"]
        regressions = comparison["regression_analysis"]
        improvements = comparison["improvement_analysis"]
        
        # Regression-based recommendations
        if regressions["severe_regressions"]:
            recommendations.append(
                "Critical: Severe performance regressions detected. "
                "Consider reverting changes or investigating root causes immediately."
            )
        
        if regressions["most_concerning"]:
            metric = regressions["most_concerning"]["metric_name"]
            change = regressions["most_concerning"]["percentage_change"]
            recommendations.append(
                f"Investigate regression in {metric} ({change:.1f}% worse). "
                "This may indicate a fundamental issue with the changes."
            )
        
        # Improvement-based recommendations
        if improvements["significant_improvements"]:
            recommendations.append(
                "Performance improvements detected. Document successful changes "
                "and consider applying similar optimizations to other areas."
            )
        
        # Configuration-based recommendations
        config_changes = comparison["config_comparison"]["summary"]["total_changes"]
        if config_changes > 10:
            recommendations.append(
                f"Many configuration changes ({config_changes}) detected. "
                "Consider isolating changes to identify which ones drive performance differences."
            )
        
        # Training efficiency recommendations
        training_comp = comparison["training_comparison"]
        efficiency = training_comp.get("efficiency_analysis", {})
        
        if efficiency.get("time_efficiency", {}).get("efficiency_change") == "slower":
            recommendations.append(
                "Training time increased. Check for computational inefficiencies "
                "or consider optimizing hyperparameters."
            )
        
        return recommendations
    
    def _save_comparisons(self) -> None:
        """Save all comparison results"""
        if not self.comparisons:
            return
        
        # Save detailed comparisons
        self.save_json(self.comparisons, "training_run_comparisons.json")
        
        # Save run data
        self.save_json(self.run_data, "loaded_run_data.json")
        
        # Create summary report
        summary_report = {
            "total_comparisons": len(self.comparisons),
            "runs_analyzed": len(self.run_data),
            "comparison_summary": self._generate_overall_summary(),
        }
        
        self.save_json(summary_report, "training_differ_summary.json")
        
        if self.verbose:
            self.logger.info(f"Saved {len(self.comparisons)} training comparisons to {self.output_dir}")
    
    def _generate_overall_summary(self) -> Dict[str, Any]:
        """Generate overall summary across all comparisons"""
        if not self.comparisons:
            return {}
        
        total_improvements = sum(c["improvement_analysis"]["total_improvements"] for c in self.comparisons)
        total_regressions = sum(c["regression_analysis"]["total_regressions"] for c in self.comparisons)
        
        return {
            "total_improvements": total_improvements,
            "total_regressions": total_regressions,
            "net_improvements": total_improvements - total_regressions,
            "improvement_rate": total_improvements / (total_improvements + total_regressions) if (total_improvements + total_regressions) > 0 else 0,
            "comparison_outcomes": [c["summary"]["overall_assessment"] for c in self.comparisons],
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get differ statistics"""
        return {
            "loaded_runs": len(self.run_data),
            "completed_comparisons": len(self.comparisons),
            "run_ids": list(self.run_data.keys()),
            "recent_comparisons": [c["comparison_name"] for c in self.comparisons[-3:]],
        }