"""
Loss curve logger for tracking training progress
"""

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..core.base import BaseDebugTool


class LossLogger(BaseDebugTool):
    """Log and track loss curves and training metrics"""
    
    def __init__(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        verbose: bool = True,
        auto_save_interval: int = 10,  # Save every N epochs
        plot_realtime: bool = False,
        save_plots: bool = True,
    ):
        super().__init__(output_dir, verbose)
        self.auto_save_interval = auto_save_interval
        self.plot_realtime = plot_realtime
        self.save_plots = save_plots
        
        self.metrics_history: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        self.epoch_times: List[float] = []
        self.current_epoch = 0
        self.training_start_time = None
        self.epoch_start_time = None
        
        # Plotting setup
        if self.plot_realtime:
            plt.ion()
            self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 10))
            self.fig.suptitle("Training Progress")
    
    def enable(self) -> None:
        """Enable loss logger"""
        self.enabled = True
        self.training_start_time = time.time()
        if self.verbose:
            self.logger.info("Loss logger enabled")
    
    def disable(self) -> None:
        """Disable loss logger and save final results"""
        self.enabled = False
        self._save_all_data()
        if self.plot_realtime:
            plt.ioff()
            plt.close(self.fig)
        if self.verbose:
            self.logger.info("Loss logger disabled and data saved")
    
    def start_epoch(self, epoch: int) -> None:
        """Mark the start of an epoch"""
        if not self.enabled:
            return
        
        self.current_epoch = epoch
        self.epoch_start_time = time.time()
        
        if self.verbose and epoch % 10 == 0:  # Log every 10 epochs
            self.logger.info(f"Starting epoch {epoch}")
    
    def end_epoch(self) -> None:
        """Mark the end of an epoch"""
        if not self.enabled or self.epoch_start_time is None:
            return
        
        epoch_duration = time.time() - self.epoch_start_time
        self.epoch_times.append(epoch_duration)
        
        # Auto-save periodically
        if self.current_epoch % self.auto_save_interval == 0:
            self._save_current_state()
        
        # Update plots if real-time plotting is enabled
        if self.plot_realtime:
            self._update_plots()
        
        self.epoch_start_time = None
    
    def log_metric(
        self, 
        metric_name: str, 
        value: float, 
        phase: str = "train",
        step: Optional[int] = None
    ) -> None:
        """Log a single metric value"""
        if not self.enabled:
            return
        
        if step is None:
            step = self.current_epoch
        
        self.metrics_history[phase][metric_name].append(value)
        
        # Also log the step/epoch for this metric
        step_key = f"{metric_name}_steps"
        if len(self.metrics_history[phase][step_key]) < len(self.metrics_history[phase][metric_name]):
            self.metrics_history[phase][step_key].append(step)
    
    def log_metrics(
        self, 
        metrics: Dict[str, float], 
        phase: str = "train",
        step: Optional[int] = None
    ) -> None:
        """Log multiple metrics at once"""
        for metric_name, value in metrics.items():
            self.log_metric(metric_name, value, phase, step)
    
    def log_lr(self, learning_rate: float, step: Optional[int] = None) -> None:
        """Log learning rate"""
        self.log_metric("learning_rate", learning_rate, "optimizer", step)
    
    def log_batch_metrics(
        self,
        metrics: Dict[str, float],
        batch_idx: int,
        phase: str = "train"
    ) -> None:
        """Log metrics for a specific batch"""
        batch_phase = f"{phase}_batch"
        for metric_name, value in metrics.items():
            batch_key = f"{metric_name}_batch"
            self.metrics_history[batch_phase][batch_key].append(value)
            self.metrics_history[batch_phase][f"{batch_key}_indices"].append(batch_idx)
    
    def get_current_metrics(self, phase: str = "train") -> Dict[str, float]:
        """Get the most recent metrics for a phase"""
        current_metrics = {}
        
        for metric_name, values in self.metrics_history[phase].items():
            if values and not metric_name.endswith("_steps"):
                current_metrics[metric_name] = values[-1]
        
        return current_metrics
    
    def get_metric_history(self, metric_name: str, phase: str = "train") -> List[float]:
        """Get the full history of a specific metric"""
        return self.metrics_history[phase].get(metric_name, [])
    
    def get_best_metric(self, metric_name: str, phase: str = "train", mode: str = "min") -> Dict[str, Any]:
        """Get the best value of a metric"""
        values = self.get_metric_history(metric_name, phase)
        
        if not values:
            return {"value": None, "epoch": None, "step": None}
        
        if mode == "min":
            best_idx = np.argmin(values)
            best_value = min(values)
        else:  # mode == "max"
            best_idx = np.argmax(values)
            best_value = max(values)
        
        # Get corresponding step/epoch
        step_key = f"{metric_name}_steps"
        steps = self.metrics_history[phase].get(step_key, list(range(len(values))))
        best_step = steps[best_idx] if best_idx < len(steps) else best_idx
        
        return {
            "value": best_value,
            "epoch": best_step,
            "step": best_step,
            "index": best_idx
        }
    
    def calculate_smoothed_metrics(
        self, 
        metric_name: str, 
        phase: str = "train", 
        window_size: int = 5
    ) -> List[float]:
        """Calculate smoothed version of metrics using moving average"""
        values = self.get_metric_history(metric_name, phase)
        
        if len(values) < window_size:
            return values
        
        smoothed = []
        for i in range(len(values)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(values), i + window_size // 2 + 1)  
            window_values = values[start_idx:end_idx]
            smoothed.append(np.mean(window_values))
        
        return smoothed
    
    def detect_plateaus(
        self, 
        metric_name: str, 
        phase: str = "train",
        patience: int = 10,
        min_delta: float = 1e-4
    ) -> List[Dict[str, Any]]:
        """Detect plateaus in metric curves"""
        values = self.get_metric_history(metric_name, phase)
        
        if len(values) < patience * 2:
            return []
        
        plateaus = []
        current_plateau_start = None
        
        for i in range(patience, len(values)):
            # Check if the last 'patience' values show no significant improvement
            recent_values = values[i-patience:i]
            improvement = abs(max(recent_values) - min(recent_values))
            
            if improvement < min_delta:
                if current_plateau_start is None:
                    current_plateau_start = i - patience
            else:
                if current_plateau_start is not None:
                    plateaus.append({
                        "start_epoch": current_plateau_start,
                        "end_epoch": i - 1,
                        "duration": i - 1 - current_plateau_start,
                        "value_range": improvement,
                        "avg_value": np.mean(values[current_plateau_start:i])
                    })
                    current_plateau_start = None
        
        # Handle plateau that extends to the end
        if current_plateau_start is not None:
            plateaus.append({
                "start_epoch": current_plateau_start,
                "end_epoch": len(values) - 1,
                "duration": len(values) - 1 - current_plateau_start,
                "value_range": abs(max(values[current_plateau_start:]) - min(values[current_plateau_start:])),
                "avg_value": np.mean(values[current_plateau_start:])
            })
        
        return plateaus
    
    def detect_overfitting(
        self,
        train_metric: str = "loss",
        val_metric: str = "loss",
        patience: int = 5,
        threshold: float = 0.1
    ) -> Dict[str, Any]:
        """Detect potential overfitting by comparing train/val metrics"""
        train_values = self.get_metric_history(train_metric, "train")
        val_values = self.get_metric_history(val_metric, "val")
        
        if len(train_values) < patience or len(val_values) < patience:
            return {"overfitting_detected": False, "reason": "Insufficient data"}
        
        # Align lengths
        min_length = min(len(train_values), len(val_values))
        train_values = train_values[:min_length]
        val_values = val_values[:min_length]
        
        # Look for divergence: train continues to improve while val gets worse
        overfitting_start = None
        
        for i in range(patience, len(train_values)):
            train_trend = np.mean(train_values[i-patience:i]) - np.mean(train_values[i-patience*2:i-patience])
            val_trend = np.mean(val_values[i-patience:i]) - np.mean(val_values[i-patience*2:i-patience])
            
            # For loss: train_trend should be negative (improving), val_trend positive (getting worse)
            # For accuracy: train_trend should be positive, val_trend negative
            is_loss_metric = "loss" in train_metric.lower()
            
            if is_loss_metric:
                diverging = train_trend < -threshold and val_trend > threshold
            else:
                diverging = train_trend > threshold and val_trend < -threshold
            
            if diverging and overfitting_start is None:
                overfitting_start = i - patience
                break
        
        result = {
            "overfitting_detected": overfitting_start is not None,
            "overfitting_start_epoch": overfitting_start,
            "train_final": train_values[-1] if train_values else None,
            "val_final": val_values[-1] if val_values else None,
            "gap": abs(train_values[-1] - val_values[-1]) if train_values and val_values else None,
        }
        
        if overfitting_start:
            result["recommendation"] = f"Consider early stopping around epoch {overfitting_start}"
        
        return result
    
    def _update_plots(self) -> None:
        """Update real-time plots"""
        if not self.plot_realtime or not hasattr(self, 'axes'):
            return
        
        # Clear previous plots
        for ax in self.axes.flat:
            ax.clear()
        
        # Plot 1: Loss curves
        ax1 = self.axes[0, 0]
        for phase in ["train", "val"]:
            if "loss" in self.metrics_history[phase]:
                loss_values = self.metrics_history[phase]["loss"]
                ax1.plot(loss_values, label=f"{phase}_loss")
        ax1.set_title("Loss Curves")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: Accuracy/metric curves
        ax2 = self.axes[0, 1]
        for phase in ["train", "val"]:
            for metric_name in self.metrics_history[phase]:
                if "acc" in metric_name.lower() and not metric_name.endswith("_steps"):
                    values = self.metrics_history[phase][metric_name]
                    ax2.plot(values, label=f"{phase}_{metric_name}")
        ax2.set_title("Accuracy/Metrics")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Value")
        ax2.legend()
        ax2.grid(True)
        
        # Plot 3: Learning rate
        ax3 = self.axes[1, 0]
        if "learning_rate" in self.metrics_history.get("optimizer", {}):
            lr_values = self.metrics_history["optimizer"]["learning_rate"]
            ax3.plot(lr_values, color='red')
            ax3.set_title("Learning Rate")
            ax3.set_xlabel("Epoch")
            ax3.set_ylabel("Learning Rate")
            ax3.set_yscale('log')
            ax3.grid(True)
        
        # Plot 4: Training time per epoch
        ax4 = self.axes[1, 1]
        if self.epoch_times:
            ax4.plot(self.epoch_times, color='green')
            ax4.set_title("Epoch Duration")
            ax4.set_xlabel("Epoch")
            ax4.set_ylabel("Time (seconds)")
            ax4.grid(True)
        
        plt.tight_layout()
        plt.pause(0.01)
    
    def create_plots(self, save_path: Optional[Path] = None) -> Path:
        """Create comprehensive plots of training progress"""
        if save_path is None:
            save_path = self.output_dir / "training_plots.png"
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("Training Analysis Dashboard", fontsize=16)
        
        # Plot 1: Loss curves
        ax1 = axes[0, 0]
        for phase in ["train", "val", "test"]:
            if "loss" in self.metrics_history.get(phase, {}):
                loss_values = self.metrics_history[phase]["loss"]
                ax1.plot(loss_values, label=f"{phase}_loss", linewidth=2)
                
                # Add smoothed version
                smoothed = self.calculate_smoothed_metrics("loss", phase)
                ax1.plot(smoothed, label=f"{phase}_loss_smooth", linestyle='--', alpha=0.7)
        
        ax1.set_title("Loss Curves")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Accuracy curves
        ax2 = axes[0, 1]
        for phase in ["train", "val", "test"]:
            phase_metrics = self.metrics_history.get(phase, {})
            for metric_name in phase_metrics:
                if ("acc" in metric_name.lower() or "f1" in metric_name.lower()) and not metric_name.endswith("_steps"):
                    values = phase_metrics[metric_name]
                    ax2.plot(values, label=f"{phase}_{metric_name}", linewidth=2)
        
        ax2.set_title("Performance Metrics")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Score")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Learning rate schedule
        ax3 = axes[0, 2]
        if "learning_rate" in self.metrics_history.get("optimizer", {}):
            lr_values = self.metrics_history["optimizer"]["learning_rate"]
            ax3.plot(lr_values, color='red', linewidth=2)
            ax3.set_title("Learning Rate Schedule")
            ax3.set_xlabel("Epoch")
            ax3.set_ylabel("Learning Rate")
            ax3.set_yscale('log')
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Training time analysis
        ax4 = axes[1, 0]
        if self.epoch_times:
            ax4.plot(self.epoch_times, color='green', linewidth=2)
            # Add moving average
            if len(self.epoch_times) > 5:
                moving_avg = pd.Series(self.epoch_times).rolling(window=5).mean()
                ax4.plot(moving_avg, color='darkgreen', linestyle='--', label='5-epoch avg')
                ax4.legend()
            ax4.set_title("Training Time per Epoch")
            ax4.set_xlabel("Epoch")
            ax4.set_ylabel("Time (seconds)")
            ax4.grid(True, alpha=0.3)
        
        # Plot 5: Loss distribution/histogram
        ax5 = axes[1, 1]
        if "loss" in self.metrics_history.get("train", {}):
            train_losses = self.metrics_history["train"]["loss"]
            if "loss" in self.metrics_history.get("val", {}):
                val_losses = self.metrics_history["val"]["loss"]
                ax5.hist([train_losses, val_losses], bins=20, alpha=0.7, label=['train', 'val'])
                ax5.legend()
            else:
                ax5.hist(train_losses, bins=20, alpha=0.7, color='blue')
            ax5.set_title("Loss Distribution")
            ax5.set_xlabel("Loss Value")
            ax5.set_ylabel("Frequency")
            ax5.grid(True, alpha=0.3)
        
        # Plot 6: Training summary
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        # Create summary text
        summary_text = self._generate_training_summary()
        ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        if self.verbose:
            self.logger.info(f"Training plots saved to {save_path}")
        
        return save_path
    
    def _generate_training_summary(self) -> str:
        """Generate text summary of training progress"""
        summary_lines = ["Training Summary", "=" * 15, ""]
        
        # Basic info
        total_epochs = self.current_epoch
        summary_lines.append(f"Total Epochs: {total_epochs}")
        
        if self.training_start_time:
            total_time = time.time() - self.training_start_time
            summary_lines.append(f"Total Time: {total_time/3600:.2f}h")
        
        if self.epoch_times:
            avg_epoch_time = np.mean(self.epoch_times)
            summary_lines.append(f"Avg Epoch Time: {avg_epoch_time:.2f}s")
        
        summary_lines.append("")
        
        # Best metrics
        for phase in ["train", "val", "test"]:
            if phase in self.metrics_history:
                summary_lines.append(f"{phase.capitalize()} Metrics:")
                
                # Loss
                if "loss" in self.metrics_history[phase]:
                    best_loss = self.get_best_metric("loss", phase, "min")
                    summary_lines.append(f"  Best Loss: {best_loss['value']:.4f} @ epoch {best_loss['epoch']}")
                
                # Accuracy
                for metric_name in self.metrics_history[phase]:
                    if "acc" in metric_name.lower() and not metric_name.endswith("_steps"):
                        best_acc = self.get_best_metric(metric_name, phase, "max")
                        summary_lines.append(f"  Best {metric_name}: {best_acc['value']:.4f} @ epoch {best_acc['epoch']}")
                
                summary_lines.append("")
        
        # Overfitting detection
        if "train" in self.metrics_history and "val" in self.metrics_history:
            overfitting = self.detect_overfitting()
            if overfitting["overfitting_detected"]:
                summary_lines.append("⚠ Overfitting detected!")
                summary_lines.append(f"  Started around epoch {overfitting['overfitting_start_epoch']}")
                summary_lines.append("")
        
        return "\n".join(summary_lines)
    
    def _save_current_state(self) -> None:
        """Save current training state"""
        timestamp = int(time.time())
        
        # Save metrics as JSON
        metrics_data = {
            "timestamp": timestamp,
            "current_epoch": self.current_epoch,
            "metrics_history": dict(self.metrics_history),
            "epoch_times": self.epoch_times,
            "training_duration": time.time() - self.training_start_time if self.training_start_time else None,
        }
        
        self.save_json(metrics_data, f"training_state_epoch_{self.current_epoch}.json")
        
        # Save as CSV for easy analysis
        self._save_metrics_csv()
        
        # Create plots if enabled
        if self.save_plots:
            self.create_plots()
    
    def _save_all_data(self) -> None:
        """Save all collected data"""
        # Final state save
        self._save_current_state()
        
        # Summary statistics
        summary = self._generate_summary_statistics()
        self.save_json(summary, "training_summary.json")
        
        # Analysis results
        analysis = self._generate_analysis_report()
        self.save_json(analysis, "training_analysis.json")
        
        if self.verbose:
            self.logger.info(f"All training data saved to {self.output_dir}")
    
    def _save_metrics_csv(self) -> None:
        """Save metrics in CSV format for easy analysis"""
        # Combine all metrics into a single DataFrame
        all_data = []
        
        for phase, phase_metrics in self.metrics_history.items():
            for metric_name, values in phase_metrics.items():
                if not metric_name.endswith("_steps") and not metric_name.endswith("_indices"):
                    # Get corresponding steps/epochs
                    steps_key = f"{metric_name}_steps"
                    steps = phase_metrics.get(steps_key, list(range(len(values))))
                    
                    for i, value in enumerate(values):
                        step = steps[i] if i < len(steps) else i
                        all_data.append({
                            "phase": phase,
                            "metric": metric_name,
                            "step": step,
                            "value": value
                        })
        
        if all_data:
            df = pd.DataFrame(all_data)
            csv_path = self.output_dir / "training_metrics.csv"
            df.to_csv(csv_path, index=False)
            
            if self.verbose:
                self.logger.info(f"Metrics saved to CSV: {csv_path}")
    
    def _generate_summary_statistics(self) -> Dict[str, Any]:
        """Generate summary statistics"""
        summary = {
            "training_info": {
                "total_epochs": self.current_epoch,
                "total_time_hours": (time.time() - self.training_start_time) / 3600 if self.training_start_time else None,
                "avg_epoch_time_seconds": np.mean(self.epoch_times) if self.epoch_times else None,
            },
            "phases": {},
            "best_metrics": {},
            "final_metrics": {},
        }
        
        for phase, phase_metrics in self.metrics_history.items():
            phase_summary = {
                "total_metrics": len([k for k in phase_metrics.keys() if not k.endswith("_steps")]),
                "metric_names": [k for k in phase_metrics.keys() if not k.endswith("_steps")],
            }
            
            # Calculate statistics for each metric
            for metric_name, values in phase_metrics.items():
                if not metric_name.endswith("_steps") and values:
                    best_info = self.get_best_metric(metric_name, phase, "min" if "loss" in metric_name.lower() else "max")
                    summary["best_metrics"][f"{phase}_{metric_name}"] = best_info
                    summary["final_metrics"][f"{phase}_{metric_name}"] = values[-1]
            
            summary["phases"][phase] = phase_summary
        
        return summary
    
    def _generate_analysis_report(self) -> Dict[str, Any]:
        """Generate comprehensive analysis report"""
        analysis = {
            "convergence_analysis": {},
            "stability_analysis": {},
            "overfitting_analysis": {},
            "plateau_detection": {},
            "recommendations": [],
        }
        
        # Convergence analysis
        for phase in ["train", "val"]:
            if "loss" in self.metrics_history.get(phase, {}):
                loss_values = self.metrics_history[phase]["loss"]
                if len(loss_values) > 5:
                    # Check if loss is converging (decreasing trend)
                    recent_trend = np.polyfit(range(len(loss_values[-10:])), loss_values[-10:], 1)[0]
                    analysis["convergence_analysis"][f"{phase}_loss_trend"] = {
                        "slope": recent_trend,
                        "converging": recent_trend < 0,
                        "final_loss": loss_values[-1],
                        "best_loss": min(loss_values),
                    }
        
        # Stability analysis
        for phase in ["train", "val"]:
            phase_metrics = self.metrics_history.get(phase, {})
            for metric_name, values in phase_metrics.items():
                if not metric_name.endswith("_steps") and len(values) > 10:
                    # Calculate coefficient of variation for last 20% of training
                    recent_values = values[-len(values)//5:]  # Last 20%
                    cv = np.std(recent_values) / np.mean(recent_values) if np.mean(recent_values) != 0 else float('inf')
                    
                    analysis["stability_analysis"][f"{phase}_{metric_name}"] = {
                        "coefficient_of_variation": cv,
                        "stability": "stable" if cv < 0.1 else "moderate" if cv < 0.3 else "unstable",
                        "recent_std": np.std(recent_values),
                        "recent_mean": np.mean(recent_values),
                    }
        
        # Overfitting analysis
        if "train" in self.metrics_history and "val" in self.metrics_history:
            overfitting_result = self.detect_overfitting()
            analysis["overfitting_analysis"] = overfitting_result
        
        # Plateau detection
        for phase in ["train", "val"]:
            if "loss" in self.metrics_history.get(phase, {}):
                plateaus = self.detect_plateaus("loss", phase)
                analysis["plateau_detection"][f"{phase}_loss"] = plateaus
        
        # Generate recommendations
        analysis["recommendations"] = self._generate_recommendations(analysis)
        
        return analysis
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate training recommendations based on analysis"""
        recommendations = []
        
        # Overfitting recommendations
        overfitting = analysis.get("overfitting_analysis", {})
        if overfitting.get("overfitting_detected"):
            recommendations.append(
                f"Overfitting detected around epoch {overfitting['overfitting_start_epoch']}. "
                "Consider early stopping, regularization, or reducing model complexity."
            )
        
        # Plateau recommendations
        plateau_detection = analysis.get("plateau_detection", {})
        for metric_key, plateaus in plateau_detection.items():
            if plateaus:
                longest_plateau = max(plateaus, key=lambda x: x["duration"])
                if longest_plateau["duration"] > 10:
                    recommendations.append(
                        f"Long plateau detected in {metric_key} "
                        f"(epochs {longest_plateau['start_epoch']}-{longest_plateau['end_epoch']}). "
                        "Consider adjusting learning rate or using learning rate scheduling."
                    )
        
        # Stability recommendations
        stability_analysis = analysis.get("stability_analysis", {})
        unstable_metrics = [k for k, v in stability_analysis.items() if v.get("stability") == "unstable"]
        if unstable_metrics:
            recommendations.append(
                f"Unstable training detected in: {', '.join(unstable_metrics)}. "
                "Consider reducing learning rate, increasing batch size, or adding gradient clipping."
            )
        
        # Convergence recommendations
        convergence_analysis = analysis.get("convergence_analysis", {})
        for metric_key, conv_info in convergence_analysis.items():
            if not conv_info.get("converging"):
                recommendations.append(
                    f"No convergence detected in {metric_key}. "
                    "Training may need to run longer or hyperparameters may need adjustment."
                )
        
        return recommendations
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current logging statistics"""
        return {
            "current_epoch": self.current_epoch,
            "phases_tracked": list(self.metrics_history.keys()),
            "metrics_per_phase": {
                phase: len([k for k in metrics.keys() if not k.endswith("_steps")])
                for phase, metrics in self.metrics_history.items()
            },
            "total_datapoints": sum(
                len(values) for phase_metrics in self.metrics_history.values()
                for key, values in phase_metrics.items()
                if not key.endswith("_steps")
            ),
            "training_duration_hours": (time.time() - self.training_start_time) / 3600 if self.training_start_time else None,
        }