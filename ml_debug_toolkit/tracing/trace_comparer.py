"""
Compare traces across different runs to detect performance changes and kernel differences
"""

import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from ..core.base import BaseDebugTool
from .trace_parser import TraceParser


class TraceComparer(BaseDebugTool):
    """Compare traces to identify performance differences and kernel launch changes"""
    
    def __init__(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        verbose: bool = True,
        significance_threshold: float = 0.05,  # 5% change threshold
        min_duration_threshold_us: int = 1000,  # 1ms minimum to be considered significant
    ):
        super().__init__(output_dir, verbose)
        self.significance_threshold = significance_threshold
        self.min_duration_threshold_us = min_duration_threshold_us
        
        self.parser = TraceParser(output_dir, verbose)
        self.comparisons: List[Dict[str, Any]] = []
        
    def enable(self) -> None:
        """Enable trace comparer"""
        self.enabled = True
        self.parser.enable()
        if self.verbose:
            self.logger.info("Trace comparer enabled")
    
    def disable(self) -> None:
        """Disable trace comparer and save results"""
        self.enabled = False
        self.parser.disable()
        self._save_comparisons()
        if self.verbose:
            self.logger.info("Trace comparer disabled and results saved")
    
    def compare_traces(
        self,
        trace1_path: Union[str, Path],
        trace2_path: Union[str, Path],
        comparison_name: Optional[str] = None,
        trace1_label: str = "baseline",
        trace2_label: str = "modified",
    ) -> Dict[str, Any]:
        """Compare two trace files comprehensively"""
        if not self.enabled:
            raise RuntimeError("TraceComparer is not enabled")
        
        # Parse both traces
        trace1_data = self.parser.parse_trace_file(trace1_path, f"{trace1_label}_trace")
        trace2_data = self.parser.parse_trace_file(trace2_path, f"{trace2_label}_trace")
        
        comparison_name = comparison_name or f"{trace1_label}_vs_{trace2_label}"
        
        if self.verbose:
            self.logger.info(f"Comparing traces: {trace1_path} vs {trace2_path}")
        
        comparison_result = {
            "comparison_name": comparison_name,
            "trace1_label": trace1_label,
            "trace2_label": trace2_label,
            "trace1_path": str(trace1_path),
            "trace2_path": str(trace2_path),
            "trace1_events": trace1_data["total_events"],
            "trace2_events": trace2_data["total_events"],
            "timestamp": self.parser.parsed_traces[f"{trace1_label}_trace"]["analysis"]["metadata_events"][0]["ts"] if self.parser.parsed_traces[f"{trace1_label}_trace"]["analysis"]["metadata_events"] else 0,
            
            # Core comparisons
            "function_comparison": self._compare_functions(trace1_data, trace2_data),
            "kernel_comparison": self._compare_kernels(trace1_data, trace2_data),
            "memory_comparison": self._compare_memory_usage(trace1_data, trace2_data),
            "timeline_comparison": self._compare_timelines(trace1_data, trace2_data),
            "efficiency_analysis": self._analyze_efficiency_changes(trace1_data, trace2_data),
            
            # Summary
            "summary": {},
            "recommendations": [],
        }
        
        # Generate summary and recommendations
        comparison_result["summary"] = self._generate_comparison_summary(comparison_result)
        comparison_result["recommendations"] = self._generate_recommendations(comparison_result)
        
        self.comparisons.append(comparison_result)
        
        if self.verbose:
            self.logger.info(f"Comparison completed: {comparison_name}")
        
        return comparison_result
    
    def _compare_functions(self, trace1: Dict[str, Any], trace2: Dict[str, Any]) -> Dict[str, Any]:
        """Compare function call patterns and performance"""
        analysis1 = trace1["analysis"]
        analysis2 = trace2["analysis"]
        
        # Get function statistics
        func_stats1 = self._extract_function_stats(analysis1["duration_events"])
        func_stats2 = self._extract_function_stats(analysis2["duration_events"])
        
        all_functions = set(func_stats1.keys()) | set(func_stats2.keys())
        
        comparison = {
            "total_functions_trace1": len(func_stats1),
            "total_functions_trace2": len(func_stats2),
            "common_functions": len(set(func_stats1.keys()) & set(func_stats2.keys())),
            "unique_to_trace1": list(set(func_stats1.keys()) - set(func_stats2.keys())),
            "unique_to_trace2": list(set(func_stats2.keys()) - set(func_stats1.keys())),
            "function_changes": [],
            "significant_changes": [],
        }
        
        # Compare common functions
        for func_name in all_functions:
            stats1 = func_stats1.get(func_name, {"total_time": 0, "call_count": 0, "avg_time": 0})
            stats2 = func_stats2.get(func_name, {"total_time": 0, "call_count": 0, "avg_time": 0})
            
            change_info = {
                "function_name": func_name,
                "trace1_total_time": stats1["total_time"],
                "trace2_total_time": stats2["total_time"],
                "trace1_call_count": stats1["call_count"],
                "trace2_call_count": stats2["call_count"],
                "trace1_avg_time": stats1["avg_time"],
                "trace2_avg_time": stats2["avg_time"],
                "time_change_percent": 0,
                "call_count_change": stats2["call_count"] - stats1["call_count"],
                "avg_time_change_percent": 0,
                "is_significant": False,
            }
            
            # Calculate percentage changes
            if stats1["total_time"] > 0:
                change_info["time_change_percent"] = (
                    (stats2["total_time"] - stats1["total_time"]) / stats1["total_time"] * 100
                )
            
            if stats1["avg_time"] > 0:
                change_info["avg_time_change_percent"] = (
                    (stats2["avg_time"] - stats1["avg_time"]) / stats1["avg_time"] * 100
                )
            
            # Check if change is significant
            if (abs(change_info["time_change_percent"]) > self.significance_threshold * 100 and
                max(stats1["total_time"], stats2["total_time"]) > self.min_duration_threshold_us):
                change_info["is_significant"] = True
                comparison["significant_changes"].append(change_info)
            
            comparison["function_changes"].append(change_info)
        
        # Sort by absolute time change
        comparison["function_changes"].sort(
            key=lambda x: abs(x["time_change_percent"]), reverse=True
        )
        comparison["significant_changes"].sort(
            key=lambda x: abs(x["time_change_percent"]), reverse=True
        )
        
        return comparison
    
    def _compare_kernels(self, trace1: Dict[str, Any], trace2: Dict[str, Any]) -> Dict[str, Any]:
        """Compare CUDA kernel launches and GPU operations"""
        kernels1 = trace1["analysis"]["gpu_kernels"]
        kernels2 = trace2["analysis"]["gpu_kernels"]
        
        kernel_stats1 = self._extract_kernel_stats(kernels1)
        kernel_stats2 = self._extract_kernel_stats(kernels2)
        
        all_kernels = set(kernel_stats1.keys()) | set(kernel_stats2.keys())
        
        comparison = {
            "total_kernels_trace1": len(kernels1),
            "total_kernels_trace2": len(kernels2),
            "unique_kernel_types_trace1": len(kernel_stats1),
            "unique_kernel_types_trace2": len(kernel_stats2),
            "common_kernels": len(set(kernel_stats1.keys()) & set(kernel_stats2.keys())),
            "new_kernels": list(set(kernel_stats2.keys()) - set(kernel_stats1.keys())),
            "removed_kernels": list(set(kernel_stats1.keys()) - set(kernel_stats2.keys())),
            "kernel_changes": [],
            "efficiency_analysis": {},
        }
        
        # Compare kernel performance
        for kernel_name in all_kernels:
            stats1 = kernel_stats1.get(kernel_name, {"count": 0, "total_time": 0, "avg_time": 0, "max_time": 0})
            stats2 = kernel_stats2.get(kernel_name, {"count": 0, "total_time": 0, "avg_time": 0, "max_time": 0})
            
            change_info = {
                "kernel_name": kernel_name,
                "trace1_launch_count": stats1["count"],
                "trace2_launch_count": stats2["count"],
                "trace1_total_time": stats1["total_time"],
                "trace2_total_time": stats2["total_time"],
                "trace1_avg_time": stats1["avg_time"],
                "trace2_avg_time": stats2["avg_time"],
                "launch_count_change": stats2["count"] - stats1["count"],
                "total_time_change_percent": 0,
                "avg_time_change_percent": 0,
                "efficiency_change": "unknown",
            }
            
            # Calculate changes
            if stats1["total_time"] > 0:
                change_info["total_time_change_percent"] = (
                    (stats2["total_time"] - stats1["total_time"]) / stats1["total_time"] * 100
                )
            
            if stats1["avg_time"] > 0:
                change_info["avg_time_change_percent"] = (
                    (stats2["avg_time"] - stats1["avg_time"]) / stats1["avg_time"] * 100
                )
            
            # Determine efficiency change
            if stats1["count"] > 0 and stats2["count"] > 0:
                if stats2["avg_time"] < stats1["avg_time"]:
                    change_info["efficiency_change"] = "improved"
                elif stats2["avg_time"] > stats1["avg_time"]:
                    change_info["efficiency_change"] = "degraded"
                else:
                    change_info["efficiency_change"] = "unchanged"
            elif stats1["count"] == 0 and stats2["count"] > 0:
                change_info["efficiency_change"] = "new_kernel"
            elif stats1["count"] > 0 and stats2["count"] == 0:
                change_info["efficiency_change"] = "removed_kernel"
            
            comparison["kernel_changes"].append(change_info)
        
        # Efficiency analysis
        total_gpu_time1 = sum(stats["total_time"] for stats in kernel_stats1.values())
        total_gpu_time2 = sum(stats["total_time"] for stats in kernel_stats2.values())
        
        comparison["efficiency_analysis"] = {
            "total_gpu_time_trace1": total_gpu_time1,
            "total_gpu_time_trace2": total_gpu_time2,
            "gpu_time_change_percent": (
                (total_gpu_time2 - total_gpu_time1) / total_gpu_time1 * 100
            ) if total_gpu_time1 > 0 else 0,
            "kernel_launch_count_change": len(kernels2) - len(kernels1),
            "avg_kernel_time_trace1": total_gpu_time1 / len(kernels1) if kernels1 else 0,
            "avg_kernel_time_trace2": total_gpu_time2 / len(kernels2) if kernels2 else 0,
        }
        
        return comparison
    
    def _compare_memory_usage(self, trace1: Dict[str, Any], trace2: Dict[str, Any]) -> Dict[str, Any]:
        """Compare memory usage patterns"""
        memory1 = trace1["analysis"]["memory_events"]
        memory2 = trace2["analysis"]["memory_events"]
        
        counter1 = trace1["analysis"]["counter_events"]
        counter2 = trace2["analysis"]["counter_events"]
        
        # Extract memory-related counters
        memory_counters1 = [e for e in counter1 if "memory" in e.get("name", "").lower()]
        memory_counters2 = [e for e in counter2 if "memory" in e.get("name", "").lower()]
        
        comparison = {
            "memory_events_trace1": len(memory1),
            "memory_events_trace2": len(memory2),
            "memory_counter_samples_trace1": len(memory_counters1),
            "memory_counter_samples_trace2": len(memory_counters2),
            "peak_memory_analysis": {},
            "memory_efficiency": {},
        }
        
        # Analyze peak memory usage from counters
        if memory_counters1 and memory_counters2:
            peak_analysis = self._analyze_peak_memory(memory_counters1, memory_counters2)
            comparison["peak_memory_analysis"] = peak_analysis
        
        # Memory allocation efficiency
        if memory1 and memory2:
            total_memory_time1 = sum(event["duration_us"] for event in memory1)
            total_memory_time2 = sum(event["duration_us"] for event in memory2)
            
            comparison["memory_efficiency"] = {
                "memory_operation_count_trace1": len(memory1),
                "memory_operation_count_trace2": len(memory2),
                "total_memory_time_trace1": total_memory_time1,
                "total_memory_time_trace2": total_memory_time2,
                "avg_memory_operation_time_trace1": total_memory_time1 / len(memory1),
                "avg_memory_operation_time_trace2": total_memory_time2 / len(memory2),
                "memory_time_change_percent": (
                    (total_memory_time2 - total_memory_time1) / total_memory_time1 * 100
                ) if total_memory_time1 > 0 else 0,
            }
        
        return comparison
    
    def _compare_timelines(self, trace1: Dict[str, Any], trace2: Dict[str, Any]) -> Dict[str, Any]:
        """Compare overall timeline characteristics"""
        timeline1 = trace1["timeline"]
        timeline2 = trace2["timeline"]
        
        comparison = {
            "total_duration_trace1_ms": timeline1.get("total_duration_ms", 0),
            "total_duration_trace2_ms": timeline2.get("total_duration_ms", 0),
            "duration_change_percent": 0,
            "concurrent_events_trace1": len(timeline1.get("concurrent_events", [])),
            "concurrent_events_trace2": len(timeline2.get("concurrent_events", [])),
            "concurrency_change": 0,
            "timeline_similarity": 0,
        }
        
        # Calculate duration change
        if timeline1.get("total_duration_ms", 0) > 0:
            comparison["duration_change_percent"] = (
                (timeline2.get("total_duration_ms", 0) - timeline1.get("total_duration_ms", 0)) /
                timeline1.get("total_duration_ms", 0) * 100
            )
        
        # Concurrency analysis
        comparison["concurrency_change"] = (
            len(timeline2.get("concurrent_events", [])) - len(timeline1.get("concurrent_events", []))
        )
        
        # Timeline similarity (simplified)
        # This could be enhanced with more sophisticated sequence alignment algorithms
        events1 = [e.get("name", "") for e in timeline1.get("timeline_events", [])]
        events2 = [e.get("name", "") for e in timeline2.get("timeline_events", [])]
        
        if events1 or events2:
            common_events = len(set(events1) & set(events2))
            total_unique_events = len(set(events1) | set(events2))
            comparison["timeline_similarity"] = common_events / total_unique_events if total_unique_events > 0 else 0
        
        return comparison
    
    def _analyze_efficiency_changes(self, trace1: Dict[str, Any], trace2: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze overall efficiency changes between traces"""
        summary1 = trace1["summary"]
        summary2 = trace2["summary"]
        
        metrics1 = summary1.get("performance_metrics", {})
        metrics2 = summary2.get("performance_metrics", {})
        
        analysis = {
            "overall_performance_change": {},
            "bottleneck_analysis": {},
            "efficiency_score": {},
            "recommendations_based_on_changes": [],
        }
        
        # Overall performance comparison
        if metrics1 and metrics2:
            analysis["overall_performance_change"] = {
                "total_execution_time_change_percent": self._calculate_percentage_change(
                    metrics1.get("total_execution_time_ms", 0),
                    metrics2.get("total_execution_time_ms", 0)
                ),
                "avg_function_duration_change_percent": self._calculate_percentage_change(
                    metrics1.get("avg_function_duration_us", 0),
                    metrics2.get("avg_function_duration_us", 0)
                ),
                "total_function_calls_change": (
                    metrics2.get("total_calls", 0) - metrics1.get("total_calls", 0)
                ),
                "function_count_change": (
                    metrics2.get("total_functions", 0) - metrics1.get("total_functions", 0)
                ),
            }
        
        # Bottleneck analysis
        bottlenecks1 = summary1.get("bottlenecks", [])
        bottlenecks2 = summary2.get("bottlenecks", [])
        
        analysis["bottleneck_analysis"] = {
            "bottleneck_count_trace1": len(bottlenecks1),
            "bottleneck_count_trace2": len(bottlenecks2),
            "new_bottlenecks": [
                b["function"] for b in bottlenecks2
                if b["function"] not in [b1["function"] for b1 in bottlenecks1]
            ],
            "resolved_bottlenecks": [
                b["function"] for b in bottlenecks1
                if b["function"] not in [b2["function"] for b2 in bottlenecks2]
            ],
            "persistent_bottlenecks": [
                b["function"] for b in bottlenecks1
                if b["function"] in [b2["function"] for b2 in bottlenecks2]
            ],
        }
        
        # Efficiency score (simplified)
        total_time1 = metrics1.get("total_execution_time_us", 1)
        total_time2 = metrics2.get("total_execution_time_us", 1)
        function_count1 = metrics1.get("total_functions", 1)
        function_count2 = metrics2.get("total_functions", 1)
        
        efficiency1 = function_count1 / (total_time1 / 1000000)  # functions per second
        efficiency2 = function_count2 / (total_time2 / 1000000)
        
        analysis["efficiency_score"] = {
            "efficiency_trace1": efficiency1,
            "efficiency_trace2": efficiency2,
            "efficiency_change_percent": self._calculate_percentage_change(efficiency1, efficiency2),
            "interpretation": "improved" if efficiency2 > efficiency1 else "degraded" if efficiency2 < efficiency1 else "unchanged"
        }
        
        return analysis
    
    def _extract_function_stats(self, events: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Extract function statistics from duration events"""
        stats = defaultdict(lambda: {"total_time": 0, "call_count": 0, "times": []})
        
        for event in events:
            name = event.get("name", "unknown")
            duration = event.get("duration_us", 0)
            
            stats[name]["total_time"] += duration
            stats[name]["call_count"] += 1
            stats[name]["times"].append(duration)
        
        # Calculate averages
        for name, stat in stats.items():
            if stat["call_count"] > 0:
                stat["avg_time"] = stat["total_time"] / stat["call_count"]
                stat["max_time"] = max(stat["times"])
                stat["min_time"] = min(stat["times"])
            else:
                stat["avg_time"] = 0
                stat["max_time"] = 0
                stat["min_time"] = 0
        
        return dict(stats)
    
    def _extract_kernel_stats(self, events: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Extract kernel statistics from GPU events"""
        stats = defaultdict(lambda: {"count": 0, "total_time": 0, "times": []})
        
        for event in events:
            name = event.get("name", "unknown")
            duration = event.get("duration_us", 0)
            
            stats[name]["count"] += 1
            stats[name]["total_time"] += duration
            stats[name]["times"].append(duration)
        
        # Calculate statistics
        for name, stat in stats.items():
            if stat["count"] > 0:
                stat["avg_time"] = stat["total_time"] / stat["count"]
                stat["max_time"] = max(stat["times"])
                stat["min_time"] = min(stat["times"])
            else:
                stat["avg_time"] = 0
                stat["max_time"] = 0
                stat["min_time"] = 0
        
        return dict(stats)
    
    def _analyze_peak_memory(self, counters1: List[Dict[str, Any]], counters2: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze peak memory usage from counter events"""
        # Extract memory values
        memory_values1 = []
        memory_values2 = []
        
        for event in counters1:
            args = event.get("args", {})
            for key, value in args.items():
                if "memory" in key.lower() and "used" in key.lower():
                    try:
                        memory_values1.append(float(value))
                    except (ValueError, TypeError):
                        pass
        
        for event in counters2:
            args = event.get("args", {})
            for key, value in args.items():
                if "memory" in key.lower() and "used" in key.lower():
                    try:
                        memory_values2.append(float(value))
                    except (ValueError, TypeError):
                        pass
        
        analysis = {}
        
        if memory_values1:
            analysis["peak_memory_trace1"] = max(memory_values1)
            analysis["avg_memory_trace1"] = statistics.mean(memory_values1)
        
        if memory_values2:
            analysis["peak_memory_trace2"] = max(memory_values2)
            analysis["avg_memory_trace2"] = statistics.mean(memory_values2)
        
        if memory_values1 and memory_values2:
            analysis["peak_memory_change_percent"] = self._calculate_percentage_change(
                max(memory_values1), max(memory_values2)
            )
            analysis["avg_memory_change_percent"] = self._calculate_percentage_change(
                statistics.mean(memory_values1), statistics.mean(memory_values2)
            )
        
        return analysis
    
    def _calculate_percentage_change(self, old_value: float, new_value: float) -> float:
        """Calculate percentage change between two values"""
        if old_value == 0:
            return 0 if new_value == 0 else float('inf')
        return (new_value - old_value) / old_value * 100
    
    def _generate_comparison_summary(self, comparison: Dict[str, Any]) -> Dict[str, Any]:
        """Generate high-level summary of comparison"""
        func_comp = comparison["function_comparison"]
        kernel_comp = comparison["kernel_comparison"]
        timeline_comp = comparison["timeline_comparison"]
        efficiency = comparison["efficiency_analysis"]
        
        summary = {
            "overall_assessment": "unknown",
            "key_changes": [],
            "performance_impact": "neutral",
            "confidence": "medium",
            "significant_findings": [],
        }
        
        # Determine overall assessment
        duration_change = timeline_comp.get("duration_change_percent", 0)
        significant_func_changes = len(func_comp.get("significant_changes", []))
        new_kernels = len(kernel_comp.get("new_kernels", []))
        removed_kernels = len(kernel_comp.get("removed_kernels", []))
        
        if abs(duration_change) > 10:  # >10% change
            summary["overall_assessment"] = "significant_change"
            summary["confidence"] = "high"
        elif significant_func_changes > 0 or new_kernels > 0 or removed_kernels > 0:
            summary["overall_assessment"] = "moderate_change"
            summary["confidence"] = "medium"
        else:
            summary["overall_assessment"] = "minimal_change"
            summary["confidence"] = "low"
        
        # Performance impact
        if duration_change < -5:  # >5% improvement
            summary["performance_impact"] = "improvement"
        elif duration_change > 5:  # >5% degradation
            summary["performance_impact"] = "degradation"
        else:
            summary["performance_impact"] = "neutral"
        
        # Key changes
        if abs(duration_change) > 1:
            summary["key_changes"].append(f"Total duration changed by {duration_change:.1f}%")
        
        if new_kernels:
            summary["key_changes"].append(f"{len(new_kernels)} new kernel types detected")
        
        if removed_kernels:
            summary["key_changes"].append(f"{len(removed_kernels)} kernel types removed")
        
        if significant_func_changes > 0:
            summary["key_changes"].append(f"{significant_func_changes} functions with significant changes")
        
        # Significant findings
        for change in func_comp.get("significant_changes", [])[:3]:  # Top 3
            summary["significant_findings"].append(
                f"Function '{change['function_name']}' changed by {change['time_change_percent']:.1f}%"
            )
        
        return summary
    
    def _generate_recommendations(self, comparison: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on comparison results"""
        recommendations = []
        
        summary = comparison["summary"]
        func_comp = comparison["function_comparison"]
        kernel_comp = comparison["kernel_comparison"]
        efficiency = comparison["efficiency_analysis"]
        
        # Performance recommendations
        if summary["performance_impact"] == "degradation":
            recommendations.append("Performance has degraded. Consider reverting recent changes or optimizing bottlenecks.")
        elif summary["performance_impact"] == "improvement":
            recommendations.append("Performance has improved. Document the changes for future reference.")
        
        # Function-specific recommendations
        bottleneck_analysis = efficiency.get("bottleneck_analysis", {})
        if bottleneck_analysis.get("new_bottlenecks"):
            recommendations.append(
                f"New bottlenecks detected: {', '.join(bottleneck_analysis['new_bottlenecks'][:3])}. "
                "Consider optimizing these functions."
            )
        
        # Kernel recommendations
        if kernel_comp.get("new_kernels"):
            recommendations.append(
                "New GPU kernels detected. Verify they are optimal and not causing memory issues."
            )
        
        if kernel_comp.get("removed_kernels"):
            recommendations.append(
                "Some GPU kernels were removed. Ensure this doesn't impact functionality."
            )
        
        # Memory recommendations
        memory_comp = comparison["memory_comparison"]
        peak_analysis = memory_comp.get("peak_memory_analysis", {})
        if peak_analysis.get("peak_memory_change_percent", 0) > 20:
            recommendations.append(
                "Peak memory usage increased significantly. Monitor for potential memory leaks."
            )
        
        return recommendations
    
    def _save_comparisons(self) -> None:
        """Save all comparison results"""
        if not self.comparisons:
            return
        
        # Save detailed comparisons
        self.save_json(self.comparisons, "trace_comparisons.json")
        
        # Create summary report
        summary_report = {
            "total_comparisons": len(self.comparisons),
            "comparison_names": [c["comparison_name"] for c in self.comparisons],
            "performance_trends": self._analyze_performance_trends(),
            "common_issues": self._identify_common_issues(),
        }
        
        self.save_json(summary_report, "comparison_summary_report.json")
        
        if self.verbose:
            self.logger.info(f"Saved {len(self.comparisons)} trace comparisons to {self.output_dir}")
    
    def _analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze trends across multiple comparisons"""
        if not self.comparisons:
            return {}
        
        improvements = sum(1 for c in self.comparisons if c["summary"]["performance_impact"] == "improvement")
        degradations = sum(1 for c in self.comparisons if c["summary"]["performance_impact"] == "degradation")
        
        return {
            "total_comparisons": len(self.comparisons),
            "improvements": improvements,
            "degradations": degradations,
            "neutral": len(self.comparisons) - improvements - degradations,
            "improvement_rate": improvements / len(self.comparisons) if self.comparisons else 0,
        }
    
    def _identify_common_issues(self) -> Dict[str, Any]:
        """Identify common issues across comparisons"""
        common_bottlenecks = defaultdict(int)
        common_new_kernels = defaultdict(int)
        
        for comparison in self.comparisons:
            # Count bottlenecks
            bottlenecks = comparison["efficiency_analysis"].get("bottleneck_analysis", {}).get("new_bottlenecks", [])
            for bottleneck in bottlenecks:
                common_bottlenecks[bottleneck] += 1
            
            # Count new kernels
            new_kernels = comparison["kernel_comparison"].get("new_kernels", [])
            for kernel in new_kernels:
                common_new_kernels[kernel] += 1
        
        return {
            "frequent_bottlenecks": dict(sorted(common_bottlenecks.items(), key=lambda x: x[1], reverse=True)[:5]),
            "frequent_new_kernels": dict(sorted(common_new_kernels.items(), key=lambda x: x[1], reverse=True)[:5]),
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comparer statistics"""
        return {
            "total_comparisons": len(self.comparisons),
            "parser_stats": self.parser.get_statistics(),
            "recent_comparisons": [c["comparison_name"] for c in self.comparisons[-5:]],
        }