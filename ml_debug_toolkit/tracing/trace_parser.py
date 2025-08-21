"""
Chrome trace parser for analyzing performance traces
"""

import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from ..core.base import BaseDebugTool


class TraceParser(BaseDebugTool):
    """Parse and analyze Chrome trace format files"""
    
    def __init__(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        verbose: bool = True,
    ):
        super().__init__(output_dir, verbose)
        self.parsed_traces: Dict[str, Dict[str, Any]] = {}
        
    def enable(self) -> None:
        """Enable trace parser"""
        self.enabled = True
        if self.verbose:
            self.logger.info("Trace parser enabled")
    
    def disable(self) -> None:
        """Disable trace parser"""
        self.enabled = False
        if self.verbose:
            self.logger.info("Trace parser disabled")
    
    def parse_trace_file(self, trace_file: Union[str, Path], trace_id: Optional[str] = None) -> Dict[str, Any]:
        """Parse a Chrome trace format file"""
        if not self.enabled:
            raise RuntimeError("TraceParser is not enabled")
        
        trace_file = Path(trace_file)
        if not trace_file.exists():
            raise FileNotFoundError(f"Trace file not found: {trace_file}")
        
        if trace_id is None:
            trace_id = trace_file.stem
        
        if self.verbose:
            self.logger.info(f"Parsing trace file: {trace_file}")
        
        # Load trace data
        with open(trace_file, 'r') as f:
            trace_data = json.load(f)
        
        # Extract events
        events = trace_data.get("traceEvents", [])
        
        # Parse events
        parsed_data = {
            "trace_id": trace_id,
            "source_file": str(trace_file),
            "metadata": trace_data.get("metadata", {}),
            "total_events": len(events),
            "events": events,
            "analysis": self._analyze_events(events),
            "timeline": self._build_timeline(events),
            "summary": self._generate_summary(events),
        }
        
        self.parsed_traces[trace_id] = parsed_data
        
        if self.verbose:
            self.logger.info(f"Parsed {len(events)} events from {trace_file}")
        
        return parsed_data
    
    def _analyze_events(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze trace events for patterns and performance metrics"""
        analysis = {
            "event_types": defaultdict(int),
            "categories": defaultdict(int),
            "processes": set(),
            "threads": set(),
            "duration_events": [],
            "counter_events": [],
            "instant_events": [],
            "metadata_events": [],
            "function_calls": [],
            "gpu_kernels": [],
            "memory_events": [],
        }
        
        # Stack for tracking begin/end pairs
        call_stack = defaultdict(list)
        
        for event in events:
            ph = event.get("ph")
            analysis["event_types"][ph] += 1
            
            category = event.get("cat", "unknown")
            analysis["categories"][category] += 1
            
            pid = event.get("pid")
            tid = event.get("tid")
            if pid is not None:
                analysis["processes"].add(pid)
            if tid is not None:
                analysis["threads"].add(tid)
            
            # Categorize events
            if ph == "B":  # Begin
                call_stack[tid].append(event)
            elif ph == "E":  # End
                if call_stack[tid]:
                    begin_event = call_stack[tid].pop()
                    duration = event.get("ts", 0) - begin_event.get("ts", 0)
                    
                    duration_event = {
                        "name": begin_event.get("name"),
                        "category": begin_event.get("cat"),
                        "start_ts": begin_event.get("ts"),
                        "end_ts": event.get("ts"),
                        "duration_us": duration,
                        "duration_ms": duration / 1000.0,
                        "pid": pid,
                        "tid": tid,
                        "args": begin_event.get("args", {}),
                    }
                    
                    analysis["duration_events"].append(duration_event)
                    
                    # Categorize specific types
                    if "cuda" in category.lower() or "gpu" in category.lower():
                        analysis["gpu_kernels"].append(duration_event)
                    elif "memory" in category.lower():
                        analysis["memory_events"].append(duration_event)
                    else:
                        analysis["function_calls"].append(duration_event)
            
            elif ph == "X":  # Complete event (has duration)
                duration = event.get("dur", 0)
                duration_event = {
                    "name": event.get("name"),
                    "category": event.get("cat"),
                    "start_ts": event.get("ts"),
                    "end_ts": event.get("ts", 0) + duration,
                    "duration_us": duration,
                    "duration_ms": duration / 1000.0,
                    "pid": pid,
                    "tid": tid,
                    "args": event.get("args", {}),
                }
                analysis["duration_events"].append(duration_event)
            
            elif ph == "C":  # Counter
                analysis["counter_events"].append(event)
            elif ph == "i":  # Instant
                analysis["instant_events"].append(event)
            elif ph == "M":  # Metadata
                analysis["metadata_events"].append(event)
        
        # Convert sets to lists for JSON serialization
        analysis["processes"] = list(analysis["processes"])
        analysis["threads"] = list(analysis["threads"])
        
        return analysis
    
    def _build_timeline(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build timeline analysis from events"""
        timeline = {
            "start_time": None,
            "end_time": None,
            "total_duration_us": 0,
            "total_duration_ms": 0,
            "timeline_events": [],
            "concurrent_events": [],
        }
        
        # Find time bounds
        timestamps = []
        for event in events:
            ts = event.get("ts")
            if ts is not None:
                timestamps.append(ts)
                
                # Also check for end timestamps
                if event.get("ph") == "X" and event.get("dur"):
                    timestamps.append(ts + event.get("dur"))
        
        if timestamps:
            timeline["start_time"] = min(timestamps)
            timeline["end_time"] = max(timestamps)
            timeline["total_duration_us"] = timeline["end_time"] - timeline["start_time"]
            timeline["total_duration_ms"] = timeline["total_duration_us"] / 1000.0
        
        # Build timeline of significant events
        duration_events = []
        for event in events:
            if event.get("ph") in ["B", "E", "X"]:
                duration_events.append(event)
        
        # Sort by timestamp
        duration_events.sort(key=lambda x: x.get("ts", 0))
        timeline["timeline_events"] = duration_events[:100]  # Limit to first 100
        
        # Find concurrent events (overlapping time ranges)
        timeline["concurrent_events"] = self._find_concurrent_events(events)
        
        return timeline
    
    def _find_concurrent_events(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find events that execute concurrently"""
        # This is a simplified implementation
        # A full implementation would use interval trees for efficiency
        
        duration_events = []
        call_stack = defaultdict(list)
        
        # Build duration events from begin/end pairs
        for event in events:
            ph = event.get("ph")
            tid = event.get("tid")
            
            if ph == "B":
                call_stack[tid].append(event)
            elif ph == "E" and call_stack[tid]:
                begin_event = call_stack[tid].pop()
                duration_events.append({
                    "name": begin_event.get("name"),
                    "start": begin_event.get("ts", 0),
                    "end": event.get("ts", 0),
                    "tid": tid,
                    "pid": event.get("pid"),
                })
        
        # Find overlapping events
        concurrent = []
        for i, event1 in enumerate(duration_events):
            for j, event2 in enumerate(duration_events[i+1:], i+1):
                # Check if events overlap and are on different threads
                if (event1["tid"] != event2["tid"] and
                    event1["start"] < event2["end"] and
                    event2["start"] < event1["end"]):
                    concurrent.append({
                        "event1": event1["name"],
                        "event2": event2["name"],
                        "overlap_start": max(event1["start"], event2["start"]),
                        "overlap_end": min(event1["end"], event2["end"]),
                        "overlap_duration": min(event1["end"], event2["end"]) - max(event1["start"], event2["start"])
                    })
        
        return concurrent[:50]  # Limit results
    
    def _generate_summary(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary statistics"""
        summary = {
            "total_events": len(events),
            "event_type_counts": defaultdict(int),
            "category_counts": defaultdict(int),
            "top_functions_by_duration": [],
            "top_functions_by_calls": [],
            "performance_metrics": {},
            "bottlenecks": [],
        }
        
        # Count event types and categories
        for event in events:
            summary["event_type_counts"][event.get("ph", "unknown")] += 1
            summary["category_counts"][event.get("cat", "unknown")] += 1
        
        # Convert to regular dicts
        summary["event_type_counts"] = dict(summary["event_type_counts"])
        summary["category_counts"] = dict(summary["category_counts"])
        
        # Analyze duration events for performance
        duration_events = []
        call_stack = defaultdict(list)
        
        for event in events:
            ph = event.get("ph")
            tid = event.get("tid")
            
            if ph == "B":
                call_stack[tid].append(event)
            elif ph == "E" and call_stack[tid]:
                begin_event = call_stack[tid].pop()
                duration = event.get("ts", 0) - begin_event.get("ts", 0)
                duration_events.append({
                    "name": begin_event.get("name"),
                    "duration": duration,
                    "category": begin_event.get("cat"),
                })
            elif ph == "X":
                duration_events.append({
                    "name": event.get("name"),
                    "duration": event.get("dur", 0),
                    "category": event.get("cat"),
                })
        
        if duration_events:
            # Function analysis
            function_stats = defaultdict(lambda: {"total_time": 0, "call_count": 0, "times": []})
            
            for event in duration_events:
                name = event["name"]
                duration = event["duration"]
                function_stats[name]["total_time"] += duration
                function_stats[name]["call_count"] += 1
                function_stats[name]["times"].append(duration)
            
            # Top functions by total duration
            by_duration = sorted(
                [(name, stats) for name, stats in function_stats.items()],
                key=lambda x: x[1]["total_time"],
                reverse=True
            )[:10]
            
            summary["top_functions_by_duration"] = [
                {
                    "name": name,
                    "total_time_us": stats["total_time"],
                    "total_time_ms": stats["total_time"] / 1000.0,
                    "call_count": stats["call_count"],
                    "avg_time_us": stats["total_time"] / stats["call_count"],
                    "max_time_us": max(stats["times"]),
                    "min_time_us": min(stats["times"]),
                }
                for name, stats in by_duration
            ]
            
            # Top functions by call count
            by_calls = sorted(
                [(name, stats) for name, stats in function_stats.items()],
                key=lambda x: x[1]["call_count"],
                reverse=True
            )[:10]
            
            summary["top_functions_by_calls"] = [
                {
                    "name": name,
                    "call_count": stats["call_count"],
                    "total_time_us": stats["total_time"],
                    "avg_time_us": stats["total_time"] / stats["call_count"],
                }
                for name, stats in by_calls
            ]
            
            # Performance metrics
            all_durations = [event["duration"] for event in duration_events]
            if all_durations:
                summary["performance_metrics"] = {
                    "total_functions": len(function_stats),
                    "total_calls": len(duration_events),
                    "total_execution_time_us": sum(all_durations),
                    "total_execution_time_ms": sum(all_durations) / 1000.0,
                    "avg_function_duration_us": statistics.mean(all_durations),
                    "median_function_duration_us": statistics.median(all_durations),
                    "max_function_duration_us": max(all_durations),
                    "min_function_duration_us": min(all_durations),
                    "std_function_duration_us": statistics.stdev(all_durations) if len(all_durations) > 1 else 0,
                }
            
            # Identify bottlenecks (functions taking >10ms or >1% of total time)
            total_time = sum(all_durations)
            threshold_time = max(10000, total_time * 0.01)  # 10ms or 1% of total
            
            for name, stats in function_stats.items():
                if stats["total_time"] > threshold_time:
                    summary["bottlenecks"].append({
                        "function": name,
                        "total_time_us": stats["total_time"],
                        "percentage_of_total": (stats["total_time"] / total_time) * 100,
                        "call_count": stats["call_count"],
                        "avg_time_us": stats["total_time"] / stats["call_count"],
                    })
            
            # Sort bottlenecks by total time
            summary["bottlenecks"].sort(key=lambda x: x["total_time_us"], reverse=True)
        
        return summary
    
    def save_analysis(self, trace_id: str, filename: Optional[str] = None) -> Path:
        """Save parsed analysis to file"""
        if trace_id not in self.parsed_traces:
            raise ValueError(f"Trace {trace_id} not found")
        
        if filename is None:
            filename = f"trace_analysis_{trace_id}.json"
        
        analysis_data = self.parsed_traces[trace_id]
        
        # Create a copy without the raw events for cleaner output
        clean_data = {
            "trace_id": analysis_data["trace_id"],
            "source_file": analysis_data["source_file"],
            "metadata": analysis_data["metadata"],
            "total_events": analysis_data["total_events"],
            "analysis": analysis_data["analysis"],
            "timeline": analysis_data["timeline"],
            "summary": analysis_data["summary"],
        }
        
        filepath = self.save_json(clean_data, filename)
        
        if self.verbose:
            self.logger.info(f"Saved trace analysis to {filepath}")
        
        return filepath
    
    def get_trace_summary(self, trace_id: str) -> Dict[str, Any]:
        """Get summary for a specific trace"""
        if trace_id not in self.parsed_traces:
            raise ValueError(f"Trace {trace_id} not found")
        
        return self.parsed_traces[trace_id]["summary"]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get parser statistics"""
        return {
            "parsed_traces": len(self.parsed_traces),
            "trace_ids": list(self.parsed_traces.keys()),
            "total_events_parsed": sum(
                trace["total_events"] for trace in self.parsed_traces.values()
            ),
        }