"""
Function/module/training run tester with comprehensive analysis
"""

import importlib
import inspect
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from ..core.base import BaseDebugTool, get_memory_info


class TestRunner(BaseDebugTool):
    """Comprehensive testing utility for functions, modules, and training runs"""
    
    def __init__(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        verbose: bool = True,
        capture_output: bool = True,
        track_memory: bool = True,
        timeout: Optional[float] = None,
        max_retries: int = 3,
    ):
        super().__init__(output_dir, verbose)
        self.capture_output = capture_output
        self.track_memory = track_memory
        self.timeout = timeout
        self.max_retries = max_retries
        
        self.test_results: List[Dict[str, Any]] = []
        
    def enable(self) -> None:
        """Enable test runner"""
        self.enabled = True
        if self.verbose:
            self.logger.info("Test runner enabled")
    
    def disable(self) -> None:
        """Disable test runner and save results"""
        self.enabled = False
        self._save_results()
        if self.verbose:
            self.logger.info("Test runner disabled and results saved")
    
    def test_function(
        self,
        func: Callable,
        test_cases: List[Dict[str, Any]],
        test_name: Optional[str] = None,
        expected_outputs: Optional[List[Any]] = None,
        comparison_fn: Optional[Callable] = None,
        setup_fn: Optional[Callable] = None,
        teardown_fn: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """Test a function with multiple test cases"""
        if not self.enabled:
            raise RuntimeError("TestRunner is not enabled")
        
        test_name = test_name or f"test_{func.__name__}"
        
        if self.verbose:
            self.logger.info(f"Testing function {func.__name__} with {len(test_cases)} test cases")
        
        test_result = {
            "test_name": test_name,
            "function_name": func.__name__,
            "function_module": func.__module__,
            "function_signature": str(inspect.signature(func)),
            "timestamp": time.time(),
            "test_cases": [],
            "summary": {
                "total_cases": len(test_cases),
                "passed": 0,
                "failed": 0,
                "errors": 0,
                "total_time": 0,
            }
        }
        
        start_time = time.time()
        
        for i, test_case in enumerate(test_cases):
            case_result = self._run_single_test_case(
                func, test_case, i, expected_outputs[i] if expected_outputs else None,
                comparison_fn, setup_fn, teardown_fn
            )
            test_result["test_cases"].append(case_result)
            
            # Update summary
            if case_result["status"] == "passed":
                test_result["summary"]["passed"] += 1
            elif case_result["status"] == "failed":
                test_result["summary"]["failed"] += 1
            else:
                test_result["summary"]["errors"] += 1
        
        test_result["summary"]["total_time"] = time.time() - start_time
        test_result["summary"]["success_rate"] = test_result["summary"]["passed"] / len(test_cases)
        
        self.test_results.append(test_result)
        
        if self.verbose:
            self.logger.info(
                f"Test {test_name} completed: "
                f"{test_result['summary']['passed']}/{len(test_cases)} passed "
                f"({test_result['summary']['success_rate']:.2%})"
            )
        
        return test_result
    
    def _run_single_test_case(
        self,
        func: Callable,
        test_case: Dict[str, Any],
        case_index: int,
        expected_output: Any = None,
        comparison_fn: Optional[Callable] = None,
        setup_fn: Optional[Callable] = None,
        teardown_fn: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """Run a single test case"""
        case_result = {
            "case_index": case_index,
            "inputs": test_case,
            "timestamp": time.time(),
            "status": "unknown",
            "execution_time": 0,
            "memory_before": None,
            "memory_after": None,
            "output": None,
            "expected_output": expected_output,
            "error": None,
            "stdout": "",
            "stderr": "",
        }
        
        try:
            # Setup
            if setup_fn:
                setup_fn()
            
            # Memory tracking
            if self.track_memory:
                case_result["memory_before"] = get_memory_info()
            
            # Prepare arguments
            args = test_case.get("args", ())
            kwargs = test_case.get("kwargs", {})
            
            # Execute with output capture
            start_time = time.time()
            
            if self.capture_output:
                stdout_capture = StringIO()
                stderr_capture = StringIO()
                
                with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                    output = func(*args, **kwargs)
                
                case_result["stdout"] = stdout_capture.getvalue()
                case_result["stderr"] = stderr_capture.getvalue()
            else:
                output = func(*args, **kwargs)
            
            case_result["execution_time"] = time.time() - start_time
            case_result["output"] = output
            
            # Memory tracking after
            if self.track_memory:
                case_result["memory_after"] = get_memory_info()
            
            # Compare with expected output
            if expected_output is not None:
                if comparison_fn:
                    matches = comparison_fn(output, expected_output)
                else:
                    matches = self._default_comparison(output, expected_output)
                
                case_result["status"] = "passed" if matches else "failed"
                if not matches:
                    case_result["comparison_details"] = {
                        "expected": expected_output,
                        "actual": output,
                        "match": False
                    }
            else:
                case_result["status"] = "passed"
            
            # Teardown
            if teardown_fn:
                teardown_fn()
        
        except Exception as e:
            case_result["execution_time"] = time.time() - start_time if 'start_time' in locals() else 0
            case_result["status"] = "error"
            case_result["error"] = {
                "type": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc()
            }
            
            if self.track_memory and case_result["memory_before"]:
                case_result["memory_after"] = get_memory_info()
            
            # Teardown even on error
            if teardown_fn:
                try:
                    teardown_fn()
                except Exception as teardown_error:
                    case_result["teardown_error"] = str(teardown_error)
        
        return case_result
    
    def _default_comparison(self, actual: Any, expected: Any) -> bool:
        """Default comparison function"""
        try:
            # Handle tensor comparisons
            if hasattr(actual, 'shape') and hasattr(expected, 'shape'):
                import torch
                if isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor):
                    return torch.allclose(actual, expected, rtol=1e-5, atol=1e-8)
            
            # Handle numpy arrays
            import numpy as np
            if isinstance(actual, np.ndarray) and isinstance(expected, np.ndarray):
                return np.allclose(actual, expected, rtol=1e-5, atol=1e-8)
            
            # Default equality
            return actual == expected
            
        except Exception:
            # Fallback to string comparison
            return str(actual) == str(expected)
    
    def test_module(
        self,
        module_path: Union[str, Path],
        test_functions: Optional[List[str]] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Test a Python module by importing and running its functions"""
        module_path = Path(module_path)
        if not module_path.exists():
            raise FileNotFoundError(f"Module not found: {module_path}")
        
        module_name = module_path.stem
        timeout = timeout or self.timeout
        
        test_result = {
            "module_path": str(module_path),
            "module_name": module_name,
            "timestamp": time.time(),
            "functions_tested": [],
            "import_success": False,
            "import_error": None,
            "summary": {
                "total_functions": 0,
                "successful_imports": 0,
                "failed_imports": 0,
                "total_time": 0,
            }
        }
        
        start_time = time.time()
        
        try:
            # Import module
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            test_result["import_success"] = True
            
            # Get functions to test
            if test_functions is None:
                test_functions = [
                    name for name, obj in inspect.getmembers(module, inspect.isfunction)
                    if not name.startswith('_') and obj.__module__ == module_name
                ]
            
            test_result["summary"]["total_functions"] = len(test_functions)
            
            # Test each function
            for func_name in test_functions:
                func_result = self._test_module_function(module, func_name, timeout)
                test_result["functions_tested"].append(func_result)
                
                if func_result["success"]:
                    test_result["summary"]["successful_imports"] += 1
                else:
                    test_result["summary"]["failed_imports"] += 1
        
        except Exception as e:
            test_result["import_error"] = {
                "type": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc()
            }
        
        test_result["summary"]["total_time"] = time.time() - start_time
        self.test_results.append(test_result)
        
        return test_result
    
    def _test_module_function(
        self, 
        module: Any, 
        func_name: str, 
        timeout: Optional[float]
    ) -> Dict[str, Any]:
        """Test a single function from a module"""
        func_result = {
            "function_name": func_name,
            "success": False,
            "signature": None,
            "docstring": None,
            "error": None,
            "inspection_details": {}
        }
        
        try:
            func = getattr(module, func_name)
            func_result["signature"] = str(inspect.signature(func))
            func_result["docstring"] = inspect.getdoc(func)
            
            # Basic inspection
            func_result["inspection_details"] = {
                "is_coroutine": inspect.iscoroutinefunction(func),
                "is_generator": inspect.isgeneratorfunction(func),
                "source_lines": len(inspect.getsourcelines(func)[0]) if hasattr(inspect, 'getsourcelines') else None,
                "annotations": getattr(func, '__annotations__', {}),
            }
            
            func_result["success"] = True
            
        except Exception as e:
            func_result["error"] = {
                "type": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc()
            }
        
        return func_result
    
    def run_training_script(
        self,
        script_path: Union[str, Path],
        args: Optional[List[str]] = None,
        env_vars: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
        working_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Run a training script as a subprocess and monitor it"""
        script_path = Path(script_path)
        if not script_path.exists():
            raise FileNotFoundError(f"Script not found: {script_path}")
        
        args = args or []
        timeout = timeout or self.timeout
        working_dir = Path(working_dir) if working_dir else script_path.parent
        
        run_result = {
            "script_path": str(script_path),
            "args": args,
            "env_vars": env_vars,
            "working_dir": str(working_dir),
            "timestamp": time.time(),
            "success": False,
            "return_code": None,
            "execution_time": 0,
            "stdout": "",
            "stderr": "",
            "timeout_occurred": False,
            "memory_monitoring": [],
            "error": None,
        }
        
        cmd = [sys.executable, str(script_path)] + args
        
        if self.verbose:
            self.logger.info(f"Running training script: {' '.join(cmd)}")
        
        start_time = time.time()
        
        try:
            # Set up environment
            import os
            env = os.environ.copy()
            if env_vars:
                env.update(env_vars)
            
            # Start process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=working_dir,
                env=env
            )
            
            # Monitor process
            if self.track_memory:
                memory_thread = self._start_memory_monitoring(process, run_result)
            
            # Wait for completion
            try:
                stdout, stderr = process.communicate(timeout=timeout)
                run_result["return_code"] = process.returncode
                run_result["stdout"] = stdout
                run_result["stderr"] = stderr
                run_result["success"] = process.returncode == 0
                
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                run_result["timeout_occurred"] = True
                run_result["stdout"] = stdout
                run_result["stderr"] = stderr
                run_result["return_code"] = -1
            
            # Stop memory monitoring
            if self.track_memory:
                memory_thread.stop()
        
        except Exception as e:
            run_result["error"] = {
                "type": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc()
            }
        
        run_result["execution_time"] = time.time() - start_time
        self.test_results.append(run_result)
        
        if self.verbose:
            status = "succeeded" if run_result["success"] else "failed"
            self.logger.info(f"Training script {status} in {run_result['execution_time']:.2f}s")
        
        return run_result
    
    def _start_memory_monitoring(self, process, run_result: Dict[str, Any]):
        """Start monitoring memory usage of a process"""
        import threading
        import psutil
        
        class MemoryMonitor(threading.Thread):
            def __init__(self):
                super().__init__(daemon=True)
                self.running = True
                
            def run(self):
                try:
                    ps_process = psutil.Process(process.pid)
                    while self.running and process.poll() is None:
                        try:
                            memory_info = ps_process.memory_info()
                            run_result["memory_monitoring"].append({
                                "timestamp": time.time(),
                                "rss": memory_info.rss,
                                "vms": memory_info.vms,
                                "cpu_percent": ps_process.cpu_percent()
                            })
                            time.sleep(1)  # Monitor every second
                        except psutil.NoSuchProcess:
                            break
                except Exception:
                    pass
            
            def stop(self):
                self.running = False
        
        monitor = MemoryMonitor()
        monitor.start()
        return monitor
    
    def _save_results(self) -> None:
        """Save all test results"""
        if not self.test_results:
            return
        
        # Save detailed results
        self.save_json(self.test_results, "test_results.json")
        
        # Create summary
        summary = {
            "total_tests": len(self.test_results),
            "test_types": {},
            "overall_stats": {
                "total_functions_tested": 0,
                "total_test_cases": 0,
                "total_execution_time": 0,
            }
        }
        
        for result in self.test_results:
            if "function_name" in result:
                test_type = "function_test"
                summary["overall_stats"]["total_test_cases"] += result["summary"]["total_cases"]
            elif "module_name" in result:
                test_type = "module_test"
                summary["overall_stats"]["total_functions_tested"] += result["summary"]["total_functions"]
            elif "script_path" in result:
                test_type = "training_script"
            else:
                test_type = "unknown"
            
            if test_type not in summary["test_types"]:
                summary["test_types"][test_type] = 0
            summary["test_types"][test_type] += 1
            
            if "summary" in result and "total_time" in result["summary"]:
                summary["overall_stats"]["total_execution_time"] += result["summary"]["total_time"]
            elif "execution_time" in result:
                summary["overall_stats"]["total_execution_time"] += result["execution_time"]
        
        self.save_json(summary, "test_summary.json")
        
        if self.verbose:
            self.logger.info(f"Saved {len(self.test_results)} test results to {self.output_dir}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current testing statistics"""
        if not self.test_results:
            return {"message": "No tests run yet"}
        
        stats = {
            "total_tests": len(self.test_results),
            "recent_results": self.test_results[-5:] if self.test_results else []
        }
        
        return stats