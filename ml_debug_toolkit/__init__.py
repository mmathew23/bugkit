"""
ML Debug Toolkit - Comprehensive debugging and troubleshooting toolkit for ML frameworks
"""

__version__ = "0.1.0"
__author__ = "ML Debug Toolkit Contributors"

# Core utilities
from .core.logger import IOLogger
from .core.debug_inserter import DebugInserter

# Testing and comparison
from .testing.tensor_compare import TensorComparer
from .testing.runner import TestRunner
from .testing.differ import TrainingDiffer

# Tracing and profiling
from .tracing.chrome_tracer import ChromeTracer
from .tracing.trace_parser import TraceParser
from .tracing.trace_comparer import TraceComparer

# Analysis tools
from .analysis.loss_logger import LossLogger
from .analysis.loss_analyzer import LossAnalyzer

# Framework-specific utilities
from .frameworks.storage import DiskTensorStorage, MultiDtypeComparer
from .frameworks.huggingface import (
    HuggingFaceDebugger, 
    auto_debug_model, 
    benchmark_model, 
    compare_models,
    benchmark_training_step,
    analyze_attention_patterns,
    compare_layer_outputs,
    profile_memory_usage,
    PEFTDebugger,
    auto_peft_debug,
    TRLDebugger,
    auto_trl_debug,
    AccelerateDebugger,
    auto_accelerate_debug,
)
from .frameworks.pytorch import (
    PyTorchDebugger,
    auto_debug_module,
    profile_forward_pass,
    profile_backward_pass,
    compare_optimizers,
    GradientHook,
    ActivationHook,
    hook_model_layers,
    CUDADebugger,
    profile_cuda_operation,
    auto_cuda_debug,
    TritonDebugger,
    profile_triton_kernel,
    auto_triton_debug,
    BitsAndBytesDebugger,
    auto_quantization_debug,
    DistributedDebugger,
    auto_distributed_debug,
)
# Capture and comparison utilities
from .capture import (
    capture_forward_pass_non_streaming,
    capture_forward_pass,
    compare_captures_streaming,
    load_capture_any,
    compare_captures,
    summarize_compare_csv,
)


__all__ = [
    # Core utilities
    "IOLogger",
    "DebugInserter", 
    
    # Testing and comparison
    "TensorComparer",
    "TestRunner",
    "TrainingDiffer",
    
    # Tracing and profiling
    "ChromeTracer",
    "TraceParser", 
    "TraceComparer",
    
    # Analysis tools
    "LossLogger",
    "LossAnalyzer",
    
    # Storage and multi-dtype
    "DiskTensorStorage",
    "MultiDtypeComparer",
    
    # HuggingFace utilities
    "HuggingFaceDebugger",
    "auto_debug_model",
    "benchmark_model", 
    "compare_models",
    "benchmark_training_step",
    "analyze_attention_patterns",
    "compare_layer_outputs",
    "profile_memory_usage",
    "PEFTDebugger",
    "auto_peft_debug",
    "TRLDebugger",
    "auto_trl_debug",
    "AccelerateDebugger",
    "auto_accelerate_debug",
    
    # PyTorch utilities
    "PyTorchDebugger",
    "auto_debug_module",
    "profile_forward_pass",
    "profile_backward_pass", 
    "compare_optimizers",
    "GradientHook",
    "ActivationHook",
    "hook_model_layers",
    
    # CUDA utilities
    "CUDADebugger",
    "profile_cuda_operation",
    "auto_cuda_debug",
    
    # Triton utilities
    "TritonDebugger",
    "profile_triton_kernel", 
    "auto_triton_debug",
    
    # BitsAndBytes utilities
    "BitsAndBytesDebugger",
    "auto_quantization_debug",
    
    # Distributed utilities
    "DistributedDebugger",
    "auto_distributed_debug",
    
    # Capture and comparison utilities
    "capture_forward_pass_non_streaming",
    "capture_forward_pass",
    "compare_captures_streaming",
    "load_capture_any",
    "compare_captures",
    "summarize_compare_csv",
]