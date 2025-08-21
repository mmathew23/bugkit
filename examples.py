#!/usr/bin/env python3
"""
ML Debug Toolkit - Usage Examples
Demonstrates key functionality across different frameworks
"""

import torch
import torch.nn as nn
import numpy as np

print("🚀 ML Debug Toolkit - Usage Examples")
print("=" * 50)

def example_1_pytorch_debugging():
    """Example 1: Basic PyTorch Model Debugging"""
    print("\n📱 Example 1: PyTorch Model Debugging")
    print("-" * 40)
    
    from ml_debug_toolkit import auto_debug_module, profile_forward_pass
    
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(50, 10),
        nn.Softmax(dim=1)
    )
    
    # Debug the model
    debugger = auto_debug_module(model, "simple_classifier", verbose=False)
    model_debug = debugger.debug_model(model)
    
    print(f"✓ Model Parameters: {model_debug['basic_info']['total_parameters']:,}")
    print(f"✓ Architecture Depth: {model_debug['architecture_analysis']['depth_analysis']['max_depth']}")
    
    # Profile performance
    inputs = torch.randn(64, 100)
    profile_results = profile_forward_pass(model, inputs, num_runs=50, warmup_runs=5)
    
    print(f"✓ Average Inference: {profile_results['timing']['mean_ms']:.2f}ms")
    print(f"✓ Throughput: {profile_results['throughput']['samples_per_second']:.0f} samples/sec")
    
    debugger.disable()

def example_2_tensor_operations():
    """Example 2: Tensor Comparison and Storage"""
    print("\n🔍 Example 2: Tensor Comparison & Storage")
    print("-" * 40)
    
    from ml_debug_toolkit import TensorComparer, DiskTensorStorage
    
    # Create test tensors
    tensor1 = torch.randn(500, 500)
    tensor2 = tensor1 + torch.randn(500, 500) * 0.01  # Add small noise
    
    # Compare tensors
    comparer = TensorComparer(verbose=False)
    comparer.enable()
    
    comparison = comparer.compare(tensor1, tensor2, name1="original", name2="noisy")
    match_pct = comparison["match_percentage"]
    mae = comparison["statistics"]["difference_stats"]["mean_abs_diff"]
    
    print(f"✓ Tensor Match: {match_pct:.1f}%")
    print(f"✓ Mean Abs Error: {mae:.6f}")
    
    # Test disk storage
    storage = DiskTensorStorage(verbose=False, compress=True)
    storage.enable()
    
    large_tensor = torch.randn(1000, 1000)  # ~4MB tensor
    storage_key = storage.store_tensor(large_tensor, "large_tensor")
    loaded_tensor, info = storage.load_tensor(storage_key)
    
    print(f"✓ Stored tensor: {info['size_mb']:.1f}MB")
    print(f"✓ Compression: {'Yes' if info.get('compressed', False) else 'No'}")
    print(f"✓ Load successful: {torch.allclose(large_tensor, loaded_tensor)}")
    
    storage.disable()
    comparer.disable()

def example_3_cuda_debugging():
    """Example 3: CUDA Memory Analysis"""
    print("\n🎮 Example 3: CUDA Memory Analysis")
    print("-" * 40)
    
    from ml_debug_toolkit import auto_cuda_debug
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available - showing CPU analysis")
        cuda_debugger = auto_cuda_debug(verbose=False)
        info = cuda_debugger.get_cuda_info()
        print(f"✓ CUDA Available: {info['cuda_available']}")
        cuda_debugger.disable()
        return
    
    # CUDA debugging
    cuda_debugger = auto_cuda_debug(verbose=False)
    
    # Take memory snapshot
    cuda_debugger.snapshot_memory("before_allocation")
    
    # Allocate some GPU memory
    gpu_tensors = []
    for i in range(3):
        tensor = torch.randn(1000, 1000, device='cuda')
        gpu_tensors.append(tensor)
    
    cuda_debugger.snapshot_memory("after_allocation")
    
    # Analyze memory usage
    comparison = cuda_debugger.compare_memory_snapshots("before_allocation", "after_allocation")
    if "device_comparisons" in comparison:
        for device, stats in comparison["device_comparisons"].items():
            allocated_mb = stats["allocated_diff"] / (1024 * 1024)  # Convert bytes to MB
            print(f"✓ {device}: {allocated_mb:.1f}MB allocated")
    
    # Memory fragmentation analysis
    fragmentation = cuda_debugger.analyze_memory_fragmentation()
    if "device_analyses" in fragmentation:
        for device, analysis in fragmentation["device_analyses"].items():
            frag_ratio = analysis["external_fragmentation_ratio"]
            print(f"✓ {device}: {frag_ratio:.1%} fragmentation")
    
    cuda_debugger.disable()

def example_4_huggingface_debugging():
    """Example 4: HuggingFace Model Analysis"""
    print("\n🤗 Example 4: HuggingFace-Style Model Analysis")
    print("-" * 40)
    
    from ml_debug_toolkit import HuggingFaceDebugger
    
    # Create a transformer-like model
    model = nn.Sequential(
        nn.Embedding(1000, 128),
        nn.TransformerEncoderLayer(d_model=128, nhead=8, batch_first=True),
        nn.Linear(128, 2)
    )
    
    # Setup HF-style debugging
    hf_debugger = HuggingFaceDebugger(
        model, 
        capture_level="layer",
        capture_attention=False,  # Skip attention for this simple example
        verbose=False
    )
    hf_debugger.enable()
    
    # Test with dummy data
    batch_size, seq_len = 4, 16
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    # Forward pass - data is captured automatically
    embeddings = model[0](input_ids)  # Embedding layer
    transformer_out = model[1](embeddings)  # Transformer layer
    logits = model[2](transformer_out.mean(dim=1))  # Classification head
    
    # Get analysis
    analysis = hf_debugger.get_analysis()
    print(f"✓ Model Architecture: {analysis['model_info']['architecture']}")
    print(f"✓ Total Layers: {len(analysis['model_info']['layers'])}")
    print(f"✓ Attention Layers: {len(analysis['model_info']['attention_layers'])}")
    
    hf_debugger.disable()

def example_5_distributed_analysis():
    """Example 5: Distributed Training Analysis"""
    print("\n🌐 Example 5: Distributed Training Analysis")
    print("-" * 40)
    
    from ml_debug_toolkit import auto_distributed_debug
    
    # Setup distributed debugging (works even without actual distributed setup)
    dist_debugger = auto_distributed_debug(verbose=False)
    
    # Analyze distributed environment
    setup_info = dist_debugger.distributed_info
    print(f"✓ Distributed Available: {setup_info['distributed_available']}")
    print(f"✓ Distributed Active: {setup_info['distributed_initialized']}")
    print(f"✓ GPU Count: {setup_info['device_count']}")
    
    # Demonstrate load balance analysis (simulated)
    # In real usage, you'd gather this from actual distributed ranks
    batch_sizes = {0: 32, 1: 30, 2: 34, 3: 28}  # Simulated uneven batches
    processing_times = {0: 0.12, 1: 0.15, 2: 0.11, 3: 0.18}  # Simulated times
    
    load_analysis = dist_debugger.monitor_load_balance(
        batch_sizes, processing_times, "simulated_analysis"
    )
    
    if "load_balance_metrics" in load_analysis:
        efficiency = load_analysis["load_balance_metrics"]["parallel_efficiency"]
        print(f"✓ Parallel Efficiency: {efficiency:.1f}%")
        
        if load_analysis["recommendations"]:
            print("📝 Recommendations:")
            for rec in load_analysis["recommendations"][:2]:  # Show first 2
                print(f"   - {rec}")
    
    dist_debugger.disable()

def example_6_multi_dtype_analysis():
    """Example 6: Multi-dtype Precision Analysis"""
    print("\n🎯 Example 6: Multi-dtype Precision Analysis")
    print("-" * 40)
    
    from ml_debug_toolkit import MultiDtypeComparer
    
    # Create test data
    base_tensor = torch.randn(100, 100, dtype=torch.float32)
    
    # Setup multi-dtype comparison
    comparer = MultiDtypeComparer(
        comparison_dtypes=["float32", "float16", "bfloat16"],
        verbose=False
    )
    comparer.enable()
    
    # Compare across dtypes
    comparison = comparer.compare_dtypes(base_tensor, "test_weights")
    
    print("✓ Precision Analysis:")
    for dtype, stats in comparison["dtype_comparisons"].items():
        if "error_metrics" in stats:
            mse = stats["error_metrics"]["mse"]
            size_mb = stats["memory_info"]["size_mb"]
            print(f"   {dtype:>8}: MSE={mse:.2e}, Size={size_mb:.1f}MB")
    
    comparer.disable()

def main():
    """Run all examples"""
    examples = [
        example_1_pytorch_debugging,
        example_2_tensor_operations,
        example_3_cuda_debugging,
        example_4_huggingface_debugging,
        example_5_distributed_analysis,
        example_6_multi_dtype_analysis,
    ]
    
    for i, example in enumerate(examples, 1):
        try:
            example()
        except Exception as e:
            print(f"⚠️  Example {i} encountered an issue: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 All examples completed!")
    print("\nNext steps:")
    print("  - Check the generated debug_output/ directories")
    print("  - Try the examples with your own models")
    print("  - Explore advanced features in the README")
    print("\nHappy debugging! 🐛→✨")

if __name__ == "__main__":
    main()