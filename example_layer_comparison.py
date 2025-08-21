#!/usr/bin/env python3
"""
ML Debug Toolkit - Focused Layer Comparison Example

A streamlined example that demonstrates:
1. Creating two similar transformer layers
2. Feeding identical inputs to both layers
3. Benchmarking performance differences
4. Comprehensive tensor comparison analysis
5. Multi-dtype precision analysis
6. Disk storage for large tensors

This is a more accessible version that works without downloading large models.

Usage:
    python example_layer_comparison.py
"""

import torch
import torch.nn as nn
import time
from pathlib import Path

# Import ML Debug Toolkit
from ml_debug_toolkit import (
    TensorComparer, 
    DiskTensorStorage,
    MultiDtypeComparer,
    profile_forward_pass,
    auto_cuda_debug
)

def create_transformer_layers():
    """Create two similar transformer layers for comparison"""
    print("🏗️  Creating transformer layers...")
    
    # Layer configuration
    hidden_size = 768
    num_heads = 12
    intermediate_size = 3072
    
    class TransformerLayer(nn.Module):
        def __init__(self, hidden_size, num_heads, intermediate_size, dropout=0.1):
            super().__init__()
            self.self_attention = nn.MultiheadAttention(
                hidden_size, num_heads, dropout=dropout, batch_first=True
            )
            self.norm1 = nn.LayerNorm(hidden_size)
            self.norm2 = nn.LayerNorm(hidden_size)
            
            # Feed-forward network
            self.ffn = nn.Sequential(
                nn.Linear(hidden_size, intermediate_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(intermediate_size, hidden_size),
                nn.Dropout(dropout)
            )
            
        def forward(self, x):
            # Self-attention with residual connection
            attn_output, _ = self.self_attention(x, x, x)
            x = self.norm1(x + attn_output)
            
            # Feed-forward with residual connection
            ffn_output = self.ffn(x)
            x = self.norm2(x + ffn_output)
            
            return x
    
    # Create Layer 2 and Layer 3 (with slight differences)
    layer_2 = TransformerLayer(hidden_size, num_heads, intermediate_size, dropout=0.1)
    layer_3 = TransformerLayer(hidden_size, num_heads, intermediate_size, dropout=0.15)  # Different dropout
    
    # Initialize Layer 3 with similar weights to Layer 2 for interesting comparison
    with torch.no_grad():
        for p2, p3 in zip(layer_2.parameters(), layer_3.parameters()):
            if p2.shape == p3.shape:
                # Add small random variation
                p3.data = p2.data + torch.randn_like(p2.data) * 0.01
    
    print(f"✓ Layer 2: {sum(p.numel() for p in layer_2.parameters()):,} parameters")
    print(f"✓ Layer 3: {sum(p.numel() for p in layer_3.parameters()):,} parameters")
    
    return {"layer_2": layer_2, "layer_3": layer_3}

def create_test_inputs(batch_sizes=[8, 16, 32], seq_lengths=[64, 128, 256], hidden_size=768):
    """Create test inputs of varying sizes"""
    print(f"\n📝 Creating test inputs...")
    
    inputs = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    for i, (batch_size, seq_len) in enumerate(zip(batch_sizes, seq_lengths)):
        input_tensor = torch.randn(
            batch_size, seq_len, hidden_size, 
            device=device, dtype=torch.float32
        ) * 0.02  # Scale to reasonable range
        
        input_name = f"input_{i+1}_b{batch_size}_s{seq_len}"
        inputs[input_name] = {
            "tensor": input_tensor,
            "batch_size": batch_size,
            "seq_length": seq_len,
            "description": f"Batch={batch_size}, Seq={seq_len}"
        }
        
        print(f"   {input_name}: {input_tensor.shape} ({input_tensor.numel() * 4 / 1024**2:.1f}MB)")
    
    print(f"✓ Created {len(inputs)} test inputs")
    return inputs

def benchmark_layer_performance(layers, inputs):
    """Benchmark performance of both layers"""
    print(f"\n⚡ Benchmarking layer performance...")
    
    results = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Move layers to device
    for name, layer in layers.items():
        layers[name] = layer.to(device).eval()
    
    for layer_name, layer in layers.items():
        print(f"\n   Benchmarking {layer_name}...")
        layer_results = {}
        
        for input_name, input_data in inputs.items():
            # Use the ML Debug Toolkit profiling function
            input_tensor = input_data["tensor"]
            
            profile_result = profile_forward_pass(
                layer, input_tensor, 
                num_runs=50, warmup_runs=10
            )
            
            layer_results[input_name] = {
                "mean_ms": profile_result["timing"]["mean_ms"],
                "std_ms": profile_result["timing"]["std_ms"],
                "min_ms": profile_result["timing"]["min_ms"],
                "max_ms": profile_result["timing"]["max_ms"],
                "throughput": profile_result["throughput"]["samples_per_second"],
                "tokens_per_sec": profile_result["throughput"]["samples_per_second"] * input_data["seq_length"],
                "memory_info": profile_result.get("memory", {}),
                "stability": "stable" if profile_result["timing"]["std_ms"] < profile_result["timing"]["mean_ms"] * 0.15 else "variable"
            }
            
            print(f"      {input_name}: {layer_results[input_name]['mean_ms']:.2f}ms ± {layer_results[input_name]['std_ms']:.2f}ms")
        
        results[layer_name] = layer_results
    
    print("✓ Performance benchmarking completed")
    return results

def generate_layer_outputs(layers, inputs):
    """Generate outputs from both layers for the same inputs"""
    print(f"\n🔄 Generating layer outputs...")
    
    outputs = {}
    
    for layer_name, layer in layers.items():
        print(f"   Processing {layer_name}...")
        layer_outputs = {}
        
        with torch.no_grad():
            for input_name, input_data in inputs.items():
                output = layer(input_data["tensor"])
                layer_outputs[input_name] = output
                
                print(f"      {input_name}: {list(input_data['tensor'].shape)} → {list(output.shape)}")
        
        outputs[layer_name] = layer_outputs
    
    print("✓ Layer outputs generated")
    return outputs

def comprehensive_comparison_analysis(outputs, inputs, storage):
    """Perform comprehensive comparison analysis"""
    print(f"\n🔍 Comprehensive comparison analysis...")
    
    # Setup comparison tools
    tensor_comparer = TensorComparer(
        rtol=1e-6,  # Research-level tolerance
        atol=1e-8,
        verbose=False
    )
    tensor_comparer.enable()
    
    multi_dtype_comparer = MultiDtypeComparer(
        comparison_dtypes=["float32", "float16", "bfloat16"],
        verbose=False
    )
    multi_dtype_comparer.enable()
    
    analysis_results = {}
    layer_names = list(outputs.keys())
    
    if len(layer_names) < 2:
        print("⚠️  Need at least 2 layers for comparison")
        return {}
    
    layer1, layer2 = layer_names[0], layer_names[1]
    
    for input_name in inputs.keys():
        print(f"\n   Analyzing {input_name}...")
        
        output1 = outputs[layer1][input_name]
        output2 = outputs[layer2][input_name]
        
        # Store tensors to disk (demonstrate disk storage capability)
        storage_key1 = storage.store_tensor(output1, f"{layer1}_{input_name}")
        storage_key2 = storage.store_tensor(output2, f"{layer2}_{input_name}")
        
        # Basic tensor comparison
        comparison = tensor_comparer.compare(
            output1, output2,
            name1=f"{layer1}_{input_name}",
            name2=f"{layer2}_{input_name}"
        )
        
        # Multi-dtype analysis on first layer output
        dtype_analysis = multi_dtype_comparer.compare_across_dtypes(output1, f"{layer1}_{input_name}")
        
        # Comprehensive analysis
        analysis_results[input_name] = {
            "basic_comparison": {
                "match_percentage": comparison["match_percentage"],
                "mean_abs_diff": comparison["statistics"]["difference_stats"]["mean_abs_diff"],
                "max_abs_diff": comparison["statistics"]["difference_stats"]["max_abs_diff"],
                "relative_error": comparison["statistics"]["difference_stats"]["mean_abs_diff"] / (torch.mean(torch.abs(output1)).item() + 1e-8),
                "cosine_similarity": float(torch.cosine_similarity(
                    output1.flatten(), output2.flatten(), dim=0
                ))
            },
            "dtype_analysis": {},
            "statistical_analysis": {
                "layer1_stats": {
                    "mean": float(torch.mean(output1)),
                    "std": float(torch.std(output1)),
                    "min": float(torch.min(output1)),
                    "max": float(torch.max(output1)),
                    "sparsity": float(torch.sum(torch.abs(output1) < 1e-6)) / output1.numel()
                },
                "layer2_stats": {
                    "mean": float(torch.mean(output2)),
                    "std": float(torch.std(output2)),
                    "min": float(torch.min(output2)),
                    "max": float(torch.max(output2)),
                    "sparsity": float(torch.sum(torch.abs(output2) < 1e-6)) / output2.numel()
                }
            },
            "storage_info": {
                "storage_keys": [storage_key1, storage_key2],
                "original_size_mb": output1.numel() * 4 / (1024**2),
                "compressed": True
            }
        }
        
        # Process dtype analysis results
        for dtype, stats in dtype_analysis["dtype_comparisons"].items():
            if "error_metrics" in stats:
                analysis_results[input_name]["dtype_analysis"][dtype] = {
                    "mse": stats["error_metrics"]["mse"],
                    "mae": stats["error_metrics"]["mae"],
                    "size_mb": stats["memory_info"]["size_mb"] if "memory_info" in stats else 0
                }
        
        # Print summary for this input
        match_pct = analysis_results[input_name]["basic_comparison"]["match_percentage"]
        mae = analysis_results[input_name]["basic_comparison"]["mean_abs_diff"]
        cosine_sim = analysis_results[input_name]["basic_comparison"]["cosine_similarity"]
        
        print(f"      Match: {match_pct:.2f}%, MAE: {mae:.6f}, Cosine: {cosine_sim:.4f}")
        
        # Show dtype comparison
        dtype_results = analysis_results[input_name]["dtype_analysis"]
        if dtype_results:
            print(f"      Precision analysis: {len(dtype_results)} dtypes")
            for dtype, metrics in list(dtype_results.items())[:2]:
                print(f"        {dtype}: MSE={metrics['mse']:.2e}, Size={metrics['size_mb']:.1f}MB")
    
    tensor_comparer.disable()
    multi_dtype_comparer.disable()
    
    print("✓ Comprehensive comparison analysis completed")
    return analysis_results

def cuda_memory_analysis():
    """Demonstrate CUDA memory analysis if available"""
    if not torch.cuda.is_available():
        print("\n⚠️  CUDA not available - skipping memory analysis")
        return {}
    
    print(f"\n🎮 CUDA memory analysis...")
    
    cuda_debugger = auto_cuda_debug(verbose=False)
    
    # Take snapshot before creating tensors
    cuda_debugger.snapshot_memory("before_tensors")
    
    # Create some large tensors
    large_tensors = []
    for i in range(3):
        tensor = torch.randn(1024, 1024, device='cuda')
        large_tensors.append(tensor)
        print(f"   Created tensor {i+1}: {tensor.shape} ({tensor.numel() * 4 / 1024**2:.1f}MB)")
    
    # Take snapshot after
    cuda_debugger.snapshot_memory("after_tensors")
    
    # Analyze memory usage
    memory_comparison = cuda_debugger.compare_memory_snapshots("before_tensors", "after_tensors")
    
    # Memory fragmentation analysis
    fragmentation = cuda_debugger.analyze_memory_fragmentation()
    
    analysis = {
        "memory_usage": memory_comparison,
        "fragmentation": fragmentation,
        "recommendations": []
    }
    
    # Print summary
    if "device_comparisons" in memory_comparison:
        for device, stats in memory_comparison["device_comparisons"].items():
            allocated_mb = stats["allocated_diff"] / (1024 * 1024)
            print(f"   {device}: +{allocated_mb:.1f}MB allocated")
    
    if "device_analyses" in fragmentation:
        for device, frag_analysis in fragmentation["device_analyses"].items():
            frag_ratio = frag_analysis["external_fragmentation_ratio"]
            print(f"   {device}: {frag_ratio:.1%} fragmentation")
            
            if frag_ratio > 0.2:
                analysis["recommendations"].append(f"High fragmentation on {device} - consider torch.cuda.empty_cache()")
    
    cuda_debugger.disable()
    print("✓ CUDA memory analysis completed")
    return analysis

def generate_final_report(benchmark_results, comparison_results, cuda_analysis=None):
    """Generate comprehensive final report"""
    print(f"\n📊 Generating final report...")
    
    report = {
        "summary": {
            "layers_compared": list(benchmark_results.keys()),
            "inputs_tested": len(list(benchmark_results.values())[0]),
            "analysis_timestamp": time.time()
        },
        "performance_comparison": {},
        "output_analysis": comparison_results,
        "cuda_analysis": cuda_analysis or {},
        "insights": [],
        "recommendations": []
    }
    
    # Performance comparison between layers
    layer_names = list(benchmark_results.keys())
    if len(layer_names) >= 2:
        layer1, layer2 = layer_names[0], layer_names[1]
        
        for input_name in benchmark_results[layer1].keys():
            perf1 = benchmark_results[layer1][input_name]
            perf2 = benchmark_results[layer2][input_name]
            
            speedup = perf1["mean_ms"] / perf2["mean_ms"] if perf2["mean_ms"] > 0 else 1.0
            throughput_diff = perf2["throughput"] - perf1["throughput"]
            
            report["performance_comparison"][input_name] = {
                "layer1_time_ms": perf1["mean_ms"],
                "layer2_time_ms": perf2["mean_ms"],
                "speedup_ratio": speedup,
                "throughput_difference": throughput_diff,
                "faster_layer": layer1 if perf1["mean_ms"] < perf2["mean_ms"] else layer2
            }
    
    # Generate insights
    if comparison_results:
        avg_similarity = sum(
            result["basic_comparison"]["match_percentage"] 
            for result in comparison_results.values()
        ) / len(comparison_results)
        
        if avg_similarity > 95:
            report["insights"].append("Layers produce very similar outputs - potential for optimization")
        elif avg_similarity < 70:
            report["insights"].append("Layers show significant differences - good layer diversity")
        
        avg_cosine = sum(
            result["basic_comparison"]["cosine_similarity"]
            for result in comparison_results.values()
        ) / len(comparison_results)
        
        if avg_cosine > 0.98:
            report["insights"].append("High cosine similarity suggests similar representation spaces")
    
    # Performance insights
    if benchmark_results:
        all_times = []
        for layer_data in benchmark_results.values():
            for perf in layer_data.values():
                all_times.append(perf["mean_ms"])
        
        if all_times:
            time_variance = (max(all_times) - min(all_times)) / max(all_times)
            if time_variance > 0.5:
                report["insights"].append("High performance variance across inputs")
    
    # Recommendations
    if report["performance_comparison"]:
        faster_counts = {}
        for comp in report["performance_comparison"].values():
            faster_layer = comp["faster_layer"]
            faster_counts[faster_layer] = faster_counts.get(faster_layer, 0) + 1
        
        if faster_counts:
            consistently_faster = max(faster_counts, key=faster_counts.get)
            if faster_counts[consistently_faster] >= len(report["performance_comparison"]) * 0.8:
                report["recommendations"].append(f"{consistently_faster} consistently faster - consider using its architecture")
    
    print("✓ Final report generated")
    return report

def save_results(report, output_dir="layer_comparison_results"):
    """Save all results to files"""
    print(f"\n💾 Saving results to {output_dir}/...")
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    import json
    
    # Save detailed report
    with open(output_path / "analysis_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    # Generate human-readable summary
    summary = f"""
ML Debug Toolkit - Layer Comparison Analysis
==========================================

Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Layers Compared
{', '.join(report['summary']['layers_compared'])}

## Performance Summary
"""
    
    if report["performance_comparison"]:
        for input_name, comp in report["performance_comparison"].items():
            summary += f"\n{input_name}:\n"
            summary += f"  Layer 1: {comp['layer1_time_ms']:.2f}ms\n"
            summary += f"  Layer 2: {comp['layer2_time_ms']:.2f}ms\n"
            summary += f"  Faster: {comp['faster_layer']} ({comp['speedup_ratio']:.2f}x)\n"
    
    if report["output_analysis"]:
        summary += "\n## Output Similarity\n"
        for input_name, analysis in report["output_analysis"].items():
            match_pct = analysis["basic_comparison"]["match_percentage"]
            cosine_sim = analysis["basic_comparison"]["cosine_similarity"]
            summary += f"{input_name}: {match_pct:.1f}% match, {cosine_sim:.4f} cosine similarity\n"
    
    if report["insights"]:
        summary += "\n## Key Insights\n"
        for insight in report["insights"]:
            summary += f"• {insight}\n"
    
    if report["recommendations"]:
        summary += "\n## Recommendations\n"
        for rec in report["recommendations"]:
            summary += f"• {rec}\n"
    
    with open(output_path / "summary.txt", "w") as f:
        f.write(summary)
    
    print(f"✓ Results saved to {output_path}/")
    return output_path

def main():
    """Main analysis pipeline"""
    print("🚀 ML Debug Toolkit - Layer Comparison Example")
    print("=" * 60)
    print()
    print("This example demonstrates:")
    print("• Creating and comparing transformer layers")
    print("• Performance benchmarking with detailed metrics") 
    print("• Comprehensive tensor comparison analysis")
    print("• Multi-dtype precision analysis")
    print("• Disk storage for large tensors")
    print("• CUDA memory analysis (if available)")
    print()
    
    try:
        # Setup storage
        print("🔧 Setting up debugging tools...")
        storage = DiskTensorStorage(
            storage_dir="layer_comparison_results/tensors",
            compress=True,
            max_memory_mb=50,  # Store tensors > 50MB to disk
            verbose=False
        )
        storage.enable()
        print("✓ Disk tensor storage enabled")
        
        # Create transformer layers
        layers = create_transformer_layers()
        
        # Create test inputs
        inputs = create_test_inputs()
        
        # Benchmark performance
        benchmark_results = benchmark_layer_performance(layers, inputs)
        
        # Generate outputs
        layer_outputs = generate_layer_outputs(layers, inputs)
        
        # Comprehensive comparison analysis
        comparison_results = comprehensive_comparison_analysis(layer_outputs, inputs, storage)
        
        # CUDA memory analysis
        cuda_analysis = cuda_memory_analysis()
        
        # Generate final report
        report = generate_final_report(benchmark_results, comparison_results, cuda_analysis)
        
        # Save results
        output_path = save_results(report)
        
        # Print summary
        print("\n" + "=" * 60)
        print("🎉 Analysis Complete!")
        print("=" * 60)
        
        # Performance summary
        if report["performance_comparison"]:
            print(f"\n📈 Performance Summary:")
            for input_name, comp in report["performance_comparison"].items():
                faster = comp["faster_layer"]
                speedup = comp["speedup_ratio"]
                print(f"   {input_name}: {faster} is {speedup:.2f}x faster")
        
        # Similarity summary
        if comparison_results:
            print(f"\n🔍 Similarity Summary:")
            for input_name, analysis in comparison_results.items():
                match_pct = analysis["basic_comparison"]["match_percentage"]
                cosine_sim = analysis["basic_comparison"]["cosine_similarity"]
                print(f"   {input_name}: {match_pct:.1f}% match, {cosine_sim:.4f} cosine similarity")
        
        # Key insights
        if report["insights"]:
            print(f"\n💡 Key Insights:")
            for insight in report["insights"]:
                print(f"   • {insight}")
        
        # Storage summary
        storage_stats = storage.get_storage_stats()
        print(f"\n💾 Storage Summary:")
        print(f"   • {storage_stats['total_tensors']} tensors stored")
        print(f"   • {storage_stats['memory_usage_mb']:.1f}MB in memory")
        print(f"   • {storage_stats['disk_usage_mb']:.1f}MB on disk")
        
        # Cleanup
        storage.disable()
        
        print(f"\n📁 Detailed results saved to: {output_path}/")
        print("   • analysis_report.json: Complete analysis data")
        print("   • summary.txt: Human-readable summary")
        print("   • tensors/: Stored tensor data")
        
        print("\n🎯 This example showcases the comprehensive debugging capabilities")
        print("   of the ML Debug Toolkit for layer-level analysis!")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())