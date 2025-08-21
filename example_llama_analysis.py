#!/usr/bin/env python3
"""
ML Debug Toolkit - Comprehensive Llama Model Analysis Example

This script demonstrates advanced usage by:
1. Loading the Llama-3.2-1B-Instruct model from HuggingFace
2. Extracting and analyzing layers 2 and 3 individually
3. Feeding identical inputs to both layers
4. Benchmarking layer performance
5. Comparing outputs using all comparison tooling
6. Demonstrating multi-dtype analysis
7. Showcasing CUDA profiling and memory analysis
8. Using disk storage for large tensors

Usage:
    python example_llama_analysis.py
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import time

# Import ML Debug Toolkit
from ml_debug_toolkit import (
    auto_debug_model, 
    auto_cuda_debug,
    TensorComparer, 
    DiskTensorStorage,
    MultiDtypeComparer,
    ChromeTracer,
    profile_forward_pass
)

def load_llama_model():
    """Load Llama model and extract specific layers"""
    print("🦙 Loading Llama-3.2-1B-Instruct model...")
    
    try:
        from transformers import AutoModel, AutoTokenizer, AutoConfig
        
        model_name = "unsloth/Llama-3.2-1B-Instruct"
        
        # Load model configuration first to check architecture
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        print(f"✓ Model config loaded: {config.num_hidden_layers} layers")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        model = AutoModel.from_pretrained(
            model_name, 
            torch_dtype=torch.float32,  # Use fp32 for better analysis
            device_map="auto" if torch.cuda.is_available() else "cpu",
            trust_remote_code=True
        )
        
        print(f"✓ Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        return model, tokenizer, config
        
    except Exception as e:
        print(f"⚠️  Failed to load Llama model: {e}")
        print("Creating a mock transformer model for demonstration...")
        
        # Create a mock transformer model for demonstration
        class MockLlamaLayer(nn.Module):
            def __init__(self, hidden_size=2048, num_heads=32):
                super().__init__()
                self.self_attn = nn.MultiheadAttention(
                    hidden_size, num_heads, batch_first=True
                )
                self.mlp = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size * 4),
                    nn.GELU(),
                    nn.Linear(hidden_size * 4, hidden_size)
                )
                self.input_layernorm = nn.LayerNorm(hidden_size)
                self.post_attention_layernorm = nn.LayerNorm(hidden_size)
                
            def forward(self, hidden_states):
                # Self attention
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)
                attn_output, _ = self.self_attn(hidden_states, hidden_states, hidden_states)
                hidden_states = residual + attn_output
                
                # MLP
                residual = hidden_states
                hidden_states = self.post_attention_layernorm(hidden_states)
                hidden_states = self.mlp(hidden_states)
                hidden_states = residual + hidden_states
                
                return hidden_states
        
        class MockTokenizer:
            def __init__(self):
                self.pad_token = "<pad>"
                self.vocab_size = 32000
                
            def __call__(self, text, return_tensors=None, padding=True, truncation=True, max_length=512):
                # Simple mock tokenization
                tokens = torch.randint(1, self.vocab_size, (1, min(len(text.split()), max_length)))
                return {"input_ids": tokens, "attention_mask": torch.ones_like(tokens)}
        
        # Create mock model with embedding and layers
        hidden_size = 2048
        mock_model = nn.ModuleDict({
            "embed_tokens": nn.Embedding(32000, hidden_size),
            "layers": nn.ModuleList([
                MockLlamaLayer(hidden_size) for _ in range(6)  # 6 layers for demo
            ])
        })
        
        class MockConfig:
            num_hidden_layers = 6
            hidden_size = hidden_size
        
        return mock_model, MockTokenizer(), MockConfig()

def extract_specific_layers(model, layer_indices=[2, 3]):
    """Extract specific layers from the model"""
    print(f"\n🔍 Extracting layers {layer_indices}...")
    
    extracted_layers = {}
    
    try:
        # Try to access transformer layers (common patterns)
        if hasattr(model, 'layers'):
            layers = model.layers
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            layers = model.transformer.h
        elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layers = model.model.layers
        else:
            # Fallback for mock model
            layers = model.get('layers', None)
            if layers is None:
                raise AttributeError("Could not find transformer layers")
        
        for idx in layer_indices:
            if idx < len(layers):
                extracted_layers[f"layer_{idx}"] = layers[idx]
                print(f"✓ Extracted layer {idx}: {type(layers[idx]).__name__}")
            else:
                print(f"⚠️  Layer {idx} not found (model has {len(layers)} layers)")
        
        return extracted_layers
        
    except Exception as e:
        print(f"⚠️  Failed to extract layers: {e}")
        return {}

def prepare_test_inputs(tokenizer, device="cpu"):
    """Prepare test inputs for layer analysis"""
    print(f"\n📝 Preparing test inputs...")
    
    # Test prompts of varying complexity
    test_prompts = [
        "The quick brown fox jumps over the lazy dog.",
        "In the realm of artificial intelligence, large language models have revolutionized natural language processing.",
        "What is the meaning of life, the universe, and everything according to Douglas Adams?",
    ]
    
    inputs = {}
    
    for i, prompt in enumerate(test_prompts):
        print(f"   Processing prompt {i+1}: '{prompt[:50]}...'")
        
        # Tokenize
        tokenized = tokenizer(
            prompt, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=128
        )
        
        # Move to device
        for key in tokenized:
            tokenized[key] = tokenized[key].to(device)
        
        # Create hidden states (simulate embedding output)
        batch_size, seq_len = tokenized["input_ids"].shape
        hidden_size = 2048  # Typical Llama hidden size
        
        # Generate realistic hidden states
        hidden_states = torch.randn(
            batch_size, seq_len, hidden_size, 
            device=device, dtype=torch.float32
        ) * 0.02  # Scale to reasonable range
        
        inputs[f"prompt_{i+1}"] = {
            "hidden_states": hidden_states,
            "attention_mask": tokenized.get("attention_mask"),
            "prompt_text": prompt,
            "token_count": seq_len
        }
    
    print(f"✓ Prepared {len(inputs)} test inputs")
    return inputs

def benchmark_layers(layers, inputs, cuda_debugger=None):
    """Benchmark layer performance with detailed profiling"""
    print(f"\n⚡ Benchmarking layer performance...")
    
    benchmark_results = {}
    
    for layer_name, layer in layers.items():
        print(f"\n   Benchmarking {layer_name}...")
        layer_results = {}
        
        # Set layer to eval mode
        layer.eval()
        
        for input_name, input_data in inputs.items():
            hidden_states = input_data["hidden_states"]
            
            # Take memory snapshot before
            if cuda_debugger and torch.cuda.is_available():
                cuda_debugger.snapshot_memory(f"{layer_name}_{input_name}_before")
            
            # Benchmark forward pass
            with torch.no_grad():
                # Warmup
                for _ in range(3):
                    _ = layer(hidden_states)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                
                # Actual benchmarking
                start_time = time.time()
                times = []
                
                for run in range(10):
                    if torch.cuda.is_available():
                        start_event = torch.cuda.Event(enable_timing=True)
                        end_event = torch.cuda.Event(enable_timing=True)
                        start_event.record()
                    
                    run_start = time.time()
                    output = layer(hidden_states)
                    
                    if torch.cuda.is_available():
                        end_event.record()
                        torch.cuda.synchronize()
                        cuda_time = start_event.elapsed_time(end_event)
                        times.append(cuda_time)
                    else:
                        cpu_time = (time.time() - run_start) * 1000  # Convert to ms
                        times.append(cpu_time)
                
                # Calculate statistics
                import statistics
                layer_results[input_name] = {
                    "mean_ms": statistics.mean(times),
                    "median_ms": statistics.median(times),
                    "std_ms": statistics.stdev(times) if len(times) > 1 else 0,
                    "min_ms": min(times),
                    "max_ms": max(times),
                    "throughput_tokens_per_sec": (hidden_states.shape[1] * 1000) / statistics.mean(times),
                    "output_shape": list(output.shape),
                    "memory_mb": output.numel() * output.element_size() / (1024 * 1024)
                }
            
            # Take memory snapshot after
            if cuda_debugger and torch.cuda.is_available():
                cuda_debugger.snapshot_memory(f"{layer_name}_{input_name}_after")
                
                # Analyze memory usage
                memory_comparison = cuda_debugger.compare_memory_snapshots(
                    f"{layer_name}_{input_name}_before",
                    f"{layer_name}_{input_name}_after"
                )
                layer_results[input_name]["memory_analysis"] = memory_comparison
            
            mean_time = layer_results[input_name]["mean_ms"]
            throughput = layer_results[input_name]["throughput_tokens_per_sec"]
            print(f"      {input_name}: {mean_time:.2f}ms avg, {throughput:.0f} tokens/sec")
        
        benchmark_results[layer_name] = layer_results
    
    print("✓ Layer benchmarking completed")
    return benchmark_results

def generate_layer_outputs(layers, inputs):
    """Generate outputs from each layer for comparison"""
    print(f"\n🔄 Generating layer outputs...")
    
    layer_outputs = {}
    
    for layer_name, layer in layers.items():
        print(f"   Processing {layer_name}...")
        layer.eval()
        
        layer_outputs[layer_name] = {}
        
        with torch.no_grad():
            for input_name, input_data in inputs.items():
                hidden_states = input_data["hidden_states"]
                output = layer(hidden_states)
                
                layer_outputs[layer_name][input_name] = {
                    "output": output,
                    "input_shape": list(hidden_states.shape),
                    "output_shape": list(output.shape),
                    "prompt_text": input_data["prompt_text"]
                }
    
    print("✓ Layer outputs generated")
    return layer_outputs

def comprehensive_tensor_comparison(layer_outputs, storage):
    """Perform comprehensive tensor comparison analysis"""
    print(f"\n🔍 Comprehensive tensor comparison...")
    
    # Setup comparers
    tensor_comparer = TensorComparer(tolerance_profile="research", verbose=False)
    tensor_comparer.enable()
    
    multi_dtype_comparer = MultiDtypeComparer(
        comparison_dtypes=["float32", "float16", "bfloat16"],
        verbose=False
    )
    multi_dtype_comparer.enable()
    
    comparison_results = {}
    
    # Compare outputs between layers for each input
    layer_names = list(layer_outputs.keys())
    input_names = list(layer_outputs[layer_names[0]].keys())
    
    for input_name in input_names:
        print(f"\n   Analyzing {input_name}...")
        comparison_results[input_name] = {}
        
        # Get outputs from both layers
        outputs = {}
        for layer_name in layer_names:
            outputs[layer_name] = layer_outputs[layer_name][input_name]["output"]
        
        if len(layer_names) >= 2:
            layer1, layer2 = layer_names[0], layer_names[1]
            output1, output2 = outputs[layer1], outputs[layer2]
            
            # Store tensors to disk for large tensor analysis
            storage_key1 = storage.store_tensor(output1, f"{layer1}_{input_name}")
            storage_key2 = storage.store_tensor(output2, f"{layer2}_{input_name}")
            
            print(f"      Stored tensors: {storage_key1[:8]}... and {storage_key2[:8]}...")
            
            # Tensor comparison
            comparison = tensor_comparer.compare(
                output1, output2,
                name1=f"{layer1}_{input_name}",
                name2=f"{layer2}_{input_name}"
            )
            
            comparison_results[input_name]["tensor_comparison"] = {
                "match_percentage": comparison["match_percentage"],
                "mean_abs_diff": comparison["statistics"]["difference_stats"]["mean_abs_diff"],
                "max_abs_diff": comparison["statistics"]["difference_stats"]["max_abs_diff"],
                "l2_norm": comparison["statistics"]["difference_stats"]["l2_norm"],
                "cosine_similarity": comparison.get("cosine_similarity", "N/A"),
                "storage_keys": [storage_key1, storage_key2]
            }
            
            # Multi-dtype analysis on first layer output
            dtype_comparison = multi_dtype_comparer.compare_across_dtypes(
                output1, f"{layer1}_{input_name}"
            )
            
            comparison_results[input_name]["dtype_analysis"] = {
                "precision_comparison": {},
                "memory_usage": {}
            }
            
            for dtype, stats in dtype_comparison["dtype_comparisons"].items():
                if "error_metrics" in stats:
                    comparison_results[input_name]["dtype_analysis"]["precision_comparison"][dtype] = {
                        "mse": stats["error_metrics"]["mse"],
                        "mae": stats["error_metrics"]["mae"]
                    }
                if "memory_info" in stats:
                    comparison_results[input_name]["dtype_analysis"]["memory_usage"][dtype] = {
                        "size_mb": stats["memory_info"]["size_mb"]
                    }
            
            # Print summary
            match_pct = comparison_results[input_name]["tensor_comparison"]["match_percentage"]
            mae = comparison_results[input_name]["tensor_comparison"]["mean_abs_diff"]
            print(f"      Layer comparison: {match_pct:.2f}% match, MAE={mae:.6f}")
            
            # Print dtype analysis summary
            dtype_analysis = comparison_results[input_name]["dtype_analysis"]["precision_comparison"]
            if dtype_analysis:
                print(f"      Precision analysis: {len(dtype_analysis)} dtypes compared")
                for dtype, metrics in list(dtype_analysis.items())[:2]:  # Show first 2
                    print(f"        {dtype}: MSE={metrics['mse']:.2e}")
    
    tensor_comparer.disable()
    multi_dtype_comparer.disable()
    
    print("✓ Comprehensive tensor comparison completed")
    return comparison_results

def analyze_attention_patterns(layer_outputs):
    """Analyze attention patterns if available"""
    print(f"\n👁️  Analyzing attention patterns...")
    
    attention_analysis = {}
    
    for layer_name, layer_data in layer_outputs.items():
        attention_analysis[layer_name] = {}
        
        for input_name, output_data in layer_data.items():
            output = output_data["output"]
            
            # Simple attention pattern analysis
            # For real attention analysis, you'd need access to attention weights
            attention_analysis[layer_name][input_name] = {
                "activation_statistics": {
                    "mean": float(torch.mean(output)),
                    "std": float(torch.std(output)),
                    "max": float(torch.max(output)),
                    "min": float(torch.min(output)),
                    "sparsity": float(torch.sum(torch.abs(output) < 1e-6)) / output.numel()
                },
                "sequence_analysis": {
                    "sequence_length": output.shape[1],
                    "hidden_dim": output.shape[2],
                    "mean_per_position": torch.mean(output, dim=[0, 2]).tolist()[:5]  # First 5 positions
                }
            }
    
    print("✓ Attention pattern analysis completed")
    return attention_analysis

def generate_performance_report(benchmark_results, comparison_results, attention_analysis):
    """Generate comprehensive performance report"""
    print(f"\n📊 Generating performance report...")
    
    report = {
        "summary": {
            "layers_analyzed": list(benchmark_results.keys()),
            "inputs_tested": len(list(benchmark_results.values())[0]),
            "total_comparisons": len(comparison_results),
            "timestamp": time.time()
        },
        "performance_analysis": {},
        "comparison_analysis": {},
        "attention_analysis": attention_analysis,
        "recommendations": []
    }
    
    # Performance analysis
    for layer_name, layer_data in benchmark_results.items():
        layer_perf = {}
        for input_name, metrics in layer_data.items():
            layer_perf[input_name] = {
                "inference_time_ms": metrics["mean_ms"],
                "throughput_tokens_sec": metrics["throughput_tokens_per_sec"],
                "memory_usage_mb": metrics["memory_mb"],
                "stability": "stable" if metrics["std_ms"] < metrics["mean_ms"] * 0.1 else "variable"
            }
        report["performance_analysis"][layer_name] = layer_perf
    
    # Comparison analysis summary
    total_comparisons = 0
    high_similarity_count = 0
    
    for input_name, comp_data in comparison_results.items():
        if "tensor_comparison" in comp_data:
            total_comparisons += 1
            match_pct = comp_data["tensor_comparison"]["match_percentage"]
            if match_pct > 90:
                high_similarity_count += 1
            
            report["comparison_analysis"][input_name] = {
                "layer_similarity": match_pct,
                "mean_absolute_error": comp_data["tensor_comparison"]["mean_abs_diff"],
                "precision_analysis": comp_data.get("dtype_analysis", {})
            }
    
    # Generate recommendations
    if total_comparisons > 0:
        similarity_ratio = high_similarity_count / total_comparisons
        if similarity_ratio > 0.8:
            report["recommendations"].append("Layers show high similarity - consider layer sharing or pruning")
        elif similarity_ratio < 0.3:
            report["recommendations"].append("Layers show diverse representations - good for model expressiveness")
    
    # Performance recommendations
    avg_times = []
    for layer_data in benchmark_results.values():
        for metrics in layer_data.values():
            avg_times.append(metrics["mean_ms"])
    
    if avg_times:
        overall_avg = sum(avg_times) / len(avg_times)
        if overall_avg > 50:
            report["recommendations"].append("Consider optimization - average inference time is high")
        
        time_variance = max(avg_times) / min(avg_times) if min(avg_times) > 0 else 1
        if time_variance > 2:
            report["recommendations"].append("High performance variance between inputs - check input preprocessing")
    
    print("✓ Performance report generated")
    return report

def save_analysis_results(report, comparison_results, output_dir="llama_analysis"):
    """Save analysis results to files"""
    print(f"\n💾 Saving analysis results to {output_dir}/...")
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    import json
    
    # Save main report
    with open(output_path / "performance_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    # Save detailed comparison results
    with open(output_path / "detailed_comparisons.json", "w") as f:
        json.dump(comparison_results, f, indent=2, default=str)
    
    # Generate human-readable summary
    summary_text = f"""
ML Debug Toolkit - Llama Layer Analysis Summary
===============================================

Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Layers Analyzed
{', '.join(report['summary']['layers_analyzed'])}

## Performance Summary
"""
    
    for layer_name, layer_perf in report["performance_analysis"].items():
        summary_text += f"\n### {layer_name}\n"
        for input_name, metrics in layer_perf.items():
            summary_text += f"- {input_name}: {metrics['inference_time_ms']:.2f}ms, {metrics['throughput_tokens_sec']:.0f} tokens/sec\n"
    
    summary_text += "\n## Layer Comparison Summary\n"
    for input_name, comp_data in report["comparison_analysis"].items():
        similarity = comp_data["layer_similarity"]
        mae = comp_data["mean_absolute_error"]
        summary_text += f"- {input_name}: {similarity:.1f}% similarity, MAE={mae:.6f}\n"
    
    if report["recommendations"]:
        summary_text += "\n## Recommendations\n"
        for rec in report["recommendations"]:
            summary_text += f"- {rec}\n"
    
    with open(output_path / "analysis_summary.txt", "w") as f:
        f.write(summary_text)
    
    print(f"✓ Results saved to {output_path}/")
    print(f"   - performance_report.json: Detailed analysis data")
    print(f"   - detailed_comparisons.json: Tensor comparison results")  
    print(f"   - analysis_summary.txt: Human-readable summary")

def main():
    """Main analysis pipeline"""
    print("🚀 ML Debug Toolkit - Comprehensive Llama Layer Analysis")
    print("=" * 70)
    print()
    print("This script demonstrates advanced debugging capabilities by:")
    print("• Loading and analyzing specific Llama model layers")
    print("• Benchmarking layer performance with detailed profiling")
    print("• Comparing layer outputs using comprehensive tensor analysis")
    print("• Multi-dtype precision analysis and storage optimization")
    print("• CUDA memory profiling and optimization recommendations")
    print()
    
    try:
        # Setup debugging tools
        print("🔧 Setting up debugging environment...")
        
        # CUDA debugging setup
        cuda_debugger = None
        if torch.cuda.is_available():
            cuda_debugger = auto_cuda_debug(verbose=False)
            print("✓ CUDA debugging enabled")
        else:
            print("! CUDA not available - using CPU analysis")
        
        # Disk storage setup
        storage = DiskTensorStorage(
            storage_dir="llama_analysis/tensors",
            compress=True,
            max_memory_mb=100,  # Store tensors > 100MB to disk
            verbose=False
        )
        storage.enable()
        print("✓ Disk tensor storage enabled")
        
        # Chrome tracing setup
        tracer = ChromeTracer(
            output_dir="llama_analysis/traces",
            include_cuda=torch.cuda.is_available()
        )
        tracer.enable()
        print("✓ Chrome tracing enabled")
        
        # Load model and extract layers
        with tracer.trace("model_loading", "setup"):
            model, tokenizer, config = load_llama_model()
            layers = extract_specific_layers(model, layer_indices=[2, 3])
        
        if not layers:
            print("❌ Failed to extract layers - exiting")
            return 1
        
        # Prepare test inputs
        with tracer.trace("input_preparation", "setup"):
            device = next(iter(layers.values())).weight.device if hasattr(next(iter(layers.values())), 'weight') else 'cpu'
            inputs = prepare_test_inputs(tokenizer, device)
        
        # Benchmark layer performance
        with tracer.trace("performance_benchmarking", "analysis"):
            benchmark_results = benchmark_layers(layers, inputs, cuda_debugger)
        
        # Generate layer outputs
        with tracer.trace("output_generation", "analysis"):
            layer_outputs = generate_layer_outputs(layers, inputs)
        
        # Comprehensive tensor comparison
        with tracer.trace("tensor_comparison", "analysis"):
            comparison_results = comprehensive_tensor_comparison(layer_outputs, storage)
        
        # Attention pattern analysis
        with tracer.trace("attention_analysis", "analysis"):
            attention_analysis = analyze_attention_patterns(layer_outputs)
        
        # Generate performance report
        with tracer.trace("report_generation", "analysis"):
            report = generate_performance_report(
                benchmark_results, comparison_results, attention_analysis
            )
        
        # Save results
        with tracer.trace("result_saving", "io"):
            save_analysis_results(report, comparison_results)
        
        # Print final summary
        print("\n" + "=" * 70)
        print("🎉 Analysis Complete!")
        print("=" * 70)
        
        print(f"\n📈 Performance Summary:")
        for layer_name in report["summary"]["layers_analyzed"]:
            layer_perf = report["performance_analysis"][layer_name]
            avg_time = sum(metrics["inference_time_ms"] for metrics in layer_perf.values()) / len(layer_perf)
            avg_throughput = sum(metrics["throughput_tokens_sec"] for metrics in layer_perf.values()) / len(layer_perf)
            print(f"   {layer_name}: {avg_time:.2f}ms avg, {avg_throughput:.0f} tokens/sec")
        
        print(f"\n🔍 Comparison Summary:")
        for input_name, comp_data in report["comparison_analysis"].items():
            similarity = comp_data["layer_similarity"]
            mae = comp_data["mean_absolute_error"]
            print(f"   {input_name}: {similarity:.1f}% similarity, MAE={mae:.6f}")
        
        if report["recommendations"]:
            print(f"\n💡 Key Recommendations:")
            for rec in report["recommendations"]:
                print(f"   • {rec}")
        
        print(f"\n📁 Results saved to: llama_analysis/")
        print(f"   • Tensor storage: {storage.get_storage_stats()['total_tensors']} tensors stored")
        print(f"   • Chrome traces: Available for visualization")
        print(f"   • Detailed reports: JSON and text formats")
        
        # Cleanup
        storage.disable()
        tracer.disable()
        if cuda_debugger:
            cuda_debugger.disable()
        
        print("\n🎯 Analysis demonstrates the full power of ML Debug Toolkit!")
        print("   Check the generated files for detailed insights.")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())