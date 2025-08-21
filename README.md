# ML Debug Toolkit

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

> **Comprehensive debugging and troubleshooting toolkit for ML frameworks**  
> Making ML ecosystem debugging **trivial** with minimal setup and maximum insight.

## 🚀 Quick Start

```bash
pip install -e .  # Install the package
```

**One-line debugging setup:**
```python
from ml_debug_toolkit import auto_debug_model, auto_cuda_debug

# Debug any HuggingFace model instantly
debugger = auto_debug_model(model, capture_attention=True)

# Profile CUDA operations 
cuda_debugger = auto_cuda_debug()
```

## ✨ Key Features

### 🎯 **Framework-Specific Quality of Life**
- **HuggingFace**: Auto-wrap any transformer with comprehensive I/O capture
- **PEFT**: LoRA/AdaLoRA efficiency analysis and rank optimization
- **TRL**: Reward distribution, KL divergence, and value function debugging
- **Accelerate**: Device placement analysis and distributed training profiling

### 🔧 **Multi-GPU & Distributed**
- **Gradient synchronization** profiling with AllReduce timing
- **Load balancing** analysis across ranks
- **Communication collective** benchmarking
- **Memory fragmentation** detection and analysis

### 💾 **Advanced Storage & Comparison**
- **Disk storage** for large tensors with compression
- **Multi-dtype comparison** (fp32, fp16, bfloat16, int8, etc.)
- **Quantization analysis** with BitsAndBytes integration
- **Chrome tracing** with kernel launch comparison

---

## 📖 Basic Usage

### 1. **HuggingFace Model Debugging**

```python
from ml_debug_toolkit import auto_debug_model
import torch
from transformers import AutoModel, AutoTokenizer

# Load any HuggingFace model
model = AutoModel.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# One-line debugging setup
debugger = auto_debug_model(
    model, 
    capture_level="layer",        # "model", "layer", "attention", "all"
    capture_attention=True,       # Capture attention patterns
    storage_mode="auto",          # Auto disk storage for large tensors
    verbose=True
)

# Use your model normally - everything is captured automatically
inputs = tokenizer("Hello world", return_tensors="pt")
outputs = model(**inputs)

# Get comprehensive analysis
analysis = debugger.get_analysis()
print(f"Captured {len(analysis['layer_analysis']['activation_statistics'])} layers")
print(f"Attention patterns: {len(analysis['attention_analysis'])}")

debugger.disable()  # Saves all data automatically
```

### 2. **PyTorch Model Debugging**

```python
from ml_debug_toolkit import auto_debug_module, profile_forward_pass
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 10)
)

# Debug model architecture and parameters
debugger = auto_debug_module(model, "my_model")
model_analysis = debugger.debug_model(model, analyze_architecture=True)

print(f"Total parameters: {model_analysis['basic_info']['total_parameters']:,}")
print(f"Architecture type: {model_analysis['architecture_analysis']}")

# Profile performance
inputs = torch.randn(64, 784)
profile_results = profile_forward_pass(model, inputs, num_runs=100)
print(f"Average inference time: {profile_results['timing']['mean_ms']:.2f}ms")

debugger.disable()
```

### 3. **CUDA Memory Debugging**

```python
from ml_debug_toolkit import auto_cuda_debug

# Start CUDA monitoring
cuda_debugger = auto_cuda_debug()

# Take memory snapshot before training
cuda_debugger.snapshot_memory("before_training")

# Your training code here...
model = model.cuda()
for batch in dataloader:
    # Training step
    pass

# Check for memory leaks
cuda_debugger.snapshot_memory("after_training")
leak_analysis = cuda_debugger.detect_memory_leaks(threshold_mb=50)

if leak_analysis["potential_leaks"]:
    print("⚠️ Potential memory leaks detected!")
    for leak in leak_analysis["potential_leaks"]:
        print(f"  {leak['type']}: {leak['growth_mb']:.1f}MB growth")

cuda_debugger.disable()
```

### 4. **Tensor Comparison & Storage**

```python
from ml_debug_toolkit import TensorComparer, DiskTensorStorage
import torch

# Compare tensors with detailed analysis
comparer = TensorComparer(tolerance_profile="strict")
comparer.enable()

tensor1 = torch.randn(1000, 1000)
tensor2 = tensor1 + torch.randn(1000, 1000) * 0.01  # Similar tensor

comparison = comparer.compare(tensor1, tensor2, name1="original", name2="perturbed")
print(f"Match percentage: {comparison['match_percentage']:.2f}%")
print(f"Mean absolute difference: {comparison['statistics']['difference_stats']['mean_abs_diff']:.6f}")

# Store large tensors to disk with compression
storage = DiskTensorStorage(compress=True, max_memory_mb=100)
storage.enable()

large_tensor = torch.randn(10000, 10000)  # ~400MB tensor
storage_key = storage.store_tensor(large_tensor, "large_tensor")
print(f"Stored tensor with key: {storage_key}")

# Load it back later
loaded_tensor, info = storage.load_tensor(storage_key)
print(f"Loaded tensor shape: {loaded_tensor.shape}")

storage.disable()
comparer.disable()
```

---

## 🚀 Comprehensive Examples

### **Real-World Model Analysis**

For detailed examples showing the toolkit in action:

```bash
# Comprehensive Llama model layer analysis (downloads model)
python example_llama_analysis.py

# Focused layer comparison (no downloads required)
python example_layer_comparison.py
```

**`example_llama_analysis.py`** demonstrates:
- Loading and analyzing specific Llama-3.2-1B-Instruct layers
- Layer-by-layer performance benchmarking
- Comprehensive tensor comparison across layers
- Multi-dtype precision analysis
- CUDA memory profiling and optimization
- Chrome tracing for performance visualization
- Disk storage for large tensors

**`example_layer_comparison.py`** demonstrates:
- Creating and comparing transformer layers
- Performance benchmarking with statistical analysis
- Tensor comparison with multiple tolerance profiles
- Multi-dtype analysis (fp32, fp16, bfloat16)
- CUDA memory analysis and fragmentation detection
- Comprehensive reporting and insights

Both examples generate detailed reports and demonstrate the full power of the toolkit for real debugging scenarios.

---

## 🎯 Advanced Usage

### 1. **PEFT (LoRA) Model Analysis**

```python
from ml_debug_toolkit import auto_peft_debug
from peft import LoraConfig, get_peft_model
from transformers import AutoModel

# Create PEFT model
base_model = AutoModel.from_pretrained("bert-base-uncased")
lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["query", "value"])
peft_model = get_peft_model(base_model, lora_config)

# Analyze PEFT efficiency
peft_debugger = auto_peft_debug(peft_model)

# Comprehensive efficiency analysis
efficiency = peft_debugger.analyze_adapter_efficiency()
print(f"Parameter efficiency: {efficiency['efficiency_metrics']['parameter_efficiency']:.4%}")
print(f"Memory reduction: {efficiency['efficiency_metrics']['memory_reduction_ratio']:.2%}")

# LoRA rank analysis
lora_analysis = peft_debugger.analyze_lora_ranks()
if lora_analysis.get("rank_statistics"):
    print(f"Average LoRA rank: {lora_analysis['rank_statistics']['mean_rank']:.1f}")
    print(f"Rank distribution: {lora_analysis['rank_statistics']['rank_distribution']}")

# Gradient analysis
dummy_output = peft_model(**tokenizer("test", return_tensors="pt"))
grad_analysis = peft_debugger.compare_adapter_gradients(dummy_output.last_hidden_state)

peft_debugger.disable()
```

### 2. **TRL (Reinforcement Learning) Debugging**

```python
from ml_debug_toolkit import auto_trl_debug
import torch

# Setup TRL debugging
trl_debugger = auto_trl_debug()

# Analyze reward distribution
rewards = torch.tensor([0.1, 0.5, -0.2, 0.8, 0.3, -0.1, 0.9])
reward_analysis = trl_debugger.analyze_reward_distribution(rewards, "episode_1")

print(f"Mean reward: {reward_analysis['reward_statistics']['mean']:.3f}")
print(f"Reward distribution: {reward_analysis['distribution_analysis']['distribution_type']}")

if reward_analysis["potential_issues"]:
    print("Issues detected:")
    for issue in reward_analysis["potential_issues"]:
        print(f"  - {issue}")

# KL divergence analysis
log_probs_current = torch.randn(100)
log_probs_reference = torch.randn(100)

kl_analysis = trl_debugger.analyze_kl_divergence(
    log_probs_current, log_probs_reference, 
    target_kl=0.1
)

print(f"Mean KL divergence: {kl_analysis['kl_statistics']['mean_kl']:.6f}")
if not kl_analysis["constraint_analysis"]["constraint_satisfied"]:
    print("⚠️ KL constraint violated!")

trl_debugger.disable()
```

### 3. **Multi-GPU Distributed Training**

```python
from ml_debug_toolkit import auto_distributed_debug
import torch
import torch.distributed as dist
import torch.nn as nn

# Initialize distributed training (in your training script)
# dist.init_process_group(backend="nccl")

# Setup distributed debugging
dist_debugger = auto_distributed_debug()

# Analyze distributed setup
setup_info = dist_debugger.distributed_info
print(f"Distributed backend: {setup_info.get('backend', 'Not initialized')}")
print(f"World size: {setup_info.get('world_size', 'N/A')}")

# Profile gradient synchronization
model = nn.Linear(1000, 100)
optimizer = torch.optim.Adam(model.parameters())
loss_fn = nn.MSELoss()

inputs = {"input": torch.randn(32, 1000)}
targets = torch.randn(32, 100)

sync_profile = dist_debugger.profile_gradient_sync(
    model, optimizer, loss_fn, inputs, targets
)

print(f"Forward pass: {sync_profile['timing']['forward_ms']:.2f}ms")
print(f"Backward pass: {sync_profile['timing']['backward_ms']:.2f}ms")
print(f"Sync overhead: {sync_profile['timing']['sync_overhead_ms']:.2f}ms")

# Communication profiling (if distributed is initialized)
if dist.is_initialized():
    test_tensor = torch.randn(1000, device="cuda")
    comm_profile = dist_debugger.profile_communication_collective(
        "allreduce", test_tensor, "test_allreduce"
    )
    print(f"AllReduce bandwidth: {comm_profile['bandwidth_analysis']['bandwidth_gbps']:.2f} Gbps")

dist_debugger.disable()
```

### 4. **Accelerate Integration**

```python
from ml_debug_toolkit import auto_accelerate_debug
from accelerate import Accelerator
import torch.nn as nn

# Initialize Accelerator
accelerator = Accelerator(mixed_precision="fp16")

# Setup debugging with accelerator
acc_debugger = auto_accelerate_debug(accelerator)

model = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))
optimizer = torch.optim.Adam(model.parameters())

# Prepare with accelerator
model, optimizer = accelerator.prepare(model, optimizer)

# Analyze device placement
device_analysis = acc_debugger.analyze_device_placement(model)
print(f"Model spans {len(device_analysis['model_device_mapping'])} device(s)")
print(f"Memory distribution: {device_analysis['memory_analysis']}")

# Profile training step
data_batch = {"input": torch.randn(32, 784), "labels": torch.randint(0, 10, (32,))}
loss_fn = nn.CrossEntropyLoss()

step_profile = acc_debugger.profile_training_step(
    model, optimizer, data_batch, 
    lambda outputs, batch: loss_fn(outputs, batch["labels"])
)

print(f"Total step time: {step_profile['timing_breakdown']['total_ms']:.2f}ms")
print(f"Mixed precision: {step_profile['accelerator_info']['mixed_precision']}")

acc_debugger.disable()
```

### 5. **Chrome Tracing & Performance Analysis**

```python
from ml_debug_toolkit import ChromeTracer, TraceComparer
import torch
import torch.nn as nn

# Setup Chrome tracing
tracer = ChromeTracer(output_dir="traces", include_cuda=True)
tracer.enable()

model = nn.TransformerEncoderLayer(d_model=512, nhead=8)
input_data = torch.randn(32, 100, 512)

# Trace model execution
with tracer.trace("forward_pass", "inference"):
    with tracer.trace("transformer_layer", "model"):
        output = model(input_data)

# Add custom events
tracer.add_event("data_loading", "io", 0.0, 5.5)
tracer.add_event("preprocessing", "compute", 5.5, 2.3)

tracer.disable()  # Saves Chrome trace file

# Compare two trace files
comparer = TraceComparer()
if len(tracer.events) > 0:
    # In practice, you'd compare two different trace files
    analysis = comparer.compare_traces("traces/trace_1.json", "traces/trace_2.json")
    print("Trace comparison completed")
```

### 6. **Comprehensive Multi-dtype Analysis**

```python
from ml_debug_toolkit import MultiDtypeComparer
import torch

# Compare across different precisions
comparer = MultiDtypeComparer(
    dtypes=["float32", "float16", "bfloat16", "int8"],
    output_dir="dtype_analysis"
)
comparer.enable()

# Original tensor
base_tensor = torch.randn(1000, 1000, dtype=torch.float32)

# Compare across dtypes
comparison = comparer.compare_across_dtypes(base_tensor, "weight_matrix")

print("Precision Analysis:")
for dtype, stats in comparison["dtype_comparisons"].items():
    if "error_metrics" in stats:
        print(f"  {dtype}: MSE={stats['error_metrics']['mse']:.2e}")

# Quantization scheme testing
quantization_test = comparer.test_quantization_schemes(
    base_tensor, 
    schemes=["symmetric", "asymmetric", "per_channel"]
)

print("\nQuantization Analysis:")
for scheme, results in quantization_test["scheme_results"].items():
    if "compression_ratio" in results:
        print(f"  {scheme}: {results['compression_ratio']:.1f}x compression")

comparer.disable()
```

---

## 🏗️ Architecture

```
ml_debug_toolkit/
├── core/                   # Base functionality
│   ├── base.py            # BaseDebugTool abstract class
│   ├── logger.py          # I/O logging utilities
│   └── debug_inserter.py  # AST-based debug insertion
│
├── frameworks/            # Framework-specific utilities
│   ├── huggingface/       # HuggingFace ecosystem
│   │   ├── debugger.py    # Main HF debugger
│   │   ├── peft_utils.py  # PEFT/LoRA analysis
│   │   ├── trl_utils.py   # TRL debugging
│   │   └── accelerate_utils.py # Accelerate integration
│   │
│   ├── pytorch/           # PyTorch ecosystem  
│   │   ├── debugger.py    # PyTorch debugger
│   │   ├── cuda.py        # CUDA utilities
│   │   ├── distributed.py # Multi-GPU debugging
│   │   ├── triton_utils.py # Triton kernel debugging
│   │   └── bitsandbytes_utils.py # Quantization
│   │
│   └── storage.py         # Advanced tensor storage
│
├── testing/               # Testing & comparison
│   ├── tensor_compare.py  # Multi-dtype comparison
│   ├── runner.py          # Test execution
│   └── differ.py          # Training run comparison
│
├── tracing/               # Performance tracing
│   ├── chrome_tracer.py   # Chrome trace format
│   ├── trace_parser.py    # Trace analysis
│   └── trace_comparer.py  # Trace comparison
│
└── analysis/              # Analysis tools
    ├── loss_logger.py     # Loss curve tracking
    └── loss_analyzer.py   # Loss analysis
```

---

## 🔧 Installation & Dependencies

### Basic Installation
```bash
pip install -e .
```

### Framework-Specific Dependencies
```bash
# HuggingFace ecosystem
pip install -e .[huggingface]

# CUDA debugging
pip install -e .[cuda]  

# Triton kernel debugging
pip install -e .[triton]

# Quantization analysis
pip install -e .[quantization]

# Everything
pip install -e .[all]
```

### Development Setup
```bash
pip install -e .[dev]
pre-commit install
```

---

## 🚀 Performance & Scale

### **Memory Efficiency**
- **Automatic disk storage** for tensors > 1GB
- **Compressed storage** with 3-5x reduction
- **Memory-mapped loading** for large tensors
- **Lazy evaluation** of expensive analyses

### **Multi-GPU Ready**
- **Distributed training** profiling
- **Cross-rank communication** analysis
- **Load balancing** monitoring
- **NCCL collective** benchmarking

### **Production Safe**
- **Minimal overhead** in disable mode
- **Graceful degradation** for missing dependencies
- **Thread-safe** operations
- **Exception handling** with recovery

---

## 📊 Use Cases

### **Research & Development**
- **Model architecture** comparison and optimization
- **Attention pattern** analysis and visualization
- **Gradient flow** debugging and vanishing gradient detection
- **Training dynamics** analysis across different configurations

### **Production Debugging**
- **Memory leak** detection in long-running training
- **Performance regression** analysis between model versions
- **Distributed training** bottleneck identification
- **Quantization impact** assessment on model accuracy

### **Optimization**
- **PEFT efficiency** analysis for parameter-efficient fine-tuning
- **Multi-GPU load balancing** optimization
- **Communication overhead** reduction in distributed training
- **Memory fragmentation** analysis and optimization

---

## 🤝 Contributing

We welcome contributions! The toolkit follows these principles:

1. **Minimal setup**: One-line debugging activation
2. **Framework agnostic**: Support for all major ML frameworks
3. **Production ready**: Robust error handling and minimal overhead
4. **Comprehensive**: Cover all aspects of ML debugging

### Development Guidelines
- Follow existing code patterns and architecture
- Add comprehensive docstrings and examples
- Include integration tests for new features
- Ensure backward compatibility

---

## 📄 License

This project is licensed under the **GPL-3.0 License** - see the [LICENSE](LICENSE) file for details.

> **Why GPL-3.0?** We believe debugging tools should remain open and accessible. The GPL-3.0 ensures that any improvements or derivatives also remain open source, benefiting the entire ML community.

---

## 🙏 Acknowledgments

Built for the ML community to make debugging **trivial** across all major frameworks:
- **HuggingFace** Transformers, PEFT, TRL, Accelerate
- **PyTorch** core, distributed, and CUDA ecosystems  
- **Triton** kernel development and optimization
- **BitsAndBytes** quantization workflows

**Happy Debugging!** 🐛→✨