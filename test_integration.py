#!/usr/bin/env python3
"""
Integration test script to verify core functionality
"""

import torch
import torch.nn as nn
import numpy as np

def test_basic_imports():
    """Test that all major components can be imported"""
    print("Testing basic imports...")
    
    try:
        # Core utilities
        from ml_debug_toolkit import IOLogger, TensorComparer, ChromeTracer
        print("✓ Core utilities imported successfully")
        
        # PyTorch utilities
        from ml_debug_toolkit import PyTorchDebugger, CUDADebugger, auto_debug_module
        print("✓ PyTorch utilities imported successfully")
        
        # HuggingFace utilities
        from ml_debug_toolkit import HuggingFaceDebugger, auto_debug_model
        print("✓ HuggingFace utilities imported successfully")
        
        # Storage utilities
        from ml_debug_toolkit import DiskTensorStorage, MultiDtypeComparer
        print("✓ Storage utilities imported successfully")
        
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_tensor_operations():
    """Test tensor comparison and storage"""
    print("\nTesting tensor operations...")
    
    try:
        from ml_debug_toolkit import TensorComparer, DiskTensorStorage
        
        # Create test tensors
        tensor1 = torch.randn(100, 50)
        tensor2 = tensor1 + torch.randn(100, 50) * 0.01  # Similar tensor
        
        # Test tensor comparison
        comparer = TensorComparer(verbose=False)
        comparer.enable()
        
        comparison = comparer.compare(tensor1, tensor2, name1="tensor1", name2="tensor2")
        assert "statistics" in comparison
        assert "difference_stats" in comparison["statistics"]
        assert "mean_abs_diff" in comparison["statistics"]["difference_stats"]  # This is MAE
        assert "l2_norm" in comparison["statistics"]["difference_stats"]  # This relates to MSE
        
        comparer.disable()
        print("✓ Tensor comparison works")
        
        # Test disk storage
        storage = DiskTensorStorage(verbose=False)
        storage.enable()
        
        storage_key = storage.store_tensor(tensor1, "test_tensor")
        loaded_tensor, info = storage.load_tensor(storage_key)
        
        assert torch.allclose(tensor1, loaded_tensor)
        storage.disable()
        print("✓ Disk tensor storage works")
        
        return True
    except Exception as e:
        print(f"✗ Tensor operations failed: {e}")
        return False

def test_pytorch_debugging():
    """Test PyTorch debugging functionality"""
    print("\nTesting PyTorch debugging...")
    
    try:
        from ml_debug_toolkit import PyTorchDebugger, profile_forward_pass
        
        # Create simple model
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 1)
        )
        
        # Test model debugging
        debugger = PyTorchDebugger(verbose=False)
        debugger.enable()
        
        model_debug = debugger.debug_model(model, "test_model")
        assert "basic_info" in model_debug
        assert "total_parameters" in model_debug["basic_info"]
        
        debugger.disable()
        print("✓ PyTorch model debugging works")
        
        # Test profiling
        inputs = torch.randn(32, 10)
        profile_results = profile_forward_pass(model, inputs, num_runs=10, warmup_runs=2)
        
        assert "timing" in profile_results
        assert "mean_ms" in profile_results["timing"]
        print("✓ PyTorch profiling works")
        
        return True
    except Exception as e:
        print(f"✗ PyTorch debugging failed: {e}")
        return False

def test_huggingface_debugging():
    """Test HuggingFace debugging functionality"""
    print("\nTesting HuggingFace debugging...")
    
    try:
        from ml_debug_toolkit import HuggingFaceDebugger
        
        # Create simple transformer-like model
        model = nn.Sequential(
            nn.Embedding(1000, 128),
            nn.TransformerEncoderLayer(d_model=128, nhead=8, batch_first=True),
            nn.Linear(128, 1)
        )
        
        # Test HF debugging
        hf_debugger = HuggingFaceDebugger(model, verbose=False)
        hf_debugger.enable()
        
        # Basic test - just check that it doesn't crash
        assert hf_debugger.enabled
        hf_debugger.disable()
        print("✓ HuggingFace debugging works")
        
        return True
    except Exception as e:
        print(f"✗ HuggingFace debugging failed: {e}")
        return False

def test_cuda_functionality():
    """Test CUDA debugging if available"""
    print("\nTesting CUDA functionality...")
    
    try:
        from ml_debug_toolkit import CUDADebugger, auto_cuda_debug
        
        if torch.cuda.is_available():
            cuda_debugger = auto_cuda_debug(verbose=False)
            
            memory_snapshot = cuda_debugger.snapshot_memory("test_snapshot")
            assert "memory_stats" in memory_snapshot
            
            cuda_debugger.disable()
            print("✓ CUDA debugging works")
        else:
            # Test that it handles no CUDA gracefully
            cuda_debugger = CUDADebugger(verbose=False)
            cuda_debugger.enable()
            
            snapshot = cuda_debugger.snapshot_memory("no_cuda_test")
            assert "error" in snapshot
            
            cuda_debugger.disable()
            print("✓ CUDA debugging handles no CUDA correctly")
        
        return True
    except Exception as e:
        print(f"✗ CUDA functionality failed: {e}")
        return False

def test_distributed_functionality():
    """Test distributed debugging"""
    print("\nTesting distributed functionality...")
    
    try:
        from ml_debug_toolkit import DistributedDebugger, auto_distributed_debug
        
        # Test basic initialization
        dist_debugger = auto_distributed_debug(verbose=False)
        
        # Test distributed setup analysis
        setup_info = dist_debugger.distributed_info
        assert "distributed_available" in setup_info
        
        dist_debugger.disable()
        print("✓ Distributed debugging initialization works")
        
        return True
    except Exception as e:
        print(f"✗ Distributed functionality failed: {e}")
        return False

def main():
    """Run all tests"""
    print("ML Debug Toolkit Integration Test")
    print("=" * 40)
    
    tests = [
        test_basic_imports,
        test_tensor_operations,
        test_pytorch_debugging,
        test_huggingface_debugging,
        test_cuda_functionality,
        test_distributed_functionality,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
    
    print("\n" + "=" * 40)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The toolkit is working correctly.")
        return 0
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    exit(main())