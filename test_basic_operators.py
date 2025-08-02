#!/usr/bin/env python3
"""
Basic Operator Tests for Qwen3 Debugging

This script runs tests on the fundamental operators used in Qwen3:
- RMSNorm
- Linear layers (GEMM)
- SwiGLU activation

The goal is to establish that the basic building blocks work correctly
before testing more complex operators like attention.
"""

import os
import sys
import json
import time
import warnings
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Try to import the existing test infrastructure
try:
    # Check if we can run existing InfiniCore operator tests
    infinicore_test_path = Path('/home/runner/work/copilot-test/copilot-test/InfiniCore-main/test/infiniop')
    if infinicore_test_path.exists():
        sys.path.insert(0, str(infinicore_test_path.parent))
        print(f"✓ Found InfiniCore test directory: {infinicore_test_path}")
        CAN_RUN_EXISTING_TESTS = True
    else:
        print(f"⚠ InfiniCore test directory not found: {infinicore_test_path}")
        CAN_RUN_EXISTING_TESTS = False
except Exception as e:
    print(f"✗ Error setting up InfiniCore test path: {e}")
    CAN_RUN_EXISTING_TESTS = False


@dataclass
class BasicOperatorTestResult:
    """Results from testing a basic operator"""
    operator_name: str
    test_case: str
    input_shapes: List[Tuple[int, ...]]
    output_shape: Tuple[int, ...]
    python_execution_time: float
    theoretical_ops: int
    error_message: Optional[str] = None


def test_pytorch_rmsnorm():
    """Test PyTorch RMSNorm implementation with various configurations"""
    print("=" * 60)
    print("TESTING PYTORCH RMSNORM")
    print("=" * 60)
    
    results = []
    
    # Test configurations similar to Qwen3
    test_configs = [
        # (batch, seq_len, hidden_size, eps)
        (1, 16, 128, 1e-6),
        (1, 32, 2048, 1e-6),
        (1, 64, 4096, 1e-6),
        (16, 128, 2048, 1e-6),  # Larger batch
    ]
    
    class PyTorchRMSNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.variance_epsilon = eps

        def forward(self, hidden_states):
            input_dtype = hidden_states.dtype
            hidden_states = hidden_states.to(torch.float32)
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
            result = self.weight * hidden_states
            return result.to(input_dtype)  # Ensure we return to original dtype
    
    for batch, seq_len, hidden_size, eps in test_configs:
        print(f"Testing RMSNorm: batch={batch}, seq_len={seq_len}, hidden_size={hidden_size}, eps={eps}")
        
        try:
            # Create test data
            torch.manual_seed(42)
            input_tensor = torch.randn(batch, seq_len, hidden_size, dtype=torch.float16)
            
            # Create RMSNorm layer
            rmsnorm = PyTorchRMSNorm(hidden_size, eps)
            rmsnorm.weight.data.normal_(mean=1.0, std=0.1)
            
            # Run forward pass with timing
            start_time = time.time()
            with torch.no_grad():
                output = rmsnorm(input_tensor)
            execution_time = time.time() - start_time
            
            # Calculate theoretical operations
            # RMSNorm: for each element: power(2) + mean + rsqrt + multiply
            theoretical_ops = batch * seq_len * hidden_size * 4  # Approximate
            
            # Verify output properties
            assert output.shape == input_tensor.shape, f"Shape mismatch: {output.shape} vs {input_tensor.shape}"
            assert output.dtype == input_tensor.dtype, f"Dtype mismatch: {output.dtype} vs {input_tensor.dtype}"
            assert not torch.isnan(output).any(), "Output contains NaN values"
            assert not torch.isinf(output).any(), "Output contains Inf values"
            
            result = BasicOperatorTestResult(
                operator_name="RMSNorm",
                test_case=f"b{batch}_s{seq_len}_h{hidden_size}",
                input_shapes=[(batch, seq_len, hidden_size)],
                output_shape=tuple(output.shape),
                python_execution_time=execution_time,
                theoretical_ops=theoretical_ops,
                error_message=None
            )
            
            print(f"  ✓ PASS - Time: {execution_time:.6f}s, Shape: {output.shape}")
            
        except Exception as e:
            result = BasicOperatorTestResult(
                operator_name="RMSNorm",
                test_case=f"b{batch}_s{seq_len}_h{hidden_size}",
                input_shapes=[(batch, seq_len, hidden_size)],
                output_shape=(0,),
                python_execution_time=0.0,
                theoretical_ops=0,
                error_message=str(e)
            )
            print(f"  ✗ FAIL - Error: {e}")
        
        results.append(result)
    
    return results


def test_pytorch_linear():
    """Test PyTorch Linear layers with various configurations"""
    print("=" * 60)
    print("TESTING PYTORCH LINEAR LAYERS")
    print("=" * 60)
    
    results = []
    
    # Test configurations for different parts of Qwen3
    test_configs = [
        # (batch, seq_len, input_dim, output_dim, bias)
        (1, 16, 128, 512, True),    # Small test
        (1, 32, 2048, 2048, True),  # Q/K/V projection
        (1, 32, 2048, 16384, False), # Gate/Up projection  
        (1, 32, 16384, 2048, False), # Down projection
        (16, 128, 2048, 2048, True), # Larger batch
    ]
    
    for batch, seq_len, input_dim, output_dim, bias in test_configs:
        print(f"Testing Linear: batch={batch}, seq_len={seq_len}, {input_dim}->{output_dim}, bias={bias}")
        
        try:
            # Create test data
            torch.manual_seed(42)
            input_tensor = torch.randn(batch, seq_len, input_dim, dtype=torch.float16)
            
            # Create Linear layer
            linear = nn.Linear(input_dim, output_dim, bias=bias)
            linear = linear.to(torch.float16)  # Match input dtype
            
            # Run forward pass with timing
            start_time = time.time()
            with torch.no_grad():
                output = linear(input_tensor)
            execution_time = time.time() - start_time
            
            # Calculate theoretical operations (GEMM)
            theoretical_ops = batch * seq_len * input_dim * output_dim * 2  # Multiply-add
            if bias:
                theoretical_ops += batch * seq_len * output_dim  # Add bias
            
            # Verify output properties
            expected_shape = (batch, seq_len, output_dim)
            assert output.shape == expected_shape, f"Shape mismatch: {output.shape} vs {expected_shape}"
            assert output.dtype == input_tensor.dtype, f"Dtype mismatch: {output.dtype} vs {input_tensor.dtype}"
            assert not torch.isnan(output).any(), "Output contains NaN values"
            assert not torch.isinf(output).any(), "Output contains Inf values"
            
            result = BasicOperatorTestResult(
                operator_name="Linear",
                test_case=f"b{batch}_s{seq_len}_{input_dim}to{output_dim}_bias{bias}",
                input_shapes=[(batch, seq_len, input_dim)],
                output_shape=tuple(output.shape),
                python_execution_time=execution_time,
                theoretical_ops=theoretical_ops,
                error_message=None
            )
            
            print(f"  ✓ PASS - Time: {execution_time:.6f}s, Shape: {output.shape}")
            
        except Exception as e:
            result = BasicOperatorTestResult(
                operator_name="Linear",
                test_case=f"b{batch}_s{seq_len}_{input_dim}to{output_dim}_bias{bias}",
                input_shapes=[(batch, seq_len, input_dim)],
                output_shape=(0,),
                python_execution_time=0.0,
                theoretical_ops=0,
                error_message=str(e)
            )
            print(f"  ✗ FAIL - Error: {e}")
        
        results.append(result)
    
    return results


def test_pytorch_swiglu():
    """Test SwiGLU activation function"""
    print("=" * 60)
    print("TESTING PYTORCH SWIGLU")
    print("=" * 60)
    
    results = []
    
    class SwiGLU(nn.Module):
        def __init__(self, input_dim, hidden_dim):
            super().__init__()
            self.gate_proj = nn.Linear(input_dim, hidden_dim, bias=False)
            self.up_proj = nn.Linear(input_dim, hidden_dim, bias=False)
            self.down_proj = nn.Linear(hidden_dim, input_dim, bias=False)
            self.act_fn = nn.SiLU()
        
        def forward(self, x):
            gate = self.act_fn(self.gate_proj(x))
            up = self.up_proj(x)
            return self.down_proj(gate * up)
    
    # Test configurations
    test_configs = [
        # (batch, seq_len, input_dim, hidden_dim)
        (1, 16, 128, 512),
        (1, 32, 2048, 11008),  # Qwen3-like configuration
        (1, 64, 4096, 14336),
        (8, 128, 2048, 11008), # Larger batch
    ]
    
    for batch, seq_len, input_dim, hidden_dim in test_configs:
        print(f"Testing SwiGLU: batch={batch}, seq_len={seq_len}, {input_dim}->{hidden_dim}->{input_dim}")
        
        try:
            # Create test data
            torch.manual_seed(42)
            input_tensor = torch.randn(batch, seq_len, input_dim, dtype=torch.float16)
            
            # Create SwiGLU module
            swiglu = SwiGLU(input_dim, hidden_dim)
            swiglu = swiglu.to(torch.float16)
            
            # Run forward pass with timing
            start_time = time.time()
            with torch.no_grad():
                output = swiglu(input_tensor)
            execution_time = time.time() - start_time
            
            # Calculate theoretical operations
            # Gate proj + Up proj + Down proj + SiLU + element-wise multiply
            theoretical_ops = (
                batch * seq_len * input_dim * hidden_dim * 2 * 3 +  # 3 GEMM operations
                batch * seq_len * hidden_dim * 2  # SiLU + element-wise multiply
            )
            
            # Verify output properties
            assert output.shape == input_tensor.shape, f"Shape mismatch: {output.shape} vs {input_tensor.shape}"
            assert output.dtype == input_tensor.dtype, f"Dtype mismatch: {output.dtype} vs {input_tensor.dtype}"
            assert not torch.isnan(output).any(), "Output contains NaN values"
            assert not torch.isinf(output).any(), "Output contains Inf values"
            
            result = BasicOperatorTestResult(
                operator_name="SwiGLU",
                test_case=f"b{batch}_s{seq_len}_{input_dim}h{hidden_dim}",
                input_shapes=[(batch, seq_len, input_dim)],
                output_shape=tuple(output.shape),
                python_execution_time=execution_time,
                theoretical_ops=theoretical_ops,
                error_message=None
            )
            
            print(f"  ✓ PASS - Time: {execution_time:.6f}s, Shape: {output.shape}")
            
        except Exception as e:
            result = BasicOperatorTestResult(
                operator_name="SwiGLU",
                test_case=f"b{batch}_s{seq_len}_{input_dim}h{hidden_dim}",
                input_shapes=[(batch, seq_len, input_dim)],
                output_shape=(0,),
                python_execution_time=0.0,
                theoretical_ops=0,
                error_message=str(e)
            )
            print(f"  ✗ FAIL - Error: {e}")
        
        results.append(result)
    
    return results


def run_existing_infinicore_tests():
    """Run existing InfiniCore operator tests if available"""
    print("=" * 60)
    print("CHECKING EXISTING INFINICORE TESTS")
    print("=" * 60)
    
    if not CAN_RUN_EXISTING_TESTS:
        print("⚠ Cannot run existing InfiniCore tests - path not available")
        return []
    
    results = []
    
    # Check what tests are available
    test_dir = Path('/home/runner/work/copilot-test/copilot-test/InfiniCore-main/test/infiniop')
    test_files = [
        'rms_norm.py',
        'gemm.py', 
        'linear.py',
        'swiglu.py',
        'attention.py'
    ]
    
    for test_file in test_files:
        test_path = test_dir / test_file
        if test_path.exists():
            print(f"✓ Found test: {test_file}")
            # We could potentially run these tests here if the library was built
            # For now, just document that they exist
            results.append(f"Available: {test_file}")
        else:
            print(f"✗ Missing test: {test_file}")
    
    print(f"\nFound {len(results)} existing test files")
    print("Note: To run these tests, InfiniCore library needs to be built first")
    
    return results


def create_precision_test_vectors():
    """Create test vectors specifically for debugging precision issues"""
    print("=" * 60)
    print("CREATING PRECISION TEST VECTORS")
    print("=" * 60)
    
    # Create test vectors that might expose precision differences
    test_vectors = {}
    
    print("Creating test vectors for common Qwen3 scenarios...")
    
    # Small values that might be sensitive to precision
    torch.manual_seed(12345)
    test_vectors['small_values'] = torch.randn(1, 32, 2048) * 1e-3
    
    # Large values that might overflow/underflow in different precisions
    test_vectors['large_values'] = torch.randn(1, 32, 2048) * 10.0
    
    # Values with specific patterns that might expose implementation differences
    test_vectors['alternating_pattern'] = torch.zeros(1, 32, 2048)
    test_vectors['alternating_pattern'][:, :, ::2] = 1.0
    test_vectors['alternating_pattern'][:, :, 1::2] = -1.0
    
    # Edge cases
    test_vectors['zeros'] = torch.zeros(1, 32, 2048)
    test_vectors['ones'] = torch.ones(1, 32, 2048)
    test_vectors['near_zero'] = torch.full((1, 32, 2048), 1e-7)
    
    # Save test vectors for later use with C++ implementation
    output_dir = Path('test_vectors')
    output_dir.mkdir(exist_ok=True)
    
    for name, tensor in test_vectors.items():
        torch.save(tensor, output_dir / f'{name}.pt')
        print(f"  Saved test vector: {name}.pt with shape {tensor.shape}")
    
    print(f"\nSaved {len(test_vectors)} test vectors to {output_dir}/")
    
    return test_vectors


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Basic Operator Tests for Qwen3 Debugging")
    parser.add_argument(
        "--output", 
        type=str,
        default="basic_operator_test_results.json",
        help="Output JSON file for results"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("BASIC OPERATOR TESTS FOR QWEN3 DEBUGGING")
    print("=" * 80)
    print("Testing fundamental operators to establish baseline performance")
    print("and verify that basic building blocks work correctly.")
    print()
    
    all_results = []
    
    # Run PyTorch tests
    try:
        rmsnorm_results = test_pytorch_rmsnorm()
        all_results.extend(rmsnorm_results)
    except Exception as e:
        print(f"✗ RMSNorm tests failed: {e}")
    
    try:
        linear_results = test_pytorch_linear()
        all_results.extend(linear_results)
    except Exception as e:
        print(f"✗ Linear tests failed: {e}")
    
    try:
        swiglu_results = test_pytorch_swiglu()
        all_results.extend(swiglu_results)
    except Exception as e:
        print(f"✗ SwiGLU tests failed: {e}")
    
    # Check existing tests
    try:
        existing_tests = run_existing_infinicore_tests()
    except Exception as e:
        print(f"✗ Existing test check failed: {e}")
        existing_tests = []
    
    # Create precision test vectors
    try:
        test_vectors = create_precision_test_vectors()
    except Exception as e:
        print(f"✗ Test vector creation failed: {e}")
        test_vectors = {}
    
    # Calculate overall results
    passed_tests = [r for r in all_results if r.error_message is None]
    failed_tests = [r for r in all_results if r.error_message is not None]
    
    print("=" * 80)
    print("BASIC OPERATOR TEST SUMMARY")
    print("=" * 80)
    print(f"Total tests run: {len(all_results)}")
    print(f"Passed: {len(passed_tests)}")
    print(f"Failed: {len(failed_tests)}")
    print()
    
    if failed_tests:
        print("Failed tests:")
        for result in failed_tests:
            print(f"  {result.operator_name} ({result.test_case}): {result.error_message}")
        print()
    
    # Performance summary
    if passed_tests:
        print("Performance summary:")
        by_operator = {}
        for result in passed_tests:
            if result.operator_name not in by_operator:
                by_operator[result.operator_name] = []
            by_operator[result.operator_name].append(result)
        
        for op_name, results in by_operator.items():
            total_time = sum(r.python_execution_time for r in results)
            total_ops = sum(r.theoretical_ops for r in results)
            avg_time = total_time / len(results)
            print(f"  {op_name}: {len(results)} tests, avg time: {avg_time:.6f}s")
        print()
    
    # Save results
    output_data = {
        'test_results': [
            {
                'operator_name': r.operator_name,
                'test_case': r.test_case,
                'input_shapes': r.input_shapes,
                'output_shape': r.output_shape,
                'python_execution_time': r.python_execution_time,
                'theoretical_ops': r.theoretical_ops,
                'error_message': r.error_message
            }
            for r in all_results
        ],
        'summary': {
            'total_tests': len(all_results),
            'passed': len(passed_tests),
            'failed': len(failed_tests),
            'existing_tests_available': existing_tests,
            'test_vectors_created': list(test_vectors.keys()) if test_vectors else []
        }
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Results saved to: {args.output}")
    print()
    
    # Provide next steps
    print("=" * 80)
    print("NEXT STEPS FOR DEBUGGING")
    print("=" * 80)
    print("1. Build InfiniCore library to enable C++ operator testing")
    print("2. Run existing InfiniCore operator tests: rms_norm.py, gemm.py, etc.")
    print("3. Use created test vectors to compare C++ vs Python implementations")
    print("4. Focus on operators that show the largest precision differences")
    print("5. Run layer-by-layer Qwen3 verification with working operators")
    
    # Exit with appropriate code
    sys.exit(0 if len(failed_tests) == 0 else 1)


if __name__ == "__main__":
    main()