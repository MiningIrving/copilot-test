#!/usr/bin/env python3
"""
Enhanced Qwen3 Layer Analysis Tool

This script analyzes the existing validation results and provides detailed
debugging guidance for the Qwen3 implementation differences between C++ and Python.

Based on the validation results, it creates focused tests to help identify
the root cause of the computational differences.
"""

import os
import sys
import json
import math
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, asdict

# Suppress warnings
warnings.filterwarnings("ignore")


@dataclass
class ValidationAnalysis:
    """Analysis of validation results"""
    tokenization_status: bool
    token_matches: List[bool]
    generation_outputs_match: bool
    word_overlap_scores: List[float]
    cpp_vs_python_time_ratio: float
    identified_issues: List[str]
    recommended_tests: List[str]


def analyze_existing_validation_results() -> ValidationAnalysis:
    """Analyze the existing qwen3_validation_results.json"""
    
    print("=" * 80)
    print("ANALYZING EXISTING VALIDATION RESULTS")
    print("=" * 80)
    
    # Load existing validation results
    results_file = Path("/home/runner/work/copilot-test/copilot-test/InfiniCore-Infer-main/qwen3_validation_results.json")
    
    if not results_file.exists():
        print(f"⚠ Validation results file not found: {results_file}")
        return ValidationAnalysis(
            tokenization_status=False,
            token_matches=[],
            generation_outputs_match=False,
            word_overlap_scores=[],
            cpp_vs_python_time_ratio=0.0,
            identified_issues=["Validation results file not found"],
            recommended_tests=["Run initial validation to generate results"]
        )
    
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        print("✓ Loaded validation results")
        print(f"  File: {results_file}")
        
        # Analyze tokenization
        tokenization_status = results.get("tokenization", {}).get("passed", False)
        print(f"  Tokenization passed: {tokenization_status}")
        
        # Analyze single step results
        token_matches = []
        cpp_times = []
        python_times = []
        
        step_keys = [k for k in results.keys() if k.startswith("single_step_")]
        for key in sorted(step_keys):
            step_data = results[key]
            token_match = step_data.get("tokens_match", False)
            token_matches.append(token_match)
            
            cpp_time = step_data.get("cpp_time", 0.0)
            python_time = step_data.get("pytorch_time", 0.0)
            cpp_times.append(cpp_time)
            python_times.append(python_time)
            
            print(f"  {key}: tokens_match={token_match}, cpp_token={step_data.get('cpp_token', 'N/A')}, pytorch_token={step_data.get('pytorch_token', 'N/A')}")
        
        # Analyze generation results
        generation_keys = [k for k in results.keys() if k.startswith("generation_")]
        word_overlap_scores = []
        generation_outputs_match = True
        
        for key in sorted(generation_keys):
            gen_data = results[key]
            outputs_match = gen_data.get("outputs_match", False)
            word_overlap = gen_data.get("word_overlap", 0.0)
            word_overlap_scores.append(word_overlap)
            
            if not outputs_match:
                generation_outputs_match = False
            
            print(f"  {key}: outputs_match={outputs_match}, word_overlap={word_overlap}")
            print(f"    C++ output: '{gen_data.get('cpp_output', 'N/A')}'")
            print(f"    Python output: '{gen_data.get('pytorch_output', 'N/A')}'")
        
        # Calculate time ratios
        total_cpp_time = sum(cpp_times)
        total_python_time = sum(python_times)
        time_ratio = total_cpp_time / total_python_time if total_python_time > 0 else float('inf')
        
        print(f"  Time ratio (C++/Python): {time_ratio:.2f}")
        
        # Identify issues
        identified_issues = []
        if not tokenization_status:
            identified_issues.append("Tokenization failing - fundamental input processing issue")
        
        if not any(token_matches):
            identified_issues.append("No token matches - complete output divergence from first step")
        
        if not generation_outputs_match:
            identified_issues.append("Generation outputs completely different")
        
        if all(score == 0.0 for score in word_overlap_scores):
            identified_issues.append("Zero word overlap - outputs are completely unrelated")
        
        if time_ratio > 100:
            identified_issues.append("C++ implementation much slower than expected")
        
        # Generate recommended tests
        recommended_tests = []
        
        if not tokenization_status:
            recommended_tests.extend([
                "Test tokenizer configuration consistency",
                "Verify input preprocessing steps",
                "Check encoding/decoding pipelines"
            ])
        
        if not any(token_matches):
            recommended_tests.extend([
                "Test embedding layer with identical inputs",
                "Verify model weight loading",
                "Check first transformer layer computation"
            ])
        
        if not generation_outputs_match:
            recommended_tests.extend([
                "Test final layer normalization",
                "Verify logits computation",
                "Check sampling/generation logic"
            ])
        
        recommended_tests.extend([
            "Run layer-by-layer output comparison",
            "Test individual operators (RMSNorm, Attention, MLP)",
            "Verify numerical precision settings"
        ])
        
        analysis = ValidationAnalysis(
            tokenization_status=tokenization_status,
            token_matches=token_matches,
            generation_outputs_match=generation_outputs_match,
            word_overlap_scores=word_overlap_scores,
            cpp_vs_python_time_ratio=time_ratio,
            identified_issues=identified_issues,
            recommended_tests=recommended_tests
        )
        
        return analysis
        
    except Exception as e:
        print(f"✗ Error analyzing validation results: {e}")
        return ValidationAnalysis(
            tokenization_status=False,
            token_matches=[],
            generation_outputs_match=False,
            word_overlap_scores=[],
            cpp_vs_python_time_ratio=0.0,
            identified_issues=[f"Analysis error: {e}"],
            recommended_tests=["Fix analysis errors and retry"]
        )


def create_targeted_debugging_tests(analysis: ValidationAnalysis) -> Dict[str, Any]:
    """Create targeted tests based on the analysis results"""
    
    print("=" * 80)
    print("CREATING TARGETED DEBUGGING TESTS")
    print("=" * 80)
    
    tests = {}
    
    # Test 1: Basic tensor operations
    print("Creating basic tensor operation tests...")
    tests['basic_tensor_ops'] = create_basic_tensor_tests()
    
    # Test 2: Embedding layer tests
    if not analysis.tokenization_status or not any(analysis.token_matches):
        print("Creating embedding layer tests...")
        tests['embedding_tests'] = create_embedding_tests()
    
    # Test 3: Layer normalization tests  
    print("Creating layer normalization tests...")
    tests['normalization_tests'] = create_normalization_tests()
    
    # Test 4: Attention mechanism tests
    print("Creating attention mechanism tests...")
    tests['attention_tests'] = create_attention_tests()
    
    # Test 5: MLP tests
    print("Creating MLP tests...")
    tests['mlp_tests'] = create_mlp_tests()
    
    # Test 6: Configuration consistency tests
    print("Creating configuration consistency tests...")
    tests['config_tests'] = create_config_consistency_tests()
    
    return tests


def create_basic_tensor_tests() -> Dict[str, torch.Tensor]:
    """Create basic tensor operation tests"""
    
    tests = {}
    
    # Test different data types and shapes that might expose precision issues
    dtypes = [torch.float16, torch.float32]
    shapes = [
        (1, 1, 2048),      # Single token
        (1, 32, 2048),     # Small sequence
        (1, 128, 2048),    # Medium sequence
        (8, 32, 2048),     # Small batch
    ]
    
    for i, (dtype, shape) in enumerate(zip(dtypes * len(shapes), shapes * len(dtypes))):
        # Create deterministic test tensors
        torch.manual_seed(42 + i)
        
        # Normal distribution (most common case)
        tests[f'normal_{dtype}_{shape}'] = torch.randn(shape, dtype=dtype) * 0.02
        
        # Small values (might expose underflow issues)
        tests[f'small_{dtype}_{shape}'] = torch.randn(shape, dtype=dtype) * 1e-4
        
        # Large values (might expose overflow issues)
        tests[f'large_{dtype}_{shape}'] = torch.randn(shape, dtype=dtype) * 10.0
        
        # Edge cases
        tests[f'zeros_{dtype}_{shape}'] = torch.zeros(shape, dtype=dtype)
        tests[f'ones_{dtype}_{shape}'] = torch.ones(shape, dtype=dtype)
    
    return tests


def create_embedding_tests() -> Dict[str, Any]:
    """Create embedding layer tests"""
    
    tests = {}
    
    # Test different vocabulary sizes and hidden dimensions
    configs = [
        {"vocab_size": 1000, "hidden_size": 128},
        {"vocab_size": 151936, "hidden_size": 2048},  # Qwen3-like
        {"vocab_size": 151936, "hidden_size": 4096},  # Larger model
    ]
    
    for i, config in enumerate(configs):
        vocab_size = config["vocab_size"]
        hidden_size = config["hidden_size"]
        
        # Create test token sequences
        torch.manual_seed(42 + i)
        
        # Short sequence
        short_tokens = torch.randint(0, vocab_size, (1, 8))
        tests[f'embedding_short_{vocab_size}_{hidden_size}'] = {
            'tokens': short_tokens,
            'config': config
        }
        
        # Medium sequence  
        medium_tokens = torch.randint(0, vocab_size, (1, 32))
        tests[f'embedding_medium_{vocab_size}_{hidden_size}'] = {
            'tokens': medium_tokens,
            'config': config
        }
        
        # Edge case: first and last tokens
        edge_tokens = torch.tensor([[0, vocab_size-1, vocab_size//2]])
        tests[f'embedding_edge_{vocab_size}_{hidden_size}'] = {
            'tokens': edge_tokens,
            'config': config
        }
    
    return tests


def create_normalization_tests() -> Dict[str, Any]:
    """Create layer normalization tests"""
    
    tests = {}
    
    # Different normalization scenarios
    hidden_sizes = [128, 2048, 4096]
    eps_values = [1e-6, 1e-5, 1e-8]
    
    for hidden_size in hidden_sizes:
        for eps in eps_values:
            # Test different input characteristics
            torch.manual_seed(42)
            
            # Normal case
            normal_input = torch.randn(1, 32, hidden_size) * 0.02
            tests[f'rmsnorm_normal_{hidden_size}_eps{eps}'] = {
                'input': normal_input,
                'hidden_size': hidden_size,
                'eps': eps
            }
            
            # Very small values (near underflow)
            small_input = torch.randn(1, 32, hidden_size) * 1e-6
            tests[f'rmsnorm_small_{hidden_size}_eps{eps}'] = {
                'input': small_input,
                'hidden_size': hidden_size,
                'eps': eps
            }
            
            # Large values
            large_input = torch.randn(1, 32, hidden_size) * 5.0
            tests[f'rmsnorm_large_{hidden_size}_eps{eps}'] = {
                'input': large_input,
                'hidden_size': hidden_size,
                'eps': eps
            }
    
    return tests


def create_attention_tests() -> Dict[str, Any]:
    """Create attention mechanism tests"""
    
    tests = {}
    
    # Different attention configurations
    configs = [
        {"hidden_size": 128, "num_heads": 8, "num_kv_heads": 2, "seq_len": 16},
        {"hidden_size": 2048, "num_heads": 16, "num_kv_heads": 16, "seq_len": 32},
        {"hidden_size": 4096, "num_heads": 32, "num_kv_heads": 8, "seq_len": 64},
    ]
    
    for i, config in enumerate(configs):
        hidden_size = config["hidden_size"]
        num_heads = config["num_heads"]
        seq_len = config["seq_len"]
        
        torch.manual_seed(42 + i)
        
        # Test different input patterns
        # Normal case
        normal_input = torch.randn(1, seq_len, hidden_size) * 0.02
        tests[f'attention_normal_{hidden_size}_{num_heads}_{seq_len}'] = {
            'input': normal_input,
            'config': config
        }
        
        # All same values (might expose normalization issues)
        uniform_input = torch.ones(1, seq_len, hidden_size) * 0.1
        tests[f'attention_uniform_{hidden_size}_{num_heads}_{seq_len}'] = {
            'input': uniform_input,
            'config': config
        }
        
        # Sequence with pattern (might expose position encoding issues)
        pattern_input = torch.zeros(1, seq_len, hidden_size)
        for pos in range(seq_len):
            pattern_input[0, pos, :] = 0.1 * (pos + 1) / seq_len
        tests[f'attention_pattern_{hidden_size}_{num_heads}_{seq_len}'] = {
            'input': pattern_input,
            'config': config
        }
    
    return tests


def create_mlp_tests() -> Dict[str, Any]:
    """Create MLP tests"""
    
    tests = {}
    
    # Different MLP configurations
    configs = [
        {"hidden_size": 128, "intermediate_size": 512},
        {"hidden_size": 2048, "intermediate_size": 11008},  # Qwen3-like
        {"hidden_size": 4096, "intermediate_size": 14336},
    ]
    
    for i, config in enumerate(configs):
        hidden_size = config["hidden_size"]
        intermediate_size = config["intermediate_size"]
        
        torch.manual_seed(42 + i)
        
        # Different activation patterns
        # Normal case
        normal_input = torch.randn(1, 32, hidden_size) * 0.02
        tests[f'mlp_normal_{hidden_size}_{intermediate_size}'] = {
            'input': normal_input,
            'config': config
        }
        
        # Positive values (SiLU behavior)
        positive_input = torch.abs(torch.randn(1, 32, hidden_size)) * 0.5
        tests[f'mlp_positive_{hidden_size}_{intermediate_size}'] = {
            'input': positive_input,
            'config': config
        }
        
        # Negative values (SiLU behavior)
        negative_input = -torch.abs(torch.randn(1, 32, hidden_size)) * 0.5
        tests[f'mlp_negative_{hidden_size}_{intermediate_size}'] = {
            'input': negative_input,
            'config': config
        }
    
    return tests


def create_config_consistency_tests() -> Dict[str, Any]:
    """Create configuration consistency tests"""
    
    tests = {}
    
    # Common configuration issues
    test_configs = [
        {
            "name": "qwen3_1.7b_like",
            "vocab_size": 151936,
            "hidden_size": 2048,
            "intermediate_size": 11008,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "num_key_value_heads": 16,
            "max_position_embeddings": 32768,
            "rope_theta": 1000000.0,
            "rms_norm_eps": 1e-6,
        },
        {
            "name": "qwen3_7b_like",
            "vocab_size": 151936,
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "num_hidden_layers": 28,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "max_position_embeddings": 32768,
            "rope_theta": 1000000.0,
            "rms_norm_eps": 1e-6,
        }
    ]
    
    for config in test_configs:
        tests[f'config_{config["name"]}'] = config
    
    return tests


def save_debugging_tests(tests: Dict[str, Any], output_dir: str = "debugging_tests"):
    """Save debugging tests to files"""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"Saving debugging tests to {output_path}/")
    
    # Save tensor tests
    if 'basic_tensor_ops' in tests:
        tensor_dir = output_path / "tensor_tests"
        tensor_dir.mkdir(exist_ok=True)
        
        for test_name, tensor in tests['basic_tensor_ops'].items():
            torch.save(tensor, tensor_dir / f"{test_name}.pt")
    
    # Save other tests as JSON (serializable parts)
    for category, test_data in tests.items():
        if category == 'basic_tensor_ops':
            continue  # Already saved as .pt files
        
        # Convert tensors to lists for JSON serialization
        serializable_data = {}
        for test_name, test_info in test_data.items():
            serializable_info = {}
            for key, value in test_info.items():
                if isinstance(value, torch.Tensor):
                    serializable_info[key] = {
                        'data': value.tolist(),
                        'dtype': str(value.dtype),
                        'shape': list(value.shape)
                    }
                else:
                    serializable_info[key] = value
            serializable_data[test_name] = serializable_info
        
        with open(output_path / f"{category}.json", 'w') as f:
            json.dump(serializable_data, f, indent=2)
    
    print(f"✓ Saved {len(tests)} test categories")
    return output_path


def generate_debugging_report(analysis: ValidationAnalysis, tests: Dict[str, Any]) -> str:
    """Generate a comprehensive debugging report"""
    
    report = []
    report.append("=" * 80)
    report.append("QWEN3 DEBUGGING ANALYSIS REPORT")
    report.append("=" * 80)
    report.append("")
    
    # Summary of issues
    report.append("IDENTIFIED ISSUES:")
    for i, issue in enumerate(analysis.identified_issues, 1):
        report.append(f"{i}. {issue}")
    report.append("")
    
    # Analysis details
    report.append("DETAILED ANALYSIS:")
    report.append(f"- Tokenization working: {analysis.tokenization_status}")
    report.append(f"- Token matches: {sum(analysis.token_matches)}/{len(analysis.token_matches)}")
    report.append(f"- Generation outputs match: {analysis.generation_outputs_match}")
    report.append(f"- Average word overlap: {np.mean(analysis.word_overlap_scores):.3f}")
    report.append(f"- C++/Python time ratio: {analysis.cpp_vs_python_time_ratio:.2f}")
    report.append("")
    
    # Root cause analysis
    report.append("ROOT CAUSE ANALYSIS:")
    if not analysis.tokenization_status:
        report.append("- PRIMARY ISSUE: Tokenization failure indicates input processing problems")
        report.append("  * Check tokenizer configuration consistency")
        report.append("  * Verify vocabulary and special token handling")
    elif not any(analysis.token_matches):
        report.append("- PRIMARY ISSUE: Complete output divergence from first token")
        report.append("  * Check embedding layer implementation")
        report.append("  * Verify weight loading and initialization")
        report.append("  * Test first transformer layer computation")
    else:
        report.append("- PRIMARY ISSUE: Computational differences in model layers")
        report.append("  * Focus on layer-by-layer comparison")
        report.append("  * Check operator precision and implementation differences")
    report.append("")
    
    # Recommended testing order
    report.append("RECOMMENDED TESTING ORDER:")
    for i, test in enumerate(analysis.recommended_tests, 1):
        report.append(f"{i}. {test}")
    report.append("")
    
    # Available tests
    report.append("AVAILABLE DEBUGGING TESTS:")
    for category, test_data in tests.items():
        report.append(f"- {category}: {len(test_data)} tests")
    report.append("")
    
    # Next steps
    report.append("IMMEDIATE NEXT STEPS:")
    report.append("1. Build InfiniCore library to enable C++ operator testing")
    report.append("2. Run basic operator precision tests (RMSNorm, Linear, SwiGLU)")
    report.append("3. Use created test vectors to compare implementations")
    report.append("4. Focus on the identified primary issue area")
    report.append("5. Implement fixes incrementally and verify with layer tests")
    report.append("")
    
    return "\n".join(report)


def main():
    """Main entry point"""
    
    print("QWEN3 ENHANCED DEBUGGING ANALYSIS")
    print("This tool analyzes existing validation results and creates targeted debugging tests")
    print()
    
    # Analyze existing validation results
    analysis = analyze_existing_validation_results()
    
    # Create targeted debugging tests
    tests = create_targeted_debugging_tests(analysis)
    
    # Save tests
    test_dir = save_debugging_tests(tests)
    
    # Generate report
    report = generate_debugging_report(analysis, tests)
    
    # Save and display report
    report_file = test_dir / "debugging_report.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(report)
    print(f"Full report saved to: {report_file}")
    
    # Save analysis results
    analysis_file = test_dir / "analysis_results.json"
    with open(analysis_file, 'w') as f:
        json.dump(asdict(analysis), f, indent=2)
    
    print(f"Analysis results saved to: {analysis_file}")


if __name__ == "__main__":
    main()